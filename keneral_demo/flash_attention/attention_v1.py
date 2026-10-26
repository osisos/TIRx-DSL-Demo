from typing import cast

import tvm
from tvm.ir import Var
from tvm.tirx import script as T
from tvm.tirx.layout import TileLayout, S, R, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.script import tile as Tx

BLOCK_CNT = 66
WARP_PER_CTA = 4

MMA_M = 16
MMA_K = 16
MMA_N = 8

Q_TILE_M = 16
K_TILE_N = 8
V_TILE_N = 8

OUTER_STAGE = 2  # 这是K_TILE, V_TILE 从 global memory -> share memory的流水线
INNER_STAGE = 2  # 在Q x K^T，以及计算Attention x V 的块，存在 share memory -> register 的加载，我们定义为Inner stage


def flash_attention_v1(M: int, N: int, D: int):
    """
    核心设计，每个warp负责一个Q_TILE及其相关的计算。
    e.g. 当前warp负责的Q_TILE为Q[N:N+16, D]，那么需要的K_TILE、V_TILE为[0:N+16, D]。
    实际执行过程中，Q_TILE一次性加载到register中；K V tile 不断从 global memory 加载到 share memory中，这个是outer stage的流水线。
    在计算 S_tile = Q_tile @ K_tile^T ([16, D] x [D, 8] -> [16, 8]) 时：
      - 沿 D 维以步长 MMA_K=16 切分为 D // 16 步（inner stage 流水线）：
      - 每步通过 ldmatrix 将 KSMEM[8, 16] 预取到 K_REG，与常驻寄存器的 Q_REG[16, 16]
        调用 mma.m16n8k16 指令累加：QK_REG[16, 8] += Q_REG[16, k:k+16] @ K_REG[8, k:k+16]^T
    """

    SOFTMAX_SCALE = D**-0.5  # Q x K^T 的缩放系数

    Q_TILES = D // MMA_K
    RES_TILES = D // MMA_N

    Q_REG_PER_THREAD = (D * Q_TILE_M * 2) // (32 * 4)
    # 每个thread中Q_TILE的占用寄存器数量
    Q_REG_CP_TURNS = Q_REG_PER_THREAD // 4
    # 一次搬运填满4个32bit寄存器

    # Q 寄存器
    Q_LAYOUT = TileLayout(
        S[(2, 8, Q_TILES, 2, 4, 2) : (2, 4 @ laneid, 8, 4, 1 @ laneid, 1)],
    )

    # K share memory
    K_SMEM_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [OUTER_STAGE, K_TILE_N, D],
    )

    V_SMEM_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [OUTER_STAGE, V_TILE_N, D],
    )

    # RES 寄存器，存储最终的计算结果
    RES_LAYOUT = TileLayout(
        S[(2, 8, RES_TILES, 4, 2) : (2, 4 @ laneid, 4, 1 @ laneid, 1)],
    )

    # K 寄存器
    # K_TILE不需要完整的加载到寄存器，只需要按照规约维度的索引，不断从smem -> reg进行搬运，因此寄存器的K是for tensor core的计算视图
    # 实际 Q_TILE @ K_TILE^T: m16n8k16 tensor mma
    K_LAYOUT = TileLayout(
        S[(INNER_STAGE, 8, 2, 4, 2) : (4, 4 @ laneid, 2, 1 @ laneid, 1)]
    )

    # V 寄存器
    # 同K_TILE，仅用来计算 RES = RES + Attention[16:8] @ V[8:D]，按照INNER流水线加载到REG即可
    # 实际执行: mma m16n8k8（因为Attention的形状为[16,8]，所以中间的规约维度也只有8）
    V_LAYOUT = TileLayout(S[(INNER_STAGE, 4, 2, 8) : (2, 1 @ laneid, 1, 4 @ laneid)])

    # Q @ K^T 的直接计算结果 REG，32bit
    # NOTE: 之前一版代码，QK_LAYOUT展开了OUTER_STAGE，是不需要的。
    # 原因: 实际的计算过程中，OUTER_PIPELINE:
    # for k in range inner_stage:
    #   cp.async(k+1) -> attention(k) # k表示 inner stage
    #   实际上流水线实现的伪并行，完全依赖cp.async是一个 异步 动作，发射完cp.async后，后续执行的 attention过程完全同步！！！因此，可以保证一个时间段只有一K_TILE会参与计算。
    #   这里，如果是hopper+的硬件，真正执行producer consumer 的计算流程是，是需要展开的。
    #   后续的所有Attention相关都可以移除OUTSTAGE展开
    #   TODO: 分析下这类是如何解决Local Memory SPILL风险的
    QK_LAYOUT = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])

    # # 似乎这个不太有用？可以删掉
    # ATTN_TILE_ROW_MAX_LAYOUT = TileLayout(
    #     S[(OUTER_STAGE, 2, 8) : (2, 1, 4 @ laneid)] + R[4 : 1 @ laneid]
    # )

    # Q_TILE @ K_TILE ^ T, 计算出所有的attention的最大值
    QK_ROW_MAX_LAYOUT = TileLayout(S[(2, 8) : (1, 4 @ laneid)] + R[4 : 1 @ laneid])

    # CURRENT QK MAX，当前QK Tile的最大值
    CURR_QK_ROW_MAX_LAYOUT = TileLayout(S[(2, 8) : (1, 4 @ laneid)] + R[4 : 1 @ laneid])

    # current Q_TILE@K_TILE ^T - 全局最大值 (row)
    QK_ROW_MAX_DIFF_LAYOUT = TileLayout(S[(2, 8) : (1, 4 @ laneid)] + R[4 : 1 @ laneid])

    # ATTN 开头的变量名，表示是exp过的
    QK_ROW_SUM_LAYOUT = TileLayout(S[(2, 8) : (1, 4 @ laneid)] + R[4 : 1 @ laneid])

    # 最终的 exp((Q_TILE@K_TILE^T - QK_ROW_MAX) / D ^ -0.5) 对比真实的attention，仅缺少除 ATTN_ROW_SUM [16, 8]
    ATTN_TILE_LAYOUT = TileLayout(S[(2, 8, 4, 2) : (2, 4 @ laneid, 1 @ laneid, 1)])

    @T.prim_func
    def kernel(
        Q: T.Buffer((M, D), "float16"),
        K: T.Buffer((N, D), "float16"),
        V: T.Buffer((N, D), "float16"),
        O: T.Buffer((M, D), "float16"),
    ):
        T.device_entry()

        bx = T.cta_id([BLOCK_CNT])
        warp_id = T.warp_id([WARP_PER_CTA])
        lane_id = T.lane_id([32])

        """
        核心设计，首先，因为sm89实际上不支持tma加载，并且mma tensor core 是不支持跨小单元调用的，因此，每个mma启动的
        """

        pool = T.SMEMPool()

        KSMEM = pool.alloc(
            [OUTER_STAGE, K_TILE_N, D],
            "float16",
            layout=K_SMEM_LAYOUT,
        )
        VSMEM = pool.alloc(
            [OUTER_STAGE, K_TILE_N, D],
            "float16",
            layout=V_SMEM_LAYOUT,
        )

        SMEM_READY_BAR = pool.alloc((OUTER_STAGE,), "uint64", align=8)
        SMEM_FREE_BAR = pool.alloc((OUTER_STAGE,), "uint64", align=8)

        pool.commit()

        # --- Q寄存器 ---
        """
            NOTE: 物理地址、layout、逻辑地址，三个元素构成三个函数，任意确定其中两个，第三个也确定了。
            这里的Q_LOCAL 有一种错误的写法，Q_LOCAL = Q_REG.local(Q_REG_CP_TURNS, 8, Q_LAYOUT.storage())
            这里来不及了，后面补一下张量代数。
            注意，当Layout不是双射的时候，可能引入其他的问题？如何确定Layout不是双射？？？torch编译栈中好像有种技术。
            但是后面我的Q_LOCAL好像写的有点不爽，我在这里需要的是强连续访问。不知道为什么很容易写错

            感觉隐含了一个代数方程
            我已知原始的逻辑、和Laytout，那么显然唯一确定了物理；接下来，我需要做一种变换，求出一个laytout，
            这个laytout，可以让我确定的 逻辑:[Q_TURN, ACC_PER_THREAD]，只需要迭代PER_THREAD 维度，就可以按照某个顺序访问到原始Laytout的中间某个维度的迭代。。。

            此处我在Gemini-notebook 中有个笔记后面慢慢看吧（这里太小，写不开——
        """

        # 4B对齐
        # NOTE: [Local Memory 风险 - 寄存器容量溢出 (Register Spill)]
        # Q_REG 占 D/4 个 32-bit 寄存器/线程，RES 占 D/2 个寄存器/线程（合计 3D/4）。
        # 当 D >= 256 时，仅 Q_REG + RES 就占 192 个寄存器，加上 K/V/QK 缓冲与展开后的临时变量，
        # 极易超过 SM89 单线程 255 寄存器上限，触发 ptxas 被动 Spill 到 Local Memory。

        # --- Q ---
        Q = T.alloc_local((Q_TILE_M, D), "float16", layout=Q_LAYOUT, align=4)
        # Global memeory -> register 的拷贝视图
        Q_LOCAL = Q.local(Q_REG_CP_TURNS, 8)  # 直接物理连续视图
        Q_CP_VIEW = Q_LOCAL.view("uint32")

        # --- RES ---
        RES = T.alloc_local((Q_TILE_M, D), "float32", layout=RES_LAYOUT, align=4)
        # 每个 tile 对应一条 m16n8k16 MMA 的四个 FP32 累加操作数。
        # 按行缩放时，第 h 行对应 RES_LOCAL[tile, 2*h : 2*h+2]。
        RES_LOCAL = RES.local(RES_TILES, 4)
        for tile in T.unroll(RES_TILES):  # pyright: ignore[reportGeneralTypeIssues]
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                RES_LOCAL[tile, i] = T.float32(0.0)

        # --- K ---
        K = T.alloc_local(
            (INNER_STAGE, K_TILE_N, 16),
            "float16",
            layout=K_LAYOUT,
            align=4,
        )

        K_LOCAL = K.local(INNER_STAGE, 4)  # 直接物理连续视图
        K_B32_VIEW = K_LOCAL.view(
            "uint32"
        )  # TODO: 未必真的可行，用于直接使用32bit视图，对齐寄存器地址，来发起tensor core mma

        V = T.alloc_local(
            (INNER_STAGE, V_TILE_N, MMA_N),
            "float16",
            layout=V_LAYOUT,
            align=4,
        )
        V_LOCAL = V.local(INNER_STAGE, 2)
        V_B32_VIEW = V_LOCAL.view("uint32")  # 同 K

        # Q/K 输入为 FP16，QK 使用 FP32 累加，对应 MMA 的 f32.f16.f16.f32。
        # NOTE: [Local Memory 风险 - 动态 Stage 索引与跨位宽指针重解释]
        # 1. QK_REG / ATTN_TILE_ROW_MAX / ATTN_TILE_DIFF 包含了 OUTER_STAGE 维度。
        #    若外层 K/V 循环未按 OUTER_STAGE 展开，传入运行时的 outer_stage 变量作下标，
        #    会导致动态寻址并强制退化到 Local Memory（实际上寄存器层无需 OUTER_STAGE 维度）。
        # 2. 对同一个 float32 数组同时创建 float16 / uint32 视图并做原位半宽覆写（In-place Type Punning），
        #    容易破坏 NVCC 的 SROA（标量替换）别名分析，导致整个数组落入 Local Memory。
        QK = T.alloc_local(
            (16, 8),  # 这个常数，后续可以优化下
            "float32",
            layout=QK_LAYOUT,
            align=4,
        )
        # local 视图
        QK_LOCAL_VIEW = QK.local(4)
        QK_ACC_VIEW = QK.local(4)  # 每个 lane 每个 stage 持有 4 个 FP32 累加元素。

        # NOTE: attn row max 规约的中间结果，之所以是这个排布是因为 row:1 的相关的寄存器在[0-3] laneid 中
        # NOTE: [Local Memory 风险] 同 QK_REG，若以运行时变量 outer_stage 索引该局部缓冲，会退化到 Local Memory。
        # QK_ROW_MAX = T.alloc_local(
        #     (16, 1), "float32", layout=ATTN_TILE_ROW_MAX_LAYOUT, align=4
        # )
        # ATTN_TILE_ROW_MAX_LOCAL = QK_ROW_MAX.local(2)

        QK_ROW_MAX = T.alloc_local(
            (16, 1), "float32", layout=QK_ROW_MAX_LAYOUT, align=4
        )
        QK_ROW_MAX_LOCAL = QK_ROW_MAX.local()  # 需要初始化为 负无穷
        for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
            QK_ROW_MAX_LOCAL[i] = T.float32(float("-inf"))

        # TODO：明天从这里继续 ------

        # NOTE: [Local Memory 风险] 同 QK_REG，若以运行时变量 outer_stage 索引该局部缓冲，会退化到 Local Memory。
        QK_ROW_MAX_DIFF = T.alloc_local(
            (16, 1), "float32", layout=QK_ROW_MAX_DIFF_LAYOUT, align=4
        )
        QK_ROW_MAX_DIFF_LOCAL = QK_ROW_MAX_DIFF.local(2)

        CURR_QK_ROW_MAX = T.alloc_local(
            (16, 1), "float32", layout=CURR_QK_ROW_MAX_LAYOUT, align=4
        )
        CURR_QK_ROW_MAX_LOCAL = CURR_QK_ROW_MAX.local(2)

        # Q_TILE @ K_TILE ^T 的 row reduce sum
        QK_ROW_SUM = T.alloc_local(
            (16, 1), "float32", layout=QK_ROW_SUM_LAYOUT, align=4
        )
        QK_ROW_SUM_LOCAL = QK_ROW_SUM.local()
        for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
            QK_ROW_SUM_LOCAL[i] = T.float32(0.0)

        ATTN = T.alloc_local((16, 8), "float16", layout=ATTN_TILE_LAYOUT, align=4)
        ATTN_LOCAL = ATTN.local((4))
        ATTN_B32_VIEW = ATTN_LOCAL.view("uint32")

        # 这里的核心设计是，4个warp共同加载一个 k_tile [8, head_dim] 到 smem（完整的），K V 有一个CTA的4个warp的共同负责
        @T.inline
        def load_k_to_smem(stage, row_offset):
            # 一轮加载的总字节数//一轮总加载数(4 * 32) * 16B
            turns = (K_TILE_N * D * 2) // (4 * 32 * 16)
            # if turns == 0: # 不足一轮 D < 128
            row_per_turns = (128 * 16) // (D * 2)

            # 计算搬运起始位置，假设head dim > 128
            for i in range(turns):
                row = i * row_per_turns + ((warp_id * 32 + lane_id) * 16) // (D * 2)
                col = ((warp_id * 32 + lane_id) * 8) % D

                # 根据地址进行搬运，经过L2缓存
                T.ptx.cp_async(
                    KSMEM.ptr_to([stage, row, col]),
                    K.ptr_to([row_offset + row, col]),
                    16,
                )

        @T.inline
        def load_v_to_smem(outer_stage, row_offset):
            turns = (V_TILE_N * D * 2) // (4 * 32 * 16)
            row_per_turns = (128 * 16) // (D * 2)

            for i in range(turns):
                row = i * row_per_turns + ((warp_id * 32 + lane_id) * 16) // (D * 2)
                col = ((warp_id * 32 + lane_id) * 8) % D

                # 根据地址进行搬运，经过L2缓存
                T.ptx.cp_async(
                    VSMEM.ptr_to([outer_stage, row, col]),
                    K.ptr_to([row_offset + row, col]),
                    16,
                )

        # Q加载到REG，由于Q是每个warp单独负责计算的，因此Q直接写入每个warp的REG，可以参与的线程只有32个
        @T.inline
        def load_q_to_req():
            # Q_tile = [16, 16]

            # NOTE: [Local Memory 风险 - 循环未展开导致动态索引]
            # 此处使用普通 range(Q_TILES) 而非 T.unroll(Q_TILES)，循环变量 i 在 TIR 中是运行时变量。
            # 下方 Q_CP_VIEW[i, j] 使用 i 索引寄存器视图，若后端未自动完全展开该循环，
            # 会导致整个 Q_REG 因运行时动态寻址而被分配到 Local Memory！应改为 T.unroll(Q_TILES)。

            for i in range(Q_TILES):
                # 内层循环，一个循环负责加载4个元素，(16 * 16) / 32 = 4
                for j in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                    row = lane_id // 4 + (j % 2) * 8  # 行数
                    col = (lane_id % 4) * 2 + (j // 2) * 8 + i * 16

                    Q_CP_VIEW[i, j] = T.ptx.ld(
                        Q.ptr_to([row, col]),
                        "uint32",
                        "b32",
                        space="global",
                    )

        @T.inline
        def flash_attn(q_tile, k, v, o_tile, q_row_start):
            # Buffer / view 参数引用已有存储，不在接口处复制数据。
            # q_tile: float16[Q_TILE_M, D]，完整 Q 块，块内行号从 0 开始。
            # k、v:   float16[N, D]，完整 K/V 序列的全局内存 Buffer。
            # o_tile: float16[Q_TILE_M, D]，写入当前 Q 块对应的输出。
            # 当前接口约定：非 causal，无尾块；N % K_V_TILE_N == 0，D % MMA_K == 0。
            # 目标：o_tile = softmax(q_tile @ k.T / sqrt(D)) @ v。
            # TODO: 初始化逐行 max、sum 和 float32 输出累加器。
            # TODO: 遍历 K/V tiles，执行 QK、online softmax 和 PV 累加。
            # TODO: 用逐行 sum 归一化累加结果，转换为 float16 写入 o_tile。
            # 仅定义接口，计算实现待补充。
            T.evaluate(0)

        @T.inline
        def CLEAN_QK_REG():
            for i in T.unroll((4)):  # pyright: ignore[reportGeneralTypeIssues]
                QK_LOCAL_VIEW[i] = T.float32(0.0)

        # 内层PIPE LINE
        @T.inline
        def CAL_Q_K(outer_stage, k_offset):

            load_k_to_smem(outer_stage, k_offset)
            # TODO: barrier 同步加载完成，等待CTA内的所有warp加载完成

            # 内部实现二级流水线，计算 Q_tile * K_tile 的结果[16, 8]
            # 清理QK Acc
            CLEAN_QK_REG()

            MMA_Q_K(outer_stage)  # 计算出一个Attention的小块

            # Q x K_TILE^T的片段行内规约，online_softmax(Q x K_TILE ^T)

            # attention_tile x v_tile, 结果累加到 res 中

        @T.inline
        def ATTN_ROW(outer_stage):
            """
            Attention 计算过程分为两步
            1. 计算 Q_TILE x K_TILE ^ T, 并执行规约
            """
            # QK 连续视图
            # [0, 1] <-> [[0,0], [0,1]]
            # [2, 3] <-> [[8,0], [8,1]]

            # NOTE: [Local Memory 风险 - 运行时 outer_stage 索引局部数组]
            # 本函数中所有 [outer_stage, ...] 下标访问（ATTN_TILE_ROW_MAX_LOCAL、QK_ACC_VIEW、ATTN_TILE_DIFF_LOCAL），
            # 如果调用方传入的 outer_stage 是运行时变量（例如 ko % OUTER_STAGE），
            # 这三个 T.alloc_local 缓冲区都会因非常量索引退化到 Local Memory。

            # softmax 的输入为 QK / sqrt(D)，先缩放再计算逐行 max、exp 和 sum。
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                QK_ACC_VIEW[i] *= T.float32(SOFTMAX_SCALE)

            # 计算Attention的最大值，并且维护
            for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                CURR_QK_ROW_MAX[i] = T.max(QK_ACC_VIEW[i * 2], QK_ACC_VIEW[i * 2 + 1])

            # 外层迭代，迭代laneid
            for st in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                # 内层迭代，迭代每个线程需要维护的2 row
                for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]

                    # warp shuffle 规约，迭代每一行的最大值
                    QK_ROW_MAX_LOCAL[i] = T.max(
                        QK_ROW_MAX_LOCAL[i],
                        T.tvm_warp_shuffle_xor(
                            T.uint32(0xFFFFFFFF),
                            QK_ROW_MAX_LOCAL[i],
                            1 << st,  # lane_mask，
                            4,  # 4个lane [0,3]
                            32,  # warp size
                        ),
                    )

            # 以上计算完了每一行的最大值，接下来计算 diff，也就是新旧最大值的差值

            # NOTE：[已解决]有可能存在乱序问题吗？考虑两次，显然可能导致有一次漏加。同OUTER pipeline
            # 接下来维护 QK_TILE 的最大值变化情况
            for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                z = CURR_QK_ROW_MAX_LOCAL[i] - QK_ROW_MAX_LOCAL[i]
                QK_ROW_MAX_DIFF_LOCAL[i] = T.max(T.float32(0.0), z)  # 更新QK的变化值
                QK_ROW_MAX_LOCAL[i] = T.max(
                    QK_ROW_MAX_LOCAL[i], CURR_QK_ROW_MAX_LOCAL[i]
                )

            # alpha: exp(-QK_ROW_DIFF)
            # exp(x_1 - z_old) * exp(-(z_new - z_old)) = exp(x_1 - z_new)
            # res = res * exp(x_1 - z_old)
            # 外层迭代行数
            for r in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                alpha = T.exp(-QK_ROW_MAX_DIFF_LOCAL[outer_stage, r])
                QK_ROW_SUM_LOCAL[r] *= alpha
                # 内层迭代 res tiles
                for i in T.unroll(
                    RES_TILES
                ):  # pyright: ignore[reportGeneralTypeIssues]
                    RES_LOCAL[i, r * 2] *= alpha
                    RES_LOCAL[i, r * 2 + 1] *= alpha

            # 更新当前attention, atteniton = attention - attn_row_max
            # Attention = exp(QK[i,j] - max(QK[i]))
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                QK_ACC_VIEW[i] = T.exp(QK_ACC_VIEW[i] - QK_ROW_MAX_LOCAL[i // 2])

            # 一行额外的规约，计算 qk_row_sum
            for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                s = QK_ACC_VIEW[i * 2] + QK_ACC_VIEW[i * 2 + 1]
                # warp shuffle calculate row sum
                for j in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                    s = s + T.tvm_warp_shuffle_xor(
                        T.uint32(0xFFFFFFFF),
                        s,
                        1 << j,  # lane_mask，
                        4,  # 4个lane [0,3]
                        32,  # warp size
                    )
                # 每4个相邻的thread一组，维护qk_row_sum
                QK_ROW_SUM_LOCAL[i] += s

            # fp32 -> fp16
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                ATTN_LOCAL[i] = T.float16(QK_ACC_VIEW[i])

        # 接下来真实计算
        # res = exp(QK - Attn_max) @ V + res * exp(-attn_diff)

        @T.inline
        def MMA_ATTN_V(out_stage):
            # 计算ATTN x V，规约的中间维度 = 8
            # RES = RES + ATTN x V，其中, ATTN = fp32, V = fp16

            # NOTE: [Local Memory 风险 - 原位类型转换与寄存器移位]
            # 1. 不要在 QK_REG (float32) 上通过 QK_FP16_VIEW 原位写入半精度并强转指针传给 mma，
            #    跨位宽的原位指针别名会阻碍编译器 SROA 标量化，导致 QK_REG 落入 Local Memory。
            #    正确做法：单独 T.alloc_local((4,), "float16") 存放 P_frag，编译器会通过生命周期分析自动复用物理寄存器。
            # 2. 切勿使用非 T.unroll 的循环对寄存器数组做移位/重排，任何非常量下标或手动行列转置循环都会退化到 Local Memory。

            # 计算 Attn_tile x V
            # 本次使用复制出一段新的寄存器

            N_STPS = D // MMA_N  # 沿输出列分块，PV 的规约维度固定为 8。

            # 本函数计算过程和QxK^T计算相同

            # 二级流水线，预填充
            for s in T.unroll(
                INNER_STAGE - 1
            ):  # pyright: ignore[reportGeneralTypeIssues]
                if s < N_STPS:
                    T.warp.copy(
                        V[s, 0:8, 0:MMA_N],
                        VSMEM[out_stage, 0:8, (s * MMA_N) : ((s + 1) * MMA_N)],
                        dispatch="ldstmatrix",
                    )

            # 执行流水线
            for n in T.unroll(N_STPS):  # pyright: ignore[reportGeneralTypeIssues]
                pre_n: T.let = n + INNER_STAGE - 1
                pre_s: T.let = pre_n % INNER_STAGE

                if pre_n < N_STPS:
                    T.warp.copy(
                        V[pre_s, 0:8, 0:MMA_N],
                        VSMEM[0:8, (pre_n * MMA_N) : ((pre_n + 1) * MMA_N)],
                        dispatch="ldstmatrix",
                    )

                s: T.let = n % INNER_STAGE

                attn_ptrs = [ATTN_B32_VIEW.ptr_to([i]) for i in range(2)]
                # 当前流水线槽内的一个 b32，保存 V 的两个 FP16 元素。
                v_ptrs = [V_B32_VIEW.ptr_to([s, 0])]

                res_ptrs = [RES_LOCAL.ptr_to([n, 0])]

                T.ptx.mma(
                    "m16n8k8",
                    "row",
                    "col",
                    "float32",
                    "float16",
                    "float16",
                    "float32",
                    res_ptrs,
                    attn_ptrs,
                    v_ptrs,
                    res_ptrs,
                )

            # 计算流水线结束。接下来 / attn_row_max
            for n in T.unroll(N_STPS):  # pyright: ignore[reportGeneralTypeIssues]
                for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                    RES[n, i] = RES[n, i] // QK_ROW_SUM_LOCAL[i // 2]

        @T.inline
        def MMA_Q_K(out_stage):
            # QxK的中间规约维度是16
            K_STPS = D // 16
            # 实现一个内层流水线，注意，load_smem_to_reg 是可以使用寄存器依赖检查来同步的，不需要手动写同步原语

            # 预填充
            # inner stage - 1 法
            for s in T.unroll(
                INNER_STAGE - 1
            ):  # pyright: ignore[reportGeneralTypeIssues]
                if s < K_STPS:
                    # TODO: 确认TIRx lowering的结果，是否支持swizzle
                    T.warp.copy(
                        K[s, 0:8, 0:MMA_K],
                        KSMEM[out_stage, 0:8, (s * MMA_K) : ((s + 1) * MMA_K)],
                        dispatch="ldstmatrix",
                    )

            for ki in T.unroll(K_STPS):  # pyright: ignore[reportGeneralTypeIssues]
                # 预发起下一段的加载，加载的部分可以轻微错开

                # NOTE: [Local Memory 风险 - T.let 绑定运行时 Var]
                # T.let 会生成 TIR LetStmt 运行时变量（而非编译期常量）。
                # 若 T.warp.copy 或 ptr_to 在 Lowering 时未先将 pre_s / s 常量折叠，
                # K_REG[pre_s, ...] 和 K_B32_VIEW.ptr_to([s, i]) 会被视为动态索引，导致 K_REG 退化到 Local Memory。

                pre_k: T.let = ki + INNER_STAGE - 1
                pre_s: T.let = pre_k % INNER_STAGE

                if pre_k < K_STPS:
                    T.warp.copy(
                        K[pre_s, 0:8, 0:MMA_K],
                        KSMEM[out_stage, 0:8, (pre_k * MMA_K) : ((pre_k + 1) * MMA_K)],
                        dispatch="ldstmatrix",
                    )

                # 发起MMA计算

                s: T.let = ki % INNER_STAGE  # 计算归属的inner pipeline status

                # NOTE: [Local Memory 风险 - ptr_to 取动态偏移指针]
                # T.ptx.mma 要求操作数绑定到静态寄存器。若 out_stage、ki、s 中有任何一个在生成 C 代码时
                # 不是编译期常量，对局部数组取地址 &arr[var]（Pointer Escape）会直接迫使对应数组落入 Local Memory。

                c_ptrs = [QK_ACC_VIEW.ptr_to([i]) for i in range(4)]
                q_ptrs = [Q_CP_VIEW.ptr_to([ki, i]) for i in range(4)]
                k_ptrs = [K_B32_VIEW.ptr_to([s, i]) for i in range(2)]

                T.ptx.mma(
                    "m16n8k16",
                    "row",
                    "col",
                    "float32",
                    "float16",
                    "float16",
                    "float32",
                    c_ptrs,
                    q_ptrs,
                    k_ptrs,
                    c_ptrs,
                )

    return kernel
