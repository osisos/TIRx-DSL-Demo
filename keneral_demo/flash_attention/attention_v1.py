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

OUTER_STAGE = 2  # 先暂定3，通信掩盖延迟
INNER_STAGE = 2  # smem load to register 的深度

"""
    核心设计：每个warp负责一个Q_tile的对应的计算
"""


def flash_attention_v1(M: int, N: int, D: int):

    Q_TILES = D // MMA_K
    K_TILES = D // MMA_N
    V_TILES = D // MMA_N
    RES_TILES = D // MMA_N

    Q_REG_PER_THREAD = (D * Q_TILE_M * 2) // (32 * 4)
    # 每个thread负责的Q寄存器存储个数（总字节数/(总线程数 * 每个寄存器的字节数)
    Q_REG_CP_TURNS = Q_REG_PER_THREAD // 4  # 一次搬运填满4个32bit寄存器

    # 以下布局针对 mma.m16n8k16；8 个 lane group、每组 4 lanes 等是固定映射。
    # Q layout[16, D]，假设 D % MMA_K == 0。
    Q_LAYOUT = TileLayout(
        S[(2, 8, Q_TILES, 2, 4, 2) : (2, 4 @ laneid, 8, 4, 1 @ laneid, 1)],
    )

    # Out Layout
    RES_LAYOUT = TileLayout(
        S[(2, 8, RES_TILES, 4, 2) : (2, 4 @ laneid, 4, 1 @ laneid, 1)],
    )

    K_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [OUTER_STAGE, K_TILE_N, D],
    )

    V_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [OUTER_STAGE, V_TILE_N, D],
    )

    K_REG_LAYOUT = TileLayout(
        S[(INNER_STAGE, 8, 2, 4, 2) : (4, 4 @ laneid, 2, 1 @ laneid, 1)]
    )

    V_REG_LAYOUT = TileLayout(
        S[(INNER_STAGE, 2, 4, 2, 8) : (4, 2, 1 @ laneid, 1, 4 @ laneid)]
    )

    QK_REG_LAYOUT = TileLayout(
        S[(OUTER_STAGE, 2, 8, 4, 2) : (4, 2, 4 @ laneid, 1 @ laneid, 1)]
    )

    ATTN_TILE_ROW_MAX_LAYOUT = TileLayout(
        S[(OUTER_STAGE, 2, 8) : (2, 1, 4 @ laneid)] + R[4 : 1 @ laneid]
    )

    ATTN_OLD_ROW_MAX_LAYOUT = TileLayout(
        S[(2, 8) : (1, 4 @ laneid)] + R[4 : 1 @ laneid]
    )

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
            layout=K_LAYOUT,
        )
        VSEME = pool.alloc(
            [OUTER_STAGE, K_TILE_N, D],
            "float16",
            layout=V_LAYOUT,
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
        Q_REG = T.alloc_local((Q_TILE_M, D), "float16", layout=Q_LAYOUT, align=4)
        # Global memeory -> register 的拷贝视图
        Q_LOCAL = Q_REG.local(Q_REG_CP_TURNS, 8)  # 直接物理连续视图
        Q_CP_VIEW = Q_LOCAL.view("uint32")

        # K 寄存器
        K_REG = T.alloc_local(
            (INNER_STAGE, K_TILE_N, 16),
            "float16",
            layout=K_REG_LAYOUT,
        )

        K_LOCAL = K_REG.local(INNER_STAGE, 4)  # 直接物理连续视图
        K_B32_VIEW = K_LOCAL.view("uint32")

        V_REG = T.alloc_local(
            (INNER_STAGE, V_TILE_N, 16),
            "float16",
            layout=V_REG_LAYOUT,
        )

        # Q/K 输入为 FP16，QK 使用 FP32 累加，对应 MMA 的 f32.f16.f16.f32。
        QK_REG = T.alloc_local(
            (OUTER_STAGE, 16, 8),  # 这个常数，后续可以优化下
            "float32",
            layout=QK_REG_LAYOUT,
        )

        # local 视图
        QK_LOCAL_VIEW = QK_REG.local()
        QK_ACC_VIEW = QK_REG.local(
            OUTER_STAGE, 4
        )  # 每个 lane 每个 stage 持有 4 个 FP32 累加元素。

        # NOTE: attn row max 规约的中间结果，之所以是这个排布是因为 row:1 的相关的寄存器在[0-3] laneid 中
        ATTN_TILE_ROW_MAX = T.alloc_local(
            (OUTER_STAGE, 16, 1), "float32", layout=ATTN_TILE_ROW_MAX_LAYOUT
        )
        ATTN_TILE_ROW_MAX_LOCAL = ATTN_TILE_ROW_MAX.local()

        ATTN_ROW_MAX = T.alloc_local((16, 1), "float32", layout=ATTN_OLD_ROW_MAX_LAYOUT)
        ATTN_ROW_MAX_LOCAL = ATTN_ROW_MAX.local()

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
                    VSEME.ptr_to([outer_stage, row, col]),
                    K.ptr_to([row_offset + row, col]),
                    16,
                )

        # Q加载到REG，由于Q是每个warp单独负责计算的，因此Q直接写入每个warp的REG，可以参与的线程只有32个
        @T.inline
        def load_q_to_req():
            # Q_tile = [16, 16]
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
            for i in T.unroll(
                (4 * OUTER_STAGE)
            ):  # pyright: ignore[reportGeneralTypeIssues]
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
            # QK 连续视图，[0,1,2,3]
            ATTN_TILE_ROW_MAX_LOCAL[0] = T.max(QK_ACC_VIEW[0], QK_ACC_VIEW[1])
            ATTN_TILE_ROW_MAX_LOCAL[1] = T.max(QK_ACC_VIEW[2], QK_ACC_VIEW[3])

            # warp_shuffle
            for h in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                idx = outer_stage * 2

            # TODO: 没写完

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
                        K_REG[s, 0:8, 0:MMA_K],
                        KSMEM[out_stage, 0:8, (s * MMA_K) : ((s + 1) * MMA_K)],
                        dispatch="ldstmatrix",
                    )

            # TODO: 看下这里的普通循环写法能不能自动展开
            for ki in T.unroll(K_STPS):  # pyright: ignore[reportGeneralTypeIssues]

                # 预发起下一段的加载，加载的部分可以轻微错开
                pre_k: T.let = ki + INNER_STAGE - 1  #
                pre_s: T.let = pre_k % INNER_STAGE

                if pre_k < K_STPS:
                    T.warp.copy(
                        K_REG[pre_s, 0:8, 0:MMA_K],
                        KSMEM[out_stage, 0:8, (pre_k * MMA_K) : ((pre_k + 1) * MMA_K)],
                        dispatch="ldstmatrix",
                    )

                # 发起MMA计算

                s: T.let = ki % INNER_STAGE  # 计算归属的inner pipeline status

                c_ptrs = [QK_ACC_VIEW.ptr_to([out_stage, i]) for i in range(4)]

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
