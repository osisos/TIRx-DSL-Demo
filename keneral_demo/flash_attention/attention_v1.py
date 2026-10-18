from typing import cast

import tvm
from tvm.ir import Var
from tvm.tirx import script as T
from tvm.tirx.layout import TileLayout, S, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.script import tile as Tx

BLOCK_CNT = 66
WARP_PER_CTA = 4

MMA_M = 16
MMA_K = 16
MMA_N = 8

Q_TILE_M = 16
K_V_TILE_N = 8

KV_GMEM_SMEM_PIPE_STEP = 2  # 先暂定3，通信掩盖延迟
SMEM_TO_REG_STEP = 2  # smem load to register 的深度

"""
    核心设计：每个warp负责一个Q_tile的对应的计算
"""


def flash_attention_v1(M: int, N: int, D: int):

    Q_TILES = D // MMA_K
    K_TILES = D // MMA_N
    V_TILES = D // MMA_N
    OUT_TILES = D // MMA_N

    Q_REG_PER_THREAD = (D * Q_TILE_M * 2) // (32 * 4)
    # 每个thread负责的Q寄存器存储个数（总字节数/(总线程数 * 每个寄存器的字节数)
    Q_REG_CP_TURNS = Q_REG_PER_THREAD // 4  # 一次搬运填满4个32bit寄存器

    # Q layout[16, head_dim]，先假设head dim % 16 = 0
    Q_LAYOUT = TileLayout(
        S[(2, 8, Q_TILES, 2, 4, 2) : (2, 4 @ laneid, 8, 4, 1 @ laneid, 1)],
    )

    # Out Layout
    OUT_LAYOUT = TileLayout(
        S[(2, 8, OUT_TILES, 4, 2) : (2, 4 @ laneid, 4, 1 @ laneid, 1)],
    )

    K_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [KV_GMEM_SMEM_PIPE_STEP, K_V_TILE_N, D],
    )

    V_LAYOUT = mma_shared_layout(
        "float16",
        SwizzleMode.SWIZZLE_128B_ATOM,
        [KV_GMEM_SMEM_PIPE_STEP, K_V_TILE_N, D],
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
            [KV_GMEM_SMEM_PIPE_STEP, K_V_TILE_N, D],
            "float16",
            layout=K_LAYOUT,
        )
        VSEME = pool.alloc(
            [KV_GMEM_SMEM_PIPE_STEP, K_V_TILE_N, D],
            "float16",
            layout=V_LAYOUT,
        )

        pool.commit()

        # 申请Q缓存
        Q_REG = T.alloc_local((M, D), "float16", layout=Q_LAYOUT, align=4)  # 4B对齐
        Q_LOCAL = Q_REG.local(
            Q_REG_CP_TURNS, 8, layout=Q_LAYOUT.storage()
        )  # Global memeory -> register 的拷贝视图
        Q_CP_VIEW = Q_LOCAL.view("uint32")

        # 这里的核心设计是，4个warp共同加载一个 k_tile [8, head_dim] 到 smem（完整的），K V 有一个CTA的4个warp的共同负责
        @T.inline
        def load_kv_to_smem(stage, row_offset):
            # 一轮加载的总字节数//一轮总加载数(4 * 32) * 16B
            turns = (K_V_TILE_N * D * 2) // (4 * 32 * 16)
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

                T.ptx.cp_async(
                    VSEME.ptr_to([stage, row, col]),
                    V.ptr([row_offset + row, col]),
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

    return kernel
