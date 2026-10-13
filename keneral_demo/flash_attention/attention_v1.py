import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode

SM_COUNT = 66

MMA_M = 16
MMA_K = 16
MMA_N = 8

Q_TILE_N = 16
K_V_TILE_N = 8

KV_GMEM_SMEM_PIPE_STEP = 3  # 先暂定3，通信掩盖延迟
SMEM_TO_REG_STEP = 2  # smem load to register 的深度


def flash_attention_v1(M: int, N: int, D: int):

    Q_TILES = D // MMA_K
    K_TILES = D // MMA_N
    V_TILES = D // MMA_N
    OUT_TILES = D // MMA_N

    # Q layout[16, head_dim]，先假设head dim % 16 = 0
    Q_LAYOUT = TileLayout(
        S[(2, 8, Q_TILES, 2, 4, 2) : (2, 4 @ laneid, 8, 4, 1 @ laneid, 1)],
    )

    # Out Layout
    OUT_LAYOUT = TileLayout(
        S[(2, 8, OUT_TILES, 4, 2) : (2, 4 @ laneid, 4, 1 @ laneid, 1)],
    )

    K_LAYOUT = mma_shared_layout()

    @T.prim_func
    def kernel(
        Q: T.Buffer((M, D), "float16"),
        K: T.Buffer((N, D), "float16"),
        V: T.Buffer((N, D), "float16"),
        O: T.Buffer((M, D), "float16"),
    ):
        T.device_entry()
        T.evaluate(0)  # TODO: attention

        # 每个CTA负责一个Q_tile[M，D]，及其关联的所有输出。
        # Q、O 都驻留在寄存器中
        # K, V

    return kernel
