import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg

SM_COUNT = 66

MMA_M = 16
MMA_K = 16
MMA_N = 8

Q_TILE_N = 16
K_V_TILE_N = 8

KV_GMEM_SMEM_PIPE_STEP = 3  # 先暂定3，通信掩盖延迟
SMEM_TO_REG_STEP = 2  # smem load to register 的深度


def flash_attention_v1(M: int, N: int, D: int):

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
