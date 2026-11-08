import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def Gemm(M,N,K, block_M = 128, block_K = 64, block_N = 128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((N, N), "float32"),
    ):
        m_tiles = T.ceildiv(M, block_M)
        n_tiles = T.ceildiv(N, block_N)



