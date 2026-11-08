import tilelang
import tilelang.language as T


def Gemm(M, N, K, block_M=128, block_K=64, block_N=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        m_tiles = T.ceildiv(M, block_M)
        n_tiles = T.ceildiv(N, block_N)
        num_ctas = 66  # 4070 ti super

        with T.Kernel(num_ctas, threads=128) as pid:
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_N, block_K), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for bx, by in T.Persistent(
                [m_tiles, n_tiles],
                num_ctas,
                pid,
                group_size=8,
            ):
                m_idx = block_M * bx
                n_idx = block_N * by

                T.clear(C_local)

                # 不加软件流水线，用最纯粹的串行循环观察编译 Pass 的变化
                for ko in T.serial(T.ceildiv(K, block_K)):
                    k_idx = block_K * ko
                    T.copy(A[m_idx : m_idx + block_M, k_idx : k_idx + block_K], A_shared)
                    T.copy(B[n_idx : n_idx + block_N, k_idx : k_idx + block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[m_idx : m_idx + block_M, n_idx : n_idx + block_N])

    return main


if __name__ == "__main__":
    func = Gemm(4096, 4096, 4096)

    print("=== 1. 初始高层 TileLang TIR (未经过 Lowering) ===")
    print(func.script())

    # 执行完整编译流程（可通过环境变量 TILELANG_PASS_DIFF=html 或 terminal 观察每个 Pass 的前后变化）
    artifact = tilelang.lower(func, target="cuda")

    print("\n=== 2. Lower 后的底层 Device TIR ===")
    print(artifact.device_mod.script())

    print("\n=== 3. 最终生成的 CUDA C++ 源码 ===")
    print(artifact.kernel_source)
