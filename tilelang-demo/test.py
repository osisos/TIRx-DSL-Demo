import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2])
def vector_add(n=1024, block=128):
    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        # 每个 block 处理 128 个元素，每个线程计算一次加法。
        with T.Kernel(T.ceildiv(n, block), threads=128) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]

    return main


if __name__ == "__main__":
    a = torch.randn(1024, device="cuda")
    b = torch.randn_like(a)

    kernel = vector_add()
    c = kernel(a, b)  # out_idx=[2] 表示 C 由 TileLang 分配并返回。

    torch.testing.assert_close(c, a + b)
    print("PASS: TileLang vector_add 与 PyTorch 结果一致")
