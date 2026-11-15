"""复现自定义 shared-memory 布局与 MMA 不兼容的问题。

运行（在仓库根目录）：
    conda activate tilelang
    python tilelang-demo/bug-annotation.py --baseline  # 默认布局，校验正确性
    python tilelang-demo/bug-annotation.py             # 自定义布局，复现问题

两个模式请分别启动进程；CUDA misaligned address 后不能继续复用该进程。
"""

import argparse

import tilelang
import tilelang.language as T
import torch

M = N = K = 64


@tilelang.jit
def bad_mma_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    annotate: bool = True,
):
    C = T.empty((M, N), "float32")
    with T.Kernel(1, threads=128) as bx:
        A_s = T.alloc_shared((M, K), "float16")
        B_s = T.alloc_shared((K, N), "float16")
        E = T.alloc_fragment((M, N), "float32")
        T.clear(E)

        # 双射（bijective）的细粒度行内异或布局：(i, j) -> (i, j ^ (i % 2))
        # 若仅做 T.copy 往返完全正确，但与 MMA 底层的 ldmatrix 16B 对齐要求不兼容
        if annotate:
            T.annotate_layout({B_s: T.Layout((K, N), lambda i, j: [i, j ^ (i % 2)])})

        T.copy(A, A_s)
        T.copy(B, B_s)
        T.gemm(A_s, B_s, E)  # Bug 触发点：MMA 路径直接跳过 B_s 布局校验并复用
        T.copy(E, C)
    return C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", action="store_true", help="不添加自定义布局，运行正确性对照"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("此示例需要 CUDA GPU，请使用本项目的 tilelang 环境运行。")

    torch.manual_seed(0)
    print(
        f"GPU: {torch.cuda.get_device_name()} (sm_{''.join(map(str, torch.cuda.get_device_capability()))})",
        flush=True,
    )
    print(
        f"模式: {'baseline（默认布局）' if args.baseline else 'bug（自定义 XOR 布局）'}",
        flush=True,
    )
    A = torch.randint(-2, 3, (M, K), device="cuda", dtype=torch.float16)
    B = torch.randint(-2, 3, (K, N), device="cuda", dtype=torch.float16)
    # 小整数输入的结果可精确表示，避免浮点误差干扰布局问题的判断。
    reference = A.float() @ B.float()
    torch.cuda.synchronize()

    print("开始编译并执行 GEMM...", flush=True)
    try:
        C = bad_mma_gemm(A, B, annotate=not args.baseline)
        torch.cuda.synchronize()  # 在这里捕获异步 CUDA 执行错误
    except RuntimeError:
        print(
            "GEMM 失败，详细错误见下方 traceback；请在新进程中运行下一次实验。",
            flush=True,
        )
        raise

    torch.testing.assert_close(C, reference, rtol=0, atol=0)
    print("PASS: GEMM 与 PyTorch 结果完全一致。", flush=True)
    if not args.baseline:
        print("当前环境未复现该布局问题。", flush=True)


if __name__ == "__main__":
    main()
