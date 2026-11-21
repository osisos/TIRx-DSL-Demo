"""bug-4056.py 的 Triton 对照：Out = A @ B + sum(H, axis=1)[:, None]。

    python tilelang-demo/bug-4056-triton.py
    python tilelang-demo/bug-4056-triton.py --run

默认离线编译到 sm_89，导出 TTIR、TTGIR、LLVM IR 和 PTX；不分配 GPU 张量。
与 TileLang 原例仅 lowering 不同，这里完成编译（含 ptxas，不需要 NVCC）。
--run 额外在本机 GPU 上执行并与 PyTorch FP32 参考值比较。
"""

import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def kernel(A, B, H, Out):
    # 单个 CTA，64x64 tile；num_warps=4 对应原例的 128 个线程。
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    k = tl.arange(0, 64)
    a = tl.load(A + rows[:, None] * 64 + k[None, :])
    b = tl.load(B + k[:, None] * 64 + cols[None, :])
    acc = tl.zeros((64, 64), tl.float32)
    acc = tl.dot(a, b, acc)

    # 独立输入先转 FP32，再做逐行归约，对应 wide / stat。
    wide = tl.load(H + rows[:, None] * 64 + cols[None, :]).to(tl.float32)
    stat = tl.sum(wide, axis=1)

    # 关键对照点：归约结果广播后，与 GEMM 累加器相加。
    out = stat[:, None] + acc
    tl.store(Out + rows[:, None] * 64 + cols[None, :], out)


def compile_and_dump(output_dir):
    compiled = triton.compile(
        ASTSource(
            fn=kernel,
            signature={"A": "*fp16", "B": "*fp16", "H": "*fp16", "Out": "*fp32"},
        ),
        target=GPUTarget("cuda", 89, 32),
        options={"num_warps": 4, "num_stages": 1},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        (output_dir / f"bug-4056.{stage}").write_text(compiled.asm[stage])
    print(f"Triton {triton.__version__}: sm_89 编译成功，IR: {output_dir}")
    print("可在 TTGIR 中搜索 convert_layout、tt.dot、tt.reduce，观察布局衔接。")


def verify():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--run 需要可用的 CUDA GPU；仅导出 IR 时无需 --run。")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    for case in ("random", "zero_h", "zero_gemm"):
        a, b, h = [torch.randn((64, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
        if case == "zero_h":
            h.zero_()
        elif case == "zero_gemm":
            a.zero_()
        out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
        kernel[(1,)](a, b, h, out, num_warps=4, num_stages=1)
        ref = a.float() @ b.float() + h.float().sum(dim=1, keepdim=True)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
        print(f"{case}: PASS, max_abs_error={(out - ref).abs().max().item():.8g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="额外执行 GPU 数值验证")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "tmp" / "triton-4056",
    )
    args = parser.parse_args()
    compile_and_dump(args.output_dir)
    if args.run:
        verify()
