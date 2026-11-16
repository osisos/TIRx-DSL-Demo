"""Issue #8706 的原版 kernel 与一维索引加载版本对照。

运行：conda activate triton && python tilelang-demo/gather-test.py
IR 默认保存到 tmp/gather-8706，可搜索 ttg.convert_layout 并查看其所在循环。
来源：https://github.com/triton-lang/triton/issues/8706
原文的近 5x 加速来自 Ascend 上修改编译器后的实验，本脚本不修改编译器。
"""

import argparse
from pathlib import Path

import triton
import triton.testing
import torch
import triton.language as tl


@triton.jit
def test_gather(
    indices_ptr,  # [batch_size, seq_size] [128, 4000]
    table_ptr,  # [vocab_size, dim] [9000, 128]
    output_ptr,
    batch_size: tl.constexpr = 128,
    dim: tl.constexpr = 128,
    seq_size: tl.constexpr = 4000,
    vocab_size: tl.constexpr = 9000,
    BLOCK_R: tl.constexpr = 32,
):
    # 1 cta 负责 1 行
    b = tl.program_id(0)  # 这一行的含义是什么？cta id?

    d = tl.arange(0, dim)  # [0..32]

    r_base = tl.arange(0, BLOCK_R)

    # 临时收集，从这发起规约
    acc = tl.full(
        (BLOCK_R, dim), 0, tl.float32
    )  # 这个Acc, (BLOCK_R, dim) 是形状吗？ [BLOCK_R, dim] 是可以的吗？

    # [0, 32 ... 4000]
    for start in range(0, seq_size, BLOCK_R):  # 似乎这个一定是时间顺序？

        r = start + r_base
        valid = r < seq_size

        # 加载 idx [start, start + r]
        idx = tl.load(
            indices_ptr + b * seq_size + r,  # 偏移量？ row = b * seq_size,  col = r
            mask=valid,
            other=0,
        )

        idx = tl.where(idx < 0, idx + vocab_size, idx)  # ? 这是常用的写法吗

        values = tl.load(
            table_ptr + idx[:, None] * dim + d[None, :],  # 这个是 parallel？
            # 语法，生成二维矩阵指针
            # [M] -> [M, 1]
            # [D] -> [1, D]
            mask=valid[:, None],
            other=0.0,
        )

        acc += values

    result = tl.sum(acc, axis=0)

    # output [b, d]
    tl.store(output_ptr + b * dim + d, result)


@triton.jit
def issue_gather(
    indices_ptr,
    table_ptr,
    output_ptr,
    output_numel,
    seq_size,
    grid_size,
    BLOCK_D: tl.constexpr,  # 128
    BLOCK_R: tl.constexpr,  # 32
):
    # 保留 issue 的二维索引加载和 grid-output_stride 循环，以观察原始布局问题。
    output_numel = 16384  # 【128， 128】
    seq_size = 4000
    pid = tl.program_id(0)
    output_start = pid * BLOCK_D
    output_stride = grid_size * BLOCK_D

    for output_offset in range(output_start, output_numel, output_stride):
        output_index = output_offset + tl.arange(0, BLOCK_D)[:, None]
        r_base = tl.arange(0, BLOCK_R)[None, :]
        b = output_index // 128
        d = output_index % 128
        acc = tl.full([BLOCK_D, BLOCK_R], 0, tl.float32)
        for start in range(0, seq_size, BLOCK_R):
            r = start + r_base
            valid = r < seq_size
            idx = tl.load(indices_ptr + r + 4000 * b, valid, other=0.0)
            adjusted = idx + tl.full([BLOCK_D, BLOCK_R], 9000, tl.int32)
            idx = tl.where(idx < 0, adjusted, idx)
            values = tl.load(table_ptr + d + 128 * idx, valid)
            values = tl.broadcast_to(values, [BLOCK_D, BLOCK_R])
            acc = tl.where(valid, acc + values, acc)
        result = tl.sum(acc, 1)[:, None]
        tl.store(output_ptr + output_index, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=["both", "issue", "gather"], default="both")
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=range(1, 129),
        default=128,
        metavar="1..128",
        help="仅用于 issue kernel",
    )
    parser.add_argument("--num-warps", type=int, choices=[4, 8, 16, 32], default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-dir", type=Path, default=Path("tmp/gather-8706"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 和安装了 PyTorch/Triton 的环境。")

    print(
        f"PyTorch {torch.__version__}; Triton {triton.__version__}; CUDA {torch.version.cuda}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name()}; capability: {torch.cuda.get_device_capability()}"
    )
    print(f"seed={args.seed}, num_warps={args.num_warps}, grid_size={args.grid_size}")
    torch.manual_seed(args.seed)
    indices = torch.randint(-1000, 9000, (128, 4000), dtype=torch.int64, device="cuda")
    table = torch.randn((9000, 128), dtype=torch.float32, device="cuda")
    adjusted = torch.where(indices < 0, indices + 9000, indices)
    reference = table[adjusted].sum(dim=1)
    args.dump_dir.mkdir(parents=True, exist_ok=True)

    for name in ("issue", "gather"):
        if args.kernel not in ("both", name):
            continue
        output = torch.full((128, 128), float("nan"), device="cuda")

        def launch():
            if name == "issue":
                return issue_gather[(args.grid_size,)](
                    indices,
                    table,
                    output,
                    16384,
                    4000,
                    args.grid_size,
                    BLOCK_D=128,
                    BLOCK_R=32,
                    num_warps=args.num_warps,
                )
            return test_gather[(128,)](
                indices,
                table,
                output,
                num_warps=args.num_warps,
            )

        compiled = launch()
        torch.cuda.synchronize()
        # 在校验前导出，方便即使数值校验失败也能检查生成代码。
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            if stage in compiled.asm:
                (args.dump_dir / f"{name}.{stage}").write_text(compiled.asm[stage])
        error = (output - reference).abs().max().item()
        # 不同归约顺序在接近零处需要绝对容差。
        torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-4)
        ms = triton.testing.do_bench(launch)
        ir = compiled.asm.get("ttgir", "")
        conversions = ir.count("ttg.convert_layout")
        print(
            f"{name}: PASS, max_abs_error={error:.6g}, latency={ms:.4f} ms, "
            f"shared={compiled.metadata.shared} bytes, convert_layout 总数={conversions}"
        )
        for line_no, line in enumerate(ir.splitlines(), 1):
            if "ttg.convert_layout" in line:
                print(f"  {name}.ttgir:{line_no}: {line.strip()}")
    print(f"IR 已保存：{args.dump_dir.resolve()}")
    print("总数包含循环外转换；请查看循环内 embedding load 前的指针布局转换。")


if __name__ == "__main__":
    main()
