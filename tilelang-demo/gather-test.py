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


import linecache
import re


def _strip_loc_exprs(line: str) -> str:
    """移除行内所有的 `loc(...)` 表达式（支持嵌套括号）。"""
    out = []
    i = 0
    n = len(line)
    while i < n:
        m = re.search(r"\s*\bloc\(", line[i:])
        if not m:
            out.append(line[i:])
            break
        out.append(line[i : i + m.start()])
        paren_start = i + m.end() - 1
        depth = 0
        j = paren_start
        in_str = False
        while j < n:
            ch = line[j]
            if ch == '"' and (j == 0 or line[j - 1] != "\\"):
                in_str = not in_str
            elif not in_str:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1
        i = j
    return "".join(out).rstrip()


def format_triton_ir(ir_text: str) -> str:
    """后处理 Triton TTIR/TTGIR：解析 #loc 映射，内联 Python 源码注释，并移除 #loc 噪音。"""
    lines = ir_text.splitlines()

    # 1. 解析所有 #loc 定义
    raw_locs: dict[str, tuple] = {}
    loc_def_re = re.compile(r"^#(loc\w*)\s*=\s*loc\((.+)\)\s*$")
    file_loc_re = re.compile(r'^"([^"]+)":(\d+):(\d+)$')
    named_loc_re = re.compile(r'^"[^"]*"\(#(loc\w*)\)$')
    callsite_re = re.compile(r"^callsite\(#(loc\w*)\s+at\s+#(loc\w*)\)$")

    body_lines: list[str] = []
    for line in lines:
        m = loc_def_re.match(line.strip())
        if m:
            loc_id, payload = m.group(1), m.group(2).strip()
            if payload == "unknown":
                raw_locs[loc_id] = ("unknown",)
            elif fm := file_loc_re.match(payload):
                raw_locs[loc_id] = ("file", fm.group(1), int(fm.group(2)))
            elif nm := named_loc_re.match(payload):
                raw_locs[loc_id] = ("ref", nm.group(1))
            elif cm := callsite_re.match(payload):
                # callsite(callee at caller)：优先展示用户代码中的 caller 位置
                raw_locs[loc_id] = ("callsite", cm.group(2), cm.group(1))
            continue
        body_lines.append(line)

    # 若文件中已经没有 #loc 定义（例如已被格式化过），直接原样返回，保证幂等
    if not raw_locs:
        return ir_text

    def resolve_loc(
        loc_id: str, visited: set[str] | None = None
    ) -> tuple[str, int] | None:
        if visited is None:
            visited = set()
        if loc_id in visited or loc_id not in raw_locs:
            return None
        visited.add(loc_id)
        entry = raw_locs[loc_id]
        kind = entry[0]
        if kind == "file":
            return (entry[1], entry[2])
        if kind == "ref":
            return resolve_loc(entry[1], visited)
        if kind == "callsite":
            caller, callee = entry[1], entry[2]
            return resolve_loc(caller, visited) or resolve_loc(callee, visited)
        return None

    # 2. 提取每行末尾的 loc(#locXX)，并借助括号栈把块尾 loc 回填到块首（如 scf.for / tt.reduce）
    trailing_loc_re = re.compile(r"\bloc\(#(loc\w*)\)\s*$")
    line_locs: list[tuple[str, int] | None] = [None] * len(body_lines)
    block_stack: list[int] = []

    for idx, line in enumerate(body_lines):
        stripped = line.strip()
        m = trailing_loc_re.search(stripped)
        loc_info = resolve_loc(m.group(1)) if m else None
        line_locs[idx] = loc_info

        if stripped.startswith("}") and block_stack:
            open_idx = block_stack.pop()
            if line_locs[open_idx] is None and loc_info is not None:
                line_locs[open_idx] = loc_info
            line_locs[idx] = None

        if stripped.endswith("{"):
            block_stack.append(idx)

    # 3. 生成清理后的 IR，在源码行变化处插入注释
    out_lines: list[str] = []
    last_source_loc: tuple[str, int] | None = None

    for idx, line in enumerate(body_lines):
        clean_line = _strip_loc_exprs(line)
        stripped = clean_line.strip()
        indent = clean_line[: len(clean_line) - len(clean_line.lstrip())]

        # 将超长的 tt.func 签名按参数拆成多行
        if (
            stripped.startswith("tt.func ")
            and "(" in clean_line
            and ") attributes" in clean_line
        ):
            prefix, rest = clean_line.split("(", 1)
            args_str, suffix = rest.rsplit(") attributes", 1)
            args = [a.strip() for a in args_str.split(", %")]
            if len(args) > 1:
                out_lines.append(f"{prefix}(")
                for a_idx, arg in enumerate(args):
                    arg_text = arg if a_idx == 0 else f"%{arg}"
                    comma = "," if a_idx < len(args) - 1 else ""
                    out_lines.append(f"{indent}    {arg_text}{comma}")
                out_lines.append(f"{indent}) attributes{suffix}")
                last_source_loc = line_locs[idx]
                continue

        loc_info = line_locs[idx]
        if (
            loc_info is not None
            and loc_info != last_source_loc
            and not stripped.startswith(("module", "tt.func", "tt.return", "^bb"))
            and "arith.constant" not in stripped
        ):
            filepath, lineno = loc_info
            src_line = linecache.getline(filepath, lineno).strip()
            short_file = Path(filepath).name
            if (
                out_lines
                and out_lines[-1].strip() != ""
                and not out_lines[-1].strip().endswith("{")
            ):
                out_lines.append("")
            if src_line:
                out_lines.append(f"{indent}// [{short_file}:{lineno}] {src_line}")
            else:
                out_lines.append(f"{indent}// [{short_file}:{lineno}]")
            last_source_loc = loc_info

        out_lines.append(clean_line)

    return "\n".join(out_lines) + "\n"


def format_ir_dir(ir_dir: Path) -> None:
    """批量格式化指定目录下的 .ttir / .ttgir 文件。"""
    files = sorted(list(ir_dir.rglob("*.ttir")) + list(ir_dir.rglob("*.ttgir")))
    for path in files:
        original = path.read_text()
        formatted = format_triton_ir(original)
        if formatted != original:
            path.write_text(formatted)
            print(f"已格式化: {path}")
        else:
            print(f"跳过（已是格式化状态）: {path}")


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
    parser.add_argument(
        "--raw-ir",
        action="store_true",
        help="保存未经去噪后处理的原始 TTIR/TTGIR",
    )
    parser.add_argument(
        "--format-dir",
        type=Path,
        default=None,
        help="仅对指定目录下的 .ttir/.ttgir 文件执行可读性后处理并退出（无需 GPU）",
    )
    args = parser.parse_args()
    if args.format_dir is not None:
        format_ir_dir(args.format_dir)
        return
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
                text = compiled.asm[stage]
                if stage in ("ttir", "ttgir") and not args.raw_ir:
                    text = format_triton_ir(text)
                (args.dump_dir / f"{name}.{stage}").write_text(text)
        error = (output - reference).abs().max().item()
        # 不同归约顺序在接近零处需要绝对容差。
        torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-4)
        ms = triton.testing.do_bench(launch)
        ir = compiled.asm.get("ttgir", "")
        if not args.raw_ir:
            ir = format_triton_ir(ir)
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
