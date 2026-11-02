"""验证 flash_attention_v1：单 head、FP16 输入/输出、包含对角线的 causal mask。

仓库根目录运行：
    .venv/bin/python -m keneral_demo.flash_attention.verify_attention_v1 --reference-only
    .venv/bin/python -m keneral_demo.flash_attention.verify_attention_v1
    .venv/bin/python -m keneral_demo.flash_attention.verify_attention_v1 --run-cuda

默认验证参考实现并检查 CUDA 代码生成；--run-cuda 才执行 GPU 数值对比。
加 --long 覆盖超过 BLOCK_CNT 个 CTA 的 Q 块调度。无需 PyTorch / pytest。
当前测试按 CAUSAL_MASK 的 k <= q 语义，而非 Flash_Attention 中旧的非 causal 注释。
"""

import argparse

import numpy as np


def reference_attention(q, k, v):
    """以实际 FP16 输入为基准，用 FP64 稳定 softmax 构造独立 CPU oracle。

    分块处理 Q 以限制长序列测试的内存；非方阵也采用左上对齐的 causal mask。
    """
    m, d = q.shape
    n = k.shape[0]
    k64, v64 = k.astype(np.float64), v.astype(np.float64)
    result = np.empty((m, d), dtype=np.float64)
    for start in range(0, m, 64):
        stop = min(start + 64, m)
        scores = (q[start:stop].astype(np.float64) @ k64.T) / np.sqrt(d)
        mask = np.arange(n)[None, :] > np.arange(start, stop)[:, None]
        scores[mask] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=1, keepdims=True)
        result[start:stop] = weights @ v64
    return result


def check_reference():
    # Q=0 时，第 i 行应为 V[:i+1] 的均值，直接验证缩放以外的 mask/归一化。
    q = np.zeros((16, 128), dtype=np.float16)
    k = np.ones((24, 128), dtype=np.float16)
    v = (np.arange(24 * 128).reshape(24, 128) % 31 - 15).astype(np.float16)
    expected = np.cumsum(v.astype(np.float64), axis=0)[:16]
    expected /= np.arange(1, 17)[:, None]
    np.testing.assert_allclose(reference_attention(q, k, v), expected, atol=1e-12)

    # 两个 key 的解析结果，独立检查 1/sqrt(D) 缩放及 QK 的方向。
    q.fill(1)
    k.fill(0)
    k[1, 0] = 8
    v.fill(0)
    v[1] = 1
    expected_row = 1 / (1 + np.exp(-8 / np.sqrt(128)))
    np.testing.assert_allclose(reference_attention(q, k, v)[1], expected_row)

    # 较大 logits 不应溢出；所有 query 都不可见的后缀不应影响结果。
    q.fill(32)
    k.fill(32)
    baseline = reference_attention(q, k, v)
    k[16:] = -32
    v[16:] = 100
    np.testing.assert_array_equal(reference_attention(q, k, v), baseline)
    if not np.isfinite(baseline).all():
        raise AssertionError("参考实现产生 NaN/Inf")
    print("PASS: CPU 参考实现（prefix mean、缩放、causal mask、稳定 softmax）")


def cases(m, n, d):
    rng = np.random.default_rng(42)
    q = rng.normal(size=(m, d)).astype(np.float16)
    k = rng.normal(size=(n, d)).astype(np.float16)
    v = rng.normal(size=(n, d)).astype(np.float16)
    yield "random", q, k, v
    # 同一 module 连续换输入，可暴露状态重置和输出复用问题。
    yield "zero_q_prefix_mean", np.zeros_like(q), k, v
    yield "zero_v", q, k, np.zeros_like(v)
    # 每列不同，能同时检查归一化及输出列布局。
    constant_v = np.broadcast_to(
        np.linspace(-1, 1, d, dtype=np.float16), (n, d)
    ).copy()
    yield "constant_v", q, k, constant_v
    yield "large_logits", q * np.float16(8), k * np.float16(8), v
    # 后续 K tile 不断刷新 row max，检验 online softmax 对旧 RES/sum 的重缩放。
    increasing_k = np.broadcast_to(
        np.linspace(-4, 4, n, dtype=np.float16)[:, None], (n, d)
    ).copy()
    yield "increasing_row_max", np.ones_like(q), increasing_k, v


def build_kernel(tvm, attention, m, n, d):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"}, host="llvm")
    module = tvm.tirx.build(
        attention.flash_attention_v1(m, n, d), target=target, pipeline="tirx"
    )
    source = module.imports[0].inspect_source()
    # 仅能编译并不代表入口执行了计算：未调用的 @T.inline 不会生成 MMA。
    for instruction in ("mma.sync.aligned.m16n8k16", "mma.sync.aligned.m16n8k8"):
        if instruction not in source:
            raise AssertionError(
                f"CUDA 源码缺少 {instruction}；检查主入口是否调用了 QK/PV 计算。"
                "当前实现仅定义 inline helper 时也可能编译成功。"
            )
    print(f"PASS: CUDA 代码生成及 QK/PV MMA，shape=({m}, {n}, {d})")
    return module


def check_cuda(tvm, module, m, n, d):
    device = tvm.cuda(0)
    q_dev = tvm.runtime.empty((m, d), "float16", device=device)
    k_dev = tvm.runtime.empty((n, d), "float16", device=device)
    v_dev = tvm.runtime.empty((n, d), "float16", device=device)
    out_dev = tvm.runtime.empty((m, d), "float16", device=device)
    for name, q, k, v in cases(m, n, d):
        expected = reference_attention(q, k, v)
        q_dev.copyfrom(q)
        k_dev.copyfrom(k)
        v_dev.copyfrom(v)
        # NaN 哨兵让漏写、仅写部分行、空 kernel 都明确失败。
        out_dev.copyfrom(np.full((m, d), np.nan, dtype=np.float16))
        module["kernel"](q_dev, k_dev, v_dev, out_dev)
        device.sync()
        actual = out_dev.numpy().astype(np.float64)
        label = f"{name}, shape=({m}, {n}, {d})"
        if not np.isfinite(actual).all():
            bad = np.argwhere(~np.isfinite(actual))[0]
            raise AssertionError(f"{label}: 输出含 NaN/Inf，首个位置 {tuple(bad)}；检查写回及 softmax")
        # FP32 累加，但 P 和最终 O 会转 FP16；近零值主要由 atol 约束。
        np.testing.assert_allclose(
            actual, expected, rtol=5e-3, atol=5e-3, err_msg=label
        )
        # causal 第一行只有一个有效 key，输出必须为 V[0]。
        np.testing.assert_allclose(actual[0], v[0], rtol=1e-3, atol=1e-3, err_msg=label)
        print(f"PASS: {label}, max_abs_error={np.max(np.abs(actual - expected)):.6g}")


def main(run_cuda=False, reference_only=False, long=False):
    check_reference()
    if reference_only:
        print("仅验证 CPU 参考实现；未验证 attention kernel")
        return

    import tvm

    from . import attention_v1 as attention

    if run_cuda and (not tvm.runtime.enabled("cuda") or not tvm.cuda(0).exist):
        raise RuntimeError("--run-cuda 需要启用 CUDA 的 TVM 和 NVIDIA GPU")

    # 当前无尾块处理：M 为 64 的倍数；N 为 8 的倍数且 N>=M。
    # K/V 每轮由 128 线程各搬 8 个 FP16，因此 D 取 128 的倍数。
    shapes = [(64, 64, 128), (128, 128, 128), (64, 72, 128), (128, 128, 256)]
    if long:
        rows = (attention.BLOCK_CNT + 1) * attention.WARP_PER_CTA * attention.Q_TILE_M
        shapes.append((rows, rows, 128))
    for m, n, d in shapes:
        module = build_kernel(tvm, attention, m, n, d)
        if run_cuda:
            check_cuda(tvm, module, m, n, d)
    print("PASS: GPU 数值验证" if run_cuda else "未运行 GPU 数值验证；在 CUDA 环境添加 --run-cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference-only", action="store_true", help="仅检查 CPU 参考实现")
    mode.add_argument("--run-cuda", action="store_true", help="执行 GPU 数值对比")
    parser.add_argument("--long", action="store_true", help="额外覆盖超过一轮 CTA 调度的序列")
    args = parser.parse_args()
    main(args.run_cuda, args.reference_only, args.long)
