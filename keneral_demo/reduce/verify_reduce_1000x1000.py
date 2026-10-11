"""验证逐行最大值归约。

仓库根目录运行：
    .venv/bin/python -m keneral_demo.reduce.verify_reduce_1000x1000
    .venv/bin/python -m keneral_demo.reduce.verify_reduce_1000x1000 --run-cuda

默认只检查 TIRx lowering 和 CUDA 源码生成；--run-cuda 才运行 GPU 数值测试。
"""

import argparse

import numpy as np
import tvm

from .reduce_1000x1000 import N, TARGET, SM89_1000x1000_REDUCE_LINE_MAX


def cases():
    rng = np.random.default_rng(42)
    yield "随机有限值", rng.uniform(-100, 100, (N, N)).astype(np.float16)
    yield "全负数", -rng.uniform(1, 100, (N, N)).astype(np.float16)
    yield "全部相等", np.full((N, N), -7, dtype=np.float16)

    values = np.full((N, N), -100, dtype=np.float16)
    values[:, [0, 511, 999]] = 17
    yield "跨轮次重复最大值", values

    values = np.full((N, N), -100, dtype=np.float16)
    values[:, -1] = np.arange(N, dtype=np.float16)
    yield "最后一列最大且每行结果不同", values

    # 第 i 行的唯一最大值在第 i 列，一次覆盖所有列、轮次和 lane。
    # 尤其覆盖只读三轮的 lane 29～31，防止尾部处理清掉它们的已有结果。
    values = np.full((N, N), -100, dtype=np.float16)
    values[np.arange(N), np.arange(N)] = np.arange(N, dtype=np.float16)
    yield "逐列唯一最大值", values


def main(run_cuda=False):
    if run_cuda and (not tvm.runtime.enabled("cuda") or not tvm.cuda(0).exist):
        raise RuntimeError("--run-cuda 需要启用 CUDA 的 TVM 和 NVIDIA GPU")

    module = tvm.tirx.build(
        SM89_1000x1000_REDUCE_LINE_MAX(),
        target=tvm.target.Target(TARGET, host="llvm"),
        pipeline="tirx",
    )
    source = module.imports[0].inspect_source()
    assert "__launch_bounds__(256)" in source
    assert "ld.global.v4.b32" in source
    assert source.count("__shfl_xor_sync(") == 5
    print("PASS: TIRx lowering、256 线程/block、向量加载、5 层 warp shuffle 代码生成")

    if not run_cuda:
        print("未运行 GPU 数值验证；在 CUDA 环境添加 --run-cuda")
        return

    device = tvm.cuda(0)
    x = tvm.runtime.empty((N, N), "float16", device=device)
    y = tvm.runtime.empty((N,), "float16", device=device)
    for name, values in cases():
        x.copyfrom(values)
        y.copyfrom(np.full(N, np.nan, dtype=np.float16))
        module["kernel"](x, y)
        device.sync()
        np.testing.assert_array_equal(y.numpy(), values.max(axis=1), err_msg=name)
        print(f"PASS: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cuda", action="store_true")
    main(parser.parse_args().run_cuda)
