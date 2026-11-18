"""复现 https://github.com/tile-ai/tilelang/issues/4056 的布局推导错误。

运行（仓库根目录，tilelang 环境）：
    python tilelang-demo/bug-4056.py

GEMM 累加器 + 独立输入的逐行和，受影响版本报错：
    InternalError: Check failed: (has_best) is false: no available layout found
只做 lowering，不调用 NVCC，不分配 GPU 张量，也不执行 kernel。
"""

from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang.tools import lower_trace
from tvm.target import Target


def make_kernel():
    @T.prim_func
    def main(
        A: T.Tensor((64, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        H: T.Tensor((64, 64), "float16"),
        Out: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            As = T.alloc_shared((64, 64), "float16")
            Bs = T.alloc_shared((64, 64), "float16")
            acc = T.alloc_fragment((64, 64), "float32")
            wide = T.alloc_fragment((64, 64), "float32")
            stat = T.alloc_fragment((64,), "float32")
            out = T.alloc_fragment((64, 64), "float32")

            T.copy(A, As)
            T.copy(B, Bs)
            T.clear(acc)
            T.gemm(As, Bs, acc)

            for i, j in T.Parallel(64, 64):
                wide[i, j] = T.cast(H[i, j], "float32")
            T.reduce_sum(wide, stat, dim=1, clear=True)

            # 关键：把独立归约得到的行统计量广播到 GEMM 的固定布局中。
            for i, j in T.Parallel(64, 64):
                out[i, j] = stat[i] + acc[i, j]
            T.copy(out, Out)

    return main


if __name__ == "__main__":
    lower_trace.enable(
        mode="both",
        trace_dir=str(Path(__file__).resolve().parents[1] / "tmp" / "lower_trace"),
    )
    with Target({"kind": "cuda", "arch": "sm_89"}) as target:
        tilelang.lower(
            make_kernel(),
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
