"""练习目标：用 TIRx 实现 1000 × 1000 矩阵的逐行最大值归约。

输入：X，形状 (1000, 1000)，行优先连续存储，dtype 为 float16。
输出：Y，形状 (1000,)，dtype 为 float16，只返回最大值，不返回索引。
计算：Y[i] = max(X[i, j] for j in range(1000))，即沿 axis=1 归约。

验收要求：
- 与 NumPy 参考结果 X.max(axis=1) 一致。
- 覆盖全负数、重复最大值，以及最大值位于每行最后一列的情况。
- 正确处理 1000 列的边界，不能越界读取，也不能遗漏末尾元素。
- 本练习先限定输入为有限值，暂不考虑 NaN / Inf 的语义。

沿用现有练习的 FP16 数据类型和 CUDA SM89 目标；kernel 实现自行完成。
依赖：NumPy，以及包含 tvm.tirx 的 TVM 构建。
GPU 运行还需要启用 CUDA 的 TVM 和对应的 NVIDIA GPU。
"""

import numpy as np
import tvm
from tvm.tirx import script as T

N = 1000
DTYPE = "float16"
TARGET = {"kind": "cuda", "arch": "sm_89"}

BLOCK_CNT = 125
WARP_PER_CTA = 8

"""
    首先，每个warp负责一行
    1个block负责1行，1block8warp

"""


# 运气很好，你要求1000x1000的算子，1000 x 2 % 16 = 0，需要输入16对齐
def SM89_1000x1000_REDUCE_LINE_MAX():

    @T.prim_func
    def kernel(
        X: T.Buffer(
            (
                N,
                N,
            ),
            "float16",
        ),
        OUT: T.Buffer((N,), "float16"),
    ):

        T.device_entry()

        bx = T.cta_id([BLOCK_CNT])  # 125
        warp_id = T.warp_id([WARP_PER_CTA])  # 8
        lane_id = T.lane_id([32])

        num_buffer = T.alloc_local((4,), "float32")
        num = num_buffer.view("float16")

        lane_res = T.alloc_local((1,), "float16")
        lane_res[0] = T.float16(float("-inf"))

        @T.inline
        def load_gmem_to_reg(ptr):
            # float4 向量化加载
            T.ptx.ld(
                ptr,
                "uint32",
                "b32",
                dst=num.ptr_to([0]),
                space="global",
                vec="v4",
            )

        rounds = N // 256  # 8 * 32 = warp 每一轮可以处理的元素数

        tail_start = N - (N % 256)
        tail_lane = N % 256
        tail_thread = tail_lane // 8

        # 计算当前warp负责的行
        row = bx * 8 + warp_id

        # 主干过程
        for r in range(rounds):
            col = r * 32 * 8 + lane_id * 8
            load_gmem_to_reg(X.ptr_to([row, col]))
            # 真正执行规约

            # 绑定8个元素，手动展开循环
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                num[i] = T.max(num[i], num[i + 4])

            for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                num[i] = T.max(num[i], num[i + 2])

            num[0] = T.max(num[0], num[1])
            lane_res[0] = T.max(num[0], lane_res[0])

        # tail 处理
        if lane_id < tail_thread:
            col = tail_start + lane_id * 8
            load_gmem_to_reg(X.ptr_to([row, col]))
            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                num[i] = T.max(num[i], num[i + 4])

            for i in T.unroll(2):  # pyright: ignore[reportGeneralTypeIssues]
                num[i] = T.max(num[i], num[i + 2])

            num[0] = T.max(num[0], num[1])
            lane_res[0] = T.max(lane_res[0], num[0])

        # warp shuffle
        for stage in T.unroll(5):  # pyright: ignore[reportGeneralTypeIssues]
            lane_res[0] = T.max(
                lane_res[0],
                T.tvm_warp_shuffle_xor(
                    T.uint32(0xFFFFFFFF), lane_res[0], 16 >> stage, 32, 32
                ),
            )

        # write back
        if lane_id == 0:
            OUT[row] = lane_res[0]

    return kernel
