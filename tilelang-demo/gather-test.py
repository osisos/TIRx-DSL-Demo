import triton
import torch
import triton.language as tl


@triton.jit
def test_gather(
    indices_ptr,
    table_ptr,
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

        # indices[b, r]: [32]
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
