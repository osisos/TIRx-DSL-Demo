// test_gather — TTIR 阅读版
// 来源：tilelang-demo/gather-test.py；GPU：RTX 4070 Ti SUPER (sm_89)
// Triton 3.6.0；num_warps=32；grid_size=128；seed=0；数值校验 PASS
// 输入 indices[128,4000]、table[9000,128]；输出 output[128,128]。
// 输出含义：每行收集 4000 个 embedding，沿序列维度求和。
// [gather-test.py:N] 注释对应 Python 源码；原始调试位置保存在 raw/。
// 阅读顺序：参数/常量 → 索引与指针 → scf.for → tt.reduce → tt.store。

module {
  tt.func public @test_gather(
      %indices_ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32},
      %table_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<32xi64>
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<128> : tensor<32x1xi64>
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<9000> : tensor<32xi64>
    %cst_2 = arith.constant dense<4000> : tensor<32xi32>
    %c4000_i32 = arith.constant 4000 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf32>

    // [gather-test.py:30] b = tl.program_id(0)  # 这一行的含义是什么？cta id?
    %b = tt.get_program_id x : i32

    // [gather-test.py:32] d = tl.arange(0, dim)  # [0..32]
    %d = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>

    // [gather-test.py:34] r_base = tl.arange(0, BLOCK_R)
    %r_base = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>

    // [gather-test.py:42] for start in range(0, seq_size, BLOCK_R):  # 似乎这个一定是时间顺序？
    // 循环：分块遍历序列，每次处理 BLOCK_R=32 个索引。
    %acc = scf.for %start = %c0_i32 to %c4000_i32 step %c32_i32 iter_args(%acc_4 = %cst_3) -> (tensor<32x128xf32>)  : i32 {
      // [gather-test.py:44] r = start + r_base
      %r = tt.splat %start : i32 -> tensor<32xi32>
      %r_5 = arith.addi %r, %r_base : tensor<32xi32>

      // [gather-test.py:45] valid = r < seq_size
      %valid = arith.cmpi slt, %r_5, %cst_2 : tensor<32xi32>

      // [gather-test.py:49] indices_ptr + b * seq_size + r,  # 偏移量？ row = b * seq_size,  col = r
      %idx = arith.muli %b, %c4000_i32 : i32
      %idx_6 = tt.addptr %indices_ptr, %idx : !tt.ptr<i64>, i32
      %idx_7 = tt.splat %idx_6 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>>
      %idx_8 = tt.addptr %idx_7, %r_5 : tensor<32x!tt.ptr<i64>>, tensor<32xi32>
      %idx_9 = tt.load %idx_8, %valid, %cst : tensor<32x!tt.ptr<i64>>

      // [gather-test.py:54] idx = tl.where(idx < 0, idx + vocab_size, idx)  # ? 这是常用的写法吗
      %idx_10 = arith.cmpi slt, %idx_9, %cst : tensor<32xi64>
      %idx_11 = arith.addi %idx_9, %cst_1 : tensor<32xi64>
      %idx_12 = arith.select %idx_10, %idx_11, %idx_9 : tensor<32xi1>, tensor<32xi64>

      // [gather-test.py:61] mask=valid[:, None],
      %values = tt.expand_dims %valid {axis = 1 : i32} : tensor<32xi1> -> tensor<32x1xi1>

      // [gather-test.py:57] table_ptr + idx[:, None] * dim + d[None, :],  # 这个是 parallel？
      %values_13 = tt.expand_dims %idx_12 {axis = 1 : i32} : tensor<32xi64> -> tensor<32x1xi64>
      %values_14 = arith.muli %values_13, %cst_0 : tensor<32x1xi64>
      %values_15 = tt.splat %table_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
      %values_16 = tt.addptr %values_15, %values_14 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi64>
      %values_17 = tt.expand_dims %d {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %values_18 = tt.broadcast %values_16 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x128x!tt.ptr<f32>>
      %values_19 = tt.broadcast %values_17 : tensor<1x128xi32> -> tensor<32x128xi32>
      %values_20 = tt.addptr %values_18, %values_19 : tensor<32x128x!tt.ptr<f32>>, tensor<32x128xi32>
      %values_21 = tt.broadcast %values : tensor<32x1xi1> -> tensor<32x128xi1>
      %values_22 = tt.load %values_20, %values_21, %cst_3 : tensor<32x128x!tt.ptr<f32>>

      // [gather-test.py:65] acc += values
      %acc_23 = arith.addf %acc_4, %values_22 : tensor<32x128xf32>
      scf.yield %acc_23 : tensor<32x128xf32>
    }

    // [gather-test.py:67] result = tl.sum(acc, axis=0)
    // 归约：把分块累加器沿序列维度求和，得到各 embedding 通道的结果。
    %result = "tt.reduce"(%acc) <{axis = 0 : i32}> ({
    ^bb0(%result_4: f32, %result_5: f32):
      %result_6 = arith.addf %result_4, %result_5 : f32
      tt.reduce.return %result_6 : f32
    }) : (tensor<32x128xf32>) -> tensor<128xf32>

    // [gather-test.py:70] tl.store(output_ptr + b * dim + d, result)
    %0 = arith.muli %b, %c128_i32 : i32
    %1 = tt.addptr %output_ptr, %0 : !tt.ptr<f32>, i32
    %2 = tt.splat %1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %3 = tt.addptr %2, %d : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %3, %result : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
