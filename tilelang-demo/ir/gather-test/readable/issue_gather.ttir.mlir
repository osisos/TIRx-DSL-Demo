// issue_gather — TTIR 阅读版
// 来源：tilelang-demo/gather-test.py；GPU：RTX 4070 Ti SUPER (sm_89)
// Triton 3.6.0；num_warps=32；grid_size=128；seed=0；数值校验 PASS
// 输入 indices[128,4000]、table[9000,128]；输出 output[128,128]。
// 输出含义：每行收集 4000 个 embedding，沿序列维度求和。
// [gather-test.py:N] 注释对应 Python 源码；原始调试位置保存在 raw/。
// 阅读顺序：参数/常量 → 索引与指针 → scf.for → tt.reduce → tt.store。

module {
  tt.func public @issue_gather(
      %indices_ptr: !tt.ptr<i64> {tt.divisibility = 16 : i32},
      %table_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %output_numel: i32 {tt.divisibility = 16 : i32},
      %seq_size: i32 {tt.divisibility = 16 : i32},
      %grid_size: i32 {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %cst = arith.constant dense<9000> : tensor<128x32xi64>
    %cst_0 = arith.constant dense<0> : tensor<128x32xi64>
    %c32_i32 = arith.constant 32 : i32
    %c4000_i32 = arith.constant 4000 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %cst_1 = arith.constant dense<128> : tensor<128x32xi64>
    %cst_2 = arith.constant dense<4000> : tensor<128x1xi32>
    %cst_3 = arith.constant dense<4000> : tensor<1x32xi32>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
    %cst_5 = arith.constant dense<128> : tensor<128x1xi32>
    %c128_i32 = arith.constant 128 : i32

    // [gather-test.py:87] pid = tl.program_id(0)
    %pid = tt.get_program_id x : i32

    // [gather-test.py:88] output_start = pid * BLOCK_D
    %output_start = arith.muli %pid, %c128_i32 : i32

    // [gather-test.py:89] output_stride = grid_size * BLOCK_D
    %output_stride = arith.muli %grid_size, %c128_i32 : i32

    // [gather-test.py:91] for output_offset in range(output_start, output_numel, output_stride):
    scf.for %output_offset = %output_start to %c16384_i32 step %output_stride  : i32 {
      // [gather-test.py:92] output_index = output_offset + tl.arange(0, BLOCK_D)[:, None]
      %output_index = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %output_index_6 = tt.expand_dims %output_index {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %output_index_7 = tt.splat %output_offset : i32 -> tensor<128x1xi32>
      %output_index_8 = arith.addi %output_index_7, %output_index_6 : tensor<128x1xi32>

      // [gather-test.py:93] r_base = tl.arange(0, BLOCK_R)[None, :]
      %r_base = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
      %r_base_9 = tt.expand_dims %r_base {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>

      // [gather-test.py:94] b = output_index // 128
      %b = arith.divsi %output_index_8, %cst_5 : tensor<128x1xi32>

      // [gather-test.py:95] d = output_index % 128
      %d = arith.remsi %output_index_8, %cst_5 : tensor<128x1xi32>

      // [gather-test.py:97] for start in range(0, seq_size, BLOCK_R):
      // 循环：分块遍历序列，每次处理 BLOCK_R=32 个索引。
      %acc = scf.for %start = %c0_i32 to %c4000_i32 step %c32_i32 iter_args(%acc_11 = %cst_4) -> (tensor<128x32xf32>)  : i32 {
        // [gather-test.py:98] r = start + r_base
        %r = tt.splat %start : i32 -> tensor<1x32xi32>
        %r_12 = arith.addi %r, %r_base_9 : tensor<1x32xi32>

        // [gather-test.py:99] valid = r < seq_size
        %valid = arith.cmpi slt, %r_12, %cst_3 : tensor<1x32xi32>

        // [gather-test.py:100] idx = tl.load(indices_ptr + r + 4000 * b, valid, other=0.0)
        %idx = tt.splat %indices_ptr : !tt.ptr<i64> -> tensor<1x32x!tt.ptr<i64>>
        %idx_13 = tt.addptr %idx, %r_12 : tensor<1x32x!tt.ptr<i64>>, tensor<1x32xi32>
        %idx_14 = arith.muli %b, %cst_2 : tensor<128x1xi32>
        %idx_15 = tt.broadcast %idx_13 : tensor<1x32x!tt.ptr<i64>> -> tensor<128x32x!tt.ptr<i64>>
        %idx_16 = tt.broadcast %idx_14 : tensor<128x1xi32> -> tensor<128x32xi32>
        %idx_17 = tt.addptr %idx_15, %idx_16 : tensor<128x32x!tt.ptr<i64>>, tensor<128x32xi32>
        %idx_18 = tt.broadcast %valid : tensor<1x32xi1> -> tensor<128x32xi1>
        %idx_19 = tt.load %idx_17, %idx_18, %cst_0 : tensor<128x32x!tt.ptr<i64>>

        // [gather-test.py:101] adjusted = idx + tl.full([BLOCK_D, BLOCK_R], 9000, tl.int32)
        %adjusted = arith.addi %idx_19, %cst : tensor<128x32xi64>

        // [gather-test.py:102] idx = tl.where(idx < 0, adjusted, idx)
        %idx_20 = arith.cmpi slt, %idx_19, %cst_0 : tensor<128x32xi64>
        %idx_21 = arith.select %idx_20, %adjusted, %idx_19 : tensor<128x32xi1>, tensor<128x32xi64>

        // [gather-test.py:103] values = tl.load(table_ptr + d + 128 * idx, valid)
        %values = tt.splat %table_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
        %values_22 = tt.addptr %values, %d : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
        %values_23 = arith.muli %idx_21, %cst_1 : tensor<128x32xi64>
        %values_24 = tt.broadcast %values_22 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x32x!tt.ptr<f32>>
        %values_25 = tt.addptr %values_24, %values_23 : tensor<128x32x!tt.ptr<f32>>, tensor<128x32xi64>
        %values_26 = tt.load %values_25, %idx_18 : tensor<128x32x!tt.ptr<f32>>

        // [gather-test.py:105] acc = tl.where(valid, acc + values, acc)
        %acc_27 = arith.addf %acc_11, %values_26 : tensor<128x32xf32>
        %acc_28 = arith.select %idx_18, %acc_27, %acc_11 : tensor<128x32xi1>, tensor<128x32xf32>
        scf.yield %acc_28 : tensor<128x32xf32>
      }

      // [gather-test.py:106] result = tl.sum(acc, 1)[:, None]
      // 归约：把分块累加器沿序列维度求和，得到各 embedding 通道的结果。
      %result = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%result_11: f32, %result_12: f32):
        %result_13 = arith.addf %result_11, %result_12 : f32
        tt.reduce.return %result_13 : f32
      }) : (tensor<128x32xf32>) -> tensor<128xf32>
      %result_10 = tt.expand_dims %result {axis = 1 : i32} : tensor<128xf32> -> tensor<128x1xf32>

      // [gather-test.py:107] tl.store(output_ptr + output_index, result)
      %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
      %1 = tt.addptr %0, %output_index_8 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
      tt.store %1, %result_10 : tensor<128x1x!tt.ptr<f32>>
    }
    tt.return
  }
}
