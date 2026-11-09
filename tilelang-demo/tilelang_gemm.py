import tilelang
import tilelang.language as T


def Gemm(M, N, K, block_M=128, block_K=64, block_N=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        m_tiles = T.ceildiv(M, block_M)
        n_tiles = T.ceildiv(N, block_N)
        num_ctas = 66  # 4070 ti super

        with T.Kernel(num_ctas, threads=128) as pid:
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_N, block_K), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for bx, by in T.Persistent(
                [m_tiles, n_tiles],
                num_ctas,
                pid,
                group_size=8,
            ):
                m_idx = block_M * bx
                n_idx = block_N * by

                T.clear(C_local)

                # 不加软件流水线，用最纯粹的串行循环观察编译 Pass 的变化
                for ko in T.serial(T.ceildiv(K, block_K)):
                    k_idx = block_K * ko
                    T.copy(
                        A[m_idx : m_idx + block_M, k_idx : k_idx + block_K], A_shared
                    )
                    T.copy(
                        B[n_idx : n_idx + block_N, k_idx : k_idx + block_K], B_shared
                    )
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[m_idx : m_idx + block_M, n_idx : n_idx + block_N])

    return main


if __name__ == "__main__":
    from pathlib import Path
    from tilelang.tools import lower_trace
    from tilelang.tools.lower_trace.core import get_last_session
    from tilelang.transform import PassContext, PassConfigKey
    from tvm.target import Target

    trace_dir = Path(__file__).resolve().parent / "trace_output"
    trace_dir.mkdir(parents=True, exist_ok=True)

    # 0. 开启 DEBUG 级别日志 + Lower Trace（捕获所有 Pass 的前后 TIR、终端 Diff、HTML 报告及 Codegen）
    tilelang.set_log_level("DEBUG")
    lower_trace.enable(mode="both", trace_dir=str(trace_dir))

    func = Gemm(4096, 4096, 4096)

    print("=== 0. 初始高层 TIR 的原始 AST 语法树结构 (ASTPrinter) ===")
    ast_printer = tilelang.analysis.ASTPrinter()
    # ASTPrinter 会打印 PrimFunc -> SBlock -> For -> CallNode(tl.tileop.*) 的底层节点树
    ast_printer(tilelang.tvm.IRModule({"main": func}))

    print("\n=== 1. 初始高层 TileLang TIR (TVMScript) ===")
    print(func.script())

    # 配置全量编译器内部诊断开关（纯 TileLang/TVM 编译期，不触发 NVCC）：
    # - TL_PASS_PROFILE: 统计每个 Pass 的耗时
    # - TL_LAYOUT_VISUALIZATION_ENABLE + "svg": 打印 Fragment 映射关系并生成彩色 2D 布局图 (.svg)
    # - TL_ENABLE_VECTORIZE_PLANNER_VERBOSE: 打印 VectorizeLoop 如何推导每个 Buffer 的向量化宽度 (float4/half8)
    # - TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: 打印动态共享内存池如何做生命周期分析与地址合并
    # - TL_EMIT_LINE_DIRECTIVES (level=2): 每一行生成的 CUDA C++ 都插入 #line 精确锚定回 Python 源码行
    pass_configs = {
        PassConfigKey.TL_PASS_PROFILE: True,
        PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS: 0,
        PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "svg",
        PassConfigKey.TL_ENABLE_VECTORIZE_PLANNER_VERBOSE: True,
        PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: True,
        PassConfigKey.TL_EMIT_LINE_DIRECTIVES: 2,
    }

    # enable_host_codegen=True 生成 Host C 源码，enable_device_compile=False 不调用 NVCC 编译二进制
    # 布局推导中的向量化规划器需要当前 Target 上下文。
    with Target("cuda") as target, PassContext(config=pass_configs):
        artifact = tilelang.lower(
            func,
            target=target,
            target_host="c",
            enable_host_codegen=True,
            enable_device_compile=False,
        )

    print(
        "\n=== 2. Host 端生成的 CPU 驱动 C 源码 (设置动态 SMEM 上限与 cuLaunchKernel 参数打包) ==="
    )
    # enable_host_codegen=True 时，rt_mod 保存生成的 Host 运行时模块。
    assert artifact.rt_mod is not None
    host_source = artifact.rt_mod.inspect_source()
    print(host_source)
    (trace_dir / "host_launcher.c").write_text(host_source, encoding="utf-8")

    print("\n=== 3. Lower 后的 Device 端底层 TIR ===")
    print(artifact.device_mod.script())

    print("\n=== 4. 最终生成的 Device CUDA C++ 源码 (带 #line 映射) ===")
    print(artifact.kernel_source)
    (trace_dir / "device_kernel.cu").write_text(
        artifact.kernel_source, encoding="utf-8"
    )

    # 5. 汇总打印本次编译经历的所有 Pass 清单与产物路径
    session = get_last_session()
    if session is not None and session.records:
        print(
            "\n=== 5. 全链路编译 Pass 轨迹汇总 (含 Host Codegen 与 Device Codegen) ==="
        )
        print(f"{'序号':<6} {'阶段':<16} {'Pass 名称':<38} {'状态':<10} {'行数变化'}")
        print("-" * 86)
        for rec in session.records:
            status = (
                "CODEGEN"
                if rec.status == "codegen"
                else ("CHANGED" if rec.changed else "NO-OP")
            )
            delta = (
                f"+{rec.add_lines}/-{rec.del_lines}"
                if (rec.changed or rec.status == "codegen")
                else "-"
            )
            print(
                f"{rec.index:<6d} {rec.phase:<16} {rec.name:<38} {status:<10} {delta}"
            )
        print("-" * 86)
        if session.script_dir:
            print(
                f"1) 交互式 Side-by-Side HTML 报告 : {Path(session.script_dir) / 'report.html'}"
            )
        if session.run_dir:
            print(f"2) 每一步 Pass 的原始 .tir 快照目录: {session.run_dir}")
        print(f"3) Host C 发射源码               : {trace_dir / 'host_launcher.c'}")
        print(f"4) Device CUDA C++ 源码          : {trace_dir / 'device_kernel.cu'}")
