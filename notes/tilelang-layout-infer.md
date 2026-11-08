# TileLang Layout Infer 分析

## expr

`VisitExpr_`实现：

处理两类节点

* `CallNode`：块级算子调用，TileOp，例如`Gemm`、`Copy`、`Reduce`，把算子整体加入推导列表`infer_list_`，建立 算子<->Buffer的依赖关系
* `BufferLoadNode`：标量级元素读取，`A[i, j]`，？TODO: 补充下具体的例子

首先看`CallNode`的处理

```cpp
  void VisitExpr_(const CallNode *op) final {
    /*
    	参数里往往包含bufferLoadNode，
    	T.region(A[0.0], ...)
    */
    // 递归遍历当前CallNode的所有子表达式？这个函数的作用是什么
    IRVisitorWithAnalyzer::VisitExpr_(op);
    // Do not analysis the call node to the global function.

    // 全局函数引用，一定不是tileLang内置算子，why?
    if (op->op.as<GlobalVarNode>())
      return;
    TileOperator p;

    // 将 call 解析成 tile operator
    try {
      p = ParseOperator(GetRef<Call>(op));
    } catch (const std::bad_optional_access &e) {
      LOG(FATAL) << "bad_optional_access while parsing tile op call: "
                 << GetRef<Call>(op);
    }

    // 如果是普通的数学调用，全部跳过

    // p.defined() 意味着 CopyOp、GemmOp、ReduceOp等被注册好的算子,参与布局推导的注册好的算子。
    if (p.defined()) {
      for (const auto &arg : op->args) {
        // 提取出被操作的Buffer
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          // 参数是buffer load node
          addToUseList(buffer.value());
        } else if (auto buffer = getBufferFromRegion(arg)) {
          addToUseList(buffer.value());
        }

        /*
        	接下来处理 间接索引依赖。

        	idx = block_mask_frag[i]          # 在 TIR 中生成一条 BindNode: idx = block_mask_frag[i]
					T.copy(A[idx, 0:64], A_shared)    # arg 是 T.region(A[idx, 0], ...)

        	似乎是间接索引？
        	T.copy 依赖了 block_mask_frag ？
        	block_mask_frage 也需要关联到这个 T.copy算子上。
					细节
        */

        // Check if the argument uses any Bind variables that reference
        // fragment buffers. If so, add those buffers to the use list.
        // This handles cases like: a = block_mask_f[i]; T.copy(A[a, 0], ...)
        CollectFragmentBuffersFromExpr(arg);
      }

      /*
      	算子正式登记，推进5个数组，这5个数组拥有完全相同的下标
      */

      // Compute thread_index and thread_bounds
      thread_index_vec_.push_back(CurrentThreadIndex());
      //

      thread_bounds_vec_.push_back(CurrentThreadBounds());
      analyzer_vec_.push_back(analyzer_.Clone());
      // Add the tile operator to infer_list_
      infer_list_stmt_.push_back(GetRef<ObjectRef>(op));
      infer_list_.push_back(std::move(p));
    }
  }
```

