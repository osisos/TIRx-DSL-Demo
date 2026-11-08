# 本机 GPU 开发环境

保留两个独立的 Conda 开发环境，避免 TVM 命名空间和 TVM-FFI ABI 冲突。

| 环境 | 用途 | Python 解释器 |
| --- | --- | --- |
| `tilelang` | TileLang 与它自带的 TVM / `tvm.tirx` | `/home/kael/miniconda3/envs/tilelang/bin/python` |
| `tirx` | 最新独立 `tvm_tirx` | `/home/kael/miniconda3/envs/tirx/bin/python` |

两个环境均使用 Python 3.10、PyTorch CUDA，以及系统 CUDA Toolkit 12.9。
源码路径通过各环境的 `.pth` 配置加载；直接使用解释器运行时也有效。
环境配置生成器：`~/dev/scripts/configure-gpu-envs.py`。

## TileLang 与内置 TIRx

```bash
conda activate tilelang
python tilelang-demo/test.py
python ~/dev/scripts/smoke-tilelang-tirx.py
```

导入方式：

```python
import tilelang
import tilelang.language as T
import tvm
from tvm.tirx import script as Tx
```

VS Code 打开 `tilelang.code-workspace`。修改 TileLang C++ 源码后运行：

```bash
bash ~/dev/scripts/build-tilelang.sh
```

此环境的 `tvm` 来自 `~/dev/src/tilelang/3rdparty/tvm/python`。
它支持基础 `tvm.tirx` 编译，但与本仓库旧版 TVM 0.26 示例的扩展 DSL 不完全兼容：
例如 `T.device_entry()`、`tvm.tirx.layout`、名为 `tirx` 的 pipeline 不能假定存在。
这些示例需要按目标版本迁移，不能仅修改导入路径。

## 独立最新版 TIRx

```bash
conda activate tirx
python ~/dev/scripts/smoke-tvm-tirx.py
```

导入方式：

```python
import tvm_tirx as tx
from tvm_tirx.script import tirx as T
```

VS Code 打开 `tirx.code-workspace`。修改 C++ 源码后运行：

```bash
bash ~/dev/scripts/build-tirx.sh
```

源码：`~/dev/src/tvm`；独立 TIRx：其 `3rdparty/tvm-tirx` 子模块。
截至 2026-11-08 查询到的上游 HEAD：TVM `1fa4b862`，TIRx `27772b26`。
TIRx 子模块显式使用上游最新提交，领先于上层 TVM 固定的子模块版本。
此环境仅构建独立 TIRx 目标，不导入上层 `tvm`；当前上层 TVM 与最新独立 TIRx 存在 C++ API 差异。
需要 `import tvm` 时使用 `tilelang` 环境中的配套版本。
以后执行 `git submodule update` 会恢复上层 TVM 固定的 TIRx 版本。

## 验证与环境切换

```bash
bash ~/dev/scripts/verify.sh
```

原来的 `source ~/dev/scripts/activate-tilelang.sh` 和
`source ~/dev/scripts/activate-tvm.sh` 入口仍可用，两者均激活 `tilelang`。
独立 TIRx 的入口是 `source ~/dev/scripts/activate-tirx.sh`。
切换环境后启动新的 Python 进程或重启 Notebook kernel，不在同一进程混用两套 TVM。

依赖清单在 `~/dev/envs/*-requirements.lock.txt` 和 `*-conda-explicit.txt`。
迁移前的环境清单、脚本及本地包备份在 `~/dev/backups/python-envs-20261108/`。
Conda `base`、系统 Python 和 `sgl-env` 保留。
