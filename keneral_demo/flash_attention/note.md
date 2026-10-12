# Flash-Attention

尝试实现`flash-attention`

分工：

首先是否根据当前的硬件和`Q`、`K`、`V`的尺寸，计算出分工。

尝试实现，固定一个`Q`tile，然后迭代依赖的`K`、`V` Tile

？这是一个下三角矩阵，



1.  最简单，一个隐藏层的输出给一个block_size 。但是显然存在，Q、K、V 会被不同的输出模块分别依赖，需要加载进入
2. 但是按照输出划分工作量很复杂，$$O_n$$的计算，依赖$$Q_{0..n}$$和$$K_{0..n}$$的计算，似乎不能重计算。运气好的是，行之间没有依赖关系。哦我之前说的不对，不存在这个依赖关系。唯一的问题是，似乎不同的行的工作量差异过大？
3. K V 会重复加载，这是无法避免的。

## 当前实现方案（讨论整理）

目标硬件：RTX 4070 Ti SUPER，Ada / sm89。记隐藏维度为 `H`（代码中的 `D`），每个 CTA 处理 `Br` 行 Q，每次迭代处理 `Bc` 行 K/V。

当前选择：**Br=16、Bc=8**。固定 Q tile，沿序列方向遍历 K/V，Q 和输出累加器 result 尽量始终驻留寄存器，K/V 通过共享内存流水加载。

### 硬件容量与单位

| 资源 | sm89 上限 |
|---|---:|
| 每 SM 共享内存 | 100 KiB = 102400 bytes |
| 每 CTA 可用共享内存 | 99 KiB = 101376 bytes |
| 默认每 CTA 共享内存上限 / 静态分配上限 | 48 KiB = 49152 bytes |
| 每 SM 寄存器文件 | 65536 个 32-bit 寄存器 = 256 KiB |
| 每线程寄存器 | 最多 255 个 32-bit 寄存器 |

CUDA 为每个 CTA 保留 1 KiB 共享内存。超过 48 KiB 的分配需要动态共享内存，并由启动端显式 opt-in。100 KiB 是 SM 的最大配置，不是每个 CTA 都能独占的预算。

寄存器文件的 64K 单位是“个 32-bit 寄存器”，不是 byte；因此每 SM 的寄存器总容量确实大于共享内存。但寄存器分配给各个线程，不能像 CTA shared memory 一样任意共同访问。布局、复制、分配粒度和其他驻留 CTA 都会影响实际可用资源。

来源：[NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/)。

### Q、K、V、result 的容量

先按 Q/K/V 都是 FP16、result 也选择 FP16 累加计算：

| 数据 | 形状 | 数据容量 |
|---|---|---:|
| Q | `16 × H` | `32H` bytes |
| result | `16 × H` | `32H` bytes |
| 每组 K | `8 × H` | `16H` bytes |
| 每组 V | `8 × H` | `16H` bytes |

因此：

- Q + result：**64H bytes / CTA**，计划保留在寄存器。
- 每个 K/V smem 槽位：**32H bytes / CTA**。
- 外层 S 级缓冲：**32HS bytes / CTA**，未计 padding、barrier 等开销。
- `H=128` 时，Q + result 为 8 KiB；每个 K/V 槽位为 4 KiB，2/3/4 级分别为 8/12/16 KiB。

此前讨论过 K/V 每次也加载 16 行的情况：那时 K、V 各为 32H bytes，一组 K/V 合计 64H bytes。现在改成每次 8 行，单组容量相应减半。Q 的行数 Br 与 K/V 的行数 Bc 可以独立选择。

FP32 累加不是硬件强制要求。sm89 的 FP16 MMA 支持 FP16 或 FP32 累加；若 result 改为 FP32，Q + result 的容量变为 **96H bytes / CTA**。FP16 累加需要验证长序列、反复重缩放时的数值误差；softmax 状态和求和精度可以单独选择。

上述数字是数据本身的容量，不等于编译后的寄存器占用。FP16 是否打包、线程布局及重复持有数据都会影响结果，还需要计入 K/V fragment、分数/概率、softmax 状态、地址和临时变量。CTA 的线程数尚未确定，不能只凭总字节数断言每线程寄存器足够。

### Q 和 result 驻留寄存器

Q 加载一次后，按 MMA 所需的 lane / fragment 布局驻留寄存器，在所有 K/V 迭代中复用。result 同样保留在寄存器，最终归一化后写回 global。

sm89 的 `mma.sync` 使用寄存器操作数，因此 Q 不必经过 smem；直接 global → register 的前提是正确安排加载与 fragment 布局。经 smem 再用 `ldmatrix` 是一种方便的布局转换方式，并非硬件上的必经路径。直接写回 result 也需要匹配输出布局和合并访问。

### 两层流水与槽位生命周期

外层：global → CTA smem，使用 `cp.async` 提前加载后续 K/V 组。

内层：smem → register，按计算所需的 fragment 提前加载，在两套寄存器缓冲之间交替使用。这里是通过指令调度进行预取，不应把它理解成另一个 global → shared 的异步拷贝机制。

稳态执行思路：

1. 提前发出后续 K/V 组的 global → smem 加载。
2. 等待当前 smem 组加载完成，完成必要的线程间同步。
3. 提前加载下一份寄存器 fragment，计算当前 fragment。
4. 当前组完成 QKᵀ、online softmax 和 PV，更新 result。
5. 所有消费者完成当前 smem 槽位读取后，允许生产者复用槽位。

流水还需要处理启动预填充、结束排空和尾部边界。单纯用索引取模不能保证安全；异步拷贝完成等待与消费者同步都不可省略。多 warp 消费时尤其要避免过早覆盖 smem。

### Online softmax 与 causal 边界

不能对每组 K/V 独立 softmax 后直接把输出相加。每行 Q 需要保存运行最大值 m、归一化和 l，以及尚未最终归一化的 result。

对本组已经缩放并完成 causal mask 的分数 S，概念上的更新为：

```text
m_new = max(m, rowmax(S))
alpha = exp(m - m_new)
P = exp(S - m_new)
l_new = alpha * l + rowsum(P)
result_new = alpha * result + P @ V
```

所有组处理完后，输出 `result / l`。实际实现需要处理初始化以及整行被 mask 等边界，避免 `-inf - (-inf)` 产生 NaN。MMA 所需的 P 类型转换、线程布局转换也要纳入设计。

标准包含对角线的 causal attention 使用 `k_index <= q_index`。一个 CTA 内的 16 行 Q 有不同边界：遍历到该 Q tile 最后一行允许访问的 K/V 范围，对靠近对角线的分数组逐元素 mask，不能只按 Q tile 第一行的索引截断。不同位置的 CTA 因此具有不同迭代次数。

### Bc=8 与 MMA 形状

`Q[16,H] × Kᵀ[H,8]` 得到 `16×8`，可沿 H 维使用 `mma.m16n8k16` 分块。

`P[16,8] × V[8,H]` 的归约维为 8，可考虑支持的 FP16 `mma.m16n8k8`，沿输出 H 维分块。具体指令选择、布局及 TVM lowering 支持仍需验证，不能仅根据矩阵尺寸假定自动得到理想指令。

来源：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

### Br 并非越小越好

Br 变小会减少 Q/result 的寄存器需求，使调度和资源分配更灵活。但固定 Bc 时，K/V 每槽位的 smem 占用不变；若受 smem 限制，减小 Br 未必增加驻留 CTA。

同时，更小的 Br 减少每次 K/V 加载服务的 Q 行数，增加不同 CTA 的重复加载。缓存命中可能缓解 global 流量，但不会消除所有加载开销。更多驻留 CTA 也不等于更高执行吞吐。当前以 Br=16 为起点，后续如有必要再比较其他 Br。

### 流水深度的起始选择

区分“层数”和“深度”：两层指 global → smem → register；深度指每层的缓冲槽位数量。

**起始配置：外层 smem 3 级，内层 register 2 级。** 这是经验性的调优起点，没有在当前 kernel 上实测，不能保证占满流水线。

| 位置 | 起始深度 | 优先比较范围 |
|---|---:|---|
| global → smem | 3 个 K/V 槽位 | 2、3、4 |
| smem → register | 2 套 fragment | 1、2；有证据再试 3 |

外层 3 级表示消费当前组时，最多提前准备两组。内层 2 级表示计算当前 fragment 时，提前准备下一份。

用简化的单 CTA 模型估计：加载到可用的延迟为 L，处理一组的时间为 C，S 个槽位能提供约 `(S-1)C` 的提前量。隐藏加载延迟大致需要 `(S-1)C >= L`。实际还取决于加载发出时机、槽位释放、缓存、资源竞争和其他驻留 warp，不能用这个公式直接确定最佳深度。

当前 Br=16、Bc=8，不计 softmax、Q 首次加载和最终写回：

- QKᵀ 与 PV 合计约 `4BrBcH = 512H` FLOPs。
- 每组 K/V 加载 `4BcH = 32H` bytes。
- 相对于这些 K/V 加载字节，计算量为 16 FLOPs/byte；这不是考虑缓存后的实际 DRAM 算术强度。
- H=128 时，每组约 65536 FLOPs、4 KiB K/V 数据。

每组计算较小，因此外层双缓冲可能不够；3～4 级提供更长预取距离。内层先用双缓冲，避免增加 fragment 寄存器后挤占其他状态、降低驻留数量。

加深流水只能改善延迟隐藏，不能突破带宽上限，也不能消除 softmax、累加器等依赖链。深度增加还可能降低 occupancy。若外层 4 级仍明显等数据，先检查合并加载、缓存命中、驻留 warp 和同步开销，再决定是否加深。

外层加载粒度也可以大于内层计算粒度：一次向 smem 加载 16 或 32 行 K/V，内部仍每 8 行计算，以减少流水管理开销；代价是更高 smem 占用。

验证时比较实际执行时间、编译后的每线程寄存器数和 spill、smem 占用及等待原因。需要同时检查数值误差，并覆盖不同序列长度与 causal 对角线边界；“能隐藏加载延迟”和“Tensor Core 跑满”是不同目标。

参考：[CUDA 异步数据搬运与延迟隐藏](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)。
