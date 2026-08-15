# vLLM Scheduler 调度可视化

用逐步动画演示 vLLM v1 Scheduler 的调度过程：调度决策、CPU Python 对象与
GPU 张量的实时变化（模拟）、KV 缓存从创建到占用的完整生命周期。

**不需要 GPU**：驱动的是真实的 `Scheduler` / `AsyncScheduler` 类，但只用到
其纯 CPU 调度逻辑（`schedule()` / `update_from_output()`），模型输出用 mock
替代。

## 快速开始

```bash
# 跑全部场景，输出单个自包含 HTML
uv run python sched_demo.py

# 指定输出路径
uv run python sched_demo.py -o docs/sched_viz.html

# 只跑某个场景
uv run python sched_demo.py -s spec

# 列出所有场景
uv run python sched_demo.py --list

# 生成 KV Cache 排布与逻辑↔物理映射页（独立页面，与调度页无关）
# 该页依赖可导入 vllm 的 Python（数据来自本地 vllm 源码树的规划代码），
# 用 build.sh 自动探测（同级 vllm 源码树 / VLLM_PYTHON 环境变量）：
./build.sh                       # -> kv_cache_layout.html
./build.sh -o out.html           # 其余参数透传给 kv_layout.py
# 兜底（若环境恰好装好 vllm）：
# uv run python kv_layout.py -o kv_cache_layout.html
```

用浏览器打开生成的 HTML 即可交互：调度页支持上一步 / 下一步 / 自动播放 / 拖拽进度条；
排布页支持左侧大纲 / 底部翻页 / demo 内调度步切换。

## 场景

按学习路径分两组（标签栏按组连续排列）：

| 组 | 场景 | 演示的机制 |
|----|------|-----------|
| **基础循环** | `base` 基础调度 | FCFS 队列、token 预算、KV 块分配、waiting 排队 |
| | `chunked` Chunked Prefill | 长 prompt 拆多步、prefill/decode 混排 |
| | `prefix` 前缀缓存 | 共享前缀命中、`num_computed_tokens` 跳跃 |
| | `preemption` 抢占 | 内存不足时驱逐、回退重算 |
| | `chunked_prefix` Chunked+前缀 | 分块 prefill + 前缀缓存叠加 |
| | `encoder_decoder` Encoder-Decoder | encoder 编码 → decoder 自回归（手绘） |
| | `long_prefill` Long-Prefill 阈值 | `long_prefill_token_threshold` 截断大 prefill |
| | `reserve_full_isl` 整序列预留 | `scheduler_reserve_full_isl` 准入决策（手绘） |
| | `streaming` 流式输入 | `resumable` 会话暂停/续接（`streaming_queue`） |
| | `mamba` Mamba 块对齐缓存 | `mamba_cache_mode=align` 的块边界状态缓存 |
| | `diffusion` 离散扩散 (dLLM) | `canvas_length` 整块去噪、`num_sampled_tokens_per_step=0` |
| | `spec` 投机解码 | draft token 接受/拒绝、回退 |
| | `structured` 结构化输出 | grammar FSM 约束 token |
| | `spec_structured` 投机+结构化 | spec draft 受 grammar 校验 |
| **工程架构** | `async` 异步调度 | `num_output_placeholders` 超前流水线 |
| | `async_pp` 异步+PP | placeholders + `next_decode_eligible_step` 微拍 |
| | `async_spec` 异步+投机 | placeholders + draft |
| | `pp_spec` PP+投机 | PP 微拍节流 + draft 加速 |
| | `parallel` PD 分离 | producer chunked prefill → KV 传输 → consumer decode |
| | `pp` 流水线并行 PP | `next_decode_eligible_step = current_step + pp_size` 微拍 |
| | `dp_balance` DP 预填充均衡 | `defer_prefills` / `prefill_capacity_bound`（手绘） |

## 界面

单 HTML，无外部依赖，按「讲解 → 变化 → 全貌」的阅读顺序设计。每个场景包含：

- **本步发生了什么（story）**：顶部逐步讲解卡片——STEP 徽标 + 一句话动作摘要 +
  渲染好的讲解条目（**加粗**/`代码` 自动排版）+ 事件徽章（prefill / decode+draft /
  被抢占 / 完成），随 step 变化，`aria-live` 播报。
- **时间线圆点（剧情走向）**：每个 step 一个小圆点，颜色表示该步事件类型
  （蓝=prefill 调度、紫=投机 draft、红=抢占、灰=完成、绿=全部完成、斜纹=稳定循环），
  点击任意圆点直接跳转；配套可拖拽进度条。
- **变化高亮（diff）**：本步新调度/被抢占/完成的请求卡片会闪烁；块池中新分配的
  块有光环动画；KV 格子柜里本步新写入的格子放大弹出；队列里新进入/移出的请求
  以实心/删除线标记呈现。两个面板标题栏还带有「本步变化」徽章
  （CPU：调度/抢占/完成/新块数；GPU：本步新写 slot 数），先看徽章、再看高亮。
- **交叉联动**：点请求卡片 → 高亮它的 KV 块（块池）、BlockTable 行、slot_mapping
  行与缓存格子（其余变暗）；点块池方块 → GPU 侧定位并高亮对应缓存块。
- **使用引导 & 术语表**（顶部可折叠，默认展开）：说明页面怎么读、如何交互，
  并对 CPU/GPU 侧、KV 块、prefill/decode、slot_mapping、抢占等核心概念给出通俗解释。
- **初始状态 step 0**：每个调度场景从"调度前"的初始帧开始——请求在 `waiting` 排队、
  尚未分配 KV 块，让读者先看清起点，再逐步进入实际调度。
- **阅读稳定（不跳动）**：默认开启「固定视口」——讲解、面板与图例放进一个高度恒定的
  可视区（内容在区内滚动），页面总高度不随步进变化，从结构上杜绝页面级跳动；同时视口按
  **元素级锚定**：你正在看的 KV 块 / 请求卡片 / 张量行在重渲染后被精确放回原位，
  元素消失时逐级回退到所在区块。取消勾选「固定视口」可回到流式长页面，锚定依然生效。
- **CPU 侧**（python 对象）：`Request` 卡片（computed/total token 进度、
  状态、优先级、prefill chunk）、`waiting`/`running`/`skipped` 队列（带容量
  `Running 2/4`，直观看出为何有人排队）、KV 块池（每块 ref_cnt / 是否缓存 / 占用率）。
- **GPU 侧**（模拟 tensor，标注 `device=cuda`）：`BlockTable`（每请求的
  block_id 行）、`slot_mapping`（CPU block → GPU slot）、`InputBatch`、
  **KV 缓存格子柜**（真实渲染：每个 slot 一个格子，显示 token id 并按请求着色，
  本步新写入的格子放大弹出；默认只显示已用块，可切换显示全部）。
- **事件 + 讲解栏**：每步的调度事件（scheduled / preempted / finished /
  draft 验证）与原理讲解。
- **循环检测**：当 token 级状态（每请求 computed/total/status/块集）重复时，
  自动停止并标注"已进入稳定状态"。
- **播放控制**：播放/暂停、0.5×–4× 变速、首/尾跳转；键盘
  `←`/`→` 步进、`空格` 播放/暂停、`Home`/`End` 首尾；勾选
  「播放时跟随写入」后，画面自动跟随新写入的缓存块滚动。
- **深链分享**：URL 带 `#场景id/步数`，可刷新保持位置、可分享给他人直达某一步。
- **无障碍**：语义化 `<button>`/`<h1–4>`、focus-visible 焦点样式、ARIA 标签、
  键盘操作、`prefers-reduced-motion` 下关闭动画。

> `parallel`（PD 分离）场景是一个真实的多步演示：左边 Prefill 实例（producer）
> 用真实 Scheduler 做 chunked prefill 逐步产出 KV 块，中间 KV 传输步骤把块移交
> 给右边 Decode 实例（consumer），随后在 consumer 上逐步 decode。两个实例的
> computed / out / 块所有权随 step 变化清晰可见。
>
> 多数场景（base/chunked/prefix/preemption/spec/structured/long_prefill/streaming/
> mamba/diffusion/async/spec_structured/async_pp/async_spec/pp_spec/pp）驱动的是
> **真实** `Scheduler`/`AsyncScheduler`，因此其行为与真实 vLLM 一致。少数场景
> （`reserve_full_isl`、`encoder_decoder`、`parallel`、`dp_balance`）因需完整的多模态
> budget / KV connector / DP 集群设置，用**手绘**决策条来呈现该特性。

## KV Cache 排布页（独立页面）

`kv_layout.py` 生成一个**独立**页面 `kv_cache_layout.html`，不涉及调度过程，专注
**KV cache 的内存排布与逻辑↔物理映射**：

- **逻辑寻址**：token → manager block → kernel block → 物理 slot（两个块层级，
  对照真实 worker 侧 `block_table.py` 的展开逻辑）。
- **物理布局**：`KVCacheTensor` 原始分配、**L/B/H/N/C 维度顺序**（num-blocks-first、
  NHD/HND、ROCM kv-first、层维）、多组共享池、DeepSeek-V4 packed 打包、页填充与量化缩放。
- **模型差异**：GQA FullAttention / SlidingWindow / MLA / DSA / Mamba-hybrid / Encoder-Decoder，
  每种都调用**真实** `get_kv_cache_groups` + `get_kv_cache_config_from_groups` 计算真实
  `num_blocks`、kv_cache_groups、张量计划并渲染。
- **逻辑→物理链路**：`pos → block_id → kernel_id → slot → 地址`，用真实 Scheduler 分配 +
  worker 展开逻辑复算 slot_mapping 交互展示。
  - 映射 demo 是**真实调度**的逐步回放（开启 chunked prefill、`max_num_batched_tokens=6`）：
    **第 0 步 = A 的 prefill 独占帧** → `A decode + B prefill` → `B decode`，每步带阶段标签，
    黄色高亮本步新写入的 token（相对上一步），第 0 步即"初始 prefill 一次写入"；
  - 每个请求配有**物理 slot 网格**：每格 = 一个物理 slot（格内 token pos、下方 slot 号），
    与下方 slot 映射表及公式 `slot = kernel_block_id·kernel_bs + offset` 一一对应；
  - 页面本身是**讲义式排版**：左侧章节目录（6 章 21 节）、顶部面包屑、底部翻页器、
    `#章节id/节号` 深链直达（浏览器后退/前进可用）、键盘无障碍与 `prefers-reduced-motion` 支持。

数据来源：`layout/models.py`（真实 vLLM 规划代码）、`layout/mapping.py`（真实调度分配）。

## 代码结构

```
├── sched_demo.py        # 调度页 CLI 入口
├── engine.py            # 通用调度驱动引擎 + token 级循环检测
├── samplers.py          # ModelRunnerOutput mock（基础/spec/diffusion）
├── kv_sim.py            # KV 块池快照提取
├── tensor_view.py       # CPU/GPU 双世界张量提取
├── template.html        # 调度页渲染模板（内嵌 CSV+JS，无外部依赖）
├── kv_layout.py         # KV Cache 排布页 CLI 入口
├── build.sh             # 排布页一键构建（自动探测可导入 vllm 的 Python）
└── layout/
    ├── template.html    # 排布页渲染模板（复用 template.html 的主题 CSS）
    ├── models.py        # 真实 get_kv_cache_groups + get_kv_cache_config_from_groups
    ├── mapping.py       # 真实调度分配（chunked prefill 3 步回放）+ worker 展开 → slot_mapping
    └── sections.py      # 章节/小节内容组装
and scenarios/
    ├── common.py        # 共享 scheduler/request 构造（支持多种特性 flag）
    ├── base.py          # 场景框架 + 公共 extractor + explain 钩子
    ├── core.py          # base/chunked/prefix/preemption/chunked_prefix
    ├── features.py      # encoder_decoder/long_prefill/reserve/streaming/mamba/diffusion
    ├── advanced.py      # spec/structured/async
    ├── combo.py         # spec_structured/async_pp/async_spec/pp_spec
    └── parallel.py      # PD 分离、PP、DP 均衡
```

## 说明

- 场景驱动的是**真实** `Scheduler`/`AsyncScheduler`，因此抢占、前缀缓存、
  chunked prefill、投机解码回退、PP 微拍、语法失败等行为与真实 vLLM 一致。
- PP 微拍 / 异步需 `use_v2_model_runner=True`（mock 环境强制关闭 GPU runner，但
  `AsyncScheduler` 的 `next_decode_eligible_step` cadence 依赖该 flag 才生效）。
- GPU 张量是**模拟**的：不真正分配 CUDA 内存，而是用调度器 CPU 侧的
  `CpuGpuBuffer` numpy 值 + 标注 `device/shape/dtype` 来呈现。
- 环境要求：CPU 版 PyTorch + vLLM 源码（排布页构建由 `./build.sh` 自动探测，见快速开始）。