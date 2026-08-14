# SPDX-License-Identifier: Apache-2.0
"""Assemble the educational page content: chapters -> sections -> render blocks.

The layout page is about *how KV cache is arranged in memory* and how a
logical token position maps to a physical slot / tensor address. It is not a
scheduler walkthrough. Real data comes from two sources:

* ``layout.models.build_all_snapshots`` -- real ``KVCacheConfig`` produced by
  vLLM's own planning code for 6 representative model structures.
* ``layout.mapping.build_mapping_demo`` -- a real scheduler allocation the
  page re-expands into kernel blocks + physical slots (worker-side logic).
"""
from __future__ import annotations

from typing import Any

from layout.mapping import build_mapping_demo
from layout.models import build_all_snapshots


def P(md: str) -> dict[str, Any]:
    return {"t": "p", "md": md}


def CODE(file: str, line: int, label: str) -> dict[str, Any]:
    return {"t": "code", "file": file, "line": line, "label": label}


def TABLE(head: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {"t": "table", "head": head, "rows": rows}


def KVMODEL(name: str) -> dict[str, Any]:
    return {"t": "kvcfg", "model": name}


def DIA(kind: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"t": "dia", "kind": kind, "data": data or {}}


def CALL(md: str) -> dict[str, Any]:
    return {"t": "callout", "md": md}


def MAPPING() -> dict[str, Any]:
    return {"t": "mapping"}


def KVCOMPARE(models: list[str]) -> dict[str, Any]:
    return {"t": "kvcfg_compare", "models": models}


def _ch(ch_id: str, title: str, color: str, sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": ch_id, "title": title, "color": color, "sections": sections}


def _sec(title: str, body: list[dict[str, Any]]) -> dict[str, Any]:
    return {"title": title, "body": body}


def build_page() -> dict[str, Any]:
    mapping = build_mapping_demo()
    models = {s["name"]: s for s in build_all_snapshots()}

    chapters = [
        # ------------------------------------------------------------------
        _ch("logical", "逻辑寻址", "#4f8ff7", [
            _sec("为什么 KV cache 要分块", [
                P("decode 每生成一个 token，注意力都要读它**之前所有 token** 的 K/V。"
                  "若不缓存，每步得把整段历史重新算一遍，代价随序列长度 **O(n²) 增长**。"),
                P("KV cache 就是把这些 K/V 存进显存，用**固定的块（block/page）**管理。"
                  "每个块固定装 **block_size 个 token** 的 KV，块是分配、释放、缓存的最小单元。"),
                P("分块（PagedAttention）相比 Transformers 的整段连续缓存："),
                TABLE(["连续缓存", "vLLM 分块缓存"], [
                    ["一个序列预留连续整段，长度不定导致碎片", "块可散落，用 block table 记录每请求的块"],
                    ["按最大序列长度预留 → 显存浪费", "按实际 token 用量动态分配/回收"],
                    ["并发序列互相挤占", "块池共享，空闲块按需分配"],
                ]),
                CODE("vllm/v1/worker/gpu/block_table.py", 107, "BlockTable：每请求的 block_id 表"),
            ]),
            _sec("两个块层级：manager block 与 kernel block", [
                P("vLLM 里有两层块，千万别混："),
                TABLE(["层级", "大小", "谁在用", "作用"], [
                    ["manager block", "`block_size`（如 4/16/64）", "scheduler / KVCacheManager", "逻辑分配、前缀缓存、引用计数"],
                    ["kernel block", "`kernel_block_size`（如 16/64）", "attention 内核", "内核按固定粒度访问显存"],
                ]),
                P("worker 把 manager block **展开**成若干 kernel block：设 "
                  "`bpk = block_size // kernel_block_size`，manager 块 `b` 变成 "
                  "`b*bpk + 0 … b*bpk + bpk-1` 共 `bpk` 个 kernel 块。"),
                CODE("vllm/v1/worker/gpu/block_table.py", 118, "append_block_ids：b -> b*bpk + k"),
            ]),
            _sec("真实映射：pos → block → slot", [
                P("下面用真实 Scheduler 分配 2 个短请求，再按 worker 逻辑把每个 token 位置"
                  "展开成 manager 块 → kernel 块 → 物理 slot。可切换调度步查看。"),
                MAPPING(),
            ]),
        ]),
        # ------------------------------------------------------------------
        _ch("physical", "物理布局", "#b07cf7", [
            _sec("原始分配与 KVCacheTensor", [
                P("物理上，KV cache 是一段段 **int8 原始字节**。`KVCacheConfig.kv_cache_tensors` "
                  "描述如何初始化：`size`（字节）、`shared_by`（共享该张量的层名）、"
                  "`offset`（打包时层在该块 slab 内的字节偏移）、`block_stride`（打包时块步长）。"),
                CODE("vllm/v1/kv_cache_interface.py", 1007, "KVCacheTensor"),
                P("worker 先 `torch.zeros` 分配原始张量，再按各层 spec `reshape` 成注意力内核要的形状。" ),
                CODE("vllm/v1/worker/gpu/attn_utils.py", 199, "_allocate_kv_cache"),
            ]),
            _sec("num-blocks-first 与 L·B·H·N·C 维度顺序", [
                P("最常见的物理排布是 **num-blocks-first**：块是张量的第 0 维。注意力内核的 "
                  "`get_kv_cache_shape` 给出**逻辑形状**，`get_kv_cache_stride_order` 给出到物理内存的排列。"),
                DIA("stride_order"),
                P("对 FlashAttention，逻辑形状是 `(B, H, N, 2D)` —— K 与 V 沿最后一维拼接 "
                  "（`2*head_size`），`N = block_size`。真正的物理顺序由 `get_kv_cache_stride_order` 决定："),
                DIA("stride_variants"),
                CODE("vllm/v1/attention/backends/flash_attn.py", 196, "get_kv_cache_stride_order"),
            ]),
            _sec("多组共享池（general case）", [
                P("当模型有**多个 kv_cache_group**（如 Mamba + 注意力），且各层页大小可统一时，"
                  "物理上按 `group_size` 个池分配，**每个池被每个组的一层按同一层下标共享**。"),
                DIA("shared_pool"),
                P("例如 3 组 `(full.0, full.1)`、`(sw.0, sw.2)`、`(sw.1, padding)`："
                  "`full.0 / sw.0 / sw.1` 共享一个 Tensor，`full.1 / sw.2` 共享另一个。"),
                CODE("vllm/v1/core/kv_cache_utils.py", 1327, "get_kv_cache_config_from_groups"),
            ]),
            _sec("packed 打包布局（DeepSeek-V4）", [
                P("当各层页大小不同又无法统一（如 DeepSeek-V4 的 MLA 与 SWA-MLA），vLLM 改用 "
                  "**packed 打包布局**：所有组叠进**同一块 slab**，每层在块内拥有一个字节 `offset`，"
                  "块内总步长记在 `block_stride`。"),
                DIA("packed"),
                CODE("vllm/v1/core/kv_cache_utils.py", 1249, "_get_packed_kv_cache_layout"),
            ]),
            _sec("页填充与量化缩放", [
                P("某些布局要求页对齐：`page_size_padded` 会在每页后补字节，worker 用 "
                  "`torch.as_strided` 跳过填充（num-blocks-first 才支持）。"),
                CODE("vllm/v1/worker/gpu/attn_utils.py", 294, "page_size_padded -> as_strided"),
                P("量化 KV（如 fp8 per-token-head、int4）时，每 token 的缩放因子也**从原始 KV 分配里切出**"
                  "独立张量，需在 `page_size_bytes` 里计入。"),
                CODE("vllm/v1/kv_cache_interface.py", 217, "unpadded_page_size_bytes 计入 per-token scales"),
            ]),
        ]),
        # ------------------------------------------------------------------
        _ch("models", "模型差异", "#35d07f", [
            _sec("GQA Full Attention（Llama-7B）", [
                P("GQA：`num_kv_heads < num_query_heads`，多个 query head 共享一组 K/V。"
                  "每 token 存 K/V 各 `num_kv_heads` 个 head，`head_size=128`，fp16。"),
                KVMODEL("llama"),
                P("**关键数字**：每 token 单层 = 8 heads × 128 × 2 字节 × (K+V) = **4096 B**；"
                  "16 token/块 → 每块 64 KiB。全模型 ×32 层。" ),
            ]),
            _sec("Sliding Window（Mistral-7B）", [
                P("滑窗注意力只保留最近 `sliding_window` 个 token。窗口外历史块的 KV 不再需要 → "
                  "替换为 null 块并释放，每请求 KV 上界 O(window) 而非 O(seq_len)。"),
                KVMODEL("mistral"),
                DIA("swa"),
            ]),
            _sec("MLA（DeepSeek-V3/R1）", [
                P("MLA 把每层的 K/V 压缩进低秩潜在 `kv_lora`（=512），再叠加 RoPE 的 `k_rope`（=64），"
                  "故 `head_size = 576`，`num_kv_heads = 1`。"),
                KVMODEL("deepseek_v3"),
                P("fp8_ds_mla 自定义排布：每 token / 每层只占 **656 B**（对比 GQA 的 4096 B × 多 head），"
                  "大幅省显存。压缩的是**缓存内容**，不是块结构。"),
                DIA("mla"),
            ]),
            _sec("DeepSeek Sparse Attention（DSA）", [
                P("DSA 在滑窗基础上，用 **indexer** 为每个 query 选出 top-k 个全局 token 参与注意力。"
                  "**稀疏只改变『注意哪些 token』，KV cache 布局与 MLA 完全相同** —— 仍是"
                  "压缩潜在 + k_rope，num_kv_heads=1。"),
                KVMODEL("dsa"),
                P("所以在排布层面，DSA ≈ MLA；区别体现在注意力后端（`DeepseekSparseSWABackend`）"
                  "与稀疏索引，而非缓存张量。"),
                CODE("vllm/v1/attention/backends/mla/sparse_swa.py", 112, "DeepseekSparseSWABackend"),
            ]),
            _sec("Hybrid：Mamba + Full Attention（Jamba）", [
                P("Mamba 层不存 token 化的 K/V，而是每块存一段**状态**（conv + ssm），"
                  "物理上以 `[num_blocks, 1, 1, page_size_bytes]` 的页视图存放。"),
                KVMODEL("jamba"),
                P("Mamba 与注意力形成**多个 kv_cache_group**，各自独立 block table，但共享同一 BlockPool。"
                  "块大小可能在页统一时被拉大（下面 Mamba 组 bs=16、注意力组 bs=64）。"),
                DIA("mamba"),
            ]),
            _sec("六模型排布对比", [
                P("全部由真实 `get_kv_cache_groups` + `get_kv_cache_config_from_groups` 计算"
                  "（8 GiB KV 显存，block_size=16）。"),
                KVCOMPARE(["llama", "mistral", "deepseek_v3", "dsa", "jamba", "deepseek_v4"]),
            ]),
        ]),
        # ------------------------------------------------------------------
        _ch("hybrid", "Hybrid 多组", "#ffc53d", [
            _sec("kv_cache_group 与块大小统一", [
                P("具同一 KV cache 规格的层归为一组（`KVCacheGroupSpec`）。每组的 `block_size` 可能不同。"
                  "上层用两个额外粒度协调："),
                TABLE(["粒度", "取值", "作用"], [
                    ["scheduler_block_size", "各 group block_size 的 **LCM**", "调度器 token 对齐"],
                    ["hash_block_size", "各 group block_size 的 **GCD**", "前缀缓存最低匹配粒度"],
                ]),
                CODE("vllm/v1/core/kv_cache_utils.py", 607, "resolve_kv_cache_block_sizes"),
            ]),
            _sec("一组 BlockPool，多组 BlockTable", [
                P("所有 group 共享同一个 `BlockPool`（同一批 KVCacheBlock、同一空闲队列），"
                  "但**每个 group 各自维护 `req_to_blocks`（block table）**。"
                  "调度器按 LCM 粒度对齐后，把每个 group 的块分别写入各自的表。"),
                DIA("hybrid_groups"),
                CODE("vllm/v1/core/kv_cache_coordinator.py", 63, "KVCacheCoordinator 协调多组"),
            ]),
            _sec("DeepSeek-V4 的 UniformType 打包", [
                P("DeepSeek-V4 的 MLA 与不同窗口的 SWA-MLA 页大小不同 → 按 "
                  "`(block_size, sliding_window)` 分组为多个 `UniformTypeKVCacheSpecs`，"
                  "再走 packed 布局：组各自在 slab 里排布，层按字节偏移。"),
                KVMODEL("deepseek_v4"),
                DIA("packed"),
                CODE("vllm/v1/core/kv_cache_utils.py", 1545, "group_and_unify_kv_cache_specs"),
            ]),
        ]),
        # ------------------------------------------------------------------
        _ch("mapping", "逻辑→物理链路", "#ff6b6b", [
            _sec("完整链路", [
                P("把前面几章串起来，一个逻辑 token 位置映射到物理缓存地址的完整链路："),
                DIA("slot_chain"),
                P("公式：`pos // block_size` = manager 块索引 → 查 BlockPool 得 manager `block_id`；"
                  "`block_id * bpk + (pos % block_size) // kernel_block_size` = kernel 块；"
                  "`slot = kernel_block_id * kernel_block_size + (pos % kernel_block_size)`。"
                  "内核再以 `base + layer_offset + slot * stride` 定位。"),
                CODE("vllm/v1/worker/gpu/block_table.py", 262, "_compute_slot_mappings_kernel"),
            ]),
            _sec("跨层 KV sharing 与量化缩放", [
                P("层可以 **别名** 到目标层的 KV 张量（`kv_sharing_target_layer_name`），"
                  "多个 Attention 模块共享同一份缓存，不重复分配。"),
                CODE("vllm/v1/worker/gpu/attn_utils.py", 460, "共享层 -> 目标层张量"),
                P("量化每 token/每 head 的缩放张量从原始 KV 分配中切出，由注意力后端托管，"
                  "但字节在 `page_size_bytes` 中已预算。"),
            ]),
        ]),
        # ------------------------------------------------------------------
        _ch("compare", "实战对比", "#22d3ee", [
            _sec("从模型 config 到 KV cache 张量", [
                P("一个模型结构 → 一组 kv_cache_group + 一个张量计划。下面用真实 vLLM 规划代码"
                  "（`get_kv_cache_groups` + `get_kv_cache_config_from_groups`）对 6 种结构"
                  "在同一 8 GiB 显存、block_size=16 下算出结果。"),
                KVCOMPARE(["llama", "mistral", "deepseek_v3", "dsa", "jamba", "deepseek_v4"]),
            ]),
            _sec("读图要点", [
                P("**每 token 字节是核心差异**：GQA 贵在 KV head 数多；MLA 靠压缩潜在把单 token "
                  "压到数百字节；Mamba 存的是固定状态而非 token。"),
                P("**布局决定并发容量**：`num_blocks` 越大，能同时跑的序列/上下文越多。"
                  "MLA/DSA 因每 token 字节小，`num_blocks` 是 GQA 的数倍。"),
                P("**packed vs num-blocks-first**：层页大小统一 → num-blocks-first；不统一 → packed。"
                  "这是 DeepSeek-V4 与大多数模型在物理排布上的分水岭。"),
            ]),
        ]),
    ]

    return {"chapters": chapters, "data": {"models": models, "mapping": mapping}}