# SPDX-License-Identifier: Apache-2.0
"""Build real vLLM KV-cache configs for representative model structures.

We never instantiate a model or touch a GPU. Instead we hand-craft the
per-layer ``KVCacheSpec`` dict that the model's attention layers would report,
then drive the *real* vLLM planning code (``get_kv_cache_groups`` +
``get_kv_cache_config_from_groups``) to produce the real ``KVCacheConfig``:
the KV cache groups, the per-group block size / page size, the physical
``KVCacheTensor`` plan (num-blocks-first vs packed), and ``num_blocks``.

The result is a JSON-serializable snapshot the front-end renders as diagrams
and tables.
"""
from __future__ import annotations

import os
from typing import Any

import torch

os.environ.setdefault("PYTHONHASHSEED", "0")

from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    AttentionSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.core.single_type_kv_cache_manager import (  # noqa: E402
    register_all_kvcache_specs,
)

from scenarios.common import _ensure_fake_model  # noqa: E402

AVAILABLE_MEM_BYTES = 8 * 1024**3  # 8 GiB of KV cache memory for the examples


def _make_vllm_config(block_size: int = 16) -> VllmConfig:
    _ensure_fake_model()
    model_config = ModelConfig(
        model="/tmp/fake_model",
        trust_remote_code=False,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        device_config=DeviceConfig(device="cpu"),
    )
    # We run on a CPU-only platform, where vLLM auto-disables the hybrid KV
    # cache manager (the platform cannot run it). That would make genuine
    # hybrid models (e.g. Jamba: Mamba + attention) raise in
    # ``unify_hybrid_kv_cache_specs`` for a *demonstration* of the layout. The
    # real planning path we want to show is the one with HMA enabled, so force
    # it back on -- this only affects how the KV cache groups / tensors are
    # planned, never the GPU runtime (we never touch a GPU).
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return vllm_config


def _spec_kind(spec: Any) -> str:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return "uniform_type"
    return type(spec).__name__


def _group_info(group: KVCacheGroupSpec) -> dict[str, Any]:
    spec = group.kv_cache_spec
    page_bytes = int(spec.page_size_bytes)
    per_token = page_bytes // spec.block_size
    info: dict[str, Any] = {
        "kind": _spec_kind(spec),
        "num_layers": len(group.layer_names),
        "layer_names": list(group.layer_names),
        "block_size": spec.block_size,
        "page_size_bytes": page_bytes,
        "per_token_bytes": per_token,
    }
    if isinstance(spec, AttentionSpec):
        info["num_kv_heads"] = spec.num_kv_heads
        info["head_size"] = spec.head_size
        info["kv_quant_mode"] = spec.kv_quant_mode.name
    if isinstance(spec, MambaSpec):
        info["shapes"] = [list(s) for s in spec.shapes]
        info["mamba_cache_mode"] = spec.mamba_cache_mode
    if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
        info["kv_lora_rank"] = spec.head_size - 64  # (qk_rope_head_dim = 64)
        info["qk_rope_head_dim"] = 64
        info["cache_dtype_str"] = spec.cache_dtype_str
        info["compress_ratio"] = spec.compress_ratio
        info["storage_block_size"] = spec.storage_block_size
    if isinstance(spec, (SlidingWindowMLASpec, SlidingWindowSpec)):
        info["sliding_window"] = spec.sliding_window
    return info


def _snapshot(
    name: str,
    label: str,
    note: str,
    vllm_config: VllmConfig,
    specs: dict[str, Any],
    available_memory: int = AVAILABLE_MEM_BYTES,
) -> dict[str, Any]:
    groups = get_kv_cache_groups(vllm_config, specs)
    kvcfg = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=available_memory
    )
    group_infos = [_group_info(g) for g in groups]

    # Per-layer page size. For a UniformTypeKVCacheSpecs each layer has its own
    # page size; otherwise all layers of the group share the spec's page size.
    layer_page: dict[str, int] = {}
    for g in groups:
        spec = g.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            for ln in g.layer_names:
                layer_page[ln] = int(spec.kv_cache_specs[ln].page_size_bytes)
        else:
            for ln in g.layer_names:
                layer_page[ln] = int(spec.page_size_bytes)

    # In the packed layout, several layers (one per group) share the same
    # byte_offset inside the block slab. The segment that starts at that offset
    # must span the *largest* page size among the layers sharing it, so the
    # front-end can draw non-overlapping-backed segments correctly.
    tensors = [
        {
            "size": t.size,
            "shared_by": list(t.shared_by),
            "offset": t.offset,
            "block_stride": t.block_stride,
            "page_size": max(layer_page[n] for n in t.shared_by),
        }
        for t in kvcfg.kv_cache_tensors
    ]
    packed = any(t["block_stride"] > 0 for t in tensors)
    uniform_type = any(g["kind"] == "uniform_type" for g in group_infos)
    return {
        "name": name,
        "label": label,
        "note": note,
        "num_blocks": int(kvcfg.num_blocks),
        "available_memory_bytes": available_memory,
        "layout": "packed" if packed else ("uniform_type" if uniform_type else "num-blocks-first"),
        "groups": group_infos,
        "tensors": tensors,
        "num_tensors": len(tensors),
    }


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_llama7b() -> tuple[VllmConfig, dict[str, Any], str]:
    """Llama-2-7B: GQA, full attention, 32 layers."""
    vc = _make_vllm_config(16)
    specs = {
        f"model.layers.{i}.self_attn": FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
        )
        for i in range(32)
    }
    return vc, specs, (
        "GQA 全注意力：每 token 存 num_kv_heads(=8) 组 K/V，head_size=128，fp16。"
        "所有层规格相同 → 归为 1 个 kv_cache_group。"
    )


def _build_mistral() -> tuple[VllmConfig, dict[str, Any], str]:
    """Mistral-7B: sliding-window attention, 32 layers."""
    vc = _make_vllm_config(16)
    specs = {
        f"model.layers.{i}.self_attn": SlidingWindowSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            sliding_window=4096,
        )
        for i in range(32)
    }
    return vc, specs, (
        "滑窗注意力：窗口外历史 token 不再参与注意力，其块被替换为 null 块并释放，"
        "每请求 KV 上界 O(window)，而非 O(seq_len)。"
    )


def _build_deepseek_v3() -> tuple[VllmConfig, dict[str, Any], str]:
    """DeepSeek-V3/R1: MLA (compressed latent + RoPE), 61 layers."""
    vc = _make_vllm_config(16)
    specs = {
        f"model.layers.{i}.self_attn": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="fp8_ds_mla",
        )
        for i in range(61)
    }
    return vc, specs, (
        "MLA：把 K/V 压缩进一个低秩潜在 kv_lora(=512) 再叠加 RoPE 的 k_rope(=64)，"
        "故 head_size=576，num_kv_heads=1。fp8_ds_mla 自定义排布：每 token 656B。"
    )


def _build_deepseek_v4() -> tuple[VllmConfig, dict[str, Any], str]:
    """DeepSeek-V4: MLA + SWA-MLA, varying page sizes -> packed layout."""
    vc = _make_vllm_config(16)
    specs: dict[str, Any] = {}
    for i in range(32):
        if i % 3 == 0:
            specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                compress_ratio=8,
            )
        else:
            specs[f"model.layers.{i}.self_attn"] = SlidingWindowMLASpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
                sliding_window=4096,
                cache_dtype_str="fp8_ds_mla",
                model_version="deepseek_v4",
                compress_ratio=4,
            )
    return vc, specs, (
        "DeepSeek-V4：一部分层是 MLA、其余是不同窗口的 SWA-MLA，页大小不同 → "
        "按 UniformType 分组，并采用 **packed 打包布局**（单 slab、逐层字节偏移 + block_stride）。"
    )


def _build_jamba() -> tuple[VllmConfig, dict[str, Any], str]:
    """Jamba-like hybrid: Mamba state + Full attention, interleaved."""
    vc = _make_vllm_config(16)
    mamba_spec = MambaSpec(
        shapes=((16, 4096),),
        dtypes=(torch.float32,),
        block_size=16,
        mamba_cache_mode="align",
    )
    specs: dict[str, Any] = {}
    for i in range(24):
        if i % 4 == 0:
            specs[f"model.layers.{i}.mamba"] = mamba_spec
        else:
            specs[f"model.layers.{i}.self_attn"] = FullAttentionSpec(
                block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
            )
    return vc, specs, (
        "Hybrid：Mamba 层（存 conv/ssm 状态块）与 FullAttention 层交错，按层下标模 4 "
        "归为 4 个 kv_cache_group（1 个 Mamba 组 + 3 个注意力组），各自独立 block table，"
        "但共享同一 BlockPool。"
    )


def _build_dsa() -> tuple[VllmConfig, dict[str, Any], str]:
    """DeepSeek sparse attention (DSA): sparse window + global indexer."""
    vc = _make_vllm_config(16)
    specs = {
        f"model.layers.{i}.self_attn": SlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            sliding_window=4096,
            cache_dtype_str="fp8_ds_mla",
        )
        for i in range(32)
    }
    return vc, specs, (
        "DSA（DeepSeek Sparse Attention）：局部滑窗 + 全局 top-k 索引（indexer）。"
        "稀疏只改变『注意力关系』，KV cache 布局与 MLA 完全相同。"
    )


MODEL_BUILDERS: dict[str, tuple[callable, str]] = {
    "llama": (_build_llama7b, "Llama-7B · GQA Full Attention"),
    "mistral": (_build_mistral, "Mistral-7B · Sliding Window"),
    "deepseek_v3": (_build_deepseek_v3, "DeepSeek-V3/R1 · MLA"),
    "deepseek_v4": (_build_deepseek_v4, "DeepSeek-V4 · MLA + SWA (packed)"),
    "jamba": (_build_jamba, "Jamba · Mamba + Full Attention hybrid"),
    "dsa": (_build_dsa, "DeepSeek Sparse Attention · DSA"),
}


def build_model_snapshot(name: str, available_memory: int = AVAILABLE_MEM_BYTES) -> dict[str, Any]:
    builder, label = MODEL_BUILDERS[name]
    vc, specs, note = builder()
    return _snapshot(name, label, note, vc, specs, available_memory)


def build_all_snapshots() -> list[dict[str, Any]]:
    return [build_model_snapshot(name) for name in MODEL_BUILDERS]


# Models and memory budgets used by the "实战对比" chapter's capacity-scaling
# comparison. num_blocks grows linearly with the memory budget, but the slope
# differs by model (inverse of per-token bytes), which is the practical takeaway.
MEMORY_SERIES_MODELS = ("llama", "deepseek_v3", "jamba")
MEMORY_SERIES_GIB = (2, 4, 8, 16)


def build_memory_series(
    names: tuple[str, ...] = MEMORY_SERIES_MODELS,
    memories: tuple[int, ...] = MEMORY_SERIES_GIB,
) -> dict[str, Any]:
    """How many KV blocks each representative model fits at growing memory budgets.

    Reuses the same real planning path as ``_snapshot`` (get_kv_cache_groups +
    get_kv_cache_config_from_groups) so the numbers are faithful.
    """
    models: dict[str, Any] = {}
    for name in names:
        builder, label = MODEL_BUILDERS[name]
        vc, specs, note = builder()
        points = []
        for mem in memories:
            s = _snapshot(name, label, note, vc, specs, available_memory=mem * 1024**3)
            points.append({"mem_gib": mem, "num_blocks": s["num_blocks"]})
        # Representative per-token bytes: the largest group's per-token cost.
        rep_per_token = max(g["per_token_bytes"] for g in s["groups"])
        models[name] = {"label": label, "rep_per_token": rep_per_token, "points": points}
    return {"mems": list(memories), "models": models}