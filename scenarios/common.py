"""SPDX-License-Identifier: Apache-2.0
Shared helpers for building schedulers and requests in the visualization
scenarios (mirrors the test infrastructure in tests/v1/core/utils.py).
"""
from __future__ import annotations

import os
from collections.abc import Callable

# Deterministic block hashes: without a fixed PYTHONHASHSEED, init_none_hash()
# draws os.urandom(32) each call, giving different NONE_HASH per request and
# silently breaking prefix cache hits. Pin it before any hashing runs.
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    DiffusionConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256

# A tiny local model directory so ModelConfig does not try to download
# anything from the Hugging Face hub. Newer vLLM validates the model path and
# requires a recognized config file, so a minimal one is written if missing.
FAKE_MODEL = "/tmp/fake_model"

_FAKE_MODEL_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "float16",
}


def _ensure_fake_model() -> None:
    """Create a minimal local model config so ModelConfig accepts FAKE_MODEL."""
    import json
    from pathlib import Path

    config = Path(FAKE_MODEL) / "config.json"
    if not config.exists():
        Path(FAKE_MODEL).mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(_FAKE_MODEL_CONFIG), encoding="utf-8")

_block_hasher_cache: dict[int, Callable] = {}


def _block_hasher(block_size: int):
    if block_size not in _block_hasher_cache:
        init_none_hash(sha256)
        _block_hasher_cache[block_size] = get_request_block_hasher(block_size, sha256)
    return _block_hasher_cache[block_size]


def make_scheduler(
    *,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 32,
    max_model_len: int = 64,
    block_size: int = 4,
    num_blocks: int = 20,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = True,
    policy: str = "fcfs",
    async_scheduling: bool = False,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    num_speculative_tokens: int | None = None,
    spec_method: str | None = None,
    long_prefill_token_threshold: int = 0,
    use_v2_model_runner: bool = False,
    scheduler_reserve_full_isl: bool = True,
    mamba_cache_mode: str | None = None,
    canvas_length: int | None = None,
) -> Scheduler | AsyncScheduler:
    _ensure_fake_model()
    model_config = ModelConfig(
        model=FAKE_MODEL,
        trust_remote_code=False,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    if canvas_length is not None:
        model_config.hf_config.canvas_length = canvas_length
        # Diffusion forces the V2 model runner (which needs Triton, unavailable
        # in the mock environment). Force it back off so the CPU scheduler can
        # run; the diffusion cadence is still visible via num_sampled_tokens.
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    parallel_config = ParallelConfig(
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
        policy=policy,
        watermark=0.0,
        scheduler_reserve_full_isl=scheduler_reserve_full_isl,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode=mamba_cache_mode or "none",
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        device_config=DeviceConfig(device="cpu"),
        diffusion_config=(
            DiffusionConfig(canvas_length=canvas_length)
            if canvas_length is not None
            else None
        ),
        speculative_config=(
            SpeculativeConfig(
                model=spec_method or "ngram",
                method=spec_method,
                num_speculative_tokens=num_speculative_tokens,
                target_model_config=model_config,
                target_parallel_config=parallel_config,
            )
            if num_speculative_tokens is not None and num_speculative_tokens > 0
            else None
        ),
    )
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    kv_groups = [
        KVCacheGroupSpec(
            ["layer"],
            FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
    ]
    if mamba_cache_mode:
        kv_groups.append(
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    shapes=((block_size, 1),),
                    dtypes=(torch.float32,),
                    block_size=block_size,
                    mamba_cache_mode=mamba_cache_mode,
                ),
            )
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=kv_groups,
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
    scheduler = scheduler_cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    scheduler.use_v2_model_runner = use_v2_model_runner
    return scheduler


def make_request(
    request_id: str,
    prompt_len: int,
    max_tokens: int = 8,
    *,
    prompt_token: int | None = None,
    priority: int = 0,
    arrival_time: float | None = None,
    block_size: int = 4,
    structured_outputs=None,
    resumable: bool = False,
) -> Request:
    init_none_hash(sha256)
    block_hasher = _block_hasher(block_size)
    sampling_params = SamplingParams(
        ignore_eos=False,
        max_tokens=max_tokens,
        structured_outputs=structured_outputs,
    )
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    token = prompt_token if prompt_token is not None else int(request_id)
    return Request(
        request_id=request_id,
        prompt_token_ids=[token] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
        priority=priority,
        arrival_time=arrival_time,
        resumable=resumable,
    )