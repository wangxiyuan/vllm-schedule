"""SPDX-License-Identifier: Apache-2.0
CPU/GPU two-world snapshot extraction for the vLLM scheduler visualization.

The scheduler itself only touches CPU python objects (``Request``,
``KVCacheBlock``, ``BlockPool``). The GPU world is built by the worker from
``SchedulerOutput``: ``BlockTable`` (CpuGpuBuffer), ``slot_mapping``,
``InputBatch`` and the KV cache tensors. We never allocate a real GPU; we
simulate the GPU tensors by describing their shape/dtype/device and showing
the real values that live in the CPU-side numpy buffers of the corresponding
``CpuGpuBuffer``s.
"""
from __future__ import annotations

from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput


def _reqs_in_batch(scheduler_output: SchedulerOutput | None) -> list[str]:
    if scheduler_output is None:
        return []
    ids: list[str] = []
    for req in scheduler_output.scheduled_new_reqs:
        ids.append(req.req_id)
    ids.extend(scheduler_output.scheduled_cached_reqs.req_ids)
    return ids


def extract_cpu(scheduler, step: int, scheduler_output: SchedulerOutput | None) -> dict[str, Any]:
    """CPU world: requests, queues, block pool, prefix-cache map."""
    reqs: dict[str, Any] = {}
    for rid, req in scheduler.requests.items():
        reqs[rid] = {
            "id": rid,
            "status": req.status.name,
            "priority": int(getattr(req, "priority", 0)),
            "num_prompt_tokens": int(getattr(req, "num_prompt_tokens", 0)),
            "num_computed_tokens": int(getattr(req, "num_computed_tokens", 0)),
            "num_output_tokens": len(getattr(req, "_output_token_ids", [])),
            "num_tokens": int(getattr(req, "num_tokens", 0)),
            "num_output_placeholders": int(
                getattr(req, "num_output_placeholders", 0)
            ),
            "is_prefill_chunk": bool(getattr(req, "is_prefill_chunk", False)),
            "num_preemptions": int(getattr(req, "num_preemptions", 0)),
            "num_stale_output_tokens": int(
                getattr(req, "num_stale_output_tokens", 0)
            ),
        }

    def _queue_ids(q) -> list[str]:
        return [r.request_id for r in q]

    return {
        "requests": reqs,
        "waiting": _queue_ids(scheduler.waiting),
        "skipped_waiting": _queue_ids(scheduler.skipped_waiting),
        "running": _queue_ids(scheduler.running),
        "num_waiting_for_streaming_input": getattr(
            scheduler, "num_waiting_for_streaming_input", 0
        ),
        "token_budget": getattr(scheduler, "max_num_scheduled_tokens", 0),
        "max_num_running": getattr(scheduler, "max_num_running_reqs", 0),
        "current_step": getattr(scheduler, "current_step", step),
    }


def extract_gpu(
    scheduler, step: int, scheduler_output: SchedulerOutput | None
) -> dict[str, Any]:
    """Simulated GPU world: block table, slot mapping, input batch, KV cache."""
    req_ids = _reqs_in_batch(scheduler_output)
    block_size = scheduler.block_size

    # Per-request block table (what the worker's BlockTable rows would hold).
    block_table: dict[str, list[int]] = {}
    for rid in req_ids:
        try:
            kb = scheduler.kv_cache_manager.get_blocks(rid)
            ids = kb.get_block_ids() if kb else None
            block_table[rid] = list(ids[0]) if ids else []
        except Exception:
            block_table[rid] = []

    # num_scheduled_tokens -> the InputBatch would hold these many tokens.
    num_tokens_map = (
        scheduler_output.num_scheduled_tokens if scheduler_output is not None else {}
    )

    # Simulated slot_mapping: for each token scheduled this step, the KV slot =
    # block_id * block_size + (token_pos % block_size). token_pos runs over the
    # range actually computed this step, i.e. [num_computed - ntok, num_computed).
    slot_mapping: dict[str, list[int]] = {}
    for rid, ntok in num_tokens_map.items():
        req = scheduler.requests.get(rid)
        if req is None:
            slot_mapping[rid] = []
            continue
        end = req.num_computed_tokens
        start = end - ntok
        table = block_table.get(rid, [])
        slots = []
        for t in range(ntok):
            pos = start + t
            block_idx = pos // block_size
            if block_idx < len(table):
                slots.append(table[block_idx] * block_size + (pos % block_size))
            else:
                slots.append(-1)
        slot_mapping[rid] = slots

    # Simulated KV cache cell contents: for every computed token of each
    # scheduled request, the slot it occupies -> the token id stored there.
    # This lets the front-end render the KV cache as a filled grid instead of
    # a bare shape/dtype string.
    kv_content: dict[int, str] = {}
    for rid in req_ids:
        req = scheduler.requests.get(rid)
        if req is None:
            continue
        table = block_table.get(rid, [])
        all_tokens = list(getattr(req, "all_token_ids", []))
        for pos in range(req.num_computed_tokens):
            if pos >= len(all_tokens):
                break
            block_idx = pos // block_size
            if block_idx >= len(table):
                continue
            slot = table[block_idx] * block_size + (pos % block_size)
            # annotate with request id so the front-end can color it
            kv_content[slot] = f"{all_tokens[pos]}|{rid}"

    return {
        "device": "cuda",  # simulated
        "block_size": block_size,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "kv_content": kv_content,
        "input_batch": {
            "req_ids": req_ids,
            "num_scheduled_tokens": num_tokens_map,
            "total_num_scheduled_tokens": (
                scheduler_output.total_num_scheduled_tokens
                if scheduler_output is not None
                else 0
            ),
        },
        "kv_cache_tensor": {
            "num_blocks": scheduler.kv_cache_config.num_blocks,
            "dtype": "int8",
            "layout": "num-blocks-first",
        },
    }


def extract_tensor_worlds(
    scheduler, step: int, scheduler_output: SchedulerOutput
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        extract_cpu(scheduler, step, scheduler_output),
        extract_gpu(scheduler, step, scheduler_output),
    )