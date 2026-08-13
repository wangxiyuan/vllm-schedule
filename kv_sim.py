"""SPDX-License-Identifier: Apache-2.0
KV cache lifecycle extraction for the vLLM scheduler visualization.

Turns the real scheduler's ``BlockPool``/``KVCacheManager`` state into a
JSON-serializable snapshot: pool occupancy, per-request block ownership,
free queue, prefix-cache hits, and a per-block lifecycle timeline entry.
"""
from __future__ import annotations

from typing import Any


def sniff_block_pool(scheduler) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the KV block pool."""
    block_pool = scheduler.kv_cache_manager.block_pool
    blocks = getattr(block_pool, "blocks", [])
    num_blocks = len(blocks)

    per_block: list[dict[str, Any]] = []
    for blk in blocks:
        per_block.append(
            {
                "block_id": blk.block_id,
                "is_null": bool(getattr(blk, "is_null", False)),
                "ref_cnt": int(getattr(blk, "ref_cnt", 0)),
                "is_cached": getattr(blk, "block_hash", None) is not None,
                "in_free": getattr(blk, "next_free_block", None) is not None,
            }
        )

    # free queue order (eviction order). FreeKVCacheBlockQueue is a doubly
    # linked list threaded through KVCacheBlock.next_free_block with a fake head.
    free_order: list[int] = []
    fq = getattr(block_pool, "free_block_queue", None)
    if fq is not None:
        blk = getattr(fq, "fake_free_list_head", None)
        seen = 0
        while blk is not None and seen < num_blocks + 1:
            blk = getattr(blk, "next_free_block", None)
            if blk is None or getattr(blk, "block_id", None) == -1:
                break
            free_order.append(getattr(blk, "block_id", -1))
            seen += 1

    usage = block_pool.get_usage() if hasattr(block_pool, "get_usage") else 0.0
    return {
        "num_blocks": num_blocks,
        "usage": usage,
        "per_block": per_block,
        "free_order": free_order,
    }


def request_blocks(scheduler, request_id: str) -> dict[str, Any]:
    """Return the block ids owned by a request (all KV groups)."""
    try:
        kb = scheduler.kv_cache_manager.get_blocks(request_id)
    except Exception:
        return {"groups": []}
    if kb is None:
        return {"groups": []}
    groups = []
    for group in getattr(kb, "blocks", []):
        groups.append([b.block_id for b in group])
    return {"groups": groups}


def extract_kv(scheduler, step: int) -> dict[str, Any]:
    """Extract a full KV-lifecycle snapshot for a step."""
    return {
        "step": step,
        "pool": sniff_block_pool(scheduler),
        "requests": {
            rid: request_blocks(scheduler, rid)
            for rid in sorted(scheduler.requests.keys())
        },
    }