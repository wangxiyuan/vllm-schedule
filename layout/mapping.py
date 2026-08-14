# SPDX-License-Identifier: Apache-2.0
"""Logical -> physical mapping demo.

Drive a real ``Scheduler`` (CPU mock) to allocate real KV blocks for a couple
of short requests, then replicate the worker-side expansion that turns a
logical token position into a physical slot in the KV tensor:

    pos -> manager block index -> manager block_id (BlockPool)
        -> kernel block id (''b * bpk + k'' in worker block_table.py)
        -> physical slot (''slot = kernel_block_id * kernel_block_size + offset'')

This mirrors ``vllm/v1/worker/gpu/block_table.py`` (append_block_ids +
_compute_slot_mappings_kernel) so the numbers are faithful to how vLLM really
addresses the num-blocks-first KV tensor.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("PYTHONHASHSEED", "0")

from scenarios.common import make_request, make_scheduler  # noqa: E402
from samplers import one_token_per_req  # noqa: E402


def build_mapping_demo() -> dict[str, Any]:
    # Small, zoomed-in config so the block table is human-readable.
    block_size = 4
    kernel_block_size = 2  # bpk = block_size // kernel_block_size = 2
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=block_size,
        num_blocks=8,
        enable_chunked_prefill=False,
        policy="fcfs",
    )
    req_a = make_request("A", prompt_len=6, max_tokens=2, prompt_token=1)
    req_b = make_request("B", prompt_len=4, max_tokens=2, prompt_token=2)
    sched.add_request(req_a)
    sched.add_request(req_b)

    steps = []
    step = 0
    while sched.has_requests() and step <= 8:
        so = sched.schedule()

        # Per-request manager block ids (from the real block pool).
        infos: dict[str, dict[str, Any]] = {}
        for rid in sched.requests:
            req = sched.requests[rid]
            try:
                kb = sched.kv_cache_manager.get_blocks(rid)
                manager_blocks = list(kb.get_block_ids()[0]) if kb else []
            except Exception:
                manager_blocks = []
            req_tokens = [
                f"t{req.num_computed_tokens + i}" for i in range(so.num_scheduled_tokens.get(rid, 0))
            ]
            # Worker expansion: manager block b -> kernel blocks b*bpk + k.
            bpk = block_size // kernel_block_size
            kernel_blocks = [
                b * bpk + k for b in manager_blocks for k in range(bpk)
            ]
            # slot_mapping: pos -> kernel block id * kernel_bs + offset.
            slot_map: list[dict[str, Any]] = []
            for pos in range(req.num_computed_tokens):
                kernel_idx = pos // kernel_block_size
                offset = pos % kernel_block_size
                if kernel_idx < len(kernel_blocks):
                    slot = kernel_blocks[kernel_idx] * kernel_block_size + offset
                else:
                    slot = -1
                manager_idx = pos // block_size
                slot_map.append(
                    {
                        "pos": pos,
                        "manager_block_idx": manager_idx,
                        "manager_block_id": (
                            manager_blocks[manager_idx] if manager_idx < len(manager_blocks) else -1
                        ),
                        "kernel_block_id": (
                            kernel_blocks[kernel_idx] if kernel_idx < len(kernel_blocks) else -1
                        ),
                        "slot": slot,
                    }
                )
            infos[rid] = {
                "manager_blocks": manager_blocks,
                "kernel_blocks": kernel_blocks,
                "computed": req.num_computed_tokens,
                "num_tokens": req.num_tokens,
                "slot_map": slot_map,
            }

        steps.append(
            {
                "step": step,
                "block_size": block_size,
                "kernel_block_size": kernel_block_size,
                "bpk": block_size // kernel_block_size,
                "requests": infos,
                "scheduled": dict(so.num_scheduled_tokens),
            }
        )

        mro = one_token_per_req()(sched, so)
        sched.update_from_output(so, mro)
        step += 1

    return {
        "block_size": block_size,
        "kernel_block_size": kernel_block_size,
        "bpk": block_size // kernel_block_size,
        "steps": steps,
    }