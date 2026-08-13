"""SPDX-License-Identifier: Apache-2.0
Scenario definitions: each builds a real scheduler + requests and produces
mock model outputs. Frames are produced by the shared extractor.
"""
from __future__ import annotations

from ..samplers import one_token_per_req
from .base import make_scenario
from .common import make_request, make_scheduler


# --------------------------------------------------------------------------
# 1. Base scheduling (FCFS)
# --------------------------------------------------------------------------
def _build_base():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=False,
        policy="fcfs",
    )
    # 3 requests but max_num_seqs=2: the third must wait in the waiting queue
    # until one of the first two finishes, so the demo shows the waiting queue.
    reqs = [
        make_request("A", prompt_len=6, max_tokens=4, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=4, prompt_token=2),
        make_request("C", prompt_len=4, max_tokens=4, prompt_token=3),
    ]
    return sched, reqs, one_token_per_req()


def _build_chunked():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
    )
    reqs = [
        make_request("A", prompt_len=12, max_tokens=6, prompt_token=1),
        make_request("B", prompt_len=10, max_tokens=6, prompt_token=2),
    ]
    return sched, reqs, one_token_per_req()


def _build_prefix():
    sched = make_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        policy="fcfs",
    )
    # A and B share the exact same prompt prefix (same token -> same block
    # hashes). B is injected only after A's prefill has been cached, so its
    # prefix blocks are reused instead of recomputed.
    req_a = make_request("A", prompt_len=8, max_tokens=6, prompt_token=7)
    req_b = make_request("B", prompt_len=12, max_tokens=4, prompt_token=7)

    def inject_b(sched, step):
        # A runs until just before it finishes (8 prompt + 6 output = 14 tokens),
        # so its prefix blocks are cached; then admit B to hit the prefix cache.
        if step == 5 and "B" not in sched.requests:
            sched.add_request(req_b)

    return sched, [req_a], one_token_per_req(), inject_b


def _annotate_prefix(sched, step, so):
    lines = []
    for rid in [r.req_id for r in so.scheduled_new_reqs]:
        req = sched.requests.get(rid)
        if req is None:
            continue
        new_ids = req.num_computed_tokens
        cached = new_ids - so.num_scheduled_tokens.get(rid, 0)
        if cached <= 0:
            continue
        lines.append(
            f"前缀缓存命中：**{rid}** 与已缓存的 A 共享前缀 token=7，"
            f"前 {cached} 个 prompt token 直接复用 A 的 KV 块（不重算），"
            f"本步只调度 {so.num_scheduled_tokens.get(rid, 0)} 个新 token。"
        )
    return lines


def _annotate_preemption(sched, step, so):
    lines = []
    if so.preempted_req_ids:
        for rid in so.preempted_req_ids:
            req = sched.requests.get(rid)
            if req is None:
                continue
            lines.append(
                f"抢占原因：KV 块池已满——running 中的请求需要新的 KV 块"
                "（decode 扩容或续 prefill）但无可分配块。"
                f"priority 策略选择驱逐 priority 值最大（最低优先级）的 **{rid}**"
                f"（priority={req.priority}），释放其 KV 块。"
            )
    # Detect a resumed (recomputed) preempted request: it was preempted (computed
    # reset to 0) and is scheduled again via the resumed path.
    resumed_ids = getattr(so.scheduled_cached_reqs, "resumed_req_ids", set()) or set()
    for rid in resumed_ids:
        req = sched.requests.get(rid)
        if req is None:
            continue
        lines.append(
            f"抢占代价：**{rid}** 之前被抢占（第 {req.num_preemptions} 次），"
            f"已算好的 KV 全部作废，需从头重算 prompt（本步重算 "
            f"{so.num_scheduled_tokens.get(rid, 0)} 个 token），这是抢占的主要开销。"
        )
    return lines


def _build_preemption():
    # Few blocks so the KV pool runs out mid-run and forces a preemption.
    sched = make_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=6,
        enable_chunked_prefill=False,
        policy="priority",
    )
    reqs = [
        make_request("A", prompt_len=8, max_tokens=6, prompt_token=1, priority=0),
        make_request("B", prompt_len=8, max_tokens=8, prompt_token=2, priority=1),
        make_request("C", prompt_len=10, max_tokens=6, prompt_token=3, priority=2),
    ]
    return sched, reqs, one_token_per_req()


def _build_chunked_prefix():
    # Chunked prefill + prefix caching stacked: a long first request is split
    # into chunks, and a later request sharing its (already cached) prefix is
    # admitted to reuse the cached KV blocks instead of recomputing them.
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        policy="fcfs",
    )
    req_a = make_request("A", prompt_len=16, max_tokens=4, prompt_token=7)
    req_b = make_request("B", prompt_len=12, max_tokens=4, prompt_token=7)

    def inject_b(sched, step):
        # Admit B after A's prefix is cached (A is mid-chunked-prefill), so it
        # hits the shared prefix AND continues via chunking.
        if step == 2 and "B" not in sched.requests:
            sched.add_request(req_b)

    return sched, [req_a], one_token_per_req(), inject_b


def _annotate_chunked_prefix(sched, step, so):
    lines = []
    for rid in [r.req_id for r in so.scheduled_new_reqs]:
        req = sched.requests.get(rid)
        if req is None:
            continue
        new_ids = req.num_computed_tokens
        cached = new_ids - so.num_scheduled_tokens.get(rid, 0)
        if cached > 0:
            lines.append(
                f"前缀命中 + 分块：**{rid}** 复用 A 已缓存的 {cached} 个前缀 token"
                f"（不重算），本步只调度 {so.num_scheduled_tokens.get(rid, 0)} 个新 token"
                "继续 chunked prefill。"
            )
    return lines


SCENARIOS = [
    make_scenario(
        id="base",
        title="基础调度",
        group="基础循环",
        color="#4f8ff7",
        build=_build_base,
    ),
    make_scenario(
        id="chunked",
        title="Chunked Prefill",
        group="基础循环",
        color="#00b8a9",
        build=_build_chunked,
    ),
    make_scenario(
        id="prefix",
        title="前缀缓存",
        group="基础循环",
        color="#f7b84f",
        build=_build_prefix,
        annotate=_annotate_prefix,
    ),
    make_scenario(
        id="preemption",
        title="抢占",
        group="基础循环",
        color="#f75f5f",
        build=_build_preemption,
        annotate=_annotate_preemption,
    ),
    make_scenario(
        id="chunked_prefix",
        title="Chunked + 前缀",
        group="基础循环",
        color="#00b8a9",
        build=_build_chunked_prefix,
        annotate=_annotate_chunked_prefix,
    ),
]