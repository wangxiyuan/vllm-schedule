"""SPDX-License-Identifier: Apache-2.0
Parallel / PD-disaggregation scenarios.

``parallel`` is a real, multi-step PD-disaggregation walkthrough: it drives a
real ``Scheduler`` as the *prefill instance* (producer) through chunked
prefill, then hands the produced KV blocks to a *decode instance* (consumer)
for the decode phase. Each frame captures the two instances' state plus the
in-flight KV transfer, so the viewer sees the whole prefill → KV transfer →
decode lifecycle.

``pp`` drives a real ``AsyncScheduler`` with pipeline parallelism, showing the
PP micro-batching cadence (``next_decode_eligible_step = current_step +
pp_size``).

``dp_balance`` is a hand-built DP prefill-balancing decision strip (no real
scheduler) illustrating the scheduler's real ``defer_prefills`` /
``prefill_capacity_bound`` cadence logic.
"""
from __future__ import annotations

from typing import Any

from .base import make_scenario
from .common import make_request, make_scheduler
from ..samplers import one_token_per_req

# How many output tokens the decode (consumer) instance produces per step.
_DECODE_TOKENS_PER_STEP = 1


def _prefill_only_output(scheduler, scheduler_output):
    """Model output for a prefill-only producer: never emit any output token,
    so the producer instance only ever computes prompt KV (no decode). Requests
    whose prefill is already complete return empty and simply idle until the
    KV is transferred to the decode instance."""
    from ..samplers import _req_ids

    req_ids = _req_ids(scheduler_output)
    return _make_output(req_ids, [[] for _ in req_ids])


def _make_output(req_ids, sampled):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


def _build_parallel():
    """Prellfill instance (real scheduler) runs chunked prefill; the output
    blocks are then "transferred" to a decode instance which decodes them."""
    block_size = 4
    # ---- Prefill instance (producer): chunked prefill of two requests ----
    prod = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        block_size=block_size,
        num_blocks=24,
        enable_chunked_prefill=True,
        policy="fcfs",
    )
    req_a = make_request("A", prompt_len=14, max_tokens=10, prompt_token=1)
    req_b = make_request("B", prompt_len=10, max_tokens=8, prompt_token=2)
    prod.add_request(req_a)
    prod.add_request(req_b)

    frames: list[dict[str, Any]] = []
    ALLRID = ["A", "B"]
    # run producer through its prefill steps, recording each step's state
    producer_steps: list[dict[str, Any]] = []
    step = 1

    def _both_prefilled() -> bool:
        return all(
            rid not in prod.requests
            or prod.requests[rid].num_computed_tokens >= prod.requests[rid].num_tokens
            for rid in ALLRID
        )

    while prod.has_requests() and step <= 20 and not _both_prefilled():
        so = prod.schedule()
        producer_steps.append(
            {
                "prod_blocks": {
                    rid: list(prod.kv_cache_manager.get_blocks(rid).get_block_ids()[0])
                    if prod.kv_cache_manager.get_blocks(rid) is not None
                    else []
                    for rid in prod.requests
                },
                "prod_computed": {
                    rid: prod.requests[rid].num_computed_tokens
                    for rid in prod.requests
                },
                "prod_out": {
                    rid: len(prod.requests[rid]._output_token_ids)
                    for rid in prod.requests
                },
                "prod_prefill_chunk": {
                    rid: prod.requests[rid].is_prefill_chunk for rid in prod.requests
                },
                "scheduled": dict(so.num_scheduled_tokens),
            }
        )
        # Prefill-only producer: never emit output tokens during the prefill
        # phase, so all generated tokens happen on the decode instance.
        mro = _prefill_only_output(prod, so)
        prod.update_from_output(so, mro)
        step += 1

    def _blocks_of(ps):
        return {rid: ps["prod_blocks"].get(rid, []) for rid in ALLRID}

    final_blocks = _blocks_of(producer_steps[-1])

    # ---- Decode instance (consumer): decode the transferred KV ----
    # decode produces tokens on top of the (already computed) prompt KV
    decode_blocks = dict(final_blocks)  # consumer now owns these blocks
    decode_out = {rid: 0 for rid in ALLRID}
    # Final computed tokens per request come from the last producer step.
    decode_computed = {
        rid: producer_steps[-1]["prod_computed"].get(rid, 0) for rid in ALLRID
    }

    # Build frames: first the producer prefill steps, then the transfer moment,
    # then the decode steps.
    for i, ps in enumerate(producer_steps):
        active = [rid for rid in ALLRID if ps["scheduled"].get(rid)]
        frames.append(
            _pd_frame(
                step=i + 1,
                phase="prefill",
                producer_hint="Prefill 实例（producer）",
                active=active,
                prod_blocks=ps["prod_blocks"],
                prod_computed=ps["prod_computed"],
                prod_out=ps["prod_out"],
                prod_prefill_chunk=ps["prod_prefill_chunk"],
                decode_blocks={rid: [] for rid in ALLRID},
                decode_out={rid: 0 for rid in ALLRID},
                decode_computed={rid: 0 for rid in ALLRID},
                transferred=[],
                explanation=_prefill_expl(step=i + 1, scheduled=ps["scheduled"]),
            )
        )

    # The KV transfer step: producer's blocks move to the decode instance.
    transfer_step = len(producer_steps) + 1
    frames.append(
        _pd_frame(
            step=transfer_step,
            phase="transfer",
            producer_hint="KV connector 传输",
            active=ALLRID,
            prod_blocks={rid: [] for rid in ALLRID},
            prod_computed={rid: 0 for rid in ALLRID},
            prod_out={rid: 0 for rid in ALLRID},
            prod_prefill_chunk={rid: False for rid in ALLRID},
            decode_blocks=decode_blocks,
            decode_out={rid: 0 for rid in ALLRID},
            decode_computed=decode_computed,
            transferred=list(final_blocks.keys()),
            explanation=_transfer_expl(final_blocks),
        )
    )

    # Decode steps on the consumer instance.
    for d in range(1, 6):
        step = transfer_step + d
        decode_out = {rid: d for rid in ALLRID}
        decode_computed = {
            rid: decode_computed[rid] + _DECODE_TOKENS_PER_STEP for rid in ALLRID
        }
        frames.append(
            _pd_frame(
                step=step,
                phase="decode",
                producer_hint="Decode 实例（consumer）",
                active=ALLRID,
                prod_blocks={rid: [] for rid in ALLRID},
                prod_computed={rid: 0 for rid in ALLRID},
                prod_out={rid: 0 for rid in ALLRID},
                prod_prefill_chunk={rid: False for rid in ALLRID},
                decode_blocks=decode_blocks,
                decode_out=decode_out,
                decode_computed=decode_computed,
                transferred=[],
                explanation=_decode_expl(step=step, out=decode_out),
            )
        )

    meta = {
        "id": "parallel",
        "title": "PD 分离",
        "group": "工程架构",
        "color": "#7f8c8d",
        "special_view": "parallel_topology",
        "finished_reason": "all_done",
        "repeat_at": None,
        "num_steps": len(frames),
    }
    return None, frames, meta


def _pd_frame(
    *,
    step: int,
    phase: str,
    producer_hint: str,
    active: list[str],
    prod_blocks: dict[str, list[int]],
    prod_computed: dict[str, int],
    prod_out: dict[str, int],
    prod_prefill_chunk: dict[str, bool],
    decode_blocks: dict[str, list[int]],
    decode_out: dict[str, int],
    decode_computed: dict[str, int],
    transferred: list[str],
    explanation: list[str],
) -> dict[str, Any]:
    def _reqs(blocks, computed, out, chunk):
        r = {}
        for rid, blks in blocks.items():
            r[rid] = {
                "id": rid,
                "status": "RUNNING",
                "num_computed_tokens": computed.get(rid, 0),
                "num_output_tokens": out.get(rid, 0),
                "blocks": blks,
                "is_prefill_chunk": chunk.get(rid, False),
            }
        return r

    return {
        "step": step,
        "cpu": {
            "requests": {},
            "waiting": [],
            "running": [],
            "skipped_waiting": [],
            "pd": {
                "phase": phase,
                "producer_hint": producer_hint,
                "active": active,
                "transferred": transferred,
                "producer": _reqs(
                    prod_blocks, prod_computed, prod_out, prod_prefill_chunk
                ),
                "consumer": _reqs(
                    decode_blocks, decode_computed, decode_out, {}
                ),
            },
        },
        "gpu": {"device": "cuda", "node": "pd_topology"},
        "kv": {
            "pool": {"num_blocks": 0, "usage": 0.0, "per_block": [], "free_order": []},
            "requests": {},
        },
        "events": [],
        "explanation": explanation,
        "meta": {},
    }


def _prefill_expl(step: int, scheduled: dict[str, int]) -> list[str]:
    sched = " ".join(f"{k}={n} token" for k, n in scheduled.items())
    if step == 1:
        return [
            "PD 分离的第 1 步：请求 A、B 进入 **Prefill 实例**（kv_role=producer）。",
            "producer 用 chunked prefill 逐步吃掉 prompt，每步调度 max_num_batched_tokens "
            "个 token，KV 块随之增长。",
            f"本步调度：{sched}。处于 prefill chunk 的请求暂不产出输出 token。",
        ]
    return [
        f"Prefill 实例继续 chunked prefill。本步调度：{sched}。",
        "KV 块在 producer 的块池中分配，块 id 随计算进度增长。",
    ]


def _transfer_expl(final_blocks: dict[str, list[int]]) -> list[str]:
    blk = "，".join(f"{rid}:{b}" for rid, b in final_blocks.items())
    return [
        "**KV 传输**：prefill 完成后，producer 把每个请求的 KV 块通过 KV connector "
        "（kv_role=producer → consumer）传给 decode 实例。",
        f"A、B 的 prompt 全部计算完毕，KV 块所有权移交：{blk}。",
        "decode 实例拿到 KV 后，无需重新 prefill，直接从已有 KV 开始 generate。",
    ]


def _decode_expl(step: int, out: dict[str, int]) -> list[str]:
    o = "，".join(f"{rid}:{n}" for rid, n in out.items())
    return [
        f"Decode 实例每步为每个请求生成 {_DECODE_TOKENS_PER_STEP} 个输出 token。",
        f"当前输出进度：{o}。KV 块保持被持有，供后续 decode 继续追加。",
        "与单实例相比，decode 实例不为长 prompt 反复做 prefill，吞吐更高。",
    ]


def _build_dp_balance():
    # Hand-built DP prefill-balancing decision strip (no real scheduler),
    # mirroring the scheduler's real logic:
    #   defer_prefills = throttle_prefills AND NOT prefill_capacity_bound
    #                    AND any running request is decoding
    #   prefill_capacity_bound = waiting queue is non-empty (prefill demand is
    #                            the bottleneck, so prefill must NOT be deferred)
    # On cadence-aligned steps the prefill instance admits and drains the
    # waiting queue (capacity-bound, no defer); on the throttled (off-cadence)
    # steps in between, the queue is empty so prefill is not saturated and the
    # instance defers prefill compute to let decode run ahead.
    steps = []
    for step in range(1, 13):
        cadence_aligned = step % 2 == 0
        has_decode = step >= 2
        # New prefill demand arrives before each aligned step; throttled steps
        # see it drained.
        waiting = 3 if cadence_aligned else 0
        defer = (not cadence_aligned) and has_decode and waiting == 0
        prefill_capacity_bound = waiting > 0
        steps.append(
            {
                "step": step,
                "cpu": {
                    "requests": {},
                    "waiting": ["r1", "r2", "r3"] if waiting else [],
                    "running": [],
                    "skipped_waiting": [],
                    "dp_balance": {
                        "cadence_aligned": cadence_aligned,
                        "has_decode": has_decode,
                        "waiting": waiting,
                        "defer_prefills": bool(defer),
                        "prefill_capacity_bound": prefill_capacity_bound,
                    },
                },
                "gpu": {"device": "cuda"},
                "kv": {
                    "pool": {
                        "num_blocks": 0,
                        "usage": 0.0,
                        "per_block": [],
                        "free_order": [],
                    },
                    "requests": {},
                },
                "events": [],
                "explanation": [
                    f"step {step}: cadence_aligned={cadence_aligned}, "
                    f"has_decode={has_decode}, waiting={waiting} → "
                    f"defer_prefills={bool(defer)}"
                    + (
                        "（off-cadence 步 + waiting 已清空 → prefill 未饱和，"
                        "defer 让位给 decode）"
                        if defer
                        else (
                            "（cadence 对齐步，集中投放 prefill）"
                            if cadence_aligned
                            else "（无 decode 可让位）"
                        )
                    ),
                ],
                "meta": {},
            }
        )
    meta = {
        "id": "dp_balance",
        "title": "DP 预填充均衡",
        "group": "工程架构",
        "color": "#34495e",
        "special_view": "dp_balance",
        "finished_reason": "all_done",
        "repeat_at": None,
        "num_steps": len(steps),
    }
    return None, steps, meta


def _build_pp():
    """Pipeline parallelism: a real AsyncScheduler with pp_size=2. The PP
    micro-batching cadence limits each request to a decode every `pp_size`
    steps (`next_decode_eligible_step = current_step + pp_size`), visible as
    the alternating blank / decode steps."""
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
        async_scheduling=True,
        pipeline_parallel_size=2,
        use_v2_model_runner=True,
    )
    reqs = [
        make_request("A", prompt_len=8, max_tokens=6, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=6, prompt_token=2),
    ]
    return sched, reqs, one_token_per_req()


def _annotate_pp(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        lines.append(
            f"PP 微拍：**{rid}** 相邻两次 decode 至少间隔 pp_size=2 步"
            "（`next_decode_eligible_step = current_step + pp_size`），"
            "本步是其可 decode 步。"
        )
    return lines


def make_parallel_scenarios():
    return [
        make_scenario(
            id="parallel",
            title="PD 分离",
            group="工程架构",
            color="#7f8c8d",
            build=_build_parallel,
            special_view="parallel_topology",
        ),
        make_scenario(
            id="pp",
            title="流水线并行 PP",
            group="工程架构",
            color="#2980b9",
            build=_build_pp,
            annotate=_annotate_pp,
        ),
        make_scenario(
            id="dp_balance",
            title="DP 预填充均衡",
            group="工程架构",
            color="#34495e",
            build=_build_dp_balance,
            special_view="dp_balance",
        ),
    ]