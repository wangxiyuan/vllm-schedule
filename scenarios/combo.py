"""SPDX-License-Identifier: Apache-2.0
Combo scenarios: scheduling features that genuinely stack and are driven by a
real scheduler, so the visualization shows the combined effect faithfully.

  - spec_structured : speculative decoding + structured-output grammar
  - async_pp        : async scheduling + pipeline parallelism (PP micro-batching)
  - async_spec      : async scheduling + speculative decoding
  - pp_spec         : pipeline parallelism + speculative decoding
"""
from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from samplers import _req_ids, make_draft_token_ids, one_token_per_req
from scenarios.advanced import _DigitGrammar
from scenarios.base import make_scenario
from scenarios.common import make_request, make_scheduler


def _spec_sampler(num_spec_tokens: int):
    """Emit 1 target + N accepted drafts (all conforming) for spec requests."""

    def sampler(sched, so: SchedulerOutput) -> ModelRunnerOutput:
        req_ids = _req_ids(so)
        spec = so.scheduled_spec_decode_tokens
        sampled = []
        for rid in req_ids:
            req = sched.requests[rid]
            if getattr(req, "is_prefill_chunk", False):
                sampled.append([])
            elif rid in spec:
                sampled.append([1] + [2] * num_spec_tokens)
            else:
                sampled.append([1])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
        )

    return sampler


# --------------------------------------------------------------------------
# spec + structured output
# --------------------------------------------------------------------------
def _build_spec_structured():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=False,
        policy="fcfs",
        num_speculative_tokens=3,
    )
    from vllm.sampling_params import StructuredOutputsParams

    req = make_request(
        "A",
        prompt_len=6,
        max_tokens=12,
        prompt_token=1,
        structured_outputs=StructuredOutputsParams(regex="[0-9]+"),
    )
    req.structured_output_request.grammar = _DigitGrammar()
    reqs = [req]

    def after_update(sched_obj, so: SchedulerOutput, model_output):
        sched_obj.update_draft_token_ids(make_draft_token_ids(so, 3))

    return sched, reqs, _spec_sampler(3), None, after_update


def _annotate_spec_structured(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        if not spec:
            continue
        lines.append(
            f"投机 + 结构化：**{rid}** 的 {len(spec)} 个 draft 同时受 grammar 约束"
            "（`[0-9]+`）校验，非法 draft 会被过滤回退；本步保留合法 draft 加速解码。"
        )
    return lines


# --------------------------------------------------------------------------
# async + PP micro-batching
# --------------------------------------------------------------------------
def _build_async_pp():
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


def _annotate_async_pp(sched, step, so):
    lines = []
    decoded = [rid for rid in so.scheduled_cached_reqs.req_ids]
    for rid in decoded:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        lines.append(
            f"PP 微拍：**{rid}** 受 pp_size=2 约束，相邻两次 decode 至少间隔 2 步"
            "（`next_decode_eligible_step = current_step + pp_size`），"
            "本步正好到达可 decode 的时机。"
        )
    return lines


# --------------------------------------------------------------------------
# async + spec
# --------------------------------------------------------------------------
def _build_async_spec():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
        async_scheduling=True,
        num_speculative_tokens=3,
        spec_method="ngram_gpu",
    )
    reqs = [
        make_request("A", prompt_len=8, max_tokens=12, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=12, prompt_token=2),
    ]

    def after_update(sched_obj, so: SchedulerOutput, model_output):
        sched_obj.update_draft_token_ids(make_draft_token_ids(so, 3))

    return sched, reqs, _spec_sampler(3), None, after_update


def _annotate_async_spec(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        req0 = sched.requests[rid]
        place = getattr(req0, "num_output_placeholders", 0)
        if not spec:
            continue
        lines.append(
            f"异步 + 投机：**{rid}** 的 {len(spec)} 个 draft 已排入调度；"
            f"异步调度同时登记 {place} 个输出占位（placeholders）超前流水线。"
        )
    return lines


# --------------------------------------------------------------------------
# PP + spec
# --------------------------------------------------------------------------
def _build_pp_spec():
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
        num_speculative_tokens=3,
        spec_method="ngram_gpu",
    )
    reqs = [
        make_request("A", prompt_len=8, max_tokens=12, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=12, prompt_token=2),
    ]

    def after_update(sched_obj, so: SchedulerOutput, model_output):
        sched_obj.update_draft_token_ids(make_draft_token_ids(so, 3))

    return sched, reqs, _spec_sampler(3), None, after_update


def _annotate_pp_spec(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        if not spec:
            continue
        lines.append(
            f"PP + 投机：**{rid}** 同时受 pp_size=2 微拍错拍与 {len(spec)} 个 draft"
            "加速约束；decode 被 pp 节奏节流，但每次 decode 仍验证多个 draft。"
        )
    return lines


SCENARIOS = [
    make_scenario(
        id="spec_structured",
        title="投机 + 结构化",
        group="基础循环",
        color="#8e44ad",
        build=_build_spec_structured,
        annotate=_annotate_spec_structured,
        special_view="spec_acceptance",
    ),
    make_scenario(
        id="async_pp",
        title="异步 + 流水线并行",
        group="工程架构",
        color="#d35400",
        build=_build_async_pp,
        annotate=_annotate_async_pp,
        special_view="pipeline",
    ),
    make_scenario(
        id="async_spec",
        title="异步 + 投机",
        group="工程架构",
        color="#c0392b",
        build=_build_async_spec,
        annotate=_annotate_async_spec,
        special_view="pipeline",
    ),
    make_scenario(
        id="pp_spec",
        title="流水线并行 + 投机",
        group="工程架构",
        color="#16a085",
        build=_build_pp_spec,
        annotate=_annotate_pp_spec,
    ),
]