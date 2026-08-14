"""SPDX-License-Identifier: Apache-2.0
Advanced scenarios: speculative decoding, structured output, async
scheduling, and parallel / PD-disaggregation.

Speculative decoding and structured output drive a real scheduler and mock
outputs. Async scheduling drives a real ``AsyncScheduler``. Parallel / PD
disaggregation is a conceptual animation (no real scheduler) that also
illustrates the scheduler's real DP prefill-balancing logic.
"""
from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.structured_output import StructuredOutputGrammar

from samplers import _build_output, make_draft_token_ids, one_token_per_req
from scenarios.base import make_scenario
from scenarios.common import make_request, make_scheduler


# --------------------------------------------------------------------------
# 5. Speculative decoding
# --------------------------------------------------------------------------
def _build_spec():
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
    reqs = [
        make_request("A", prompt_len=8, max_tokens=12, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=12, prompt_token=2),
    ]

    # Acceptance cadence shared with _annotate_spec: even steps (2, 4, 6)
    # accept all drafts, odd steps (3, 5) reject them all. Deriving it from
    # the step number (instead of a closure toggle) guarantees the narration
    # and the simulated counters always agree.
    def sampler(sched, so: SchedulerOutput):
        accept = (getattr(sched, "current_step", 1) % 2) == 0
        n_acc = 3 if accept else 0
        return _build_output(sched, so, lambda rid, n: [1] + [2] * n_acc)

    def after_update(sched_obj, so: SchedulerOutput, model_output):
        sched_obj.update_draft_token_ids(make_draft_token_ids(so, 3))

    return sched, reqs, sampler, None, after_update


def _annotate_spec(sched, step, so):
    """Narrate the spec-decode accept/reject outcome.

    ``_build_spec``'s sampler accepts on even steps (2, 4, 6) and rejects on
    odd steps (3, 5), so this annotation mirrors that exact cadence.
    """
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        if not spec:
            continue
        accept = (step % 2) == 0
        n = len(spec)
        if accept:
            lines.append(
                f"投机命中：**{rid}** 的 {n} 个 draft 全部被接受，"
                f"本步一次推进 1 目标 + {n} 个 draft = {1 + n} 个 token（加速）。"
            )
        else:
            lines.append(
                f"投机未命中：**{rid}** 的 {n} 个 draft 被拒绝，"
                "回退（rollback）重算，本步只得 1 个目标 token。"
            )
    return lines


# --------------------------------------------------------------------------
# 6. Structured output
# --------------------------------------------------------------------------
class _DigitGrammar(StructuredOutputGrammar):
    """A minimal StructuredOutputGrammar that only accepts integer digits,
    driving the scheduler's structured-output promotion + bitmask path without
    needing a real tokenizer."""

    def __init__(self) -> None:
        self._tokens: list[int] = []
        self.terminated = False

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        ok = all(0 <= t <= 9 for t in tokens)
        if ok:
            self._tokens.extend(tokens)
        return ok

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        out = []
        for t in tokens:
            if 0 <= t <= 9:
                out.append(t)
            else:
                break
        return out

    def rollback(self, num_tokens: int) -> None:
        if num_tokens:
            del self._tokens[-num_tokens:]

    def fill_bitmask(self, bitmask, batch_index: int) -> None:
        # Allow digits 0-9, reject everything else.

        bitmask[batch_index].zero_()
        bitmask[batch_index, 0:10] = 1

    def is_terminated(self) -> bool:
        return self.terminated

    def reset(self) -> None:
        self._tokens = []
        self.terminated = False

    @property
    def state_text(self) -> str:
        return "digits: " + "".join(str(t) for t in self._tokens)


def _build_structured():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=False,
        policy="fcfs",
    )
    from vllm.sampling_params import StructuredOutputsParams

    reqs = [
        make_request(
            "A",
            prompt_len=6,
            max_tokens=8,
            prompt_token=1,
            structured_outputs=StructuredOutputsParams(regex="[0-9]+"),
        ),
    ]
    # Provide a working grammar so the request can be promoted to WAITING.
    grammar = _DigitGrammar()
    reqs[0].structured_output_request.grammar = grammar

    def before_step(sched, step):
        # Update the grammar's visible state for the top panel.
        pass

    return sched, reqs, one_token_per_req(), before_step


def _annotate_structured(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        grammar = getattr(getattr(req, "structured_output_request", None), "grammar", None)
        state = ""
        if grammar is not None:
            try:
                state = f"，当前状态：{grammar.state_text}"
            except Exception:
                state = ""
        lines.append(
            f"结构化约束：**{rid}** 受 grammar（正则 `[0-9]+`）约束，"
            f"每步只允许生成 0-9 的合法 token{state}。"
        )
    return lines


# --------------------------------------------------------------------------
# 7. Async scheduling
# --------------------------------------------------------------------------
def _build_async():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
        async_scheduling=True,
        pipeline_parallel_size=2,
    )
    reqs = [
        make_request("A", prompt_len=10, max_tokens=6, prompt_token=1),
        make_request("B", prompt_len=8, max_tokens=6, prompt_token=2),
    ]
    return sched, reqs, one_token_per_req()


SCENARIOS = [
    make_scenario(
        id="spec",
        title="投机解码",
        group="基础循环",
        color="#9b59b6",
        build=_build_spec,
        annotate=_annotate_spec,
        special_view="spec_acceptance",
    ),
    make_scenario(
        id="structured",
        title="结构化输出",
        group="基础循环",
        color="#2ecc71",
        build=_build_structured,
        annotate=_annotate_structured,
        special_view="structured_fsm",
    ),
    make_scenario(
        id="async",
        title="异步调度",
        group="工程架构",
        color="#e67e22",
        build=_build_async,
        special_view="pipeline",
    ),
]