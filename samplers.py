"""SPDX-License-Identifier: Apache-2.0
Mock ``ModelRunnerOutput`` samplers for the vLLM scheduler visualization.

Each returns a ``ModelRunnerOutput`` that mimics what the GPU worker would
produce for a just-scheduled step, without running any model.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput


def _req_ids(scheduler_output: SchedulerOutput) -> list[str]:
    ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
    ids.extend(scheduler_output.scheduled_cached_reqs.req_ids)
    return ids


def _is_prefill(scheduler, req_id: str, scheduler_output: SchedulerOutput) -> bool:
    """Whether this request's just-scheduled forward is a (partial) prefill.

    Mirrors the real worker: a request is still prefilling when its computed
    tokens have not yet caught up to num_tokens + placeholders. We read the
    post-schedule ``is_prefill_chunk`` flag, which is authoritative.
    """
    req = scheduler.requests.get(req_id)
    if req is None:
        return False
    return bool(getattr(req, "is_prefill_chunk", False))


def _build_output(
    scheduler,
    scheduler_output: SchedulerOutput,
    tokens_for: Callable[[str, int], list[int]],
) -> ModelRunnerOutput:
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    req_ids = _req_ids(scheduler_output)
    sampled = []
    for r in req_ids:
        if _is_prefill(scheduler, r, scheduler_output):
            # A partial prefill forward samples no token.
            sampled.append([])
        elif r not in spec_tokens:
            # Full prefill (this step) or plain decode: exactly one target
            # token, no spec drafts.
            sampled.append([1])
        else:
            n = scheduler_output.num_scheduled_tokens.get(r, 0)
            sampled.append(tokens_for(r, n))
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
    )


def one_token_per_req() -> Callable[[Any, SchedulerOutput], ModelRunnerOutput]:
    """Each (non-prefill) request emits exactly one new token (decode)."""
    return lambda sched, so: _build_output(sched, so, lambda rid, n: [1])


def spec_decode(
    num_spec_tokens: int, accept_rate: float = 0.8
) -> Callable[[Any, SchedulerOutput], ModelRunnerOutput]:
    """Speculative decoding sampler.

    Each request emits ``1 + round(accept_rate * num_spec_tokens)`` tokens in
    the sampled output. The scheduler's update_from_output compares this count
    against the scheduled draft tokens and rolls back the rejected drafts.
    """
    n_accepted = max(0, min(num_spec_tokens, int(round(accept_rate * num_spec_tokens))))

    def sampler(sched, so: SchedulerOutput) -> ModelRunnerOutput:
        return _build_output(sched, so, lambda rid, n: [1] + [2] * n_accepted)

    return sampler


def structured_output() -> Callable[[Any, SchedulerOutput], ModelRunnerOutput]:
    """Structured output: each (non-prefill) request emits one grammar-conforming
    token."""
    return lambda sched, so: _build_output(sched, so, lambda rid, n: [1])


def diffusion(
    canvas_length: int,
) -> Callable[[Any, SchedulerOutput], ModelRunnerOutput]:
    """Discrete-diffusion (dLLM) sampler.

    Diffusion models have ``num_sampled_tokens_per_step == 0``: instead of one
    AR target token, each denoising step emits the whole canvas (spec) block.
    Emitting exactly the scheduled canvas block makes every block accepted, so
    the scheduler advances ``canvas_length`` tokens per step on the decode path.
    """
    del canvas_length  # count comes from the scheduled spec tokens

    def sampler(sched, so: SchedulerOutput) -> ModelRunnerOutput:
        spec_tokens = so.scheduled_spec_decode_tokens
        req_ids = []
        sampled = []
        for r in so.scheduled_new_reqs:
            req_ids.append(r.req_id)
            sampled.append([])
        for rid in so.scheduled_cached_reqs.req_ids:
            req_ids.append(rid)
            sp = spec_tokens.get(rid)
            # num_sampled_tokens_per_step==0: emit the canvas block only, with
            # no bonus AR token.
            sampled.append([1] * len(sp) if sp else [1])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
        )

    return sampler


def make_draft_token_ids(
    scheduler_output: SchedulerOutput, num_spec_tokens: int
) -> DraftTokenIds:
    """Build draft (spec) token ids for the next step."""
    req_ids = _req_ids(scheduler_output)
    return DraftTokenIds(
        req_ids=req_ids,
        draft_token_ids=[[2] * num_spec_tokens for _ in req_ids],
    )