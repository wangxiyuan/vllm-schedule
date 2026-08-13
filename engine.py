"""SPDX-License-Identifier: Apache-2.0
Common scheduling-driver engine for the vLLM scheduler visualization.

Drives a real ``Scheduler``/``AsyncScheduler`` instance (no GPU required),
runs the ``schedule() -> mock model output -> update_from_output()`` loop,
and records a per-step frame (JSON-serializable dict) capturing the CPU
python objects and (simulated) GPU tensors.

A driver is given:
  * ``extract(scheduler, step, scheduler_output)`` -> dict with keys
    ``cpu``, ``gpu``, ``kv``, ``events``, ``explanation``.
  * ``make_output(scheduler_output)`` -> a mock ``ModelRunnerOutput``.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput


class SchedulerDriver:
    """Drives a real scheduler through scheduling steps and records frames."""

    def __init__(
        self,
        scheduler,
        *,
        extract: Callable[..., dict[str, Any]],
        make_output: Callable[[Any, SchedulerOutput], ModelRunnerOutput],
        max_steps: int = 200,
        token_level_detect: bool = True,
        before_step: Callable[[Any, int], None] | None = None,
        after_update: Callable[[Any, SchedulerOutput, ModelRunnerOutput], None] | None = None,
        capture_initial: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.extract = extract
        self.make_output = make_output
        self.max_steps = max_steps
        self.token_level_detect = token_level_detect
        self.before_step = before_step
        self.after_update = after_update
        self.capture_initial = capture_initial
        self.frames: list[dict[str, Any]] = []
        self._seen_states: dict[str, int] = {}
        self.finished_reason: str | None = None
        self.repeat_at: int | None = None

    def run(self) -> list[dict[str, Any]]:
        """Run scheduling steps until the system reaches a repeated state
        (token-level) or the hard step cap."""
        # Capture the pre-schedule initial state as step 0 so readers see the
        # starting point (requests queued in waiting) before the first
        # schedule() moves them into running.
        if self.capture_initial:
            payload = self.extract(self.scheduler, 0, None)
            self.frames.append(
                {
                    "step": 0,
                    "cpu": payload.get("cpu", {}),
                    "gpu": payload.get("gpu", {}),
                    "kv": payload.get("kv", {}),
                    "events": payload.get("events", []),
                    "explanation": payload.get("explanation", []),
                    "meta": {"initial": True},
                }
            )

        for step in range(1, self.max_steps + 1):
            if self.before_step:
                self.before_step(self.scheduler, step)
            scheduler_output = self.scheduler.schedule()
            payload = self.extract(self.scheduler, step, scheduler_output)
            frame = {
                "step": step,
                "cpu": payload.get("cpu", {}),
                "gpu": payload.get("gpu", {}),
                "kv": payload.get("kv", {}),
                "events": payload.get("events", []),
                "explanation": payload.get("explanation", []),
                "meta": {},
            }
            self.frames.append(frame)

            if self.token_level_detect:
                state = self._state_signature()
                if state in self._seen_states:
                    self.repeat_at = self._seen_states[state]
                    self.finished_reason = "loop"
                    break
                self._seen_states[state] = step

            if (
                not self.scheduler.has_requests()
                # A paused streaming session (WAITING_FOR_STREAMING_REQ) is
                # excluded from vLLM's unfinished count but is not finished:
                # keep the run alive so the continuation can be fed in.
                and getattr(
                    self.scheduler, "num_waiting_for_streaming_input", 0
                )
                == 0
            ):
                self.finished_reason = "all_done"
                break

            model_output = self.make_output(self.scheduler, scheduler_output)
            self.scheduler.update_from_output(scheduler_output, model_output)
            if self.after_update:
                self.after_update(self.scheduler, scheduler_output, model_output)
        else:
            self.finished_reason = "cap"

        if self.frames:
            self.frames[-1]["meta"]["finished_reason"] = self.finished_reason
            if self.repeat_at is not None:
                self.frames[-1]["meta"]["repeat_at"] = self.repeat_at
        return self.frames

    def _state_signature(self) -> str:
        """Token-level signature: for each request, the tuple
        (status, num_computed_tokens, num_tokens, block ids)."""
        sched = self.scheduler
        parts = []
        for rid in sorted(sched.requests.keys()):
            req = sched.requests[rid]
            blocks: list[int] = []
            try:
                kb = sched.kv_cache_manager.get_blocks(rid)
                if kb is not None:
                    for group in kb.blocks:
                        blocks.extend(b.block_id for b in group)
            except Exception:
                blocks = []
            parts.append(
                (
                    rid,
                    req.status.name,
                    req.num_computed_tokens,
                    req.num_tokens,
                    tuple(sorted(blocks)),
                )
            )
        return hashlib.sha256(repr(parts).encode()).hexdigest()