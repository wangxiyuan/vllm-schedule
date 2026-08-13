"""SPDX-License-Identifier: Apache-2.0
Scenario registry and the shared frame extractor.

Each scenario is a dict describing how to build its scheduler/requests and
how to produce mock outputs. The shared extractor turns the scheduler +
SchedulerOutput into a JSON frame (cpu / gpu / kv / events / explanation).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput

from ..engine import SchedulerDriver
from ..kv_sim import extract_kv
from ..tensor_view import extract_tensor_worlds


def explain_schedule(
    sched,
    step: int,
    so: SchedulerOutput | None,
    *,
    scenario: str = "调度",
) -> list[str]:
    """Build a step-specific explanation from the real scheduler state.

    Mirrors the code path in ``Scheduler.schedule()``:
      1. running requests are scheduled first (decode / continued prefill),
      2. then requests are pulled off ``waiting`` (new prefill) until either
         the token budget runs out or ``max_num_running_reqs`` is reached,
      3. finished requests are removed and their KV blocks freed.
    """
    if so is None:
        waiting_ids = [r.request_id for r in sched.waiting]
        skipped_ids = [r.request_id for r in sched.skipped_waiting]
        where = []
        if waiting_ids:
            where.append(f"`waiting` 队列（{', '.join(waiting_ids)}）")
        if skipped_ids:
            where.append(f"`skipped_waiting` 队列（{', '.join(skipped_ids)}）")
        lines0 = [
            f"**第 0 帧（{scenario}前，step 0）**：请求通过 `add_request()` 加入调度器。",
            "每个请求经 `_enqueue_waiting_request()` 归类入队，尚未分配任何 KV 块"
            + (f"：{'; '.join(where)}。" if where else "。"),
        ]
        if skipped_ids:
            lines0.append(
                "部分请求因前置条件未满足（如等待结构化输出 grammar / 远程 KV）"
                "进入 `skipped_waiting`，待条件满足后才会被调度。"
            )
        lines0.append(
            "点「下一步 ▶」执行第 1 次 `schedule()`：调度器会优先调度 running 请求，"
            "再从 waiting / skipped_waiting 取新请求做 prefill。"
        )
        return lines0
    lines: list[str] = []
    tokens = so.num_scheduled_tokens
    new_ids = [r.req_id for r in so.scheduled_new_reqs]
    cached_ids = so.scheduled_cached_reqs.req_ids
    resumed_ids = getattr(so.scheduled_cached_reqs, "resumed_req_ids", set()) or set()
    # ``num_output_tokens`` is the request's output produced in PRIOR steps
    # (the extractor runs after schedule() but before update_from_output, which
    # appends this step's output). So out==0 means the request is still working
    # through its prompt (prefill); out>0 means it has started decoding.
    already_decoding = {
        rid: sched.requests[rid].num_output_tokens > 0
        for rid in cached_ids
        if rid in sched.requests
    }

    for rid in new_ids:
        req = sched.requests.get(rid)
        n = tokens.get(rid, 0)
        if req is not None:
            lines.append(
                f"第 {step} 步：从 waiting 取出 **{rid}**"
                f"（prompt {req.num_prompt_tokens} token）置为 RUNNING 做 prefill，"
                f"本步调度 {n} 个 token，为其分配 KV 块。"
            )
    for rid in cached_ids:
        req = sched.requests.get(rid)
        n = tokens.get(rid, 0)
        if req is None:
            continue
        if rid in resumed_ids:
            lines.append(
                f"**{rid}** 从抢占中恢复：重新进入 running，"
                f"从头重算 prompt（本步 {n} 个 token），不受之前已计算影响。"
            )
        elif not already_decoding.get(rid):
            done = req.num_computed_tokens >= req.num_tokens
            lines.append(
                f"**{rid}** 继续 prefill：本步再计算 {n} 个 prompt token"
                f"（已计算 {req.num_computed_tokens}/{req.num_tokens}）"
                + ("，本步完成 prefill，下步开始产出输出 token。" if done else "，尚未产出输出 token。")
            )
        else:
            if n > 1:
                lines.append(
                    f"**{rid}** decode（投机/多 token）：本步调度 {n} 个 token"
                    "（1 个目标 + 若干 draft）。"
                )
            else:
                lines.append(f"**{rid}** decode：本步只算 1 个新输出 token。")

    if not lines:
        lines.append(f"第 {step} 步：没有新调度的请求。")

    waiting_ids = [r.request_id for r in sched.waiting]
    if waiting_ids:
        # Distinguish the two real reasons a request stays in waiting: the
        # running-req slot limit, or the KV pool / full-ISL reservation.
        max_running = getattr(sched, "max_num_running_reqs", None)
        num_running = len(sched.running) + getattr(
            sched, "num_waiting_for_streaming_input", 0
        )
        if max_running is not None and num_running >= max_running:
            reason = f"受 max_num_running_reqs（{max_running}）限制，无法全部进入 running"
        else:
            reason = "空闲 KV 块不足（或整序列预留限制），暂无法分配块"
        lines.append(
            f"仍在 **waiting** 排队：{', '.join(waiting_ids)}（{reason}）。"
        )
    if so.finished_req_ids:
        lines.append(
            f"本步完成：{', '.join(sorted(so.finished_req_ids))}，"
            "从 running 移除并释放其 KV 块。"
        )
    if so.preempted_req_ids:
        lines.append(
            f"本步被抢占：{', '.join(sorted(so.preempted_req_ids))}，"
            "回到 waiting 且 num_computed_tokens 清零。"
        )
    return lines


def default_extract(scheduler, step: int, scheduler_output: SchedulerOutput | None) -> dict:
    """Shared extractor used by most scenarios.

    ``scheduler_output`` is None for the initial (step 0, pre-schedule) frame,
    in which case no scheduling events are reported.
    """
    cpu, gpu = extract_tensor_worlds(scheduler, step, scheduler_output)
    kv = extract_kv(scheduler, step)
    events: list[dict[str, Any]] = []

    if scheduler_output is not None:
        # The events enrichments below are added by the scenario via a hook if
        # needed; the base just reports scheduling counters.
        for rid, ntok in scheduler_output.num_scheduled_tokens.items():
            spec = scheduler_output.scheduled_spec_decode_tokens.get(rid)
            events.append(
                {
                    "type": "scheduled",
                    "req_id": rid,
                    "num_tokens": ntok,
                    "spec_tokens": list(spec) if spec else None,
                }
            )
        for rid in scheduler_output.preempted_req_ids or ():
            events.append({"type": "preempted", "req_id": rid})
        for rid in scheduler_output.finished_req_ids:
            events.append({"type": "finished", "req_id": rid})

    return {
        "cpu": cpu,
        "gpu": gpu,
        "kv": kv,
        "events": events,
        "explanation": [],
    }


def make_scenario(
    *,
    id: str,
    title: str,
    group: str,
    color: str,
    build: Callable[[], tuple[Any, list[Any], Callable[[Any, SchedulerOutput], Any], Any | None]],
    special_view: str | None = None,
    max_steps: int = 200,
    after_update: Callable[[SchedulerOutput, Any], None] | None = None,
    annotate: Callable[[Any, int, SchedulerOutput], list[str]] | None = None,
    explain: Callable[[Any, int, SchedulerOutput | None], list[str] | None] | None = None,
) -> dict:
    return {
        "id": id,
        "title": title,
        "group": group,
        "color": color,
        "build": build,
        "special_view": special_view,
        "max_steps": max_steps,
        "after_update": after_update,
        "annotate": annotate,
        "explain": explain,
    }


def run_scenario(scenario: dict) -> tuple[list[dict], dict]:
    """Build a scenario's scheduler and run it, returning JSON frames + meta.

    A scenario whose build returns ``scheduler=None`` is a hand-built,
    scheduler-free scene (e.g. parallel topology / DP balance): its build
    returns ``(None, frames, meta)`` directly.
    """
    built = scenario["build"]()
    if built[0] is None:
        # Hand-built scene: (None, frames, meta).
        frames, meta = built[1], built[2]
        for fr in frames:
            fr.setdefault("bg", [])
        return frames, meta

    scheduler, requests, make_output = built[0], built[1], built[2]
    # Optional 4th element: before_step(scheduler, step) to inject requests
    # mid-run (used by prefix-caching so the shared prefix is cached first).
    before_step = built[3] if len(built) > 3 else None
    # Optional 5th element: after_update(scheduler, scheduler_output, output)
    # to feed extra state after each step (e.g. draft tokens for spec decode).
    after_update = built[4] if len(built) > 4 else None
    for r in requests:
        scheduler.add_request(r)

    annotate = scenario.get("annotate")
    explain = scenario.get("explain")

    def extract(sched, step, so):
        # Per-step: only the dynamic, step-specific explanation of what the
        # scheduler did (step 0 = pre-schedule initial state). Static background
        # text is not repeated on every frame.
        payload = default_extract(sched, step, so)
        if explain is not None:
            payload["explanation"] = explain(sched, step, so) or []
        else:
            payload["explanation"] = explain_schedule(sched, step, so)
        if annotate is not None and so is not None:
            payload["explanation"].extend(annotate(sched, step, so))
        return payload

    driver = SchedulerDriver(
        scheduler,
        extract=extract,
        make_output=make_output,
        max_steps=scenario.get("max_steps", 200),
        before_step=before_step,
        after_update=after_update,
    )
    frames = driver.run()

    json_frames = []
    for f in frames:
        json_frames.append(
            {
                "step": f["step"],
                "cpu": f["cpu"],
                "gpu": f["gpu"],
                "kv": f["kv"],
                "events": f["events"],
                "explanation": f["explanation"],
                "bg": [],
                "meta": f["meta"],
            }
        )
    return json_frames, {
        "id": scenario["id"],
        "title": scenario["title"],
        "group": scenario["group"],
        "color": scenario["color"],
        "special_view": scenario.get("special_view"),
        "finished_reason": driver.finished_reason,
        "repeat_at": driver.repeat_at,
        "num_steps": len(json_frames),
    }