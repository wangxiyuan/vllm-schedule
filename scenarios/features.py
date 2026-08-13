"""SPDX-License-Identifier: Apache-2.0
Feature scenarios: reservation / long-prefill / streaming / Mamba / diffusion.

Each drives a real scheduler (CPU mock) with a config feature enabled, so the
animation matches the exact code path that feature exercises.
"""

from __future__ import annotations

from vllm.v1.core.sched.output import SchedulerOutput

from ..samplers import diffusion, make_draft_token_ids, one_token_per_req
from .base import make_scenario
from .common import make_request, make_scheduler


# --------------------------------------------------------------------------
# Long-prefill threshold
# --------------------------------------------------------------------------
def _build_long_prefill():
    # A prompt far above the prefill budget: the threshold caps each chunk so
    # the scheduler yields the token budget to other requests mid-prefill.
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
        long_prefill_token_threshold=8,
    )
    reqs = [
        make_request("A", prompt_len=20, max_tokens=4, prompt_token=1),
        make_request("B", prompt_len=6, max_tokens=4, prompt_token=2),
    ]
    return sched, reqs, one_token_per_req()


def _annotate_long_prefill(sched, step, so):
    lines = []
    for rid in so.num_scheduled_tokens:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens > 0:
            continue
        n = so.num_scheduled_tokens.get(rid, 0)
        lines.append(
            f"long_prefill 阈值：**{rid}** 的 prompt 超过阈值，"
            f"本步 prefill 被限制为 {n} 个 token（其余请求得以调度）。"
        )
    return lines


# --------------------------------------------------------------------------
# Scheduler full-ISL reservation
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Scheduler full-ISL reservation (hand-built decision strip)
# --------------------------------------------------------------------------
def _build_reserve_full_isl():
    # Hand-built contrast of scheduler_reserve_full_isl on/off. The real
    # scheduler's admission code (kv_cache_manager.allocate_slots) checks
    # ``full_sequence_must_fit``: when True, a new request is admitted only if
    # its full sequence (request.num_tokens at admission = the prompt) fits in
    # the free pool up front; when False, it may be admitted and only preempted
    # later.
    #
    # Pool = 4 blocks = 16 tokens (block_size=4). A's prompt (8 tokens) is
    # admitted first and holds 2 blocks. B's prompt (12 tokens) needs 3 blocks
    # at admission — more than the 2 free blocks left, so with the reserve on
    # B is rejected; with it off B is admitted anyway.
    pool_blocks = 4
    a_blocks = 2
    free_after_a = pool_blocks - a_blocks
    b_full_blocks = 3  # 12 prompt tokens / block_size 4

    def _strip(reserve: bool):
        steps = []
        # Step 1: A admitted (2 blocks), B requests admission.
        fits = free_after_a >= b_full_blocks
        admitted = fits if reserve else True
        steps.append(
            {
                "step": 1,
                "cpu": {
                    "requests": {},
                    "waiting": ["B"] if not admitted else [],
                    "running": ["A"],
                    "skipped_waiting": [],
                    "reserve": {
                        "reserve_full_isl": reserve,
                        "a_blocks": a_blocks,
                        "b_full_blocks": b_full_blocks,
                        "free_after_a": free_after_a,
                        "fits": fits,
                        "admitted": admitted,
                        "decision": (
                            "整序列无法容纳 → 拒绝准入，B 留在 waiting"
                            if not admitted
                            else "准予准入 B"
                        ),
                    },
                },
                "gpu": {"device": "cuda"},
                "kv": {
                    "pool": {
                        "num_blocks": pool_blocks,
                        "usage": a_blocks / pool_blocks,
                        "per_block": [],
                        "free_order": [],
                    },
                    "requests": {},
                },
                "events": [],
                "explanation": [
                    (
                        "**"
                        + (
                            "reserve_full_isl=True"
                            if reserve
                            else "reserve_full_isl=False"
                        )
                        + "**：A 已占用 "
                        + str(a_blocks)
                        + " 块，剩余 "
                        + str(free_after_a)
                        + " 块。"
                    ),
                    (
                        "B 的完整序列（准入时 `num_tokens` = prompt 12 token，"
                        "block_size=4）需要 "
                        + str(b_full_blocks)
                        + " 块，"
                        + ("可以容纳" if fits else "无法容纳")
                        + "。"
                    ),
                    (
                        "整序列预留开启：B 的完整序列必须一次性装下，否则拒绝准入，"
                        "B 只能留在 waiting 等 A 释放。"
                        if reserve
                        else "整序列预留关闭：B 先准入，输出不足再抢占。"
                    ),
                ],
                "meta": {},
            }
        )
        # Step 2: outcome.
        steps.append(
            {
                "step": 2,
                "cpu": {
                    "requests": {},
                    "waiting": ["B"] if not admitted else [],
                    "running": ["A"] + ([] if not admitted else ["B"]),
                    "skipped_waiting": [],
                    "reserve": {
                        "reserve_full_isl": reserve,
                        "a_blocks": a_blocks,
                        "b_full_blocks": b_full_blocks,
                        "free_after_a": free_after_a,
                        "fits": fits,
                        "admitted": admitted,
                        "decision": "等待中" if not admitted else "已准入",
                    },
                },
                "gpu": {"device": "cuda"},
                "kv": {
                    "pool": {
                        "num_blocks": pool_blocks,
                        "usage": a_blocks / pool_blocks,
                        "per_block": [],
                        "free_order": [],
                    },
                    "requests": {},
                },
                "events": [],
                "explanation": [
                    (
                        "B 停在 waiting：需等 A 释放后空出 1 块才能容纳 B 的完整序列。"
                        if not admitted
                        else "B 已准入，与 A 并行（后续输出不足时可能被抢占）。"
                    ),
                    (
                        "结论：`reserve_full_isl=True` 在准入时就拒绝装不下的请求，"
                        "避免运行到一半因 KV 不足被抢占；代价是长请求要等更久。"
                    ),
                ],
                "meta": {},
            }
        )
        return steps

    reserve_true = _strip(True)
    reserve_false = _strip(False)
    # Combine: show True strip then False strip.
    all_steps = []
    for off, steps in ((True, reserve_true), (False, reserve_false)):
        for st in steps:
            st = dict(st)
            st["step"] = st["step"] + (0 if off else 2)
            all_steps.append(st)
    meta = {
        "id": "reserve_full_isl",
        "title": "整序列预留",
        "group": "基础循环",
        "color": "#0d9488",
        "special_view": "reserve_full_isl",
        "finished_reason": "all_done",
        "repeat_at": None,
        "num_steps": len(all_steps),
    }
    return None, all_steps, meta


# --------------------------------------------------------------------------
# Resumable / streaming input
# --------------------------------------------------------------------------
def _build_streaming():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=False,
        policy="fcfs",
    )
    req = make_request("A", prompt_len=4, max_tokens=4, prompt_token=1, resumable=True)
    reqs = [req]

    # Inject exactly one continuation chunk; when that second bundle finishes,
    # end the session (the stream is over). Without the guard, every pause
    # would be resumed and the request would grow forever. The resume is
    # delayed by one step so the paused (WAITING_FOR_STREAMING_REQ) state is
    # actually visible in a frame.
    resumed = {"done": False}
    paused_at: list[int | None] = [None]

    def after_update(sched_obj, so, model_output):
        req_a = sched_obj.requests.get("A")
        if req_a is None:
            return
        step = sched_obj.current_step
        if req_a.status.name == "WAITING_FOR_STREAMING_REQ":
            if paused_at[0] is None:
                # Just paused: leave it paused for one frame.
                paused_at[0] = step
                return
            # One step later: resume with the next chunk, or end the session
            # if the stream is over.
            if not resumed["done"]:
                resumed["done"] = True
                cont = make_request(
                    "A", prompt_len=3, max_tokens=2, prompt_token=9, resumable=True
                )
                sched_obj.add_request(cont)
                # Request.max_tokens is cached at construction; align it with
                # the continuation's max_tokens for the stop check.
                sched_obj.requests["A"].max_tokens = 2
            else:
                from vllm.v1.request import RequestStatus

                sched_obj.finish_requests("A", RequestStatus.FINISHED_STOPPED)
        else:
            paused_at[0] = None

    return sched, reqs, one_token_per_req(), None, after_update


def _annotate_streaming(sched, step, so):
    lines = []
    for rid, req in sched.requests.items():
        if req.status.name == "WAITING_FOR_STREAMING_REQ":
            lines.append(
                f"流式暂停：**{rid}** 已生成完当前束的输出（out=max_tokens），"
                "暂停进入 `WAITING_FOR_STREAMING_REQ`，"
                "等待 `streaming_queue` 送入下一段输入。"
            )
    # A resumed streaming session is re-scheduled from the waiting queue with
    # its output counter reset and its prompt extended by the new chunk
    # (num_prompt_tokens grows), so it looks like a fresh prefill+decode.
    for nr in so.scheduled_new_reqs:
        rid = nr.req_id
        req = sched.requests.get(rid)
        if req is None or not getattr(req, "resumable", False):
            continue
        if req.num_prompt_tokens > 4 and req.num_output_tokens == 0:
            lines.append(
                f"流式续接：**{rid}** 收到下一段输入（追加 prompt token），"
                "恢复为 WAITING 并重新调度，继续 prefill + decode。"
            )
    return lines


# --------------------------------------------------------------------------
# Mamba (block-aligned state caching)
# --------------------------------------------------------------------------
def _build_mamba():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        block_size=4,
        num_blocks=16,
        enable_chunked_prefill=True,
        policy="fcfs",
        mamba_cache_mode="align",
    )
    # A 12-token prompt with block_size=4: prefill chunks must stop exactly at
    # block boundaries so reusable SSM state is materialized (8 then 4).
    reqs = [
        make_request("A", prompt_len=12, max_tokens=6, prompt_token=1),
        make_request("B", prompt_len=10, max_tokens=6, prompt_token=2),
    ]
    return sched, reqs, one_token_per_req()


def _annotate_mamba(sched, step, so):
    lines = []
    for rid in so.num_scheduled_tokens:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens > 0:
            continue
        n = so.num_scheduled_tokens.get(rid, 0)
        bs = getattr(sched, "block_size", 4)
        # A chunk that ends on a block boundary materializes reusable SSM state;
        # the final chunk is the unavoidable remainder.
        aligned = req.num_computed_tokens % bs == 0
        lines.append(
            f"Mamba 块对齐：**{rid}** 的 prefill 块被裁到块边界（block_size={bs}），"
            f"本步计算 {n} 个 token"
            + ("" if aligned else "（剩余尾块，无法再对齐）")
            + "；块边界处物化可复用的 SSM 状态。"
        )
    return lines


# --------------------------------------------------------------------------
# Discrete diffusion (dLLM)
# --------------------------------------------------------------------------
def _build_diffusion():
    sched = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=32,
        max_model_len=32,
        block_size=4,
        num_blocks=20,
        enable_chunked_prefill=True,
        policy="fcfs",
        canvas_length=8,
    )
    reqs = [
        make_request("A", prompt_len=4, max_tokens=24, prompt_token=1),
    ]

    def after_update(sched_obj, so: SchedulerOutput, model_output):
        sched_obj.update_draft_token_ids(make_draft_token_ids(so, 8))

    return sched, reqs, diffusion(8), None, after_update


def _annotate_diffusion(sched, step, so):
    lines = []
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None or req.num_output_tokens == 0:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        if not spec:
            continue
        lines.append(
            f"扩散去噪：**{rid}** 本步一次性生成 {len(spec)} 个 canvas token"
            "（num_sampled_tokens_per_step=0，无逐 token 自回归），一次去噪推进一个块。"
        )
    return lines


def _explain_diffusion(sched, step, so):
    """Diffusion avoids the AR decode/prefill narrative: each decode step is a
    whole-canvas denoise, and computed tokens advance by the canvas length."""
    if so is None:
        return [
            "**第 0 帧（离散扩散(dLLM)前，step 0）**：请求 **A** 通过 `add_request()` "
            "加入调度器，尚未分配 KV 块。",
            "离散扩散用固定长度 canvas 迭代去噪生成，而非逐 token 自回归。",
            "点「下一步 ▶」执行第 1 次 `schedule()`：先做 prompt prefill。",
        ]
    lines = []
    for nr in so.scheduled_new_reqs:
        rid = nr.req_id
        req = sched.requests.get(rid)
        if req is None:
            continue
        lines.append(
            f"第 {step} 步：从 waiting 取出 **{rid}**（prompt {req.num_prompt_tokens} "
            "token）置为 RUNNING 做 prefill，为其分配 KV 块。"
        )
    for rid in so.scheduled_cached_reqs.req_ids:
        req = sched.requests.get(rid)
        if req is None:
            continue
        spec = so.scheduled_spec_decode_tokens.get(rid)
        if req.num_output_tokens == 0 and not spec:
            lines.append(f"**{rid}** 继续 prefill：本步再计算部分 prompt token。")
        elif spec:
            lines.append(
                f"**{rid}** 扩散去噪：canvas_length=8，本步对固定长度 canvas 做一次"
                "迭代去噪，一次性推进 8 个 token（无逐 token 自回归）。"
            )
    if not lines:
        lines.append(f"第 {step} 步：没有新调度的请求。")
    if so.finished_req_ids:
        lines.append(
            f"本步完成：{', '.join(sorted(so.finished_req_ids))}，"
            "从 running 移除并释放其 KV 块。"
        )
    return lines


# --------------------------------------------------------------------------
# Encoder-decoder (hand-built conceptual strip)
# --------------------------------------------------------------------------
def _build_encoder_decoder():
    """Encoder-decoder models first run the encoder over the (multimodal)
    input to produce embeddings, then the decoder autoregressively generates
    output. Driving a real encoder-decoder scheduler needs a full multimodal
    budget + encoder cache setup, so this is a hand-built strip mirroring the
    scheduler's encoder-input scheduling path (``_try_schedule_encoder_inputs``
    allocates encoder cache and decrements the encoder token budget)."""
    steps = []
    # encoder steps
    for i, n in enumerate((4, 4, 2), 1):
        computed = i * 4
        steps.append(
            {
                "step": i,
                "cpu": {
                    "requests": {},
                    "waiting": [],
                    "running": ["A"],
                    "skipped_waiting": [],
                    "encoder_decoder": {
                        "phase": "encoder",
                        "encoder_tokens": min(computed, 10),
                        "enc_total": 10,
                        "decoder_tokens": 0,
                        "dec_total": 0,
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
                    f"Encoder 阶段 step {i}：调度器把 encoder 输入（10 token）分块送入，"
                    f"本步已编码 {min(computed, 10)}/10。",
                    "encoder 输出（embedding）写入 encoder cache，不占 decoder 的 KV 块。",
                ],
                "meta": {},
            }
        )
    # decoder steps
    for i in range(1, 5):
        steps.append(
            {
                "step": i + 3,
                "cpu": {
                    "requests": {},
                    "waiting": [],
                    "running": ["A"],
                    "skipped_waiting": [],
                    "encoder_decoder": {
                        "phase": "decoder",
                        "encoder_tokens": 10,
                        "enc_total": 10,
                        "decoder_tokens": i,
                        "dec_total": 4,
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
                    f"Decoder 阶段：encoder 已全部编码（10/10），decoder 基于 embedding "
                    f"自回归生成，本步已产出 {i} 个输出 token。",
                    "encoder 输入只编码一次并缓存，后续 decoder 步直接复用，不重复编码。",
                ],
                "meta": {},
            }
        )
    meta = {
        "id": "encoder_decoder",
        "title": "Encoder-Decoder",
        "group": "基础循环",
        "color": "#059669",
        "special_view": "encoder_decoder",
        "finished_reason": "all_done",
        "repeat_at": None,
        "num_steps": len(steps),
    }
    return None, steps, meta


SCENARIOS = [
    make_scenario(
        id="encoder_decoder",
        title="Encoder-Decoder",
        group="基础循环",
        color="#059669",
        build=_build_encoder_decoder,
        special_view="encoder_decoder",
    ),
    make_scenario(
        id="long_prefill",
        title="Long-Prefill 阈值",
        group="基础循环",
        color="#16a34a",
        build=_build_long_prefill,
        annotate=_annotate_long_prefill,
    ),
    make_scenario(
        id="reserve_full_isl",
        title="整序列预留",
        group="基础循环",
        color="#0d9488",
        build=_build_reserve_full_isl,
        special_view="reserve_full_isl",
    ),
    make_scenario(
        id="streaming",
        title="流式输入",
        group="基础循环",
        color="#2563eb",
        build=_build_streaming,
        annotate=_annotate_streaming,
    ),
    make_scenario(
        id="mamba",
        title="Mamba 块对齐缓存",
        group="基础循环",
        color="#7c3aed",
        build=_build_mamba,
        annotate=_annotate_mamba,
    ),
    make_scenario(
        id="diffusion",
        title="离散扩散 (dLLM)",
        group="基础循环",
        color="#db2777",
        build=_build_diffusion,
        annotate=_annotate_diffusion,
        explain=_explain_diffusion,
        special_view="diffusion",
    ),
]
