#!/usr/bin/env python3
"""SPDX-License-Identifier: Apache-2.0
vLLM Scheduler visualization — CLI entry point.

Runs one (or all) scheduling scenarios against a real vLLM Scheduler /
AsyncScheduler (no GPU required), captures per-step frames (CPU python objects
+ simulated GPU tensors + KV lifecycle), and renders them into a single
self-contained HTML file.

Usage:
    uv run python tools/sched_visualize/sched_demo.py              # all scenarios
    uv run python tools/sched_visualize/sched_demo.py -s spec      # one scenario
    uv run python tools/sched_visualize/sched_demo.py -o out.html  # output path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the repo root is importable regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONHASHSEED", "0")

from tools.sched_visualize.scenarios import ALL_SCENARIOS, SCENARIO_BY_ID  # noqa: E402
from tools.sched_visualize.scenarios.base import run_scenario  # noqa: E402


def render(scenes: list[dict]) -> str:
    template_path = Path(__file__).resolve().parent / "template.html"
    template = template_path.read_text(encoding="utf-8")
    data_json = json.dumps(scenes, ensure_ascii=False)
    # JSON strings are not HTML-escaped; guard "</" so no explanation text can
    # terminate the embedded <script> and break the self-contained HTML.
    data_json = data_json.replace("</", "<\\/")
    return template.replace("__DATA__", data_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-s",
        "--scenario",
        help="Run only this scenario id (default: all).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="scheduler_visualization.html",
        help="Output HTML path (default: scheduler_visualization.html).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for sc in ALL_SCENARIOS:
            print(f"  {sc['id']:<16} [{sc['group']}] {sc['title']}")
        return

    ids = [args.scenario] if args.scenario else [sc["id"] for sc in ALL_SCENARIOS]
    scenes = []
    for sid in ids:
        sc = SCENARIO_BY_ID.get(sid)
        if sc is None:
            print(f"Unknown scenario: {sid}", file=sys.stderr)
            raise SystemExit(1)
        print(f"  running {sc['id']} ...", flush=True)
        frames, meta = run_scenario(sc)
        scenes.append({"meta": meta, "frames": frames})
        print(
            f"    done: {meta['num_steps']} steps, "
            f"finished_reason={meta['finished_reason']}",
            flush=True,
        )

    html = render(scenes)
    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(html) / 1024:.1f} KB, "
          f"{len(scenes)} scenario(s)).")


if __name__ == "__main__":
    main()