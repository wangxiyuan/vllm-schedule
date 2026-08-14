#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vLLM KV-cache layout lecture — CLI entry point.

Builds a standalone, self-contained HTML page (``kv_cache_layout.html``)
explaining how vLLM arranges the KV cache in memory and how a logical token
position maps to a physical slot / tensor address. Driven by real vLLM
planning code (``get_kv_cache_groups`` + ``get_kv_cache_config_from_groups``)
and a real scheduler allocation for the slot-mapping demo. Independent of the
scheduler visualization page.

Usage:
    uv run python kv_layout.py                   # -> kv_cache_layout.html
    uv run python kv_layout.py -o out.html       # custom output path
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONHASHSEED", "0")

from layout.sections import build_page  # noqa: E402


def extract_theme_css(template_path: Path) -> str:
    """Extract the shared theme <style> block from the scheduler template so the
    lecture page shares the same visual language (single source of truth).

    The CSS is used verbatim: the template's ``/* ---------- banner */`` comments
    are harmless inside a <style> block, and any regex attempt to strip them can
    leave a dangling ``/*`` that swallows the rest of the stylesheet.
    """
    text = template_path.read_text(encoding="utf-8")
    m = re.search(r"<style>(.*?)</style>", text, re.S)
    if not m:
        return ""
    return m.group(1)


def render(page: dict, template_text: str, theme_css: str) -> str:
    data_json = json.dumps(page, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    out = template_text.replace("__THEME_CSS__", theme_css)
    return out.replace("__PAGE_DATA__", data_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", default="kv_cache_layout.html", help="Output HTML path."
    )
    args = parser.parse_args()

    print("  building page content ...", flush=True)
    page = build_page()
    num_chapters = len(page["chapters"])
    num_sections = sum(len(c["sections"]) for c in page["chapters"])

    template_path = ROOT / "layout" / "template.html"
    theme_css = extract_theme_css(ROOT / "template.html")
    html = render(page, template_path.read_text(encoding="utf-8"), theme_css)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(
        f"Wrote {out} ({len(html) / 1024:.1f} KB, {num_chapters} chapters, "
        f"{num_sections} sections)."
    )


if __name__ == "__main__":
    main()