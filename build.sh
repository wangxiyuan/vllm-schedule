#!/usr/bin/env bash
# 构建 kv_cache_layout.html（讲义模板 + 注入的 scheduler 主题 + 实时 vLLM 规划数据）。
#
# kv_layout.py 依赖能导入 vllm 的 Python（数据来自本地 vllm 源码树的规划代码），
# 普通 `python3` / `uv run` 环境里通常没有，所以这里自动探测：
#   1. 环境变量 VLLM_PYTHON=/path/to/python  显式指定解释器
#   2. 与仓库同级的 vllm 源码树（本仓库常见布局：~/code/{sched_visualize,vllm}）
#   3. 系统 python3（若恰好能导入 vllm）
#
# 用法：
#   ./build.sh                     # -> kv_cache_layout.html
#   ./build.sh -o out.html         # 其余参数原样透传给 kv_layout.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${VLLM_PYTHON:-}" ]]; then
  PY="$VLLM_PYTHON"
else
  VLLM_ROOT="${VLLM_ROOT:-$(dirname "$ROOT")/vllm}"
  if [[ -x "$VLLM_ROOT/.venv/bin/python" ]]; then
    PY="$VLLM_ROOT/.venv/bin/python"
    export PYTHONPATH="$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  else
    PY="$(command -v python3)"
  fi
fi

if ! "$PY" -c "import vllm" >/dev/null 2>&1; then
  echo "错误：$PY 无法导入 vllm。可用 VLLM_PYTHON=/path/to/python 指定（vllm 源码树内建好环境后指向其 .venv/bin/python）。" >&2
  exit 1
fi

exec "$PY" "$ROOT/kv_layout.py" "$@"
