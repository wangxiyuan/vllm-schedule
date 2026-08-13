"""SPDX-License-Identifier: Apache-2.0
Scenario registry: combine scenarios from all modules.
"""
from __future__ import annotations

from .advanced import SCENARIOS as ADVANCED
from .combo import SCENARIOS as COMBO
from .core import SCENARIOS as CORE
from .features import SCENARIOS as FEATURES
from .parallel import make_parallel_scenarios

# Order: 基础循环 (core + features + spec/structured + spec_structured), then
# 工程架构 (async + combos + parallel family). Groups render contiguously in
# the tab bar.
_CORE = CORE + FEATURES
_ADVANCED = [s for s in ADVANCED if s["group"] == "基础循环"]
_ARCH = [s for s in ADVANCED if s["group"] != "基础循环"]
_COMBO_CORE = [s for s in COMBO if s["group"] == "基础循环"]
_COMBO_ARCH = [s for s in COMBO if s["group"] != "基础循环"]

ALL_SCENARIOS = (
    _CORE + _ADVANCED + _COMBO_CORE + _ARCH + _COMBO_ARCH + make_parallel_scenarios()
)
SCENARIO_BY_ID = {s["id"]: s for s in ALL_SCENARIOS}