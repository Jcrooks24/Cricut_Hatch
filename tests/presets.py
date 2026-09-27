"""
Named HatchConfig presets to compare across the test-image suite.

`baseline` is the current-best methodology: the **tonal** cross-hatch engine —
polygonize tonal regions, darkness (1-10) sets how many overlapping angled
layers a region gets, each layer stitched into ONE path via a hidden-travel
greedy tour that minimises pen lifts without visible connectors.

To try an idea, add a preset that clones `baseline` and changes a few knobs,
then run the suite on all presets, grade, and use compare.py to check it wins on
EVERY image before promoting.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Dict

# Ensure the repo root is importable even if presets is imported first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hatch_ui_nocairo import HatchConfig


# The tonal cross-hatch method is the baseline (10 darkness layers).
BASELINE = replace(HatchConfig(), tonal=True, tonal_max_layers=10)


PRESETS: Dict[str, HatchConfig] = {
    "baseline": BASELINE,

    # Add experiment presets here, e.g.:
    # "layers6": replace(BASELINE, tonal_max_layers=6),
    # "denser":  replace(BASELINE, tonal_spacing_px=3.5),
}


def get(name: str) -> HatchConfig:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Known: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]
