"""
Named HatchConfig presets to compare across the test-image suite.

WORKFLOW
    - "baseline" is the current-best methodology (the app's default HatchConfig).
    - To try an idea, add a new preset that starts from baseline and changes a
      few knobs. Give it a descriptive name.
    - Run the suite on ALL presets, grade the previews, then use compare.py to
      see whether the new preset beats baseline on EVERY test image (not just
      the one you were looking at). That is what stops regressions.

Add presets freely; keep them cheap to reason about (change one idea at a time).
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


# The current accepted methodology. Do not edit in place to test ideas —
# clone it into a new preset instead, so baseline stays a stable reference.
BASELINE = HatchConfig()


PRESETS: Dict[str, HatchConfig] = {
    "baseline": BASELINE,

    # ── Example variants (edit / delete / add your own) ──────────────────────

    # Flow lines along iso-brightness contours (engraving look).
    "contour_flow": replace(
        BASELINE,
        contour_hatch=True,
        contour_hatch_strength=1.0,
        contour_hatch_blur=8.0,
    ),

    # Polygon-free raster hatching — cumulative-threshold crosshatch.
    "raster": replace(
        BASELINE,
        raster_hatch=True,
        raster_passes=3,
    ),

    # Tighter path budget — stress-test Cricut friendliness.
    "lean_paths": replace(
        BASELINE,
        max_paths=3000,
        band_cap=400,
        micro_enabled=False,
    ),
}


def get(name: str) -> HatchConfig:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Known: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]
