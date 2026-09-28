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


# The tonal cross-hatch method is the baseline: 10 darkness layers for depth.
#   band_gamma=1.0 : neutral preprocessing, so tonal_gamma is the SINGLE tone
#                    control (the default 1.2 was double-darkening before tonal).
#   tonal_gamma<1  : tone-reproduction curve — compensates for the non-linear
#                    darkening of overlapping passes so mid tones aren't too dark.
#   tonal_dither   : Bayer dither on the thresholds to break contour banding in
#                    smooth (esp. dark) regions into a stipple.
BASELINE = replace(
    HatchConfig(),
    tonal=True,
    tonal_max_layers=10,
    band_gamma=1.0,
    tonal_gamma=0.8,
    # --- fine-detail wins (from the ultracode methodology search) -------------
    tonal_blur_px=0.8,             # sharper region boundaries -> readable eyes/texture
    tonal_min_area_px2=40,         # keep features but merge dark-ground speckle
    local_contrast_radius=10,      # finer local contrast separates facial planes
    local_contrast_strength=0.8,
    stroke_width=0.5,              # finer lines keep dense texture legible
    # --- tonal-nuance wins ----------------------------------------------------
    tonal_angle_step=111.25,       # golden-ish angle spread -> less moire, smoother
    tonal_dither=0.6,              # anti-banding stipple (less needed w/ fine seg)
    # --- final trace pass (dark regions only) ---------------------------------
    tonal_trace=True,
    tonal_trace_layers=1,          # outline only the darkest layer (fewer rings)
    tonal_trace_min_area_px2=600.0,
)


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
