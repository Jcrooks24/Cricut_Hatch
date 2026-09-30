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

    # ── New adaptive features (from the ultracode implementation pass) ─────────
    # All default OFF in HatchConfig, so `baseline` is unchanged. These presets
    # turn them on so you can A/B them in the cockpit and decide what to promote.

    # RECOMMENDED upgrade: quantile/adaptive thresholds place the tonal layers
    # where each image's tonal mass actually is — dramatically fixes the low-key
    # over-fill (the near-solid dark portrait becomes a readable face) with no
    # regression on normal images. + dither-floor keeps true blacks solid.
    "adaptive": replace(
        BASELINE,
        tonal_adaptive_thresholds=True,
        tonal_dither_floor=0.14,
    ),

    # Crisp texture: spatially-varying min-area keeps detailed foreground sharp
    # while de-speckling flat dark grounds. NOTE: slow / heavy on very detailed
    # images (explodes polygon count) — opt-in per image, not a default.
    "texture": replace(
        BASELINE,
        tonal_adaptive_min_area=True,
        tonal_adaptive_thresholds=True,
        tonal_dither_floor=0.14,
    ),

    # Busy-scene taming: auto-scales min-area up / layers down / dither off when
    # the scene is complex; a no-op on normal images (verified identical output).
    "auto": replace(
        BASELINE,
        tonal_auto_complexity=True,
        tonal_adaptive_thresholds=True,
        tonal_dither_floor=0.14,
    ),

    # Single continuous stroke: ALL layers collapsed into ONE path with ONE pen
    # lift, ordered region-first (fills an area before moving on). Maximally
    # Cricut-friendly; connectors cross white where the path must travel — those
    # are removed later by the manual stitch-break tool. Pairs well with the
    # adaptive tonal fix.
    "single_path": replace(
        BASELINE,
        tonal_single_path=True,
        tonal_adaptive_thresholds=True,
        tonal_dither_floor=0.14,
    ),

    # METHODOLOGY 2 (experimental): darkness from variable line SPACING, not
    # overlapping layers. One family of parallel lines that bunch in shadows and
    # spread in highlights. ls_step = tightest spacing in the darkest areas.
    "line_spacing": replace(
        BASELINE,
        line_spacing=True,
        ls_step=5.0,
        ls_gamma=1.0,
        ls_angle=0.0,
    ),
}


def get(name: str) -> HatchConfig:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Known: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]
