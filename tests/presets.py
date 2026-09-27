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

    # First improvement pass (config-only) targeting the confirmed baseline
    # failures on the portrait tests:
    #   - light backgrounds filled with scan-line streaks -> leave them white
    #     by lowering the highlight-protect threshold (baseline 0.92 protected 0%).
    #   - dark tones collapsing/muddy -> lift shadows with band_gamma < 1.
    #   - subject unreadable -> crisper local contrast + a stronger edge layer
    #     so silhouettes/features carry the drawing.
    "readable_v2": replace(
        BASELINE,
        highlight_protect_thr=0.80,   # leave light areas as clean paper
        band_gamma=0.90,              # lift shadows so darks separate, not blob
        local_contrast_radius=10.0,   # finer detail
        local_contrast_strength=0.75,
        base_spacing_px=8.0,          # slightly denser tone overall
        detail_pct=82.0,              # more of the edge map becomes ink
        detail_budget=1600,
        detail_spacing_mult=0.40,
        darkedge_pct=85.0,
        darkedge_budget=550,
        lightedge_pct=88.0,
        lightedge_budget=650,
    ),

    # Edge-forward line-art: vectorize HED edges into contour strokes that trace
    # the subject, plus light hatching in the darkest regions. This is the new
    # architecture meant to make subjects actually recognizable.
    "line_art": replace(
        BASELINE,
        line_art=True,
        line_edge_source="hed",
        line_hysteresis=True,      # clean connected contours (not a speckly web)
        line_edge_thr_hi=0.50,     # only strong salient contours (less webby)
        line_edge_thr_lo=0.20,
        line_despur_px=12.0,       # prune dead-end fuzz
        line_min_len_px=14.0,
        line_smooth_px=0.8,
        line_shade=True,           # graduated crosshatch shading
        line_shade_levels=3,       # max 3 overlapping passes -> dark, not solid
        line_shade_hi=0.58,
        line_shade_lo=0.10,
        line_shade_spacing_px=6.0, # a touch wider so 3 passes stays readable
        max_pen_lifts=1000,        # override live in the cockpit per your Cricut
    ),
}


def get(name: str) -> HatchConfig:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Known: {', '.join(sorted(PRESETS))}"
        )
    return PRESETS[name]
