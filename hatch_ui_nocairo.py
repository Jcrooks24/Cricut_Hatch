#!/usr/bin/env python3
"""
Image -> pen-and-ink hatched SVG (single-line strokes) for a Cricut pen plotter.

TONAL cross-hatch engine (run_tonal): polygonize tonal regions, map darkness to a
number of overlapping angled hatch layers, stitch each layer into one path with a
greedy nearest-neighbour tour, plus optional adaptive thresholds/min-area, dither,
a dark-region trace pass, and a single-continuous-path mode (1 pen lift) for the
manual stitch-break workflow. Output SVGs are coordinate-compressed for size.

This is a library; the interactive front-end is the batch cockpit
(tests/batch_ui.py) and the stitch-break tool (tests/break_tool.py).

Dependencies:
  py -3.11 -m pip install pillow numpy shapely svgwrite scikit-image scipy dearpygui
"""

import os
import sys
import math
import time
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    raise RuntimeError(
        "Missing dependency 'dearpygui'.\n"
        "Install with:  pip install dearpygui"
    ) from None

# Geometry
try:
    from shapely.geometry import Polygon, MultiPolygon, LineString, box
    from shapely.ops import unary_union
except Exception as e:
    raise RuntimeError("Missing dependency 'shapely'. Install with: pip install shapely") from e

# SVG writing
try:
    import svgwrite
except Exception as e:
    raise RuntimeError("Missing dependency 'svgwrite'. Install with: pip install svgwrite") from e

# scikit-image (recommended, optional)
_SKIMAGE_OK = True
try:
    from skimage import measure
except Exception:
    _SKIMAGE_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HatchConfig:
    # Output
    # tonal-engine config (legacy fields removed)

    # Output
    out_width_in: float  = 12.0

    out_height_in: float = 12.0

    # Preprocess
    resize_long_edge_px: int = 1800

    invert: bool             = False

    band_gamma: float        = 1.2

    # Local contrast enhancement (unsharp-mask style).
    # Smaller radius = captures finer detail (eyes behind glasses, fine texture).
    local_contrast: bool          = True

    local_contrast_radius: float  = 18.0

    local_contrast_strength: float = 0.55

    # Polygon cleanup
    simplify_tolerance_px: float = 0.25  # low value keeps inter-band boundaries tight (reduces seam gaps)

    # Cricut struggles with many pen lifts (each M = a lift). Cap the total;
    # least-efficient strokes (fewest drawn segments per lift) are dropped first.
    # 0 = disabled.
    max_pen_lifts: int         = 0

    # ── Tonal mode ────────────────────────────────────────────────────────────
    # Pure cross-hatch tonal rendering (no edge tracing). Darkness -> number of
    # overlapping angled layers; each layer is one stitched path (1 layer =
    # 1 path). Darker regions fall inside more layers -> more passes -> darker.
    tonal: bool                = False

    tonal_max_layers: int      = 10     # darkness rank 1-10 -> up to 10 crosshatch

                                        # layers; more layers = finer tone + depth
    tonal_hi: float            = 0.62   # lightest layer threshold (lighter = white)

    tonal_lo: float            = 0.08   # darkest layer threshold

    tonal_spacing_px: float    = 4.5

    tonal_base_angle: float    = 25.0

    tonal_angle_step: float    = 40.0   # angle offset per layer (step 6)

    tonal_blur_px: float       = 2.0

    tonal_min_area_px2: float  = 30.0

    # ── Spatially-varying min-area (adaptive) ─────────────────────────────────
    # When enabled, replace the single tonal_min_area_px2 filter with a per-polygon
    # threshold interpolated from a DETAIL map (local high-frequency energy of the
    # working tone). Detailed foreground (studs/laces/eyelets) keeps a low min-area
    # so fine texture survives; flat ground gets a high min-area so speckle is culled.
    tonal_adaptive_min_area: bool  = False   # OFF -> global tonal_min_area_px2 (unchanged)

    tonal_min_area_detail: float   = 8.0     # min-area where detail is HIGH (foreground)

    tonal_min_area_flat: float     = 90.0    # min-area where detail is LOW (flat ground)

    tonal_detail_blur_px: float    = 24.0    # blur radius for the high-freq detail map

    # Tone-curve before thresholding. <1 lifts mid/light tones so they fall into
    # fewer layers (lighter overall) while pure blacks stay dark — keeps 10-layer
    # depth without the whole image going too dark. 1.0 = linear (no change).
    tonal_gamma: float         = 1.0

    # Ordered (Bayer) dither amplitude, in units of one threshold step. Breaks up
    # the hard contour banding between adjacent pass-counts into a stipple.
    # 0 = off; ~1.0 = full step.
    tonal_dither: float        = 0.0

    # Tone-floor dither suppression. When >0, the dither amplitude is scaled per
    # pixel by a smooth ramp using the CLEAN (pre-dither) tone: 0 below this floor
    # (deep shadows stay solid) ramping to full by ~floor+0.08, so true blacks
    # lose their speckle while mid/light tones keep their anti-banding stipple.
    # 0.0 = off (dither applied uniformly, as before).
    tonal_dither_floor: float  = 0.0

    # ── Quantile / adaptive layer thresholds ──────────────────────────────────
    # By default the layer thresholds are linspace(tonal_hi, tonal_lo, layers).
    # For images whose tonal mass is clustered (near-all-black low-key portraits,
    # bimodal plastic) that wastes layers on empty tone ranges and dumps the whole
    # clustered mass below every threshold -> all layers -> near-solid. When
    # enabled, derive the thresholds from the QUANTILES of the post-gamma,
    # pre-dither tone over non-highlight pixels (tone < tonal_hi) at evenly spaced
    # quantile levels, so layers concentrate where the tonal mass actually lives.
    # OFF -> pure linspace (unchanged baseline).
    tonal_adaptive_thresholds: bool  = False

    # Blend factor between the quantile thresholds (1.0) and the linear ones (0.0).
    tonal_adaptive_blend: float      = 1.0

    # Final pass: outline the darkest regions for crisp definition (drawn on top
    # of the shading). Only dark regions are traced — outlining light regions
    # looks odd. One extra path.
    tonal_trace: bool          = False

    tonal_trace_layers: int    = 2       # outline the darkest N layers' regions

    tonal_trace_min_area_px2: float = 200.0  # only trace significant masses

    tonal_trace_holes: bool    = True    # also outline interior holes

    # ── Scene-complexity auto-scaling ─────────────────────────────────────────
    # Busy scenes (dense edges / texture, e.g. a crowded market) fragment into
    # thousands of tiny regions, exploding path/lift counts and cluttering the
    # result. When enabled, a complexity score (edge/gradient density of the
    # working tone, normalised 0..1) is computed ONCE at the start of run_tonal;
    # if it exceeds tonal_complexity_threshold the EFFECTIVE render params for
    # THIS render are scaled toward simpler output, PROPORTIONALLY to the score
    # (cfg itself is never mutated):
    #   * min-area filters raised (global + adaptive detail/flat) -> cull speckle
    #   * tonal_max_layers lowered toward tonal_complexity_min_layers
    #   * tonal_dither pushed toward 0
    # OFF -> no score computed, params untouched (baseline byte-identical).
    tonal_auto_complexity: bool       = False

    tonal_complexity_edge_thr: float  = 0.06   # |gradient| above this = a "detail" pixel

    tonal_complexity_threshold: float = 0.25   # score onset; below this -> no scaling

    tonal_complexity_span: float      = 0.12   # score range above onset over which

                                               # the scaling strength ramps 0 -> 1
    tonal_complexity_min_layers: int  = 6      # layer floor at full complexity

    tonal_complexity_area_mult: float = 3.0    # extra min-area gain at full complexity

    # ── Single-path mode ──────────────────────────────────────────────────────
    # Collapse ALL tonal layers (and the trace) into ONE continuous path drawn
    # with a single pen lift. Strokes are ordered region-first: the drawing is
    # partitioned into square cells (tonal_region_px), the cells swept in a
    # serpentine order, and within each cell a greedy nearest-neighbour tour
    # draws every stroke (all angles/layers) before moving on — so the pen
    # completes an area before travelling elsewhere. Connectors run pen-DOWN
    # across gaps as needed; the (future) manual stitch-break tool cuts the ones
    # that show. OFF -> normal 1-path-per-layer output (baseline byte-identical).
    tonal_single_path: bool    = False

    tonal_region_px: float     = 64.0   # spatial cell size for region-first ordering

    # Greedy nearest-neighbour stitching to cut pen lifts. Segments within
    # join_mult * spacing connect pen-DOWN (short travel line); farther jumps
    # lift the pen. Larger = fewer lifts but more visible travel moves.
    tonal_greedy_stitch: bool  = True

    tonal_join_mult: float     = 1.6    # used only when tonal_hide_travel is off

    # Hidden-travel stitching: keep the pen DOWN whenever a travel move stays
    # inside the hatched region (invisible), lift only when it would cross a
    # light area. Aggressively minimises pen lifts with no visible connectors.
    tonal_hide_travel: bool    = True

    # SVG stroke
    stroke_width: float    = 0.55

    stroke_linecap: str    = "round"

    stroke_linejoin: str   = "round"

    # SVG file-size compression. Coordinates are rounded to svg_decimals places,
    # trailing zeros stripped, and repeated 'L' commands dropped (implicit
    # lineto). 1 decimal = ~0.04mm precision at the working resolution — visually
    # identical for a plotter, ~35-45% smaller files. Set 2 to keep more digits.
    svg_decimals: int      = 1

    # ── Methodology 2: variable-spacing parallel lines (engraving style) ───────
    # Darkness comes from LINE SPACING, not overlapping layers: one family of
    # parallel lines whose perpendicular spacing shrinks in dark areas and widens
    # (or vanishes) in light ones. Implemented as iso-level contours of the
    # cumulative darkness integrated across the line direction — so lines stay
    # continuous across the image and bunch/spread smoothly with tone.
    line_spacing: bool      = False
    ls_step: float          = 5.0    # tightest spacing (px) in the DARKEST areas
    ls_gamma: float         = 1.0    # tone curve on darkness (>1 = more contrast)
    ls_angle: float         = 0.0    # line angle in degrees (0 = horizontal lines)
    ls_min_len: float       = 6.0    # drop fragments shorter than this
    ls_single_path: bool    = True   # collapse to one continuous stroke + sidecar
    # flowing mode drip control: clamp darkness before integrating so large black
    # areas don't pile into vertical streaks (1.0 = no clamp).
    ls_dark_cap: float      = 1.0
    # straight mode: truly parallel straight lines; darkness -> line density via
    # tonal bands drawn from one global grid (aligned spacing, no drift/drips).
    ls_straight: bool       = False
    ls_bands: int           = 6      # darkness bands
    ls_max_stride: int      = 9      # lightest band keeps every Nth global line


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────


def safe_simplify(poly: Polygon, tol: float) -> Polygon:
    try:
        p2 = poly.simplify(tol, preserve_topology=True)
        return poly if p2.is_empty else p2
    except Exception:
        return poly

def iter_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    out = []
    try:
        for g in geom.geoms:
            out.extend(iter_polygons(g))
    except Exception:
        pass
    return out


def load_and_preprocess_image(path: str, cfg: HatchConfig) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    w, h = im.size
    long_edge = max(w, h)
    if long_edge > cfg.resize_long_edge_px:
        scale = cfg.resize_long_edge_px / float(long_edge)
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    g = ImageOps.grayscale(im)

    if cfg.local_contrast and cfg.local_contrast_strength > 0.0:
        g_blur   = g.filter(ImageFilter.GaussianBlur(radius=max(1.0, cfg.local_contrast_radius)))
        arr_base = np.asarray(g).astype(np.float32) / 255.0
        arr_blur = np.asarray(g_blur).astype(np.float32) / 255.0
        arr = arr_base + float(cfg.local_contrast_strength) * (arr_base - arr_blur)
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.asarray(g).astype(np.float32) / 255.0

    if cfg.invert:
        arr = 1.0 - arr
    if cfg.band_gamma != 1.0:
        arr = np.clip(arr, 0, 1) ** float(cfg.band_gamma)

    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Mask → polygons
# ─────────────────────────────────────────────────────────────────────────────

def _contours_to_holed_polys(contours, min_area: float,
                              simplify_tol: float) -> List[Polygon]:
    """
    Convert skimage contours to Shapely polygons with correct holes.

    A glasses frame is a dark RING: find_contours returns two contours —
    the outer edge of the ring AND the inner edge (the lens opening).
    Previously, unary_union of both produced a solid filled disk, causing
    dense hatching across the entire lens.  Here we detect nesting depth:
      even depth (0, 2, …) = solid fill region
      odd  depth (1, 3, …) = hole (punched out of its parent)
    """
    # ── 1. Build raw Simple-Polygon list (iter_polys to avoid MultiPolygon) ──
    raw: List[Polygon] = []
    for c in contours:
        if len(c) < 6:
            continue
        pts = [(float(p[1]), float(p[0])) for p in c]
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            for p in iter_polygons(poly):       # handles MultiPolygon safely
                if not p.is_empty and p.area >= 4:
                    raw.append(p)
        except Exception:
            continue

    if not raw:
        return []

    raw.sort(key=lambda p: p.area, reverse=True)   # largest first
    n = len(raw)
    parent = [-1] * n   # index of tightest containing polygon

    # ── 2. Assign parents using bbox pre-filter + contains ───────────────────
    bounds_cache = [p.bounds for p in raw]  # (minx, miny, maxx, maxy)

    for i in range(n):
        try:
            cx = raw[i].centroid.x
            cy = raw[i].centroid.y
        except Exception:
            continue
        for j in range(i):          # raw[j].area >= raw[i].area
            bx1, by1, bx2, by2 = bounds_cache[j]
            if not (bx1 <= cx <= bx2 and by1 <= cy <= by2):
                continue            # bbox miss → skip expensive contains
            try:
                if raw[j].contains(raw[i].centroid):
                    if parent[i] == -1 or raw[parent[i]].area > raw[j].area:
                        parent[i] = j   # keep smallest (tightest) container
            except Exception:
                continue

    # ── 3. Nesting depth → solid vs hole ─────────────────────────────────────
    def nesting_level(i: int) -> int:
        lvl, k, seen = 0, parent[i], set()
        while k != -1 and k not in seen:
            lvl += 1
            seen.add(k)
            k = parent[k]
        return lvl

    is_hole = [nesting_level(i) % 2 == 1 for i in range(n)]

    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        if parent[i] != -1:
            children[parent[i]].append(i)

    # ── 4. Build holed polygons ───────────────────────────────────────────────
    result: List[Polygon] = []
    for i in range(n):
        if is_hole[i]:
            continue

        hole_coords: List[list] = []
        for child in children[i]:
            if is_hole[child]:
                try:
                    hole_coords.append(list(raw[child].exterior.coords))
                except Exception:
                    continue

        try:
            if hole_coords:
                poly = Polygon(list(raw[i].exterior.coords), hole_coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            else:
                poly = raw[i]
            for p in iter_polygons(poly):
                if p.area >= min_area:
                    result.append(safe_simplify(p, simplify_tol))
        except Exception:
            # Hole construction failed — fall back to the unfixed outer polygon
            if raw[i].area >= min_area:
                result.append(safe_simplify(raw[i], simplify_tol))

    return result


def mask_to_polygons(mask: np.ndarray, cfg: HatchConfig,
                     min_area: float, simplify_tol: float) -> List[Polygon]:
    polys: List[Polygon] = []
    if mask.sum() == 0:
        return polys

    if _SKIMAGE_OK:
        m = mask.astype(np.uint8)
        try:
            result = _contours_to_holed_polys(
                measure.find_contours(m, 0.5), min_area, simplify_tol
            )
            if result:
                return result
        except Exception:
            pass   # fall through to pixel-union fallback

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    cells = [box(x, y, x + 1, y + 1) for x, y in zip(xs, ys)]
    try:
        u   = unary_union(cells)
        out = []
        for p in iter_polygons(u):
            if p.area >= min_area:
                out.append(safe_simplify(p, simplify_tol))
        return out
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Hatching — 1 polygon = 1 SVG <path> element
# ─────────────────────────────────────────────────────────────────────────────

def make_parallel_lines(bounds: Tuple[float, float, float, float],
                        spacing: float, angle_deg: float,
                        phase: float = 0.0) -> List[LineString]:
    minx, miny, maxx, maxy = bounds
    angle = math.radians(angle_deg)
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx

    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    projs   = [cx * nx + cy * ny for (cx, cy) in corners]
    pmin, pmax = min(projs), max(projs)

    phase_off = float(phase) * spacing
    L  = math.hypot(maxx - minx, maxy - miny) * 3.0 + 10.0
    k0 = int(math.floor((pmin - phase_off) / spacing)) - 2
    k1 = int(math.ceil ((pmax - phase_off) / spacing)) + 2

    lines: List[LineString] = []
    for k in range(k0, k1 + 1):
        off = k * spacing + phase_off
        if abs(ny) > 1e-9:
            x0, y0 = 0.0, off / ny
        else:
            x0, y0 = off / nx, 0.0
        lines.append(LineString([
            (x0 - dx * L, y0 - dy * L),
            (x0 + dx * L, y0 + dy * L),
        ]))
    return lines

def clip_lines_to_polygon(lines: List[LineString], poly: Polygon) -> List[LineString]:
    segs: List[LineString] = []
    for ln in lines:
        inter = ln.intersection(poly)
        if inter.is_empty:
            continue
        if isinstance(inter, LineString):
            if len(inter.coords) >= 2 and inter.length > 0:
                segs.append(inter)
        else:
            try:
                for g in inter.geoms:
                    if isinstance(g, LineString) and len(g.coords) >= 2 and g.length > 0:
                        segs.append(g)
            except Exception:
                pass
    return segs

def _travel_hidden(x0: float, y0: float, x1: float, y1: float,
                   mask: np.ndarray, samples: int = 6) -> bool:
    """True if the INTERIOR of the segment (x0,y0)->(x1,y1) stays inside the dark
    mask — a pen-down travel there runs through hatched area and is invisible.
    Endpoints are skipped: they sit on the clipped region boundary and would
    false-negative."""
    h, w = mask.shape
    for t in np.linspace(0.15, 0.85, samples):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return False
    return True


def stitch_segs_greedy(segs: List[LineString], join_jump: float,
                       mask: Optional[np.ndarray] = None,
                       short_join: float = 0.0) -> str:
    """
    Stitch many hatch segments into ONE d-string via a greedy nearest-neighbour
    tour: always travel to the closest unused segment.

    Pen-down vs pen-up per jump:
      - jumps <= short_join are ALWAYS pen-down (normal adjacent-line turns; tiny
        and effectively invisible) — keeps the pen-lift count low.
      - longer jumps (up to join_jump) stay down only when a `mask` is given AND
        the travel stays inside the hatched region (invisible); otherwise lift.
        This lets join_jump be raised to connect same-region gaps through dark
        without ever drawing a connector across a white/light area.
      - without `mask`: pen-down for jumps <= join_jump, else lift.

    A NN tour finishes one region before jumping, so lifts drop sharply vs a
    perpendicular sort. Falls back to the boustrophedon stitcher if scipy is
    unavailable.
    """
    if not segs:
        return ""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return stitch_segs_to_d_string(segs, join_jump)

    n = len(segs)
    if n == 1:
        c = list(segs[0].coords)
        return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in c)

    # endpoint 0 and 1 of every segment; row 2*i and 2*i+1
    ends = np.array([[s.coords[0], s.coords[-1]] for s in segs],
                    dtype=float).reshape(2 * n, 2)
    tree = cKDTree(ends)
    used = np.zeros(n, dtype=bool)

    parts: List[str] = []
    cur = 0
    a, b = ends[2 * cur], ends[2 * cur + 1]
    parts.append(f"M {a[0]:.2f},{a[1]:.2f}")
    parts.append(f"L {b[0]:.2f},{b[1]:.2f}")
    used[cur] = True
    cur_end = b
    remaining = n - 1

    while remaining > 0:
        # Gather the nearest unused endpoints as candidates (each endpoint is a
        # possible entry side of its segment).
        cand = []
        k = 16
        while not cand:
            k = min(k, 2 * n)
            dists, idxs = tree.query(cur_end, k=k)
            for d, ei in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                seg = int(ei) // 2
                if not used[seg]:
                    cand.append((float(d), seg, int(ei) % 2))
            if not cand and k >= 2 * n:
                break
            if not cand:
                k *= 2
        if not cand:
            break
        cand.sort(key=lambda c: c[0])

        # Prefer the nearest candidate whose connector stays pen-DOWN. With a
        # mask, "pen-down" means the travel stays INSIDE the hatched region
        # (invisible) — the distance clause alone is NOT enough, or a short jump
        # would still draw a visible line across a white gap. This also picks the
        # entry SIDE of the next line that avoids a white crossing. Only if no
        # nearby candidate qualifies do we accept a lift (nearest).
        pick, pen_down = None, False
        for d, seg, which in cand:
            a = ends[2 * seg + which]
            if d <= short_join:
                ok = True                                    # normal short turn
            elif mask is not None:
                ok = _travel_hidden(cur_end[0], cur_end[1], a[0], a[1], mask)  # hidden, any distance
            else:
                ok = d <= join_jump
            if ok:
                pick, pen_down = (d, seg, which), True
                break
        if pick is None:
            pick = cand[0]

        _, seg, which = pick
        a = ends[2 * seg + which]
        b = ends[2 * seg + (1 - which)]
        parts.append(f"{'L' if pen_down else 'M'} {a[0]:.2f},{a[1]:.2f}")
        parts.append(f"L {b[0]:.2f},{b[1]:.2f}")
        used[seg] = True
        cur_end = b
        remaining -= 1

    return " ".join(parts)


def stitch_single_path(strokes: List[List[Tuple[float, float]]],
                       w: int, h: int, cell_px: float) -> str:
    """
    Collapse many strokes (each a list of (x,y) points) into ONE continuous
    d-string with a SINGLE pen lift (one leading 'M', everything else 'L').

    Region-first ordering: bucket strokes into square cells of ~cell_px, sweep
    the cells in a serpentine (boustrophedon) order, and inside each cell run a
    greedy nearest-neighbour tour over ALL its strokes (every layer/angle). The
    pen therefore completes one area of the drawing before travelling to the
    next, minimising long travel connectors. Every connector is drawn pen-DOWN;
    the manual stitch-break tool removes the ones that show later.
    """
    return strokes_to_single_d(order_strokes_region_first(strokes, w, h, cell_px))


def order_strokes_region_first(strokes, w, h, cell_px):
    """Order + orient strokes region-first: bucket into serpentine-swept square
    cells (cell_px), greedy nearest-neighbour tour within each cell, so the pen
    completes one area before travelling on. Returns the ordered, oriented list
    of strokes (each a list of (x, y)). This is the connector-tagging enabler:
    connector k is the gap ordered[k].end -> ordered[k+1].start."""
    strokes = [list(s) for s in strokes if s and len(s) >= 2]
    if not strokes:
        return []
    cell = max(8.0, float(cell_px))

    def d2(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, s in enumerate(strokes):
        cx = 0.5 * (s[0][0] + s[-1][0])
        cy = 0.5 * (s[0][1] + s[-1][1])
        buckets.setdefault((int(cy // cell), int(cx // cell)), []).append(idx)

    keys = sorted(buckets.keys())
    order: List[Tuple[int, int]] = []
    row_y = None
    row: List[Tuple[int, int]] = []
    for k in keys:
        if k[0] != row_y:
            if row:
                order.extend(row if (row_y % 2 == 0) else row[::-1])
            row_y, row = k[0], []
        row.append(k)
    if row:
        order.extend(row if (row_y % 2 == 0) else row[::-1])

    ordered: List[List[Tuple[float, float]]] = []
    prev = None
    for key in order:
        idxs = buckets[key]
        while idxs:
            if prev is None:
                j = idxs[0]
            else:
                j = min(idxs, key=lambda i: min(d2(prev, strokes[i][0]),
                                                d2(prev, strokes[i][-1])))
            idxs.remove(j)
            s = strokes[j]
            if prev is not None and d2(prev, s[-1]) < d2(prev, s[0]):
                s = s[::-1]                      # enter from the nearer end
            ordered.append(s)
            prev = s[-1]
    return ordered


def strokes_to_single_d(ordered, broken=None):
    """Build a d-string from ordered strokes. Each stroke joins the previous one
    pen-DOWN ('L' to its start) => one pen lift total. If `broken` is given (a
    set of connector indices, where connector k precedes stroke k+1), those
    joins become pen-UP ('M') instead — one extra pen lift each. This is how the
    manual stitch-break tool cuts connectors that show."""
    broken = broken or set()
    parts: List[str] = []
    for k, s in enumerate(ordered):
        lift = (k == 0) or ((k - 1) in broken)
        parts.append(("M " if lift else "L ") + f"{s[0][0]:.2f},{s[0][1]:.2f}")
        parts.extend(f"L {x:.2f},{y:.2f}" for x, y in s[1:])
    return " ".join(parts)


def stitch_segs_to_d_string(segs: List[LineString], stitch_jump: float) -> str:
    """
    Stitch hatch segments for ONE polygon into a single SVG path d-string.
    O(n log n) boustrophedon: sort by centre position, single pass choosing
    the nearest endpoint direction per segment.
    Gaps ≤ stitch_jump → L (pen down).  Gaps > stitch_jump → M (pen lift).
    """
    if not segs:
        return ""

    # Pre-extract coord arrays once (avoid repeated .coords access)
    pts_list: List[List[Tuple[float, float]]] = [
        [(float(x), float(y)) for x, y in s.coords] for s in segs
    ]

    # Estimate dominant line angle from segments, then sort by the
    # perpendicular component of each segment's centre.  For parallel
    # hatch lines at angle θ the perpendicular sort puts spatially
    # adjacent lines next to each other → boustrophedon gap ≈ spacing.
    if len(pts_list) >= 2:
        dxs = [p[-1][0] - p[0][0] for p in pts_list]
        dys = [p[-1][1] - p[0][1] for p in pts_list]
        # Mean angle in [0, π) via circular mean of doubled angles
        sin2 = sum(math.sin(2 * math.atan2(dy, dx)) for dx, dy in zip(dxs, dys))
        cos2 = sum(math.cos(2 * math.atan2(dy, dx)) for dx, dy in zip(dxs, dys))
        dom_angle = 0.5 * math.atan2(sin2, cos2)          # dominant direction
        perp_cos  = math.cos(dom_angle + math.pi / 2)
        perp_sin  = math.sin(dom_angle + math.pi / 2)
    else:
        perp_cos, perp_sin = 0.0, 1.0                     # fallback: sort by y

    def seg_key(pts: List[Tuple[float, float]]) -> float:
        cx = 0.5 * (pts[0][0] + pts[-1][0])
        cy = 0.5 * (pts[0][1] + pts[-1][1])
        return cx * perp_cos + cy * perp_sin

    pts_list.sort(key=seg_key)

    parts: List[str] = []
    first = True
    prev_end: Tuple[float, float] = pts_list[0][0]

    for pts in pts_list:
        p0, p1 = pts[0], pts[-1]
        d0 = math.hypot(prev_end[0] - p0[0], prev_end[1] - p0[1])
        d1 = math.hypot(prev_end[0] - p1[0], prev_end[1] - p1[1])

        # Pick the closer endpoint; flip if tail is closer
        if d1 < d0:
            pts = list(reversed(pts))
            gap = d1
        else:
            gap = d0

        if first or gap > stitch_jump:
            parts.append(f"M {pts[0][0]:.2f},{pts[0][1]:.2f}")
            first = False
        else:
            parts.append(f"L {pts[0][0]:.2f},{pts[0][1]:.2f}")
        for x, y in pts[1:]:
            parts.append(f"L {x:.2f},{y:.2f}")
        prev_end = pts[-1]

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SVG writing
# ─────────────────────────────────────────────────────────────────────────────

def _svg_physical_size(img_w: int, img_h: int, cfg: HatchConfig) -> Tuple[float, float]:
    """
    Compute SVG width/height in inches so the physical aspect ratio matches
    the image pixel aspect ratio.  This prevents SVG viewers from letterboxing
    the content and centering it (which can make layers look misaligned).
    """
    aspect = img_w / max(img_h, 1)
    if aspect >= 1.0:           # landscape or square
        w_in = cfg.out_width_in
        h_in = cfg.out_width_in / aspect
    else:                       # portrait
        h_in = cfg.out_height_in
        w_in = cfg.out_height_in * aspect
    return w_in, h_in


def _compress_d(d: str, decimals: int) -> str:
    """
    Shrink a path 'd' string losslessly-enough for a plotter:
      * round coordinates to `decimals` places and strip trailing zeros
        (123.45 -> 123.5 at 1dp; 0.50 -> 0.5; 123.00 -> 123)
      * emit implicit lineto: a run of 'L' commands becomes 'L x,y x,y x,y'
    Geometry is preserved to sub-pixel precision; output stays valid SVG.
    """
    if not d:
        return d
    toks = d.split()
    out: List[str] = []
    last_cmd = None
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t in ("M", "L") and i + 1 < n:
            xs, ys = toks[i + 1].split(",")
            x = round(float(xs), decimals)
            y = round(float(ys), decimals)
            coord = f"{x:g},{y:g}"
            if t == "L" and last_cmd == "L":
                out.append(coord)                 # implicit lineto
            else:
                out.append(t + " " + coord)
            last_cmd = t
            i += 2
        else:
            out.append(t)
            last_cmd = None
            i += 1
    return " ".join(out)


def write_svg(out_path: str, d_strings: List[str],
              img_w: int, img_h: int, cfg: HatchConfig,
              status_cb=None) -> None:
    w_in, h_in = _svg_physical_size(img_w, img_h, cfg)
    dwg = svgwrite.Drawing(
        out_path,
        size=(f"{w_in:.4f}in", f"{h_in:.4f}in"),
        viewBox=f"0 0 {img_w} {img_h}",
        profile="tiny",
    )
    decimals = int(getattr(cfg, "svg_decimals", 1))
    for idx, d in enumerate(d_strings):
        if status_cb and idx % 250 == 0:
            status_cb(f"Writing SVG: {idx}/{len(d_strings)}")
        if not d:
            continue
        dwg.add(dwg.path(
            d=_compress_d(d, decimals), fill="none", stroke="black",
            stroke_width=cfg.stroke_width,
            stroke_linecap=cfg.stroke_linecap,
            stroke_linejoin=cfg.stroke_linejoin,
        ))
    dwg.save()


# ─────────────────────────────────────────────────────────────────────────────
# Preview renderer (PIL — no Inkscape dependency)
# ─────────────────────────────────────────────────────────────────────────────

def render_preview(d_strings: List[str], img_w: int, img_h: int,
                   max_dim: int = 1000) -> Image.Image:
    """
    Parse SVG path d-strings and draw them to a PIL Image for in-app preview.
    No external tools required — uses the same coordinate data from the pipeline.
    """
    scale = min(max_dim / max(img_w, 1), max_dim / max(img_h, 1))
    pw = max(1, int(img_w * scale))
    ph = max(1, int(img_h * scale))

    im   = Image.new("RGB", (pw, ph), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    for d in d_strings:
        tokens  = d.split()
        current = None
        cmd     = None            # current command; supports implicit lineto
        i       = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("M", "L"):
                cmd = tok
                i += 1
                continue
            # a coordinate token, applied under the active command
            try:
                x, y = map(float, tok.split(","))
            except ValueError:
                i += 1
                continue
            pt = (x * scale, y * scale)
            if cmd == "L" and current is not None:
                draw.line([current, pt], fill=(15, 15, 15), width=1)
            current = pt
            if cmd == "M":
                cmd = "L"         # subsequent bare coords after M are lineto
            i += 1

    return im


# ─────────────────────────────────────────────────────────────────────────────
# Tonal mode — cross-hatch by darkness, one stitched path per angle-layer
# ─────────────────────────────────────────────────────────────────────────────

def _bayer_matrix(n: int) -> np.ndarray:
    """n x n ordered-dither matrix normalised to [0,1) (n a power of two)."""
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / (m.shape[0] * m.shape[1])


def run_tonal(png_path: str, arr: np.ndarray, cfg: HatchConfig,
              out_svg_path: str, t0: float, status_cb=None, log=None) -> Dict:
    """
    Steps: 1) polygonize tonal regions  2) simplify  3) darkness -> layer count
    4) each darker region falls inside more layers  5) stitch each layer into
    ONE path  6) offset the hatch angle per layer so passes don't overlap.
    """
    h, w = arr.shape
    bounds = (0.0, 0.0, float(w), float(h))

    # ── Scene-complexity auto-scaling (self-contained, additive) ──────────────
    # Compute a complexity score for the working tone and, if the scene is busy,
    # scale the EFFECTIVE render params for THIS render only by rebinding the
    # local `cfg` to a scaled copy. All downstream blocks (adaptive-min-area,
    # dither, layer count, global min-area) then transparently read the scaled
    # values. `cfg` is a @dataclass; replace() returns a NEW instance, so the
    # caller's cfg is never mutated. When the flag is OFF this block is skipped
    # entirely and cfg is untouched -> baseline byte-identical.
    if cfg.tonal_auto_complexity:
        from dataclasses import replace as _dc_replace
        # Complexity = edge/gradient density: the fraction of pixels whose local
        # gradient magnitude exceeds tonal_complexity_edge_thr. High for busy,
        # textured scenes (a crowded market); ~0 for smooth low-contrast
        # portraits. Already a fraction, so inherently normalised to 0..1.
        _gy, _gx = np.gradient(arr.astype(np.float32))
        _gmag = np.sqrt(_gx * _gx + _gy * _gy)
        _score = float(np.mean(_gmag > float(cfg.tonal_complexity_edge_thr)))
        _score = max(0.0, min(1.0, _score))
        _thr = float(cfg.tonal_complexity_threshold)
        if _score > _thr:
            # Proportional strength s: 0 at the onset threshold, ramping to 1
            # across tonal_complexity_span of score above it (then saturating).
            _span = max(1e-6, float(cfg.tonal_complexity_span))
            _s = max(0.0, min(1.0, (_score - _thr) / _span))
            _area_mult   = 1.0 + _s * float(cfg.tonal_complexity_area_mult)
            _min_layers  = int(cfg.tonal_complexity_min_layers)
            _base_layers = int(cfg.tonal_max_layers)
            _eff_layers  = int(round(_base_layers - _s * (_base_layers - _min_layers)))
            _eff_layers  = max(_min_layers, min(_base_layers, _eff_layers))
            _base_dither = float(cfg.tonal_dither)
            _eff_dither  = _base_dither * (1.0 - _s)
            cfg = _dc_replace(
                cfg,
                tonal_min_area_px2    = float(cfg.tonal_min_area_px2)    * _area_mult,
                tonal_min_area_detail = float(cfg.tonal_min_area_detail) * _area_mult,
                tonal_min_area_flat   = float(cfg.tonal_min_area_flat)   * _area_mult,
                tonal_max_layers      = _eff_layers,
                tonal_dither          = _eff_dither,
            )
            if log:
                log(f"Auto-complexity: score={_score:.3f} > thr={_thr:.2f} "
                    f"(s={_s:.2f}) -> min_area x{_area_mult:.2f}, "
                    f"layers {_base_layers}->{_eff_layers}, "
                    f"dither {_base_dither:.2f}->{_eff_dither:.2f}")
        else:
            if log:
                log(f"Auto-complexity: score={_score:.3f} <= thr={_thr:.2f} "
                    f"-> no scaling (params unchanged)")
    # ---------------------------------------------------------------------------

    arr_s = arr
    if cfg.tonal_blur_px > 0:
        arr_s = np.asarray(
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(float(cfg.tonal_blur_px)))
        ).astype(np.float32) / 255.0

    # Tone curve: gamma < 1 lifts mid/light tones (fewer layers -> lighter),
    # pure black stays black. Applied to the darkness used for thresholding.
    if cfg.tonal_gamma != 1.0:
        arr_s = np.clip(arr_s, 0.0, 1.0) ** float(cfg.tonal_gamma)

    layers = max(1, int(cfg.tonal_max_layers))

    # Clean (pre-dither) tone for the trace pass, so outlines aren't speckled.
    arr_clean = arr_s.copy()

    # ── Adaptive min-area: detail map (computed ONCE) ─────────────────────────
    # DETAIL = local high-frequency energy of the working tone, i.e. how far the
    # tone deviates from its own gaussian-blurred (low-frequency) version. High
    # in textured foreground (sole studs, laces, eyelets), ~0 on flat ground.
    # Normalised 0..1. Used later to pick a per-polygon min-area threshold.
    detail_map = None
    if cfg.tonal_adaptive_min_area:
        lo_freq = np.asarray(
            Image.fromarray((np.clip(arr_clean, 0, 1) * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(float(cfg.tonal_detail_blur_px)))
        ).astype(np.float32) / 255.0
        hf = np.abs(arr_clean - lo_freq)
        mx = float(hf.max())
        detail_map = (hf / mx) if mx > 1e-6 else np.zeros_like(hf)
    # ---------------------------------------------------------------------------

    # Ordered dither: nudge each pixel's tone by up to +/- half a threshold step
    # using a tiled Bayer matrix, so the hard boundary between k and k+1 passes
    # becomes a stippled transition instead of a visible contour band.
    if cfg.tonal_dither > 0 and layers > 1:
        step = (float(cfg.tonal_hi) - float(cfg.tonal_lo)) / (layers - 1)
        bayer = _bayer_matrix(8) - 0.5                       # -0.5..+0.5, deterministic
        th, tw = arr_s.shape
        tile = np.tile(bayer, (th // 8 + 1, tw // 8 + 1))[:th, :tw]
        dither = tile * step * float(cfg.tonal_dither)
        # --- tone-floor dither suppression -------------------------------------
        # Gate the dither by a smooth ramp on the CLEAN (pre-dither) tone so deep
        # shadows stay solid (no speckle) while mid/light tones keep the stipple.
        # amp = 0 for clean tone < floor, ramps smoothly to 1 by floor+0.08.
        floor = float(cfg.tonal_dither_floor)
        if floor > 0.0:
            width = 0.08
            t = np.clip((arr_clean - floor) / width, 0.0, 1.0)
            amp = t * t * (3.0 - 2.0 * t)   # smoothstep 0->1
            dither = dither * amp
        # -----------------------------------------------------------------------
        arr_s = np.clip(arr_s + dither, 0.0, 1.0)
    # Nested thresholds: a pixel darker than ths[i] receives layer i. Darker
    # pixels pass more thresholds -> more layers -> darker (steps 3-4).
    ths = np.linspace(float(cfg.tonal_hi), float(cfg.tonal_lo), layers)
    # ── Quantile / adaptive layer thresholds ──────────────────────────────────
    # Self-contained, additive: when enabled, replace (or blend into) the linear
    # thresholds above with quantiles of the working tone so layers cluster where
    # the tonal mass is, instead of being evenly spread over empty tone ranges.
    #  * sample = post-gamma, pre-dither tone (arr_clean) over NON-highlight pixels
    #    (tone < tonal_hi), so highlights don't skew the shadow/mid distribution.
    #  * quantile levels are evenly spaced and descend hi->lo to mirror linspace
    #    (ths[0] = lightest threshold, ths[-1] = darkest).
    #  * blend with the linear thresholds by tonal_adaptive_blend (1.0 = pure
    #    quantile, 0.0 = pure linear), then clamp into [tonal_lo, tonal_hi].
    #  * CRITICAL: enforce strict monotonic separation (a tiny epsilon between
    #    adjacent thresholds) so collapsed/duplicate quantiles (common when the
    #    mass is heavily clustered) cannot create empty or inverted layers — this
    #    was a known past bug.
    if cfg.tonal_adaptive_thresholds and layers > 1:
        hi = float(cfg.tonal_hi); lo = float(cfg.tonal_lo)
        sample = arr_clean[arr_clean < hi]
        if sample.size > 0:
            qlevels = np.linspace(1.0, 0.0, layers)      # descending -> hi..lo order
            q_ths = np.quantile(sample, qlevels)
            blend = float(cfg.tonal_adaptive_blend)
            ths = blend * q_ths + (1.0 - blend) * ths
            ths = np.clip(ths, lo, hi)
            eps = 1e-4                                   # strict separation gap
            for _i in range(1, len(ths)):
                if ths[_i] >= ths[_i - 1] - eps:
                    ths[_i] = ths[_i - 1] - eps
    # ---------------------------------------------------------------------------
    spacing = max(1.0, float(cfg.tonal_spacing_px))

    all_d: List[str] = []
    single_strokes: List[List[Tuple[float, float]]] = []  # for single-path mode
    for i, th in enumerate(ths):
        if status_cb: status_cb(f"Tonal layer {i+1}/{layers}…")
        mask = arr_s < float(th)
        # close border-touching regions so dark backgrounds polygonize
        mask[0, :] = mask[-1, :] = False
        mask[:, 0] = mask[:, -1] = False
        if mask.sum() == 0:
            continue
        # ── Adaptive min-area filtering ──────────────────────────────────────
        # When enabled, polygonize with a SMALL min-area (drop nothing early),
        # then keep each polygon whose area >= a threshold interpolated from the
        # detail map at its centroid: high detail -> tonal_min_area_detail,
        # flat -> tonal_min_area_flat. Otherwise fall back to the global filter.
        if cfg.tonal_adaptive_min_area and detail_map is not None:
            a_detail = float(cfg.tonal_min_area_detail)
            a_flat   = float(cfg.tonal_min_area_flat)
            small    = max(0.0, min(a_detail, a_flat))   # never pre-drop a keepable poly
            raw = mask_to_polygons(mask, cfg,
                                   min_area=small,
                                   simplify_tol=cfg.simplify_tolerance_px)
            polys = []
            for poly in raw:
                c  = poly.centroid
                cx = int(min(max(c.x, 0.0), w - 1))
                cy = int(min(max(c.y, 0.0), h - 1))
                d  = float(detail_map[cy, cx])           # 0..1, high => detailed
                thr = a_flat + d * (a_detail - a_flat)   # lerp flat->detail
                if poly.area >= thr:
                    polys.append(poly)
        else:
            polys = mask_to_polygons(mask, cfg,                   # steps 1-2
                                     min_area=float(cfg.tonal_min_area_px2),
                                     simplify_tol=cfg.simplify_tolerance_px)
        # ---------------------------------------------------------------------
        if not polys:
            continue
        angle = float(cfg.tonal_base_angle) + float(cfg.tonal_angle_step) * i  # step 6
        lines = make_parallel_lines(bounds, spacing, angle, phase=0.0)
        segs: List[LineString] = []
        for poly in polys:
            segs.extend(clip_lines_to_polygon(lines, poly))
        if not segs:
            continue
        if cfg.tonal_single_path:
            # Collect this layer's strokes; they'll be woven into ONE path later.
            single_strokes.extend(list(s.coords) for s in segs)
            if log: log(f"Tonal layer {i+1}: tone<{th:.2f} ang={angle:.0f} "
                        f"polys={len(polys)} segs={len(segs)} -> collected")
            continue
        # step 5: 1 layer = 1 path, greedy NN tour to minimise pen lifts
        if cfg.tonal_greedy_stitch:
            d = stitch_segs_greedy(
                segs, spacing * float(cfg.tonal_join_mult),
                mask=mask if cfg.tonal_hide_travel else None,
                short_join=spacing * 1.6)
        else:
            d = stitch_segs_to_d_string(segs, spacing * 1.4)
        if d:
            all_d.append(d)
            if log: log(f"Tonal layer {i+1}: tone<{th:.2f} ang={angle:.0f} "
                        f"polys={len(polys)} segs={len(segs)} -> 1 path")

    # Final pass: outline the darkest N layers' regions (crisp shadow definition).
    # Only dark regions — outlining light regions looks odd. One extra path.
    if cfg.tonal_trace and layers > 0:
        if status_cb: status_cb("Tonal: tracing dark regions…")
        rings: List[List] = []
        n_trace = max(1, int(cfg.tonal_trace_layers))
        for i in range(max(0, layers - n_trace), layers):
            m = arr_clean < float(ths[i])          # clean tone -> smooth outlines
            m[0, :] = m[-1, :] = False
            m[:, 0] = m[:, -1] = False
            if m.sum() == 0:
                continue
            for p in mask_to_polygons(m, cfg,
                                      min_area=float(cfg.tonal_trace_min_area_px2),
                                      simplify_tol=cfg.simplify_tolerance_px):
                rings.append(list(p.exterior.coords))
                if cfg.tonal_trace_holes:
                    for r in p.interiors:
                        rings.append(list(r.coords))
        if rings and cfg.tonal_single_path:
            single_strokes.extend(r for r in rings if len(r) >= 2)
            if log: log(f"Trace pass: {len(rings)} outlines -> collected")
        elif rings:
            band = spacing * 4.0
            rings.sort(key=lambda r: (int(r[0][1] // band),
                                      r[0][0] if int(r[0][1] // band) % 2 == 0 else -r[0][0]))
            parts: List[str] = []
            for r in rings:
                if len(r) < 2:
                    continue
                parts.append(f"M {r[0][0]:.2f},{r[0][1]:.2f}")
                parts.extend(f"L {x:.2f},{y:.2f}" for x, y in r[1:])
            if parts:
                all_d.append(" ".join(parts))
                if log: log(f"Trace pass: {len(rings)} dark-region outlines -> 1 path")

    # Single-path mode: weave every collected stroke into ONE continuous path,
    # and write a sidecar JSON of the ordered strokes so the manual stitch-break
    # tool can identify/cut the connectors between them.
    if cfg.tonal_single_path:
        if status_cb: status_cb("Tonal: weaving single continuous path…")
        ordered = order_strokes_region_first(
            single_strokes, w, h, float(cfg.tonal_region_px))
        d = strokes_to_single_d(ordered)
        all_d = [d] if d else []
        try:
            import json as _json
            side = os.path.splitext(out_svg_path)[0] + "_strokes.json"
            with open(side, "w") as _f:
                _json.dump({"width": w, "height": h,
                            "strokes": [[[round(px, 1), round(py, 1)]
                                         for px, py in s] for s in ordered]}, _f)
        except Exception as _e:
            if log: log(f"strokes sidecar error: {_e}")
        if log: log(f"Single-path: {len(single_strokes)} strokes -> 1 path "
                    f"(+ _strokes.json sidecar)")

    if not all_d:
        all_d = ["M 0,0 L 0,0"]

    if status_cb: status_cb("Writing SVG…")
    write_svg(out_svg_path, all_d, w, h, cfg, status_cb=status_cb)

    preview_img = None
    try:
        preview_img = render_preview(all_d, w, h)
        preview_img.save(os.path.splitext(out_svg_path)[0] + "_preview.png")
    except Exception as _e:
        if log: log(f"Preview error: {_e}")

    return {
        "paths":         len(all_d),
        "pen_lifts_est": sum(d.count("M ") for d in all_d),
        "elapsed_sec":   time.time() - t0,
        "work_w":        w, "work_h": h,
        "aux_used":      len(all_d),
        "line_strokes":  0,
        "preview_image": preview_img,
        "band_results":  [], "band_caps": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


_PREVIEW_W = 700
_PREVIEW_H = 560

# Flat RGBA float32 list for a dark-gray placeholder texture
_PLACEHOLDER_DATA = [0.14, 0.14, 0.14, 1.0] * (_PREVIEW_W * _PREVIEW_H)


# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Methodology 2: variable-spacing parallel lines (engraving style)
# ─────────────────────────────────────────────────────────────────────────────

def run_linespacing(png_path: str, arr: np.ndarray, cfg: HatchConfig,
                    out_svg_path: str, t0: float, status_cb=None, log=None) -> Dict:
    """
    Darkness via LINE SPACING instead of overlapping layers. Integrate darkness
    across the line direction into a cumulative field D; the iso-level contours of
    D (every `ls_step`) are the lines. Where it's dark, D climbs fast so contours
    bunch (tight spacing = min ~ls_step px); where light, D is flat so they spread
    or vanish. Contours are continuous across the image -> smooth engraving lines.
    """
    import math
    h, w = arr.shape
    angle = float(cfg.ls_angle) % 180.0
    dark = np.clip(1.0 - arr, 0.0, 1.0) ** float(cfg.ls_gamma)
    step = max(0.5, float(cfg.ls_step))
    simp = float(cfg.simplify_tolerance_px)
    min_len = float(cfg.ls_min_len)
    strokes: List[List[Tuple[float, float]]] = []

    if cfg.ls_straight:
        # STRAIGHT parallel lines: one global grid at `step`; each darkness band
        # keeps every Nth line (dense in shadows, sparse in highlights). Lines are
        # drawn from the SAME grid so spacing is always aligned (no drift/drips).
        if status_cb: status_cb("Line-spacing: straight bands…")
        bounds = (0.0, 0.0, float(w), float(h))
        grid = make_parallel_lines(bounds, step, angle, phase=0.0)
        bands = max(1, int(cfg.ls_bands))
        edges = np.linspace(0.12, 0.95, bands + 1)
        for bi in range(bands):
            d_lo = float(edges[bi])
            d_hi = float(edges[bi + 1]) if bi < bands - 1 else 1.01
            mask = (dark >= d_lo) & (dark < d_hi)
            mask[0, :] = mask[-1, :] = False
            mask[:, 0] = mask[:, -1] = False
            if mask.sum() == 0:
                continue
            polys = mask_to_polygons(mask, cfg, min_area=float(cfg.tonal_min_area_px2),
                                     simplify_tol=simp)
            if not polys:
                continue
            d_mid = 0.5 * (d_lo + d_hi)
            stride = max(1, int(round(cfg.ls_max_stride * (1.0 - d_mid))))
            for ln in grid[::stride]:
                for poly in polys:
                    for seg in clip_lines_to_polygon([ln], poly):
                        cs = list(seg.coords)
                        if len(cs) >= 2:
                            strokes.append(cs)
        if log: log(f"Line-spacing STRAIGHT: {bands} bands, {len(grid)} grid lines "
                    f"-> {len(strokes)} segments (step={step}, angle={angle:.0f})")
    else:
        # FLOWING contours of the cumulative darkness field.
        if float(cfg.ls_dark_cap) < 1.0:
            dark = np.minimum(dark, float(cfg.ls_dark_cap))   # limit drip in black areas
        if abs(angle) < 1e-6:
            field = dark
        else:
            if status_cb: status_cb("Line-spacing: rotating field…")
            from scipy.ndimage import rotate as _rot
            field = _rot(dark, angle, reshape=True, order=1, mode="constant", cval=0.0)
        Hf, Wf = field.shape
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        in_c  = ((h - 1) / 2.0, (w - 1) / 2.0)
        out_c = ((Hf - 1) / 2.0, (Wf - 1) / 2.0)
        R = ((ca, sa), (-sa, ca))
        off0 = in_c[0] - (R[0][0] * out_c[0] + R[0][1] * out_c[1])
        off1 = in_c[1] - (R[1][0] * out_c[0] + R[1][1] * out_c[1])

        def to_image(rr, cc):                  # field (row,col) -> image (x,y)
            return (R[1][0] * rr + R[1][1] * cc + off1,
                    R[0][0] * rr + R[0][1] * cc + off0)

        if status_cb: status_cb("Line-spacing: building contours…")
        D = np.cumsum(field, axis=0)
        levels = np.arange(step, float(D.max()) + step, step)
        if _SKIMAGE_OK:
            for lvl in levels:
                for c in measure.find_contours(D, float(lvl)):
                    if len(c) < 2:
                        continue
                    pl = [to_image(float(rr), float(cc)) for rr, cc in c]
                    ls = LineString(pl)
                    if simp > 0:
                        ls = ls.simplify(simp, preserve_topology=False)
                    if ls.length >= min_len and len(ls.coords) >= 2:
                        strokes.append(list(ls.coords))
        if log: log(f"Line-spacing FLOWING: {len(levels)} levels -> {len(strokes)} "
                    f"strokes (step={step}, angle={angle:.0f}, cap={cfg.ls_dark_cap})")

    if cfg.ls_straight:
        # Straight segments: boustrophedon with pen-UPS across gaps — adjacent
        # parallel lines connect pen-down, everything else lifts. No long
        # cross-connectors (those only make sense for continuous flowing strokes).
        segs = [LineString(s) for s in strokes if len(s) >= 2]
        d = stitch_segs_greedy(segs, step * 1.5, mask=None, short_join=step * 1.5)
        all_d = [d] if d else []
    elif cfg.ls_single_path:
        # Flowing: one continuous path (+ break-tool sidecar).
        ordered = order_strokes_region_first(strokes, w, h, float(cfg.tonal_region_px))
        d = strokes_to_single_d(ordered)
        all_d = [d] if d else []
        try:
            import json as _json
            with open(os.path.splitext(out_svg_path)[0] + "_strokes.json", "w") as _f:
                _json.dump({"width": w, "height": h,
                            "strokes": [[[round(px, 1), round(py, 1)]
                                         for px, py in s] for s in ordered]}, _f)
        except Exception as _e:
            if log: log(f"strokes sidecar error: {_e}")
    else:
        all_d = [strokes_to_single_d([s]) for s in strokes]
    if not all_d:
        all_d = ["M 0,0 L 0,0"]

    if status_cb: status_cb("Writing SVG…")
    write_svg(out_svg_path, all_d, w, h, cfg, status_cb=status_cb)
    preview_img = None
    try:
        preview_img = render_preview(all_d, w, h)
        preview_img.save(os.path.splitext(out_svg_path)[0] + "_preview.png")
    except Exception as _e:
        if log: log(f"Preview error: {_e}")

    return {
        "paths":         len(all_d),
        "pen_lifts_est": sum(d.count("M ") for d in all_d),
        "elapsed_sec":   time.time() - t0,
        "work_w":        w, "work_h": h,
        "aux_used":      0, "line_strokes": len(strokes),
        "preview_image": preview_img,
        "band_results":  [], "band_caps": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def hatch_pipeline(png_path: str, out_svg_path: str, cfg: HatchConfig,
                   progress_cb=None, status_cb=None, log_cb=None) -> Dict:
    """Load + preprocess the image, then render it with the selected engine."""
    t0 = time.time()
    def log(s):
        if log_cb:
            log_cb(s)
    if status_cb:
        status_cb("Loading + preprocessing image…")
    arr = load_and_preprocess_image(png_path, cfg)
    if cfg.line_spacing:
        return run_linespacing(png_path, arr, cfg, out_svg_path, t0,
                               status_cb=status_cb, log=log)
    return run_tonal(png_path, arr, cfg, out_svg_path, t0,
                     status_cb=status_cb, log=log)


def main():
    print("hatch_ui_nocairo is the tonal hatch library. Use the batch cockpit:")
    print("    py -3.11 tests/batch_ui.py")


if __name__ == "__main__":
    main()
