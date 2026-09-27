#!/usr/bin/env python3
"""
PNG -> pen-and-ink hatched SVG (single-line strokes)
Tone bands (quantile) + Detail + LightEdge + DarkEdge aux layers
Optimized for depth, clean shadows, crisp edge detail, and Cricut compatibility.
Each polygon = 1 SVG path element to minimize Cricut path count.

Dependencies:
  py -3.11 -m pip install pillow numpy shapely svgwrite scikit-image dearpygui
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

import shutil
import urllib.request

# Optional: OpenCV DNN for HED edge detection
try:
    import cv2 as _cv2
    _CV2_OK = hasattr(_cv2, "dnn")
except ImportError:
    _cv2    = None
    _CV2_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# HED (Holistically-nested Edge Detection) — optional
# ─────────────────────────────────────────────────────────────────────────────

_HED_DIR   = os.path.join(os.path.expanduser("~"), ".hatch_models")
_HED_PROTO = os.path.join(_HED_DIR, "hed_deploy.prototxt")
_HED_CAFFEMODEL = os.path.join(_HED_DIR, "hed_pretrained_bsds.caffemodel")

_HED_PROTO_URL = (
    "https://raw.githubusercontent.com/s9xie/hed"
    "/master/examples/hed/deploy.prototxt"
)
_HED_MODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"

_hed_net: object = None   # cached cv2.dnn.Net


def hed_model_ready() -> bool:
    return _CV2_OK and os.path.isfile(_HED_PROTO) and os.path.isfile(_HED_CAFFEMODEL)


def download_hed_model(log_cb=None) -> bool:
    """Download HED prototxt (~3 KB) and caffemodel (~56 MB). Returns True on success."""
    def log(s):
        if log_cb:
            log_cb(s)

    if not _CV2_OK:
        log("ERROR: opencv-python is required for HED.\n"
            "Install with:  pip install opencv-python")
        return False

    os.makedirs(_HED_DIR, exist_ok=True)

    # ── prototxt (~3 KB) ─────────────────────────────────────────────────────
    if os.path.isfile(_HED_PROTO):
        log(f"Prototxt already present: {_HED_PROTO}")
    else:
        log(f"Downloading HED prototxt…\n  {_HED_PROTO_URL}")
        try:
            urllib.request.urlretrieve(_HED_PROTO_URL, _HED_PROTO)
            log("  Prototxt OK.")
        except Exception as e:
            log(f"  Prototxt download failed: {e}")
            return False

    # ── caffemodel (~56 MB) ──────────────────────────────────────────────────
    if os.path.isfile(_HED_CAFFEMODEL):
        log(f"Caffemodel already present: {_HED_CAFFEMODEL}")
    else:
        log(f"Downloading HED caffemodel (~56 MB)…\n  {_HED_MODEL_URL}")
        tmp = _HED_CAFFEMODEL + ".tmp"
        try:
            def _hook(count, block, total):
                if count % 200 == 0 and total > 0:
                    mb_done  = count * block // (1024 * 1024)
                    mb_total = total // (1024 * 1024)
                    log(f"  {mb_done} / {mb_total} MB…")
            urllib.request.urlretrieve(_HED_MODEL_URL, tmp, reporthook=_hook)
            os.replace(tmp, _HED_CAFFEMODEL)
            log("  Caffemodel OK.")
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            log(f"  Caffemodel download failed: {e}\n"
                f"  Manual download:\n    {_HED_MODEL_URL}\n"
                f"  Save to:\n    {_HED_CAFFEMODEL}")
            return False

    log("HED model ready.")
    return True


def run_hed(img_rgb: np.ndarray, max_px: int = 512) -> np.ndarray:
    """
    Run HED on an RGB uint8 (H, W, 3) image.
    Downscales to max_px for CPU speed, then upscales result back.
    Returns edge probability map (H, W) float32 in [0, 1].
    """
    global _hed_net
    if _hed_net is None:
        _hed_net = _cv2.dnn.readNetFromCaffe(_HED_PROTO, _HED_CAFFEMODEL)

    h, w = img_rgb.shape[:2]

    # Downscale to max_px on the long edge for CPU speed
    scale = min(max_px / max(h, w, 1), 1.0)
    rh, rw = max(1, int(h * scale)), max(1, int(w * scale))
    if scale < 1.0:
        small = _cv2.resize(img_rgb, (rw, rh), interpolation=_cv2.INTER_AREA)
    else:
        small = img_rgb

    # HED: BGR with ImageNet mean subtraction
    blob = _cv2.dnn.blobFromImage(
        small[:, :, ::-1].astype(np.float32),
        scalefactor=1.0,
        size=(rw, rh),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    _hed_net.setInput(blob)
    raw = _hed_net.forward()          # (1, 1, rh, rw)
    edge_small = np.squeeze(raw)      # (rh, rw) float32

    # Upscale back to original resolution
    if scale < 1.0:
        edge_full = _cv2.resize(edge_small, (w, h), interpolation=_cv2.INTER_LINEAR)
    else:
        edge_full = edge_small

    return np.clip(edge_full, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HatchConfig:
    # Output
    out_width_in: float  = 12.0
    out_height_in: float = 12.0
    dpi: int             = 300

    # Preprocess
    resize_long_edge_px: int = 1800
    invert: bool             = False
    band_gamma: float        = 1.2

    # Local contrast enhancement (unsharp-mask style).
    # Smaller radius = captures finer detail (eyes behind glasses, fine texture).
    local_contrast: bool          = True
    local_contrast_radius: float  = 18.0
    local_contrast_strength: float = 0.55

    # Highlight protection: pixels BRIGHTER than this are excluded from tone-band
    # hatching. Keeps lens reflections, eye whites, skin highlights clean white.
    highlight_protect_thr: float = 0.92   # lower = more of lens/whites protected

    # Tone bands
    bands: int = 8

    # Hatching base
    base_spacing_px: float   = 10.0
    global_spacing_mult: float = 1.0
    base_angle_deg: float    = 25.0

    # Spacing easing — dramatic range for pen & ink depth
    dark_mult: float  = 0.32
    light_mult: float = 2.2
    ease_power: float = 2.2

    # Striping reduction — golden-ratio phase gives maximum angular separation
    # between adjacent bands, preventing moiré/constructive interference.
    angle_wobble_deg: float = 4.0    # ±4° sweep dark→light (~1.1° per band step)
    # Was 15° — caused visible angular seams in large flat-tone areas (background)
    # because each band polygon had a noticeably different hatch angle.  Reducing
    # to 4° keeps per-step angle change to ~1°, making band boundaries nearly
    # invisible (only density changes carry the tonal gradient, not angle changes).
    phase_step: float       = 0.37   # used by aux layers

    # Crosshatch (keep low for portraits: 1–2)
    cross_enabled: bool = True
    cross_dark_n: int   = 1

    # Complexity caps (now counts polygons, not segments)
    max_paths: int = 4500
    band_cap: int  = 600

    # Band mask blur — softens the iso-brightness contours that separate tone
    # bands.  Without this, smooth gradients produce nearly-straight polygon
    # edges that appear as a hard diagonal line in the output.
    band_blur_radius: float = 18.0  # px; 0 = off

    # Polygon cleanup
    simplify_tolerance_px: float = 0.25  # low value keeps inter-band boundaries tight (reduces seam gaps)
    min_area_dark_px2: float     = 50.0
    min_area_light_px2: float    = 400.0

    # Aux layer simplify
    aux_simplify_tol_px: float = 0.25

    # FaceShade — OFF by default (noisy in smooth areas)
    face_enabled: bool        = False
    face_pct: float           = 90.0
    face_budget: int          = 450
    face_spacing_mult: float  = 0.65
    face_cross: bool          = False
    face_min_area_mult: float = 0.8
    face_angle_offset: float  = 0.0

    # Detail edges — primary aux layer for pen & ink contours
    detail_enabled: bool        = True
    detail_pct: float           = 87.0
    detail_budget: int          = 1000
    detail_spacing_mult: float  = 0.45
    detail_min_area_px2: float  = 20.0
    detail_angle_offset: float  = 0.0
    detail_cross: bool          = False

    # LightEdge
    lightedge_enabled: bool       = True
    lightedge_tone_thr: float     = 0.65
    lightedge_pct: float          = 91.0
    lightedge_budget: int         = 500
    lightedge_spacing_mult: float = 0.35
    lightedge_min_area_px2: float = 8.0
    lightedge_cross: bool         = False

    # DarkEdge
    darkedge_enabled: bool       = True
    darkedge_tone_thr: float     = 0.28
    darkedge_pct: float          = 88.0
    darkedge_budget: int         = 350
    darkedge_spacing_mult: float = 0.40
    darkedge_min_area_px2: float = 10.0
    darkedge_cross: bool         = False

    # MicroDetail — OFF by default (main noise source)
    micro_enabled: bool          = False
    micro_pct: float             = 96.5
    micro_budget: int            = 450
    micro_spacing_mult: float    = 0.33
    micro_min_area_px2: float    = 3.0
    micro_cross: bool            = False
    micro_prefer_small_polys: bool = True
    micro_angle_offset: float    = 12.0

    # Recommendation system
    recommend_enabled: bool   = True
    use_recommended_caps: bool = True
    rec_floor: int            = 10
    rec_power: float          = 0.72
    light_boost: float        = 0.10
    rec_alpha_area: float     = 0.85
    rec_beta_dark: float      = 1.4
    rec_gamma_edge: float     = 0.6

    # HED edge detection (requires opencv-python + downloaded model)
    hed_enabled: bool  = False
    hed_max_px:  int   = 320    # downscale to this before HED inference (CPU speed)

    # Contour-following hatching
    # Each polygon's hatch angle is derived from the local gradient direction,
    # so lines flow along iso-brightness contours (like engraving or pen & ink).
    contour_hatch: bool           = False
    contour_hatch_strength: float = 1.0   # 0 = global angle, 1 = pure contour
    contour_hatch_blur: float     = 8.0   # pre-blur px for gradient field (smooths flow)

    # ── Raster hatch passes (no polygon finding) ─────────────────────────────
    # Each pass scans parallel lines at a different angle and emits segments
    # wherever brightness < its threshold.  Cumulative thresholds mean darker
    # pixels receive more overlapping passes → natural crosshatch shading.
    # Completely replaces the polygon band section when enabled.
    raster_hatch: bool        = False
    raster_passes: int        = 3      # number of angle passes
    raster_dark_thr: float    = 0.30   # threshold for darkest pass (deep shadows only)
    raster_light_thr: float   = 0.75   # threshold for lightest pass (most non-highlight areas)
    raster_angle_step: float  = 45.0   # degrees between pass angles

    # ── Iso-contour lines ─────────────────────────────────────────────────────
    # Draws iso-brightness curves directly as pen strokes — like a topo map of
    # the brightness field.  Very hand-drawn, no polygon artifacts.
    iso_contour_enabled: bool   = False
    iso_contour_levels: int     = 20    # number of contour levels
    iso_contour_blur_px: float  = 2.0   # smooth arr before contour finding
    iso_contour_min_len: float  = 20.0  # min contour length in pixels
    iso_contour_budget: int     = 2000  # max path elements
    iso_contour_gamma: float    = 0.5   # level compression (<1 = more in darks)

    # ── Line-art mode ─────────────────────────────────────────────────────────
    # Edge-forward: vectorize an edge map (HED or Canny) into polyline strokes
    # that TRACE the subject's contours, then add light hatching only in the
    # darkest regions for shading.  Produces a recognizable pen-and-ink drawing,
    # unlike the tone-band fills which only shade.  When True this REPLACES the
    # band/aux machinery entirely.
    line_art: bool             = False
    line_edge_source: str      = "hed"    # "hed" | "canny"
    line_hed_px: int           = 640      # HED inference long-edge (finer = more detail)
    line_edge_thr: float       = 0.28     # single-threshold fallback (hed path)
    # Hysteresis thresholding: keep edges above _hi, plus edges above _lo that
    # connect to a _hi edge. Yields clean connected contours instead of the
    # speckly closed-loop "web" a single flat threshold produces.
    line_hysteresis: bool      = True
    line_edge_thr_hi: float    = 0.32     # strong-edge seed
    line_edge_thr_lo: float    = 0.10     # grow into weak edges connected to seeds
    line_despur_px: float      = 7.0      # drop dead-end skeleton spurs shorter than this
    line_canny_lo: int         = 60
    line_canny_hi: int         = 160
    line_smooth_px: float      = 1.0      # blur edge map before thresholding
    line_min_len_px: float     = 9.0      # drop polylines shorter than this (noise)
    line_simplify_px: float    = 1.0      # Douglas-Peucker tolerance
    line_stitch_jump_px: float = 2.0      # only rejoin skeleton fragments this close
                                          # (larger draws travel lines between
                                          # unrelated contours -> a "web")
    line_subpaths_per_path: int = 1500    # pack this many strokes into one <path>
    line_max_paths: int        = 4500
    # Cricut struggles with many pen lifts (each M = a lift). Cap the total;
    # least-efficient strokes (fewest drawn segments per lift) are dropped first.
    # 0 = disabled.
    max_pen_lifts: int         = 0
    # Shading (secondary): graduated crosshatch UNDER the contour lines.
    # Multiple tone levels from line_shade_hi (lightest shaded) down to
    # line_shade_lo (darkest); each level hatches at a different angle, so
    # darker tones — which fall inside more levels — accumulate more passes
    # and read darker. This gives a smooth tonal gradient across the subject,
    # not a single rigid diagonal stripe.
    line_shade: bool           = True
    line_shade_levels: int     = 4
    line_shade_hi: float       = 0.60    # tones lighter than this stay white
    line_shade_lo: float       = 0.06    # darkest shaded tone
    line_shade_blur_px: float  = 2.0     # smooth tone masks (less staircase)
    line_shade_spacing_px: float = 5.0   # per-pass spacing (constant)
    line_shade_angle: float    = 30.0    # base angle; +45° per level
    line_shade_min_area_px2: float = 40.0
    line_shade_thr: float      = 0.32    # (legacy, unused by graduated shading)

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


@dataclass
class CMYConfig:
    """
    CMY(K) halftone-style hatch pipeline.
    Each channel's value at a pixel drives local line density:
    high channel value → dense hatching → more ink → correct color mix.
    Traditional halftone screen angles minimise moiré between channels.
    """
    # Screen angles (classic offset-press values)
    cyan_angle_deg:    float = 105.0   # 15° equivalent, avoids moiré with M
    magenta_angle_deg: float =  75.0
    yellow_angle_deg:  float =  90.0   # least visible, any angle works
    black_angle_deg:   float =  45.0

    # Spacing range: maps channel value 0→1 to max→min line spacing
    min_spacing_px: float = 3.0    # at 100 % ink (channel = 1.0)
    max_spacing_px: float = 40.0   # at threshold (near 0 % ink)

    # Ignore pixels below this channel value (avoids hatching near-white areas)
    min_channel_thr: float = 0.10

    # Quantise each channel into this many density steps (smoother = more bands)
    density_bands: int = 10

    # Under-colour removal: generate a K (black) plate from shared CMY
    use_black:  bool  = True
    ucr_amount: float = 0.85   # 0 = no K, 1 = maximum K extraction

    # Crosshatch the black plate for extra depth in shadow areas
    black_cross: bool = False

    # Polygon options (shared across all channels)
    min_area_px2:          float = 30.0
    simplify_tol_px:       float = 0.25
    max_paths_per_channel: int   = 1400


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def filter_slivers(polys: List[Polygon], max_aspect: float = 5.5) -> List[Polygon]:
    """Remove polygons whose bounding-box aspect ratio exceeds max_aspect.
    Slivers (long thin polygons) appear as diagonal stripes cutting through
    the image when hatched, because their narrow shape restricts all hatch
    lines to one direction that visually reads as a single angular line.
    """
    out = []
    for p in polys:
        minx, miny, maxx, maxy = p.bounds
        W = max(maxx - minx, 1e-3)
        H = max(maxy - miny, 1e-3)
        if max(W / H, H / W) <= max_aspect:
            out.append(p)
    return out


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

def quantile_thresholds(arr01: np.ndarray, bands: int) -> List[float]:
    qs   = [i / bands for i in range(1, bands)]
    flat = arr01.reshape(-1)
    thr  = [float(np.quantile(flat, q)) for q in qs]
    out, last = [], -1.0
    for t in thr:
        if t <= last:
            t = last + 1e-6
        out.append(min(t, 1.0))
        last = t
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

def grad_mag(arr01: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(arr01)
    return np.hypot(gx, gy).astype(np.float32)

def laplacian_abs(arr01: np.ndarray) -> np.ndarray:
    c  = arr01
    up = np.roll(c,  1, axis=0)
    dn = np.roll(c, -1, axis=0)
    lf = np.roll(c,  1, axis=1)
    rt = np.roll(c, -1, axis=1)
    return np.abs(up + dn + lf + rt - 4.0 * c).astype(np.float32)


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


def mask_dominant_angle(mask: np.ndarray,
                        gx_arr: np.ndarray,
                        gy_arr: np.ndarray) -> float:
    """
    Dominant iso-brightness CONTOUR direction (degrees) for all pixels in mask.

    Samples the gradient field at mask pixels and applies the structure tensor
    to find the dominant gradient direction; the contour direction is +90°.
    Using the full mask (not per-polygon bounding boxes) gives one consistent
    angle for all polygons in a band, avoiding seam artifacts at polygon borders.
    """
    if not mask.any():
        return 0.0

    gx = gx_arr[mask].ravel()
    gy = gy_arr[mask].ravel()

    # Structure tensor components
    E = float(np.mean(gx * gx))
    G = float(np.mean(gy * gy))
    F = float(np.mean(gx * gy))

    # Principal gradient direction → contour direction is +90°
    grad_angle_rad = 0.5 * math.atan2(2.0 * F, E - G)
    return math.degrees(grad_angle_rad) + 90.0


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
                       mask: Optional[np.ndarray] = None) -> str:
    """
    Stitch many hatch segments into ONE d-string via a greedy nearest-neighbour
    tour: always travel to the closest unused segment.

    Pen-down vs pen-up per jump:
      - with `mask` (the layer's dark region): pen stays DOWN whenever the travel
        runs through hatched area (invisible) and only LIFTS when it would cross a
        light/unhatched area — minimises lifts with no visible connectors.
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

        # Prefer the nearest candidate whose connector stays pen-DOWN — a short
        # turn, or (with a mask) a travel that doesn't cross whitespace. This
        # picks the entry SIDE of the next line that avoids a white crossing;
        # only if no nearby candidate qualifies do we accept a lift (nearest).
        pick, pen_down = None, False
        for d, seg, which in cand:
            a = ends[2 * seg + which]
            if d <= join_jump or (mask is not None and
                    _travel_hidden(cur_end[0], cur_end[1], a[0], a[1], mask)):
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

def hatch_polys_to_d_strings(polys: List[Polygon],
                              bounds: Tuple[float, float, float, float],
                              spacing: float, angle: float, phase: float,
                              cross: bool, stitch_jump: float, cap: int,
                              prefer_small: bool = False) -> List[str]:
    """1 d-string per polygon → 1 Cricut path element per polygon."""
    if not polys or cap <= 0:
        return []

    lines  = make_parallel_lines(bounds, spacing, angle, phase=phase)
    lines2 = make_parallel_lines(bounds, spacing, angle + 90.0, phase=phase) if cross else None

    result: List[str] = []
    for poly in sorted(polys, key=lambda p: p.area, reverse=not prefer_small):
        if len(result) >= cap:
            break
        segs1 = clip_lines_to_polygon(lines, poly)
        if not segs1 and lines2 is None:
            continue
        # Stitch each angle group separately so the boustrophedon sort
        # only sees collinear segments → far fewer pen lifts.
        d = stitch_segs_to_d_string(segs1, stitch_jump) if segs1 else ""
        if lines2 is not None:
            segs2 = clip_lines_to_polygon(lines2, poly)
            if segs2:
                d2 = stitch_segs_to_d_string(segs2, stitch_jump)
                d = (d + " " + d2).strip() if d else d2
        if d:
            result.append(d)

    return result[:cap]


# ─────────────────────────────────────────────────────────────────────────────
# Raster hatch pass (Option 6 — no polygon finding)
# ─────────────────────────────────────────────────────────────────────────────

def raster_hatch_pass(arr: np.ndarray, highlight_mask: np.ndarray,
                      bounds, angle_deg: float, spacing: float,
                      threshold: float, stitch_jump: float) -> str:
    """
    Scan parallel lines across the brightness array and emit segments where
    arr < threshold AND highlight_mask is True.  No polygon finding at all —
    samples the raster directly.  Returns a single stitched SVG path d-string
    for the entire pass (all lines boustrophedon-stitched together).
    """
    h, w = arr.shape
    minx, miny, maxx, maxy = bounds

    angle_r = math.radians(angle_deg)
    dx, dy  = math.cos(angle_r), math.sin(angle_r)
    nx, ny  = -dy, dx  # perpendicular (normal) direction

    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    projs   = [cx * nx + cy * ny for cx, cy in corners]
    pmin, pmax = min(projs), max(projs)
    k0 = int(math.floor(pmin / spacing)) - 1
    k1 = int(math.ceil(pmax  / spacing)) + 1
    L  = math.hypot(maxx - minx, maxy - miny) + 20.0

    segs: List[LineString] = []

    for k in range(k0, k1 + 1):
        off = k * spacing
        if abs(ny) > 1e-9:
            ox, oy = 0.0, off / ny
        else:
            ox, oy = off / nx, 0.0

        # Sample at ~1px intervals along the line
        n_pts = int(L) + 4
        ts = np.linspace(-L / 2, L / 2, n_pts)
        xs = ox + ts * dx
        ys = oy + ts * dy

        # Bounds check
        in_b = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xi = np.clip(xs.astype(np.int32), 0, w - 1)
        yi = np.clip(ys.astype(np.int32), 0, h - 1)

        # Sample brightness + highlight
        vals   = arr[yi, xi]
        hl     = highlight_mask[yi, xi]
        active = in_b & (vals < threshold) & hl

        # Find contiguous active runs
        padded = np.concatenate([[False], active, [False]])
        diff   = np.diff(padded.astype(np.int8))
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            # Lines are straight — only need endpoints
            segs.append(LineString([(xs[s], ys[s]), (xs[e - 1], ys[e - 1])]))

    if not segs:
        return ""
    return stitch_segs_to_d_string(segs, stitch_jump)


# ─────────────────────────────────────────────────────────────────────────────
# Iso-brightness contour lines (Option 2)
# ─────────────────────────────────────────────────────────────────────────────

def iso_contour_d_strings(arr: np.ndarray, highlight_mask: np.ndarray,
                          n_levels: int, highlight_thr: float,
                          min_length: float, blur_px: float,
                          gamma: float, cap: int) -> List[str]:
    """
    Draw iso-brightness contour polylines as SVG paths.
    Levels are gamma-compressed toward dark tones so shadows get more lines.
    Contours that pass mostly through highlight-protected areas are skipped.
    Returns one d-string per kept contour polyline.
    """
    if not _SKIMAGE_OK:
        return []

    if blur_px > 0:
        _pil = Image.fromarray((arr * 255).astype(np.uint8))
        _pil = _pil.filter(ImageFilter.GaussianBlur(radius=blur_px))
        arr_s = np.asarray(_pil).astype(np.float32) / 255.0
    else:
        arr_s = arr

    # gamma < 1 compresses levels toward dark (0→ more contours in shadows)
    t      = np.linspace(0.0, 1.0, n_levels + 2)[1:-1]   # skip 0 and 1 exactly
    levels = (t ** gamma) * float(highlight_thr)

    h, w = highlight_mask.shape
    d_strings: List[str] = []

    for level in levels:
        if len(d_strings) >= cap:
            break
        try:
            contours = measure.find_contours(arr_s, float(level))
        except Exception:
            continue

        for c in contours:
            if len(d_strings) >= cap:
                break
            if len(c) < max(3, int(min_length / 2)):
                continue
            # c shape: (N, 2) = (row, col)
            pts = [(float(c[i, 1]), float(c[i, 0])) for i in range(len(c))]

            # Skip contours that mostly pass through highlight-protected pixels
            n_ok = sum(
                1 for x, y in pts
                if 0 <= int(y) < h and 0 <= int(x) < w
                   and highlight_mask[int(y), int(x)]
            )
            if n_ok < len(pts) * 0.5:
                continue

            d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            d_strings.append(d)

    return d_strings


# ─────────────────────────────────────────────────────────────────────────────
# Recommended allocation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_band_percentages(arr, edge_metric, band_masks, cfg):
    scores, stats = [], []
    for m in band_masks:
        area = float(m.mean())
        if m.sum() == 0:
            mean_t, edge = 1.0, 0.0
        else:
            mean_t = float(arr[m].mean())
            edge   = float(edge_metric[m].mean())
        dark = 1.0 - mean_t
        s = (max(area, 1e-9) ** cfg.rec_alpha_area *
             max(dark, 1e-9) ** cfg.rec_beta_dark  *
             max(edge, 1e-9) ** cfg.rec_gamma_edge)
        scores.append(s)
        stats.append((area, mean_t, dark, edge, s))
    ssum = sum(scores) or 1.0
    return [100.0 * s / ssum for s in scores], stats

def build_improved_caps(rec_perc, total_paths, cfg):
    n = len(rec_perc)
    p = np.array(rec_perc, dtype=np.float32) / 100.0
    p = np.power(np.clip(p, 1e-9, 1.0), float(cfg.rec_power))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    p = p * (1.0 + float(cfg.light_boost) * t)
    p = p / max(float(p.sum()), 1e-9)

    floor = max(0, int(cfg.rec_floor))
    caps  = np.full(n, floor, dtype=np.int32)
    remaining = total_paths - int(caps.sum())

    if remaining <= 0:
        i = 0
        while int(caps.sum()) > total_paths and i < n * 3:
            j = i % n
            if caps[j] > 0:
                caps[j] -= 1
            i += 1
        return caps.tolist()

    raw  = p * float(remaining)
    add  = np.floor(raw).astype(np.int32)
    caps += add
    drift = remaining - int(add.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    k = 0
    while drift > 0:
        caps[order[k % n]] += 1
        drift -= 1
        k += 1
    return caps.tolist()


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
    for idx, d in enumerate(d_strings):
        if status_cb and idx % 250 == 0:
            status_cb(f"Writing SVG: {idx}/{len(d_strings)}")
        if not d:
            continue
        dwg.add(dwg.path(
            d=d, fill="none", stroke="black",
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
        i       = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "M":
                i += 1
                if i < len(tokens):
                    x, y    = map(float, tokens[i].split(","))
                    current = (x * scale, y * scale)
                    i += 1
            elif tok == "L":
                i += 1
                if i < len(tokens):
                    x, y = map(float, tokens[i].split(","))
                    pt   = (x * scale, y * scale)
                    if current is not None:
                        draw.line([current, pt], fill=(15, 15, 15), width=1)
                    current = pt
                    i += 1
            else:
                i += 1

    return im


# ─────────────────────────────────────────────────────────────────────────────
# Line-art mode — edge contours as polyline strokes + light shading
# ─────────────────────────────────────────────────────────────────────────────

def _line_edge_map(png_path: str, arr: np.ndarray, cfg: HatchConfig, log=None) -> np.ndarray:
    """Edge-probability map (0..1) at the working-array resolution."""
    h, w = arr.shape
    src = cfg.line_edge_source
    if src == "hed" and _CV2_OK and hed_model_ready():
        pil = Image.open(png_path).convert("RGBA")
        bg  = Image.new("RGBA", pil.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(bg, pil).convert("RGB").resize((w, h), Image.LANCZOS)
        e = run_hed(np.asarray(pil), max_px=max(64, int(cfg.line_hed_px)))
        if log: log(f"Line edges: HED  ({w}x{h}, infer {cfg.line_hed_px}px)")
        return e
    # Canny / gradient fallback
    if _CV2_OK:
        g = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        e = _cv2.Canny(g, int(cfg.line_canny_lo), int(cfg.line_canny_hi))
        if log: log("Line edges: Canny (HED unavailable)")
        return (e > 0).astype(np.float32)
    gm = grad_mag(arr)
    if log: log("Line edges: gradient fallback")
    return gm / (float(gm.max()) + 1e-6)


def _trace_skeleton(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Trace a 1-px skeleton into polyline pixel-chains (each edge walked once)."""
    pts = set(zip(*np.where(skel)))
    if not pts:
        return []
    NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def neighbors(p):
        r, c = p
        return [(r + dr, c + dc) for dr, dc in NB if (r + dr, c + dc) in pts]

    deg  = {p: len(neighbors(p)) for p in pts}
    seen = set()                       # frozenset({p, q}) edges already walked
    chains: List[List[Tuple[int, int]]] = []

    def walk(start, second):
        chain = [start, second]
        seen.add(frozenset((start, second)))
        prev, cur = start, second
        while deg.get(cur, 0) == 2:    # keep going through simple path pixels
            nxt = None
            for q in neighbors(cur):
                if q != prev and frozenset((cur, q)) not in seen:
                    nxt = q
                    break
            if nxt is None:
                break
            seen.add(frozenset((cur, nxt)))
            chain.append(nxt)
            prev, cur = cur, nxt
        return chain

    # Start every chain at an endpoint or junction (deg != 2)
    for p in [q for q in pts if deg[q] != 2]:
        for q in neighbors(p):
            if frozenset((p, q)) not in seen:
                chains.append(walk(p, q))
    # Remaining pure loops (all deg == 2)
    for p in pts:
        for q in neighbors(p):
            if frozenset((p, q)) not in seen:
                chains.append(walk(p, q))
    return chains, deg


def edges_to_polylines(edge01: np.ndarray, cfg: HatchConfig, log=None) -> List[List[Tuple[float, float]]]:
    """Threshold -> skeletonize -> trace -> simplify -> polylines in (x, y)."""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        if log: log("edges_to_polylines: scikit-image missing")
        return []

    e = np.clip(edge01, 0.0, 1.0)
    if cfg.line_smooth_px > 0:
        e = np.asarray(
            Image.fromarray((e * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(float(cfg.line_smooth_px)))
        ).astype(np.float32) / 255.0

    # Hysteresis thresholding gives clean connected contours; a single flat
    # threshold produces a speckly web.
    binary = None
    if cfg.line_hysteresis:
        try:
            from skimage.filters import apply_hysteresis_threshold
            binary = apply_hysteresis_threshold(
                e, float(cfg.line_edge_thr_lo), float(cfg.line_edge_thr_hi))
        except Exception:
            binary = None
    if binary is None:
        binary = e >= float(cfg.line_edge_thr)
    if binary.sum() == 0:
        return []
    skel = skeletonize(binary)
    chains, deg = _trace_skeleton(skel)

    despur = float(cfg.line_despur_px)
    out: List[List[Tuple[float, float]]] = []
    n_spur = 0
    for chain in chains:
        if len(chain) < 2:
            continue
        ls = LineString([(float(c), float(r)) for (r, c) in chain])  # x=col, y=row
        # A chain with a free (degree-1) end that is short is a spur / fleck.
        has_free_end = (deg.get(chain[0], 0) == 1 or deg.get(chain[-1], 0) == 1)
        if has_free_end and ls.length < despur:
            n_spur += 1
            continue
        if ls.length < float(cfg.line_min_len_px):
            continue
        if cfg.line_simplify_px > 0:
            ls = ls.simplify(float(cfg.line_simplify_px), preserve_topology=False)
        coords = list(ls.coords)
        if len(coords) >= 2:
            out.append(coords)
    if log: log(f"Line strokes: {len(out)} polylines "
                f"(from {len(chains)} traced, {n_spur} spurs pruned)")
    return out


def polylines_to_d_strings(polylines, cfg: HatchConfig) -> List[str]:
    """Order strokes on a boustrophedon grid, pen-down-join near ones, pack
    many subpaths per <path> to keep the Cricut path count low."""
    if not polylines:
        return []

    band = max(1.0, float(cfg.line_stitch_jump_px) * 4.0)

    def key(pl):
        x0, y0 = pl[0]
        b = int(y0 // band)
        return (b, x0 if b % 2 == 0 else -x0)   # snake left/right per row band

    polylines = sorted(polylines, key=key)
    jump = float(cfg.line_stitch_jump_px)

    paths: List[str] = []
    parts: List[str] = []
    subpaths = 0
    prev_end = None

    def flush():
        nonlocal parts, subpaths, prev_end
        if parts:
            paths.append(" ".join(parts))
        parts, subpaths, prev_end = [], 0, None

    for pl in polylines:
        if prev_end is not None:
            d0 = math.hypot(prev_end[0] - pl[0][0],  prev_end[1] - pl[0][1])
            d1 = math.hypot(prev_end[0] - pl[-1][0], prev_end[1] - pl[-1][1])
            if d1 < d0:
                pl = pl[::-1]
            gap = min(d0, d1)
        else:
            gap = None

        x0, y0 = pl[0]
        if prev_end is not None and gap is not None and gap <= jump:
            parts.append(f"L {x0:.2f},{y0:.2f}")     # pen stays down
        else:
            parts.append(f"M {x0:.2f},{y0:.2f}")     # pen lift
        for x, y in pl[1:]:
            parts.append(f"L {x:.2f},{y:.2f}")
        prev_end = pl[-1]
        subpaths += 1
        if subpaths >= int(cfg.line_subpaths_per_path):
            flush()
    flush()
    return paths[:int(cfg.line_max_paths)]


def run_line_art(png_path: str, arr: np.ndarray, cfg: HatchConfig,
                 out_svg_path: str, t0: float, status_cb=None, log=None) -> Dict:
    h, w = arr.shape
    bounds = (0.0, 0.0, float(w), float(h))

    if status_cb: status_cb("Line-art: detecting edges…")
    edge = _line_edge_map(png_path, arr, cfg, log=log)

    if status_cb: status_cb("Line-art: vectorizing contours…")
    polylines = edges_to_polylines(edge, cfg, log=log)
    line_d = polylines_to_d_strings(polylines, cfg)

    # Secondary shading: graduated crosshatch under the lines. Darker tones
    # fall inside more levels → more overlapping angle passes → read darker.
    shade_d: List[str] = []
    if cfg.line_shade:
        if status_cb: status_cb("Line-art: graduated shading…")
        arr_s = arr
        if cfg.line_shade_blur_px > 0:
            arr_s = np.asarray(
                Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                     .filter(ImageFilter.GaussianBlur(float(cfg.line_shade_blur_px)))
            ).astype(np.float32) / 255.0

        levels = max(1, int(cfg.line_shade_levels))
        ths = np.linspace(float(cfg.line_shade_hi), float(cfg.line_shade_lo), levels)
        remaining = max(0, cfg.line_max_paths - len(line_d))
        spacing = max(1.0, float(cfg.line_shade_spacing_px))
        for li, th in enumerate(ths):
            if remaining <= 0:
                break
            mask = arr_s < float(th)
            # Clear the 1px border so regions touching the image edge (e.g. a
            # dark background) close into valid polygons instead of being
            # dropped by find_contours — otherwise the darkest areas get no
            # shading and the image looks tonally inverted.
            mask[0, :] = mask[-1, :] = False
            mask[:, 0] = mask[:, -1] = False
            if mask.sum() == 0:
                continue
            angle = float(cfg.line_shade_angle) + 45.0 * li   # distinct per level
            polys = mask_to_polygons(mask, cfg,
                                     min_area=float(cfg.line_shade_min_area_px2),
                                     simplify_tol=cfg.simplify_tolerance_px)
            d = hatch_polys_to_d_strings(
                polys, bounds, spacing=spacing, angle=angle, phase=0.0,
                cross=False, stitch_jump=spacing * 1.2, cap=remaining,
            )
            shade_d.extend(d)
            remaining -= len(d)
            if log: log(f"Shade L{li+1} tone<{th:.2f} ang={angle:.0f}: +{len(d)} paths")

    # Enforce a pen-lift budget (Cricut struggles with many lifts). Protect the
    # contour trace (it carries recognizability); spend the remaining budget on
    # shading, dropping the least-efficient shading strokes (fewest drawn
    # segments per lift) first.
    if cfg.max_pen_lifts and cfg.max_pen_lifts > 0:
        cap = int(cfg.max_pen_lifts)
        def _lifts(d): return d.count("M ")
        def _eff(d):   return d.count("L ") / max(1, d.count("M "))

        def _trim(dstrings, budget):
            total = sum(_lifts(d) for d in dstrings)
            if total <= budget:
                return dstrings
            keep = [True] * len(dstrings)
            for i in sorted(range(len(dstrings)), key=lambda k: _eff(dstrings[k])):
                if total <= budget:
                    break
                total -= _lifts(dstrings[i])
                keep[i] = False
            return [d for i, d in enumerate(dstrings) if keep[i]]

        line_lifts = sum(_lifts(d) for d in line_d)
        if line_lifts >= cap:
            line_d = _trim(line_d, cap)   # trace alone over budget
            shade_d = []
        else:
            shade_d = _trim(shade_d, cap - line_lifts)
        if log: log(f"Pen-lift cap {cap}: trace={sum(_lifts(d) for d in line_d)} "
                    f"shade={sum(_lifts(d) for d in shade_d)}")

    all_d = shade_d + line_d          # shading first, contours on top
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

    pen_lifts = sum(d.count("M ") for d in all_d)
    return {
        "paths":         len(all_d),
        "pen_lifts_est": pen_lifts,
        "elapsed_sec":   time.time() - t0,
        "work_w":        w,
        "work_h":        h,
        "aux_used":      len(shade_d),
        "line_strokes":  len(polylines),
        "preview_image": preview_img,
        "band_results":  [],
        "band_caps":     [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tonal mode — cross-hatch by darkness, one stitched path per angle-layer
# ─────────────────────────────────────────────────────────────────────────────

def run_tonal(png_path: str, arr: np.ndarray, cfg: HatchConfig,
              out_svg_path: str, t0: float, status_cb=None, log=None) -> Dict:
    """
    Steps: 1) polygonize tonal regions  2) simplify  3) darkness -> layer count
    4) each darker region falls inside more layers  5) stitch each layer into
    ONE path  6) offset the hatch angle per layer so passes don't overlap.
    """
    h, w = arr.shape
    bounds = (0.0, 0.0, float(w), float(h))

    arr_s = arr
    if cfg.tonal_blur_px > 0:
        arr_s = np.asarray(
            Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
                 .filter(ImageFilter.GaussianBlur(float(cfg.tonal_blur_px)))
        ).astype(np.float32) / 255.0

    layers = max(1, int(cfg.tonal_max_layers))
    # Nested thresholds: a pixel darker than ths[i] receives layer i. Darker
    # pixels pass more thresholds -> more layers -> darker (steps 3-4).
    ths = np.linspace(float(cfg.tonal_hi), float(cfg.tonal_lo), layers)
    spacing = max(1.0, float(cfg.tonal_spacing_px))

    all_d: List[str] = []
    for i, th in enumerate(ths):
        if status_cb: status_cb(f"Tonal layer {i+1}/{layers}…")
        mask = arr_s < float(th)
        # close border-touching regions so dark backgrounds polygonize
        mask[0, :] = mask[-1, :] = False
        mask[:, 0] = mask[:, -1] = False
        if mask.sum() == 0:
            continue
        polys = mask_to_polygons(mask, cfg,                       # steps 1-2
                                 min_area=float(cfg.tonal_min_area_px2),
                                 simplify_tol=cfg.simplify_tolerance_px)
        if not polys:
            continue
        angle = float(cfg.tonal_base_angle) + float(cfg.tonal_angle_step) * i  # step 6
        lines = make_parallel_lines(bounds, spacing, angle, phase=0.0)
        segs: List[LineString] = []
        for poly in polys:
            segs.extend(clip_lines_to_polygon(lines, poly))
        if not segs:
            continue
        # step 5: 1 layer = 1 path, greedy NN tour to minimise pen lifts
        if cfg.tonal_greedy_stitch:
            d = stitch_segs_greedy(
                segs, spacing * float(cfg.tonal_join_mult),
                mask=mask if cfg.tonal_hide_travel else None)
        else:
            d = stitch_segs_to_d_string(segs, spacing * 1.4)
        if d:
            all_d.append(d)
            if log: log(f"Tonal layer {i+1}: tone<{th:.2f} ang={angle:.0f} "
                        f"polys={len(polys)} segs={len(segs)} -> 1 path")

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

def hatch_pipeline(png_path: str, out_svg_path: str, cfg: HatchConfig,
                   progress_cb=None, status_cb=None, log_cb=None) -> Dict:
    t0 = time.time()

    def log(s):
        if log_cb:
            log_cb(s)

    if status_cb:
        status_cb("Loading + preprocessing image…")
    arr  = load_and_preprocess_image(png_path, cfg)
    h, w = arr.shape
    bounds = (0.0, 0.0, float(w), float(h))
    log(f"Working image: {w}×{h}")

    # Line-art mode replaces the band/aux machinery entirely.
    if cfg.line_art:
        return run_line_art(png_path, arr, cfg, out_svg_path, t0,
                            status_cb=status_cb, log=log)

    # Tonal mode: pure cross-hatch by darkness, one path per angle-layer.
    if cfg.tonal:
        return run_tonal(png_path, arr, cfg, out_svg_path, t0,
                         status_cb=status_cb, log=log)

    gmag_arr = grad_mag(arr)
    lap      = laplacian_abs(arr)

    # ── Edge metric: HED (if enabled + model ready) or gradient fallback ──────
    if cfg.hed_enabled and hed_model_ready():
        if status_cb:
            status_cb("Running HED edge detection…")
        try:
            # Need the original RGB image at working resolution
            _pil_rgb = Image.open(png_path).convert("RGBA")
            _bg      = Image.new("RGBA", _pil_rgb.size, (255, 255, 255, 255))
            _pil_rgb = Image.alpha_composite(_bg, _pil_rgb).convert("RGB")
            _ww, _wh = _pil_rgb.size
            _long    = max(_ww, _wh)
            if _long > cfg.resize_long_edge_px:
                _sc  = cfg.resize_long_edge_px / _long
                _pil_rgb = _pil_rgb.resize(
                    (max(1, int(_ww * _sc)), max(1, int(_wh * _sc))), Image.LANCZOS)
            _img_np   = np.array(_pil_rgb, dtype=np.uint8)
            edge_metric = run_hed(_img_np, max_px=cfg.hed_max_px)
            log(f"HED edge map: min={edge_metric.min():.3f} max={edge_metric.max():.3f}")
        except Exception as _hed_err:
            log(f"HED failed ({_hed_err}), falling back to gradient edges.")
            edge_metric = (0.65 * gmag_arr + 0.35 * lap).astype(np.float32)
    else:
        edge_metric = (0.65 * gmag_arr + 0.35 * lap).astype(np.float32)

    # Band-assignment array: blur arr slightly so iso-brightness contours become
    # gently curved rather than perfectly straight lines.  Prevents the hard
    # diagonal-edge artifact that appears where two tonal bands meet in smooth
    # gradient areas (e.g. studio backgrounds, glass lenses).
    if cfg.band_blur_radius > 0:
        _pil_b = Image.fromarray((arr * 255).astype(np.uint8))
        _pil_b = _pil_b.filter(ImageFilter.GaussianBlur(radius=cfg.band_blur_radius))
        arr_b  = np.asarray(_pil_b).astype(np.float32) / 255.0
    else:
        arr_b = arr

    # Highlight protection (applied to blurred array so lens/sky regions stay clean)
    highlight_mask = arr_b < float(cfg.highlight_protect_thr)
    protected_px   = int((~highlight_mask).sum())
    protected_pct  = 100.0 * protected_px / max(arr.size, 1)
    log(f"Highlight protection: thr={cfg.highlight_protect_thr:.2f}  "
        f"protected={protected_px} ({protected_pct:.1f}%)")

    # Band masks (highlight-protected)
    band_masks: List[np.ndarray] = []
    thr  = quantile_thresholds(arr_b, cfg.bands)
    cuts = [0.0] + thr + [1.0000001]
    for bi in range(cfg.bands):
        lo, hi = cuts[bi], cuts[bi + 1]
        band_masks.append((arr_b >= lo) & (arr_b < hi) & highlight_mask)

    def band_min_area(bi):
        t = bi / max(1, cfg.bands - 1)
        return cfg.min_area_dark_px2 + (cfg.min_area_light_px2 - cfg.min_area_dark_px2) * t

    # Recommended allocation
    rec_perc = None
    if cfg.recommend_enabled:
        rec_perc, rec_stats = recommend_band_percentages(arr_b, edge_metric, band_masks, cfg)
        log("\n--- Band Allocation ---")
        for i, (p, st) in enumerate(zip(rec_perc, rec_stats), start=1):
            area, mean_t, dark, edge, score = st
            log(f"  Band {i:2d}: area={area*100:5.1f}% tone={mean_t:.2f} → {p:5.1f}%")

    # ── Aux layers (hard-reserved) ───────────────────────────────────────────
    all_d: List[str] = []
    base_spacing = float(cfg.base_spacing_px) * float(cfg.global_spacing_mult)

    # ── Contour-following gradient field ─────────────────────────────────────
    # Sample the ENTIRE image gradient once → one dominant contour angle for
    # the whole image.  Apply this as a uniform offset to every band and aux
    # layer's golden-ratio angle.  All layers rotate together → no
    # misalignment.  The golden-ratio wobble still gives band-to-band variety.
    _ch_offset = 0.0
    if cfg.contour_hatch:
        if cfg.contour_hatch_blur > 0:
            _chb_pil = Image.fromarray((arr * 255).astype(np.uint8))
            _chb_pil = _chb_pil.filter(ImageFilter.GaussianBlur(radius=cfg.contour_hatch_blur))
            _arr_ch  = np.asarray(_chb_pil).astype(np.float32) / 255.0
        else:
            _arr_ch = arr
        _gy_ch, _gx_ch = np.gradient(_arr_ch)
        # Dominant contour angle for the full image
        _full_mask = np.ones(arr.shape, dtype=bool)
        _img_contour_angle = mask_dominant_angle(_full_mask, _gx_ch, _gy_ch)
        # Signed offset from user's base angle, clamped to [-90, 90]
        _raw_diff  = (_img_contour_angle - cfg.base_angle_deg + 90.0) % 180.0 - 90.0
        _ch_offset = float(cfg.contour_hatch_strength) * _raw_diff
        log(f"Contour-follow: image_contour={_img_contour_angle:.1f}°  "
            f"offset={_ch_offset:.1f}°  strength={cfg.contour_hatch_strength:.2f}")

    def _contour_blend(base_angle: float) -> float:
        """Add the global contour offset to any base angle."""
        return base_angle + _ch_offset

    def aux_jump(sp):
        # 1.2x: only stitch truly adjacent parallel lines (1 spacing apart).
        # 3.5x caused connector L-lines to cross concave gaps in irregular
        # gradient-edge polygons, appearing as odd angular shapes in the output.
        return sp * 1.2

    def run_aux(label, mask, polys_fn, spacing_mult, angle_off, phase_off, cross, budget, prefer_small=False):
        if budget <= 0:
            return []
        if status_cb:
            status_cb(f"{label} mask → polygons…")
        polys   = polys_fn()
        spacing = max(1.0, base_spacing * spacing_mult)
        result  = hatch_polys_to_d_strings(
            polys, bounds, spacing=spacing,
            angle=_contour_blend(cfg.base_angle_deg + angle_off),
            phase=phase_off, cross=cross,
            stitch_jump=aux_jump(spacing),
            cap=budget, prefer_small=prefer_small,
        )
        log(f"{label}: {len(polys)} polys → {len(result)} path elements")
        return result

    face_d: List[str] = []
    if cfg.face_enabled:
        thr_f    = float(np.quantile(gmag_arr.reshape(-1), cfg.face_pct / 100.0))
        mask_f   = gmag_arr >= thr_f
        min_f    = max(1.0, cfg.min_area_dark_px2 * cfg.face_min_area_mult)
        face_d   = run_aux("FaceShade",
                           mask_f,
                           lambda: mask_to_polygons(mask_f, cfg, min_f, cfg.aux_simplify_tol_px),
                           cfg.face_spacing_mult, cfg.face_angle_offset,
                           (0.13 + cfg.phase_step) % 1.0,
                           cfg.face_cross, cfg.face_budget)

    # ── Diagnostic mask PNGs ─────────────────────────────────────────────────
    _diag_dir = os.path.join(os.path.dirname(out_svg_path), "diag_masks")
    os.makedirs(_diag_dir, exist_ok=True)
    _stem     = os.path.splitext(os.path.basename(out_svg_path))[0]

    def _save_mask_png(mask: np.ndarray, label: str) -> None:
        """Save a downscaled white-on-black mask PNG for visual inspection."""
        try:
            scale  = min(1.0, 1000.0 / max(mask.shape[1], mask.shape[0], 1))
            pw     = max(1, int(mask.shape[1] * scale))
            ph     = max(1, int(mask.shape[0] * scale))
            m_img  = Image.fromarray((mask.astype(np.uint8) * 255))
            m_img  = m_img.resize((pw, ph), Image.NEAREST)
            path   = os.path.join(_diag_dir, f"{_stem}_{label}.png")
            m_img.save(path)
            log(f"  [diag] {label} mask → {path}")
        except Exception as _e:
            log(f"  [diag] {label} mask save failed: {_e}")

    def _log_poly_stats(polys: List[Polygon], label: str) -> None:
        """Log bounding-box aspect ratios to detect thin angular slivers."""
        if not polys:
            return
        ratios = []
        for p in polys:
            minx, miny, maxx, maxy = p.bounds
            W = max(maxx - minx, 1e-3)
            H = max(maxy - miny, 1e-3)
            ratios.append(max(W / H, H / W))
        ratios.sort(reverse=True)
        sliver_n = sum(1 for r in ratios if r > 5)
        log(f"  [diag] {label} poly aspect: max={ratios[0]:.1f}  "
            f"mean={sum(ratios)/len(ratios):.1f}  "
            f"slivers(>5:1)={sliver_n}/{len(ratios)}")
        if ratios[0] > 5:
            log(f"  [diag] {label} top-5 ratios: {[round(r,1) for r in ratios[:5]]}")

    detail_d: List[str] = []
    if cfg.detail_enabled:
        thr_d    = float(np.quantile(gmag_arr.reshape(-1), cfg.detail_pct / 100.0))
        mask_d   = gmag_arr >= thr_d
        _save_mask_png(mask_d, "detail")
        _polys_d = filter_slivers(
            mask_to_polygons(mask_d, cfg, cfg.detail_min_area_px2, cfg.aux_simplify_tol_px)
        )
        _log_poly_stats(_polys_d, "Detail")
        detail_d = run_aux("Detail",
                           mask_d,
                           lambda: _polys_d,
                           cfg.detail_spacing_mult, cfg.detail_angle_offset,
                           (0.29 + 2 * cfg.phase_step) % 1.0,
                           cfg.detail_cross, cfg.detail_budget)

    # Smooth gradient for LightEdge/DarkEdge: raw gmag_arr has a peak at every
    # texture detail → hundreds of tiny disconnected polygons that appear as odd
    # angular shapes.  A moderate blur merges nearby peaks into coherent edge
    # stripes (similar to what HED's low-res output did naturally) while keeping
    # the edges spatially accurate at full resolution.
    _gmag_blur_r = max(1.0, cfg.band_blur_radius * 0.6)
    _gmag_pil    = Image.fromarray(
        np.clip(gmag_arr * (255.0 / max(float(gmag_arr.max()), 1e-9)), 0, 255).astype(np.uint8)
    )
    _gmag_smooth = np.asarray(
        _gmag_pil.filter(ImageFilter.GaussianBlur(radius=_gmag_blur_r))
    ).astype(np.float32) / 255.0

    lightedge_d: List[str] = []
    if cfg.lightedge_enabled:
        # Intersect with highlight_mask so LightEdge respects the HL threshold.
        # Without this, LightEdge fires in highlight-protected (background) areas,
        # creating dense hatching patches in regions that should be clean white.
        zone     = (arr >= float(cfg.lightedge_tone_thr)) & highlight_mask
        src      = _gmag_smooth[zone] if zone.any() else _gmag_smooth
        thr_le   = float(np.quantile(src.reshape(-1), cfg.lightedge_pct / 100.0))
        mask_le  = zone & (_gmag_smooth >= thr_le)
        _save_mask_png(mask_le, "lightedge")
        _polys_le = filter_slivers(
            mask_to_polygons(mask_le, cfg, cfg.lightedge_min_area_px2, cfg.aux_simplify_tol_px * 0.8)
        )
        _log_poly_stats(_polys_le, "LightEdge")
        lightedge_d = run_aux("LightEdge",
                              mask_le,
                              lambda: _polys_le,
                              cfg.lightedge_spacing_mult, 18.0,
                              (0.41 + 3 * cfg.phase_step) % 1.0,
                              cfg.lightedge_cross, cfg.lightedge_budget, prefer_small=True)

    darkedge_d: List[str] = []
    if cfg.darkedge_enabled:
        # Dark areas have dense texture (hair, fabric) that produces many noise
        # blobs at the LightEdge blur radius.  Use a much heavier blur to match
        # HED's effective minimum feature size (~31px at 320px inference on 1799px).
        _gmag_dark_pil  = Image.fromarray(
            np.clip(gmag_arr * (255.0 / max(float(gmag_arr.max()), 1e-9)), 0, 255).astype(np.uint8)
        )
        _gmag_dark = np.asarray(
            _gmag_dark_pil.filter(ImageFilter.GaussianBlur(radius=_gmag_blur_r * 2.5))
        ).astype(np.float32) / 255.0
        zone     = arr <= float(cfg.darkedge_tone_thr)
        src      = _gmag_dark[zone] if zone.any() else _gmag_dark
        thr_de   = float(np.quantile(src.reshape(-1), cfg.darkedge_pct / 100.0))
        mask_de  = zone & (_gmag_dark >= thr_de)
        _save_mask_png(mask_de, "darkedge")
        _polys_de = filter_slivers(
            mask_to_polygons(mask_de, cfg, cfg.darkedge_min_area_px2, cfg.aux_simplify_tol_px * 0.85)
        )
        _log_poly_stats(_polys_de, "DarkEdge")
        darkedge_d = run_aux("DarkEdge",
                             mask_de,
                             lambda: _polys_de,
                             cfg.darkedge_spacing_mult, -12.0,
                             (0.57 + 4 * cfg.phase_step) % 1.0,
                             cfg.darkedge_cross, cfg.darkedge_budget, prefer_small=True)

    micro_d: List[str] = []
    if cfg.micro_enabled:
        micro_resp = (0.55 * edge_metric + 0.45 * lap).astype(np.float32)
        thr_mi     = float(np.quantile(micro_resp.reshape(-1), cfg.micro_pct / 100.0))
        mask_mi    = micro_resp >= thr_mi
        micro_d    = run_aux("MicroDetail",
                             mask_mi,
                             lambda: mask_to_polygons(mask_mi, cfg, cfg.micro_min_area_px2, max(0.15, cfg.aux_simplify_tol_px * 0.6)),
                             cfg.micro_spacing_mult, cfg.micro_angle_offset,
                             (0.73 + 5 * cfg.phase_step) % 1.0,
                             cfg.micro_cross, cfg.micro_budget,
                             prefer_small=cfg.micro_prefer_small_polys)

    # ── Iso-contour lines ────────────────────────────────────────────────────
    iso_d: List[str] = []
    if cfg.iso_contour_enabled:
        if status_cb:
            status_cb("Iso-contour lines…")
        iso_d = iso_contour_d_strings(
            arr_b, highlight_mask,
            n_levels     = cfg.iso_contour_levels,
            highlight_thr= float(cfg.highlight_protect_thr),
            min_length   = cfg.iso_contour_min_len,
            blur_px      = cfg.iso_contour_blur_px,
            gamma        = cfg.iso_contour_gamma,
            cap          = cfg.iso_contour_budget,
        )
        log(f"Iso-contours: {len(iso_d)} path elements "
            f"(levels={cfg.iso_contour_levels} blur={cfg.iso_contour_blur_px:.1f} "
            f"gamma={cfg.iso_contour_gamma:.2f})")

    # Hard reserve aux paths first
    max_paths = int(cfg.max_paths)
    aux_d: List[str] = []
    aux_d.extend(iso_d[:cfg.iso_contour_budget])
    aux_d.extend(face_d[:cfg.face_budget])
    aux_d.extend(detail_d[:cfg.detail_budget])
    aux_d.extend(lightedge_d[:cfg.lightedge_budget])
    aux_d.extend(darkedge_d[:cfg.darkedge_budget])
    aux_d.extend(micro_d[:cfg.micro_budget])

    aux_used          = len(aux_d)
    remaining_for_bands = max(0, max_paths - aux_used)
    all_d.extend(aux_d)
    log(f"\nAux total: {aux_used}  |  Remaining for bands: {remaining_for_bands}")

    # ── Band caps ────────────────────────────────────────────────────────────
    if cfg.use_recommended_caps and rec_perc is not None:
        band_caps = build_improved_caps(rec_perc, remaining_for_bands, cfg)
        log(f"Band caps: {band_caps}  (sum={sum(band_caps)})")
    else:
        rem, band_caps = remaining_for_bands, []
        for _ in range(cfg.bands):
            c = min(int(cfg.band_cap), rem)
            band_caps.append(c)
            rem -= c

    # ── Tone bands (polygon mode) OR raster passes ───────────────────────────
    band_results: List[Dict] = []

    if cfg.raster_hatch:
        # ── Raster hatch passes — no polygon finding ──────────────────────
        # Each pass scans at a different angle with a cumulative threshold:
        # dark pass covers only deep shadows; lighter passes add mid-tones.
        # Overlapping passes naturally crosshatch in dark areas.
        n_p   = max(1, cfg.raster_passes)
        thrs  = np.linspace(cfg.raster_dark_thr, cfg.raster_light_thr, n_p)
        astep = float(cfg.raster_angle_step)
        for pi in range(n_p):
            if progress_cb:
                progress_cb((pi + 1) / n_p)
            if remaining_for_bands <= 0:
                break
            t       = pi / max(1, n_p - 1)
            factor  = cfg.dark_mult + (cfg.light_mult - cfg.dark_mult) * (t ** cfg.ease_power)
            spacing = max(1.0, base_spacing * factor)
            angle   = _contour_blend(cfg.base_angle_deg + pi * astep)
            thr     = float(thrs[pi])
            if status_cb:
                status_cb(f"Raster pass {pi+1}/{n_p}…")
            d = raster_hatch_pass(
                arr_b, highlight_mask, bounds,
                angle_deg=angle, spacing=spacing,
                threshold=thr, stitch_jump=spacing * 1.5,
            )
            if d:
                all_d.append(d)
                remaining_for_bands -= 1
            log(f"Raster pass {pi+1}: thr={thr:.2f} spacing={spacing:.1f} "
                f"ang={angle:.1f}  paths={1 if d else 0}")
            band_results.append({
                "band": pi + 1, "on_px": 0, "polys": 0,
                "budget": 1, "paths": 1 if d else 0, "spacing": spacing,
            })
    else:
        # ── Polygon band hatching (original mode) ─────────────────────────
        for bi in range(cfg.bands):
            if progress_cb:
                progress_cb((bi + 1) / max(1, cfg.bands))
            if remaining_for_bands <= 0:
                break

            m      = band_masks[bi]
            _save_mask_png(m, f"band{bi+1:02d}")
            on_px  = int(m.sum())
            min_a  = band_min_area(bi)
            polys  = mask_to_polygons(m, cfg, min_area=min_a, simplify_tol=cfg.simplify_tolerance_px)
            _log_poly_stats(polys, f"Band{bi+1}")

            t      = bi / max(1, cfg.bands - 1)
            factor = cfg.dark_mult + (cfg.light_mult - cfg.dark_mult) * (t ** cfg.ease_power)
            spacing = max(1.0, base_spacing * factor)

            # Linear angle sweep dark→light
            _PHI  = 0.6180339887
            phase = (bi * _PHI) % 1.0
            angle = _contour_blend(cfg.base_angle_deg + cfg.angle_wobble_deg * (2.0 * t - 1.0))
            cross = cfg.cross_enabled and (bi < cfg.cross_dark_n)

            budget = min(int(band_caps[bi]), remaining_for_bands)
            if budget <= 0:
                continue

            log(f"Band {bi+1}: on_px={on_px} polys={len(polys)} "
                f"spacing={spacing:.1f} ang={angle:.1f} cross={cross} budget={budget}")

            band_d = hatch_polys_to_d_strings(
                polys, bounds, spacing=spacing, angle=angle, phase=phase,
                cross=cross, stitch_jump=spacing * 1.2, cap=budget,
            )
            all_d.extend(band_d)
            remaining_for_bands -= len(band_d)
            log(f"Band {bi+1}: +{len(band_d)}  total={len(all_d)}")
            band_results.append({
                "band":    bi + 1,
                "on_px":   on_px,
                "polys":   len(polys),
                "budget":  budget,
                "paths":   len(band_d),
                "spacing": spacing,
            })

            if len(all_d) >= max_paths:
                all_d = all_d[:max_paths]
                break

    # ── Write SVG ────────────────────────────────────────────────────────────
    if status_cb:
        status_cb("Writing SVG…")
    write_svg(out_svg_path, all_d, w, h, cfg, status_cb=status_cb)

    # ── Render preview ───────────────────────────────────────────────────────
    preview_img = None
    try:
        if status_cb:
            status_cb("Rendering preview…")
        preview_img = render_preview(all_d, w, h)
        _prev_path = os.path.splitext(out_svg_path)[0] + "_preview.png"
        preview_img.save(_prev_path)
        log(f"Preview saved: {_prev_path}")
    except Exception as _prev_err:
        if log_cb:
            log_cb(f"Preview render error: {_prev_err}")

    elapsed   = time.time() - t0
    pen_lifts = sum(d.count(" M ") + (1 if d else 0) for d in all_d)

    return {
        "paths":            len(all_d),
        "pen_lifts_est":    pen_lifts,
        "elapsed_sec":      elapsed,
        "work_w":           w,
        "work_h":           h,
        "skimage_ok":       _SKIMAGE_OK,
        "aux_used":         aux_used,
        "remaining_for_bands": max(0, max_paths - aux_used),
        "preview_image":    preview_img,
        "protected_pct":    protected_pct,
        "band_results":     band_results,
        "band_caps":        band_caps,
        "total_px":         int(arr.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run diagnostics + recommendations
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_run(stats: Dict, cfg: HatchConfig) -> str:
    """
    Analyse pipeline stats and return a human-readable diagnostic report
    with specific setting recommendations where the output may be sub-optimal.
    """
    issues:  List[str] = []
    tips:    List[str] = []

    protected_pct   = stats.get("protected_pct", 0.0)
    aux_used        = stats.get("aux_used", 0)
    total_paths     = stats.get("paths", 0)
    max_paths       = cfg.max_paths
    band_results    = stats.get("band_results", [])
    band_caps       = stats.get("band_caps", [])
    remaining_bands = stats.get("remaining_for_bands", max_paths)

    # ── Highlight protection ─────────────────────────────────────────────────
    if protected_pct > 30.0:
        issues.append(
            f"BLANK AREAS — Highlight Protect ({cfg.highlight_protect_thr:.2f}) is "
            f"excluding {protected_pct:.0f}% of pixels from hatching. "
            f"Raise it to {min(cfg.highlight_protect_thr + 0.06, 1.0):.2f}–"
            f"{min(cfg.highlight_protect_thr + 0.10, 1.0):.2f} to fill in those regions."
        )
    elif protected_pct > 18.0:
        tips.append(
            f"Highlight Protect ({cfg.highlight_protect_thr:.2f}) is shielding "
            f"{protected_pct:.0f}% of pixels — if light areas look sparse consider "
            f"raising it slightly (try {min(cfg.highlight_protect_thr + 0.04, 1.0):.2f})."
        )

    # ── Budget exhaustion ─────────────────────────────────────────────────────
    aux_fraction = aux_used / max(max_paths, 1)
    if aux_fraction > 0.55:
        issues.append(
            f"BUDGET STARVED — Aux layers consumed {aux_used}/{max_paths} paths "
            f"({aux_fraction*100:.0f}%), leaving tone bands under-filled. "
            f"Either raise 'Path Ceiling' to {int(max_paths * 1.5)} or disable "
            f"Face/Micro detail in the Advanced tab."
        )
    elif aux_fraction > 0.40:
        tips.append(
            f"Aux layers used {aux_fraction*100:.0f}% of the path budget. "
            f"If bands look thin, raise 'Path Ceiling' to {int(max_paths * 1.3)}."
        )

    # ── Bands that produced zero paths ───────────────────────────────────────
    zero_bands = [r["band"] for r in band_results if r["paths"] == 0 and r["on_px"] > 0]
    if zero_bands:
        issues.append(
            f"MISSING BANDS — Band(s) {zero_bands} had pixels but produced 0 paths "
            f"(budget hit zero). Raise 'Path Ceiling' from {max_paths} to "
            f"{int(max_paths * 1.4)} to recover them."
        )

    # ── Bands with many polygons but few paths (cap hit) ─────────────────────
    capped_bands = [
        r for r in band_results
        if r["polys"] > 0 and r["paths"] > 0 and r["budget"] > 0
        and r["paths"] >= r["budget"] * 0.95  # hit the cap
    ]
    if capped_bands:
        tips.append(
            f"Band(s) {[r['band'] for r in capped_bands]} hit their path cap — "
            f"some polygon regions were skipped. Raise 'Path Ceiling' or lower "
            f"'Min Area Light' in Advanced to reduce fragmentation."
        )

    # ── Light bands with no polygons (min_area too high) ─────────────────────
    n = len(band_results)
    light_empty = [
        r for r in band_results[max(0, n - 3):]   # top 3 lightest bands
        if r["on_px"] > 0 and r["polys"] == 0
    ]
    if light_empty:
        issues.append(
            f"LIGHT BANDS EMPTY — Band(s) {[r['band'] for r in light_empty]} have pixels "
            f"but zero polygons (all regions smaller than Min Area Light = "
            f"{cfg.min_area_light_px2:.0f} px²). Lower 'Min Area Light' in Advanced "
            f"(try {max(50, cfg.min_area_light_px2 // 4):.0f})."
        )

    # ── Hatching very dense in dark bands ────────────────────────────────────
    dark_bands = [r for r in band_results[:2] if r["paths"] > 0]
    if dark_bands and cfg.dark_mult < 0.20:
        tips.append(
            f"Dark spacing mult is very low ({cfg.dark_mult:.2f}) — dark areas will "
            f"be extremely dense and may bleed or tear on physical media. "
            f"Try raising 'Dark Spacing Mult' to 0.25–0.35."
        )

    # ── Very sparse light bands ───────────────────────────────────────────────
    n = len(band_results)
    if band_results:
        lightest = band_results[-1]
        light_spacing = lightest.get("spacing", 0)
        if light_spacing > 28.0:
            tips.append(
                f"Lightest band spacing is {light_spacing:.1f}px — light/skin-tone "
                f"areas will have very visible gaps between lines. Lower 'Light "
                f"Spacing Mult' (currently {cfg.light_mult:.1f}) to 2.0–2.5."
            )

    # ── Sparse light bands (few polygons) ────────────────────────────────────
    light_sparse = [
        r for r in band_results[max(0, n - 3):]
        if r["paths"] > 0 and r["polys"] <= 8
    ]
    if light_sparse:
        tips.append(
            f"Light band(s) {[r['band'] for r in light_sparse]} have very few "
            f"polygons ({[r['polys'] for r in light_sparse]}) — large tonal regions "
            f"with minimal line coverage. Increase 'Tone Bands' in Advanced "
            f"(currently {cfg.bands}, try {cfg.bands + 4}) to subdivide them."
        )

    # ── Budget heavily underutilised ──────────────────────────────────────────
    budget_used_pct = 100.0 * total_paths / max(max_paths, 1)
    if budget_used_pct < 45.0 and total_paths > 50:
        tips.append(
            f"Only {budget_used_pct:.0f}% of the path budget was used "
            f"({total_paths}/{max_paths}). The image has large connected tonal "
            f"regions. Try increasing 'Tone Bands' in Advanced for finer "
            f"tonal gradations, or lower 'Line Spacing' to add more detail."
        )

    # ── Overall path count very low ───────────────────────────────────────────
    coverage_bands = len([r for r in band_results if r["paths"] > 0])
    if total_paths < 200 and cfg.bands > 2:
        issues.append(
            f"VERY FEW PATHS ({total_paths}) — output will look sparse. "
            f"Check that an image is loaded and try raising 'Path Ceiling' or "
            f"lowering 'Line Spacing'."
        )
    elif coverage_bands < cfg.bands // 2:
        tips.append(
            f"Only {coverage_bands}/{cfg.bands} bands produced paths. "
            f"Raise 'Path Ceiling' to give more budget to the remaining bands."
        )

    # ── Slow run warning ──────────────────────────────────────────────────────
    elapsed = stats.get("elapsed_sec", 0)
    if elapsed > 120:
        tips.append(
            f"Run took {elapsed:.0f}s. If HED edge detection is enabled, try "
            f"lowering 'HED Max Px' in Controls (currently {cfg.hed_max_px}px, "
            f"try 256) — HED is the main CPU bottleneck. Disabling HED drops "
            f"it to gradient-based edges which are near-instant."
        )

    # ── scikit-image missing ──────────────────────────────────────────────────
    if not stats.get("skimage_ok", True):
        tips.append(
            "scikit-image is not installed — polygon extraction is using a slower "
            "fallback that may miss fine detail. "
            "Install it with:  pip install scikit-image"
        )

    # ── Build report ──────────────────────────────────────────────────────────
    lines = ["", "─" * 54, "  DIAGNOSTIC REPORT"]

    if not issues and not tips:
        lines.append("  ✓ Run looks healthy — no obvious problems detected.")
    else:
        if issues:
            lines.append(f"  {len(issues)} issue(s) found:")
            for i, msg in enumerate(issues, 1):
                lines.append(f"  [{i}] {msg}")
        if tips:
            lines.append(f"  {len(tips)} suggestion(s):")
            for i, msg in enumerate(tips, 1):
                lines.append(f"  ({i}) {msg}")

    lines.append("─" * 54)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CMY(K) halftone-style pipeline
# ─────────────────────────────────────────────────────────────────────────────

def rgb_to_cmyk(img_arr: np.ndarray,
                ucr_amount: float = 1.0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert float32 RGB (H, W, 3) in [0, 1] to four float32 channel arrays,
    each in [0, 1].

    ucr_amount controls under-colour removal:
      0.0 → no K plate; C/M/Y carry all ink
      1.0 → maximum K extraction; C/M/Y carry only the chromatic remainder
    """
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    c_full = 1.0 - r
    m_full = 1.0 - g
    y_full = 1.0 - b

    k = np.minimum(np.minimum(c_full, m_full), y_full)  # raw K

    if ucr_amount <= 0.0:
        return (c_full.astype(np.float32),
                m_full.astype(np.float32),
                y_full.astype(np.float32),
                np.zeros_like(k, dtype=np.float32))

    k_used = k * ucr_amount
    denom  = 1.0 - k_used + 1e-9
    c = np.clip((c_full - k_used) / denom, 0.0, 1.0)
    m = np.clip((m_full - k_used) / denom, 0.0, 1.0)
    y = np.clip((y_full - k_used) / denom, 0.0, 1.0)

    # Blend between full-CMY and UCR-CMY according to ucr_amount
    if ucr_amount < 1.0:
        c = c_full * (1.0 - ucr_amount) + c * ucr_amount
        m = m_full * (1.0 - ucr_amount) + m * ucr_amount
        y = y_full * (1.0 - ucr_amount) + y * ucr_amount

    return (c.astype(np.float32),
            m.astype(np.float32),
            y.astype(np.float32),
            k_used.astype(np.float32))


def _channel_density_to_spacing(density: float, cmy_cfg: "CMYConfig") -> float:
    """
    Map a channel density value [0, 1] to a line spacing in px.
    Uses exponential mapping so spacing feels perceptually linear:
      density = 1.0  →  min_spacing_px  (maximum ink)
      density = thr  →  max_spacing_px  (minimum ink)
    """
    thr = cmy_cfg.min_channel_thr
    t   = max(0.0, (density - thr) / max(1.0 - thr, 1e-9))  # 0→1
    # Exponential interpolation: log(max) → log(min) as t goes 0 → 1
    log_sp = math.log(cmy_cfg.max_spacing_px) * (1.0 - t) + \
             math.log(max(cmy_cfg.min_spacing_px, 0.5)) * t
    return max(1.0, math.exp(log_sp))


def _hatch_channel(channel: np.ndarray, angle_deg: float,
                   cfg: "HatchConfig", cmy_cfg: "CMYConfig",
                   bounds: Tuple[float, float, float, float],
                   cross: bool = False,
                   label: str = "") -> List[str]:
    """
    Hatch a single CMY(K) channel array (float32 H×W in [0,1]).
    Divides the channel into density_bands uniform quantile bands;
    each band gets hatching at the spacing that matches its ink density.
    Returns a flat list of SVG path d-strings.
    """
    thr     = cmy_cfg.min_channel_thr
    n_bands = cmy_cfg.density_bands
    active  = channel >= thr

    if not active.any():
        return []

    # Quantile band edges over the active pixels only
    active_vals = channel[active]
    band_edges  = np.quantile(active_vals,
                              np.linspace(0.0, 1.0, n_bands + 1)).tolist()
    # Ensure edges are strictly increasing
    for i in range(1, len(band_edges)):
        if band_edges[i] <= band_edges[i - 1]:
            band_edges[i] = band_edges[i - 1] + 1e-6
    band_edges[0] = thr   # floor at threshold

    cap_per_band = max(1, cmy_cfg.max_paths_per_channel // n_bands + 1)
    all_d: List[str] = []

    for bi in range(n_bands):
        lo = band_edges[bi]
        hi = band_edges[bi + 1]
        mid_density = (lo + hi) * 0.5

        mask = active & (channel >= lo) & (channel < hi)
        if mask.sum() == 0:
            continue

        polys = mask_to_polygons(mask, cfg,
                                 min_area=cmy_cfg.min_area_px2,
                                 simplify_tol=cmy_cfg.simplify_tol_px)
        if not polys:
            continue

        spacing = _channel_density_to_spacing(mid_density, cmy_cfg)
        phase   = (bi / max(n_bands, 1))   # stagger bands to avoid line-up

        d_strings = hatch_polys_to_d_strings(
            polys, bounds,
            spacing=spacing, angle=angle_deg, phase=phase,
            cross=cross, stitch_jump=spacing * 1.2,
            cap=cap_per_band,
        )
        all_d.extend(d_strings)

        if len(all_d) >= cmy_cfg.max_paths_per_channel:
            all_d = all_d[:cmy_cfg.max_paths_per_channel]
            break

    return all_d


def render_preview_colored(d_strings: List[str], img_w: int, img_h: int,
                           color_rgb: Tuple[int,int,int],
                           max_dim: int = 1000) -> Image.Image:
    """Render hatch d-strings in color_rgb onto a transparent RGBA canvas."""
    scale = min(max_dim / max(img_w, 1), max_dim / max(img_h, 1))
    pw    = max(1, int(img_w * scale))
    ph    = max(1, int(img_h * scale))
    im    = Image.new("RGBA", (pw, ph), (255, 255, 255, 0))
    draw  = ImageDraw.Draw(im)
    r, g, b = color_rgb
    fill = (r, g, b, 220)
    for d in d_strings:
        tokens  = d.split()
        current = None
        i       = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "M":
                i += 1
                if i < len(tokens):
                    x, y    = map(float, tokens[i].split(","))
                    current = (x * scale, y * scale)
                    i += 1
            elif tok == "L":
                i += 1
                if i < len(tokens):
                    x, y = map(float, tokens[i].split(","))
                    pt   = (x * scale, y * scale)
                    if current is not None:
                        draw.line([current, pt], fill=fill, width=1)
                    current = pt
                    i += 1
            else:
                i += 1
    return im


def composite_color_preview(layers: List[Tuple[Image.Image, Tuple[int,int,int]]],
                             img_w: int, img_h: int,
                             max_dim: int = 1000) -> Image.Image:
    """Paste all RGBA color-layer previews onto a white background."""
    scale  = min(max_dim / max(img_w, 1), max_dim / max(img_h, 1))
    pw     = max(1, int(img_w * scale))
    ph     = max(1, int(img_h * scale))
    canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
    for layer_im, _ in layers:
        lw, lh = layer_im.size
        if (lw, lh) != (pw, ph):
            layer_im = layer_im.resize((pw, ph), Image.LANCZOS)
        canvas.paste(layer_im, mask=layer_im.split()[3])
    return canvas


def cmy_hatch_pipeline(png_path: str, out_dir: str,
                        cfg: "HatchConfig", cmy_cfg: "CMYConfig",
                        progress_cb=None, status_cb=None,
                        log_cb=None) -> Dict:
    """
    CMY(K) halftone-style pipeline.

    Converts the image to CMYK with optional under-colour removal, then
    hatches each channel independently using density-modulated line spacing
    at the traditional halftone screen angles.  Overlapping hatching in
    C / M / Y / K optically mixes to approximate the original image colours.
    """
    t0 = time.time()

    def log(s):
        if log_cb:
            log_cb(s)

    # ── Load + resize ─────────────────────────────────────────────────────────
    if status_cb:
        status_cb("Loading image…")
    im = Image.open(png_path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    w, h = im.size
    if max(w, h) > cfg.resize_long_edge_px:
        sc = cfg.resize_long_edge_px / float(max(w, h))
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.LANCZOS)
        w, h = im.size
    log(f"Image: {w}×{h}px")

    # ── RGB → CMY(K) ──────────────────────────────────────────────────────────
    if status_cb:
        status_cb("Converting to CMY(K)…")
    img_arr = np.array(im, dtype=np.float32) / 255.0
    c_arr, m_arr, y_arr, k_arr = rgb_to_cmyk(img_arr,
                                              ucr_amount=cmy_cfg.ucr_amount
                                              if cmy_cfg.use_black else 0.0)

    thr = cmy_cfg.min_channel_thr
    log(f"Channel coverage (≥{thr:.2f}):")
    for name, ch in [("C", c_arr), ("M", m_arr), ("Y", y_arr), ("K", k_arr)]:
        pct = 100.0 * float((ch >= thr).mean())
        log(f"  {name}: {pct:.1f}%  max={ch.max():.3f}  mean={ch[ch>=thr].mean():.3f}"
            if (ch >= thr).any() else f"  {name}: 0.0% (below threshold)")

    os.makedirs(out_dir, exist_ok=True)
    bounds = (0.0, 0.0, float(w), float(h))

    # Channel definitions: (name, array, angle, RGB ink colour, hex stroke, crosshatch)
    channels = [
        ("Cyan",    c_arr, cmy_cfg.cyan_angle_deg,    (0,   200, 220), "#00c8dc", False),
        ("Magenta", m_arr, cmy_cfg.magenta_angle_deg, (220,  0,  180), "#dc00b4", False),
        ("Yellow",  y_arr, cmy_cfg.yellow_angle_deg,  (220, 200,   0), "#dcc800", False),
    ]
    if cmy_cfg.use_black:
        channels.append(
            ("Black", k_arr, cmy_cfg.black_angle_deg, (20, 20, 20), "#141414",
             cmy_cfg.black_cross)
        )

    preview_layers: List[Tuple[Image.Image, Tuple[int, int, int]]] = []
    results: List[Dict]                = []
    all_layers_d: List[Tuple[str, List[str]]] = []
    total_paths = 0

    for ci, (name, ch_arr, angle, rgb_col, hex_col, cross) in enumerate(channels):
        if progress_cb:
            progress_cb(ci / len(channels))
        if status_cb:
            status_cb(f"Hatching {name} channel…")

        if not (ch_arr >= thr).any():
            log(f"\n{name}: skipped (no pixels above threshold)")
            continue

        log(f"\n{name}  angle={angle:.0f}°  cross={cross}")

        d_strings = _hatch_channel(ch_arr, angle, cfg, cmy_cfg, bounds,
                                   cross=cross, label=name)
        log(f"  Paths: {len(d_strings)}")

        if not d_strings:
            continue

        # ── Per-channel SVG ───────────────────────────────────────────────────
        fname    = f"{name.lower()}.svg"
        out_path = os.path.join(out_dir, fname)
        _w_in, _h_in = _svg_physical_size(w, h, cfg)
        dwg = svgwrite.Drawing(
            out_path,
            size=(f"{_w_in:.4f}in", f"{_h_in:.4f}in"),
            viewBox=f"0 0 {w} {h}",
            profile="tiny",
        )
        for d in d_strings:
            if d:
                dwg.add(dwg.path(
                    d=d, fill="none", stroke=hex_col,
                    stroke_width=cfg.stroke_width,
                    stroke_linecap=cfg.stroke_linecap,
                    stroke_linejoin=cfg.stroke_linejoin,
                ))
        dwg.save()
        log(f"  Saved: {fname}")

        results.append({"channel": name, "paths": len(d_strings), "file": out_path})
        total_paths += len(d_strings)
        all_layers_d.append((hex_col, d_strings))

        try:
            layer_prev = render_preview_colored(d_strings, w, h, rgb_col)
            preview_layers.append((layer_prev, rgb_col))
        except Exception as e:
            log(f"  Preview error: {e}")

    if progress_cb:
        progress_cb(1.0)

    # ── Combined SVG ──────────────────────────────────────────────────────────
    if all_layers_d:
        try:
            base_name     = os.path.splitext(os.path.basename(png_path))[0]
            combined_path = os.path.join(out_dir, f"{base_name}_cmy_combined.svg")
            _w_in, _h_in  = _svg_physical_size(w, h, cfg)
            dwg = svgwrite.Drawing(
                combined_path,
                size=(f"{_w_in:.4f}in", f"{_h_in:.4f}in"),
                viewBox=f"0 0 {w} {h}",
                profile="tiny",
            )
            for hex_col, d_strings in all_layers_d:
                for d in d_strings:
                    if d:
                        dwg.add(dwg.path(
                            d=d, fill="none", stroke=hex_col,
                            stroke_width=cfg.stroke_width,
                            stroke_linecap=cfg.stroke_linecap,
                            stroke_linejoin=cfg.stroke_linejoin,
                        ))
            dwg.save()
            log(f"Combined: {os.path.basename(combined_path)}")
        except Exception as e:
            log(f"Combined SVG error: {e}")

    # ── Composite preview ─────────────────────────────────────────────────────
    preview_img = None
    try:
        if preview_layers:
            preview_img = composite_color_preview(preview_layers, w, h)
    except Exception as e:
        log(f"Composite preview error: {e}")

    elapsed = time.time() - t0
    log(f"\nCMY pipeline done in {elapsed:.1f}s  |  Channels: {len(results)}  "
        f"Total paths: {total_paths}")
    log(f"Output: {out_dir}")

    return {
        "layers":        results,
        "total_paths":   total_paths,
        "elapsed_sec":   elapsed,
        "work_w":        w,
        "work_h":        h,
        "preview_image": preview_img,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dear PyGui application
# ─────────────────────────────────────────────────────────────────────────────

_PREVIEW_W = 700
_PREVIEW_H = 560

# Flat RGBA float32 list for a dark-gray placeholder texture
_PLACEHOLDER_DATA = [0.14, 0.14, 0.14, 1.0] * (_PREVIEW_W * _PREVIEW_H)


class HatchApp:

    PRESETS = {
        "Fine":   dict(spacing=7.0,  depth=2.0, dark_mult=0.28, light_mult=4.0),
        "Medium": dict(spacing=10.0, depth=2.2, dark_mult=0.32, light_mult=3.5),
        "Bold":   dict(spacing=14.0, depth=2.4, dark_mult=0.38, light_mult=2.8),
    }

    def __init__(self):
        self.png_path:     Optional[str]  = None
        self.out_svg_path: Optional[str]  = None
        self.cfg       = HatchConfig()
        self.cmy_cfg = CMYConfig()
        self._running   = False
        self._last_stats: Optional[Dict] = None   # saved after each run

        # Buffered log lines (so Save Report can write them to a file)
        self._log_lines: List[str] = []

        # Thread-safe queues for worker → main-thread communication
        self._log_q:      queue.Queue = queue.Queue()
        self._status_q:   queue.Queue = queue.Queue()
        self._progress_q: queue.Queue = queue.Queue()
        self._preview_q:  queue.Queue = queue.Queue()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _frow(self, tbl, label, tag, default, hint="",
              typ="f", fmt="%.2f", step=0.1, lo=0.0, hi=100.0):
        """Add a labeled input row to a form table."""
        with dpg.table_row(parent=tbl):
            dpg.add_text(label)
            if typ == "f":
                dpg.add_input_float(
                    label=f"##{tag}", tag=tag, default_value=float(default),
                    width=-1, min_value=float(lo), max_value=float(hi),
                    step=float(step), format=fmt,
                )
            else:
                dpg.add_input_int(
                    label=f"##{tag}", tag=tag, default_value=int(default),
                    width=-1, min_value=int(lo), max_value=int(hi), step=1,
                )
            if hint:
                dpg.add_text(hint, color=(120, 120, 120, 200))

    def _make_form_table(self, parent):
        return dpg.add_table(
            header_row=False, parent=parent,
            borders_innerV=False, borders_outerV=False,
            policy=dpg.mvTable_SizingFixedFit,
        )

    def _apply_preset(self, name: str):
        p = self.PRESETS[name]
        try:
            dpg.set_value("spacing",    float(p["spacing"]))
            dpg.set_value("depth",      float(p["depth"]))
            dpg.set_value("dark_mult",  float(p["dark_mult"]))
            dpg.set_value("light_mult", float(p["light_mult"]))
            self._append_log(
                f"Preset '{name}': spacing={p['spacing']} depth={p['depth']} "
                f"dark={p['dark_mult']} light={p['light_mult']}"
            )
        except Exception as e:
            self._append_log(f"Preset error: {e}")

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_theme(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                C = dpg.add_theme_color
                S = dpg.add_theme_style
                C(dpg.mvThemeCol_WindowBg,          ( 20,  20,  22, 255))
                C(dpg.mvThemeCol_ChildBg,            ( 26,  26,  30, 255))
                C(dpg.mvThemeCol_FrameBg,            ( 42,  42,  48, 255))
                C(dpg.mvThemeCol_FrameBgHovered,     ( 58,  58,  66, 255))
                C(dpg.mvThemeCol_FrameBgActive,      ( 70,  70,  80, 255))
                C(dpg.mvThemeCol_Button,             ( 55,  95, 160, 255))
                C(dpg.mvThemeCol_ButtonHovered,      ( 75, 120, 200, 255))
                C(dpg.mvThemeCol_ButtonActive,       ( 45,  80, 140, 255))
                C(dpg.mvThemeCol_Tab,                ( 38,  38,  44, 255))
                C(dpg.mvThemeCol_TabHovered,         ( 60,  60,  72, 255))
                C(dpg.mvThemeCol_TabActive,          ( 55,  95, 160, 255))
                C(dpg.mvThemeCol_Header,             ( 55,  95, 160, 160))
                C(dpg.mvThemeCol_HeaderHovered,      ( 75, 120, 200, 180))
                C(dpg.mvThemeCol_CheckMark,          (120, 180, 255, 255))
                C(dpg.mvThemeCol_SliderGrab,         ( 75, 120, 200, 255))
                C(dpg.mvThemeCol_TitleBg,            ( 20,  20,  22, 255))
                C(dpg.mvThemeCol_TitleBgActive,      ( 38,  70, 130, 255))
                C(dpg.mvThemeCol_ScrollbarBg,        ( 20,  20,  22, 255))
                C(dpg.mvThemeCol_ScrollbarGrab,      ( 60,  60,  70, 255))
                C(dpg.mvThemeCol_PopupBg,            ( 30,  30,  36, 255))
                S(dpg.mvStyleVar_FrameRounding,      5)
                S(dpg.mvStyleVar_WindowRounding,     6)
                S(dpg.mvStyleVar_ChildRounding,      5)
                S(dpg.mvStyleVar_TabRounding,        5)
                S(dpg.mvStyleVar_FramePadding,       8, 5)
                S(dpg.mvStyleVar_ItemSpacing,        8, 6)
                S(dpg.mvStyleVar_WindowPadding,     10, 10)
        dpg.bind_theme(theme)

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _build(self):
        cfg = self.cfg

        # Pre-register preview texture
        with dpg.texture_registry():
            dpg.add_raw_texture(
                _PREVIEW_W, _PREVIEW_H,
                _PLACEHOLDER_DATA,
                tag="preview_tex",
                format=dpg.mvFormat_Float_rgba,
            )

        # File dialog
        with dpg.file_dialog(
            directory_selector=False, show=False,
            callback=self._on_file_selected,
            cancel_callback=lambda s, a, u=None: None,
            tag="file_dialog", width=720, height=480,
        ):
            dpg.add_file_extension(
                "Image Files{.png,.jpg,.jpeg,.webp,.bmp}",
                color=(150, 230, 150, 255),
            )
            dpg.add_file_extension(".*", color=(180, 180, 180, 200))

        # ── Primary window ────────────────────────────────────────────────────
        with dpg.window(tag="main_win", no_title_bar=True,
                        no_resize=True, no_move=True, no_close=True,
                        no_scrollbar=True):

            # File row
            with dpg.group(horizontal=True):
                dpg.add_button(label=" Choose Image… ", tag="btn_file",
                               callback=lambda: dpg.show_item("file_dialog"))
                dpg.add_text("  No image selected.", tag="lbl_file")
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Preset row
            with dpg.group(horizontal=True):
                dpg.add_text("Preset:", color=(170, 170, 180, 255))
                dpg.add_spacer(width=6)
                for name in ("Fine", "Medium", "Bold"):
                    dpg.add_button(
                        label=f"  {name}  ",
                        callback=lambda s, a, u: self._apply_preset(u),
                        user_data=name,
                    )
                    dpg.add_spacer(width=4)
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # ── Two-column content area ───────────────────────────────────────
            with dpg.group(horizontal=True):

                # ── LEFT PANEL: Controls ─────────────────────────────────────
                with dpg.child_window(tag="left_panel", width=440,
                                      height=-220, border=False, no_scrollbar=False):

                    with dpg.tab_bar():

                        # ── Controls tab ──────────────────────────────────────
                        with dpg.tab(label=" Controls "):
                            dpg.add_spacer(height=6)
                            with dpg.table(
                                header_row=False, tag="ctrl_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=165, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()

                                self._frow("ctrl_form", "Line Spacing (px)",
                                           "spacing", cfg.base_spacing_px,
                                           "stroke density",
                                           step=0.5, lo=1, hi=50)
                                self._frow("ctrl_form", "Shadow Depth",
                                           "depth", cfg.ease_power,
                                           "1=flat  3=dramatic",
                                           step=0.1, lo=0.5, hi=5)
                                self._frow("ctrl_form", "Hatch Angle (°)",
                                           "angle", cfg.base_angle_deg,
                                           "",
                                           step=1, lo=-90, hi=90)
                                self._frow("ctrl_form", "Tone Gamma",
                                           "gamma", cfg.band_gamma,
                                           ">1 pushes shadows darker",
                                           step=0.05, lo=0.1, hi=3)
                                self._frow("ctrl_form", "Edge Sensitivity",
                                           "edge_sens", cfg.detail_pct,
                                           "lower = more edge strokes",
                                           step=0.5, lo=50, hi=99)
                                self._frow("ctrl_form", "Local Contrast",
                                           "lc_strength", cfg.local_contrast_strength,
                                           "0=off  0.5=moderate  1=strong",
                                           step=0.05, lo=0, hi=1.5)
                                self._frow("ctrl_form", "Highlight Protect",
                                           "hl_thr", cfg.highlight_protect_thr,
                                           "0.92 = top 8% stays white",
                                           fmt="%.3f", step=0.01, lo=0.5, hi=1.0)
                                self._frow("ctrl_form", "Path Ceiling",
                                           "max_paths", cfg.max_paths,
                                           "Cricut polygon limit",
                                           typ="i", lo=100, hi=20000)

                            dpg.add_spacer(height=8)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label="Crosshatch shadows",
                                    tag="cross_enabled",
                                    default_value=cfg.cross_enabled,
                                )
                                dpg.add_spacer(width=6)
                                dpg.add_text("darkest")
                                dpg.add_input_int(
                                    label="##cross_n", tag="cross_n",
                                    default_value=cfg.cross_dark_n,
                                    width=55, min_value=0, max_value=8, step=1,
                                )
                                dpg.add_text("bands")
                            dpg.add_spacer(height=4)
                            dpg.add_checkbox(
                                label="Invert image",
                                tag="invert",
                                default_value=cfg.invert,
                            )

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_spacer(height=6)
                            dpg.add_text("HED Edge Detection",
                                         color=(170, 170, 180, 255))
                            dpg.add_spacer(height=4)

                            _hed_status = ("✓ Model ready" if hed_model_ready()
                                           else "✗ Model not downloaded")
                            _hed_col    = ((100, 220, 130, 255) if hed_model_ready()
                                           else (220, 130, 80, 255))
                            dpg.add_text(_hed_status, tag="lbl_hed_status",
                                         color=_hed_col)
                            dpg.add_spacer(height=4)

                            dpg.add_checkbox(
                                label="Use HED edges (clean hand-drawn contours)",
                                tag="hed_enabled",
                                default_value=cfg.hed_enabled,
                                enabled=hed_model_ready(),
                            )
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_text("HED Max Px:",
                                             color=(150, 150, 160, 220))
                                dpg.add_spacer(width=6)
                                dpg.add_input_int(
                                    label="##hed_max_px", tag="hed_max_px",
                                    default_value=cfg.hed_max_px,
                                    width=80, min_value=128, max_value=1024, step=32,
                                )
                                dpg.add_spacer(width=6)
                                dpg.add_text("(256=fast  512=quality)",
                                             color=(110, 110, 120, 200))
                            dpg.add_spacer(height=6)
                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="  Download HED Model (~56 MB)  ",
                                    tag="btn_hed_download",
                                    callback=self.on_download_hed,
                                )
                            dpg.add_spacer(height=2)
                            dpg.add_text(
                                "Requires: pip install opencv-python",
                                color=(110, 110, 120, 200))

                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_spacer(height=6)
                            dpg.add_text("Contour-Following Hatching",
                                         color=(170, 170, 180, 255))
                            dpg.add_spacer(height=4)
                            dpg.add_text(
                                "Lines follow local iso-brightness curves\n"
                                "(engraving / pen-and-ink form-following style).",
                                color=(110, 110, 120, 200))
                            dpg.add_spacer(height=6)
                            dpg.add_checkbox(
                                label="Contour-following lines",
                                tag="contour_hatch",
                                default_value=cfg.contour_hatch,
                            )
                            dpg.add_spacer(height=4)
                            with dpg.table(
                                header_row=False, tag="contour_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=150, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()
                                self._frow("contour_form", "Contour Strength",
                                           "contour_strength",
                                           cfg.contour_hatch_strength,
                                           "0=global angle  1=full contour",
                                           fmt="%.2f", step=0.05, lo=0.0, hi=1.0)
                                self._frow("contour_form", "Flow Blur (px)",
                                           "contour_blur",
                                           cfg.contour_hatch_blur,
                                           "smooth gradient field (8–20 recommended)",
                                           step=1.0, lo=0.0, hi=40.0)

                        # ── Advanced tab ──────────────────────────────────────
                        with dpg.tab(label=" Advanced "):
                            dpg.add_spacer(height=6)
                            with dpg.table(
                                header_row=False, tag="adv_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=165, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()

                                self._frow("adv_form", "Dark Spacing Mult",
                                           "dark_mult", cfg.dark_mult,
                                           "density of darkest band",
                                           fmt="%.3f", step=0.01, lo=0.05, hi=1.0)
                                self._frow("adv_form", "Light Spacing Mult",
                                           "light_mult", cfg.light_mult,
                                           "sparsity of lightest band",
                                           step=0.1, lo=0.1, hi=8.0)
                                self._frow("adv_form", "Angle Sweep (°)",
                                           "angle_wobble", cfg.angle_wobble_deg,
                                           "±° sweep dark→light band (0=all same, 30=wide variety)",
                                           step=1.0, lo=0.0, hi=45.0)
                                self._frow("adv_form", "Tone Bands",
                                           "bands", cfg.bands,
                                           "tonal divisions (4–16)",
                                           typ="i", lo=2, hi=20)
                                self._frow("adv_form", "Min Area Dark (px²)",
                                           "min_dark", cfg.min_area_dark_px2,
                                           "kills dark noise fragments",
                                           step=5, lo=0, hi=500)
                                self._frow("adv_form", "Min Area Light (px²)",
                                           "min_light", cfg.min_area_light_px2,
                                           "keeps light areas noise-free",
                                           step=50, lo=0, hi=10000)
                                self._frow("adv_form", "Global Spacing Mult",
                                           "gspacing", cfg.global_spacing_mult,
                                           "scales all spacing uniformly",
                                           step=0.05, lo=0.1, hi=3.0)

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=6)
                            dpg.add_text("Raster Hatch Passes",
                                         color=(170, 170, 180, 255))
                            dpg.add_spacer(height=4)
                            dpg.add_text(
                                "Scans parallel lines directly across the image\n"
                                "— no polygon finding, no seam artifacts.\n"
                                "Multiple passes at different angles crosshatch\n"
                                "dark areas naturally.",
                                color=(110, 110, 120, 200))
                            dpg.add_spacer(height=6)
                            dpg.add_checkbox(
                                label="Raster hatch (replaces polygon bands)",
                                tag="raster_hatch",
                                default_value=cfg.raster_hatch,
                            )
                            dpg.add_spacer(height=4)
                            with dpg.table(
                                header_row=False, tag="raster_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=165, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()
                                self._frow("raster_form", "Passes",
                                           "raster_passes", cfg.raster_passes,
                                           "number of angle passes (2–4 typical)",
                                           typ="i", lo=1, hi=8)
                                self._frow("raster_form", "Dark Threshold",
                                           "raster_dark_thr", cfg.raster_dark_thr,
                                           "brightness cutoff for darkest pass",
                                           fmt="%.2f", step=0.05, lo=0.05, hi=0.9)
                                self._frow("raster_form", "Light Threshold",
                                           "raster_light_thr", cfg.raster_light_thr,
                                           "brightness cutoff for lightest pass",
                                           fmt="%.2f", step=0.05, lo=0.1, hi=1.0)
                                self._frow("raster_form", "Angle Step (°)",
                                           "raster_angle_step", cfg.raster_angle_step,
                                           "degrees between passes (45 = crosshatch)",
                                           step=5.0, lo=10.0, hi=90.0)

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=6)
                            dpg.add_text("Iso-Contour Lines",
                                         color=(170, 170, 180, 255))
                            dpg.add_spacer(height=4)
                            dpg.add_text(
                                "Draws iso-brightness curves as pen strokes\n"
                                "— like a topo map of the brightness field.\n"
                                "Hand-drawn feel, zero polygon artifacts.",
                                color=(110, 110, 120, 200))
                            dpg.add_spacer(height=6)
                            dpg.add_checkbox(
                                label="Iso-contour lines",
                                tag="iso_contour_enabled",
                                default_value=cfg.iso_contour_enabled,
                            )
                            dpg.add_spacer(height=4)
                            with dpg.table(
                                header_row=False, tag="iso_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=165, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()
                                self._frow("iso_form", "Levels",
                                           "iso_levels", cfg.iso_contour_levels,
                                           "number of brightness levels (10–40)",
                                           typ="i", lo=4, hi=80)
                                self._frow("iso_form", "Budget",
                                           "iso_budget", cfg.iso_contour_budget,
                                           "max path elements",
                                           typ="i", lo=100, hi=8000)
                                self._frow("iso_form", "Min Length (px)",
                                           "iso_min_len", cfg.iso_contour_min_len,
                                           "filter tiny contour fragments",
                                           step=5.0, lo=5.0, hi=200.0)
                                self._frow("iso_form", "Blur (px)",
                                           "iso_blur", cfg.iso_contour_blur_px,
                                           "smooth before contour finding (1–5)",
                                           step=0.5, lo=0.0, hi=20.0)
                                self._frow("iso_form", "Shadow Compression",
                                           "iso_gamma", cfg.iso_contour_gamma,
                                           "< 1 = more levels in dark areas",
                                           fmt="%.2f", step=0.05, lo=0.1, hi=1.5)

                            dpg.add_spacer(height=8)
                            dpg.add_separator()
                            dpg.add_spacer(height=6)
                            dpg.add_text("Aux Layers", color=(170, 170, 180, 255))
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Detail edges",
                                                 tag="detail_on",
                                                 default_value=cfg.detail_enabled)
                                dpg.add_spacer(width=12)
                                dpg.add_checkbox(label="Light edges",
                                                 tag="le_on",
                                                 default_value=cfg.lightedge_enabled)
                                dpg.add_spacer(width=12)
                                dpg.add_checkbox(label="Dark edges",
                                                 tag="de_on",
                                                 default_value=cfg.darkedge_enabled)
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(label="Face shade",
                                                 tag="face_on",
                                                 default_value=cfg.face_enabled)
                                dpg.add_spacer(width=12)
                                dpg.add_checkbox(label="Micro detail",
                                                 tag="micro_on",
                                                 default_value=cfg.micro_enabled)
                            dpg.add_spacer(height=4)
                            dpg.add_text("(Face shade and Micro detail may add noise)",
                                         color=(100, 100, 110, 200))

                        # ── CMY Halftone tab ────────────────────────────────────
                        with dpg.tab(label=" CMY Halftone "):
                            ccfg = self.cmy_cfg
                            dpg.add_spacer(height=6)
                            dpg.add_text(
                                "Decomposes image into C / M / Y / K ink channels.",
                                color=(160, 160, 170, 220))
                            dpg.add_text(
                                "Each channel is hatched at its classic screen angle.\n"
                                "Overlapping lines optically mix to match image colour.",
                                color=(120, 120, 130, 200))
                            dpg.add_spacer(height=8)
                            with dpg.table(
                                header_row=False, tag="cmy_form",
                                borders_innerV=False, borders_outerV=False,
                                policy=dpg.mvTable_SizingFixedFit,
                            ):
                                dpg.add_table_column(init_width_or_weight=165, width_fixed=True)
                                dpg.add_table_column(init_width_or_weight=115, width_fixed=True)
                                dpg.add_table_column()

                                self._frow("cmy_form", "Min Spacing (px)",
                                           "cmy_min_sp", ccfg.min_spacing_px,
                                           "densest lines (100% ink)",
                                           step=0.5, lo=1.0, hi=20.0)
                                self._frow("cmy_form", "Max Spacing (px)",
                                           "cmy_max_sp", ccfg.max_spacing_px,
                                           "sparsest lines (near 0% ink)",
                                           step=1.0, lo=5.0, hi=100.0)
                                self._frow("cmy_form", "Ink Threshold",
                                           "cmy_thr", ccfg.min_channel_thr,
                                           "ignore channel values below this",
                                           fmt="%.3f", step=0.01, lo=0.02, hi=0.5)
                                self._frow("cmy_form", "Density Bands",
                                           "cmy_bands", ccfg.density_bands,
                                           "spacing quantisation steps",
                                           typ="i", lo=4, hi=20)
                                self._frow("cmy_form", "UCR Amount",
                                           "cmy_ucr", ccfg.ucr_amount,
                                           "0=none  1=full black extraction",
                                           fmt="%.2f", step=0.05, lo=0.0, hi=1.0)
                                self._frow("cmy_form", "Max Paths/Channel",
                                           "cmy_max_paths", ccfg.max_paths_per_channel,
                                           "Cricut ceiling per channel",
                                           typ="i", lo=100, hi=5000)
                                self._frow("cmy_form", "Min Area (px²)",
                                           "cmy_min_area", ccfg.min_area_px2,
                                           "ignore tiny ink regions",
                                           step=5.0, lo=0.0, hi=500.0)
                                self._frow("cmy_form", "Cyan Angle (°)",
                                           "cmy_c_ang", ccfg.cyan_angle_deg,
                                           "default 105°",
                                           step=1.0, lo=0.0, hi=180.0)
                                self._frow("cmy_form", "Magenta Angle (°)",
                                           "cmy_m_ang", ccfg.magenta_angle_deg,
                                           "default 75°",
                                           step=1.0, lo=0.0, hi=180.0)
                                self._frow("cmy_form", "Yellow Angle (°)",
                                           "cmy_y_ang", ccfg.yellow_angle_deg,
                                           "default 90°",
                                           step=1.0, lo=0.0, hi=180.0)
                                self._frow("cmy_form", "Black Angle (°)",
                                           "cmy_k_ang", ccfg.black_angle_deg,
                                           "default 45°",
                                           step=1.0, lo=0.0, hi=180.0)

                            dpg.add_spacer(height=8)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label="Use black (K) channel",
                                    tag="cmy_use_black",
                                    default_value=ccfg.use_black,
                                )
                            dpg.add_spacer(height=4)
                            with dpg.group(horizontal=True):
                                dpg.add_checkbox(
                                    label="Crosshatch black channel",
                                    tag="cmy_black_cross",
                                    default_value=ccfg.black_cross,
                                )
                            dpg.add_spacer(height=10)
                            dpg.add_separator()
                            dpg.add_spacer(height=8)
                            dpg.add_button(
                                label="  Generate CMY Layers  ",
                                tag="btn_generate_color",
                                callback=self.on_generate_cmy,
                            )
                            dpg.add_spacer(height=4)
                            dpg.add_text(
                                "Output: <image_folder>/<name>_cmy_layers/",
                                color=(120, 120, 130, 200))

                    # ── Generate controls (below tabs) ────────────────────────
                    dpg.add_spacer(height=10)
                    dpg.add_separator()
                    dpg.add_spacer(height=8)

                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="  Generate SVG  ",
                            tag="btn_generate",
                            callback=self.on_generate,
                        )
                        dpg.add_spacer(width=8)
                        dpg.add_button(
                            label="  Open Folder  ",
                            callback=self.on_open_folder,
                        )
                        dpg.add_spacer(width=8)
                        dpg.add_button(
                            label="  Save Report  ",
                            callback=self.on_save_report,
                        )

                    dpg.add_spacer(height=8)
                    dpg.add_progress_bar(
                        tag="progress_bar",
                        default_value=0.0,
                        width=-1,
                        height=16,
                    )
                    dpg.add_spacer(height=4)
                    dpg.add_text("Ready.", tag="lbl_status",
                                 color=(160, 200, 160, 255))

                dpg.add_spacer(width=8)

                # ── RIGHT PANEL: Preview ──────────────────────────────────────
                with dpg.child_window(tag="right_panel", width=-1,
                                      height=-220, border=True, no_scrollbar=False):
                    dpg.add_spacer(height=4)
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=4)
                        dpg.add_text("Preview", color=(170, 170, 180, 255))
                    dpg.add_spacer(height=6)
                    # Image centered in available space
                    dpg.add_image("preview_tex", tag="preview_img",
                                  width=_PREVIEW_W, height=_PREVIEW_H)

            # ── Log panel ─────────────────────────────────────────────────────
            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text("Log  (select all → Ctrl+A, copy → Ctrl+C)",
                         color=(170, 170, 180, 255))
            dpg.add_spacer(height=2)
            _init_log = ""
            if not _SKIMAGE_OK:
                _init_log = "WARNING: scikit-image not found. Polygonization fallback active.\n"
            dpg.add_input_text(
                tag="log_panel",
                default_value=_init_log,
                multiline=True,
                readonly=True,
                width=-1,
                height=-1,
                tab_input=False,
            )

        dpg.set_primary_window("main_win", True)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_file_selected(self, sender, app_data, user_data=None):
        selections = app_data.get("selections", {})
        path = (list(selections.values())[0] if selections
                else app_data.get("file_path_name", ""))
        if path and os.path.isfile(path):
            self._set_file(path)

    def _set_file(self, path: str):
        self.png_path = path
        base    = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(os.path.dirname(path), "hatched_out")
        os.makedirs(out_dir, exist_ok=True)
        self.out_svg_path = os.path.join(out_dir, f"{base}_hatched.svg")
        dpg.set_value("lbl_file", f"  {os.path.basename(path)}")
        self._append_log(f"Selected:  {path}")
        self._append_log(f"Output:    {self.out_svg_path}")
        # Show source image immediately in preview panel
        try:
            img = Image.open(path).convert("RGB")
            self._update_preview(img)
        except Exception as e:
            self._append_log(f"Preview load error: {e}")

    def on_generate(self):
        if not self.png_path:
            return
        if self._running:
            return

        self._running = True
        dpg.configure_item("btn_generate", enabled=False)
        dpg.set_value("progress_bar", 0.0)
        self._sync_cfg()
        self._append_log("\n─── GENERATE ───")
        self._append_log(
            f"Spacing={self.cfg.base_spacing_px}  Depth={self.cfg.ease_power}  "
            f"Angle={self.cfg.base_angle_deg}  Gamma={self.cfg.band_gamma}  "
            f"EdgeSens={self.cfg.detail_pct:.1f}  LC={self.cfg.local_contrast_strength:.2f}  "
            f"HL={self.cfg.highlight_protect_thr:.2f}  MaxPaths={self.cfg.max_paths}"
        )

        def worker():
            try:
                stats = hatch_pipeline(
                    self.png_path, self.out_svg_path, self.cfg,
                    progress_cb=lambda f: self._progress_q.put(f),
                    status_cb=lambda s: self._status_q.put(s),
                    log_cb=lambda s: self._log_q.put(s),
                )
                self._log_q.put(
                    f"\nDone in {stats['elapsed_sec']:.1f}s\n"
                    f"SVG paths (Cricut count): {stats['paths']}\n"
                    f"Pen lifts (M commands):   {stats['pen_lifts_est']}\n"
                    f"Aux used: {stats['aux_used']}  |  "
                    f"Band budget left: {stats['remaining_for_bands']}\n"
                    f"Saved: {self.out_svg_path}"
                )
                self._log_q.put(diagnose_run(stats, self.cfg))
                self._last_stats = stats
                self._status_q.put("Complete.")
                self._progress_q.put(1.0)
                prev = stats.get("preview_image")
                if prev is not None:
                    self._log_q.put(f"Preview image ready: {prev.size[0]}×{prev.size[1]}px — queuing…")
                    self._preview_q.put(prev)
                else:
                    self._log_q.put("Preview image: not generated (check log for errors)")
            except Exception:
                self._log_q.put("ERROR:\n" + traceback.format_exc())
                self._status_q.put("Error — see log.")
            finally:
                self._running = False
                # Re-enable button via flag; main loop handles it
                self._progress_q.put("DONE")

        threading.Thread(target=worker, daemon=True).start()

    # ── CMY pipeline ─────────────────────────────────────────────────────────

    def _sync_cmy_cfg(self):
        def _f(tag):
            v = dpg.get_value(tag)
            return float(v[0] if isinstance(v, (list, tuple)) else v)
        def _i(tag):
            v = dpg.get_value(tag)
            return int(v[0] if isinstance(v, (list, tuple)) else v)
        def _b(tag):
            v = dpg.get_value(tag)
            return bool(v[0] if isinstance(v, (list, tuple)) else v)

        c = self.cmy_cfg
        c.min_spacing_px         = max(1.0,  _f("cmy_min_sp"))
        c.max_spacing_px         = max(c.min_spacing_px + 1.0, _f("cmy_max_sp"))
        c.min_channel_thr        = max(0.01, min(0.5, _f("cmy_thr")))
        c.density_bands          = max(4,    _i("cmy_bands"))
        c.ucr_amount             = max(0.0,  min(1.0, _f("cmy_ucr")))
        c.max_paths_per_channel  = max(100,  _i("cmy_max_paths"))
        c.min_area_px2           = max(0.0,  _f("cmy_min_area"))
        c.cyan_angle_deg         = _f("cmy_c_ang")
        c.magenta_angle_deg      = _f("cmy_m_ang")
        c.yellow_angle_deg       = _f("cmy_y_ang")
        c.black_angle_deg        = _f("cmy_k_ang")
        c.use_black              = _b("cmy_use_black")
        c.black_cross            = _b("cmy_black_cross")

    def on_generate_cmy(self):
        if not self.png_path:
            return
        if self._running:
            return

        self._running = True
        dpg.configure_item("btn_generate",       enabled=False)
        dpg.configure_item("btn_generate_color", enabled=False)
        dpg.set_value("progress_bar", 0.0)
        self._sync_cfg()
        self._sync_cmy_cfg()

        base    = os.path.splitext(os.path.basename(self.png_path))[0]
        out_dir = os.path.join(
            os.path.dirname(self.png_path),
            f"{base}_cmy_layers"
        )

        k_note = " + K" if self.cmy_cfg.use_black else ""
        self._append_log(f"\n─── CMY{k_note} HALFTONE ───")
        self._append_log(f"Output folder: {out_dir}")

        def worker():
            try:
                stats = cmy_hatch_pipeline(
                    self.png_path, out_dir,
                    self.cfg, self.cmy_cfg,
                    progress_cb=lambda f: self._progress_q.put(f),
                    status_cb=lambda s:   self._status_q.put(s),
                    log_cb=lambda s:      self._log_q.put(s),
                )
                summary = (
                    f"\nDone in {stats['elapsed_sec']:.1f}s\n"
                    f"Channels written: {len(stats['layers'])}\n"
                    f"Total paths: {stats['total_paths']}\n"
                )
                for lyr in stats["layers"]:
                    summary += (f"  {lyr['channel']}  paths={lyr['paths']}  "
                                f"{os.path.basename(lyr['file'])}\n")
                self._log_q.put(summary)
                self._status_q.put("CMY layers complete.")
                self._progress_q.put(1.0)
                prev = stats.get("preview_image")
                if prev is not None:
                    self._log_q.put(f"Preview: {prev.size[0]}×{prev.size[1]}px")
                    self._preview_q.put(prev)
                else:
                    self._log_q.put("Preview: not generated")
            except Exception:
                self._log_q.put("ERROR:\n" + traceback.format_exc())
                self._status_q.put("Error — see log.")
            finally:
                self._running = False
                self._progress_q.put("DONE")

        threading.Thread(target=worker, daemon=True).start()

    def on_download_hed(self):
        if self._running:
            return
        self._running = True
        dpg.configure_item("btn_hed_download", enabled=False)
        self._append_log("\n─── Downloading HED model ───")

        def worker():
            try:
                ok = download_hed_model(log_cb=lambda s: self._log_q.put(s))
                if ok:
                    self._status_q.put("HED model ready.")
                    self._log_q.put("HED model download complete.")
                    # Update the status label and enable the checkbox
                    self._preview_q.put("__hed_ready__")
                else:
                    self._status_q.put("HED download failed — see log.")
            except Exception:
                self._log_q.put("ERROR:\n" + traceback.format_exc())
                self._status_q.put("HED download error.")
            finally:
                self._running = False
                self._progress_q.put("DONE")

        threading.Thread(target=worker, daemon=True).start()

    def on_open_folder(self):
        if not self.out_svg_path:
            return
        out_dir = os.path.dirname(self.out_svg_path)
        try:
            if sys.platform.startswith("win"):
                os.startfile(out_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
        except Exception as e:
            self._append_log(f"Could not open folder: {e}")

    # ── Config sync ──────────────────────────────────────────────────────────

    def _sync_cfg(self):
        def _f(tag):
            v = dpg.get_value(tag)
            return float(v[0] if isinstance(v, (list, tuple)) else v)

        def _i(tag):
            v = dpg.get_value(tag)
            return int(v[0] if isinstance(v, (list, tuple)) else v)

        def _b(tag):
            v = dpg.get_value(tag)
            return bool(v[0] if isinstance(v, (list, tuple)) else v)

        c = self.cfg
        c.base_spacing_px        = max(1.0,  _f("spacing"))
        c.ease_power             = max(0.5,  _f("depth"))
        c.base_angle_deg         = _f("angle")
        c.band_gamma             = max(0.05, _f("gamma"))
        c.detail_pct             = max(50.0, min(99.0, _f("edge_sens")))
        c.local_contrast_strength = max(0.0, _f("lc_strength"))
        c.local_contrast         = c.local_contrast_strength > 0.0
        c.highlight_protect_thr  = max(0.5, min(1.0, _f("hl_thr")))
        c.max_paths              = max(100,  _i("max_paths"))
        c.cross_enabled          = _b("cross_enabled")
        c.cross_dark_n           = max(0,    _i("cross_n"))
        c.invert                 = _b("invert")
        # Advanced
        c.dark_mult              = max(0.05, _f("dark_mult"))
        c.light_mult             = max(0.1,  _f("light_mult"))
        c.angle_wobble_deg       = max(0.0,  min(45.0, _f("angle_wobble")))
        c.bands                  = max(2,    _i("bands"))
        c.min_area_dark_px2      = max(0.0,  _f("min_dark"))
        c.min_area_light_px2     = max(0.0,  _f("min_light"))
        c.global_spacing_mult    = max(0.1,  _f("gspacing"))
        c.detail_enabled         = bool(dpg.get_value("detail_on"))
        c.lightedge_enabled      = bool(dpg.get_value("le_on"))
        c.darkedge_enabled       = bool(dpg.get_value("de_on"))
        c.face_enabled           = bool(dpg.get_value("face_on"))
        c.micro_enabled          = _b("micro_on")
        c.hed_enabled            = _b("hed_enabled")
        c.hed_max_px             = max(128, min(1024, _i("hed_max_px")))
        c.contour_hatch          = _b("contour_hatch")
        c.contour_hatch_strength = max(0.0, min(1.0, _f("contour_strength")))
        c.contour_hatch_blur     = max(0.0, _f("contour_blur"))
        # Raster hatch passes
        c.raster_hatch           = _b("raster_hatch")
        c.raster_passes          = max(1, _i("raster_passes"))
        c.raster_dark_thr        = max(0.05, min(0.9,  _f("raster_dark_thr")))
        c.raster_light_thr       = max(0.1,  min(1.0,  _f("raster_light_thr")))
        c.raster_angle_step      = max(10.0, _f("raster_angle_step"))
        # Iso-contour lines
        c.iso_contour_enabled    = _b("iso_contour_enabled")
        c.iso_contour_levels     = max(4, _i("iso_levels"))
        c.iso_contour_budget     = max(100, _i("iso_budget"))
        c.iso_contour_min_len    = max(5.0, _f("iso_min_len"))
        c.iso_contour_blur_px    = max(0.0, _f("iso_blur"))
        c.iso_contour_gamma      = max(0.1, _f("iso_gamma"))

    # ── Preview texture update ────────────────────────────────────────────────

    def _update_preview(self, pil_img: Image.Image):
        """Letterbox pil_img into the fixed preview texture and push to GPU."""
        pw, ph = pil_img.size
        scale  = min(_PREVIEW_W / max(pw, 1), _PREVIEW_H / max(ph, 1))
        nw, nh = int(pw * scale), int(ph * scale)
        resized = pil_img.resize((nw, nh), Image.LANCZOS)

        canvas = Image.new("RGB", (_PREVIEW_W, _PREVIEW_H), (36, 36, 40))
        ox = (_PREVIEW_W - nw) // 2
        oy = (_PREVIEW_H - nh) // 2
        canvas.paste(resized, (ox, oy))

        arr  = np.ascontiguousarray(
            np.array(canvas.convert("RGBA")).astype(np.float32) / 255.0
        ).flatten()
        dpg.set_value("preview_tex", arr)

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        """Append text to the selectable log widget and internal buffer."""
        self._log_lines.append(msg)
        current = dpg.get_value("log_panel") or ""
        dpg.set_value("log_panel", current + msg + "\n")

    def _set_status(self, msg: str):
        dpg.set_value("lbl_status", msg)

    def on_save_report(self):
        """Write a self-contained diagnostic report to the output folder."""
        import dataclasses, json

        out_dir = (os.path.dirname(self.out_svg_path)
                   if self.out_svg_path else os.path.expanduser("~"))
        report_path = os.path.join(out_dir, "hatch_diagnostic_report.txt")

        lines: List[str] = []
        sep = "=" * 60

        lines += [sep, "HATCH DIAGNOSTIC REPORT", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", sep, ""]

        # ── Source file ───────────────────────────────────────────────────────
        lines += ["SOURCE IMAGE", "-" * 40,
                  f"  {self.png_path or '(none selected)'}",
                  f"  Output SVG: {self.out_svg_path or '(none)'}",
                  ""]

        # ── HatchConfig ───────────────────────────────────────────────────────
        lines += ["HATCH CONFIG", "-" * 40]
        for f in dataclasses.fields(self.cfg):
            lines.append(f"  {f.name:35s} = {getattr(self.cfg, f.name)}")
        lines.append("")

        # ── CMYConfig ─────────────────────────────────────────────────────────
        lines += ["CMY CONFIG", "-" * 40]
        for f in dataclasses.fields(self.cmy_cfg):
            lines.append(f"  {f.name:35s} = {getattr(self.cmy_cfg, f.name)}")
        lines.append("")

        # ── Last run stats ────────────────────────────────────────────────────
        stats = self._last_stats
        if stats:
            lines += ["LAST RUN STATS", "-" * 40]
            skip_keys = {"preview_image", "band_results"}
            for k, v in stats.items():
                if k not in skip_keys:
                    lines.append(f"  {k:35s} = {v}")
            lines.append("")

            band_results = stats.get("band_results", [])
            if band_results:
                lines += ["BAND RESULTS", "-" * 40]
                lines.append(f"  {'Band':>5}  {'on_px':>8}  {'polys':>6}  "
                             f"{'budget':>7}  {'paths':>6}  {'spacing':>8}")
                for r in band_results:
                    lines.append(
                        f"  {r['band']:>5}  {r['on_px']:>8}  {r['polys']:>6}  "
                        f"  {r['budget']:>7}  {r['paths']:>6}  {r['spacing']:>8.1f}"
                    )
                lines.append("")

            # Re-run diagnostic so it's always fresh
            lines += ["DIAGNOSTIC REPORT", "-" * 40,
                      diagnose_run(stats, self.cfg), ""]
        else:
            lines += ["LAST RUN STATS", "-" * 40,
                      "  (no completed run yet — generate first)", ""]

        # ── Full session log ──────────────────────────────────────────────────
        lines += [sep, "FULL SESSION LOG", sep]
        for entry in self._log_lines:
            lines.append(entry)

        try:
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
            self._append_log(f"\nReport saved: {report_path}")
            if sys.platform.startswith("win"):
                os.startfile(os.path.dirname(report_path))  # type: ignore
        except Exception as e:
            self._append_log(f"Save report error: {e}")

    # ── Queue flusher (called every frame) ────────────────────────────────────

    def _flush_queues(self):
        changed = False

        while not self._log_q.empty():
            self._append_log(self._log_q.get_nowait())
            changed = True

        while not self._status_q.empty():
            self._set_status(self._status_q.get_nowait())

        while not self._progress_q.empty():
            val = self._progress_q.get_nowait()
            if val == "DONE":
                dpg.configure_item("btn_generate",       enabled=True)
                dpg.configure_item("btn_generate_color", enabled=True)
            else:
                dpg.set_value("progress_bar", float(val))

        while not self._preview_q.empty():
            item = self._preview_q.get_nowait()
            if item == "__hed_ready__":
                # HED model just downloaded — update status label + enable checkbox
                try:
                    dpg.set_value("lbl_hed_status", "✓ Model ready")
                    dpg.configure_item("lbl_hed_status", color=(100, 220, 130, 255))
                    dpg.configure_item("hed_enabled", enabled=True)
                    dpg.configure_item("btn_hed_download", enabled=True)
                except Exception:
                    pass
            else:
                try:
                    self._update_preview(item)
                except Exception as e:
                    self._append_log(f"Preview display error: {e}")
            changed = True

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self):
        dpg.create_context()
        self._setup_theme()
        self._build()

        dpg.create_viewport(
            title="Pen & Ink SVG Generator",
            width=1200, height=860,
            min_width=900, min_height=700,
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Manual render loop so we can flush queues each frame
        while dpg.is_dearpygui_running():
            self._flush_queues()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    HatchApp().run()


if __name__ == "__main__":
    main()
