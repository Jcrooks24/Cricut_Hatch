"""
HATCH: the desktop app.

A native window (pywebview) wrapping the Hatch web UI, wired to the hatching
pipeline. Workflow: import a photo, set the output size, fine-tune for the image,
generate the single-path line art, cut the travel connectors in the built-in
break tool, and export a plotter-ready SVG.

    py -3.11 hatch_app.py            # launch the app
    py -3.11 hatch_app.py --selftest # headless check of the pipeline path (no window)
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import time
from dataclasses import replace

import numpy as np
from PIL import Image

# Pipeline + single source of truth for the methodology. tests/ holds presets and
# the break-tool helpers the app reuses, so put it on the path.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from presets import get as get_preset                                    # noqa: E402
from hatch_ui_nocairo import hatch_pipeline                              # noqa: E402
from break_tool import load_sidecar, connectors as _connectors, export_svg  # noqa: E402

# Anchored methodology: SINGLE-PATH (one continuous stroke), then cut the travel
# connectors in the break step.
BASELINE = get_preset("single_path")
APP_OUT = os.path.join(_REPO, "results", "app")
TEMPLATE = os.path.join(_REPO, "hatch_web", "template.html")

# Rough Cricut motion model (same basis as tests/_pathstats.py): the pen draws
# slowly, travels (lifted) fast, and pays a small penalty per direction change.
DRAW_MM_S, TRAVEL_MM_S, VERTEX_S = 20.0, 100.0, 0.04


# ── pipeline logic (no GUI, unit-testable headless) ───────────────────────────

def build_cfg(size_in: float, growth: float, levels: int, detail_regions=None):
    """Single-path config with the app's dials applied. size_in = longest side."""
    cfg = replace(BASELINE, out_width_in=float(size_in), out_height_in=float(size_in))
    cfg = replace(cfg, tonal_deep_spacing_growth=max(0.0, float(growth)))
    if levels and int(levels) > 0:
        cfg = replace(cfg, tonal_max_layers=int(levels))
    if detail_regions:
        # Circled areas keep finer features; everywhere else stays at the normal
        # global min-area (so only the marked regions change).
        regions = tuple((float(r["cx"]), float(r["cy"]), float(r["r"]))
                        for r in detail_regions)
        cfg = replace(cfg, detail_regions=regions, tonal_adaptive_min_area=True,
                      tonal_min_area_flat=float(cfg.tonal_min_area_px2),
                      tonal_min_area_detail=max(2.0, float(cfg.tonal_min_area_px2) * 0.1),
                      tonal_detail_sharpen=0.5)
    return cfg


def _pen_stats(svg_path: str):
    txt = open(svg_path, encoding="utf-8").read()
    down = up = 0.0
    nverts = 0
    for d in re.findall(r'\bd="([^"]+)"', txt):
        cur = None; cmd = None; first = False
        for t in d.split():
            if t in ("M", "L"):
                cmd = t; first = (t == "M"); continue
            if "," not in t:
                continue
            x, y = (float(v) for v in t.split(","))
            if cur is not None:
                dist = math.hypot(x - cur[0], y - cur[1])
                if cmd == "M" and first:
                    up += dist
                else:
                    down += dist; nverts += 1
            first = False
            cur = (x, y)
    return down, up, nverts


def estimate(svg_path: str, work_w: int, work_h: int, size_in: float):
    down, up, nverts = _pen_stats(svg_path)
    mm_per_px = (float(size_in) * 25.4) / max(work_w, work_h, 1)
    secs = (down * mm_per_px / DRAW_MM_S + up * mm_per_px / TRAVEL_MM_S
            + nverts * VERTEX_S)
    return down * mm_per_px / 1000.0, secs / 3600.0


def _png_b64(path: str, max_dim: int = 760) -> str:
    im = Image.open(path).convert("L")
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def analyze_image(path: str) -> dict:
    """Describe a photo's tone/contrast/detail to seed the fine-tune suggestions."""
    im = Image.open(path).convert("L")
    w, h = im.size
    small = im.copy(); small.thumbnail((256, 256))
    a = np.asarray(small, dtype=np.float32) / 255.0
    mean, std = float(a.mean()), float(a.std())
    gy, gx = np.gradient(a)
    detail = float(np.hypot(gx, gy).mean())

    chars = []
    key = ("low-key" if mean < 0.40 else "high-key" if mean > 0.72 else "mid-key")
    chars.append(key)
    chars.append("high contrast" if std > 0.22 else "low contrast" if std < 0.12
                 else "balanced tone")
    chars.append("fine detail" if detail > 0.085 else "smooth" if detail < 0.04
                 else "some texture")

    lvl = 6 if "low-key" in chars else 10 if "high-key" in chars else 8
    bits = []
    if "low-key" in chars:
        bits.append("mostly dark, so fewer darkness levels keep shadows from filling in")
    elif "high-key" in chars:
        bits.append("bright and open, so more levels hold the delicate midtones")
    else:
        bits.append("a balanced tonal range, a strong candidate for crosshatching")
    reco = f"This photo is {chars[0]} and {chars[2]}."
    note = "Hatch sees " + ", ".join(chars) + ". " + bits[0][0].upper() + bits[0][1:] + "."
    aspect = max(w, h) / max(1, min(w, h))
    return {"px": f"{w} x {h}", "chars": chars, "note": note, "reco": reco,
            "aspect": round(aspect, 4), "suggest_levels": lvl}


def convert(img_path: str, out_svg: str, size_in: float, growth: float, levels: int,
            status_cb=None, detail_regions=None):
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    cfg = build_cfg(size_in, growth, levels, detail_regions)
    stats = hatch_pipeline(img_path, out_svg, cfg, status_cb=status_cb,
                           log_cb=lambda m: None)
    draw_m, hours = estimate(out_svg, stats["work_w"], stats["work_h"], size_in)
    stats.update(draw_m=draw_m, est_hours=hours, svg_path=out_svg,
                 preview_png=os.path.splitext(out_svg)[0] + "_preview.png",
                 sidecar=os.path.splitext(out_svg)[0] + "_strokes.json")
    return stats


# ── native bridge (JS ⇄ Python) ────────────────────────────────────────────────

class Api:
    def __init__(self):
        self.window = None
        self.img_path = None
        self.svg_path = None
        self.sidecar = None
        self.strokes = None
        self.w = self.h = 0
        self.cfg_snap = None
        self.size_in = 12.0

    def _push_status(self, msg):
        """Forward a pipeline status line to the page's progress bar."""
        try:
            self.window.evaluate_js("window.hatchStatus && window.hatchStatus("
                                    + json.dumps(str(msg)) + ")")
        except Exception:
            pass

    # step 1: import
    def pick_image(self):
        import webview
        res = self.window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False,
            file_types=("Images (*.jpg;*.jpeg;*.png;*.webp;*.bmp)", "All files (*.*)"))
        path = (res[0] if isinstance(res, (list, tuple)) and res else
                res if isinstance(res, str) else None)
        if not path or not os.path.isfile(path):
            return None
        self.img_path = path
        info = analyze_image(path)
        info["name"] = os.path.basename(path)
        info["thumb"] = _png_b64(path, 460)
        return info

    # step 4: generate
    def generate(self, params):
        try:
            self.size_in = float(params.get("size_in", 12.0))
            growth = float(params.get("growth", 0.0))
            levels = int(params.get("levels", 0) or 0)
            regions = params.get("detail_regions") or []
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(APP_OUT, f"hatch_{stamp}.svg")
            s = convert(self.img_path, out, self.size_in, growth, levels,
                        status_cb=self._push_status, detail_regions=regions)
            self.svg_path = s["svg_path"]; self.sidecar = s["sidecar"]
            self.w, self.h, self.strokes, self.cfg_snap = load_sidecar(self.sidecar)
            # Only the travel connectors matter for cutting; sub-4px moves are
            # in-region pen turns that are always kept. Dropping them keeps the
            # payload light (original index k preserved so export still maps).
            conns = [{"i": k, "x1": round(p0[0], 1), "y1": round(p0[1], 1),
                      "x2": round(p1[0], 1), "y2": round(p1[1], 1), "len": round(ln, 1)}
                     for (k, p0, p1, ln) in _connectors(self.strokes) if ln >= 4.0]
            return {"preview": _png_b64(s["preview_png"], 820),
                    "paths": s["paths"], "lifts": s["pen_lifts_est"],
                    "draw_m": round(s["draw_m"]), "est_hours": round(s["est_hours"], 1),
                    "img_w": self.w, "img_h": self.h, "connectors": conns}
        except Exception as e:
            return {"error": str(e)}

    # step 6: export (cuts = connector indices to break)
    def export(self, cuts):
        import webview
        if not self.strokes:
            return None
        base = os.path.splitext(os.path.basename(self.img_path or "hatch"))[0]
        res = self.window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=f"{base}_hatch.svg",
            file_types=("SVG vector (*.svg)", "All files (*.*)"))
        dest = (res[0] if isinstance(res, (list, tuple)) and res else
                res if isinstance(res, str) else None)
        if not dest:
            return None
        if not dest.lower().endswith(".svg"):
            dest += ".svg"
        export_svg(self.strokes, set(int(c) for c in cuts), self.w, self.h,
                   dest, self.cfg_snap)
        return dest


def _find_logo():
    """A logo image dropped in hatch_web/ (logo.png|jpg|jpeg|svg|webp) is used in
    the title bar in place of the built-in mark. Returns a data URI or None."""
    base = os.path.join(_REPO, "hatch_web")
    for ext, mime in (("png", "image/png"), ("svg", "image/svg+xml"),
                      ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"),
                      ("webp", "image/webp")):
        p = os.path.join(base, f"logo.{ext}")
        if os.path.exists(p):
            data = base64.b64encode(open(p, "rb").read()).decode()
            return f"data:{mime};base64,{data}"
    return None


def _page_html():
    assets = {}
    logo = _find_logo()
    if logo:
        assets["logo"] = logo
    body = open(TEMPLATE, encoding="utf-8").read().replace("__ASSETS__", json.dumps(assets))
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Hatch</title></head><body>" + body + "</body></html>")


def main():
    import traceback
    import webview
    os.makedirs(APP_OUT, exist_ok=True)
    try:
        api = Api()
        api.window = webview.create_window(
            "Hatch", html=_page_html(), js_api=api,
            width=1280, height=880, min_size=(980, 680),
            background_color="#F7F6F2")
        webview.start()
    except Exception:
        # The desktop shortcut runs pythonw (no console); record why we failed.
        with open(os.path.join(APP_OUT, "hatch_error.log"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise


def _selftest():
    """Headless: exercise the pipeline + connector extraction (no window)."""
    img = os.path.join(_REPO, "tests", "images", "texture_sneaker.jpg")
    print("analyze:", analyze_image(img))
    out = os.path.join(APP_OUT, "_selftest.svg")
    os.makedirs(APP_OUT, exist_ok=True)
    s = convert(img, out, size_in=12.0, growth=0.0, levels=0)
    w, h, strokes, snap = load_sidecar(s["sidecar"])
    conns = _connectors(strokes)
    print(f"generate: paths={s['paths']} lifts={s['pen_lifts_est']} "
          f"draw={s['draw_m']:.0f}m est~{s['est_hours']:.1f}h "
          f"img={w}x{h} connectors={len(conns)}")
    b = _png_b64(s["preview_png"], 820)
    print(f"preview b64 {len(b)//1024} KB")
    assert conns and b.startswith("data:image/png")
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
