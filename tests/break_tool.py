"""
Manual stitch-break tool for single-path tonal output.

Single-path mode draws the whole image as ONE continuous stroke (1 pen lift),
which leaves visible travel connectors where the pen crosses light areas to reach
a far region. This tool lets you cut those connectors: each break splits the path
there (one extra pen lift), so only the subtle in-region connectors remain.

    py -3.11 tests/break_tool.py [path/to/<name>_strokes.json]

If no path is given, the most recent *_strokes.json under results/runs is used.
Generate one first by rendering any image with a single-path preset, e.g. the
"single_path" preset in the batch cockpit, or:
    hatch_pipeline(img, out.svg, replace(baseline, tonal_single_path=True))

Workflow: red overlay lines are the travel connectors (filtered by length via the
slider). Click one to toggle a break (red = kept pen-down, green = broken/pen-up).
Export writes <name>_broken.svg with the breaks applied. Short connectors below
the slider are always kept pen-down (they're invisible in-region turns).
"""
from __future__ import annotations

import os
import sys
import glob
import json
import math

import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H
from hatch_ui_nocairo import HatchConfig, strokes_to_single_d, write_svg, render_preview

DISP_MAX = 1000   # max display dimension


# ── non-GUI logic (importable / testable) ──────────────────────────────────────

def find_latest_sidecar():
    cands = glob.glob(os.path.join(H.RUNS_DIR, "**", "*_strokes.json"), recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None


def load_sidecar(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    strokes = [[(float(x), float(y)) for x, y in s] for s in d["strokes"]]
    return int(d["width"]), int(d["height"]), strokes


def connectors(strokes):
    """Connector k spans strokes[k].end -> strokes[k+1].start."""
    out = []
    for k in range(len(strokes) - 1):
        p0 = strokes[k][-1]
        p1 = strokes[k + 1][0]
        out.append((k, p0, p1, math.dist(p0, p1)))
    return out


def export_svg(strokes, broken, w, h, out_path):
    """Write the SVG with `broken` connector indices as pen-ups."""
    cfg = HatchConfig()
    d = strokes_to_single_d(strokes, broken=broken)
    write_svg(out_path, [d], w, h, cfg)
    return out_path


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance from point (px,py) to segment (a,b)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
    cx, cy = ax + t * vx, ay + t * vy
    return math.hypot(px - cx, py - cy)


# ── GUI ─────────────────────────────────────────────────────────────────────

class BreakApp:
    def __init__(self, sidecar_path):
        self.path = sidecar_path
        self.w, self.h, self.strokes = load_sidecar(sidecar_path)
        self.conns = connectors(self.strokes)
        self.broken = set()
        self.scale = min(DISP_MAX / max(self.w, 1), DISP_MAX / max(self.h, 1))
        self.dw = max(1, int(self.w * self.scale))
        self.dh = max(1, int(self.h * self.scale))
        self.min_len = 15.0
        self.line_tags = {}   # connector index -> drawn line tag

    # candidates = connectors longer than the length filter
    def _candidates(self):
        return [c for c in self.conns if c[3] >= self.min_len]

    def _draw_overlay(self):
        dpg.delete_item("dl", children_only=True, slot=2)
        dpg.draw_image("bg", (0, 0), (self.dw, self.dh), parent="dl")
        self.line_tags.clear()
        for (k, p0, p1, ln) in self._candidates():
            col = (40, 200, 40, 255) if k in self.broken else (230, 30, 30, 220)
            tag = dpg.draw_line((p0[0] * self.scale, p0[1] * self.scale),
                                (p1[0] * self.scale, p1[1] * self.scale),
                                color=col, thickness=1.5, parent="dl")
            self.line_tags[k] = tag
        self._update_status()

    def _update_status(self):
        cand = self._candidates()
        dpg.set_value("status",
                      f"connectors shown: {len(cand)}  |  broken: {len(self.broken)}  "
                      f"|  resulting pen lifts: {1 + len(self.broken)}")

    def on_click(self):
        mx, my = dpg.get_mouse_pos(local=False)
        try:
            ox, oy = dpg.get_item_rect_min("dl")
        except Exception:
            return
        lx, ly = mx - ox, my - oy
        if lx < 0 or ly < 0 or lx > self.dw or ly > self.dh:
            return
        # nearest candidate connector in display space, within 8px
        best, bestd = None, 8.0
        for (k, p0, p1, ln) in self._candidates():
            d = _seg_dist(lx, ly, p0[0] * self.scale, p0[1] * self.scale,
                          p1[0] * self.scale, p1[1] * self.scale)
            if d < bestd:
                best, bestd = k, d
        if best is None:
            return
        if best in self.broken:
            self.broken.discard(best)
        else:
            self.broken.add(best)
        tag = self.line_tags.get(best)
        if tag:
            col = (40, 200, 40, 255) if best in self.broken else (230, 30, 30, 220)
            dpg.configure_item(tag, color=col)
        self._update_status()

    def on_filter(self, sender, value):
        self.min_len = float(value)
        # drop breaks that are no longer candidates
        self.broken = {k for k in self.broken
                       if self.conns[k][3] >= self.min_len}
        self._draw_overlay()

    def on_break_all_shown(self):
        for (k, p0, p1, ln) in self._candidates():
            self.broken.add(k)
        self._draw_overlay()

    def on_clear(self):
        self.broken.clear()
        self._draw_overlay()

    def on_export(self):
        out = os.path.splitext(self.path)[0].replace("_strokes", "") + "_broken.svg"
        export_svg(self.strokes, self.broken, self.w, self.h, out)
        dpg.set_value("status", f"Exported {out}  ({1 + len(self.broken)} pen lifts)")
        try:
            os.startfile(out)  # Windows
        except Exception:
            pass

    def _make_bg_texture(self):
        # strokes-only background (all connectors pen-up -> not drawn), faint.
        allbroken = set(range(len(self.strokes) - 1))
        d = strokes_to_single_d(self.strokes, broken=allbroken)
        im = render_preview([d], self.w, self.h, max_dim=DISP_MAX).convert("RGBA")
        im = im.resize((self.dw, self.dh))
        # fade to light gray so red/green connectors pop
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr[..., :3] = 1.0 - (1.0 - arr[..., :3]) * 0.45
        with dpg.texture_registry():
            dpg.add_static_texture(self.dw, self.dh, arr.flatten(), tag="bg")

    def build(self):
        with dpg.window(tag="main"):
            dpg.add_text(f"Stitch-break tool — {os.path.basename(self.path)}  "
                         f"({len(self.strokes)} strokes, {len(self.conns)} connectors)",
                         color=(120, 200, 255))
            dpg.add_text("Red = travel connector (kept). Click to toggle a break "
                         "(green = cut / pen-up). Short connectors below the filter "
                         "stay pen-down.", wrap=980)
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="min connector length (px)", tag="flt",
                                     default_value=self.min_len, min_value=2.0,
                                     max_value=120.0, width=280, callback=self.on_filter)
                dpg.add_button(label="Break all shown", callback=self.on_break_all_shown)
                dpg.add_button(label="Clear breaks", callback=self.on_clear)
                dpg.add_button(label="Export broken SVG", callback=self.on_export,
                               width=170)
            dpg.add_text("", tag="status")
            self._make_bg_texture()
            with dpg.drawlist(width=self.dw, height=self.dh, tag="dl"):
                pass
        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left,
                                        callback=self.on_click)

    def run(self):
        dpg.create_context()
        self.build()
        self._draw_overlay()
        dpg.create_viewport(title="Stitch-break tool",
                            width=min(1200, self.dw + 60),
                            height=self.dh + 200)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest_sidecar()
    if not path or not os.path.exists(path):
        print("No *_strokes.json found. Render an image with tonal_single_path=True "
              "first (e.g. the 'single_path' preset).")
        return
    print(f"Loading {path}")
    BreakApp(path).run()


if __name__ == "__main__":
    main()
