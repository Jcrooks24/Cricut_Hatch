"""
Manual stitch-break tool for single-path tonal output.

Single-path mode draws the whole image as ONE continuous stroke (1 pen lift),
which leaves visible travel connectors where the pen crosses light areas to reach
a far region. This tool lets you cut those connectors: each break splits the path
there (one extra pen lift), so only the subtle in-region connectors remain.

    py -3.11 tests/break_tool.py [path/to/<name>_strokes.json]

If no path is given, the most recent *_strokes.json under results/runs is used.

Controls (left mouse is always the break tool):
    left-click     toggle a break on the nearest connector
    left-drag      "paint" breaks across every connector you swipe over
    scroll wheel   zoom to cursor
    Space+drag     pan   (or middle-mouse drag)
Red = connector kept (pen-down). Green = broken (pen-up). Short connectors below
the length slider are always kept. Export writes <name>_broken.svg.
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

DISP_MAX = 900   # base display dimension (zoom in for detail)


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
        self.line_tags = {}          # connector index -> drawn line tag
        self.zoom = 1.0
        self.pan = [0.0, 0.0]        # drawlist-local translation (px)
        self._pan0 = [0.0, 0.0]
        self._moved = False
        self._ldown = False          # left gesture in progress
        self._mdown = False          # middle (pan) gesture in progress
        self._space = False          # Space held -> pan modifier
        self._gesture_pan = False    # current left gesture is a pan (Space at press)

    def _candidates(self):
        return [c for c in self.conns if c[3] >= self.min_len]

    # ── coordinate transforms ─────────────────────────────────────────────────
    # zoom/pan are baked into draw coordinates (no draw_node transform, which some
    # DPG versions won't accept as a draw_image parent). Full redraw on zoom/pan.
    def _to_screen(self, ix, iy):
        return (ix * self.scale * self.zoom + self.pan[0],
                iy * self.scale * self.zoom + self.pan[1])

    def _mouse_to_image(self, mx, my):
        """Screen mouse pos -> image-space (px), inverting drawlist/pan/zoom/scale."""
        try:
            ox, oy = dpg.get_item_rect_min("dl")
        except Exception:
            return None
        lx, ly = mx - ox, my - oy                       # drawlist-local screen px
        dx = (lx - self.pan[0]) / self.zoom             # display px
        dy = (ly - self.pan[1]) / self.zoom
        return dx / self.scale, dy / self.scale, lx, ly

    # ── scene build / redraw ──────────────────────────────────────────────────
    def _draw_scene(self):
        dpg.delete_item("dl", children_only=True)
        z, (px, py) = self.zoom, self.pan
        dpg.draw_image("bg", (px, py),
                       (px + self.dw * z, py + self.dh * z), parent="dl")
        self.line_tags.clear()
        for (k, p0, p1, ln) in self._candidates():
            col = (40, 200, 40, 255) if k in self.broken else (230, 30, 30, 220)
            tag = dpg.draw_line(self._to_screen(*p0), self._to_screen(*p1),
                                color=col, thickness=1.5, parent="dl")
            self.line_tags[k] = tag
        self._update_status()

    def _update_status(self):
        dpg.set_value("status",
                      f"connectors shown: {len(self._candidates())}  |  "
                      f"broken: {len(self.broken)}  |  "
                      f"resulting pen lifts: {1 + len(self.broken)}  |  "
                      f"zoom: {self.zoom:.1f}x")

    def _set_broken(self, k, value):
        if value:
            self.broken.add(k)
        else:
            self.broken.discard(k)
        tag = self.line_tags.get(k)
        if tag:
            dpg.configure_item(tag, color=(40, 200, 40, 255) if value
                               else (230, 30, 30, 220))

    def _nearest_connector(self, ix, iy, radius_img):
        best, bestd = None, radius_img
        for (k, p0, p1, ln) in self._candidates():
            d = _seg_dist(ix, iy, p0[0], p0[1], p1[0], p1[1])
            if d < bestd:
                best, bestd = k, d
        return best

    # ── mouse handlers ────────────────────────────────────────────────────────
    def on_wheel(self, sender, app_data):
        mp = dpg.get_mouse_pos(local=False)
        conv = self._mouse_to_image(*mp)
        if conv is None:
            return
        _, _, lx, ly = conv
        if not (0 <= lx <= self.dw and 0 <= ly <= self.dh):
            return
        factor = 1.15 if app_data > 0 else 1.0 / 1.15
        old, new = self.zoom, max(0.5, min(30.0, self.zoom * factor))
        disp_x = (lx - self.pan[0]) / old
        disp_y = (ly - self.pan[1]) / old
        self.pan[0] = lx - new * disp_x                 # keep point under cursor fixed
        self.pan[1] = ly - new * disp_y
        self.zoom = new
        self._draw_scene()
        self._update_status()

    # Space bar held -> a pan modifier (left-drag pans instead of breaking).
    def on_space_down(self, sender, app_data):
        self._space = True

    def on_space_up(self, sender, app_data):
        self._space = False

    # Middle-mouse drag = pan (always). Left-drag with Space held = pan too.
    def on_mid_down(self, sender, app_data):
        if not self._mdown:
            self._mdown = True
            self._pan0 = list(self.pan)

    def on_mid_drag(self, sender, app_data):
        _, ddx, ddy = app_data
        self.pan[0] = self._pan0[0] + ddx
        self.pan[1] = self._pan0[1] + ddy
        self._draw_scene()

    def on_mid_release(self, sender, app_data):
        self._mdown = False

    def on_left_down(self, sender, app_data):
        if not self._ldown:            # first frame of the gesture only
            self._ldown = True
            self._moved = False
            self._gesture_pan = self._space      # lock gesture type at press
            self._pan0 = list(self.pan)

    def on_left_drag(self, sender, app_data):
        _, ddx, ddy = app_data
        if abs(ddx) + abs(ddy) > 3:
            self._moved = True
        if self._gesture_pan:                    # Space held at press -> pan
            self.pan[0] = self._pan0[0] + ddx
            self.pan[1] = self._pan0[1] + ddy
            self._draw_scene()
            return
        mp = dpg.get_mouse_pos(local=False)
        conv = self._mouse_to_image(*mp)
        if conv is None:
            return
        ix, iy, lx, ly = conv
        if not (0 <= lx <= self.dw and 0 <= ly <= self.dh):
            return
        radius = 9.0 / (self.zoom * self.scale)         # ~9 screen px, in image px
        k = self._nearest_connector(ix, iy, radius)
        if k is not None and k not in self.broken:
            self._set_broken(k, True)                   # paint-break
            self._update_status()

    def on_left_release(self, sender, app_data):
        was_drag, was_pan = self._moved, self._gesture_pan
        self._ldown = False
        if was_drag or was_pan:
            return                                       # drag/pan already handled
        mp = dpg.get_mouse_pos(local=False)
        conv = self._mouse_to_image(*mp)
        if conv is None:
            return
        ix, iy, lx, ly = conv
        if not (0 <= lx <= self.dw and 0 <= ly <= self.dh):
            return
        radius = 9.0 / (self.zoom * self.scale)
        k = self._nearest_connector(ix, iy, radius)
        if k is not None:
            self._set_broken(k, k not in self.broken)   # click = toggle
            self._update_status()

    # ── button callbacks ──────────────────────────────────────────────────────
    def on_filter(self, sender, value):
        self.min_len = float(value)
        self.broken = {k for k in self.broken if self.conns[k][3] >= self.min_len}
        self._draw_scene()

    def on_break_all_shown(self):
        for (k, p0, p1, ln) in self._candidates():
            self.broken.add(k)
        self._draw_scene()

    def on_clear(self):
        self.broken.clear()
        self._draw_scene()

    def on_reset_view(self):
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self._draw_scene()
        self._update_status()

    def on_export(self):
        out = os.path.splitext(self.path)[0].replace("_strokes", "") + "_broken.svg"
        export_svg(self.strokes, self.broken, self.w, self.h, out)
        dpg.set_value("status", f"Exported {out}  ({1 + len(self.broken)} pen lifts)")
        try:
            os.startfile(out)  # Windows
        except Exception:
            pass

    def _make_bg_texture(self):
        allbroken = set(range(len(self.strokes) - 1))
        d = strokes_to_single_d(self.strokes, broken=allbroken)   # strokes only
        im = render_preview([d], self.w, self.h, max_dim=DISP_MAX).convert("RGBA")
        im = im.resize((self.dw, self.dh))
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr[..., :3] = 1.0 - (1.0 - arr[..., :3]) * 0.45          # fade for contrast
        with dpg.texture_registry():
            dpg.add_static_texture(self.dw, self.dh, arr.flatten(), tag="bg")

    def build(self):
        with dpg.window(tag="main"):
            dpg.add_text(f"Stitch-break tool — {os.path.basename(self.path)}  "
                         f"({len(self.strokes)} strokes, {len(self.conns)} connectors)",
                         color=(120, 200, 255))
            dpg.add_text("left-click = toggle a break  |  left-drag = paint breaks  |  "
                         "scroll = zoom  |  Space+drag or middle-drag = pan",
                         wrap=980, color=(170, 170, 170))
            with dpg.group(horizontal=True):
                dpg.add_slider_float(label="min connector length (px)", tag="flt",
                                     default_value=self.min_len, min_value=2.0,
                                     max_value=120.0, width=260, callback=self.on_filter)
                dpg.add_button(label="Break all shown", callback=self.on_break_all_shown)
                dpg.add_button(label="Clear", callback=self.on_clear)
                dpg.add_button(label="Reset view", callback=self.on_reset_view)
                dpg.add_button(label="Export broken SVG", callback=self.on_export,
                               width=170)
            dpg.add_text("", tag="status")
            self._make_bg_texture()
            with dpg.drawlist(width=self.dw, height=self.dh, tag="dl"):
                pass

        space_key = getattr(dpg, "mvKey_Spacebar", getattr(dpg, "mvKey_Space", 32))
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.on_wheel)
            # pan: middle-mouse drag, or Space held + left drag
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Middle, callback=self.on_mid_down)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=self.on_mid_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Middle, callback=self.on_mid_release)
            dpg.add_key_down_handler(space_key, callback=self.on_space_down)
            dpg.add_key_release_handler(space_key, callback=self.on_space_up)
            # break: left click = toggle, left drag = paint
            dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=self.on_left_down)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, threshold=0.0,
                                       callback=self.on_left_drag)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.on_left_release)

    def run(self):
        dpg.create_context()
        self.build()
        self._draw_scene()
        dpg.create_viewport(title="Stitch-break tool",
                            width=min(1200, self.dw + 60),
                            height=self.dh + 220)
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
