"""
HATCH — the production tool.

Load one photo, convert it with the printed baseline (tonal cross-hatch), see a
live preview + an estimated Cricut plot time, and export a plotter-ready SVG.

This is the real tool, separate from the test/grading cockpit (tests/batch_ui.py).
It is wired to the SAME "baseline" config that produced the print you liked, so
the default Convert reproduces that look; a few dials cover the choices that
actually matter for a print (output size, speed/ink, darkness levels).

    py -3.11 hatch_app.py            # launch the app
    py -3.11 hatch_app.py --selftest # headless check of the convert/estimate path
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import replace

# The printed baseline lives in tests/presets.py (single source of truth so the
# app and the grading harness never drift). Make it importable from repo root.
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from presets import get as get_preset                      # noqa: E402
from hatch_ui_nocairo import hatch_pipeline                # noqa: E402

BASELINE = get_preset("baseline")
APP_OUT = os.path.join(_REPO, "results", "app")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Rough Cricut motion model (same basis as tests/_pathstats.py): the pen draws
# slowly, travels (lifted) fast, and pays a small penalty per direction change.
DRAW_MM_S, TRAVEL_MM_S, VERTEX_S = 20.0, 100.0, 0.04


# ── pure logic (no GUI — unit-testable headless) ───────────────────────────────

def build_cfg(size_in: float, growth: float, levels: int):
    """The printed baseline with the app's few dials applied. size_in is the
    LONGEST side (out_width_in / out_height_in is picked by orientation inside
    the library, so setting both = fit the longest side to size_in)."""
    cfg = replace(BASELINE, out_width_in=float(size_in), out_height_in=float(size_in))
    if growth >= 0:
        cfg = replace(cfg, tonal_deep_spacing_growth=float(growth))
    if levels and levels > 0:
        cfg = replace(cfg, tonal_max_layers=int(levels))
    return cfg


def _pen_stats(svg_path: str):
    """Pen-down / pen-up distance (px) and vertex count from an SVG's d-strings,
    honouring the compressed implicit-lineto format the writer emits."""
    txt = open(svg_path, encoding="utf-8").read()
    down = up = 0.0
    nverts = lifts = 0
    for d in re.findall(r'\bd="([^"]+)"', txt):
        cur = None; cmd = None; first_after_m = False
        for t in d.split():
            if t in ("M", "L"):
                cmd = t; first_after_m = (t == "M"); continue
            if "," not in t:
                continue
            x, y = (float(v) for v in t.split(","))
            if cur is not None:
                dist = math.hypot(x - cur[0], y - cur[1])
                if cmd == "M" and first_after_m:
                    up += dist; lifts += 1
                else:
                    down += dist; nverts += 1
            else:
                lifts += 1
            first_after_m = False
            cur = (x, y)
    return down, up, nverts, lifts


def estimate(svg_path: str, work_w: int, work_h: int, size_in: float):
    """Return (draw_metres, est_hours) for a generated SVG at the chosen size."""
    down, up, nverts, _ = _pen_stats(svg_path)
    mm_per_px = (float(size_in) * 25.4) / max(work_w, work_h, 1)
    secs = (down * mm_per_px / DRAW_MM_S
            + up * mm_per_px / TRAVEL_MM_S
            + nverts * VERTEX_S)
    return down * mm_per_px / 1000.0, secs / 3600.0


def convert(img_path: str, out_svg: str, size_in: float, growth: float, levels: int):
    """Run the pipeline and return a metrics dict (+ preview png path)."""
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    cfg = build_cfg(size_in, growth, levels)
    stats = hatch_pipeline(img_path, out_svg, cfg, log_cb=lambda m: None)
    draw_m, hours = estimate(out_svg, stats["work_w"], stats["work_h"], size_in)
    stats["draw_m"] = draw_m
    stats["est_hours"] = hours
    stats["preview_png"] = os.path.splitext(out_svg)[0] + "_preview.png"
    stats["svg_path"] = out_svg
    return stats


# ── GUI ────────────────────────────────────────────────────────────────────────

# palette
BG      = (24, 25, 28)
PANEL   = (33, 35, 40)
FRAME   = (44, 47, 54)
FRAME_H = (54, 58, 67)
ACCENT  = (91, 141, 239)
ACCENT_H = (116, 161, 245)
ACCENT_A = (71, 121, 219)
TEXT    = (232, 233, 236)
MUTED   = (150, 156, 165)
BORDER  = (55, 58, 66)
GOOD    = (120, 200, 150)


class HatchApp:
    def __init__(self):
        self.img_path = None
        self.result = None
        self._pending = None          # result dict waiting to be shown (main thread)
        self._busy = False
        self.export_ready = False

    # -- worker --------------------------------------------------------------
    def _do_convert(self):
        import dearpygui.dearpygui as dpg
        try:
            size_in = float(dpg.get_value("size_in"))
            # Speed is opt-in: off -> g=0 = the full-quality printed look.
            growth = float(dpg.get_value("growth")) if dpg.get_value("use_speed") else 0.0
            levels = int(dpg.get_value("levels") or 0)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(APP_OUT, f"hatch_{stamp}.svg")
            self._pending = convert(self.img_path, out, size_in, growth, levels)
        except Exception as e:                       # surface, don't crash the loop
            self._pending = {"error": str(e)}
        finally:
            self._busy = False

    # -- texture helper (main thread only) -----------------------------------
    def _show_image(self, tag, png_path, slot, max_w=430, max_h=560):
        import dearpygui.dearpygui as dpg
        for t in (tag + "_img", tag):        # delete image before its texture
            if dpg.does_item_exist(t):
                dpg.delete_item(t)
        if dpg.does_item_exist(slot):
            dpg.delete_item(slot, children_only=True)
        w, h, _c, data = dpg.load_image(png_path)
        scale = min(max_w / w, max_h / h, 1.0)
        with dpg.texture_registry():
            dpg.add_static_texture(w, h, data, tag=tag)
        dpg.add_image(tag, width=int(w * scale), height=int(h * scale),
                      parent=slot, tag=tag + "_img")

    # -- native file dialogs (real Windows Explorer) -------------------------
    @staticmethod
    def _native_open():
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        root.attributes("-topmost", True); root.update()
        path = filedialog.askopenfilename(
            parent=root, title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
                       ("All files", "*.*")])
        root.destroy()
        return path

    @staticmethod
    def _native_save(default_name="hatch.svg"):
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        root.attributes("-topmost", True); root.update()
        path = filedialog.asksaveasfilename(
            parent=root, title="Export SVG", defaultextension=".svg",
            initialfile=default_name,
            filetypes=[("SVG vector", "*.svg"), ("All files", "*.*")])
        root.destroy()
        return path

    # -- callbacks -----------------------------------------------------------
    def _on_load(self):
        import dearpygui.dearpygui as dpg
        try:
            path = self._native_open()
        except Exception as e:
            dpg.set_value("status", f"File dialog error: {e}")
            return
        if not path or not os.path.isfile(path):
            return
        self.img_path = path
        dpg.set_value("filename", os.path.basename(path))
        self._show_image("src_tex", path, "src_slot")
        dpg.configure_item("btn_convert", enabled=True)
        dpg.set_value("status", "Ready. Hit Convert.")

    def _on_toggle_speed(self, sender, val):
        import dearpygui.dearpygui as dpg
        dpg.configure_item("growth", enabled=bool(val))

    def _on_convert(self):
        import dearpygui.dearpygui as dpg
        if self._busy or not self.img_path:
            return
        self._busy = True
        self.export_ready = False
        dpg.configure_item("btn_convert", enabled=False)
        dpg.configure_item("btn_export", enabled=False)
        dpg.set_value("status", "Converting…  (dense images can take a minute)")
        threading.Thread(target=self._do_convert, daemon=True).start()

    def _on_export(self):
        import dearpygui.dearpygui as dpg
        if not (self.result and self.result.get("svg_path")):
            return
        base = os.path.splitext(os.path.basename(self.img_path or "hatch"))[0]
        try:
            dest = self._native_save(default_name=f"{base}_hatch.svg")
        except Exception as e:
            dpg.set_value("status", f"File dialog error: {e}")
            return
        if not dest:
            return
        if not dest.lower().endswith(".svg"):
            dest += ".svg"
        try:
            shutil.copyfile(self.result["svg_path"], dest)
            dpg.set_value("status", f"Exported  →  {dest}")
        except Exception as e:
            dpg.set_value("status", f"Export failed: {e}")

    # -- apply a finished conversion (main thread) ---------------------------
    def _apply(self, res):
        import dearpygui.dearpygui as dpg
        dpg.configure_item("btn_convert", enabled=True)
        if res.get("error"):
            dpg.set_value("status", f"Error: {res['error']}")
            return
        self.result = res
        self._show_image("out_tex", res["preview_png"], "out_slot")
        dpg.set_value("m_paths", f"{res['paths']}")
        dpg.set_value("m_lifts", f"{res['pen_lifts_est']}")
        dpg.set_value("m_draw", f"{res['draw_m']:.0f} m")
        dpg.set_value("m_time", f"~{res['est_hours']:.1f} h")
        dpg.configure_item("btn_export", enabled=True)
        self.export_ready = True
        dpg.set_value("status", f"Done in {res['elapsed_sec']:.1f}s. "
                                f"Review, then Export SVG.")

    # -- theme ---------------------------------------------------------------
    def _theme(self):
        import dearpygui.dearpygui as dpg
        with dpg.theme() as base:
            with dpg.theme_component(dpg.mvAll):
                C = dpg.add_theme_color
                S = dpg.add_theme_style
                cat = dpg.mvThemeCat_Core
                C(dpg.mvThemeCol_WindowBg, BG, category=cat)
                C(dpg.mvThemeCol_ChildBg, PANEL, category=cat)
                C(dpg.mvThemeCol_Border, BORDER, category=cat)
                C(dpg.mvThemeCol_Text, TEXT, category=cat)
                C(dpg.mvThemeCol_TextDisabled, MUTED, category=cat)
                C(dpg.mvThemeCol_FrameBg, FRAME, category=cat)
                C(dpg.mvThemeCol_FrameBgHovered, FRAME_H, category=cat)
                C(dpg.mvThemeCol_FrameBgActive, FRAME_H, category=cat)
                C(dpg.mvThemeCol_Button, FRAME, category=cat)
                C(dpg.mvThemeCol_ButtonHovered, FRAME_H, category=cat)
                C(dpg.mvThemeCol_ButtonActive, ACCENT_A, category=cat)
                C(dpg.mvThemeCol_SliderGrab, ACCENT, category=cat)
                C(dpg.mvThemeCol_SliderGrabActive, ACCENT_H, category=cat)
                C(dpg.mvThemeCol_CheckMark, ACCENT, category=cat)
                C(dpg.mvThemeCol_Header, FRAME, category=cat)
                C(dpg.mvThemeCol_ScrollbarBg, BG, category=cat)
                C(dpg.mvThemeCol_ScrollbarGrab, FRAME, category=cat)
                S(dpg.mvStyleVar_WindowRounding, 0, category=cat)
                S(dpg.mvStyleVar_ChildRounding, 8, category=cat)
                S(dpg.mvStyleVar_FrameRounding, 6, category=cat)
                S(dpg.mvStyleVar_GrabRounding, 6, category=cat)
                S(dpg.mvStyleVar_PopupRounding, 6, category=cat)
                S(dpg.mvStyleVar_WindowPadding, 18, 18, category=cat)
                S(dpg.mvStyleVar_FramePadding, 12, 9, category=cat)
                S(dpg.mvStyleVar_ItemSpacing, 10, 10, category=cat)
                S(dpg.mvStyleVar_ChildBorderSize, 1, category=cat)
        # accent theme for the primary (Convert) button
        with dpg.theme() as accent_btn:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ACCENT_H)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, ACCENT_A)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
        return base, accent_btn

    def _fonts(self):
        import dearpygui.dearpygui as dpg
        body = title = None
        candidates = [r"C:\Windows\Fonts\segoeui.ttf",
                      r"C:\Windows\Fonts\SegoeUI.ttf",
                      r"C:\Windows\Fonts\calibri.ttf"]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path:
            with dpg.font_registry():
                body = dpg.add_font(path, 18)
                title = dpg.add_font(path, 32)
        return body, title

    # -- build ---------------------------------------------------------------
    def _build(self, accent_btn, title_font):
        import dearpygui.dearpygui as dpg

        with dpg.window(tag="main"):
            # header
            t = dpg.add_text("HATCH")
            if title_font:
                dpg.bind_item_font(t, title_font)
            dpg.add_text("photo  →  pen-plotter line art", color=MUTED)
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            with dpg.group(horizontal=True):
                # ── left: controls ──────────────────────────────────────────
                with dpg.child_window(width=330, autosize_y=True, border=True):
                    dpg.add_text("1 · IMAGE", color=MUTED)
                    dpg.add_button(label="  Load image…  ", width=-1,
                                   callback=self._on_load)
                    dpg.add_text("(none)", tag="filename", color=MUTED, wrap=290)
                    dpg.add_spacer(height=12)

                    dpg.add_text("2 · OUTPUT SIZE", color=MUTED)
                    dpg.add_slider_float(tag="size_in", default_value=12.0,
                                         min_value=4.0, max_value=24.0,
                                         format="%.1f in (longest side)", width=-1)
                    dpg.add_spacer(height=12)

                    dpg.add_text("3 · SPEED  (optional)", color=MUTED)
                    dpg.add_checkbox(label="Speed up (less ink in dark areas)",
                                     tag="use_speed", default_value=False,
                                     callback=self._on_toggle_speed)
                    dpg.add_slider_float(
                        tag="growth", default_value=0.30, enabled=False,
                        min_value=0.05, max_value=0.6,
                        format="%.2f", width=-1)
                    dpg.add_text("Off = full quality (the look you printed). "
                                 "On = trade dark-area ink for a faster plot.",
                                 color=MUTED, wrap=290)
                    dpg.add_spacer(height=12)

                    dpg.add_text("4 · DARKNESS LEVELS", color=MUTED)
                    dpg.add_input_int(tag="levels", default_value=0,
                                      min_value=0, max_value=20, step=1, width=-1)
                    dpg.add_text("0 = auto (baseline). Set the real # of tones "
                                 "for flat / graphic art.", color=MUTED, wrap=290)
                    dpg.add_spacer(height=18)

                    b = dpg.add_button(label="CONVERT", width=-1, height=42,
                                       tag="btn_convert", enabled=False,
                                       callback=self._on_convert)
                    dpg.bind_item_theme(b, accent_btn)
                    dpg.add_button(label="Export SVG…", width=-1, height=34,
                                   tag="btn_export", enabled=False,
                                   callback=self._on_export)

                # ── right: previews + metrics ───────────────────────────────
                with dpg.child_window(width=-1, autosize_y=True, border=True):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=-1, height=600, border=False):
                            with dpg.group(horizontal=True):
                                with dpg.group():
                                    dpg.add_text("SOURCE", color=MUTED)
                                    with dpg.child_window(width=450, height=580,
                                                          tag="src_slot",
                                                          border=True):
                                        pass
                                with dpg.group():
                                    dpg.add_text("HATCHED", color=MUTED)
                                    with dpg.child_window(width=-1, height=580,
                                                          tag="out_slot",
                                                          border=True):
                                        pass
                    dpg.add_spacer(height=6)
                    # metrics strip
                    with dpg.group(horizontal=True):
                        for lbl, tag in [("Paths", "m_paths"),
                                         ("Pen lifts", "m_lifts"),
                                         ("Draw length", "m_draw"),
                                         ("Est. plot time", "m_time")]:
                            with dpg.child_window(width=180, height=64, border=True):
                                dpg.add_text(lbl, color=MUTED)
                                v = dpg.add_text("—", tag=tag)
                                if tag == "m_time":
                                    dpg.bind_item_font(v, title_font) if title_font else None

            dpg.add_spacer(height=6)
            dpg.add_text("Load an image to begin.", tag="status", color=GOOD)

    def run(self):
        import dearpygui.dearpygui as dpg
        os.makedirs(APP_OUT, exist_ok=True)
        dpg.create_context()
        base, accent_btn = self._theme()
        body, title = self._fonts()
        if body:
            dpg.bind_font(body)
        dpg.bind_theme(base)
        self._build(accent_btn, title)
        dpg.create_viewport(title="Hatch", width=1240, height=820,
                            min_width=980, min_height=680)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        while dpg.is_dearpygui_running():
            if self._pending is not None:
                res, self._pending = self._pending, None
                self._apply(res)
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def _selftest():
    """Headless: exercise the convert + estimate path (no GUI) for both the
    default (speed OFF = full printed look) and speed-ON settings."""
    img = os.path.join(_REPO, "tests", "images", "portrait_lowkey.jpg")
    for label, g in [("default / speed OFF (g=0.0)", 0.0),
                     ("speed ON (g=0.30)", 0.30)]:
        out = os.path.join(APP_OUT, "_selftest.svg")
        s = convert(img, out, size_in=12.0, growth=g, levels=0)
        print(f"{label:28s} paths={s['paths']:3d} lifts={s['pen_lifts_est']:4d} "
              f"draw={s['draw_m']:4.0f} m  est~{s['est_hours']:.1f} h")
        assert os.path.exists(s["svg_path"]) and os.path.exists(s["preview_png"])
    print("SELFTEST OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        HatchApp().run()
