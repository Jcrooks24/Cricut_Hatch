"""
Batch testing cockpit — one window to Run -> Grade -> Export.

    py -3.11 tests/batch_ui.py

Workflow:
    1. Pick presets + images, hit RUN BATCH. Generation runs on a worker
       thread; each result appears as a card with its preview + metrics.
    2. Grade each card 1-5 on tone / detail / clean / overall, add notes.
    3. SAVE GRADES  -> appends to results/grades.jsonl (history).
       EXPORT FOR CLAUDE  -> writes results/feedback_<run>.md (+ .json) AND
       copies the markdown to your clipboard, ready to paste into Claude Code
       for targeted adjustments.
    4. PROMOTE BASELINE -> accept this run's metrics/grades as the reference.

This is a separate app from hatch_ui_nocairo.py on purpose: the test cockpit
must never be able to regress the generator it is testing.
"""
from __future__ import annotations

import os
import sys
import json
import queue
import threading
import traceback

import dearpygui.dearpygui as dpg
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ on path
import harness as H
import presets as P
from hatch_ui_nocairo import hatch_pipeline

AXES = ["tone", "detail", "clean", "overall"]
THUMB_MAX = 320


def _make_thumb(preview_path: str, thumb_path: str):
    try:
        im = Image.open(preview_path).convert("RGB")
        im.thumbnail((THUMB_MAX, THUMB_MAX))
        im.save(thumb_path)
        return im.size
    except Exception:
        return None


class BatchTester:
    def __init__(self):
        self.results = []          # list of record dicts
        self.run_label = None
        self.q = queue.Queue()
        self.running = False

    # ── batch generation (worker thread) ─────────────────────────────────────
    def start(self, preset_names, image_paths):
        if self.running:
            return
        self.running = True
        self.results = []
        self.run_label = f"batch_{H.now_stamp()}"
        t = threading.Thread(target=self._run, args=(preset_names, image_paths),
                             daemon=True)
        t.start()

    def _run(self, preset_names, image_paths):
        run_dir = os.path.join(H.RUNS_DIR, self.run_label)
        os.makedirs(run_dir, exist_ok=True)
        total = len(preset_names) * len(image_paths)
        done = 0
        for preset_name in preset_names:
            cfg = P.get(preset_name)
            for img_path in image_paths:
                key = H.image_key(img_path)
                stem = f"{key}__{preset_name}"
                out_svg = os.path.join(run_dir, stem + ".svg")
                self.q.put(("status", f"Generating {stem} ..."))
                try:
                    stats = hatch_pipeline(img_path, out_svg, cfg)
                    metrics = H.compute_metrics(out_svg, stats)
                    fails = H.hard_check(metrics)
                    preview = os.path.splitext(out_svg)[0] + "_preview.png"
                    thumb = os.path.splitext(out_svg)[0] + "_thumb.png"
                    if os.path.exists(preview):
                        _make_thumb(preview, thumb)
                    rec = {
                        "id": stem, "image": key, "preset": preset_name,
                        "svg": out_svg, "preview": preview if os.path.exists(preview) else None,
                        "thumb": thumb if os.path.exists(thumb) else None,
                        "metrics": metrics, "hard_fails": fails,
                        "config": H.config_fingerprint(cfg),
                    }
                except Exception as e:
                    traceback.print_exc()
                    rec = {"id": stem, "image": key, "preset": preset_name,
                           "error": str(e), "metrics": {}, "hard_fails": [],
                           "config": H.config_fingerprint(cfg)}
                self.results.append(rec)
                done += 1
                self.q.put(("result", rec))
                self.q.put(("progress", done / max(1, total)))
        # persist manifest for compare.py / grade.py reuse
        H.write_run_manifest(run_dir, {
            "run_label": self.run_label, "git_hash": H.git_hash(),
            "created": H.now_stamp(), "records": self.results,
        })
        self.q.put(("done", self.run_label))
        self.running = False

    # ── grades + export ──────────────────────────────────────────────────────
    def collect_grades(self, get_value):
        """Read grade widgets for every result -> list of graded records."""
        graded = []
        for rec in self.results:
            if rec.get("error"):
                continue
            scores = {ax: int(get_value(f"score_{rec['id']}_{ax}")) for ax in AXES}
            note = get_value(f"note_{rec['id']}") or ""
            graded.append({**rec, "scores": scores, "note": note})
        return graded

    def save_grades(self, graded):
        n = 0
        for g in graded:
            if not any(g["scores"].values()):
                continue  # skip fully ungraded
            H.append_grade({
                "run_label": self.run_label, "git_hash": H.git_hash(),
                "image": g["image"], "preset": g["preset"],
                "scores": g["scores"], "note": g["note"],
                "metrics": g["metrics"], "graded_at": H.now_stamp(),
            })
            n += 1
        return n

    def export_feedback(self, graded):
        baseline = H.load_baseline()
        base_defaults = H.config_fingerprint(P.get("baseline"))
        md = []
        md.append(f"# Hatch batch feedback — {self.run_label} (git {H.git_hash()})\n")
        md.append("Grades are 1-5 (5 = best). Use these to adjust the image-"
                  "processing methodology in hatch_ui_nocairo.py.\n")
        md.append("## Summary\n")
        md.append("| image | preset | overall | tone | detail | clean | paths | svg_kb | d_over vs base |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for g in sorted(graded, key=lambda r: (r["image"], r["preset"])):
            s, m = g["scores"], g["metrics"]
            bkey = H.baseline_key(g["image"], g["preset"])
            b_over = baseline.get(bkey, {}).get("overall")
            d = (s["overall"] - b_over) if (b_over and s["overall"]) else None
            md.append(f"| {g['image']} | {g['preset']} | {s['overall'] or '-'} | "
                      f"{s['tone'] or '-'} | {s['detail'] or '-'} | {s['clean'] or '-'} | "
                      f"{m.get('paths','-')} | {m.get('svg_kb','-')} | "
                      f"{('%+d' % d) if d is not None else '-'} |")
        md.append("\n## Per-image notes\n")
        for g in sorted(graded, key=lambda r: (r["image"], r["preset"])):
            s, m = g["scores"], g["metrics"]
            md.append(f"### {g['image']} / {g['preset']}")
            md.append(f"- scores: tone={s['tone']} detail={s['detail']} "
                      f"clean={s['clean']} overall={s['overall']}")
            md.append(f"- metrics: paths={m.get('paths')} pen_lifts={m.get('pen_lifts_est')} "
                      f"svg_kb={m.get('svg_kb')} elapsed={m.get('elapsed_sec')}s")
            if g["hard_fails"]:
                md.append(f"- HARD FAILS: {'; '.join(g['hard_fails'])}")
            if g["note"]:
                md.append(f"- note: {g['note']}")
            # config diff vs baseline defaults (only what this preset changed)
            diff = {k: v for k, v in g["config"].items()
                    if base_defaults.get(k) != v}
            if diff:
                md.append(f"- config changes vs baseline: {json.dumps(diff)}")
            md.append("")
        text = "\n".join(md)

        out_md = os.path.join(H.RESULTS_DIR, f"feedback_{self.run_label}.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(H.RESULTS_DIR, f"feedback_{self.run_label}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(graded, f, indent=2)
        return out_md, text


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class BatchApp:
    def __init__(self):
        self.bt = BatchTester()
        self.images = H.discover_images()
        self.textures = {}  # id -> texture tag

    def _selected_presets(self):
        return [n for n in sorted(P.PRESETS) if dpg.get_value(f"preset_{n}")]

    def _selected_images(self):
        return [p for p in self.images
                if dpg.get_value(f"image_{H.image_key(p)}")]

    # ── callbacks ────────────────────────────────────────────────────────────
    def on_run(self):
        presets = self._selected_presets()
        images = self._selected_images()
        if not presets or not images:
            dpg.set_value("status", "Select at least one preset and one image.")
            return
        dpg.delete_item("results", children_only=True)
        self.textures.clear()
        dpg.set_value("progress", 0.0)
        dpg.configure_item("btn_run", enabled=False)
        dpg.set_value("status", f"Running {len(images)}x{len(presets)} ...")
        self.bt.start(presets, images)

    def on_save(self):
        graded = self.bt.collect_grades(dpg.get_value)
        n = self.bt.save_grades(graded)
        dpg.set_value("status", f"Saved {n} grade(s) to results/grades.jsonl")

    def on_export(self):
        graded = self.bt.collect_grades(dpg.get_value)
        if not graded:
            dpg.set_value("status", "Nothing to export — run a batch first.")
            return
        self.bt.save_grades(graded)
        path, text = self.bt.export_feedback(graded)
        try:
            dpg.set_clipboard_text(text)
            clip = " (copied to clipboard)"
        except Exception:
            clip = ""
        dpg.set_value("status", f"Exported {os.path.basename(path)}{clip}")
        try:
            os.startfile(path)  # Windows
        except Exception:
            pass

    def on_promote(self):
        # Reuse compare.promote logic via baseline dict.
        graded = self.bt.collect_grades(dpg.get_value)
        baseline = H.load_baseline()
        for g in graded:
            baseline[H.baseline_key(g["image"], g["preset"])] = {
                "image": g["image"], "preset": g["preset"],
                "metrics": g["metrics"], "overall": g["scores"]["overall"] or None,
                "git_hash": H.git_hash(), "run_label": self.bt.run_label,
                "promoted": H.now_stamp(),
            }
        H.save_baseline(baseline)
        dpg.set_value("status", f"Promoted {len(graded)} item(s) to baseline.")

    def _open_preview(self, sender, app_data, user_data):
        try:
            os.startfile(user_data)  # full preview path
        except Exception:
            pass

    # ── build a result card (main thread) ────────────────────────────────────
    def _add_card(self, rec):
        rid = rec["id"]
        with dpg.child_window(parent="results", width=-1, height=200,
                              border=True):
            with dpg.group(horizontal=True):
                # thumbnail
                if rec.get("thumb") and os.path.exists(rec["thumb"]):
                    try:
                        w, h, c, data = dpg.load_image(rec["thumb"])
                        tag = f"tex_{rid}"
                        with dpg.texture_registry():
                            dpg.add_static_texture(w, h, data, tag=tag)
                        self.textures[rid] = tag
                        dpg.add_image(tag, width=w, height=h)
                    except Exception:
                        dpg.add_text("(preview load failed)")
                else:
                    dpg.add_text("(no preview)")

                # info + grading
                with dpg.group():
                    m = rec.get("metrics", {})
                    dpg.add_text(f"{rec['image']}   |   preset: {rec['preset']}",
                                 color=(120, 200, 255))
                    if rec.get("error"):
                        dpg.add_text(f"ERROR: {rec['error']}", color=(255, 120, 120))
                    else:
                        flag = ("   HARD FAIL: " + "; ".join(rec["hard_fails"])
                                if rec["hard_fails"] else "")
                        dpg.add_text(
                            f"paths={m.get('paths')}  pen_lifts={m.get('pen_lifts_est')}  "
                            f"svg_kb={m.get('svg_kb')}  {m.get('elapsed_sec')}s{flag}",
                            color=(255, 170, 120) if flag else (200, 200, 200))
                        for ax in AXES:
                            dpg.add_slider_int(label=ax, min_value=0, max_value=5,
                                               default_value=0, width=220,
                                               tag=f"score_{rid}_{ax}")
                        dpg.add_input_text(hint="note (optional)", width=320,
                                           tag=f"note_{rid}")
                        if rec.get("preview"):
                            dpg.add_button(label="Open full preview",
                                           user_data=rec["preview"],
                                           callback=self._open_preview)

    # ── queue flush (main thread, every frame) ───────────────────────────────
    def _flush(self):
        try:
            while True:
                kind, payload = self.bt.q.get_nowait()
                if kind == "status":
                    dpg.set_value("status", payload)
                elif kind == "progress":
                    dpg.set_value("progress", payload)
                elif kind == "result":
                    self._add_card(payload)
                elif kind == "done":
                    dpg.set_value("status", f"Done. Run '{payload}'. Grade, then Export.")
                    dpg.configure_item("btn_run", enabled=True)
        except queue.Empty:
            pass

    def _build(self):
        with dpg.window(tag="main"):
            dpg.add_text("Batch Test Cockpit — Run -> Grade -> Export", color=(120, 200, 255))
            with dpg.collapsing_header(label="1. Select presets", default_open=True):
                for n in sorted(P.PRESETS):
                    dpg.add_checkbox(label=n, tag=f"preset_{n}",
                                     default_value=(n == "baseline"))
            with dpg.collapsing_header(label="2. Select images", default_open=True):
                if not self.images:
                    dpg.add_text("No images in tests/images/ — add some first.",
                                 color=(255, 170, 120))
                for p in self.images:
                    k = H.image_key(p)
                    dpg.add_checkbox(label=k, tag=f"image_{k}", default_value=True)

            with dpg.group(horizontal=True):
                dpg.add_button(label="RUN BATCH", tag="btn_run", callback=self.on_run,
                               width=140, height=32)
                dpg.add_button(label="Save grades", callback=self.on_save, height=32)
                dpg.add_button(label="Export for Claude", callback=self.on_export,
                               width=170, height=32)
                dpg.add_button(label="Promote baseline", callback=self.on_promote,
                               height=32)
            dpg.add_progress_bar(tag="progress", default_value=0.0, width=-1)
            dpg.add_text("", tag="status")
            dpg.add_separator()
            dpg.add_text("Results")
            dpg.add_child_window(tag="results", width=-1, height=-1, border=False)

    def run(self):
        dpg.create_context()
        self._build()
        dpg.create_viewport(title="Hatch Batch Cockpit", width=1180, height=900,
                            min_width=800, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)
        while dpg.is_dearpygui_running():
            self._flush()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def main():
    BatchApp().run()


if __name__ == "__main__":
    main()
