"""
Interactive grading of a run's previews. Each output is scored 1-5 on four
axes; scores + notes are appended to results/grades.jsonl (append-only, so you
keep a full history of how the methodology evolved).

Usage:
    py -3.11 tests/grade.py                 # grade the most recent run
    py -3.11 tests/grade.py --run baseline_20260927_120000
    py -3.11 tests/grade.py --no-open       # don't auto-open preview images

Axes (1 = bad, 5 = excellent):
    tone      - tonal accuracy: do light/dark regions match the photo?
    detail    - is meaningful detail preserved (eyes, texture, edges)?
    clean     - freedom from artifacts (seams, moire, noise, stray marks)
    overall   - your gut headline score; this is what regression checks track
"""
from __future__ import annotations

import os
import argparse

import harness as H

AXES = ["tone", "detail", "clean", "overall"]


def ask_score(axis: str) -> int:
    while True:
        raw = input(f"    {axis:8} [1-5, Enter=skip]: ").strip()
        if raw == "":
            return 0
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("      enter 1-5, or Enter to skip")


def grade_run(run_dir: str, open_previews: bool):
    manifest = H.read_run_manifest(run_dir)
    records = [r for r in manifest["records"] if not r.get("error")]
    label = manifest["run_label"]
    print(f"\nGrading run '{label}'  ({len(records)} outputs)  git={manifest.get('git_hash')}")
    print("Enter 1-5 per axis (blank skips an axis). Ctrl-C to stop.\n")

    for i, rec in enumerate(records, 1):
        img, preset = rec["image"], rec["preset"]
        m = rec.get("metrics", {})
        fails = rec.get("hard_fails", [])
        print(f"[{i}/{len(records)}] {img}  |  preset={preset}")
        print(f"    paths={m.get('paths')}  pen_lifts={m.get('pen_lifts_est')}  "
              f"svg_kb={m.get('svg_kb')}  {'HARD FAIL: ' + '; '.join(fails) if fails else ''}")
        preview = rec.get("preview")
        if preview:
            abspath = os.path.join(H.REPO_ROOT, preview)
            print(f"    preview: {abspath}")
            if open_previews and os.path.exists(abspath):
                try:
                    os.startfile(abspath)  # Windows
                except Exception:
                    pass

        scores = {axis: ask_score(axis) for axis in AXES}
        note = input("    note (optional): ").strip()

        H.append_grade({
            "run_label": label,
            "git_hash":  manifest.get("git_hash"),
            "image":     img,
            "preset":    preset,
            "scores":    scores,
            "note":      note,
            "metrics":   m,
            "graded_at": H.now_stamp(),
        })
        print("    saved.\n")

    print(f"All grades appended to {H.GRADES_PATH}")
    print(f"Next:  py -3.11 tests/compare.py --run {label}")


def main():
    ap = argparse.ArgumentParser(description="Grade a run's previews.")
    ap.add_argument("--run", default=None,
                    help="Run label (folder in results/runs). Default: latest.")
    ap.add_argument("--no-open", action="store_true",
                    help="Do not auto-open preview images.")
    args = ap.parse_args()

    run_dir = (os.path.join(H.RUNS_DIR, args.run) if args.run
               else H.latest_run_dir())
    if not run_dir or not os.path.isdir(run_dir):
        print("No run found. Generate one first:  py -3.11 tests/run_suite.py")
        return
    grade_run(run_dir, open_previews=not args.no_open)


if __name__ == "__main__":
    main()
