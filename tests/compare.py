"""
Compare a run against the accepted baseline and flag regressions.

The baseline stores, per (image, preset), the accepted objective metrics and
the latest human "overall" grade. When you try a new methodology:

    1. py -3.11 tests/run_suite.py --preset all
    2. py -3.11 tests/grade.py
    3. py -3.11 tests/compare.py           # see deltas vs baseline, flagged
    4. py -3.11 tests/compare.py --promote # if it wins everywhere, accept it

A regression = any image whose overall grade dropped, or that now violates a
Cricut hard limit. Because compare looks at EVERY image, a change that helps
one subject but hurts another can't sneak through.

Usage:
    py -3.11 tests/compare.py                    # latest run vs baseline
    py -3.11 tests/compare.py --run <label>
    py -3.11 tests/compare.py --promote          # set this run as new baseline
    py -3.11 tests/compare.py --promote --preset baseline   # promote one preset
"""
from __future__ import annotations

import os
import argparse

import harness as H


def _overall(image, preset):
    g = H.latest_grade(image, preset)
    if not g:
        return None
    return g.get("scores", {}).get("overall") or None


def build_current(run_dir):
    manifest = H.read_run_manifest(run_dir)
    current = {}
    for rec in manifest["records"]:
        if rec.get("error"):
            continue
        key = H.baseline_key(rec["image"], rec["preset"])
        current[key] = {
            "image":   rec["image"],
            "preset":  rec["preset"],
            "metrics": rec.get("metrics", {}),
            "hard_fails": rec.get("hard_fails", []),
            "overall": _overall(rec["image"], rec["preset"]),
        }
    return manifest, current


def show(run_dir):
    manifest, current = build_current(run_dir)
    baseline = H.load_baseline()
    print(f"\nRun '{manifest['run_label']}'  git={manifest.get('git_hash')}  vs baseline\n")
    hdr = f"{'image::preset':38} {'paths':>7} {'Δpaths':>8} {'svg_kb':>8} {'overall':>8} {'Δover':>7}  flags"
    print(hdr)
    print("-" * len(hdr))

    regressions = []
    for key, cur in sorted(current.items()):
        base = baseline.get(key, {})
        bm, cm = base.get("metrics", {}), cur["metrics"]
        d_paths = cm.get("paths", 0) - bm.get("paths", 0) if bm else 0
        b_over  = base.get("overall")
        c_over  = cur["overall"]
        d_over  = (c_over - b_over) if (b_over and c_over) else 0

        flags = list(cur["hard_fails"])
        if b_over and c_over and c_over < b_over:
            flags.append(f"GRADE DOWN {b_over}->{c_over}")
            regressions.append(key)
        if cur["hard_fails"]:
            regressions.append(key)

        print(f"{key:38} {cm.get('paths',0):>7} {d_paths:>+8} "
              f"{cm.get('svg_kb',0):>8.1f} "
              f"{(str(c_over) if c_over else '-'):>8} "
              f"{(f'{d_over:+d}' if d_over else '-'):>7}  {'; '.join(flags)}")

    print()
    if regressions:
        print(f"REGRESSIONS on {len(set(regressions))} item(s): "
              f"{', '.join(sorted(set(regressions)))}")
    else:
        print("No regressions vs baseline.")
    return manifest, current


def promote(run_dir, only_preset):
    manifest, current = build_current(run_dir)
    baseline = H.load_baseline()
    n = 0
    for key, cur in current.items():
        if only_preset and cur["preset"] != only_preset:
            continue
        baseline[key] = {
            "image":     cur["image"],
            "preset":    cur["preset"],
            "metrics":   cur["metrics"],
            "overall":   cur["overall"],
            "git_hash":  manifest.get("git_hash"),
            "run_label": manifest["run_label"],
            "promoted":  H.now_stamp(),
        }
        n += 1
    H.save_baseline(baseline)
    print(f"Promoted {n} item(s) into baseline -> {H.BASELINE_PATH}")


def main():
    ap = argparse.ArgumentParser(description="Compare a run to baseline.")
    ap.add_argument("--run", default=None, help="Run label. Default: latest.")
    ap.add_argument("--promote", action="store_true",
                    help="Accept this run's results as the new baseline.")
    ap.add_argument("--preset", default=None,
                    help="With --promote, only promote this preset.")
    args = ap.parse_args()

    run_dir = (os.path.join(H.RUNS_DIR, args.run) if args.run
               else H.latest_run_dir())
    if not run_dir or not os.path.isdir(run_dir):
        print("No run found. Generate one first:  py -3.11 tests/run_suite.py")
        return

    show(run_dir)
    if args.promote:
        promote(run_dir, args.preset)


if __name__ == "__main__":
    main()
