"""
Batch-generate hatched SVGs + preview PNGs for every test image x preset.

Usage:
    py -3.11 tests/run_suite.py                      # baseline preset, all images
    py -3.11 tests/run_suite.py --preset all         # every preset in presets.py
    py -3.11 tests/run_suite.py --preset contour_flow raster
    py -3.11 tests/run_suite.py --images portrait_01 cat_02
    py -3.11 tests/run_suite.py --open               # open each preview when done

Output goes to results/runs/<run_label>/ with a manifest.json capturing the
exact config + metrics for every output. Hard-limit violations (path count,
file size) are flagged immediately so a broken run never reaches grading.
"""
from __future__ import annotations

import os
import argparse
import traceback

import harness as H
import presets as P
from hatch_ui_nocairo import hatch_pipeline


def run(preset_names, image_filter, run_label, open_previews):
    images = H.discover_images()
    if image_filter:
        images = [p for p in images if H.image_key(p) in set(image_filter)]
    if not images:
        print(f"No images found in {H.IMAGES_DIR}")
        print("Drop 5-10 test images (png/jpg) there first.")
        return

    if preset_names == ["all"]:
        preset_names = sorted(P.PRESETS)

    run_dir = os.path.join(H.RUNS_DIR, run_label)
    os.makedirs(run_dir, exist_ok=True)

    records = []
    print(f"Run '{run_label}'  |  {len(images)} image(s) x {len(preset_names)} preset(s)")
    print(f"Output: {run_dir}\n")

    for preset_name in preset_names:
        cfg = P.get(preset_name)
        for img_path in images:
            key = H.image_key(img_path)
            stem = f"{key}__{preset_name}"
            out_svg = os.path.join(run_dir, stem + ".svg")
            print(f"  [{preset_name:14}] {key} ...", end="", flush=True)
            try:
                stats = hatch_pipeline(img_path, out_svg, cfg)
                metrics = H.compute_metrics(out_svg, stats)
                fails = H.hard_check(metrics)
                preview = os.path.splitext(out_svg)[0] + "_preview.png"
                records.append({
                    "image":   key,
                    "preset":  preset_name,
                    "svg":     os.path.relpath(out_svg, H.REPO_ROOT),
                    "preview": os.path.relpath(preview, H.REPO_ROOT)
                               if os.path.exists(preview) else None,
                    "metrics": metrics,
                    "hard_fails": fails,
                    "config":  H.config_fingerprint(cfg),
                })
                flag = "  !! " + "; ".join(fails) if fails else ""
                print(f" {metrics['paths']:>5} paths, "
                      f"{metrics['svg_kb']:>7.1f} KB, "
                      f"{metrics['elapsed_sec']:>5.1f}s{flag}")
                if open_previews and os.path.exists(preview):
                    try:
                        os.startfile(preview)  # Windows
                    except Exception:
                        pass
            except Exception as e:
                print(f" ERROR: {e}")
                traceback.print_exc()
                records.append({
                    "image": key, "preset": preset_name,
                    "error": str(e),
                })

    manifest = {
        "run_label": run_label,
        "git_hash":  H.git_hash(),
        "created":   H.now_stamp(),
        "records":   records,
    }
    H.write_run_manifest(run_dir, manifest)

    n_fail = sum(1 for r in records if r.get("hard_fails") or r.get("error"))
    print(f"\nDone. {len(records)} output(s), {n_fail} with hard fails/errors.")
    print(f"Manifest: {os.path.join(run_dir, 'manifest.json')}")
    print(f"Next:  py -3.11 tests/grade.py --run {run_label}")


def main():
    ap = argparse.ArgumentParser(description="Batch-generate the hatch test suite.")
    ap.add_argument("--preset", nargs="+", default=["baseline"],
                    help="Preset name(s), or 'all'. Default: baseline")
    ap.add_argument("--images", nargs="*", default=None,
                    help="Restrict to these image keys (filename without ext).")
    ap.add_argument("--run-label", default=None,
                    help="Folder name for this run. Default: <preset>_<timestamp>")
    ap.add_argument("--open", action="store_true",
                    help="Open each preview PNG when generated.")
    args = ap.parse_args()

    label = args.run_label
    if not label:
        tag = "multi" if (args.preset == ["all"] or len(args.preset) > 1) else args.preset[0]
        label = f"{tag}_{H.now_stamp()}"

    run(args.preset, args.images, label, args.open)


if __name__ == "__main__":
    main()
