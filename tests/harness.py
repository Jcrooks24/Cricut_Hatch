"""
Shared helpers for the Cricut_Hatch test / grading harness.

Nothing here touches the GUI — everything drives `hatch_pipeline` directly so
runs are reproducible and headless.
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

# Make the repo root importable so `import hatch_ui_nocairo` works no matter
# where the script is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

IMAGES_DIR   = os.path.join(REPO_ROOT, "tests", "images")
RESULTS_DIR  = os.path.join(REPO_ROOT, "results")
RUNS_DIR     = os.path.join(RESULTS_DIR, "runs")
GRADES_PATH  = os.path.join(RESULTS_DIR, "grades.jsonl")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline.json")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Cricut hard limits — exceeding these is an automatic FAIL, not a judgement call.
HARD_MAX_PATHS   = 5000
WARN_SVG_KB      = 1500   # files past here have historically risked Design Space crashes


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def discover_images() -> List[str]:
    if not os.path.isdir(IMAGES_DIR):
        return []
    files = [
        os.path.join(IMAGES_DIR, f)
        for f in sorted(os.listdir(IMAGES_DIR))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    return files


def image_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(svg_path: str, stats: Dict) -> Dict:
    """Objective, deterministic metrics for one generated SVG."""
    svg_bytes = os.path.getsize(svg_path) if os.path.exists(svg_path) else 0
    return {
        "paths":         int(stats.get("paths", 0)),
        "pen_lifts_est": int(stats.get("pen_lifts_est", 0)),
        "svg_kb":        round(svg_bytes / 1024.0, 1),
        "elapsed_sec":   round(float(stats.get("elapsed_sec", 0.0)), 2),
        "work_w":        int(stats.get("work_w", 0)),
        "work_h":        int(stats.get("work_h", 0)),
        "aux_used":      int(stats.get("aux_used", 0)),
    }


def hard_check(metrics: Dict) -> List[str]:
    """Return a list of hard-limit violations (empty == passes)."""
    fails = []
    if metrics.get("paths", 0) > HARD_MAX_PATHS:
        fails.append(f"paths {metrics['paths']} > hard cap {HARD_MAX_PATHS}")
    if metrics.get("svg_kb", 0) > WARN_SVG_KB:
        fails.append(f"svg_kb {metrics['svg_kb']} > warn threshold {WARN_SVG_KB}")
    return fails


# ── Baseline (the current "accepted best" per image+preset) ────────────────────

def load_baseline() -> Dict:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_baseline(baseline: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)


def baseline_key(image: str, preset: str) -> str:
    return f"{image}::{preset}"


# ── Grades (append-only human feedback log) ────────────────────────────────────

def append_grade(record: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(GRADES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_grades() -> List[Dict]:
    if not os.path.exists(GRADES_PATH):
        return []
    out = []
    with open(GRADES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def latest_grade(image: str, preset: str) -> Optional[Dict]:
    """Most recent grade for an image+preset, or None."""
    matches = [
        g for g in load_grades()
        if g.get("image") == image and g.get("preset") == preset
    ]
    return matches[-1] if matches else None


# ── Run-folder I/O ─────────────────────────────────────────────────────────────

def write_run_manifest(run_dir: str, manifest: Dict) -> None:
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_run_manifest(run_dir: str) -> Dict:
    with open(os.path.join(run_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def latest_run_dir() -> Optional[str]:
    if not os.path.isdir(RUNS_DIR):
        return None
    subs = [
        os.path.join(RUNS_DIR, d) for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
        and os.path.exists(os.path.join(RUNS_DIR, d, "manifest.json"))
    ]
    if not subs:
        return None
    return max(subs, key=os.path.getmtime)


def config_fingerprint(cfg) -> Dict:
    """Full config as a dict, so every run records exactly what produced it."""
    try:
        return asdict(cfg)
    except Exception:
        return {}
