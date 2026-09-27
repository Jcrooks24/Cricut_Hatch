# Cricut_Hatch

Converts a photo into a **pen-and-ink hatched SVG** tuned for a Cricut pen
plotter. The image is split into tone bands, each band filled with single-line
hatching (crosshatch in the darks) plus edge/detail aux layers. The controlling
design constraint: **1 polygon = 1 SVG path = 1 pen lift**, hard-capped at
**5000 paths**, because large / many-path files crash Cricut Design Space.

The interactive app is `hatch_ui_nocairo.py` (DearPyGui). The same pipeline is
callable headlessly via `hatch_pipeline(png, out_svg, cfg)`, which is what the
test harness uses.

## Install

```bash
py -3.11 -m pip install -r requirements.txt
```

## Run the app

```bash
py -3.11 hatch_ui_nocairo.py
```

## The test / grading workflow

The point of this repo is to **stop regressions**. Every change to the image-
processing methodology is measured against a fixed set of test images, on both
objective metrics (path count, pen-lifts, file size) and recorded human grades.
A change that improves one subject but wrecks another shows up immediately.

```
tests/
  presets.py     named HatchConfig variants (start from "baseline", tweak knobs)
  run_suite.py   batch-generate SVG + preview PNG for every image x preset
  grade.py       score each preview 1-5 (tone / detail / clean / overall)
  compare.py     current run vs baseline; flags regressions; --promote to accept
  harness.py     shared metrics / baseline / grades I/O
  images/        drop your 5-10 test images here
results/
  runs/          generated outputs per run (git-ignored, regenerable)
  grades.jsonl   append-only human feedback history (tracked)
  baseline.json  current accepted metrics + grades per image (tracked)
```

### Loop

```bash
# 1. Put 5-10 varied test images in tests/images/  (portrait, landscape,
#    high-contrast, low-contrast, fine texture, etc.)

# 2. Establish the starting point
py -3.11 tests/run_suite.py --preset baseline
py -3.11 tests/grade.py
py -3.11 tests/compare.py --promote        # baseline is now the reference

# 3. Try an idea: add a preset in presets.py, then
py -3.11 tests/run_suite.py --preset all
py -3.11 tests/grade.py
py -3.11 tests/compare.py                  # deltas + regression flags

# 4. If the new methodology wins on every image, accept it
py -3.11 tests/compare.py --promote
```

See `docs/methodology.md` for the full rationale.
