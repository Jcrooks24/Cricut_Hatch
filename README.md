# Cricut_Hatch

Converts a photo into a **pen-and-ink hatched SVG** tuned for a Cricut pen
plotter, keeping the path/pen-lift count low because large files crash Cricut
Design Space.

**Recommended mode: `tonal`** — clean cross-hatch engraving-style rendering.
Polygonize tonal regions → darkness sets how many overlapping angled layers a
region gets → each layer is stitched into ONE path with a greedy
nearest-neighbour tour (few pen lifts) → each layer uses a different angle so
passes don't overlap. Produces the most recognizable output in just a handful of
paths. See `run_tonal` in `hatch_ui_nocairo.py` and the `tonal` preset.

Other modes: `line_art` (HED edge contours + shading) and the original tone-band
`baseline` (kept for reference / A-B).

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

### Batch cockpit (recommended)

One window: run the whole batch, grade each result, export the grades.

```bash
py -3.11 tests/batch_ui.py
```

1. Tick presets + images, hit **RUN BATCH** (generation runs on a worker
   thread with a progress bar).
2. Each result appears as a card with its preview + metrics; grade it 1-5 on
   tone / detail / clean / overall and add notes.
3. **Export for Claude** writes `results/feedback_<run>.md` and copies it to
   your clipboard — paste that straight into Claude Code to drive adjustments.
   **Promote baseline** accepts the run as the new reference.

Timing note: dense images (e.g. a crowded scene) can take ~1-2 min each, so a
full baseline pass over ~6 images is a few minutes. Grade while it fills in.

The command-line tools below do the same thing in steps, if you prefer scripts.

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
