# Cricut_Hatch

Converts a photo into a **pen-and-ink hatched SVG** tuned for a Cricut pen
plotter, keeping the path/pen-lift count low because large files crash Cricut
Design Space.

**The method — `run_tonal` (tonal cross-hatch):** polygonize tonal regions →
darkness sets how many overlapping angled layers a region gets → each layer is
stitched into ONE path with a greedy nearest-neighbour tour (few pen lifts) →
each layer uses a different angle so passes don't overlap. Options:

- **adaptive thresholds** — layer thresholds from the image's tonal quantiles
  (fixes near-solid low-key subjects)
- **adaptive min-area** — crisp detailed foreground, de-speckled flat grounds
- **dither + dither-floor** — anti-banding stipple that leaves true blacks solid
- **dark-region trace** — outline the darkest shapes for crisp definition
- **darkness-level count** — set the exact number of tones (flat/graphic images)
- **single-path mode** — collapse everything into ONE continuous stroke (1 pen
  lift), then cut the visible connectors in the stitch-break tool
- **SVG compression** — coordinate rounding + implicit lineto for small files

`hatch_ui_nocairo.py` is a library; call `hatch_pipeline(png, out_svg, cfg)`.

## The Hatch app (the actual tool)

```bash
py -3.11 hatch_app.py
```

A standalone desktop app wired to the printed **baseline**: load one photo →
Convert → see a live preview, path/pen-lift count, drawn length and an estimated
Cricut plot time → **Export SVG**. A few dials cover what matters for a print:

- **Output size** — longest side in inches.
- **Speed / ink** — deep-layer spacing growth; higher = less ink in the darkest
  areas = faster plot (baseline 0.30; a 6 h print drops to ~2 h at this setting).
- **Darkness levels** — 0 = auto; set the real tone count for flat / graphic art.

(`py -3.11 hatch_app.py --selftest` runs the convert/estimate path headless.)

The **batch cockpit** (`tests/batch_ui.py`) and **stitch-break tool**
(`tests/break_tool.py`) are the test/grading front-ends, below.

## Install

```bash
py -3.11 -m pip install -r requirements.txt
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
