# Methodology: iterating without regressions

## The problem this solves

Tuning the hatching pipeline by eye on a single image caused **regressions**:
a change that made one photo look better quietly made others look worse, and
because only one image was ever in view, the damage wasn't noticed until later.

## The fix: a fixed test set with measured outcomes

Every candidate change is evaluated against the **same** suite of test images,
on two kinds of signal:

1. **Objective metrics** (deterministic, no judgement):
   - `paths` — must stay under the 5000 hard cap (Cricut crash risk).
   - `pen_lifts_est` — lower is better; the whole point of stitching.
   - `svg_kb` — file size; large files also crash Design Space.
   - `elapsed_sec` — keep generation practical.

2. **Human grades** (1-5), because "looks good" is the real target:
   - `tone` — do light/dark regions match the photo?
   - `detail` — is meaningful detail preserved (eyes, texture, edges)?
   - `clean` — free of artifacts (band seams, moire, noise, stray marks)?
   - `overall` — headline score; this is what regression detection tracks.

A change is only accepted when it wins — or at least doesn't lose — across
**all** images. `compare.py` flags any image whose overall grade dropped or
that now violates a hard limit, so cross-image regressions can't hide.

## Choosing test images

Pick 5-10 images that span the failure modes you actually hit. A good set:

- a **portrait** (skin tone gradients, eyes, hair — the hardest case)
- a **high-contrast** subject (deep shadows + bright highlights)
- a **low-contrast / flat** subject (tests band seams in smooth gradients)
- a **fine-texture** subject (fur, foliage, fabric — tests detail vs noise)
- a **hard-edged / graphic** subject (logo, architecture — tests edge layers)
- a **busy scene** (stress-tests the path budget)

Keep them fixed. The value comes from comparing like-for-like over time.

## The loop

1. `run_suite.py` generates an SVG + preview PNG for every image x preset and
   records the exact `HatchConfig` used in the run manifest.
2. `grade.py` walks the previews and captures your 1-5 scores + notes into an
   append-only `grades.jsonl` (stamped with the git commit).
3. `compare.py` diffs the run against `baseline.json` and flags regressions.
4. `--promote` writes the accepted results into the baseline.

Because the config and git hash are recorded per run, any result is fully
reproducible, and `grades.jsonl` becomes a durable record of how the
methodology improved.

## Presets = one idea at a time

`presets.py` clones `baseline` and changes a few knobs per preset. Changing one
idea at a time keeps the comparison interpretable — you learn *which* change
helped, not just that the bundle did.

## Known follow-up: decouple the pipeline from the GUI

Today the harness imports `hatch_ui_nocairo.py`, which imports DearPyGui at
module load. That works because the pipeline lives at module scope, but it
couples headless runs to a GUI dependency. A clean future refactor: split the
pipeline (config, image processing, SVG writing) into a `hatchcore` module that
the GUI and the harness both import. Do it behind the test suite so the
refactor itself can't regress output.
