"""Analyze the pen path of a tonal SVG: what makes the plotter slow?

Plot time on a Cricut is dominated by (a) total pen-DOWN travel and (b) the
number of direction changes / short segments (it decelerates at every vertex).
Pen-UP moves (lifts) are comparatively fast. This splits a run's d-strings into
pen-down vs pen-up moves and reports the metrics that actually track draw time.
"""
import os, sys, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import presets as P
from hatch_ui_nocairo import hatch_pipeline

def analyze(all_d):
    pen_down = pen_up = 0.0
    nverts = 0
    seg_lens = []            # pen-down segment lengths
    long_down = 0            # pen-down "connector" moves (long -> likely hidden-travel scribble)
    lifts = 0
    for d in all_d:
        # tokens are either a command letter (M/L) or an "x,y" coord pair;
        # implicit lineto: coord pairs after an M (beyond the first) are L.
        toks = d.split()
        cur = None; cmd = None; first_after_m = False
        for t in toks:
            if t in ("M", "L"):
                cmd = t; first_after_m = (t == "M"); continue
            if "," not in t:
                continue
            x, y = (float(v) for v in t.split(","))
            if cur is not None:
                dist = math.hypot(x - cur[0], y - cur[1])
                if cmd == "M" and first_after_m:
                    pen_up += dist; lifts += 1
                else:
                    pen_down += dist; nverts += 1; seg_lens.append(dist)
            else:
                lifts += 1
            first_after_m = False
            cur = (x, y)
    seg_lens = np.array(seg_lens) if seg_lens else np.array([0.0])
    return dict(pen_down=pen_down, pen_up=pen_up, nverts=nverts, lifts=lifts,
                mean_seg=float(seg_lens.mean()), median_seg=float(np.median(seg_lens)),
                n_tiny=int((seg_lens < 2.0).sum()), n_total=len(seg_lens))

def run(name, img, preset="baseline"):
    out = f"results/runs/_stats/{name}.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cfg = P.get(preset)
    s = hatch_pipeline(img, out, cfg, log_cb=lambda m: None)
    # re-read the d-strings from stats isn't exposed; recompute from the SVG preview path?
    # Simpler: pull from the written SVG.
    import xml.etree.ElementTree as ET
    txt = open(out, encoding="utf-8").read()
    ds = re.findall(r'\bd="([^"]+)"', txt)
    a = analyze(ds)
    tot = a["pen_down"] + a["pen_up"]
    print(f"\n== {name} ({preset}) ==  paths={s['paths']} lifts~{s['pen_lifts_est']}")
    print(f"  pen-DOWN dist : {a['pen_down']:10.0f}  ({100*a['pen_down']/tot:.1f}% of travel)")
    print(f"  pen-UP   dist : {a['pen_up']:10.0f}  (lifts={a['lifts']})")
    print(f"  pen-down segs : {a['n_total']:6d}   mean={a['mean_seg']:.1f}px  median={a['median_seg']:.1f}px")
    print(f"  tiny (<2px)   : {a['n_tiny']:6d}   ({100*a['n_tiny']/max(1,a['n_total']):.0f}% of draw segs)")
    return a

def est_hours(a, mm_per_px, draw_mm_s=20.0, travel_mm_s=100.0, vtx_s=0.04):
    """Rough Cricut time: pen-down slow, pen-up fast, + per-vertex accel/decel."""
    t = (a["pen_down"] * mm_per_px / draw_mm_s
         + a["pen_up"] * mm_per_px / travel_mm_s
         + a["nverts"] * vtx_s)
    return t / 3600.0

if __name__ == "__main__":
    from dataclasses import replace
    base = P.get("baseline")
    variants = {
        "baseline (10 layers, greedy)":  base,
        "+ boustrophedon order":         replace(base, tonal_greedy_stitch=False),
        "7 layers + boustrophedon":      replace(base, tonal_max_layers=7, tonal_greedy_stitch=False),
        "6 layers + boustrophedon":      replace(base, tonal_max_layers=6, tonal_greedy_stitch=False),
    }
    MM_PER_PX = 203.4 / 427.0     # 8.0063in sheet / 427px work width
    for img, nm in [("tests/images/portrait_lowkey.jpg", "portrait_lowkey"),
                    ("tests/images/texture_sneaker.jpg", "sneaker")]:
        print("\n" + "#" * 70 + f"\n# {nm}\n" + "#" * 70)
        for label, cfg in variants.items():
            out = f"results/runs/_stats/{nm}.svg"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            s = hatch_pipeline(img, out, cfg, log_cb=lambda m: None)
            txt = open(out, encoding="utf-8").read()
            ds = re.findall(r'\bd="([^"]+)"', txt)
            a = analyze(ds)
            dn_m = a["pen_down"] * MM_PER_PX / 1000.0
            print(f"\n  [{label}]  paths={s['paths']}")
            print(f"    draw length {dn_m:6.1f} m   vertices {a['nverts']:6d}  "
                  f"lifts {a['lifts']:5d}   mean seg {a['mean_seg']:5.1f}px")
            print(f"    EST PLOT TIME ~ {est_hours(a, MM_PER_PX):4.1f} h")
