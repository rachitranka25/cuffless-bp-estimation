"""Verifies, rather than assumes, that this project's categorical colour
palette (src/viz.py) is distinguishable under colour-vision deficiency.

Simulates each colour under deuteranomaly, protanomaly, and tritanomaly at
full severity (Machado, Oliveira & Fernandes, 2009, via the colorspacious
package) for both categorical sets actually used together in the project's
figures — the four evaluation protocols and the five model architectures —
and reports the worst-case pairwise CAM02-UCS colour difference (deltaE) in
each. A deltaE below ~20 is commonly treated as "may be hard to tell apart"
for categorical colour coding; this project's threshold is 20.

Run:  python3 -m pip install colorspacious   (not a pipeline dependency,
      only needed for this check — not in requirements.txt)
      python3 scripts/check_palette_safety.py
"""
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from colorspacious import cspace_convert
import viz

THRESHOLD = 20.0


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def delta_e(c1, c2, severity, cvd_type):
    rgb1, rgb2 = np.array(hex_to_rgb(c1)), np.array(hex_to_rgb(c2))
    cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    jab1 = cspace_convert(cspace_convert(rgb1, cvd_space, "sRGB1"), "sRGB1", "CAM02-UCS")
    jab2 = cspace_convert(cspace_convert(rgb2, cvd_space, "sRGB1"), "sRGB1", "CAM02-UCS")
    return float(np.sqrt(np.sum((jab1 - jab2) ** 2)))


PALETTES = {
    "4-protocol (leaky/calbased/calfree/aami)": {
        "leaky": viz.ORANGE, "calbased": viz.YELLOW,
        "calfree": viz.BLUE, "aami": viz.AQUA,
    },
    "5-model (rf/gb/cnn/resnet/transformer)": {
        "rf": viz.BLUE, "gb": viz.YELLOW, "cnn": viz.ORANGE,
        "resnet": viz.AQUA, "transformer": viz.VIOLET,
    },
    "status (OK/WARN/BAD, e.g. AAMI/BHS pass-fail cells)": {
        "OK": viz.OK, "WARN": viz.WARN, "BAD": viz.BAD,
    },
}

failures = []
for name, palette in PALETTES.items():
    print(f"=== {name} ===")
    worst, worst_pair = 999.0, None
    for cvd_type in ("deuteranomaly", "protanomaly", "tritanomaly"):
        for a, b in itertools.combinations(palette, 2):
            d = delta_e(palette[a], palette[b], 100, cvd_type)
            if d < worst:
                worst, worst_pair = d, (a, b, cvd_type)
            if d < THRESHOLD:
                print(f"  [{cvd_type:14s}] {a:12s} vs {b:12s}: deltaE={d:5.1f}  <-- LOW")
    status = "PASS" if worst >= THRESHOLD else "FAIL"
    if status == "FAIL":
        failures.append(name)
    print(f"  worst pairwise deltaE: {worst:.1f} ({worst_pair})  {status}\n")

if failures:
    print(f"FAIL: {', '.join(failures)} has a pair below deltaE={THRESHOLD}.")
    sys.exit(1)
print(f"PASS: every pair in every palette is >= deltaE={THRESHOLD} "
      f"under deuteranomaly, protanomaly, and tritanomaly.")
