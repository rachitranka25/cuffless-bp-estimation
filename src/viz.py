"""Shared plotting style, so every figure in the project reads as one system.

Hues are assigned in fixed order and by entity, never cycled and never by rank:
ECG is always blue, PPG always orange, ABP always aqua; leaky/honest/calfree keep
their own three colours wherever they appear. Identity is never carried by colour
alone — every figure also labels its series.

The five categorical colours below (BLUE/ORANGE/AQUA/YELLOW/VIOLET) are chosen
from Okabe & Ito (2008)'s colour-blind-safe set and verified, not assumed: every
pairwise combination actually used together in this project's figures (the
four-protocol set and the five-model set) was checked with colorspacious's
Machado/Oliveira/Fernandes (2009) CVD simulation across deuteranomaly,
protanomaly, and tritanomaly at full severity, requiring CAM02-UCS deltaE >= 20
(the two sets' worst pairwise deltaE are 30.8 and 24.2 respectively — see
scripts/check_palette_safety.py, which reproduces this check). Names reflect
each colour's closest hue family, not a literal "aqua" or "violet" swatch —
AQUA is a sky blue and VIOLET a wine red, both chosen for separation from the
rest of their set over strict naming purity.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, fixed order — colour-blind-safe set (see docstring above).
BLUE, ORANGE, AQUA = "#0072B2", "#D55E00", "#56B4E9"
YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#F0E442", "#e87ba4", "#008300",
                                       "#882255", "#e34948")

# Entity -> colour. Never reassign these.
CH = {"ecg": BLUE, "ppg": ORANGE, "abp": AQUA}
PROTOCOL = {"leaky": ORANGE, "honest": BLUE, "calfree": AQUA}
DATASET = {"pulsedb": BLUE, "ppg_dalia": ORANGE}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"

# Status colours are reserved and never used as a series. OK uses a teal
# rather than a pure green specifically because green-vs-red (not teal-vs-red)
# is the classic deuteranopia/protanopia confusion; verified pairwise (see
# scripts/check_palette_safety.py's STATUS set, worst-case deltaE 22.1) —
# every PASS/FAIL cell this colours also prints its own text label, so colour
# is reinforcement, not the only signal.
OK, WARN, BAD = "#00796B", "#F0E442", "#CC3311"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "font.size": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def title(ax, text, sub=None):
    """Title plus an optional recessive one-line explanation."""
    ax.set_title(text, loc="left", pad=14 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8,
                color=INK_2, va="bottom")
