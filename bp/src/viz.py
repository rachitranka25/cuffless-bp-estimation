"""Shared plotting style, so every figure in the project reads as one system.

Hues are assigned in fixed order and by entity, never cycled and never by rank:
ECG is always blue, PPG always orange, ABP always aqua; leaky/honest/calfree keep
their own three colours wherever they appear. The three-slot categorical set is
validated for all-pairs colour-vision separation, and identity is never carried by
colour alone — every figure also labels its series.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#eda100", "#e87ba4", "#008300",
                                       "#4a3aa7", "#e34948")

# Entity -> colour. Never reassign these.
CH = {"ecg": BLUE, "ppg": ORANGE, "abp": AQUA}
PROTOCOL = {"leaky": ORANGE, "honest": BLUE, "calfree": AQUA}
DATASET = {"pulsedb": BLUE, "ppg_dalia": ORANGE}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"

# Status colours are reserved and never used as a series.
OK, WARN, BAD = "#1baf7a", "#eda100", "#e34948"


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
        "savefig.dpi": 130,
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
