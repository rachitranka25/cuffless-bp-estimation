"""Figures comparing this project's baseline against published results.

Run:  python3 scripts/plot_literature.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import FIG_LIT, ROOT, MODEL_RESULTS   # noqa: E402
from literature import rows as published, ARCHITECTURES, OURS   # noqa: E402
import viz                                   # noqa: E402

viz.use_style()
OUT = FIG_LIT
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
               "resnet": "ResNet1D", "transformer": "Transformer"}
PROTO_KEY = {"cal_free": "calfree", "cal_based": "calbased", "aami": "aami"}


def rows(protocol):
    """Published results plus whatever this project has actually trained.

    Our numbers are read from the run files rather than restated here, so this
    figure cannot disagree with results/overview/RESULTS.md. For the two
    protocols the literature reports, the unbalanced run is used when it exists:
    every published number was measured without BP-bin weighting, and comparing
    across that difference would be comparing two different things.
    """
    import json
    out = [r for r in published(protocol) if not r[3]]
    want = PROTO_KEY.get(protocol)
    if not want:
        return sorted(out, key=lambda r: r[1])

    for d in sorted(MODEL_RESULTS.glob("*/*.json")):
        if "personalization" in d.name or "domain_shift" in d.name:
            continue
        j = json.loads(d.read_text())
        if j.get("protocol") != want:
            continue
        tag = j.get("config", {}).get("tag") or ""
        if tag not in ("", "nobalance"):
            continue
        m = j["model"]
        # prefer the unbalanced run; drop the balanced one if both exist
        if tag == "" and (MODEL_RESULTS / m / f"{m}_{want}_nobalance.json").exists():
            continue
        label = f"{MODEL_LABEL[m]} (this project)"
        out.append((label, j["results"]["sbp"]["mae"], j["results"]["dbp"]["mae"], True))
    return sorted(out, key=lambda r: r[1])


def save(fig, name):
    fig.savefig(OUT / name)
    print("saved", (OUT / name).relative_to(ROOT))
    plt.close(fig)


def bars(ax, data, col, title, sub, xlabel):
    """Horizontal bars, ours highlighted; everyone else recedes."""
    labels = [d[0] for d in data]
    vals = [d[col] for d in data]
    mine = [d[3] for d in data]
    y = np.arange(len(data))[::-1]
    # alpha has to be baked into each colour — barh takes one scalar alpha only
    colors = [to_rgba(viz.ORANGE if m else viz.BLUE, 1.0 if m else .78)
              for m in mine]
    ax.barh(y, vals, color=colors, height=.62)
    for yy, v, m in zip(y, vals, mine):
        ax.text(v + max(vals) * .015, yy, f"{v:.2f}", va="center", fontsize=9,
                color=viz.INK if m else viz.INK_2,
                weight="bold" if m else "normal")
    ax.set_yticks(y, labels)
    for tick, m in zip(ax.get_yticklabels(), mine):
        if m:
            tick.set_fontweight("bold")
            tick.set_color(viz.INK)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="y", visible=False)
    viz.despine(ax)
    viz.title(ax, title, sub)


def fig_calfree():
    d = rows("cal_free")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2))
    bars(axes[0], d, 1, "Systolic BP — calibration-free",
         "subject-independent: no test patient was seen in training",
         "mean absolute error (mmHg)")
    bars(axes[1], d, 2, "Diastolic BP — calibration-free", "",
         "mean absolute error (mmHg)")
    fig.suptitle("Published results on PulseDB, honest protocol — orange is this "
                 "project, measured without BP-bin weighting so the comparison holds",
                 fontsize=11.5, x=.012, ha="left", weight="bold")
    plt.tight_layout(rect=[0, 0, 1, .93])
    save(fig, "01_calibration_free_comparison.png")


def fig_protocols():
    """The same papers under the three protocols — the spread IS the point."""
    protos = [("cal_based", "Calibration-based\n(test patients were seen)"),
              ("cal_free", "Calibration-free\n(new patients)"),
              ("aami", "AAMI subset\n(full BP range)")]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    xs, labels, colors, hatches = [], [], [], []
    pos, tickpos, ticklab = 0, [], []
    for proto, name in protos:
        d = rows(proto)
        start = pos
        for label, sbp, dbp, mine in d:
            xs.append((pos, sbp, mine, label))
            pos += 1
        tickpos.append((start + pos - 1) / 2)
        ticklab.append(name)
        pos += 1.2
    for p, v, mine, label in xs:
        ax.bar(p, v, width=.82, color=viz.ORANGE if mine else viz.BLUE,
               alpha=1.0 if mine else .78)
        ax.text(p, v + .35, f"{v:.1f}", ha="center", fontsize=8.5,
                color=viz.INK if mine else viz.INK_2,
                weight="bold" if mine else "normal")
        ax.text(p, -.7, label, ha="right", va="top", rotation=35, fontsize=7.5,
                color=viz.INK if mine else viz.MUTED,
                weight="bold" if mine else "normal")
    ax.set_xticks(tickpos, ticklab, fontsize=9.5)
    ax.tick_params(axis="x", pad=96, length=0)   # clear the rotated paper labels
    ax.set_ylabel("SBP mean absolute error (mmHg)")
    ax.set_ylim(0, 22)
    ax.grid(axis="x", visible=False)
    viz.despine(ax)
    viz.title(ax, "The protocol moves the number more than the model does",
              "same papers, same dataset — only the train/test split changes")
    plt.tight_layout(rect=[0, .06, 1, 1])
    save(fig, "02_protocol_effect.png")


def fig_architectures():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    y = np.arange(len(ARCHITECTURES))[::-1]
    ax.barh(y, [1] * len(ARCHITECTURES), color=viz.BLUE, alpha=.14, height=.7)
    for yy, (name, examples, why) in zip(y, ARCHITECTURES):
        ax.text(.012, yy + .17, name, fontsize=10.5, weight="bold",
                color=viz.INK, va="center")
        ax.text(.012, yy - .17, f"{examples} — {why}", fontsize=8.5,
                color=viz.INK_2, va="center")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    viz.despine(ax, keep=())
    viz.title(ax, "Model families used on this dataset",
              "transformers currently lead; nobody reports hand-crafted features as competitive")
    plt.tight_layout()
    save(fig, "03_architectures.png")


if __name__ == "__main__":
    fig_calfree()
    fig_protocols()
    fig_architectures()
