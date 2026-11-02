"""Publication versions of the manuscript figures, drawn at their printed size.

Every figure here is laid out for a 6.5-inch print width (the single-column
Word file; the LaTeX version places them at nearly the same width), so the font
sizes set below are the sizes that appear on the page: nothing is shrunk after
the fact. Panels that were side by side in the slide versions are stacked so
that each keeps the full width. Labels use plain technical wording.

The numbers come from the same run files as every other figure and table; the
few values the text quotes directly are asserted against the run files here.

Outputs: results/overview/journal/fig*.png and .pdf
Run:     python3 scripts/make_journal_figures.py
"""
import json
import os
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MODEL_RESULTS, OVERVIEW, ROOT   # noqa: E402
import viz                                         # noqa: E402
import plot_results as PR                          # noqa: E402
from literature import rows as published           # noqa: E402

viz.use_style()
mpl.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

WIDTH = 6.5                     # inches, printed width
OUT = OVERVIEW / "journal"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = PR.MODEL_ORDER
LABEL = PR.MODEL_LABEL
PROTOS = PR.PROTO_ORDER
PCOLOR = PR.PROTO_COLOR
PNAME = {"leaky": "leaky", "calbased": "calbased", "calfree": "calfree", "aami": "AAMI"}
TICK = {"rf": "Random\nForest", "gb": "Gradient\nBoosting", "cnn": "1D-CNN", "resnet": "ResNet1D",
        "transformer": "Transformer"}
MCOLOR = {"rf": viz.BLUE, "gb": viz.YELLOW, "cnn": viz.ORANGE, "resnet": viz.AQUA,
          "transformer": viz.VIOLET}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    with open(OUT / f"{name}.png", "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    print(f"  {name}: {w / 300:.2f} x {h / 300:.2f} in")


# --------------------------------------------------------------------------- 1
def fig_protocols():
    def box(ax, x, y, w, h, color, text, tcolor="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                    linewidth=0, facecolor=color, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8,
                color=tcolor, weight="bold", zorder=3, linespacing=1.35)

    spec = [("Leaky", "random clips, any patient", viz.ORANGE, "Training pool",
             "Test: random clips\n(patients also in training)", True, "white"),
            ("Calibration-based", "held-out clips, same patients", viz.YELLOW, "Training pool",
             "Test: held-out clips\n(patients also in training)", True, viz.INK),
            ("Calibration-free", "144 unseen patients", viz.BLUE, "Training pool\n(1,164 patients)",
             "Test: 144 new\npatients", False, "white"),
            ("AAMI", "116 unseen patients, full BP range", viz.AQUA, "Training pool\n(1,164 patients)",
             "Test: 116 new patients\n(full BP range)", False, viz.INK)]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.3))
    for ax, (name, sub, color, train, test, overlap, tcol) in zip(axes.flat, spec):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        viz.title(ax, name, sub, fontsize=9, sub_fontsize=7.5)
        box(ax, 0.06, 0.56, 0.88, 0.34, viz.INK_2, train)
        box(ax, 0.06, 0.06, 0.88, 0.34, color, test, tcol)
        if overlap:
            ax.add_patch(FancyArrowPatch((0.5, 0.56), (0.5, 0.40), arrowstyle="-|>",
                                         mutation_scale=11, color=viz.MUTED, lw=1.1, zorder=4))
            word, col = "same patients", viz.MUTED
        else:
            ax.plot([0.5, 0.5], [0.40, 0.56], color=viz.BAD, lw=1.4, zorder=4)
            word, col = "no shared patients", viz.BAD
        ax.text(0.5, 0.48, word, ha="center", va="center", fontsize=7.5, color=col,
                style="italic", zorder=5, bbox=dict(facecolor=viz.SURFACE, edgecolor="none", pad=1.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.12)
    save(fig, "fig1_protocols")


# ------------------------------------------------------------------------ 2, 3
METRICS = [("mae", "MAE (mmHg)", "Mean absolute error", "Lower is better", None),
           ("sde", "Error SD (mmHg)", "Error standard deviation", "AAMI requires SD ≤ 8 mmHg", 8.0),
           ("r", "Pearson r", "Pearson correlation", "Between predicted and reference BP", None)]


def fig_gap(runs, target, name):
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.6))
    width = 0.8 / len(PROTOS)
    for ax, (key, ylab, ttl, sub, limit) in zip(axes, METRICS):
        top = 0
        for j, p in enumerate(PROTOS):
            xs = [i - 0.4 + width * (j + 0.5) for i in range(len(MODELS))]
            vals = [runs[(m, p)]["results"][target][key] for m in MODELS]
            ax.bar(xs, vals, width=width * 0.9, color=PCOLOR[p], edgecolor="none", label=PNAME[p])
            top = max(top, max(vals))
            for x, v in zip(xs, vals):
                ax.text(x, v + top * 0.02, f"{v:.2f}" if key == "r" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7, color=viz.INK_2, rotation=90)
        if limit:
            ax.axhline(limit, ls="--", lw=1.0, color=viz.BAD)
            ax.text(1.0, limit, "  AAMI limit", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=7.5, color=viz.BAD)
        ax.set_ylim(0, top * 1.32)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([TICK[m] for m in MODELS])
        ax.set_ylabel(ylab)
        viz.title(ax, f"{target.upper()}: {ttl}", sub, fontsize=9, sub_fontsize=7.5)
        viz.despine(ax)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.02), handlelength=1.2,
                   columnspacing=1.0)
    fig.subplots_adjust(hspace=0.62)
    save(fig, name)


# --------------------------------------------------------------------------- 4
def fig_weighting():
    def mae(m, tag):
        d = json.loads((MODEL_RESULTS / m / f"{m}_calfree{'_' + tag if tag else ''}.json").read_text())
        return d["results"]["sbp"]["mae"]

    def above170(m, tag):
        z = np.load(MODEL_RESULTS / m / f"{m}_calfree{'_' + tag if tag else ''}_predictions.npz")
        y, p = z["y_true"][:, 0], z["y_pred"][:, 0]
        sel = y >= 170
        return float(np.mean(np.abs(p[sel] - y[sel])))

    overall, high = [], []
    for m in MODELS:
        w = PR.CALFREE_CANONICAL_TAG.get(m, "")
        overall.append(mae(m, w) - mae(m, "nobalance"))
        high.append(above170(m, w) - above170(m, "nobalance"))
    # the values quoted in Section 4.3
    for got, want in zip(overall, [-0.18, 1.92, 1.88, 0.74, 2.77]):
        assert abs(got - want) < 0.006, (overall, "overall")
    for got, want in zip(high, [-4.05, -13.29, -16.73, -9.06, -16.41]):
        assert abs(got - want) < 0.006, (high, "above170")

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    names = [LABEL[m] for m in MODELS]
    colors = [MCOLOR[m] for m in MODELS]
    for ax, vals, ttl, sub, ylab in (
            (a, overall, "Cost: overall SBP error", "Weighted minus unweighted; positive = higher error",
             "Change in SBP MAE (mmHg)"),
            (b, high, "Benefit: SBP above 170 mmHg", "Weighted minus unweighted; negative = lower error",
             "Change in SBP MAE (mmHg)")):
        bars = ax.bar(range(5), vals, color=colors, width=0.62)
        ax.axhline(0, color=viz.INK_2, lw=0.7)
        span = max(vals) - min(min(vals), 0)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + (0.03 if v >= 0 else -0.03) * span,
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
        lo, hi = min(min(vals), 0), max(max(vals), 0)
        ax.set_ylim(lo - 0.16 * span if lo < 0 else lo, hi + 0.16 * span if hi > 0 else hi + 0.05 * span)
        ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.5, rotation=30, ha="right",
                                                     rotation_mode="anchor")
        ax.set_ylabel(ylab)
        viz.title(ax, ttl, sub, fontsize=9, sub_fontsize=7)
        viz.despine(ax); ax.grid(axis="x", visible=False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "fig4_weighting")


# --------------------------------------------------------------------------- 5
PUB_LABEL = {"DMT (Transformer)": "DMT (Shen et al.)",
             "UTransBPNet": "UTransBPNet (Huang et al.)",
             "CNN-LSTM-Attention": "CNN-LSTM-Attention (Huang et al.)",
             "XResNet1d101 (benchmark)": "XResNet1d101 (Moulaeifard et al.)"}


def fig_literature():
    import plot_literature as PL
    data = []
    for lbl, s, d, mine in PL.rows("cal_free"):
        if mine:
            lbl = lbl.replace("(this project)", "(this study)")
        else:
            lbl = PUB_LABEL[lbl]
        data.append((lbl, s, d, mine))
    assert sum(not m for *_, m in data) == 4
    fig, axes = plt.subplots(2, 1, figsize=(5.65, 5.4))
    for ax, col, ttl in ((axes[0], 1, "SBP, calibration-free"), (axes[1], 2, "DBP, calibration-free")):
        rows = sorted(data, key=lambda r: r[col])
        y = np.arange(len(rows))[::-1]
        vals = [r[col] for r in rows]
        ax.barh(y, vals, height=0.62, color=[to_rgba(viz.ORANGE if r[3] else viz.BLUE, 1 if r[3] else .8)
                                             for r in rows])
        for yy, v, r in zip(y, vals, rows):
            ax.text(v + max(vals) * 0.012, yy, f"{v:.2f}", va="center", fontsize=7.5,
                    weight="bold" if r[3] else "normal", color=viz.INK if r[3] else viz.INK_2)
        ax.set_yticks(y, [r[0] for r in rows])
        for t, r in zip(ax.get_yticklabels(), rows):
            if r[3]:
                t.set_fontweight("bold"); t.set_color(viz.INK)
        ax.set_xlim(0, max(vals) * 1.12)
        ax.set_xlabel("MAE (mmHg)")
        ax.grid(axis="y", visible=False)
        viz.title(ax, ttl, "Orange: this study, unweighted, seed-0 runs; blue: published results",
                  fontsize=9, sub_fontsize=7.5)
        viz.despine(ax)
    fig.subplots_adjust(hspace=0.55)
    save(fig, "fig5_literature")


# ------------------------------------------------------------------------ 6, 7
def shift_rows():
    rows = []
    for m in MODELS:
        tag = PR.CALFREE_CANONICAL_TAG.get(m, "")
        a = json.loads((MODEL_RESULTS / m / f"{m}_{tag + '_' if tag else ''}dalia_domain_shift.json").read_text())
        b = json.loads((MODEL_RESULTS / m / f"{m}_nobalance_dalia_domain_shift.json").read_text())
        mean = lambda j, k: j["distribution"]["sbp"][k]["mean"]   # noqa: E731
        sd = lambda j, k: j["collapse"]["sbp"][k]                 # noqa: E731
        rows.append(dict(model=m,
                         shift_w=mean(a, "dalia_predicted") - mean(a, "pulsedb_predicted"),
                         shift_u=mean(b, "dalia_predicted") - mean(b, "pulsedb_predicted"),
                         pul_w=sd(a, "pulsedb_predicted"), dal_w=sd(a, "dalia_predicted"),
                         pul_u=sd(b, "pulsedb_predicted"), dal_u=sd(b, "dalia_predicted"),
                         true=sd(a, "pulsedb_true")))
    return rows


def fig_shift(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w / 2, [r["shift_w"] for r in rows], w, color=viz.MUTED, label="With BP-bin weighting (primary)")
    ax.bar(x + w / 2, [r["shift_u"] for r in rows], w, color=viz.BLUE, label="Without BP-bin weighting")
    top = max(r["shift_w"] for r in rows)
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r["shift_w"] + top * 0.02, f"+{r['shift_w']:.1f}", ha="center", fontsize=7.5,
                color=viz.INK_2)
        ax.text(i + w / 2, r["shift_u"] + top * 0.02, f"+{r['shift_u']:.1f}", ha="center", fontsize=7.5,
                color=viz.BLUE)
        d = r["shift_w"] - r["shift_u"]
        ax.text(i, max(r["shift_w"], r["shift_u"]) + top * 0.11,
                f"{d:.1f} ({d / r['shift_w'] * 100:.0f}%)", ha="center", fontsize=7.5,
                color=viz.BAD)
    ax.set_xticks(x); ax.set_xticklabels([TICK[r["model"]] for r in rows])
    ax.set_ylabel("PPG-DaLiA minus PulseDB\nmean predicted SBP (mmHg)")
    ax.set_ylim(0, top * 1.48)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], ls="none", marker="$\\mathrm{x}$", color=viz.BAD, ms=0,
               label="Red: shift due to weighting, mmHg (% of weighted shift)")]
    leg = ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.2)
    leg.get_texts()[-1].set_color(viz.BAD)
    viz.despine(ax); ax.grid(axis="x", visible=False)
    save(fig, "fig6_shift")


def fig_collapse(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    for i, r in enumerate(rows):
        for dx, a, b, col, lab in ((-0.14, r["pul_w"], r["dal_w"], viz.MUTED, "With BP-bin weighting"),
                                   (0.14, r["pul_u"], r["dal_u"], viz.BLUE, "Without BP-bin weighting")):
            ax.plot([i + dx, i + dx], [a, b], color=col, lw=2.2, zorder=2, label=lab if i == 0 else None)
            ax.plot(i + dx, a, "o", color=col, ms=5, zorder=3)
            ax.plot(i + dx, b, "s", color=col, ms=5, zorder=3)
            ax.text(i + dx + (0.05 if dx > 0 else -0.05), b, f"{b:.1f}", fontsize=7,
                    ha="left" if dx > 0 else "right", va="center", color=col if col == viz.BLUE else viz.INK_2)
    t = rows[0]["true"]
    ax.axhline(t, ls="--", lw=1.0, color=viz.BAD)
    ax.text(-0.55, t + 0.25, f"True between-subject SD\non PulseDB: {t:.1f} mmHg", ha="left",
            va="bottom", fontsize=7.5, color=viz.BAD, linespacing=1.3)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([TICK[r["model"]] for r in rows])
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, max(max(r["pul_w"], r["pul_u"]) for r in rows) * 1.18)
    ax.set_ylabel("Between-subject SD of\npredicted SBP (mmHg)")
    handles = ax.get_legend_handles_labels()[0][:2] + [
        Line2D([], [], marker="o", ls="none", color=viz.INK_2, label="PulseDB"),
        Line2D([], [], marker="s", ls="none", color=viz.INK_2, label="PPG-DaLiA")]
    ax.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(0, 1.16), handlelength=1.4,
              columnspacing=1.2)
    viz.despine(ax); ax.grid(axis="x", visible=False)
    save(fig, "fig7_collapse")
    w = [r["dal_w"] for r in rows]; u = [r["dal_u"] for r in rows]
    print(f"    PPG-DaLiA spread: weighted {min(w):.2f}-{max(w):.2f}, unweighted {min(u):.2f}-{max(u):.2f}")


# --------------------------------------------------------------------------- 8
def fig_personalization():
    style = {"first": "-", "random": "--", "spread": ":"}
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.8), sharex=True)
    for m in MODELS:
        tag = PR.CALFREE_CANONICAL_TAG.get(m, "")
        d = json.loads((MODEL_RESULTS / m / f"{m}_personalization_calfree{'_' + tag if tag else ''}.json").read_text())
        for ax, target in zip(axes, ("sbp", "dbp")):
            for pick, curve in d["curves"].items():
                k = [r["shots"] for r in curve]
                v = [r["results"][target]["mae"] for r in curve]
                ax.plot(k, v, ls=style[pick], color=MCOLOR[m], lw=1.3, marker="o", ms=2.6)
    for ax, target in zip(axes, ("SBP", "DBP")):
        ax.set_ylabel(f"{target} MAE (mmHg)")
        ax.set_xticks([0, 1, 3, 5, 10, 25])
        viz.title(ax, f"{target}, naive split", "k = 0 is the uncalibrated model", fontsize=9, sub_fontsize=7.5)
        viz.despine(ax)
    axes[1].set_xlabel("Calibration clips per subject (k)")
    mh = [Line2D([], [], color=MCOLOR[m], lw=1.6, label=LABEL[m]) for m in MODELS]
    rule = {"first": "first clips (storage order)", "random": "random clips", "spread": "clips spread over BP range"}
    sh = [Line2D([], [], color=viz.INK_2, lw=1.3, ls=s, label=rule[p]) for p, s in style.items()]
    fig.legend(handles=mh + sh, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.subplots_adjust(hspace=0.35)
    save(fig, "fig8_personalization")


# ------------------------------------------------------------- supplement S1, S2
def fig_training():
    fig, axes = plt.subplots(4, 2, figsize=(WIDTH, 7.6), sharex=True)
    for col, m in enumerate(("cnn", "resnet")):
        for row, p in enumerate(PROTOS):
            ax = axes[row][col]
            h = json.loads((MODEL_RESULTS / m / f"{m}_{p}.json").read_text())["info"]["history"]
            ep = [q["epoch"] for q in h]
            ax.plot(ep, [q["train_loss"] for q in h], color=viz.MUTED, lw=1.2, label="training")
            ax.plot(ep, [q["val_loss"] for q in h], color=viz.BLUE, lw=1.4, label="validation")
            best = min(h, key=lambda q: q["val_loss"])["epoch"]
            ax.axvline(best, ls=":", lw=1.0, color=viz.BAD, label="best validation epoch")
            viz.title(ax, f"{LABEL[m]}, {PNAME[p]}", fontsize=8.5)
            if col == 0:
                ax.set_ylabel("Loss")
            if row == 3:
                ax.set_xlabel("Epoch")
            viz.despine(ax)
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.subplots_adjust(hspace=0.55, wspace=0.22)
    save(fig, "figS1_training")


def fig_bland():
    cmap = mpl.colormaps["RdBu_r"]; vmax = 12.0
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 7.0))
    for ax, (t, k) in zip(axes, (("SBP", 0), ("DBP", 1))):
        bias = np.full((5, 4), np.nan); lim = np.full((5, 4), np.nan)
        for i, m in enumerate(MODELS):
            for j, p in enumerate(PROTOS):
                tag = PR.CALFREE_CANONICAL_TAG.get(m, "") if p == "calfree" else ""
                z = np.load(MODEL_RESULTS / m / f"{m}_{p}{'_' + tag if tag else ''}_predictions.npz")
                diff = z["y_pred"][:, k] - z["y_true"][:, k]
                bias[i, j] = diff.mean(); lim[i, j] = 1.96 * diff.std(ddof=1)
        im = ax.imshow(bias, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(5):
            for j in range(4):
                c = "white" if abs(bias[i, j]) > vmax * 0.55 else viz.INK
                b = round(float(bias[i, j]), 1) + 0.0   # no "-0.0"
                ax.text(j, i, f"{b:+.1f} ± {lim[i, j]:.0f}" if b else f"0.0 ± {lim[i, j]:.0f}",
                        ha="center", va="center",
                        fontsize=8, color=c)
        ax.set_xticks(range(4), [PNAME[p] for p in PROTOS])
        ax.set_yticks(range(5), [LABEL[m] for m in MODELS])
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color=viz.SURFACE, lw=2); ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        viz.title(ax, f"{t} agreement", "Cell: bias ± 95% limits of agreement (mmHg)",
                  fontsize=9, sub_fontsize=7.5)
    fig.subplots_adjust(hspace=0.38, bottom=0.13)
    cax = fig.add_axes([0.25, 0.035, 0.5, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Bias, predicted minus reference (mmHg)")
    save(fig, "figS2_bland_altman")


if __name__ == "__main__":
    runs = PR.load()
    fig_protocols()
    fig_gap(runs, "sbp", "fig2_gap_sbp")
    fig_gap(runs, "dbp", "fig3_gap_dbp")
    fig_weighting()
    fig_literature()
    rows = shift_rows()
    fig_shift(rows)
    fig_collapse(rows)
    fig_personalization()
    fig_training()
    fig_bland()
