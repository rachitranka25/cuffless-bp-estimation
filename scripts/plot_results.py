"""Figures for whatever has been trained so far.

Reads results/runs/*.json and the matching *_predictions.npz and draws only what
the data supports — a missing model or protocol leaves a gap rather than an
invented bar. Re-runnable at any point: run it after the baselines and it draws
the baselines; run it again after the deep models and it redraws everything.

Systolic and diastolic are both shown everywhere they can be: the tables report
each separately and the per-model figures carry one row per target. Only the two
domain-shift figures split into separate files, because a wearable dataset with
no labels supports only a distribution comparison and the two targets need very
different axes.

overview/ — everything that only exists by comparing models
  1_leakage_gap.png            the headline: same models, four test rules, both targets
  2_compliance.png             AAMI / BHS pass-fail table (spec K, primary), both targets
  3_shift_amount.png           systolic: how far the predictions moved, and how
                               much of that was our own weighting
  3_shift_amount_dbp.png       the same for diastolic
  3_shift_collapse.png         whether it still tells people apart

models/<name>/figures/ — one model at a time
  1_bp_band_error.png          what the BP-bin balancing costs and buys (spec L),
                               systolic on top, diastolic below
  2_bland_altman.png           agreement plot (spec K, secondary), both targets
  3_domain_shift.png           the PulseDB-trained model run on PPG-DaLiA
  4_training_curve.png         deep models only — where generalisation stops
  5_personalization.png        the few-shot curve, once it has been run

Run:  python3 scripts/plot_results.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import MODEL_RESULTS, OVERVIEW, ROOT, model_dir   # noqa: E402
import viz                                 # noqa: E402

viz.use_style()

# Fixed order and fixed colour per protocol, everywhere. Orange and yellow are
# the two that test on patients the model has already met; blue and aqua are the
# two that do not — so the honest/leaky split is legible before reading a label.
PROTO_ORDER = ["leaky", "calbased", "calfree", "aami"]
PROTO_COLOR = {"leaky": viz.ORANGE, "calbased": viz.YELLOW,
               "calfree": viz.BLUE, "aami": viz.AQUA}
PROTO_SEEN = {"leaky": True, "calbased": True, "calfree": False, "aami": False}
PROTO_NOTE = {
    "leaky": "same patients,\nrandom clips",
    "calbased": "same patients,\nofficial clips",
    "calfree": "144 unseen\npatients",
    "aami": "116 unseen\npatients",
}
MODEL_ORDER = ["rf", "gb", "cnn", "resnet", "transformer"]
MODEL_LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN",
               "resnet": "ResNet1D", "transformer": "Transformer"}


def save(fig, name, model=None):
    """Cross-model figures go to overview/; single-model ones live with the model.

    Saves both the raster PNG (for quick viewing, and for the ISGJ/docx paper,
    which cannot embed vector graphics) and a vector PDF alongside it (for the
    IEEE/LaTeX paper, which can — infinite-resolution text and lines, no
    re-rasterising at print size).
    """
    out = model_dir(model, "figures") if model else OVERVIEW
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    fig.savefig((out / name).with_suffix(".pdf"))
    print("  saved", (out / name).relative_to(ROOT), "(+ .pdf)")
    plt.close(fig)


def load():
    """{(model, protocol): run dict} for the untagged runs.

    Tagged runs (ablations such as `gb_calfree_nobalance`) share a model and a
    protocol with the run they are compared against, so they are kept out of the
    main dictionary — otherwise an ablation would silently replace the result it
    exists to be measured against.
    """
    runs = {}
    for p in sorted(MODEL_RESULTS.glob("*/*.json")):
        # personalization curves and the domain-shift report are keyed differently
        # and have their own figures
        if "personalization" in p.name or "domain_shift" in p.name:
            continue
        d = json.loads(p.read_text())
        if d.get("config", {}).get("tag"):
            continue
        runs[(d["model"], d["protocol"])] = d
    return runs


def predictions(model, protocol, tag=""):
    p = (MODEL_RESULTS / model
         / f"{model}_{protocol}{'_' + tag if tag else ''}_predictions.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["y_true"], z["y_pred"], z["subjects"]


def _present(runs):
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in runs)]
    protos = [p for p in PROTO_ORDER if any(k[1] == p for k in runs)]
    return models, protos


# --------------------------------------------------------------------------- #


# The three numbers worth plotting. MAE is what the field headlines, but a
# method can have a low MAE and still fail AAMI on spread, and correlation is
# the one that exposes a model predicting the population mean — so all three.
# The AAMI limit of 5 mmHg applies to the *bias*, not to MAE — the line on the
# MAE panel is the target the field aims at, which is a weaker claim, so it is
# labelled as such rather than borrowing AAMI's name.
METRICS = [
    ("mae", "MAE (mmHg)", "how far off it is on average — lower is better",
     5.0, "clinical target 5"),
    ("sde", "SD of error (mmHg)",
     "are the errors all similar, or all over the place — this is what AAMI limits",
     8.0, "AAMI limit 8"),
    ("r", "Pearson r",
     "1.0 = follows the patient up and down, 0.0 = one answer for everyone",
     None, None),
]


def fig_leakage_gap(runs):
    models, protos = _present(runs)
    if not models:
        return
    nrow, ncol = 2, len(METRICS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.35 * nrow),
                             squeeze=False)
    width = 0.8 / max(1, len(protos))

    for row, target in enumerate(("sbp", "dbp")):
        for col, (key, ylab, sub, limit, limit_label) in enumerate(METRICS):
            ax = axes[row][col]
            top = 0
            for j, p in enumerate(protos):
                vals, xs = [], []
                for i, m in enumerate(models):
                    r = runs.get((m, p))
                    if not r:
                        continue
                    vals.append(r["results"][target][key])
                    xs.append(i - 0.4 + width * (j + 0.5))
                if not vals:
                    continue
                ax.bar(xs, vals, width=width * 0.92, color=PROTO_COLOR[p],
                       label=p if (row == 0 and col == 0) else None,
                       edgecolor="none")
                top = max(top, max(vals))
                for x, v in zip(xs, vals):
                    ax.text(x, v + top * 0.03, f"{v:.2f}" if key == "r" else f"{v:.1f}",
                            ha="center", va="bottom", fontsize=7.2, color=viz.INK_2,
                            rotation=90)
            if limit:
                ax.axhline(limit, ls="--", lw=1.3, color=viz.BAD)
                ax.text(len(models) - 0.44, limit + top * 0.02,
                        limit_label, ha="right", fontsize=8,
                        color=viz.BAD, weight="bold")
            ax.set_ylim(0, top * 1.38)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([MODEL_LABEL[m].replace(" ", "\n") for m in models],
                                fontsize=8.0)
            ax.set_ylabel(ylab)
            viz.title(ax, f"{target.upper()} — {key.upper() if key == 'mae' else ylab.split(' (')[0]}",
                      sub)
            viz.despine(ax)
    axes[0][0].legend(ncol=2, loc="upper left", fontsize=8.4)
    fig.suptitle("Same model, same training clips — only the test patients change.  "
                 "Orange/yellow = the model had already met them; blue/sky blue = it had not.",
                 fontsize=12.5, weight="bold", x=0.006, ha="left", y=1.0)
    plt.tight_layout()
    save(fig, "1_leakage_gap.png")


SBP_BANDS = [0, 90, 110, 130, 150, 170, 1000]
BAND_LABEL = ["<90", "90-110", "110-130", "130-150", "150-170", "170+"]


def _band_mae(y_true, y_pred):
    band = np.digitize(y_true, SBP_BANDS[1:-1])
    return np.array([np.mean(np.abs(y_pred[band == i] - y_true[band == i]))
                     if (band == i).any() else np.nan
                     for i in range(len(BAND_LABEL))]), \
           np.array([(band == i).sum() for i in range(len(BAND_LABEL))])


def fig_bp_band_error(runs, protocol="calfree"):
    """What the BP-bin balancing actually buys, band by band, for every model.

    Spec section L asks for stratified sampling or a weighted loss over BP bins.
    It costs headline MAE and it is worth showing why that trade is the right
    one: without it the model concentrates on the crowded middle of the BP
    distribution, which is exactly where a cuffless monitor matters least.

    Drawn only for models that have both the balanced run and the `nobalance`
    ablation — there is nothing to compare otherwise.
    """
    for model in _present(runs)[0]:
        _one_bp_band_error(model, protocol)


def _one_bp_band_error(model, protocol):
    """Systolic on the top row, diastolic on the bottom.

    The bands are always cut on the *systolic* value, for both rows — a patient
    is hypertensive or not as one person, and splitting diastolic on its own
    scale would put the same clip in different bands on the two rows.
    """
    on = predictions(model, protocol)
    off = predictions(model, protocol, tag="nobalance")
    if on is None or off is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0),
                             gridspec_kw=dict(width_ratios=[2.1, 1]))
    x = np.arange(len(BAND_LABEL))

    for row, target in enumerate(("sbp", "dbp")):
        k = 0 if target == "sbp" else 1
        ax = axes[row][0]
        # band membership always comes from systolic, on both rows
        band = np.digitize(on[0][:, 0], SBP_BANDS[1:-1])
        n = np.array([(band == i).sum() for i in range(len(BAND_LABEL))])
        mae_on = np.array([np.mean(np.abs(on[1][band == i, k] - on[0][band == i, k]))
                           if (band == i).any() else np.nan
                           for i in range(len(BAND_LABEL))])
        mae_off = np.array([np.mean(np.abs(off[1][band == i, k] - off[0][band == i, k]))
                            if (band == i).any() else np.nan
                            for i in range(len(BAND_LABEL))])

        ax.bar(x - 0.2, mae_off, 0.38, color=viz.MUTED, label="balancing off")
        ax.bar(x + 0.2, mae_on, 0.38, color=viz.BLUE, label="balancing on")
        for i, (a, b) in enumerate(zip(mae_off, mae_on)):
            d = b - a
            ax.text(i, max(a, b) + max(mae_off) * .03, f"{d:+.1f}", ha="center",
                    fontsize=8.4, weight="bold", color=viz.BAD if d > 0 else viz.OK)
        ax.axhline(5, ls="--", lw=1.2, color=viz.BAD)

        overall_on = float(np.mean(np.abs(on[1][:, k] - on[0][:, k])))
        overall_off = float(np.mean(np.abs(off[1][:, k] - off[0][:, k])))
        ax.axhline(overall_on, ls=":", lw=1.6, color=viz.BLUE)
        ax.text(0.015, 0.97,
                f"dotted line = MAE over all {len(on[0]):,} clips\n"
                f"balancing on {overall_on:.2f}   ·   off {overall_off:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.4, color=viz.BLUE,
                weight="bold", linespacing=1.4)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{l}\n{c:,} clips" for l, c in zip(BAND_LABEL, n)],
                           fontsize=8.2)
        ax.set_ylabel(f"{target.upper()} MAE (mmHg)")
        viz.title(ax, f"{target.upper()} error by blood-pressure band",
                  f"{MODEL_LABEL[model]} · {protocol} · bands cut on systolic · "
                  f"bars average to the dotted line")
        if row == 0:
            ax.legend(loc="upper center")
        viz.despine(ax)

        # the share panel is the same both rows; draw it once and blank the other
        ax2 = axes[row][1]
        if row == 0:
            share = n / n.sum() * 100
            ax2.barh(x, share, 0.6, color=viz.ORANGE)
            for i, v in enumerate(share):
                ax2.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=8.2,
                         color=viz.INK_2)
            ax2.set_yticks(x)
            ax2.set_yticklabels(BAND_LABEL, fontsize=8.2)
            ax2.set_xlim(0, max(share) * 1.28)
            ax2.set_xlabel("share of test clips")
            viz.title(ax2, "Why it is needed", "the training set has the same shape")
            viz.despine(ax2)
        else:
            ax2.axis("off")
            ax2.grid(False)

    cost = overall_on - overall_off
    head = ("Balancing costs a little on the average and buys a lot on the patients "
            "who matter" if cost > 0.5 else
            "Balancing buys accuracy on the patients who matter at almost no cost")
    fig.suptitle(f"{head} — {MODEL_LABEL[model]}", fontsize=13, weight="bold",
                 x=0.008, ha="left", y=1.0)
    plt.tight_layout()
    save(fig, "1_bp_band_error.png", model=model)


def fig_bland_altman(runs):
    """Spec section K: the standard BP method-comparison plot.

    One figure per model — systolic on the top row, diastolic on the bottom,
    four protocols across. A cloud that tilts downwards is a model regressing to
    the population mean: it reads high for the hypotensive and low for the
    hypertensive, which an MAE alone will not show you.
    """
    models, protos = _present(runs)
    for m in models:
        cols = [p for p in protos if predictions(m, p) is not None]
        if not cols:
            continue
        fig, axes = plt.subplots(2, len(cols), figsize=(3.5 * len(cols), 6.8),
                                 squeeze=False)
        for row, target in enumerate(("sbp", "dbp")):
            k = 0 if target == "sbp" else 1
            for col, p in enumerate(cols):
                ax = axes[row][col]
                y, yp, _ = predictions(m, p)
                mean = (y[:, k] + yp[:, k]) / 2
                diff = yp[:, k] - y[:, k]
                bias, sd = diff.mean(), diff.std(ddof=1)
                idx = np.random.default_rng(0).choice(len(mean), min(6000, len(mean)),
                                                      replace=False)
                ax.scatter(mean[idx], diff[idx], s=3, alpha=.10,
                           color=PROTO_COLOR[p], edgecolors="none")
                ax.axhline(bias, color=viz.INK, lw=1.2)
                for sgn in (1.96, -1.96):
                    ax.axhline(bias + sgn * sd, color=viz.INK_2, lw=1, ls="--")
                ax.text(0.97, 0.03, f"bias {bias:+.1f}\n95% limits ±{1.96 * sd:.0f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        color=viz.INK_2)
                if row == 0:
                    ax.set_title(p, fontsize=10, color=PROTO_COLOR[p])
                ax.set_xlabel(f"mean of true and predicted {target.upper()}")
                if col == 0:
                    ax.set_ylabel(f"{target.upper()}  predicted − true (mmHg)")
                ax.set_xlim(*( (40, 210) if target == "sbp" else (20, 130) ))
                ax.set_ylim(*( (-90, 90) if target == "sbp" else (-60, 60) ))
                viz.despine(ax)
        fig.suptitle(f"Bland-Altman — {MODEL_LABEL[m]}.  A downward tilt means the "
                     f"model is pulling every patient towards the population average.",
                     fontsize=11.5, weight="bold", x=0.005, ha="left", y=1.0)
        plt.tight_layout()
        save(fig, "2_bland_altman.png", model=m)


def fig_compliance(runs):
    """AAMI pass/fail and BHS grade, as a table. Spec section K's primary metric."""
    models, protos = _present(runs)
    rows = [(m, p) for m in models for p in protos if (m, p) in runs]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(14.5, 0.40 * len(rows) + 1.7))
    ax.axis("off")
    ax.grid(False)
    cols = ["model", "protocol", "SBP MAE", "SBP SD", "AAMI", "BHS",
            "DBP MAE", "DBP SD", "AAMI", "BHS"]
    xs = [0, 15, 29, 39, 49, 57, 66, 76, 86, 94]
    ax.set_xlim(-1, 100)
    ax.set_ylim(-len(rows) - 1.6, 1.8)
    for x, c in zip(xs, cols):
        ax.text(x, 0.7, c, fontsize=9, weight="bold", color=viz.INK)
    ax.plot([-1, 100], [0.2, 0.2], color=viz.GRID, lw=1)

    for i, (m, p) in enumerate(rows):
        r = runs[(m, p)]["results"]
        y = -i - 0.55
        cells = [MODEL_LABEL[m], p]
        for t in ("sbp", "dbp"):
            cells += [f"{r[t]['mae']:.2f}", f"{r[t]['sde']:.2f}",
                      "PASS" if r[t]["aami_pass"] else "FAIL", r[t]["bhs"]]
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        vals = [cells[0], cells[1], cells[2], cells[3], cells[4], cells[5],
                cells[6], cells[7], cells[8], cells[9]]
        for x, v, k in zip(xs, vals, order):
            colour = viz.INK_2
            weight = "normal"
            if v in ("PASS", "FAIL"):
                colour = viz.OK if v == "PASS" else viz.BAD
                weight = "bold"
            elif k in (5, 9):
                colour = {"A": viz.OK, "B": viz.OK, "C": viz.WARN}.get(v, viz.BAD)
                weight = "bold"
            elif k == 1:
                colour = PROTO_COLOR[p]
                weight = "bold"
            ax.text(x, y, v, fontsize=8.8, color=colour, weight=weight)
        ax.plot([-1, 100], [y - 0.3, y - 0.3], color=viz.GRID, lw=0.6)

    ax.text(0, -len(rows) - 1.1,
            "AAMI passes when the mean error is within ±5 mmHg and its SD within 8. "
            "BHS grades on the share of errors under 5 / 10 / 15 mmHg (A is best, D is a fail).",
            fontsize=8.6, color=viz.MUTED, va="top")
    ax.set_title("Clinical compliance", loc="left", fontsize=13, pad=16)
    plt.tight_layout()
    save(fig, "2_compliance.png")


def fig_domain_shift():
    """The PulseDB-trained model turned loose on a wrist watch.

    PPG-DaLiA has no BP labels, so there is no error to plot. What can be shown
    is where the predictions land: if a model trained on ICU patients tells you
    that fifteen healthy volunteers in their twenties are all hypertensive, the
    number it reports is not usable, and that is visible without ground truth.
    """
    files = sorted(MODEL_RESULTS.glob("*/*_dalia_domain_shift.json"))
    if not files:
        return
    for f in files:
        d = json.loads(f.read_text())
        model, tag = d["model"], d.get("tag", "")
        # the prediction file shares the JSON's stem, so a tagged run cannot
        # silently pick up the untagged run's predictions
        stem = f.name[: -len("_domain_shift.json")]
        pred_file = f.parent / f"{stem}_predictions.npz"
        if not pred_file.exists():
            continue
        z = np.load(pred_file, allow_pickle=True)
        p_dalia, activity = z["y_pred"], z["activity"]
        pulse = predictions(model, "calfree", tag=tag)
        if pulse is None:
            continue
        y_pulse, p_pulse, _ = pulse

        fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.5),
                               gridspec_kw=dict(width_ratios=[1.25, 1.35, 0.95]))

        # 1 — where the predictions land
        bins = np.linspace(60, 220, 90)
        for v, lab, c, style in (
                (y_pulse[:, 0], "PulseDB — true BP", viz.MUTED, dict(alpha=.55)),
                (p_pulse[:, 0], "PulseDB — predicted", viz.BLUE, dict(alpha=.55)),
                (p_dalia[:, 0], "PPG-DaLiA — predicted", viz.ORANGE, dict(alpha=.65))):
            ax[0].hist(v, bins=bins, density=True, color=c, label=lab,
                       edgecolor="none", **style)
        for v, c in ((y_pulse[:, 0], viz.MUTED), (p_pulse[:, 0], viz.BLUE),
                     (p_dalia[:, 0], viz.ORANGE)):
            ax[0].axvline(v.mean(), color=c, lw=1.6, ls="--")
        shift = p_dalia[:, 0].mean() - p_pulse[:, 0].mean()
        # sits below the peaks so it cannot land under the legend
        yline = ax[0].get_ylim()[1] * 0.45
        ax[0].annotate("", xy=(p_dalia[:, 0].mean(), yline),
                       xytext=(p_pulse[:, 0].mean(), yline),
                       arrowprops=dict(arrowstyle="<->", color=viz.BAD, lw=1.8))
        ax[0].text((p_pulse[:, 0].mean() + p_dalia[:, 0].mean()) / 2,
                   yline * 1.06, f"+{shift:.0f} mmHg", ha="center",
                   color=viz.BAD, fontsize=10.5, weight="bold")
        ax[0].set_yticks([])
        ax[0].set_xlabel("SBP (mmHg)")
        ax[0].legend(loc="upper right", fontsize=8.2)
        viz.title(ax[0], "Where the predictions land",
                  "the same model, on ICU patients and on a wrist watch")
        viz.despine(ax[0], keep=("bottom",))

        # 2 — by activity
        acts = sorted(set(activity), key=lambda a: p_dalia[activity == a, 0].mean())
        means = [p_dalia[activity == a, 0].mean() for a in acts]
        sds = [p_dalia[activity == a, 0].std() for a in acts]
        ns = [int((activity == a).sum()) for a in acts]
        y = np.arange(len(acts))
        ax[1].barh(y, means, xerr=sds, color=viz.ORANGE, height=.62,
                   error_kw=dict(ecolor=viz.INK_2, lw=1, capsize=2.5))
        ax[1].axvline(p_pulse[:, 0].mean(), color=viz.BLUE, ls="--", lw=1.5)
        ax[1].text(p_pulse[:, 0].mean(), len(acts) - 0.35,
                   f"PulseDB predicted mean  {p_pulse[:, 0].mean():.0f}",
                   ha="center", va="bottom", fontsize=8.2, color=viz.BLUE,
                   weight="bold")
        ax[1].set_ylim(-0.7, len(acts) - 0.1)
        ax[1].set_yticks(y)
        ax[1].set_yticklabels([f"{a}  ({n:,})" for a, n in zip(acts, ns)],
                              fontsize=8.4)
        ax[1].set_xlabel("predicted SBP (mmHg)")
        viz.title(ax[1], "By what the volunteer was doing",
                  "bars are the mean, whiskers the spread within that activity")
        viz.despine(ax[1])

        # 3 — has it stopped telling people apart
        c = d["collapse"]["sbp"]
        keys = ["pulsedb_true", "pulsedb_predicted", "dalia_predicted"]
        labs = ["PulseDB\ntrue", "PulseDB\npredicted", "PPG-DaLiA\npredicted"]
        cols = [viz.MUTED, viz.BLUE, viz.ORANGE]
        ax[2].bar(range(3), [c[k] for k in keys], color=cols, width=.62)
        for i, k in enumerate(keys):
            ax[2].text(i, c[k] + 0.2, f"{c[k]:.1f}", ha="center", fontsize=9,
                       weight="bold", color=viz.INK_2)
        ax[2].set_xticks(range(3))
        ax[2].set_xticklabels(labs, fontsize=8.4)
        ax[2].set_ylabel("between-subject SD of SBP (mmHg)")
        viz.title(ax[2], "Can it still tell people apart?",
                  "lower means everyone gets the same answer")
        viz.despine(ax[2])

        imp = d["plausibility"]["pct_any_impossible"]
        fig.suptitle(
            f"{MODEL_LABEL[model]} trained on PulseDB, run on PPG-DaLiA — "
            f"{d['n_dalia_clips']:,} clips from {d['n_dalia_subjects']} volunteers.  "
            f"Only {imp:.2f}% of answers are physiologically impossible, but the "
            f"whole distribution has moved {shift:.0f} mmHg."
            + ("   [BP-bin balancing off]" if tag == "nobalance" else ""),
            fontsize=11.5, weight="bold", x=0.005, ha="left", y=1.02)
        plt.tight_layout()
        save(fig, f"3_domain_shift{'_' + tag if tag else ''}.png", model=model)


def fig_domain_shift_control(runs):
    """How much of the measured domain shift is the domain, and how much is us.

    BP-bin balancing pushes a model to spread its predictions apart. That spread
    does not survive the move to a wrist sensor, so a balanced model appears both
    to shift further and to stop telling people apart. Running the same test on
    the unbalanced model separates the two, and the difference is large enough
    that reporting only the balanced number would overstate the shift by half.
    """
    rows = []
    for m in _present(runs)[0]:
        d = MODEL_RESULTS / m
        bal = d / f"{m}_dalia_domain_shift.json"
        nob = d / f"{m}_nobalance_dalia_domain_shift.json"
        if not (bal.exists() and nob.exists()):
            continue
        a, b = (json.loads(x.read_text()) for x in (bal, nob))
        r = dict(model=m)
        for t in ("sbp", "dbp"):
            get = lambda j, k, t=t: j["distribution"][t][k]["mean"]      # noqa: E731
            col = lambda j, k, t=t: j["collapse"][t][k]                  # noqa: E731
            r[t] = dict(
                shift_bal=get(a, "dalia_predicted") - get(a, "pulsedb_predicted"),
                shift_nob=get(b, "dalia_predicted") - get(b, "pulsedb_predicted"),
                sd_pulse_bal=col(a, "pulsedb_predicted"),
                sd_dalia_bal=col(a, "dalia_predicted"),
                sd_pulse_nob=col(b, "pulsedb_predicted"),
                sd_dalia_nob=col(b, "dalia_predicted"),
                sd_true=col(a, "pulsedb_true"))
        # the shift figure keys off systolic; diastolic rides along in r["dbp"]
        r.update(**r["sbp"])
        rows.append(r)
    if not rows:
        return

    # Two separate figures: each answers a different question, and putting both
    # on one slide made each too small to read.
    x = np.arange(len(rows))
    w = 0.36

    # ---- 1. how far the predictions moved -------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(x - w / 2, [r["shift_bal"] for r in rows], w, color=viz.MUTED,
           label="with BP-bin weighting  (what we measured first)")
    ax.bar(x + w / 2, [r["shift_nob"] for r in rows], w, color=viz.BLUE,
           label="without it  (the real shift)")
    for i, r in enumerate(rows):
        ax.text(i - w / 2, r["shift_bal"] + .6, f"+{r['shift_bal']:.1f}", ha="center",
                fontsize=10, color=viz.INK_2)
        ax.text(i + w / 2, r["shift_nob"] + .6, f"+{r['shift_nob']:.1f}", ha="center",
                fontsize=10, weight="bold", color=viz.BLUE)
        diff = r["shift_bal"] - r["shift_nob"]
        ax.text(i, max(r["shift_bal"], r["shift_nob"]) + 3.2,
                f"{diff:.1f} of it — {diff / r['shift_bal'] * 100:.0f}% —\nwas the weighting",
                ha="center", fontsize=9.5, weight="bold", color=viz.BAD, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[r["model"]] for r in rows], fontsize=10.5)
    ax.set_ylabel("how much higher it reads on the watch\nthan on hospital patients (mmHg)")
    ax.set_ylim(0, max(r["shift_bal"] for r in rows) * 1.45)
    ax.legend(loc="upper left", fontsize=9.5)
    viz.title(ax, "Systolic \u2014 how far the predictions moved",
              "grey is what we measured first; blue is the same test without our weighting")
    viz.despine(ax)
    plt.tight_layout()
    save(fig, "3_shift_amount.png")

    # the same figure for diastolic
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.bar(x - w / 2, [r["dbp"]["shift_bal"] for r in rows], w, color=viz.MUTED,
           label="with BP-bin weighting  (what we measured first)")
    ax.bar(x + w / 2, [r["dbp"]["shift_nob"] for r in rows], w, color=viz.BLUE,
           label="without it  (the real shift)")
    for i, r in enumerate(rows):
        dd = r["dbp"]
        ax.text(i - w / 2, dd["shift_bal"] + .3, f"+{dd['shift_bal']:.1f}",
                ha="center", fontsize=10, color=viz.INK_2)
        ax.text(i + w / 2, dd["shift_nob"] + .3, f"+{dd['shift_nob']:.1f}",
                ha="center", fontsize=10, weight="bold", color=viz.BLUE)
        diff = dd["shift_bal"] - dd["shift_nob"]
        ax.text(i, max(dd["shift_bal"], dd["shift_nob"]) + 1.8,
                f"{diff:.1f} of it \u2014 {diff / dd['shift_bal'] * 100:.0f}% \u2014\nwas the weighting",
                ha="center", fontsize=9.5, weight="bold", color=viz.BAD, linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[r["model"]] for r in rows], fontsize=10.5)
    ax.set_ylabel("how much higher it reads on the watch\nthan on hospital patients (mmHg)")
    ax.set_ylim(0, max(r["dbp"]["shift_bal"] for r in rows) * 1.45)
    ax.legend(loc="upper left", fontsize=9.5)
    viz.title(ax, "Diastolic \u2014 how far the predictions moved",
              "same test, same models, the other target")
    viz.despine(ax)
    plt.tight_layout()
    save(fig, "3_shift_amount_dbp.png")

    # ---- 2. does it still tell people apart -----------------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for i, r in enumerate(rows):
        ax.plot([i - .16, i - .16], [r["sd_pulse_bal"], r["sd_dalia_bal"]],
                color=viz.MUTED, lw=3, marker="o", ms=7,
                label="with BP-bin weighting" if i == 0 else None)
        ax.plot([i + .16, i + .16], [r["sd_pulse_nob"], r["sd_dalia_nob"]],
                color=viz.BLUE, lw=3, marker="o", ms=7,
                label="without it" if i == 0 else None)
        top, bot = r["sd_pulse_nob"], r["sd_dalia_nob"]
        if top - bot < 1.0:
            # the two ends coincide; two labels would sit on top of each other
            ax.text(i + .30, (top + bot) / 2,
                    f"{top:.1f}→{bot:.1f} barely moves", fontsize=8,
                    weight="bold", color=viz.BLUE, va="center")
        else:
            ax.text(i + .30, top, f"{top:.1f} hospital",
                    fontsize=8, color=viz.BLUE, va="center")
            ax.text(i + .30, bot, f"{bot:.1f} watch",
                    fontsize=8, weight="bold", color=viz.BLUE, va="center")
    ax.axhline(rows[0]["sd_true"], ls="--", lw=1.4, color=viz.BAD)
    ax.text(len(rows) - .55, rows[0]["sd_true"] + .35,
            f"how different these patients really are  {rows[0]['sd_true']:.1f}",
            ha="right", fontsize=9.5, color=viz.BAD, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[r["model"]] for r in rows], fontsize=10.5)
    ax.set_ylabel("how differently it answers for different people\n(mmHg)")
    ax.set_xlim(-.5, len(rows) - .1)
    top_val = max(max(r["sd_pulse_bal"], r["sd_dalia_bal"],
                       r["sd_pulse_nob"], r["sd_dalia_nob"]) for r in rows)
    ax.set_ylim(0, max(top_val * 1.08, rows[0]["sd_true"] * 1.25))
    ax.legend(loc="lower left", fontsize=9.5)
    viz.title(ax, "Can it still tell people apart?",
              "each line drops from hospital data to the watch — a long drop means it stopped")
    viz.despine(ax)
    plt.tight_layout()
    save(fig, "3_shift_collapse.png")


def fig_training_curves(runs):
    """Where the deep models stop generalising and start memorising.

    Train and validation loss per epoch, one panel per protocol. The gap between
    the two curves is the whole story: on this task the validation loss reaches
    its floor within a few epochs while the training loss keeps falling for
    another fifteen, which is the model learning the training clips rather than
    learning blood pressure. It is the same claim as the leakage gap, visible
    inside a single run.
    """
    for m in _present(runs)[0]:
        panels = [(p, runs[(m, p)]) for p in PROTO_ORDER
                  if (m, p) in runs and runs[(m, p)]["info"].get("history")]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.6),
                                 squeeze=False, sharey=True)
        for k, (proto, r) in enumerate(panels):
            ax = axes[0][k]
            h = r["info"]["history"]
            ep = [x["epoch"] for x in h]
            tr = [x["train_loss"] for x in h]
            va = [x["val_loss"] for x in h]
            best = min(h, key=lambda x: x["val_loss"])
            ax.plot(ep, tr, color=viz.MUTED, lw=1.8, label="train")
            ax.plot(ep, va, color=PROTO_COLOR[proto], lw=2.0, label="validation")
            ax.axvline(best["epoch"], ls=":", lw=1.4, color=viz.BAD)
            ax.text(best["epoch"] + 0.3, max(tr) * 0.95,
                    f"best val\nepoch {best['epoch']}", fontsize=8,
                    color=viz.BAD, weight="bold", va="top", linespacing=1.3)
            ax.set_xlabel("epoch")
            if k == 0:
                ax.set_ylabel("loss")
                ax.legend(loc="upper right", fontsize=8.4)
            ax.set_title(proto, fontsize=10, color=PROTO_COLOR[proto])
            viz.despine(ax)
        fig.suptitle(f"{MODEL_LABEL[m]} — validation stops improving long before "
                     f"training does.  Everything after the dotted line is the model "
                     f"memorising its training clips.",
                     fontsize=11.5, weight="bold", x=0.005, ha="left", y=1.02)
        plt.tight_layout()
        save(fig, "4_training_curve.png", model=m)


def fig_personalization():
    # the primary (BP-bin balanced) curves only — the "_nobalance" ablation
    # files share the same model/source/pick labels and would otherwise
    # double every legend entry with no visual way to tell them apart.
    files = sorted(f for f in MODEL_RESULTS.glob("*/*_personalization_*.json")
                    if "nobalance" not in f.name)
    if not files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
    picks_seen = []
    for f in files:
        d = json.loads(f.read_text())
        for ax, target in zip(axes, ("sbp", "dbp")):
            for pick, curve in d["curves"].items():
                k = [r["shots"] for r in curve]
                v = [r["results"][target]["mae"] for r in curve]
                label = f"{d['model']} · {pick}"
                ax.plot(k, v, "-", marker="o", ms=4, label=label)
                picks_seen.append(label)
    for ax, target in zip(axes, ("sbp", "dbp")):
        ax.axhline(5, ls="--", color=viz.BAD, lw=1.3)
        ax.text(ax.get_xlim()[1], 5.2, "AAMI 5 mmHg", ha="right", fontsize=8,
                color=viz.BAD, weight="bold")
        ax.set_xlabel("calibration readings per new subject (k)")
        ax.set_ylabel("MAE (mmHg)")
        viz.title(ax, target.upper(), "k = 0 is the uncalibrated number")
        viz.despine(ax)
    axes[1].legend(fontsize=7.4, ncol=1, loc="upper left",
                   bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.suptitle("How many cuff readings a new user has to provide",
                 fontsize=13, weight="bold", x=0.008, ha="left", y=1.02)
    plt.tight_layout()
    save(fig, "5_personalization.png")


if __name__ == "__main__":
    runs = load()
    if not runs:
        print(f"no runs in {MODEL_RESULTS} yet — train something first")
        raise SystemExit(0)
    models, protos = _present(runs)
    print(f"{len(runs)} runs: models {models}, protocols {protos}\n")
    fig_leakage_gap(runs)
    fig_bp_band_error(runs)
    fig_bland_altman(runs)
    fig_compliance(runs)
    fig_domain_shift()
    fig_domain_shift_control(runs)
    fig_training_curves(runs)
    fig_personalization()
