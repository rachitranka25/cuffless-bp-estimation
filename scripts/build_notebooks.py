"""Generate the training notebooks.

The notebooks are launchers, not code. Every one of them mounts the data, calls
`train.run` (or `train.personalize`), and stops. Nothing about the pipeline lives
in a notebook, because five hand-edited copies of a pipeline drift, and then an
architecture difference and a batch-size difference are indistinguishable in the
results table — which would break the one comparison this project exists to make.

They are generated rather than hand-written for the same reason: the setup cells
are identical by construction, so no notebook can quietly acquire a different
learning rate.

Run:  python3 scripts/build_notebooks.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import ROOT      # noqa: E402

NB = ROOT / "notebooks"


def md(*lines):
    return dict(cell_type="markdown", metadata={}, source="\n".join(lines))


def code(*lines):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source="\n".join(lines))


def write(name, cells):
    # nbformat >= 4.5 wants a stable per-cell id; index-based keeps diffs readable
    for i, c in enumerate(cells):
        c["id"] = f"c{i:02d}"
    nb = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    NB.mkdir(parents=True, exist_ok=True)
    (NB / name).write_text(json.dumps(nb, indent=1))
    print(f"wrote notebooks/{name}  ({len(cells)} cells)")


# --------------------------------------------------------------------------- #
# the setup every Colab notebook shares
# --------------------------------------------------------------------------- #

SETUP_CODE = [
    "import os, sys, glob, shutil, time",
    "",
    "IN_COLAB = 'google.colab' in sys.modules or os.path.isdir('/content/drive')",
    "",
    "if not IN_COLAB:",
    "    # running from notebooks/ on a normal machine",
    "    sys.path.insert(0, '../src')",
    "    sys.path.insert(0, '../scripts')",
    "else:",
    "    from google.colab import drive",
    "    drive.mount('/content/drive')",
    "",
    "    # Find the bp folder wherever it was dropped in Drive.",
    "    hits = glob.glob('/content/drive/MyDrive/**/data/colab_manifest.json',",
    "                     recursive=True)",
    "    if len(hits) != 1:",
    "        raise SystemExit(f'expected exactly one bp folder, found {len(hits)}:'",
    "                         + ''.join('\\n  ' + h for h in hits))",
    "    DRIVE_ROOT = os.path.dirname(os.path.dirname(hits[0]))",
    "    LOCAL_ROOT = '/content/bp'",
    "    print('drive:', DRIVE_ROOT)",
    "",
    "    # Copy the data to local disk first. Drive is a network mount and the",
    "    # sampler reads clips in random order, so training straight off Drive",
    "    # leaves the GPU idle waiting on I/O. A few minutes here saves hours.",
    "    t0 = time.time()",
    "    for rel in ['src', 'scripts', 'data/colab_manifest.json', 'data/features',",
    "                'data/processed/pulsedb', 'data/processed/ppg_dalia/windows']:",
    "        src, dst = f'{DRIVE_ROOT}/{rel}', f'{LOCAL_ROOT}/{rel}'",
    "        if os.path.exists(dst):",
    "            continue",
    "        os.makedirs(os.path.dirname(dst), exist_ok=True)",
    "        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)",
    "        print(f'  copied {rel:40s} {time.time() - t0:5.0f}s')",
    "",
    "    # results/ points back at Drive — metrics, predictions and checkpoints",
    "    # all land under results/models/<name>/, so a dropped session does not",
    "    # take the run with it.",
    "    os.makedirs(f'{DRIVE_ROOT}/results/models', exist_ok=True)",
    "    if not os.path.islink(f'{LOCAL_ROOT}/results'):",
    "        os.symlink(f'{DRIVE_ROOT}/results', f'{LOCAL_ROOT}/results')",
    "",
    "    os.environ['BP_ROOT'] = LOCAL_ROOT     # every path follows this",
    "    sys.path.insert(0, LOCAL_ROOT + '/src')",
    "    sys.path.insert(0, LOCAL_ROOT + '/scripts')",
    "",
    "    # Fail now if the upload was incomplete, not twenty minutes into a run.",
    "    from prepare_colab_upload import verify",
    "    if verify(LOCAL_ROOT):",
    "        raise SystemExit('copy is incomplete — see the list above')",
    "    print(f'\\nBP_ROOT = {LOCAL_ROOT}   (ready in {time.time() - t0:.0f}s)')",
]


def colab_setup(gpu=True):
    cells = [
        md("## Setup",
           "",
           "Run these cells in order before anything else. They work both on Colab",
           "and on a laptop — on Colab the data is copied off Drive onto the local",
           "disk first, because Drive is a network mount and the sampler reads clips",
           "in random order, which would leave the GPU waiting on I/O. The copy also",
           "stops immediately if the upload turned out to be incomplete."),
    ]
    if gpu:
        cells.append(code(
            "# Needs a GPU. Runtime > Change runtime type > T4 GPU.",
            "!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'NO GPU'"))
    cells.append(code(*SETUP_CODE))
    if gpu:
        cells.append(code(
            "import torch",
            "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print(torch.cuda.get_device_name(0))",
            "else:",
            "    print('WARNING: no GPU — set the runtime type and restart')"))
    return cells


def result_cell():
    return code(
        "# Everything this session produced, on Drive",
        "from paths import MODEL_RESULTS",
        "for p in sorted(MODEL_RESULTS.glob('*/*.json')):",
        "    print(f'{p.stat().st_size / 1e3:8.1f} kB  {p.name}')")


# --------------------------------------------------------------------------- #
# 02 — trees, on the laptop
# --------------------------------------------------------------------------- #

def baselines():
    cells = [
        md("# 02 — Baselines: Random Forest and Gradient Boosting",
           "",
           "The first models the spec asks for (section I): *\"Random Forest /",
           "Gradient Boosting on hand-crafted pulse-transit-time and PPG-morphology",
           "features — classical, interpretable, fast baseline widely used in this",
           "exact literature\"*.",
           "",
           "No GPU needed — this reads `data/features/*.parquet`, 61 numbers per",
           "clip instead of the waveform. It runs on a laptop or on Colab.",
           "",
           "**On Colab, cap the random forest.** The free tier gives two CPU cores,",
           "and 300 trees over 465,480 clips will crawl. `run(model='rf', ..., cap=100)`",
           "caps it at 100 clips per training subject. Gradient boosting is fine",
           "either way. On a laptop, run both uncapped.",
           "",
           "This is also the cheapest sanity check in the project: if the",
           "calibration-free MAE lands near the 12-13 mmHg the published PulseDB",
           "papers report, the whole pipeline — data, features, splits, metrics — is",
           "confirmed in ten minutes rather than after an hour and a half of GPU time.",
           "",
           "Each model is scored under all four protocols. The model, the features",
           "and the training clips are identical across the four; the only thing",
           "that changes is which patients are in the test set.",
           "",
           "| protocol | who is tested | what it tells you |",
           "|---|---|---|",
           "| `leaky` | same patients, random clips | the number much of the field reports |",
           "| `calbased` | same patients, official held-out clips | a device that already knows you |",
           "| `calfree` | 144 unseen patients | **the honest number** |",
           "| `aami` | 116 unseen patients | the zero-shot end of the personalization curve |",
           "",
           "Roughly 10 minutes for gradient boosting, 40 for the random forest."),
        *colab_setup(gpu=False),
        code("from train import run, PROTOCOLS",
             "print(PROTOCOLS)"),
        md("### Gradient boosting, all four protocols"),
        code("gb = {p: run(model='gb', protocol=p) for p in PROTOCOLS}"),
        md("### Random forest, all four protocols",
           "",
           "Slower (300 trees). Pass `cap=100` for a quick pass — at most 100 clips",
           "per training subject, about five minutes."),
        code("rf = {p: run(model='rf', protocol=p) for p in PROTOCOLS}"),
        md("### The comparability run",
           "",
           "Balancing is on by default because spec section L asks for it, and it "
           "lowers the error on the hypertensive patients considerably. It also "
           "raises the headline MAE, and every published PulseDB number was measured "
           "without it — so one extra run per model gives the figure that lines up "
           "against the literature."),
        code("nb = {m: run(model=m, protocol='calfree', balance=False, tag='nobalance')",
             "      for m in ('gb', 'rf')}"),
        md("### The leakage gap in one table",
           "",
           "This is the project's actual output: one model, one training set, four",
           "answers, differing only in who was tested."),
        code("import pandas as pd",
             "",
             "rows = []",
             "for name, res in (('gb', gb), ('rf', rf)):",
             "    res = dict(res, **{'calfree (no bal.)': nb[name]})",
             "    for proto, out in res.items():",
             "        for target in ('sbp', 'dbp'):",
             "            m = out['results'][target]",
             "            rows.append(dict(model=name, protocol=proto, target=target,",
             "                             mae=m['mae'], sde=m['sde'], r=m['r'],",
             "                             aami='PASS' if m['aami_pass'] else 'FAIL',",
             "                             bhs=m['bhs']))",
             "",
             "df = pd.DataFrame(rows).round(2)",
             "display(df.pivot_table(index=['model', 'target'], columns='protocol',",
             "                       values='mae').round(2))",
             "df"),
        md("### Hyperparameters",
           "",
           "Spec section L: *\"k-fold (subject-level) within the training subjects for",
           "hyperparameter tuning\"*. Folds are formed over subjects, never over clips,",
           "so a candidate is never rewarded for memorising a patient. None of the",
           "test subsets are touched."),
        code("from train import tune",
             "",
             "grid = [dict(learning_rate=lr, max_leaf_nodes=n)",
             "        for lr in (0.03, 0.06, 0.12) for n in (31, 63)]",
             "best = tune(model='gb', grid=grid, folds=5, cap=100)"),
    ]
    write("02_baselines.ipynb", cells)


# --------------------------------------------------------------------------- #
# 03a/b/c — the deep models, on Colab
# --------------------------------------------------------------------------- #

DEEP_NB = {
    "cnn": ("03a", "1D-CNN", 50, 256,
            ["The second model in spec section I: *\"1D-CNN on raw PPG (+ECG)",
             "waveform — standard deep baseline\"*.",
             "",
             "This one never sees the 61 hand-crafted features. It gets the waveform",
             "itself — 2 x 1250 numbers per clip — and works out what matters on its",
             "own. Whether that beats measuring the pulse by hand is one of the",
             "ablations the spec asks for.",
             "",
             "About 3.5-4 hours on a T4 at 50 epochs."]),
    "resnet": ("03b", "ResNet1D", 50, 256,
               ["The third model in spec section I: *\"ResNet1D / lightweight",
                "Transformer — stronger deep baseline reported in recent",
                "PulseDB-benchmarking papers\"*.",
                "",
                "Same convolutional idea as the CNN, with skip connections so a much",
                "deeper stack still trains. This is the standard strong baseline",
                "across the PulseDB literature.",
                "",
                "About 6 hours on a T4 at 50 epochs."]),
    "transformer": ("03c", "Transformer (DMT recipe)", 50, 32,
                    ["Spec section I: *\"lightweight Transformer\"*. Architecture follows",
                     "DMT (arXiv:2606.11125): the 1250-sample clip becomes 125 patches,",
                     "six blocks of eight-head self-attention, every block FiLM-conditioned",
                     "on age, sex and BMI.",
                     "",
                     "Training recipe follows DMT's own reported settings as closely as this",
                     "project's constraints allow: PPG-only input, per-clip z-score, Adam",
                     "(betas 0.9/0.999, weight decay 1e-8) at a fixed lr of 2e-5 (no",
                     "schedule), batch size 32, and the auxiliary morphology-classification",
                     "head with Kendall-style learnable multi-task uncertainty weights (Eq. 9",
                     "of the paper) — see `Config`'s docstring in `src/train.py` for exactly",
                     "what each knob does and which ones are approximations rather than",
                     "reproductions (the classification *label* itself is this project's own",
                     "stand-in for a formula the paper cites but doesn't restate).",
                     "",
                     "Two deliberate deviations, by instruction rather than limitation: one",
                     "joint network for SBP+DBP instead of DMT's two, and 50 epochs instead",
                     "of 100, so the training *budget* stays comparable to CNN/ResNet.",
                     "",
                     "**This cannot run on the laptop.** Apple's MPS backend is slower",
                     "than its own CPU for attention at this shape (92.8 ms vs 30.5 ms",
                     "per call). On an H100, 50 epochs at batch 32 took about 4 hours in",
                     "testing (~293s/epoch) — batch 32 means ~4x the steps per epoch of",
                     "batch 128, so budget more time than the CNN/ResNet notebooks."]),
}


# Extra run() keyword arguments, spliced verbatim into every training/eval call
# in deep()'s cells — only the transformer carries any, for the DMT-recipe
# knobs (see src/train.py's Config and models.DMT's docstring for what each
# one does and where it's an approximation rather than a reproduction).
DEEP_NB_EXTRA_KW = {
    "cnn": "",
    "resnet": "",
    "transformer": (", channels=(1,), optimizer='adam', lr=2e-5, "
                    "weight_decay=1e-8, lr_schedule='constant', "
                    "zscore=True, aux_morphology=True"),
}


def deep(model):
    num, title, epochs, bs, blurb = DEEP_NB[model]
    kw = DEEP_NB_EXTRA_KW[model]
    cells = [
        md(f"# {num} — {title}",
           "",
           *blurb,
           "",
           "---",
           "",
           "**This notebook holds no pipeline code.** Everything lives in",
           "`src/train.py`, shared by all five models and all four protocols, so the",
           "difference between this notebook and the other two really is only the",
           "architecture — not a batch size, not a split, not a scaling choice. The",
           "whole comparison depends on that.",
           "",
           "To change how training works, edit `src/train.py` and re-upload `src/`",
           "to Drive (0.1 MB). The data does not need re-uploading."),
        *colab_setup(),
        md("## Training",
           "",
           "`calfree` first: 144 patients the model has never seen, and the number",
           "that goes in the write-up. If the session drops later, the result that",
           "matters is already saved.",
           "",
           "Each run writes its metrics and its predictions to Drive, so the other",
           "protocols can be run on another day without repeating this one."),
        code("from train import run",
             "",
             f"MODEL  = '{model}'",
             f"EPOCHS = {epochs}",
             f"BATCH  = {bs}",
             "",
             "calfree = run(model=MODEL, protocol='calfree',",
             f"              epochs=EPOCHS, batch_size=BATCH{kw})"),
        md("## The comparability run",
           "",
           "One extra run: `calfree` with the BP-bin balancing switched off.",
           "",
           "Balancing is a spec requirement (section L) and it is on by default, but",
           "it changes the headline MAE — it trades average error for accuracy on the",
           "hypertensive patients, who are rare. Every published PulseDB number was",
           "measured without it, and section K makes comparability the reason MAE is",
           "the primary metric. So both are reported: the balanced number is this",
           "project's model, the unbalanced one is what lines up against the papers."),
        code("calfree_nobalance = run(model=MODEL, protocol='calfree', balance=False,",
             "                        tag='nobalance',",
             f"                        epochs=EPOCHS, batch_size=BATCH{kw})"),
        md("## Seed variance",
           "",
           "Two more runs each of `calfree` and `calfree_nobalance`, seeds 1 and 2 —",
           "three total per protocol with the runs above (seed 0). A single training",
           "run is a point estimate; the headline claims (this project vs. DMT, and",
           "the personalization result built on top of the balanced run) only mean",
           "something if they survive re-training. Saved with a `seed1`/`seed2` tag,",
           "so they sit beside the canonical runs rather than overwriting them."),
        code("seeds = {}",
             "for seed in (1, 2):",
             "    seeds[('calfree', seed)] = run(model=MODEL, protocol='calfree', seed=seed,",
             f"                                    tag=f'seed{{seed}}', epochs=EPOCHS, batch_size=BATCH{kw})",
             "    seeds[('nobalance', seed)] = run(model=MODEL, protocol='calfree', seed=seed,",
             "                                      balance=False, tag=f'nobalance_seed{seed}',",
             f"                                      epochs=EPOCHS, batch_size=BATCH{kw})"),
        md("## The other three protocols",
           "",
           "Same architecture, same settings, same training clips. Only the test",
           "patients change.",
           "",
           "This cell trains three more models and takes roughly three times as long",
           "as the one above — run it once you are happy with the `calfree` result."),
        code("others = {}",
             "for proto in ('calbased', 'aami', 'leaky'):",
             "    others[proto] = run(model=MODEL, protocol=proto,",
             f"                        epochs=EPOCHS, batch_size=BATCH{kw})"),
        md("### Seed variance, on calbased and leaky too",
           "",
           "The professor's follow-up: the architecture-dependent leakage-gap ratio "
           "(calfree vs. calbased/leaky) is only as trustworthy as both sides of it — "
           "calfree already gets 3 seeds above, this does the same for calbased and "
           "leaky. `aami` is left at one seed; it isn't part of the leakage-gap ratio."),
        code("proto_seeds = {}",
             "for proto in ('calbased', 'leaky'):",
             "    for seed in (1, 2):",
             "        proto_seeds[(proto, seed)] = run(model=MODEL, protocol=proto, seed=seed,",
             f"                                          tag=f'seed{{seed}}', epochs=EPOCHS, batch_size=BATCH{kw})"),
        md("## The domain-shift test",
           "",
           "Spec sections E, H and L: the model is trained on PulseDB and then run",
           "on PPG-DaLiA, a wrist-worn free-living dataset it has never seen. That",
           "external test is the design choice the spec calls *\"the most novel ...",
           "differentiating this from typical single-dataset cuffless-BP papers\"*.",
           "",
           "PPG-DaLiA carries no BP labels, so there is no error to report. What is",
           "measured instead: whether the answers stay physiologically possible,",
           "how far the whole prediction distribution moves, whether the model can",
           "still tell the fifteen volunteers apart, and whether it degrades with",
           "motion.",
           "",
           "This reuses the checkpoint saved by the `calfree` run above rather than",
           "training again — the same model has to score both sides."),
        code("from train import evaluate_dalia",
             "",
             "# restated here so this cell runs on its own after a session restart,",
             "# without re-running the training cells above",
             f"MODEL = '{model}'",
             "",
             "shift = evaluate_dalia(model=MODEL, protocol='calfree')"),
        md("### The control",
           "",
           "The same test on the unbalanced model, and it is not optional. BP-bin",
           "balancing pushes a model to spread its predictions apart; that spread",
           "is what appears to collapse on PPG-DaLiA, and part of the distribution",
           "shift comes from it too. On gradient boosting the control cut the",
           "measured shift from +32 to +19 mmHg and removed the collapse entirely.",
           "",
           "Without this run there is no way to say how much of the domain shift is",
           "the domain and how much is the weighting."),
        code("shift_nobalance = evaluate_dalia(model=MODEL, protocol='calfree',",
             "                                 tag='nobalance')"),
        md("### Seed variance, on PPG-DaLiA too",
           "",
           "The same domain-shift check, on the seed 1 and seed 2 checkpoints from",
           "above — inference only, no training, seconds per run. Confirms whether",
           "the shift numbers are as seed-stable as the calfree MAE."),
        code("dalia_seeds = {}",
             "for seed in (1, 2):",
             "    dalia_seeds[('calfree', seed)] = evaluate_dalia(",
             "        model=MODEL, protocol='calfree', seed=seed, tag=f'seed{seed}')",
             "    dalia_seeds[('nobalance', seed)] = evaluate_dalia(",
             "        model=MODEL, protocol='calfree', seed=seed, tag=f'nobalance_seed{seed}')"),
        md("## 3-seed ensemble",
           "",
           "Averaging the three seeds' predictions is free accuracy — no extra training,",
           "just three numbers averaged per clip — and it almost always beats any single",
           "seed. Reads straight from the saved prediction files, so this cell also works",
           "on its own after a session restart."),
        code("import numpy as np",
             "from paths import model_dir",
             "",
             "def _load(tag=''):",
             "    p = model_dir(MODEL) / f\"{MODEL}_calfree{'_' + tag if tag else ''}_predictions.npz\"",
             "    d = np.load(p)",
             "    return d['y_true'], d['y_pred']",
             "",
             "y_true, p0 = _load()",
             "_, p1 = _load('seed1')",
             "_, p2 = _load('seed2')",
             "ensemble = (p0 + p1 + p2) / 3",
             "for t, name in enumerate(('sbp', 'dbp')):",
             "    single = np.abs(p0[:, t] - y_true[:, t]).mean()",
             "    ens = np.abs(ensemble[:, t] - y_true[:, t]).mean()",
             "    print(f'{name.upper()}  seed0 alone {single:.2f}  ->  3-seed ensemble {ens:.2f}')"),
        md("## Results"),
        code("import pandas as pd",
             "",
             "rows = []",
             "runs = {'calfree': calfree, 'calfree (no balancing)': calfree_nobalance,",
             "        **others}",
             "for proto, out in runs.items():",
             "    for target in ('sbp', 'dbp'):",
             "        m = out['results'][target]",
             "        rows.append(dict(protocol=proto, target=target, mae=m['mae'],",
             "                         sde=m['sde'], r=m['r'],",
             "                         aami='PASS' if m['aami_pass'] else 'FAIL',",
             "                         bhs=m['bhs']))",
             "pd.DataFrame(rows).round(2)"),
        result_cell(),
    ]
    write(f"{num}_{model}_colab.ipynb", cells)


# --------------------------------------------------------------------------- #
# 04 — personalization
# --------------------------------------------------------------------------- #

def personalization():
    cells = [
        md("# 04 — Few-shot personalization",
           "",
           "Spec section J: *\"a lightweight few-shot personalization layer — e.g. a",
           "small per-subject fine-tuning of the final regression layers (or a",
           "feature-space affine recalibration) using only a handful of labeled",
           "calibration segments per new subject\"*.",
           "",
           "The question it answers is a practical one: **how many cuff readings",
           "does a new user have to provide before the model works for them?** The",
           "answer is a curve over k = 0, 1, 3, 5, 10, 25, where k = 0 is the",
           "uncalibrated number, so any drop is readable straight off it.",
           "",
           "Spec section L: *\"for a held-out subject, use only the first small % of",
           "that subject's segments as calibration data and evaluate on the",
           "remainder\"*. That is `source='calfree'` — 144 unseen patients, 400 clips",
           "each, the first k used to calibrate and the remaining ~395 scored.",
           "Calibration clips are dropped from the metric."),
        *colab_setup(),
        md("## First: is there anything for calibration to fix?",
           "",
           "If the error were random noise, no amount of calibration could remove",
           "it. If it is a per-subject offset, one reading is enough in principle.",
           "",
           "`offset_headroom` measures which it is. It hands every subject its own",
           "best possible constant offset — fitted on that subject's own test clips,",
           "so it is a ceiling nobody can reach — and reports how far the error",
           "falls. That gives the scale to read the real gain against."),
        code("from train import personalize",
             "",
             "MODEL = 'gb'   # 'gb' | 'rf' | 'cnn' | 'resnet' | 'transformer'",
             "",
             "# A deep model is trained from scratch here (hours), because the",
             "# calibration clips and the test clips have to be scored by the *same*",
             "# model — calibrating one model and testing another measures nothing.",
             "cal = personalize(model=MODEL, source='calfree')"),
        md("## The curve",
           "",
           "Three ways of choosing which clips a subject calibrates on, reported side",
           "by side rather than one being asserted: `first` (the spec's literal",
           "reading, and what a device could actually collect), `random`, and",
           "`spread` (readings covering that person's own BP range)."),
        code("import pandas as pd",
             "",
             "rows = []",
             "for pick, curve in cal['curves'].items():",
             "    for row in curve:",
             "        for target in ('sbp', 'dbp'):",
             "            m = row['results'][target]",
             "            rows.append(dict(pick=pick, shots=row['shots'], target=target,",
             "                             n=row['n_scored'], mae=m['mae'], sde=m['sde'],",
             "                             aami='PASS' if m['aami_pass'] else 'FAIL'))",
             "df = pd.DataFrame(rows)",
             "display(df[df.target == 'sbp'].pivot(index='shots', columns='pick',",
             "                                     values='mae').round(2))",
             "df.round(2)"),
        code("import matplotlib.pyplot as plt",
             "",
             "fig, ax = plt.subplots(1, 2, figsize=(11, 4))",
             "for i, target in enumerate(('sbp', 'dbp')):",
             "    d = df[df.target == target]",
             "    for pick in d['pick'].unique():",
             "        s = d[d['pick'] == pick]",
             "        ax[i].plot(s.shots, s.mae, 'o-', label=pick)",
             "    ax[i].axhline(5, ls='--', c='#c0392b', label='clinical standard')",
             "    ax[i].set_title(target.upper())",
             "    ax[i].set_xlabel('calibration readings per subject (k)')",
             "    ax[i].set_ylabel('MAE (mmHg)')",
             "    ax[i].grid(alpha=.3)",
             "    ax[i].legend()",
             "plt.tight_layout()"),
        md("## Secondary: PulseDB's own AAMI subsets",
           "",
           "This gives a different answer, and the reason is in the data rather than",
           "in the method. `aami_test` is built to span the full BP range on purpose,",
           "so its clips average 134.9 mmHg SBP while the first five `aami_cal` clips",
           "of the same patients average 119.7. A first-k calibration therefore bakes",
           "that -15 mmHg gap into every prediction and makes the error worse.",
           "",
           "Worth reporting: where the calibration readings come from is itself a",
           "design decision, not a detail."),
        code("aami = personalize(model=MODEL, source='aami')"),
        result_cell(),
    ]
    write("04_personalization.ipynb", cells)


if __name__ == "__main__":
    baselines()
    for m in DEEP_NB:
        deep(m)
    personalization()
