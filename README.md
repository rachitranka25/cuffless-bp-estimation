# Project B — Cuffless Blood Pressure Estimation

TEEP research project. Spec: [`docs/TEEP_Research_Project_Proposals.md.pdf`](docs/) —
**Project B only** (Project A, the PTB-XL ECG work, belongs to the other student).

**Question.** How much does cuffless PPG-based BP estimation degrade under a rigorous
subject-independent, cross-setting protocol compared to the leakage-prone evaluation
commonly reported — and how much of that gap can a lightweight few-shot personalization
step recover?

**Results live in [`results/overview/RESULTS.md`](results/overview/RESULTS.md)**, generated
from the run files so it cannot drift from the numbers.

**Picking this up cold, or on another machine? Start with
[`HANDOFF.md`](HANDOFF.md)** — state of play, the findings worth defending, the
Windows-specific fixes, and what comes next.

## Layout

```
data/
  raw/            downloaded as-is, read-only              20 GB
  processed/      model-ready arrays                      3.6 GB
  features/       61 hand-crafted numbers per clip        174 MB
  splits/         manifest.json — who is in which split

src/              importable modules — all pipeline code lives here
scripts/          runnable jobs
notebooks/        launchers; they call src/, they do not contain pipeline code
docs/             the spec, and the decks written for the professor

results/
  dataset/        what the data looks like — true before any model existed
    pulsedb/          segments, quality, splits, labels, arrays
    ppg_dalia/        .npz contents, sessions, subjects, activity, motion
    both_datasets/    the two side by side, domain shift
    preprocessing/    evidence for every preprocessing claim
    literature/       published results on the same data
    plan/             which data feeds which model, and the four test rules
  models/         one folder per model — self-contained
    <model>/
      <model>_<protocol>.json              metrics
      <model>_<protocol>_predictions.npz   per-clip predictions
      figures/                             figures about this model alone
      checkpoints/                         trained weights (deep models)
  overview/       RESULTS.md and the cross-model figures
```

`raw/` is separate on purpose: preprocessing *will* change during the project, and when it
does you delete `processed/` and rebuild rather than re-downloading 20 GB.

Never hardcode paths — import them:

```python
import sys; sys.path.insert(0, "src")
from paths import PROC_PULSEDB, FEATURES, FIG_PULSEDB, model_dir
```

On Colab the same layout is mounted from Drive; set `BP_ROOT` before importing and every
path follows.

## The data

| | PulseDB | PPG-DaLiA |
|---|---|---|
| Setting | ICU / operating room | Free living |
| Sensor | Finger PPG + ECG | Wrist PPG (Empatica E4) + chest ECG |
| Subjects | 1,553 | 15 |
| Amount | 645,678 × 10 s clips | 64,682 × 10 s clips (36 hours) |
| BP labels | **Yes** — SBP/DBP per clip | **No** |
| Role | Train + honest test | External domain-shift test |
| Source | [Kaggle mirror](https://www.kaggle.com/datasets/weinanwangrutgers/pulsedb-balanced-training-and-testing) of [PulseDB](https://github.com/pulselabteam/PulseDB), CC BY-NC-SA 4.0 | [UCI](https://archive.ics.uci.edu/dataset/495/ppg+dalia) |

Per the spec (§M, §N month 2) this is the **VitalDB portion** of PulseDB; the full
MIMIC-III + VitalDB release (~380 GB plus MATLAB) is explicitly not required.

Both datasets are stored in the same shape — `(N, 2, 1250)` at 125 Hz — so a model trained
on PulseDB runs on PPG-DaLiA without a line of code changing. That is the point of the
external test.

### PulseDB splits

| Split | Clips | Subjects | Role |
|---|---|---|---|
| train | 465,480 | 1,293 | The model learns from this |
| calfree_test | 57,600 | 144 | **Headline** — subjects never seen |
| aami_cal | 70,212 | 116 | Calibration clips for personalization |
| aami_test | 666 | 116 | Clinical AAMI scoring, same 116 unseen subjects |
| calbased_test | 51,720 | 1,293 | Known patients — an upper bound, never a headline |

Verified on every run: `train ∩ calfree_test = 0`, `train ∩ aami_cal = 0`,
`train ∩ aami_test = 0`. `aami_cal ∩ aami_test = 116` and `train ∩ calbased_test = 1,293`
are intentional and recorded in `data/splits/manifest.json`.

`aami_test` is deliberately harder: it is built to span the full BP range, so its clips
average 134.9 mmHg SBP while the same patients' first `aami_cal` clips average 119.7 — a
gap that matters when choosing where calibration readings come from.

### Processed arrays

```
data/processed/pulsedb/<split>/
  signals.npy    (N, 2, 1250) float16   channel 0 = ECG, 1 = PPG, 125 Hz
  labels.npy     (N, 2) float32         SBP, DBP
  subjects.npy   (N,)                   for leakage assertions
  meta.parquet                          age, sex, height, weight, BMI, quality

data/processed/ppg_dalia/windows/       (64682, 2, 1250) — same shape, no BP labels
data/processed/ppg_dalia/S1..S15.npz    per-subject full recordings
data/features/<split>.parquet           61 features + subject + sbp + dbp
```

ABP is never written — it is the label source, and keeping it beside the inputs would hand
the model the answer. Storage is float16 because the channels are min-max scaled to [0, 1],
where its ~2e-4 precision sits below the sensor noise floor. Cast to float32 on load.

```python
X = np.load(PROC_PULSEDB/"train"/"signals.npy", mmap_mode="r")   # 2.3 GB, 0 bytes of RAM
y = np.load(PROC_PULSEDB/"train"/"labels.npy")
```

## Training

Everything goes through one file, `src/train.py`, so that a difference between two rows of
the results table is a difference in the model or the test rule and nothing else.

```python
from train import run, personalize

run(model="gb",  protocol="calfree")              # five models: rf gb cnn resnet transformer
run(model="cnn", protocol="calfree", epochs=20)   # four protocols: leaky calbased calfree aami
personalize(model="gb", source="calfree")         # the few-shot curve
```

```bash
python3 -m train --model gb --protocol calfree    # same thing from the shell
```

Built in, because the spec asks for them and they are easy to forget once training starts:

- **BP-bin balancing** (spec §L, *"stratified sampling or a weighted loss over BP value
  bins"*). 55% of the training clips sit between 90 and 120 mmHg and 1.8% above 160; left
  alone a model learns to answer "about 110" and is worst for the patients who matter most.
- **Subject-disjointness assertions** on every honest protocol. A silent leak would inflate
  exactly the number this project exists to measure.
- **No quality filtering of test sets.** Quality is scored and recorded, never used to drop
  clips — filtering the hard ones out of a test set inflates the result.

## Running things

```bash
# data — once
bash scripts/download_pulsedb.sh            # ~19 GB, no credentials needed
python3 scripts/distill_dalia.py            # 20 GB of .pkl -> 259 MB of .npz
jupyter nbconvert --execute --inplace notebooks/01_preprocessing.ipynb
python3 scripts/extract_features.py         # PulseDB -> data/features/
python3 scripts/extract_features_dalia.py   # PPG-DaLiA -> data/features/dalia.parquet

# training
python3 -m train --model gb --protocol calfree      # run from src/
jupyter notebook notebooks/02_baselines.ipynb       # rf + gb, all four protocols

# Colab, for the deep models
python3 scripts/make_drive_folder.py        # assembles bp/ to drag into Drive
python3 scripts/build_notebooks.py          # regenerates the launcher notebooks

# reporting
python3 scripts/plot_results.py             # figures, from whatever has been run
python3 scripts/make_results_report.py      # results/overview/RESULTS.md

# dataset figures and decks
python3 scripts/visualize_processed.py
python3 scripts/visualize_dalia.py
python3 scripts/plot_preprocessing_evidence.py
python3 scripts/plot_literature.py
python3 scripts/plot_training_plan.py
python3 scripts/build_report_deck.py
python3 scripts/build_training_plan_deck.py
```

## Status

Complete. All five models (RF, GB, 1D-CNN, ResNet1D, Transformer) are trained and scored
under all four protocols, three seeds each where noted; the leakage gap and the PPG-DaLiA
domain-shift control are measured; few-shot personalization is evaluated under both a
naive (storage-order) and a genuinely chronological calibration/evaluation split; and the
manuscript is written in two forms:

- [`docs/Journal_Paper_IEEE.pdf`](docs/Journal_Paper_IEEE.pdf) — IEEE two-column format,
  built from [`ieee_paper/paper.tex`](ieee_paper/paper.tex).
- [`docs/Journal_Paper_ISGJ.pdf`](docs/Journal_Paper_ISGJ.pdf) — Gerontechnology journal
  format, built from [`scripts/build_journal_paper.py`](scripts/build_journal_paper.py).

Current numbers, and how to reproduce each one, are in
[`results/overview/RESULTS.md`](results/overview/RESULTS.md).

### Where each run actually executed

Classical models and early deep-model iterations ran locally on this machine and in
[`bp/`](bp/) (a self-contained copy deployed to Google Colab — see
[`bp/README.txt`](bp/README.txt) and [`bp/HANDOFF.md`](bp/HANDOFF.md)). The bulk of the
50-epoch, three-seed CNN/ResNet1D/Random Forest/Gradient Boosting sweep ran on a Kaggle
account; the Transformer's 19-job sweep ran on a separate workstation, following
[`HANDOFF.md`](HANDOFF.md)'s brief. In every case the code executed was the same
`src/train.py` entry point committed here — only the orchestration notebooks used on
Kaggle itself were not preserved. `results/models/<model>/` holds the output of every one
of those runs (metrics, per-clip predictions, and checkpoints for the deep models), which
is what the manuscript's numbers are computed from.
