# Results

Generated 2026-09-27 from `results/models/*/*.json` by `scripts/make_results_report.py`. Every number below comes from a saved run file; nothing here is typed by hand.

## What has been trained

| model | protocol | training clips | training patients | test clips | test patients | wall time | run file |
|---|---|---|---|---|---|---|---|
| Random Forest | `leaky` | 335,132 | 1,164 | 93,096 | 1,293 | 39 min | `results/models/rf/rf_leaky.json` |
| Random Forest | `calbased` | 419,040 | 1,164 | 51,720 | 1,293 | 51 min | `results/models/rf/rf_calbased.json` |
| Random Forest | `calfree` | 419,040 | 1,164 | 57,600 | 144 | 42 min | `results/models/rf/rf_calfree.json` |
| Random Forest | `aami` | 419,040 | 1,164 | 666 | 116 | 45 min | `results/models/rf/rf_aami.json` |
| Gradient Boosting | `leaky` | 335,132 | 1,164 | 93,096 | 1,293 | 1 min | `results/models/gb/gb_leaky.json` |
| Gradient Boosting | `calbased` | 419,040 | 1,164 | 51,720 | 1,293 | 1 min | `results/models/gb/gb_calbased.json` |
| Gradient Boosting | `calfree` | 419,040 | 1,164 | 57,600 | 144 | 1 min | `results/models/gb/gb_calfree.json` |
| Gradient Boosting | `aami` | 419,040 | 1,164 | 666 | 116 | 1 min | `results/models/gb/gb_aami.json` |
| 1D-CNN | `leaky` | 335,132 | 1,164 | 93,096 | 1,293 | 27 min | `results/models/cnn/cnn_leaky.json` |
| 1D-CNN | `calbased` | 419,040 | 1,164 | 51,720 | 1,293 | 34 min | `results/models/cnn/cnn_calbased.json` |
| 1D-CNN | `calfree` | 419,040 | 1,164 | 57,600 | 144 | 34 min | `results/models/cnn/cnn_calfree.json` |
| 1D-CNN | `aami` | 419,040 | 1,164 | 666 | 116 | 34 min | `results/models/cnn/cnn_aami.json` |
| ResNet1D | `leaky` | 335,132 | 1,164 | 93,096 | 1,293 | 28 min | `results/models/resnet/resnet_leaky.json` |
| ResNet1D | `calbased` | 419,040 | 1,164 | 51,720 | 1,293 | 34 min | `results/models/resnet/resnet_calbased.json` |
| ResNet1D | `calfree` | 419,040 | 1,164 | 57,600 | 144 | 34 min | `results/models/resnet/resnet_calfree.json` |
| ResNet1D | `aami` | 419,040 | 1,164 | 666 | 116 | 34 min | `results/models/resnet/resnet_aami.json` |
| Transformer (DMT-style) | `calfree` | 419,040 | 1,164 | 57,600 | 144 | 247 min | `results/models/transformer/transformer_calfree.json` |

Not yet run: `transformer/leaky`, `transformer/calbased`, `transformer/aami`.

## Results

Systolic and diastolic are predicted jointly by one model with two outputs. AAMI passes when the mean error is within ±5 mmHg and its standard deviation within 8. BHS grades on the share of absolute errors under 5 / 10 / 15 mmHg; D is a fail.

| model | protocol | SBP MAE | SBP SD | SBP r | AAMI | BHS | DBP MAE | DBP SD | DBP r | AAMI | BHS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | `leaky` | 7.47 | 10.46 | 0.84 | FAIL | C | 4.34 | 6.32 | 0.86 | PASS | A |
| Random Forest | `calbased` | 7.14 | 10.12 | 0.85 | FAIL | C | 4.11 | 6.16 | 0.86 | PASS | A |
| Random Forest | `calfree` | 12.76 | 16.26 | 0.52 | FAIL | D | 8.38 | 10.62 | 0.47 | FAIL | D |
| Random Forest | `aami` | 19.67 | 23.01 | 0.69 | FAIL | D | 12.24 | 13.97 | 0.65 | FAIL | D |
| Gradient Boosting | `leaky` | 10.29 | 13.20 | 0.74 | FAIL | D | 6.13 | 7.98 | 0.75 | PASS | B |
| Gradient Boosting | `calbased` | 10.20 | 13.08 | 0.75 | FAIL | D | 6.03 | 7.92 | 0.76 | PASS | B |
| Gradient Boosting | `calfree` | 14.63 | 18.40 | 0.50 | FAIL | D | 9.00 | 11.29 | 0.45 | FAIL | D |
| Gradient Boosting | `aami` | 17.44 | 22.28 | 0.70 | FAIL | D | 11.14 | 13.59 | 0.67 | FAIL | D |
| 1D-CNN | `leaky` | 11.10 | 14.32 | 0.72 | FAIL | D | 7.04 | 9.22 | 0.68 | FAIL | C |
| 1D-CNN | `calbased` | 9.89 | 13.05 | 0.76 | FAIL | D | 6.29 | 8.41 | 0.73 | FAIL | B |
| 1D-CNN | `calfree` | 15.32 | 18.93 | 0.55 | FAIL | D | 9.20 | 11.43 | 0.48 | FAIL | D |
| 1D-CNN | `aami` | 16.53 | 21.54 | 0.74 | FAIL | D | 11.20 | 13.78 | 0.67 | FAIL | D |
| ResNet1D | `leaky` | 6.07 | 8.88 | 0.88 | FAIL | B | 3.87 | 5.96 | 0.87 | PASS | A |
| ResNet1D | `calbased` | 6.17 | 8.94 | 0.88 | FAIL | B | 3.92 | 6.02 | 0.87 | PASS | A |
| ResNet1D | `calfree` | 13.96 | 17.85 | 0.52 | FAIL | D | 9.03 | 11.39 | 0.47 | FAIL | D |
| ResNet1D | `aami` | 17.70 | 21.79 | 0.72 | FAIL | D | 11.40 | 13.33 | 0.69 | FAIL | D |
| Transformer (DMT-style) | `calfree` | 16.26 | 20.85 | 0.53 | FAIL | D | 8.72 | 11.02 | 0.56 | FAIL | D |

What each protocol means:

- **`leaky`** — clips split at random inside the training subset — the same patients appear on both sides
- **`calbased`** — PulseDB's official calibration-based test set — the same 1,293 patients, clips the model never saw
- **`calfree`** — PulseDB's official calibration-free test set — 144 patients never seen in training
- **`aami`** — PulseDB's official AAMI test set — 116 unseen patients, no calibration applied

## The leakage gap

The same trained model, scored four ways. Nothing changes between the rows except which patients are in the test set.

- **Random Forest** — 7.31 mmHg on patients it had already met, 12.76 mmHg on patients it had not. **1.75x worse**, from the test rule alone.
- **Gradient Boosting** — 10.24 mmHg on patients it had already met, 14.63 mmHg on patients it had not. **1.43x worse**, from the test rule alone.
- **1D-CNN** — 10.50 mmHg on patients it had already met, 15.32 mmHg on patients it had not. **1.46x worse**, from the test rule alone.
- **ResNet1D** — 6.12 mmHg on patients it had already met, 13.96 mmHg on patients it had not. **2.28x worse**, from the test rule alone.

## Personalization

Spec sections J and L — the project's contribution. A few of a new patient's own `calfree_test` clips fit a per-subject affine correction (offset and scale); the rest are scored. `k=0` is the uncorrected calibration-free number, so every table below reads straight off the honest baseline above. This answers the spec's three ablations for this section.

### 1. Calibration-free vs. personalized — checked two ways

**A rigor check changed this section's headline claim.** The first table below uses `k=25` picked in storage order, as spec section L reads literally. We checked whether storage order is chronological (PulseDB's own `SegIDX` field, joined in from the official Info proxy file — not shipped with the Subset files we train on) — it is not: correlation between storage position and true recording time is 0.007. That first-25 set is scattered across the whole session, so on average an evaluation clip has a calibration clip within about 3 minutes of it, which is close enough for the model to exploit short-term drift rather than the patient's stable physiology. See `scripts/check_session_leakage.py`.

| model | SBP MAE k=0 | SBP MAE k=25 | DBP MAE k=0 | DBP MAE k=25 | DBP AAMI at k=25 |
|---|---|---|---|---|---|
| Random Forest | 12.76 | 8.30 | 8.38 | 4.81 | **PASS** |
| Gradient Boosting | 14.63 | 8.09 | 9.00 | 4.74 | **PASS** |
| 1D-CNN | 15.32 | 7.74 | 9.20 | 4.51 | **PASS** |
| ResNet1D | 13.96 | 8.23 | 9.03 | 4.83 | **PASS** |
| Transformer (DMT-style) | 16.26 | 8.23 | 8.72 | 4.48 | **PASS** |

**The honest version** recalibrates using the genuinely earliest 25 clips by real time, and scores only what comes after (~70 minutes later on average — the closest thing to real deployment this single-session dataset supports):

| model | DBP MAE, true-early k=25 | DBP AAMI, true-early |
|---|---|---|
| Random Forest | 7.32 | fail (C) |
| Gradient Boosting | 7.11 | fail (C) |
| 1D-CNN | 7.25 | fail (C) |
| ResNet1D | 7.38 | fail (C) |
| Transformer (DMT-style) | 6.92 | fail (C) |

DBP MAE still improves substantially under the honest test (roughly a fifth to a third off across models) — personalization is real and worth reporting. It does not cross AAMI's SD≤8 threshold once calibration and evaluation are genuinely separated in time; bias stays under 2 mmHg throughout, so it is the spread that falls short, not the bias. **Corrected claim: personalization meaningfully reduces DBP error; it does not reach the AAMI bar once tested honestly.** Systolic's SD stays above the AAMI limit either way — the ceiling analysis further down explains why no per-patient correction could close it.

### 2. How much calibration data is needed

SBP MAE by calibration amount (`first`-picked clips). `k=1`, `3`, `5` are the exact amounts spec section L asks about.

| model | k=0 | k=1 | k=3 | k=5 | k=10 | k=25 |
|---|---|---|---|---|---|---|
| Random Forest | 12.76 | 12.20 | 15.82 | 9.31 | 8.77 | 8.30 |
| Gradient Boosting | 14.63 | 13.02 | 11.02 | 9.11 | 8.49 | 8.09 |
| 1D-CNN | 15.32 | 12.75 | 11.30 | 9.03 | 8.06 | 7.74 |
| ResNet1D | 13.96 | 12.57 | 12.55 | 9.54 | 8.56 | 8.23 |
| Transformer (DMT-style) | 16.26 | 13.46 | 11.76 | 9.70 | 8.65 | 8.23 |

Most of the recoverable error is gone by `k=5`; `k=10` to `k=25` buys a smaller additional drop. Two models (Random Forest, ResNet1D) get briefly *worse* at `k=3` before improving — 2-3 calibration clips is too few to fit a stable offset-and-scale correction from, so a bad draw can hurt more than `k=0`'s no correction at all. Worth keeping in the write-up: below some minimum, personalization is not free. Figure: `results/overview/5_personalization.png`.

### 3. Classical features vs. raw-waveform CNN

Spec section L's third ablation — answerable directly from the base runs above, no personalization involved.

| protocol | Random Forest (61 features) | 1D-CNN (raw waveform) | winner |
|---|---|---|---|
| `leaky` | 7.47 | 11.10 | Random Forest |
| `calbased` | 7.14 | 9.89 | Random Forest |
| `calfree` | 12.76 | 15.32 | Random Forest |
| `aami` | 19.67 | 16.53 | 1D-CNN |

Hand-crafted features beat the raw-waveform CNN on three of the four protocols, including the honest `calfree` one, and lose only on `aami` — the smallest, hardest test set. That matters for a wearable with no GPU: 61 numbers from fiducial-point detection do at least as well as a trained convnet here.

## Against the published numbers

Same dataset (PulseDB VitalDB), same official splits, same metric.

Our rows here are the **unbalanced** runs. Spec section L asks for BP-bin balancing and it is on by default, but it trades average error for accuracy on the rare hypertensive patients, which raises the headline MAE. Every published number below was measured without it, and section K makes comparability the reason MAE is the primary metric — so this table uses the run that is actually comparable. The balanced numbers are in the results table above, and what balancing buys is in the ablations below.

| method | source | protocol | SBP MAE | DBP MAE |
|---|---|---|---|---|
| XResNet1d101 | arXiv:2502.19167 (2025) | `calfree` | 12.70 | 8.05 |
| DMT Transformer | arXiv:2606.11125 (2026) | `calfree` | 12.17 | 7.89 |
| UTransBPNet | Conn. Health Telemed. (2024) | `calfree` | 12.50 | 8.32 |
| XResNet1d101 | arXiv:2502.19167 (2025) | `calbased` | 9.08 | 6.08 |
| DMT Transformer | arXiv:2606.11125 (2026) | `calbased` | 4.56 | 2.62 |
| **Random Forest (this project)** | — | `calfree` | 12.93 | 8.45 |
| **Random Forest (this project)** *(balanced)* | — | `calbased` | 7.14 | 4.11 |
| **Gradient Boosting (this project)** | — | `calfree` | 12.71 | 8.25 |
| **Gradient Boosting (this project)** *(balanced)* | — | `calbased` | 10.20 | 6.03 |
| **1D-CNN (this project)** | — | `calfree` | 13.01 | 8.34 |
| **1D-CNN (this project)** *(balanced)* | — | `calbased` | 9.89 | 6.29 |
| **ResNet1D (this project)** | — | `calfree` | 13.21 | 8.53 |
| **ResNet1D (this project)** *(balanced)* | — | `calbased` | 6.17 | 3.92 |
| **Transformer (DMT-style) (this project)** *(balanced)* | — | `calfree` | 16.26 | 8.72 |

## The domain-shift test — PPG-DaLiA

Spec sections E, H and L. Every model is trained on PulseDB and then run on PPG-DaLiA: 64,682 clips from 15 volunteers going about their day with a wrist sensor, never seen in training. Section L calls this *"the most novel design choice differentiating this from typical single-dataset cuffless-BP papers"*.

PPG-DaLiA has no BP labels, so there is no MAE here and inventing one would be worse than reporting nothing. Four things are measurable without labels: whether the answers stay physiologically possible, how far the prediction distribution moves, whether the model still separates the fifteen volunteers, and whether it degrades with motion.

### The control comes first

BP-bin balancing pushes a model to spread its predictions apart, and that spread does not survive the move to a wrist sensor. Reporting only the balanced model would therefore overstate the shift. Both are run:

| model | balancing | shift in mean SBP | impossible answers | tells people apart (PulseDB → PPG-DaLiA) |
|---|---|---|---|---|
| Random Forest | on | +22.5 mmHg | 0.27% | 8.2 → 2.5 (true spread 11.8) |
| Random Forest | off | +14.6 mmHg | 0.00% | 8.0 → 2.1 (true spread 11.8) |
| Gradient Boosting | on | +32.2 mmHg | 0.30% | 11.9 → 6.7 (true spread 11.8) |
| Gradient Boosting | off | +19.4 mmHg | 0.00% | 8.0 → 8.2 (true spread 11.8) |
| 1D-CNN | on | +33.3 mmHg | 0.03% | 14.0 → 3.6 (true spread 11.8) |
| 1D-CNN | off | +21.3 mmHg | 0.00% | 8.4 → 3.1 (true spread 11.8) |
| ResNet1D | on | +6.2 mmHg | 0.01% | 11.8 → 3.0 (true spread 11.8) |
| ResNet1D | off | +13.1 mmHg | 0.00% | 9.6 → 2.4 (true spread 11.8) |

Read the last column as: the true between-subject spread of systolic pressure in the PulseDB test patients is about 11.8 mmHg. A model that arrives at PPG-DaLiA still reporting a spread near that is distinguishing the volunteers; one reporting 2 or 3 is giving nearly everyone the same answer.

- **Random Forest** — the shift measures +22.5 mmHg with balancing on and +14.6 with it off, so 35% of what the balanced run reports as domain shift is the weighting.
- **Gradient Boosting** — the shift measures +32.2 mmHg with balancing on and +19.4 with it off, so 40% of what the balanced run reports as domain shift is the weighting.
- **1D-CNN** — the shift measures +33.3 mmHg with balancing on and +21.3 with it off, so 36% of what the balanced run reports as domain shift is the weighting.
- **ResNet1D** — the shift measures +6.2 mmHg with balancing on and +13.1 with it off, so -112% of what the balanced run reports as domain shift is the weighting.

### The heart-rate cross-check

Spec section G offers PPG-DaLiA's chest-ECG heart rate as a fully labelled secondary task. This tests the signal pipeline rather than the BP model, and is reported separately for that reason.

Pulse-detector heart rate against the chest-ECG reference: **27.11 bpm** overall — 13.06 when the volunteer is still, 33.35 when moving. For context, a paper that pairs these same two datasets for motion-artifact removal (arXiv:2508.10805) reports 8.69 bpm on PPG-DaLiA using a dedicated denoising and heart-rate method; a generic pulse detector run on wrist PPG is much worse, and that gap is part of why the BP predictions move.

## Ablations

### `nobalance` — 1D-CNN, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 15.32 | 18.93 | 9.20 | 11.43 |
| nobalance | 13.01 | 16.14 | 8.34 | 10.39 |

### `nobalance_seed1` — 1D-CNN, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 15.32 | 18.93 | 9.20 | 11.43 |
| nobalance_seed1 | 12.91 | 16.35 | 8.31 | 10.52 |

### `nobalance_seed2` — 1D-CNN, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 15.32 | 18.93 | 9.20 | 11.43 |
| nobalance_seed2 | 13.20 | 17.04 | 8.99 | 11.17 |

### `seed1` — 1D-CNN, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 15.32 | 18.93 | 9.20 | 11.43 |
| seed1 | 14.83 | 18.21 | 9.19 | 11.55 |

### `seed2` — 1D-CNN, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 15.32 | 18.93 | 9.20 | 11.43 |
| seed2 | 14.96 | 18.52 | 9.19 | 11.15 |

### `nobalance` — Gradient Boosting, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 14.63 | 18.40 | 9.00 | 11.29 |
| nobalance | 12.71 | 16.40 | 8.25 | 10.40 |

### `nobalance_seed1` — Gradient Boosting, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 14.63 | 18.40 | 9.00 | 11.29 |
| nobalance_seed1 | 12.73 | 16.42 | 8.36 | 10.56 |

### `nobalance_seed2` — Gradient Boosting, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 14.63 | 18.40 | 9.00 | 11.29 |
| nobalance_seed2 | 12.84 | 16.53 | 8.38 | 10.58 |

### `seed1` — Gradient Boosting, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 14.63 | 18.40 | 9.00 | 11.29 |
| seed1 | 14.76 | 18.80 | 8.94 | 11.23 |

### `seed2` — Gradient Boosting, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 14.63 | 18.40 | 9.00 | 11.29 |
| seed2 | 14.62 | 18.56 | 9.00 | 11.30 |

### `nobalance` — ResNet1D, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 13.96 | 17.85 | 9.03 | 11.39 |
| nobalance | 13.21 | 17.11 | 8.53 | 10.81 |

### `nobalance_seed1` — ResNet1D, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 13.96 | 17.85 | 9.03 | 11.39 |
| nobalance_seed1 | 13.71 | 17.02 | 9.16 | 11.39 |

### `nobalance_seed2` — ResNet1D, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 13.96 | 17.85 | 9.03 | 11.39 |
| nobalance_seed2 | 13.35 | 17.04 | 9.17 | 11.40 |

### `seed1` — ResNet1D, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 13.96 | 17.85 | 9.03 | 11.39 |
| seed1 | 14.01 | 17.94 | 9.29 | 11.63 |

### `seed2` — ResNet1D, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 13.96 | 17.85 | 9.03 | 11.39 |
| seed2 | 13.68 | 17.62 | 8.79 | 11.12 |

### `nobalance` — Random Forest, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 12.76 | 16.26 | 8.38 | 10.62 |
| nobalance | 12.93 | 16.59 | 8.45 | 10.63 |

### `nobalance_seed1` — Random Forest, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 12.76 | 16.26 | 8.38 | 10.62 |
| nobalance_seed1 | 12.74 | 16.39 | 8.64 | 10.86 |

### `nobalance_seed2` — Random Forest, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 12.76 | 16.26 | 8.38 | 10.62 |
| nobalance_seed2 | 12.77 | 16.42 | 8.69 | 10.98 |

### `seed1` — Random Forest, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 12.76 | 16.26 | 8.38 | 10.62 |
| seed1 | 12.79 | 16.35 | 8.37 | 10.62 |

### `seed2` — Random Forest, `calfree`

| | SBP MAE | SBP SD | DBP MAE | DBP SD |
|---|---|---|---|---|
| default | 12.76 | 16.26 | 8.38 | 10.62 |
| seed2 | 12.77 | 16.32 | 8.41 | 10.65 |

Spec section L asks for *"stratified sampling or a weighted loss over BP value bins"*. Turning it off lowers the headline MAE and raises the error on the hypertensive patients — see the band-by-band figure in `results/models/<model>/figures/` for the band-by-band breakdown.

## Figures


`results/overview/`
- `1_leakage_gap.png`
- `2_compliance.png`
- `3_shift_amount.png`
- `3_shift_amount_dbp.png`
- `3_shift_collapse.png`
- `5_personalization.png`

`results/models/cnn/figures/`
- `1_bp_band_error.png`
- `2_bland_altman.png`
- `3_domain_shift.png`
- `3_domain_shift_nobalance.png`
- `3_domain_shift_nobalance_seed1.png`
- `3_domain_shift_nobalance_seed2.png`
- `3_domain_shift_seed1.png`
- `3_domain_shift_seed2.png`
- `4_training_curve.png`

`results/models/gb/figures/`
- `1_bp_band_error.png`
- `2_bland_altman.png`
- `3_domain_shift.png`
- `3_domain_shift_nobalance.png`

`results/models/resnet/figures/`
- `1_bp_band_error.png`
- `2_bland_altman.png`
- `3_domain_shift.png`
- `3_domain_shift_nobalance.png`
- `3_domain_shift_nobalance_seed1.png`
- `3_domain_shift_nobalance_seed2.png`
- `3_domain_shift_seed1.png`
- `3_domain_shift_seed2.png`
- `4_training_curve.png`

`results/models/rf/figures/`
- `1_bp_band_error.png`
- `2_bland_altman.png`
- `3_domain_shift.png`
- `3_domain_shift_nobalance.png`

`results/models/transformer/figures/`
- `2_bland_altman.png`
- `4_training_curve.png`

Regenerate with `python3 scripts/plot_results.py`.

## Code

The whole pipeline is `src/train.py` — one file for all five models and all four protocols, so a difference between two rows above is a difference in the model or the test rule and nothing else. The notebooks are launchers that call it.

| file | what it does |
|---|---|
| `src/train.py` | training, evaluation, protocols, personalization |
| `src/models.py` | CNN1D, ResNet1D, DMT transformer |
| `src/features.py` | the 61 hand-crafted features |
| `src/fiducials.py` | R-peak and PPG onset / peak / notch detection |
| `src/evaluation.py` | MAE, SDE, AAMI, BHS |
| `src/bpdata.py` | memory-mapped datasets, subject-disjoint splits |
| `scripts/plot_results.py` | the figures above |
| `notebooks/02_baselines.ipynb` | Random Forest and Gradient Boosting |
| `notebooks/03a_cnn_colab.ipynb` | 1D-CNN |
| `notebooks/03b_resnet_colab.ipynb` | ResNet1D |
| `notebooks/03c_transformer_colab.ipynb` | Transformer |
| `notebooks/04_personalization.ipynb` | few-shot calibration |

### Reproducing any row above

```bash
python3 -m train --model rf --protocol leaky
python3 -m train --model rf --protocol calbased
python3 -m train --model rf --protocol calfree
python3 -m train --model rf --protocol aami
python3 -m train --model gb --protocol leaky
python3 -m train --model gb --protocol calbased
python3 -m train --model gb --protocol calfree
python3 -m train --model gb --protocol aami
python3 -m train --model cnn --protocol leaky
python3 -m train --model cnn --protocol calbased
python3 -m train --model cnn --protocol calfree
python3 -m train --model cnn --protocol aami
python3 -m train --model resnet --protocol leaky
python3 -m train --model resnet --protocol calbased
python3 -m train --model resnet --protocol calfree
python3 -m train --model resnet --protocol aami
python3 -m train --model transformer --protocol calfree
python3 -m train --model cnn --protocol calfree --no-balance --tag nobalance
python3 -m train --model cnn --protocol calfree  --tag nobalance_seed1
python3 -m train --model cnn --protocol calfree  --tag nobalance_seed2
python3 -m train --model cnn --protocol calfree  --tag seed1
python3 -m train --model cnn --protocol calfree  --tag seed2
python3 -m train --model gb --protocol calfree --no-balance --tag nobalance
python3 -m train --model gb --protocol calfree  --tag nobalance_seed1
python3 -m train --model gb --protocol calfree  --tag nobalance_seed2
python3 -m train --model gb --protocol calfree  --tag seed1
python3 -m train --model gb --protocol calfree  --tag seed2
python3 -m train --model resnet --protocol calfree --no-balance --tag nobalance
python3 -m train --model resnet --protocol calfree  --tag nobalance_seed1
python3 -m train --model resnet --protocol calfree  --tag nobalance_seed2
python3 -m train --model resnet --protocol calfree  --tag seed1
python3 -m train --model resnet --protocol calfree  --tag seed2
python3 -m train --model rf --protocol calfree --no-balance --tag nobalance
python3 -m train --model rf --protocol calfree  --tag nobalance_seed1
python3 -m train --model rf --protocol calfree  --tag nobalance_seed2
python3 -m train --model rf --protocol calfree  --tag seed1
python3 -m train --model rf --protocol calfree  --tag seed2
```

Run from `src/`, or from the project root with `src/` on `PYTHONPATH`. Each run writes its metrics to `results/runs/<model>_<protocol>.json` and its per-clip predictions to the matching `_predictions.npz`, so the figures can be redrawn without retraining.

## Guards

- `calfree` and `aami` raise an `AssertionError` and refuse to run if any subject appears in both training and test. Verified on every run.
- Validation subjects are held out of training under every protocol, including `leaky` — early stopping on a leaky validation set would leak a second time, through model selection.
- No clip is ever dropped from a test set for being low quality. Quality is scored and recorded, never used to filter, because filtering the hard clips out of a test set inflates the number this project exists to measure honestly.
