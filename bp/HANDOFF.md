# Handoff

Read this first if you are picking the project up cold — on another machine, in a
new session, or after a gap. It covers what is done, what is running, what broke
and how it was fixed, and what comes next. Numbers here are current as of the
last regeneration of `results/overview/RESULTS.md`; that file is generated from
the run files and is the authority.

## What the project is

TEEP Project B — cuffless blood pressure from PPG (+ECG). The spec is
`docs/TEEP_Research_Project_Proposals.md.pdf`, **Project B only** (Project A, the
PTB-XL work, belongs to a different student).

The question is not "how accurate is the model". It is **how much the answer
changes when you change who is in the test set** — and whether a few labelled
readings from a new person can close that gap.

## The one thing to understand before anything else

Every model is scored under four protocols. The model, the features and the
training clips are identical across all four. Only the test set changes.

| protocol | who is tested | has the model met them? |
|---|---|---|
| `leaky` | same patients, clips drawn at random | yes |
| `calbased` | same 1,293 patients, official held-out clips | yes |
| `calfree` | 144 patients | **no — this is the honest number** |
| `aami` | 116 patients, chosen to span the full BP range | no |

Much of the published field reports a number from the first two and describes it
as though it came from the third. Measuring that gap is the project.

## State of play

```
Month 1   literature review, environment                    done
Month 2   download, preprocessing pipeline                  done
Month 3   61 features + Random Forest / Gradient Boosting   done
Month 4   1D-CNN, ResNet1D, Transformer                     done
Month 5   leakage gap                                       done (early)
Month 6   few-shot personalization                          done  <- the contribution
Month 7   evaluate personalization, tune calibration amount  mostly done — see below
Month 8   PPG-DaLiA domain shift                            done (early)
Month 9   ablations (spec's 3: cal-free vs personalized,     done
          calibration amount, features vs raw-waveform)
Month 10  Bland-Altman, AAMI tables                         done (early)
Month 11  Draft manuscript                                  NOT STARTED
Month 12  Revise, internal review, submit                   NOT STARTED
```

### Trained so far — systolic / diastolic MAE, mmHg

| model | leaky | calbased | calfree | aami | calfree, no balancing |
|---|---|---|---|---|---|
| Random Forest | 7.47 / 4.34 | 7.14 / 4.11 | 12.76 / 8.38 | 19.67 / 12.24 | 12.93 / 8.45 |
| Gradient Boosting | 10.29 / 6.13 | 10.20 / 6.03 | 14.63 / 9.00 | 17.44 / 11.14 | 12.71 / 8.25 |
| 1D-CNN | 10.76 / 6.88 | 12.84 / 8.04 | 14.86 / 9.01 | 17.33 / 11.77 | 12.07 / 7.98 |
| ResNet1D | 8.35 / 5.49 | 8.66 / 5.40 | 13.58 / 8.97 | 17.64 / 11.82 | 13.15 / 8.67 |
| Transformer | 8.91 / 5.19 | 7.02 / 4.13 | 16.29 / 9.77 | 19.28 / 12.98 | 14.29 / 8.83 |

Every deep run: 20 epochs, batch 256, lr 3e-4, OneCycle, SmoothL1. All five
Transformer runs (leaky, calbased, calfree, calfree no-balance, aami) plus the
two PPG-DaLiA domain-shift runs finished on the PC's GPU with
`BP_SAFE_ATTENTION=1`. **Month 4 is complete.**

The Transformer does not beat the CNN or ResNet on `calfree` here (16.29 vs
12.07 / 13.58) — worse than even the interrupted Colab attempt (16.99), so this
is not a fluke of that one run. Its `leaky`/`calbased` numbers are its best
(8.91/5.19 and 7.02/4.13, actually beating the CNN on calbased), which is the
signature of a model that memorises rather than generalises — consistent with
finding 1 below. This is expected under this project's training budget: DMT's
12.17 comes from lr 2e-5, batch 32, 100 epochs, an auxiliary morphology head
and PPG-only z-scored input — none of which this run reproduces. Say this
plainly rather than treating it as a bug.

## Findings worth defending

**1. The leakage gap depends on the model family.** Ratio of calfree to the mean
of leaky and calbased:

```
Random Forest   1.75x      1D-CNN        1.26x
ResNet1D        1.60x      Gradient Boosting 1.43x
Transformer     2.05x
```

The inverse relationship is consistent, and the Transformer is now the clearest
case of it: 7.02 mmHg on calbased — the *best* of any model, beating the CNN's
12.84 — collapses to 16.29 on calfree, the *worst* of any model. So selecting on
leaky/calbased scores would have picked the Transformer as the winner and it is
in fact the weakest model calibration-free. No published paper runs five
architectures under four protocols on one dataset, so this has not been
reported.

The mechanism is two kinds of memorisation. Trees memorise the *patient* — a new
clip from a known patient stays easy. The CNN memorises the *clip* — its training
loss fell to 0.15 and that helped nothing, even on the same patient. The leaky
protocol only rewards the first kind.

**2. Most of the apparent domain shift on PPG-DaLiA was our own weighting.**

| model | with BP-bin balancing | without | how much was ours |
|---|---|---|---|
| Random Forest | +22.5 | +14.6 | 35% |
| Gradient Boosting | +32.2 | +19.4 | 40% |
| 1D-CNN | +34.2 | +11.6 | 66% |
| ResNet1D | +15.1 | +7.1 | 53% |
| Transformer | +13.3 | +11.4 | 14% |

PPG-DaLiA has no BP labels, so nothing would have caught this. Running the
control is the finding. The *collapse* — models giving all fifteen volunteers the
same answer — survives the control, so that part is real.

The Transformer's 14% is the opposite pattern from the other four (40-66%) and
we do not have a confirmed explanation. Best hypothesis, unproven: a model that
memorises training clips rather than the population-level BP relationship (see
finding 1) may not carry the weighting's correction into genuinely new data as
consistently as a model that generalises more — so more of its shift is real
domain effect and less is our own artifact. Flagged as open, not asserted.

**3. Balancing's cost depends on the model family too.** On the random forest it
is free (the average improves as well); on the CNN it costs 2.79 mmHg on the
average to buy 9.48 on the patients above 170. "Handle the imbalance" is not one
decision with one answer.

**4. A four-layer CNN matches the published state of the art, on this
benchmark's calibration-free evaluation specifically** — not a general claim
about the field. 12.07 mmHg against 12.17 for a 2026 transformer (DMT), 12.70
for XResNet1d101. Four years of architecture work spans 1.5 mmHg — less than
what changing the test rule does to a single model. **Do not claim the CNN
beats DMT**: the difference is 0.1 mmHg, diastolic goes the other way, our
model gets an extra ECG channel, and neither side reported seed variance —
we now have CNN seed variance in progress (professor's ask #2, 3 seeds), see
"What to do next".

## Where things are

```
src/train.py                the entire pipeline — 5 models, 4 protocols,
                            personalization, the PPG-DaLiA evaluation
src/models.py               CNN1D, ResNet1D, DMT transformer
src/bpdata.py               memory-mapped datasets, subject-disjoint splits
src/features.py             the 61 hand-crafted features
src/evaluation.py           MAE, SDE, AAMI, BHS
src/paths.py                every path; honours the BP_ROOT env var

results/models/<name>/      one folder per model: metrics, per-clip predictions,
                            figures, checkpoints
results/overview/           RESULTS.md and the cross-model figures
results/dataset/            figures about the data, true before any model existed

scripts/plot_results.py     regenerates every figure from the run files
scripts/make_results_report.py   regenerates RESULTS.md
scripts/build_training_deck.py   docs/Model_Training_Report.pdf, 16 slides
notebooks/                  launchers only; they contain no pipeline code
```

Any row of any results table reproduces with one command:

```
python -m train --model cnn --protocol calfree
```

Run it from `src/`, or with `src/` on `PYTHONPATH`.

## Running on a second machine

`src/paths.py` reads `BP_ROOT`, so the whole layout moves with one variable:

```python
import os, sys
BP_ROOT = r'C:\bp'
os.environ['BP_ROOT'] = BP_ROOT
sys.path.insert(0, os.path.join(BP_ROOT, 'src'))
sys.path.insert(0, os.path.join(BP_ROOT, 'scripts'))
```

`scripts/make_drive_folder.py` assembles a `bp/` staging folder with exactly what
a training machine needs (3.8 GB, hard-linked so it costs no disk). Move the
results back afterwards into `results/models/<name>/`.

## Things that broke on Windows, and the fixes

All four cost real time. They are in the code now, but the reasons are not
obvious from reading it.

**Application Control blocks compiled DLLs.** On a managed Windows machine both
`pyarrow` and `fastparquet` failed to import — the security policy blocks their
`.pyd` files, and no flag gets around it. The transformer only needs three
columns out of `meta.parquet` (age, sex, BMI), so those are exported to a plain
`demographics.npy` in each split folder, and `bpdata.demographics()` prefers it
when present. Regenerate them with the snippet in the git history, or read
`meta.parquet` on a machine that can and re-export.

Note this means **the tree models cannot run on that machine at all** — their 61
features live in parquet. That is fine; they are already trained.

**The data must sit on an SSD.** Training reads 419,040 clips in random order
every epoch. On a spinning disk each read costs a seek and an epoch took over
eight minutes with the GPU at 83% but starved. Moving to SSD fixed it.

**`num_workers` must be 0 inside a Windows notebook.** Worker processes try to
re-import the notebook and fail. On Linux and macOS this is not an issue.

**The fused attention kernel can fault.** `scaled_dot_product_attention` picks a
backend on its own, and on Windows with a recent GPU (RTX 2000 Ada, compute 8.9,
torch 2.6.0+cu124) it raised `CUDA error: an illegal memory access was
encountered` on the first training step, reproducibly, after a kernel restart.
Set `BP_SAFE_ATTENTION=1` before importing anything from `src/` and
`models.SelfAttention` falls back to plain matmuls — slower, always correct.

```python
os.environ['BP_SAFE_ATTENTION'] = '1'      # before the first import
```

## What to do next

**In progress right now (professor's post-presentation review, priority order):**

1. **Session-leakage check (professor's #1)** — done, see the CORRECTION box
   below. Personalization's DBP-AAMI-pass claim was retracted and replaced
   with the honest true-chronological version.
2. **Seed variance (professor's #2)** — professor separately raised the
   training budget to 50 epochs for every deep model and asked our
   Transformer be brought closer to DMT's recipe. That forced a full retrain,
   so this project is mid-transition:
   - RF/GB: calfree seeds 1,2 done (balanced); calfree_nobalance seeds 1,2
     running locally.
   - CNN/ResNet: all results wiped and moved to Colab T4 (Mac MPS was far too
     slow — 664s/epoch measured, ~9h for one 50-epoch run). `bp/` staging
     folder rebuilt with 50-epoch notebooks that include calfree +
     calfree_nobalance, each ×3 seeds. Being run on Colab now.
   - Transformer: being redone by hand to match DMT's recipe more closely, on
     a separate GPU machine, also at 50 epochs.
   - `docs/Final_Project_Report.pdf` (the deck that replaced
     `Model_Training_Report.pdf`) cannot fully rebuild until CNN/ResNet
     results are back — `scripts/build_final_deck.py` is otherwise fully
     updated for the corrected personalization claim, the scoped literature
     claims (#3, #6), the new SBP finding (#4), the Transformer
     weighting-asymmetry hypothesis (#5), and BHS grades on the
     personalization table (polish item). Rebuild once Colab/GPU results land.
3. **Writeup-only fixes (#3-6 + polish)** — done in the deck script (see
   above); RESULTS.md and this file's "Findings worth defending" section are
   updated too.

**Month 4 is done — all five models, all four protocols, at the OLD 20-epoch
budget; being superseded by the 50-epoch retrain above.** The Transformer's
training curve shows the sharpest overfitting of any model here — train loss
0.72 → 0.04 while validation loss stopped improving around epoch 2-3 on every
protocol. Worth saying plainly: **this is a transformer trained under this
project's budget, not a reproduction of DMT.** DMT uses lr 2e-5, batch 32, 100
epochs, separate SBP and DBP networks, PPG only, and an auxiliary
morphology-classification head with an uncertainty-weighted loss. We reproduce
its architecture (patch 10, 6 blocks, 8 heads, FiLM on age/sex/BMI) and none of
its training recipe. The missing auxiliary head is the most likely reason for
the gap, since its job is exactly to restrain overfitting — and it shows: our
Transformer is the best model on calbased (7.02, beats the CNN) and the worst on
calfree (16.29), the most extreme leaky/calfree split of any architecture here.

**1. Personalization is done — and it is the strongest finding in the project.**
Spec sections J and L, months 6 and 7. `train.personalize(model, source="calfree")`
now reads the saved `*_predictions.npz` files instead of re-fitting — a 25-clip
affine (offset+scale) recalibration per subject, all five models, on the Mac
CPU, in 23 seconds total, no GPU. Results are in
`results/models/<model>/<model>_personalization_calfree.json`.

**CORRECTION (this section originally claimed all five models pass AAMI on
diastolic BP — that claim did not survive a rigorous check and has been
retracted below. Read this whole box before quoting any number from it.**

The professor asked the right question: are the k=25 calibration clips and the
held-out evaluation clips independent in time, or could same-session
autocorrelation be doing personalization's work for it? We checked, using
PulseDB's own `SegIDX` field (pulled from the official `CalFree_Test_Info.mat`
proxy file on the authors' Drive — not shipped with the Subset files we train
on, so this required a separate download; see
`scripts/check_session_leakage.py`).

**Finding: storage order is not chronological.** Correlation between a clip's
position in the file and its true SegIDX is 0.007 — effectively zero. So the
"first 25" clips `personalize()` was using are scattered across the whole
session, not the actual earliest 25. With 400 clips spread across a session
that is ~2 hours long on average, that means every evaluation clip has a
calibration clip within about **3 minutes** of it on average — close enough
for the model to be exploiting short-term drift rather than the patient's
stable physiology. This is exactly the failure mode the professor described.

**Rerun with true chronological calibration** (earliest 25 clips by real
SegIDX; evaluation clips are ~70 minutes later on average — the closest thing
to real deployment this single-session dataset supports):

| model | DBP MAE, same-time k=25 | DBP AAMI, same-time | DBP MAE, true-early k=25 | DBP AAMI, true-early |
|---|---|---|---|---|
| Random Forest | 4.81 | pass | 7.32 | **fail** |
| Gradient Boosting | 4.74 | pass | 7.11 | **fail** |
| 1D-CNN | 4.54 | pass | 7.15 | **fail** |
| ResNet1D | 4.99 | pass | 7.60 | **fail** |
| Transformer | 5.12 | pass | 7.64 | **fail** |

DBP MAE still improves substantially under the honest test (e.g. CNN 9.01 →
7.15, roughly a fifth to a third off across models) — personalization is real
and worth reporting. It just does not cross AAMI's SD≤8 threshold once
calibration and evaluation are genuinely separated in time; bias stays under 2
mmHg throughout, so it is the spread, not the bias, that falls short. **The
corrected claim: personalization meaningfully reduces DBP error; it does not
reach the AAMI bar once tested honestly.** SBP was already known not to reach
it either way — see the ceiling analysis below for why.

The ceiling is measured. Giving every patient a perfect constant offset — fitted
on their own test labels, so unreachable — takes the CNN from 12.07 to 8.25, and
a perfect offset *and* scale takes it to 7.70; the realised 25-shot number, 7.87,
is already close to that ceiling. The reason the ceiling is 7.70 and not 1–2:
within a single patient's own recording, systolic wanders with SD 13.9 mmHg,
while the spread *between* patients is 11.8. No per-patient constant can track
that. Beating it needs beat-to-beat information, which is a different project.

All three of spec section L's ablations are now in `results/overview/RESULTS.md`
under "## Personalization": (1) calfree vs personalized, (2) calibration amount
(k=1/3/5, exactly what the spec asks), (3) classical features (RF) vs
raw-waveform (CNN) — RF wins 3 of 4 protocols, including `calfree` (12.76 vs
14.86), and loses only on `aami`. One honest wrinkle worth keeping in the
write-up: RF and ResNet briefly get *worse* at `k=3` before improving — 2-3
clips is too few to fit a stable offset+scale from, so personalization is not
free below some minimum.

Not yet done: the `source="aami"` arm still re-fits (since `aami_cal`
predictions were never saved by `run()`) — that needs the PC's GPU again for
the three deep models, so it was left out of this pass. The new personalization
+ ablation numbers are not yet in `docs/Model_Training_Report.pdf` — the deck
still reflects the pre-transformer, pre-personalization state.

**3. One question for the professor, still unanswered.** Spec section G states
PPG-DaLiA "does not contain BP labels at all"; sections E and N (month 8) ask how
much personalization helps there. Calibration needs labels. This probably means
the distribution check only, but it should be confirmed rather than assumed.

## How to talk about the results

Two habits that have already paid off:

**Separate what was measured from what was inferred.** The PPG-DaLiA shift is
measured — it is our model's average on one dataset minus its average on another,
and needs no labels. That fifteen healthy volunteers should not read 130 mmHg is
an inference from outside knowledge. Say which is which.

**Do not claim differences smaller than the noise.** Nobody in this literature
reports seed variance, including us. A 0.1 mmHg difference means nothing. Every
finding above rests on a gap of 2 mmHg or more, which is why they hold.
