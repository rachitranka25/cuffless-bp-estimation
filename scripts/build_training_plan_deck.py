"""Build docs/Model_Training_Plan.pdf — the plan for the modelling phase.

Starts from what the project spec asks for, then sets out which data feeds which
model, how each model will be scored, and what the published results are for the
same models on the same dataset.

Run:  python3 scripts/build_training_plan_deck.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import FIG_PLAN, DOCS          # noqa: E402
from deck import Deck, INK, INK2, MUTED    # noqa: E402

TP = FIG_PLAN
d = Deck()


def figure_slide(kicker, title, sub, fig, caption, top=2.15, bottom=1.1):
    s = d.slide(kicker, title, sub)
    d.image(s, fig, .6, top, 12.1, 7.5 - top - bottom)
    if caption:
        d.text(s, .7, 7.5 - bottom + .05, 11.9, bottom, caption,
               size=12.5, color=INK2, space_after=5)
    return s


# ══════════════════════════════════════════════════════════════ title
d.title_slide(
    "TEEP RESEARCH PROJECT B",
    "Model Training Plan",
    "Which models, which data goes into each, and how each one will be scored",
    "Rachit Ranka",
    "The data is prepared. This is the plan for the modelling phase.")

# ══════════════════════════════════════════════════════════════ the spec
s = d.slide("Starting point", "What the project spec asks for",
            "Section I of the spec lists the models, in this order.")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["", "The spec's words", "What that means in practice"],
    ["1", "Random Forest / Gradient Boosting on hand-crafted\npulse-transit-time and PPG-morphology features",
     "Measure things about each pulse first —\nwidth, area, timing — then fit trees to them"],
    ["2", "1D-CNN on raw PPG (+ECG) waveform",
     "Give the network the signal itself and let\nit work out what matters"],
    ["3", "ResNet1D / lightweight Transformer",
     "Deeper versions of the same idea, which is\nwhat the recent PulseDB papers use"],
], col_w=[.5, 5.2, 5.4], size=12.5, row_h=.78)
d.text(s, .8, bottom + .3, 11.5, 1.5, [
    "The spec also names one thing NOT to build: it calls the leakage-style numbers from older papers "
    "“reference-only, not to be re-implemented” — they are quoted to show the inflation, not benchmarked.",
    "",
    "And it names the method that is meant to be the contribution: “a lightweight few-shot personalization "
    "layer — a small per-subject fine-tuning of the final regression layers … using only a handful of "
    "labeled calibration segments per new subject”.",
], size=12.5, color=INK2, space_after=5)

s = d.slide("Starting point", "The pipeline the spec describes",
            "Section H, in the spec's own order.")
d.text(s, .8, 2.5, 11.5, 4.3, [
    "     Raw synchronised PPG + ECG + ABP segments (PulseDB)",
    "         ↓",
    "     Preprocessing — bandpass filtering, PPG/ECG fiducial point detection",
    "         ↓",
    "     Segmentation — use PulseDB's official 10-second segments",
    "         ↓",
    "     Feature / representation — pulse-transit-time and morphology features, plus a raw-waveform branch",
    "         ↓",
    "     Baseline models — trained and subject-independently validated on PulseDB splits",
    "         ↓",
    "     Proposed model — baseline plus few-shot personalization",
    "         ↓",
    "     Evaluation — (1) PulseDB subject-independent test, (2) domain-shift test on PPG-DaLiA",
], size=13.5, color=INK2, space_after=4)
d.text(s, .8, 6.95, 11.5, .5,
       "The first three steps are the preparation already reported. The rest is what follows.",
       size=12, color=MUTED)

# ══════════════════════════════════════════════════════════════ the plan
s = d.slide("The plan", "Which data goes into which model")
d.image(s, TP / "01_which_data_which_model.png", .35, 1.68, 12.6, 5.6)

s = d.slide("The plan", "The two ways of feeding the same clips",
            "Both families train on the identical 465,480 clips from 1,293 patients.")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["", "Model 1 — classical", "Models 2, 3, 4 — deep"],
    ["Reads", "data/features/*.parquet", "data/processed/pulsedb/*/signals.npy"],
    ["Per clip", "60 numbers", "2 signals x 1250 numbers"],
    ["What those are", "Pulse-transit time, pulse width, rise\ntime, area ratio, stiffness index …",
     "The raw ECG and PPG waveform"],
    ["Who chose them", "We did — measured with a pulse detector", "Nobody. The network learns its own"],
    ["Extra inputs", "Age, sex, height, weight, BMI", "Transformer also takes age, sex, BMI"],
    ["Runs on", "CPU, minutes", "GPU; the Transformer needs a cloud GPU"],
], col_w=[1.9, 4.6, 5.0], size=12.5, row_h=.56)
d.text(s, .8, bottom + .3, 11.5, 1.1, [
    "The comparison between the two is one of the spec's ablations: “classical features vs. raw-waveform CNN”. "
    "It answers whether hand-measuring the pulse is still worth doing, or whether the network does better "
    "left alone.",
], size=12.5, color=INK2)

s = d.slide("The plan", "The four ways each model will be scored")
d.image(s, TP / "02_evaluation_rules.png", .5, 1.9, 12.3, 4.4)
d.text(s, .7, 6.45, 11.9, .9, [
    "The first two score the model on people it has already met. The last two do not.",
    "The gap between them is what the project exists to measure, and the fourth is the fix it proposes.",
], size=12.5, color=INK2, space_after=5)

s = d.slide("The plan", "How the training runs will be set up",
            "Section L of the spec, point by point.")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["What the spec asks for", "How it will be done"],
    ["Subject-level split", "Use PulseDB's official groups rather than inventing our own"],
    ["Cross-validation", "k-fold by patient inside the training group, for tuning only"],
    ["Prevention of leakage", "Assertions in code — no patient ID shared between train,\ncalibration and test"],
    ["Target imbalance", "BP is skewed in ICU patients — stratified sampling or a\nweighted loss across BP bins"],
    ["Personalization protocol", "For a held-out patient, use only the first few of their clips\nas calibration; score on the rest"],
], col_w=[2.8, 7.4], size=12.5, row_h=.62)
d.text(s, .8, bottom + .3, 11.5, 1.1, [
    "The spec calls the leakage assertions “itself worth reporting as a methodological contribution, since it "
    "directly documents what much prior work skipped”.",
], size=12.5, color=INK2)

s = d.slide("The plan", "How the results will be measured",
            "Section K. Two metrics are primary because the clinical standard needs both.")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["Metric", "Why", "Role"],
    ["Mean absolute error (MAE)", "The number the whole field reports, for SBP and DBP", "Primary"],
    ["Standard deviation of error", "AAMI requires both the mean error and its spread", "Primary"],
    ["AAMI / BHS compliance", "Mean error ≤ 5 mmHg and SD ≤ 8 mmHg — the clinical bar", "Primary"],
    ["Pearson correlation", "Does the prediction move with the truth at all", "Secondary"],
    ["Bland-Altman plots", "The standard way of comparing two BP methods", "Secondary"],
], col_w=[2.8, 6.4, 1.6], size=12.5, row_h=.56)
d.text(s, .8, bottom + .3, 11.5, 1.2, [
    "A good average error alone is not enough. A model can be right on average and still swing wildly from "
    "reading to reading, which is why the spec makes the spread primary as well.",
], size=12.5, color=INK2)

s = d.slide("The plan", "The method the spec calls the contribution",
            "Everything above is a baseline. This is the part that is meant to be new.")
d.text(s, .8, 2.4, 11.5, 4.5, [
    "A new patient puts on the watch. The model has never seen them, and on its own it is not accurate enough "
    "to be useful.",
    "",
    "So we ask them for a handful of real cuff readings — once, at home or at a clinic visit — and use those "
    "to adjust the model to that person. Only the final regression layers are re-fitted, or a simple shift is "
    "applied in feature space. Nothing is retrained from scratch.",
    "",
    "The question that follows is the one nobody has answered:",
    "",
    "          how few readings are enough?    1?    3?    5?",
    "",
    "That is a planned ablation in the spec, and it is the difference between a device an elderly person can "
    "set up at home and one they cannot.",
], size=14, color=INK2, space_after=6)

s = d.slide("The plan", "The experiments this produces")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["Experiment", "What it compares", "What it answers"],
    ["Leakage gap", "Same model, leaky split vs patient-separated split",
     "How much does the wrong split inflate?"],
    ["Model family", "Classical features vs raw-waveform networks",
     "Is hand-measuring the pulse still worth it?"],
    ["Architecture", "CNN vs ResNet vs Transformer, all patient-separated",
     "Does a deeper model actually help here?"],
    ["Personalization", "Calibration-free vs 1, 3, 5 calibration clips",
     "How few readings does a new person need?"],
    ["Domain shift", "Hospital-trained model run on watch data",
     "Does any of it survive outside the ICU?"],
], col_w=[1.9, 5.0, 4.4], size=12.5, row_h=.6)
d.text(s, .8, bottom + .3, 11.5, .9,
       "Each row is a comparison, not a single number. The comparisons are the result.",
       size=12.5, color=INK2)

# ══════════════════════════════════════════════════════════════ literature
s = d.slide("For comparison", "What other groups reported on this same dataset",
            "All of these use PulseDB, and all use its official groups.")
bottom = d.table(s, .7, 2.35, 12.0, [
    ["Study", "Year", "Model", "Patients it had seen", "Brand new patients"],
    ["PulseDB (the dataset paper)", "2023", "1D ResNet-18", "not reported", "not reported"],
    ["Validation study", "2024", "CNN + LSTM + attention", "not reported", "13.67  /  8.56"],
    ["Validation study", "2024", "UTransBPNet (U-Net + Transformer)", "not reported", "12.50  /  8.32"],
    ["Benchmark study", "2025", "XResNet1d101 (best of five)", "9.08  /  6.08", "12.70  /  8.05"],
    ["DMT", "2026", "Transformer + age, sex, BMI", "4.56  /  2.62", "12.17  /  7.89"],
    ["rU-Net", "2024", "U-Net + ResNet + attention", "4.49  /  2.69", "4.14 / 2.48  after tuning"],
], col_w=[2.9, .7, 3.8, 2.3, 2.6], size=11.5, row_h=.52)
d.text(s, .7, bottom + .3, 12.0, 1.3, [
    "All figures are mean absolute error in mmHg, written SBP / DBP. Lower is better.",
    "The last row is the exception that matters: rU-Net's second number comes after giving the model 10% of "
    "the new patient's own readings.",
], size=12.5, color=INK2, space_after=5)

s = d.slide("For comparison", "The pattern in those numbers",
            "Read the last two columns of the previous slide across, not down.")
d.text(s, .8, 2.4, 11.5, 4.5, [
    "     Patients the model had already seen         about   4  to   9 mmHg",
    "     Patients it had never seen                  about  12  to  14 mmHg",
    "     Never seen, but given a few of their own readings    about   4 mmHg",
    "",
    "",
    "Three things follow from that.",
    "",
    "     One.   The gap between the first two lines is not one paper's weakness. Every study that reports "
    "both shows it. It is the honest difficulty of the problem.",
    "",
    "     Two.   Four years of new architectures moved the middle line by about 1.5 mmHg in total — from "
    "13.67 in 2024 to 12.17 in 2026. The model is not the bottleneck.",
    "",
    "     Three.  The third line is a different order of improvement, and it comes from giving the model a "
    "little information about the person rather than from a better network.",
], size=13.5, color=INK2, space_after=5)

s = d.slide("For comparison", "What we should expect, and what we should aim for")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["Setting", "Realistic expectation", "Why"],
    ["Leaky split", "roughly 5 to 8 mmHg", "Deliberately inflated — reported only to measure it"],
    ["Patients already seen", "roughly 5 to 9 mmHg", "Matches what every paper reports"],
    ["Brand new patients", "roughly 12 to 13 mmHg", "Where the whole field sits, regardless of model"],
    ["After personalization", "roughly 4 to 5 mmHg", "The one published precedent reports 4.14 / 2.48"],
], col_w=[2.6, 3.2, 5.6], size=12.5, row_h=.56)
d.text(s, .8, bottom + .3, 11.5, 1.6, [
    "Building a deeper network is worth doing, but it is worth roughly 1 mmHg. Personalization is worth "
    "roughly 8. The plan is weighted accordingly.",
    "",
    "The spec agrees: it says the project is a result even “without beating any state-of-the-art number”, "
    "provided the leakage gap, the personalization recovery and the domain shift are all quantified honestly.",
], size=12.5, color=INK2, space_after=5)

# ══════════════════════════════════════════════════════════════ sources
s = d.slide("Sources", "References")
d.text(s, .8, 2.5, 11.5, 4.3, [
    "TEEP Research Project Proposals — Project B, sections H, I, J, K, L.",
    "",
    "PulseDB: a large, cleaned dataset based on MIMIC-III and VitalDB — Wang et al.,",
    "Frontiers in Digital Health, 2023.    doi.org/10.3389/fdgth.2022.1090854",
    "",
    "Generalizable deep learning for PPG-based blood pressure estimation — a benchmarking study, 2025.",
    "arXiv:2502.19167",
    "",
    "DMT: Demographic Conditioning, Morphology-Enhanced Transformer, 2026.    arXiv:2606.11125",
    "",
    "Validation of deep learning models for cuffless BP estimation on a large benchmarking dataset, 2024.",
    "oaepublish.com/articles/chatmed.2023.23",
    "",
    "rU-Net, Multi-Scale Feature Fusion and Transfer Learning, IEEE JBHI, 2024.",
    "ieeexplore.ieee.org/document/10721372",
    "",
    "Investigation of Data Leakage in Deep-Learning-Based Blood Pressure Estimation, IEEE, 2023.",
], size=12.5, color=INK2, space_after=1)
d.text(s, .8, 7.02, 11.5, .4,
       "Published figures are quoted under the test rule each paper used. The rU-Net numbers come from its "
       "abstract and should be checked against the full text before being cited in a manuscript.",
       size=10, color=MUTED)

d.save_pdf(DOCS / "Model_Training_Plan.pdf")
