"""Published results on the datasets this project uses, for comparison.

Every row was read from the paper (or its abstract where the full text is
paywalled) and carries the protocol it was measured under, because a cuffless-BP
number is meaningless without one: the same model can report 4.5 or 12.9 mmHg
depending only on whether test subjects were seen during training.

Protocols
    cal_free   subject-independent — no test subject appears in training
    cal_based  subject-overlapping — test subjects were seen, different segments
    aami       PulseDB's AAMI subset, which spans the full BP range on purpose
    finetuned  a few labelled segments from the test subject are used to adapt

`verified` marks whether the number came from the full text (True) or only from
an abstract / indexed summary (False) — the latter should be re-checked before
being quoted in a manuscript.
"""

OURS = "This project"

PAPERS = [
    dict(key="pulsedb", short="PulseDB (Wang et al.)", year=2023,
         venue="Front. Digit. Health",
         url="https://doi.org/10.3389/fdgth.2022.1090854",
         model="1D ResNet-18", dataset="PulseDB (MIMIC+VitalDB)",
         note="Introduced the dataset and the three protocols. Showed the "
              "calibration-free gap but published no final MAE.",
         results=[], verified=True),

    dict(key="cnnlstm", short="CNN-LSTM-Attention", year=2024,
         venue="Conn. Health Telemed.",
         url="https://www.oaepublish.com/articles/chatmed.2023.23",
         model="CNN + BiLSTM + attention", dataset="PulseDB VitalDB",
         note="Subject-independent cross-validation on the VitalDB subset.",
         results=[("cal_free", 13.67, 8.56)], verified=True),

    dict(key="utrans", short="UTransBPNet", year=2024,
         venue="Conn. Health Telemed.",
         url="https://www.oaepublish.com/articles/chatmed.2023.23",
         model="U-Net + Transformer", dataset="PulseDB VitalDB",
         note="Same study as above; best of its five variants.",
         results=[("cal_free", 12.50, 8.32)], verified=True),

    dict(key="bench", short="XResNet1d101 (benchmark)", year=2025,
         venue="IOP / arXiv:2502.19167",
         url="https://arxiv.org/abs/2502.19167",
         model="XResNet1d101", dataset="PulseDB VitalDB",
         note="Benchmarked LeNet1D, XResNet1d50/101, Inception1D and S4, then "
              "tested out-of-distribution on four external datasets. Numbers are "
              "the VitalDB rows of Table III, matching this project's subset; the "
              "Combined rows are worse (13.97 / 8.51 calibration-free). Reports "
              "single training runs — its bootstrap intervals resample the test "
              "set, not the training seed.",
         results=[("cal_free", 12.70, 8.05), ("cal_based", 9.09, 6.09),
                  ("aami", 19.31, 12.33)], verified=True),

    dict(key="dmt", short="DMT (Transformer)", year=2026,
         venue="arXiv:2606.11125",
         url="https://arxiv.org/abs/2606.11125",
         model="Transformer + demographic FiLM conditioning",
         dataset="PulseDB",
         note="Current best calibration-free number we found. Conditions the "
              "transformer on age, sex and BMI. Its segment counts — 465,480 "
              "train, 51,720 calibration-based, 57,600 calibration-free — are "
              "identical to the VitalDB subsets used here, so the comparison is "
              "like for like. Single training run; the std column in its Table I "
              "is the error spread, not run-to-run variance.",
         results=[("cal_free", 12.17, 7.89), ("cal_based", 4.56, 2.62),
                  ("aami", 17.16, 11.31)], verified=True),

    dict(key="runet", short="rU-Net + transfer learning", year=2024,
         venue="IEEE JBHI",
         url="https://ieeexplore.ieee.org/document/10721372/",
         model="U-Net + ResNet, STFT, multi-head attention",
         dataset="PulseDB",
         note="Fine-tunes on 10% of a new subject's data — the closest published "
              "precedent for this project's personalization step.",
         results=[("cal_based", 4.49, 2.69), ("finetuned", 4.14, 2.48)],
         verified=False),

    # `results` stays empty on purpose. The comparison figures read this
    # project's numbers straight out of results/models/*/*.json instead, so a
    # chart can never disagree with the run files.
    dict(key=OURS, short="This project", year=2026,
         venue="—", url="",
         model="Random Forest, Gradient Boosting, 1D-CNN, ResNet1D, Transformer "
               "— all five trained under all four protocols, plus "
               "few-shot personalization",
         dataset="PulseDB VitalDB (+ PPG-DaLiA as an external check)",
         note="Numbers live in results/overview/RESULTS.md. Like every paper "
              "above, these are single training runs — no seed variance measured "
              "yet, which is why differences under about half a mmHg are not "
              "claimed either way.",
         results=[],
         verified=True),
]

# Architectures that appear across these papers, for the "what models are used"
# slide. Ordered roughly by how they show up in the literature.
ARCHITECTURES = [
    ("1D CNN", "LeNet1D, plain convolutional stacks", "The starting baseline everywhere"),
    ("ResNet1D", "ResNet-18, XResNet1d50/101", "The strong standard deep baseline"),
    ("Inception1D", "multi-scale convolution", "Captures several beat scales at once"),
    ("U-Net variants", "UTransBPNet, rU-Net", "Encoder-decoder, often with attention"),
    ("Recurrent + attention", "CNN-LSTM-Attention", "Older but still competitive"),
    ("State space", "S4", "Long-sequence model, benchmarked but not best"),
    ("Transformer", "DMT, UTransBPNet", "Current best; often conditioned on demographics"),
]


# Preprocessing choices, paper by paper. These are what make two numbers
# comparable or not, and they are rarely put side by side anywhere.
#
# Most of the pipeline is fixed by PulseDB itself — 125 Hz, 10 s segments,
# band-pass filtered, one segment per row — so every paper inherits it. What
# actually differs is the last three rows: which channels go in, how the signal
# is scaled, and whether SBP and DBP get one model or two.
PREPROCESSING = [
    # (aspect,            PulseDB default,          benchmark 2025,  DMT 2026,        ours)
    ("Sampling rate",     "125 Hz",                 "125 Hz",        "125 Hz",        "125 Hz"),
    ("Segment length",    "10 s = 1250 samples",    "10 s / 1250",   "1250 samples",  "10 s / 1250"),
    ("Filtering",         "done by PulseDB",        "inherited",     "not described", "inherited"),
    ("Input channels",    "ECG + PPG available",    "PPG only",      "PPG only",      "ECG + PPG"),
    ("Signal scaling",    "min-max to [0,1]",       "not stated",    "z-score",       "min-max to [0,1]"),
    ("Extra inputs",      "—",                      "none",          "age, sex, BMI", "none (in the deep model)"),
    ("SBP/DBP heads",     "—",                      "one model",     "two models",    "one model, two outputs"),
    ("Splits",            "official 3 protocols",   "official",      "official",      "official"),
]

# Training-set sizes, so "how much data" is answerable.
DATA_SCALE = [
    ("Benchmarking study 2025", "PulseDB VitalDB", "1,293", "418,986"),
    ("Benchmarking study 2025", "PulseDB combined", "2,217", "801,720"),
    ("DMT 2026", "PulseDB VitalDB", "not stated", "465,480"),
    ("This project", "PulseDB VitalDB", "1,293", "465,480"),
]


# What each paper was actually trying to do, and how it prepared its data.
# The "goal" column matters: several of these papers report very different
# numbers because they were answering different questions, not because one
# method is better than another.
PAPER_DETAIL = [
    dict(short="PulseDB", year=2023,
         goal="Build a large, clean benchmark with patient IDs and standard test rules",
         prep="Extracted from MIMIC-III and VitalDB, filtered for quality, cut to 10 s, "
              "min-max scaled, labels from the arterial line",
         amount="5,361 patients · 5.2M clips",
         model="1D ResNet-18",
         seen="not reported as a final number",
         new="not reported as a final number"),
    dict(short="Validation study", year=2024,
         goal="Check whether deep models hold up on a large benchmark",
         prep="PulseDB VitalDB as shipped; no extra filtering reported",
         amount="2,154 patients · 1.26M clips",
         model="CNN-LSTM-attention, UTransBPNet",
         seen="not reported separately",
         new="12.50 / 8.32  (best variant)"),
    dict(short="rU-Net", year=2024,
         goal="Push accuracy as high as possible, then adapt to each new patient",
         prep="PulseDB, PPG + ECG, short-time Fourier transform features added",
         amount="PulseDB (count not stated)",
         model="U-Net + ResNet + attention",
         seen="4.49 / 2.69",
         new="4.14 / 2.48 after tuning on 10% of the new patient"),
    dict(short="Benchmark study", year=2025,
         goal="Compare five architectures fairly, then test them on outside data",
         prep="PulseDB as shipped, PPG only, 125 Hz, 10 s clips",
         amount="1,293 patients · 418,986 clips (Seoul half)",
         model="LeNet1D, XResNet1d50/101, Inception1D, S4",
         seen="9.08 / 6.08",
         new="12.70 / 8.05"),
    dict(short="DMT", year=2026,
         goal="Improve accuracy by feeding the model the patient's age, sex and BMI",
         prep="PulseDB, PPG only, 1250 samples, z-score scaling, pulse-shape labels added",
         amount="465,480 clips",
         model="Transformer + FiLM demographic conditioning",
         seen="4.56 / 2.62",
         new="12.17 / 7.89"),
    dict(short="This project", year=2026,
         goal="Measure how much the test rule inflates results, then fix it with personalization",
         prep="PulseDB as shipped, ECG + PPG, 125 Hz, 10 s, min-max, float16",
         amount="1,293 patients · 465,480 clips",
         model="Random Forest, Gradient Boosting, 1D-CNN, ResNet1D, Transformer",
         seen="7.02 / 4.13 (Transformer, calbased; best of five)",
         new="12.07 / 7.98 calibration-free (CNN, best of five); "
             "7.87 / 4.54 after 25-clip personalization, DBP passes AAMI"),
]

# The one paper that uses both of our datasets. It is not a blood-pressure
# paper, which is the distinction that keeps this project's gap open.
BOTH_DATASETS = dict(
    title="Reduction of motion artifacts from PPG signals using learned convolutional sparse coding",
    year=2025, venue="Physiological Measurement / arXiv:2508.10805",
    task="Removing motion artifacts, then estimating heart rate",
    pulsedb="training — clean clips with synthetic motion artifacts added",
    dalia="testing — real motion from daily life",
    metrics="SNR -7.06 to 11.23 dB on synthetic; heart-rate error 11.29 to 8.69 bpm on PPG-DaLiA",
    bp="Never predicts or evaluates blood pressure",
)


def rows(protocol):
    """(label, sbp, dbp, is_ours) for every paper reporting that protocol."""
    out = []
    for p in PAPERS:
        for proto, sbp, dbp in p["results"]:
            if proto == protocol:
                out.append((p["short"], sbp, dbp, p["key"] == OURS))
    return sorted(out, key=lambda r: r[1])
