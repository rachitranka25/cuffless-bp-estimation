"""Build the project report decks, in plain English.

    python3 scripts/build_report_deck.py             -> Preprocessing_Report.pdf
                                                        the short version to present
    python3 scripts/build_report_deck.py --detailed  -> Preprocessing_Report_detailed.pdf
                                                        every evidence slide
    python3 scripts/build_report_deck.py --full      -> Project_B_Report.pdf
                                                        the above plus literature and results

The short deck is what gets shown; the supervisor's feedback was that thirty
slides is too many and most of it was unnecessary. The detailed one still
renders every piece of evidence, because a supervisor may ask "show me where you
did that" about any single claim and the answer should already exist.

Written for a reader who is not a signal-processing specialist: every term is
explained the first time it appears, and the preprocessing section shows real
patient data rather than asserting that the work was done.
"""

import os
import sys

FULL = "--full" in sys.argv
DETAILED = FULL or "--detailed" in sys.argv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import FIG_PREP, FIG_LIT, DOCS              # noqa: E402
from deck import Deck, INK, INK2, MUTED                # noqa: E402
from literature import PAPER_DETAIL, BOTH_DATASETS     # noqa: E402

PRE = FIG_PREP
LIT = FIG_LIT
d = Deck()


def figure_slide(kicker, title, sub, fig, caption, top=2.25, bottom=1.3):
    s = d.slide(kicker, title, sub)
    d.image(s, fig, .7, top, 11.9, 7.5 - top - bottom)
    if caption and caption != [""]:
        d.text(s, .7, 7.5 - bottom + .05, 11.9, bottom, caption,
               size=12.5, color=INK2, space_after=5)
    return s


# ══════════════════════════════════════════════════════════════ title
d.title_slide(
    "TEEP RESEARCH PROJECT B",
    "Cuffless Blood Pressure Estimation",
    "The goal, the two datasets, and how the data was prepared",
    "Rachit Ranka",
    "Every number here was measured from the data, not quoted from memory")

# ══════════════════════════════════════════════════════════════ the project
s = d.slide("The project", "What we are doing, and what we are checking",
            "Blood pressure needs a cuff. A watch already has a sensor that might replace it.")
d.text(s, .8, 2.4, 11.5, 4.5, [
    "A cuff squeezes your arm to measure blood pressure. It works, but an elderly person living alone cannot "
    "manage it five times a day. A smartwatch already shines a light into the skin and records the pulse, all "
    "day, for free. That signal is called PPG. Can a computer read blood pressure out of it?",
    "",
    "Many papers say yes. But a lot of them tested the wrong way: they put some clips from a person in the "
    "training set and other clips from the SAME person in the test set. The model had already met that person. "
    "It is like giving a student the exam paper the night before — the marks tell you nothing.",
    "",
    "     1.   How much better does that make the results look?",
    "     2.   What is the real number, on a person the model has never met?",
    "     3.   What happens on a real smartwatch instead of hospital equipment?",
    "     4.   If we give the model a few real readings from the new person, how much does it improve?",
], size=14, color=INK2, space_after=6)
d.text(s, .8, 7.02, 11.5, .4,
       "From Section B of the project spec: a “subject-independent, cross-setting protocol (train on ICU data, "
       "test on free-living wearable data) … and how much of that gap can be recovered using a lightweight, "
       "few-shot personalization calibration step”.", size=10, color=MUTED)

# ══════════════════════════════════════════════════════════════ the data
s = d.slide("The data", "Two datasets",
            "One to teach the model. One to check whether it survives outside the hospital.")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["", "PulseDB", "PPG-DaLiA"],
    ["Where it comes from", "Hospital ICU and operating rooms", "Ordinary daily life"],
    ["Sensor", "Clinical finger sensor + ECG", "Smartwatch on the wrist"],
    ["People", "1,553 patients", "15 volunteers"],
    ["Recording per person", "1 hour  (60 to 100 min, by group)", "2.4 hours  (88 to 177 min)"],
    ["Total", "1,794 hours · 646,000 clips of 10 s", "36 hours"],
    ["Blood pressure labels", "Yes — measured, one per clip", "None at all"],
    ["We use it to", "Train the model and test it", "Check it on real-world data"],
], col_w=[2.4, 3.4, 3.4], size=13, row_h=.41)
d.text(s, .8, bottom + .26, 11.5, 1.8, [
    "PulseDB is Boston (MIMIC-III) and Seoul (VitalDB) combined, 5,361 patients. The spec says to use the Seoul "
    "half only — hence the “VitalDB” filenames. It ships already split into five groups, and every published "
    "paper on PulseDB uses those same groups.",
    "",
    "Has anyone used both datasets together? One paper has (arXiv:2508.10805, 2025), but for removing motion "
    "noise and estimating heart rate — never blood pressure. The pairing exists; using it for blood pressure "
    "does not.",
], size=12.5, color=INK2, space_after=6)

if DETAILED:
    figure_slide("The data", "Three real people from each dataset",
                 "Same sensor idea, very different conditions.",
                 PRE / "08_three_subjects_each_dataset.png",
                 ["Left: hospital patients lying still, blood pressure known. Right: volunteers wearing a "
                  "watch while sitting, with no blood pressure recorded at all."],
                 top=2.2, bottom=1.0)

    figure_slide("The data", "Where the five group names came from",
                 "They are the dataset's, not ours.",
                 PRE / "07_where_splits_come_from.png",
                 ["We lower-cased the file names and changed nothing else. No patient was moved between groups."],
                 top=2.05, bottom=.9)

# ══════════════════════════════════════════════════════════════ pipeline
s = d.slide("Preprocessing", "The whole pipeline on one page")
d.image(s, PRE / "13_flowchart.png", .5, 1.72, 12.3, 5.5)

if DETAILED:
    figure_slide("Preprocessing", "Step 0 — what we actually downloaded",
                 "Opening the raw file and printing what is inside it.",
                 PRE / "01_raw_file_contents.png",
                 ["Two things to notice: ECG and PPG arrive already scaled between 0 and 1, and the pressure "
                  "signal arrives in real units."],
                 top=2.05, bottom=.9)

    figure_slide("Preprocessing", "Evidence — the file sizes quoted here",
                 "Printed off the machine.",
                 PRE / "09_disk_evidence.png",
                 ["19 GB of downloads become 3.6 GB: the pressure channel is dropped, and each number is "
                  "stored in half the space."],
                 top=2.05, bottom=.9)

# ══════════════════════════════════════════════════════════════ one clip
figure_slide("Preprocessing", "One real clip, every step",
             "Patient p003361_1, one 10-second clip, from file to stored array.",
             PRE / "02_pipeline_one_segment.png",
             ["Panel 2 is the important one — the green trace produced this clip's answer of 133 over 76, and "
              "is then thrown away. The model only ever sees the blue and orange lines.",
              "Panel 3 looks identical to panel 1 because PulseDB already scales these two channels. That step "
              "exists for PPG-DaLiA, which does not."],
             top=2.15, bottom=1.15)

figure_slide("Preprocessing", "Where the two numbers come from",
             "A blood pressure reading is the top and the bottom of one wave.",
             PRE / "03_label_derivation.png",
             ["Red triangle = the heart squeezing, the high number. Blue = the heart relaxing, the low number. "
              "Average them across the clip and you get 133 over 76.",
              "A thin tube inside the patient's artery measured this. Nobody estimated it — which is why these "
              "labels can be trusted, and why the model never sees the green line."],
             top=2.25, bottom=1.3)

if DETAILED:
    figure_slide("Preprocessing", "Every patient has a different pulse",
                 "Four patients. Left is the download, right is what we saved.",
                 PRE / "04_raw_vs_stored_subjects.png",
                 ["Four people, four clearly different pulse shapes — close to a fingerprint, which is exactly "
                  "why training and test groups must not share people.",
                  "Left and right match to within 0.00024."],
                 top=2.2, bottom=1.2)

    figure_slide("Preprocessing", "Evidence — the authors' own description",
                 "PulseDB's published preprocessing, quoted from the paper.",
                 PRE / "12_pulsedb_own_words.png",
                 ["Note the last exclusion rule: they compared PPG against the pressure trace “after "
                  "alignment”, so the authors knew the two channels do not share a clock."],
                 top=2.05, bottom=1.0)

# ══════════════════════════════════════════════════════════════ the contrast
s = d.slide("Preprocessing", "The two datasets arrive in very different states",
            "PulseDB comes finished. PPG-DaLiA does not — which is why it needed more work.")
bottom = d.table(s, .7, 2.4, 12.0, [
    ["Step", "PulseDB", "PPG-DaLiA", "So who did it?"],
    ["Sensors synchronised", "done by authors", "done by authors", "neither of us"],
    ["Signal filtered", "done — Chebyshev-II, 0.5–8 Hz", "not done", "nobody — see note"],
    ["Cut into fixed clips", "done — 10 s, non-overlapping", "not done", "we did, for DaLiA"],
    ["Bad clips removed", "done — three quality rules", "not done", "we score quality, keep all"],
    ["Signal rescaled", "done — 0 to 1", "not done", "we did, for DaLiA"],
    ["Sampling rate", "125 Hz", "64 Hz wrist, 700 Hz chest", "we resampled DaLiA to 125"],
    ["Blood pressure labels", "yes — from an arterial line", "none at all", "cannot be created"],
], col_w=[2.3, 3.4, 3.0, 3.0], size=11.5, row_h=.46)
d.text(s, .7, bottom + .28, 12.0, 1.5, [
    "PulseDB's authors shipped the data already filtered, segmented, quality-checked and labelled, so we left "
    "it exactly as it came. Filtering an already filtered signal removes real detail rather than cleaning it.",
    "PPG-DaLiA's watch does its own processing on the device and the authors ship what it produced, so we did "
    "not add a filter there either — the real-world test only means something if both are treated identically. "
    "Its readme suggests 8-second clips; we use 10 so the shape matches PulseDB.",
], size=12.5, color=INK2, space_after=5)

if DETAILED:
    figure_slide("Preprocessing", "PPG-DaLiA, the same clip through each step",
                 "Volunteer S1, ten seconds from ten minutes into the recording.",
                 PRE / "05_dalia_pipeline.png",
                 ["The watch records 640 numbers for these ten seconds; PulseDB expects 1250. Resampling "
                  "fills the difference without changing the pulse shape."],
                 top=2.25, bottom=1.1)

    figure_slide("Preprocessing", "Evidence — what was inside the PPG-DaLiA download",
                 "24 GB of sensors, of which this project uses five.",
                 PRE / "11_dalia_archive.png",
                 ["Most of it is the chest sensor suite from the stress-detection side of the study. The "
                  "archive is kept, so any dropped channel can be recovered."],
                 top=2.05, bottom=.9)

# ══════════════════════════════════════════════════════════════ output
s = d.slide("Preprocessing", "What we ended up with")
bottom = d.table(s, .8, 2.4, 11.5, [
    ["Group", "Clips", "Patients", "Per patient", "What it is for", "Size"],
    ["train", "465,480", "1,293", "60 min", "The model learns from this", "2.33 GB"],
    ["calfree_test", "57,600", "144", "67 min", "The real exam — brand new patients", "288 MB"],
    ["aami_cal", "70,212", "116", "101 min", "A few readings to tune the model", "351 MB"],
    ["aami_test", "666", "116", "1 min", "Then score it on those same people", "3 MB"],
    ["calbased_test", "51,720", "1,293", "7 min", "Easy case — patients it knows", "259 MB"],
    ["PPG-DaLiA", "64,682", "15", "144 min", "Watch data, no BP labels", "323 MB"],
], col_w=[2.0, 1.4, 1.2, 1.3, 4.0, 1.1], size=12, row_h=.44)
d.text(s, .8, bottom + .28, 11.5, 1.8, [
    "Each group is four files: the waveforms, the two BP numbers, which patient each clip came from, and a "
    "small table of age, sex, weight and signal quality. They line up row by row.",
    "",
    "One thing not to misread: PPG-DaLiA's clips are cut every 2 seconds and are 10 seconds long, so they "
    "overlap. Its recording really is 36 hours in total, not 180.",
], size=12.5, color=INK2, space_after=5)

if DETAILED:
    s = d.slide("Preprocessing", "What one finished group looks like",
                "The training group, file by file.")
    bottom = d.table(s, .8, 2.4, 11.5, [
        ["File", "What is in it", "Size"],
        ["signals.npy", "The waveforms — 465,480 clips, each 2 signals x 1250 numbers", "2,327 MB"],
        ["labels.npy", "The answers — 465,480 rows, an upper and a lower BP number", "3.7 MB"],
        ["subjects.npy", "Which patient each clip came from — 465,480 names", "29.8 MB"],
        ["meta.parquet", "Age, sex, height, weight, BMI and three signal-quality scores", "5.9 MB"],
    ], col_w=[1.8, 7.2, 1.4], size=13, row_h=.5)
    d.text(s, .8, bottom + .35, 11.5, 1.8, [
        "     clip 0    belongs to patient p000001_1,  a 48-year-old man,  blood pressure 131 / 88,",
        "               waveform 2 signals of 1250 numbers each.",
        "",
        "The waveform file is opened without loading it — a training run reads only the clips it is about to "
        "use, so a 2.3 GB file costs no memory to open.",
    ], size=13, color=INK2, space_after=5)

# ══════════════════════════════════════════════════════════════ checks
s = d.slide("Preprocessing", "Three things that could have gone wrong",
            "Each one was tested, not assumed.")
bottom = d.table(s, .7, 2.4, 12.0, [
    ["What could go wrong", "How we checked", "What came back"],
    ["The saved file might not match\nthe original",
     "Reloaded four clips from the download and compared\nthem number by number",
     "Same to within 0.00024\n— that is 0.02%"],
    ["Saving in a smaller format might\nhave damaged the signal",
     "Ran pulse detection on both versions and compared\nwhere it found the heartbeats",
     "Identical on all 300\nclips tested"],
    ["A patient used for training might\nturn up in a test group",
     "Compared the patient lists of every pair of groups",
     "Zero shared, and it is\nre-checked on every run"],
], col_w=[3.2, 5.4, 2.6], size=12.5, row_h=.84)
d.text(s, .7, bottom + .28, 12.0, 1.3, [
    "The third one matters most. If a training patient ever appeared in a test group, the results would "
    "quietly get better and nobody would notice — so the build stops instead.",
    "Two groups do share patients on purpose (aami_cal with aami_test, train with calbased_test). Both are by "
    "design, and neither is ever a headline result.",
], size=12.5, color=INK2, space_after=5)

# ══════════════════════════════════════════════════════════════ vs the papers
s = d.slide("Preprocessing", "Did we prepare the data the same way as the papers?",
            "Our settings against the two most relevant studies, and how much data each had.")
bottom = d.table(s, .7, 2.35, 12.0, [
    ["Setting", "PulseDB gives", "Benchmark 2025", "DMT 2026", "Us", "Same?"],
    ["Measurements per second", "125", "125", "125", "125", "Yes"],
    ["Clip length", "10 s / 1250", "10 s / 1250", "1250", "10 s / 1250", "Yes"],
    ["Cleaning / filtering", "already done", "kept as is", "not stated", "kept as is", "Yes"],
    ["Patient groups", "3 official ones", "official", "official", "official", "Yes"],
    ["Input signals", "ECG + PPG", "PPG only", "PPG only", "ECG + PPG", "No"],
    ["Rescaling", "0 to 1", "not stated", "z-score", "0 to 1", "No"],
    ["Training clips", "—", "418,986", "465,480", "465,480", "Yes"],
], col_w=[2.3, 1.9, 1.9, 1.5, 1.7, .9], size=11.5, row_h=.42)
d.text(s, .7, bottom + .28, 12.0, 1.5, [
    "Four of six settings match exactly, because they come with the dataset and nobody can change them. Our "
    "preprocessing is the standard preprocessing for this dataset, not something we invented.",
    "The two that differ are deliberate: we keep the ECG because the spec asks for it, and we use the 0-to-1 "
    "scaling the dataset already provides rather than adding another step.",
    "The last row matters: the best published result trained on 465,480 clips. We hold exactly the same number.",
], size=12.5, color=INK2, space_after=4)

# ══════════════════════════════════════════════════════════════ close
if not FULL:
    s = d.slide("Status", "Where this leaves us",
                "The data is ready. Nothing has been modelled yet.")
    d.text(s, .8, 2.5, 11.5, 4.3, [
        "Both datasets are downloaded, converted, checked, and stored in a form a model can train on.",
        "",
        "     1,553 patients from PulseDB, plus 15 volunteers from PPG-DaLiA",
        "     1,794 hours of hospital recording and 36 hours of watch recording",
        "     3.6 GB on disk, down from 22 GB of downloads",
        "     no training patient appears in any test group, checked on every run",
        "",
        "Clips from both datasets now have exactly the same shape, so a model trained on hospital data can be "
        "run on the watch data without a single change.",
        "",
        "Everything regenerates from scripts — delete the processed folder and one command rebuilds it.",
        "",
        "Next: train the first models, then compare against the published results on the same dataset.",
    ], size=14, color=INK2, space_after=6)

    if DETAILED:
        figure_slide("Appendix", "Raw output — opening a finished group", "",
                     PRE / "10_processed_folder.png", [""], top=1.95, bottom=.4)
        figure_slide("Appendix", "Raw output — the three checks", "",
                     PRE / "06_verification.png", [""], top=1.95, bottom=.4)

    # two columns: a single list of this length runs off the bottom of the slide
    s = d.slide("Sources", "References")
    d.text(s, .8, 2.45, 5.7, 4.5, [
        "THE TWO DATASETS",
        "",
        "PulseDB: a large, cleaned dataset based on MIMIC-III",
        "and VitalDB — Wang et al., Frontiers in Digital",
        "Health, 2023.",
        "doi.org/10.3389/fdgth.2022.1090854",
        "",
        "PPG-DaLiA — Reiss et al., UCI Machine Learning",
        "Repository, dataset 495.",
        "",
        "",
        "THE ONE PAPER USING BOTH OF THEM",
        "— for denoising and heart rate, not blood pressure",
        "",
        "Reduction of motion artifacts from PPG signals using",
        "learned convolutional sparse coding, Physiological",
        "Measurement, 2025.    arXiv:2508.10805",
    ], size=12, color=INK2, space_after=1)
    d.text(s, 6.9, 2.45, 5.6, 4.5, [
        "PAPERS THAT PREPARE THE SAME DATA THE SAME WAY",
        "",
        "Generalizable deep learning for PPG-based blood",
        "pressure estimation — a benchmarking study, 2025.",
        "arXiv:2502.19167",
        "",
        "DMT: Demographic Conditioning, Morphology-Enhanced",
        "Transformer, 2026.    arXiv:2606.11125",
        "",
        "Validation of deep learning models for cuffless BP",
        "estimation, 2024.  oaepublish.com/articles/chatmed.2023.23",
        "",
        "rU-Net, Multi-Scale Feature Fusion and Transfer",
        "Learning, IEEE JBHI, 2024.",
        "",
        "WHY THE SPLIT MATTERS",
        "",
        "Investigation of Data Leakage in Deep-Learning-Based",
        "Blood Pressure Estimation, IEEE, 2023.",
    ], size=12, color=INK2, space_after=1)

    d.save_pdf(DOCS / ("Preprocessing_Report_detailed.pdf" if DETAILED
                       else "Preprocessing_Report.pdf"))
    raise SystemExit

# ══════════════════════════════════════════════════════════════ literature (--full)
s = d.slide("Other research", "Six papers, and what each was trying to do")
bottom = d.table(s, .7, 2.4, 12.0,
                 [["Paper", "Year", "Its goal"]] +
                 [[p["short"], str(p["year"]), p["goal"]] for p in PAPER_DETAIL],
                 col_w=[2.4, .8, 8.0], size=12.5, row_h=.5)
d.text(s, .7, bottom + .3, 12.0, 1.0,
       "Their numbers differ partly because they were answering different questions, not because one method "
       "is better than another.", size=13, color=INK2)

s = d.slide("Other research", "Models used, and results on seen vs new patients")
bottom = d.table(s, .7, 2.4, 12.0,
                 [["Paper", "Model", "Patients it had seen", "Brand new patients"]] +
                 [[p["short"], p["model"], p["seen"], p["new"]] for p in PAPER_DETAIL],
                 col_w=[2.0, 4.2, 2.9, 2.9], size=11.5, row_h=.56, highlight=(6,))
d.text(s, .7, bottom + .3, 12.0, 1.4, [
    "Every paper that reports both shows the same pattern: around 4 to 9 mmHg on patients it had already seen, "
    "around 12 to 14 on new ones.",
    "That gap is the honest difficulty of the problem, and it is what this project set out to measure.",
], size=12.5, color=INK2, space_after=5)

s = d.slide("Other research", "Is there a paper using BOTH our datasets?",
            "Yes — one. The distinction matters for what we can claim.")
bottom = d.table(s, .8, 2.5, 11.5, [
    ["", BOTH_DATASETS["title"][:52] + "…"],
    ["Published", f"{BOTH_DATASETS['venue']}, {BOTH_DATASETS['year']}"],
    ["Its task", BOTH_DATASETS["task"]],
    ["Uses PulseDB for", BOTH_DATASETS["pulsedb"]],
    ["Uses PPG-DaLiA for", BOTH_DATASETS["dalia"]],
    ["Blood pressure?", BOTH_DATASETS["bp"]],
], col_w=[2.3, 9.0], size=12.5, row_h=.52)
d.text(s, .8, bottom + .3, 11.5, 1.6, [
    "So the pairing has been done — but for cleaning up noisy signals and estimating heart rate, not for blood "
    "pressure.",
    "",
    "Our claim has to be precise: nobody has trained a blood-pressure model on PulseDB and tested it on "
    "PPG-DaLiA. Saying “nobody has used these two together” would be wrong, and a reviewer would find this.",
], size=13, color=INK2, space_after=5)

figure_slide("Accuracy", "Our result against the published ones",
             "Tested on brand new patients. Lower is better.",
             LIT / "01_calibration_free_comparison.png",
             ["Ours is the orange bar — gradient boosting on 60 hand-measured features, with no tuning.",
              "0.76 mmHg behind the best Transformer. Four years of new architectures have moved this number "
              "by 1.5 mmHg in total."],
             top=2.3, bottom=1.35)

figure_slide("Accuracy", "The biggest factor is not the model",
             "Same models, same data — only which patients are in the test set changes.",
             LIT / "02_protocol_effect.png",
             ["4.5 mmHg when the test patients had been seen in training.   12.9 when they are new.   "
              "19.3 on the hardest test set."],
             top=2.25, bottom=1.05)

s = d.slide("Our training", "What we have actually trained so far",
            "Being precise, because the word “training” is easy to overstate.")
bottom = d.table(s, .8, 2.5, 11.5, [
    ["Model", "Type", "Status", "Fits run"],
    ["HistGradientBoosting", "Gradient boosting", "Trained", "18"],
    ["Random Forest", "Decision trees", "Trained", "6"],
    ["1D CNN", "Deep learning", "Code ready, not run", "0"],
    ["ResNet1D", "Deep learning", "Code ready, not run", "0"],
    ["Transformer", "Deep learning", "Code ready, runs on Colab", "0"],
], col_w=[2.8, 2.4, 3.2, 1.3], size=13, row_h=.5, highlight=(1, 2))
d.text(s, .8, bottom + .35, 11.5, 1.4, [
    "24 models fitted so far. Each run trains 6 — one for the upper and one for the lower BP number, across "
    "the three ways of splitting the patients.",
    "No deep model has been trained yet.",
], size=13, color=INK2, space_after=5)

s = d.slide("Sources", "References")
d.text(s, .8, 2.5, 11.5, 4.3, [
    "PulseDB — Wang et al., Frontiers in Digital Health, 2023.    doi.org/10.3389/fdgth.2022.1090854",
    "PPG-DaLiA — Reiss et al., UCI Machine Learning Repository, dataset 495.",
    "",
    "Generalizable deep learning for PPG-based BP estimation — a benchmarking study, 2025.    arXiv:2502.19167",
    "DMT: Demographic Conditioning, Morphology-Enhanced Transformer, 2026.    arXiv:2606.11125",
    "Validation of deep learning models for cuffless BP estimation, 2024.",
    "rU-Net, Multi-Scale Feature Fusion and Transfer Learning, IEEE JBHI, 2024.",
    "",
    "Investigation of Data Leakage in Deep-Learning-Based Blood Pressure Estimation, IEEE, 2023.",
    "Reduction of motion artifacts from PPG signals using learned convolutional sparse coding, 2025.",
    "     arXiv:2508.10805",
], size=12.5, color=INK2, space_after=3)

d.save_pdf(DOCS / "Project_B_Report.pdf")
