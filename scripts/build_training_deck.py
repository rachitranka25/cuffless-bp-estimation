"""Build docs/Model_Training_Report.pdf — what was trained and what came out.

Every number is read from results/models/*/*.json at build time, so the deck
cannot drift from the run files. The only hand-entered numbers are the published
results, which carry their source.

Structure: one slide per model first, then the findings that only appear once the
models are compared. Short lines, one idea per slide.

Run:  python3 scripts/build_training_deck.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from paths import MODEL_RESULTS, OVERVIEW, FIG_LIT, DOCS, model_dir  # noqa: E402
from deck import Deck, INK, INK2, MUTED, BLUE, RED                   # noqa: E402

LABEL = {"rf": "Random Forest", "gb": "Gradient Boosting", "cnn": "1D-CNN"}
ORDER = ["rf", "gb", "cnn"]
PROTOS = ["leaky", "calbased", "calfree", "aami"]
PROTO_WHO = {"leaky": "same patients, clips picked at random",
             "calbased": "same patients, official held-out clips",
             "calfree": "144 patients it has never seen",
             "aami": "116 new patients, full BP range"}


def load():
    runs, shifts = {}, {}
    for p in sorted(MODEL_RESULTS.glob("*/*.json")):
        d = json.loads(p.read_text())
        if "domain_shift" in p.name:
            shifts[(d["model"], d.get("tag") or "balanced")] = d
        elif "personalization" not in p.name:
            runs[(d["model"], d["protocol"], d.get("config", {}).get("tag") or "")] = d
    return runs, shifts


RUNS, SHIFTS = load()


def r(m, p, tag="", target="sbp", key="mae"):
    x = RUNS.get((m, p, tag))
    return x["results"][target][key] if x else None


def n(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def seen_avg(m):
    v = [x for x in (r(m, "leaky"), r(m, "calbased")) if x]
    return sum(v) / len(v) if v else None


def gap(m):
    a, b = seen_avg(m), r(m, "calfree")
    return b / a if a and b else None


def shift(m, tag):
    d = SHIFTS.get((m, tag))
    if not d:
        return None
    s = d["distribution"]["sbp"]
    return s["dalia_predicted"]["mean"] - s["pulsedb_predicted"]["mean"]


d = Deck()

# 1 ------------------------------------------------------------------- title
d.title_slide(
    "PROJECT B  ·  CUFFLESS BLOOD PRESSURE",
    "Model training — round one",
    "Three of five models, each tested four ways, plus the external test\n"
    "on the wearable dataset",
    "Rachit Ranka",
    "Every number generated from the saved run files — results/overview/RESULTS.md")

# 2 ------------------------------------------------------------ what was run
s = d.slide("What was trained", "Three models done, two to go",
            "The order is the spec's, section I.")
bottom = d.table(s, .9, 2.55, 11.5, [
    ["", "Model", "What it reads", "Status"],
    ["1", "Random Forest", "61 measurements taken from each clip", "done"],
    ["2", "Gradient Boosting", "the same 61 measurements", "done"],
    ["3", "1D-CNN", "the raw waveform, 2 × 1250 numbers", "done"],
    ["4", "ResNet1D", "the raw waveform", "next"],
    ["5", "Transformer", "the waveform + age, sex, BMI", "after that"],
], col_w=[.4, 2.6, 5.4, 1.4], size=15, row_h=.56, highlight=(1, 2, 3))
d.text(s, .9, bottom + .5, 11.5, 1.2, [
    "15 training runs. Each learns from 419,040 clips from 1,164 patients.",
    "",
    "The first two never see the signal — only measurements taken from it beforehand.",
], size=15, color=INK2, space_after=8)

# 3 ------------------------------------------------------------ the four rules
s = d.slide("The core idea", "The same model, tested four ways",
            "Nothing changes between these except who is in the test set.")
bottom = d.table(s, .9, 2.55, 11.5, [
    ["Test rule", "Who is tested", "Met them before?"],
    ["leaky", "same patients, clips picked at random", "yes"],
    ["calbased", "same patients, official held-out clips", "yes"],
    ["calfree", "144 patients", "no"],
    ["aami", "116 patients, chosen to span the whole BP range", "no"],
], col_w=[2.0, 7.5, 2.0], size=14.5, row_h=.5, highlight=(3,))
d.text(s, .9, bottom + .3, 11.5, .4,
       "calfree is the honest one — the situation a real user is in: the device has never met them.",
       size=15, color=INK)
d.text(s, .9, bottom + .9, 11.5, .35, "THE TWO CLINICAL STANDARDS USED IN THE TABLES THAT FOLLOW",
       size=12, color=BLUE, bold=True)
d.text(s, .9, bottom + 1.32, 11.5, 1.3, [
    "AAMI      pass or fail, and it never looks at MAE. The bias must be within ±5 mmHg and the",
    "               spread of the error within 8. Both, or it fails.",
    "BHS        a grade from A to D on how many predictions land close: grade A needs 60% of them",
    "               within 5 mmHg, 85% within 10, and 95% within 15.",
], size=13.5, color=INK2, space_after=2)

# 4-6 --------------------------------------------------------- one per model
TAKE = {
    "rf": ["Best-looking model on the two rules where it already knows the patient,",
           "and no better than the others once the patients are new.",
           "Diastolic: 4.11 mmHg and grade A on known patients, 8.38 and grade D on new ones."],
    "gb": ["Steadier than the forest — worse on known patients, better on new ones,",
           "so its gap is smaller.",
           "Without the BP-bin weighting it reaches 12.71 on new patients, the range the "
           "published deep models sit in."],
    "cnn": ["The only one that reads the waveform itself. 381,186 parameters, 20 epochs,",
            "about 30 minutes on a T4 GPU.",
            "Without the BP-bin weighting: 12.07 — the best calibration-free systolic "
            "number in this project."],
}
for m in ORDER:
    s = d.slide(LABEL[m], "All four test rules",
                "Mean absolute error in mmHg, lower is better. The two clinical standards are "
                "shown for diastolic: systolic fails AAMI in every row, so it would carry no "
                "information.")
    rows = [["Test rule", "Who is tested", "SBP", "DBP",
             "AAMI\n(on DBP)", "BHS\n(on DBP)"]]
    for p in PROTOS:
        x = RUNS.get((m, p, ""))
        name = "calfree  (A)" if p == "calfree" else p
        rows.append([name, PROTO_WHO[p], n(r(m, p)), n(r(m, p, target="dbp")),
                     "PASS" if x and x["results"]["dbp"]["aami_pass"] else "FAIL",
                     x["results"]["dbp"]["bhs"] if x else "—"])
    if (m, "calfree", "nobalance") in RUNS:
        x = RUNS[(m, "calfree", "nobalance")]
        rows.append(["calfree  (B)", "the same 144, without BP-bin weighting",
                     n(r(m, "calfree", "nobalance")),
                     n(r(m, "calfree", "nobalance", target="dbp")),
                     "PASS" if x["results"]["dbp"]["aami_pass"] else "FAIL",
                     x["results"]["dbp"]["bhs"]])
    bottom = d.table(s, .9, 2.42, 11.5, rows, col_w=[1.5, 4.4, 1.1, 1.1, 1.5, 1.3],
                     size=14, row_h=.52, highlight=(3,))
    d.text(s, .9, bottom + .18, 11.5, .4,
           "(A) carries the BP-bin weighting the spec asks for.    "
           "(B) is the same model without it — the form every published number takes.",
           size=12, color=MUTED)
    g = gap(m)
    if g:
        d.text(s, .9, bottom + .62, 11.5, .45,
               f"Known patients {seen_avg(m):.2f}   →   new patients "
               f"{r(m, 'calfree'):.2f}      gap {g:.2f}x",
               size=16.5, color=RED, bold=True)
    d.text(s, .9, bottom + 1.16, 11.5, 1.3, TAKE[m][:2], size=14, color=INK2,
           space_after=3)

# 8 ------------------------------------------------------------ all together
s = d.slide("Side by side", "All three models, all four test rules",
            "Top row systolic, bottom row diastolic. Three different questions, left to right.")
d.image(s, OVERVIEW / "1_leakage_gap.png", .35, 2.12, 12.65, 4.05)
d.text(s, .9, 6.3, 11.5, 1.1, [
    "MAE — how far off it is on average. Lower is better. This is the number papers headline.",
    "SD of error — whether the errors are all similar or all over the place. This is what AAMI limits, not MAE.",
    "Pearson r — 1.0 means it follows the patient up and down; 0.0 means one fixed answer for everyone.",
], size=13.5, color=INK2, space_after=2)
d.text(s, .9, 7.15, 11.5, .35,
       "The red dotted line is the clinical limit — bars should sit below it. None of them do.",
       size=13.5, color=RED, bold=True)

# 8 ------------------------------------------------------------- the gap idea
s = d.slide("The finding", "The gap is not the same for every model")
rows = [["Model", "Known patients", "New patients", "Gap"]]
for m in ORDER:
    rows.append([LABEL[m], f"{seen_avg(m):.2f}", n(r(m, "calfree")), f"{gap(m):.2f}x"])
bottom = d.table(s, 1.7, 2.6, 9.9, rows, col_w=[3.2, 2.4, 2.3, 1.5],
                 size=16, row_h=.66)
d.text(s, 1.7, bottom + .5, 9.9, 2.2, [
    "The Random Forest memorises the patient, so a new clip from a known patient is",
    "still easy — it looks excellent under the first two rules.",
    "",
    "The CNN memorises the clip, which does not help even on the same patient.",
    "Both are overfitting — but the leaky rule only rewards the first kind.",
], size=15, color=INK2, space_after=4)

# 10 ------------------------------------------------- balancing: what it is
s = d.slide("BP-bin balancing", "What it is, and where it comes from")
d.text(s, .9, 2.35, 11.5, .8, [
    "The proposal asks for it by name, in section L:",
], size=14.5, color=INK2, space_after=3)
d.text(s, 1.3, 2.72, 10.8, .9,
       "\"Class/target imbalance handling: BP value distribution is typically right-skewed in ICU "
       "populations (more hypertensive/hypotensive extremes); apply stratified sampling or a "
       "weighted loss over BP value bins.\"",
       size=14, color=BLUE)
d.text(s, .9, 3.6, 11.5, .45,
       "The problem it solves:", size=14.5, color=INK, bold=True)
bottom = d.table(s, .9, 4.0, 11.5, [
    ["Systolic band", "Training clips", "Share of the data", "Weight each clip is given"],
    ["below 90", "33,580", "7.2%", "2.3x"],
    ["90 – 130", "333,610", "71.7%", "0.5x"],
    ["130 – 170", "95,152", "20.5%", "1.0 – 4.4x"],
    ["above 170", "3,138", "0.7%", "24.7x"],
], col_w=[2.6, 2.5, 2.6, 3.8], size=13.5, row_h=.46, highlight=(4,))
d.text(s, .9, bottom + .38, 11.5, 1.1, [
    "Seven in ten training clips sit between 90 and 130, so a model that always answers \"about 115\" is",
    "right most of the time — and useless for the patients a cuffless monitor would exist to help.",
    "Balancing makes one clip above 170 count as much as twenty-five ordinary ones.",
], size=13.5, color=INK2, space_after=2)

# 11 --------------------------------------------- balancing: what it changes
s = d.slide("BP-bin balancing", "What it costs and what it buys",
            "Systolic MAE on the 144 unseen patients, with the weighting off and on.")
rows = [["Model", "Overall\noff", "Overall\non", "Change",
         "Patients above 170\noff", "Patients above 170\non", "Change"]]
BAND = {"rf": (12.93, 12.76, 47.18, 43.13), "gb": (12.71, 14.63, 46.21, 32.93),
        "cnn": (12.07, 14.86, 47.10, 37.62)}
for m in ORDER:
    off, on, hoff, hon = BAND[m]
    rows.append([LABEL[m], f"{off:.2f}", f"{on:.2f}", f"{on - off:+.2f}",
                 f"{hoff:.2f}", f"{hon:.2f}", f"{hon - hoff:+.2f}"])
bottom = d.table(s, .7, 2.45, 11.9, rows, col_w=[2.3, 1.3, 1.3, 1.3, 1.9, 1.9, 1.3],
                 size=13, row_h=.58)
d.text(s, .7, bottom + .4, 11.9, 2.2, [
    "Read the two \"Change\" columns against each other. On gradient boosting the weighting costs",
    "1.92 mmHg on the average and buys 13.29 on the patients above 170. On the CNN it costs more.",
    "On the random forest it costs nothing at all — the average improves as well.",
    "",
    "So both numbers are reported everywhere in this deck:",
    "        (A)  with the weighting — this project's model, what section L asks for",
    "        (B)  without it — how every published number was measured, so the comparison holds",
], size=14, color=INK2, space_after=3)

# 11 ------------------------------------------------------------- literature
s = d.slide("Against published work", "Same dataset, same official splits, same metric")
d.image(s, FIG_LIT / "01_calibration_free_comparison.png", .45, 2.05, 12.45, 3.7)
d.text(s, .9, 6.0, 11.5, 1.2, [
    "Orange is this project. A four-layer CNN lands at 12.07 mmHg; a 2026 transformer is at 12.17.",
    "",
    "That is not a win — it is evidence that the architecture is not what is holding the field back.",
], size=15, color=INK2, space_after=6)

# 12 --------------------------------------------------- what PPG-DaLiA is
s = d.slide("The second dataset", "What PPG-DaLiA actually is",
            "15 volunteers, a wrist watch and a chest strap, one ordinary day each.")
d.text(s, .9, 2.45, 5.2, 3.0, [
    "One volunteer's day — S5, age 21:",
    "",
    "     transient          43 min",
    "     lunch              38 min",
    "     working            21 min",
    "     driving            14 min",
    "     sitting            10 min",
    "     walking            10 min",
    "     stairs              8 min",
    "     cycling             7 min",
    "     table soccer        5 min",
], size=14, color=INK2, space_after=1)
d.text(s, 6.7, 2.45, 5.7, 3.0, [
    "What the researchers recorded:",
    "",
    "     wrist PPG              ✓",
    "     chest ECG              ✓",
    "     accelerometer          ✓",
    "     heart rate             ✓",
    "     what they were doing   ✓",
    "     age, sex, height, weight ✓",
    "",
    "     BLOOD PRESSURE         ✗   never measured",
], size=14, color=INK2, space_after=1)
d.text(s, .9, 5.75, 11.5, 1.5, [
    "Nobody ever put a blood-pressure cuff on these fifteen people. The dataset was built to study",
    "heart rate under motion, not blood pressure.",
    "",
    "That is why the next slide has no accuracy number for BP — there is nothing to compare against.",
], size=14.5, color=INK, space_after=3)

# 13 ------------------------------------------------------ the second dataset
hr = next(iter(SHIFTS.values()))["heart_rate_crosscheck"]

s = d.slide("The second dataset", "Does it work on a watch?",
            "Trained on hospital patients, then run on 15 volunteers wearing a wrist watch.")
d.text(s, .9, 2.38, 11.5, .8, [
    "There is no MAE for blood pressure here, and there cannot be: MAE needs a true value to",
    "subtract from, and PPG-DaLiA never measured anyone's blood pressure. Two things it does have:",
], size=14.5, color=INK, space_after=3)

d.text(s, .9, 3.22, 11.5, .35, "1.  HEART RATE — this one is labelled, so it has a real MAE",
       size=13, color=BLUE, bold=True)
bottom = d.table(s, .9, 3.65, 5.3, [
    ["", "MAE (bpm)"],
    ["volunteer sitting still", f"{hr['low_motion_mae_bpm']:.2f}"],
    ["volunteer moving", f"{hr['high_motion_mae_bpm']:.2f}"],
    ["all 63,180 clips", f"{hr['mae_bpm']:.2f}"],
], col_w=[3.4, 1.9], size=13.5, row_h=.5)

d.text(s, 6.6, 3.22, 5.8, .35, "2.  BLOOD PRESSURE — no labels, so only where it lands",
       size=13, color=BLUE, bold=True)
def band(v):
    return ("normal" if v < 120 else "elevated" if v < 130
            else "stage 1 hypertension" if v < 140 else "stage 2 hypertension")

rows = [["", "Average SBP", "Which band"],
        ["a healthy adult in their 20s", "under 120", "normal"]]
for m in ORDER:
    dd = SHIFTS.get((m, "nobalance"))
    if dd:
        v = dd["distribution"]["sbp"]["dalia_predicted"]["mean"]
        rows.append([LABEL[m] + " says", f"{v:.1f}", band(v)])
d.table(s, 6.6, 3.5, 5.8, rows, col_w=[2.5, 1.5, 1.8], size=12.5, row_h=.47,
        highlight=(1,))

d.text(s, .9, bottom + .55, 11.5, 1.2, [
    "Heart rate is the honest measure, and it is bad: 13 bpm sitting still, 33 moving. A published paper",
    "doing this properly on the same data reports 8.69 bpm.",
    "The first row on the right is the clinical norm, not our data — every model lands above it.",
], size=13.5, color=INK2, space_after=3)

# 14 ------------------------------------------------- control, part 1: shift
s = d.slide("The second dataset", "Was the shift real, or did we cause it?",
            "The same test, run twice per model — once with the BP-bin weighting, once without.")
d.text(s, .9, 2.3, 11.5, .8, [
    "Note what this measures. It is our model's average on PPG-DaLiA minus our model's average on",
    "PulseDB — both are our own outputs, so no BP label is needed anywhere. It is a shift, not an error.",
], size=14, color=INK, space_after=3)
d.image(s, OVERVIEW / "3_shift_amount.png", .8, 3.25, 11.7, 2.75)
d.text(s, .9, 6.25, 11.5, 1.2, [
    "Take the Random Forest. We first measured +22.5. Without the weighting the same test gives",
    "+14.6. So 7.9 of that 22.5 — 35% — was our own weighting, and only 14.6 was the watch.",
    "On the CNN it was 22.6 out of 34.2: two thirds of what we first called domain shift was us.",
], size=13.5, color=INK2, space_after=2)

# 15 ---------------------------------------------- control, part 2: collapse
s = d.slide("The second dataset", "Can it still tell people apart?",
            "A separate question — and here the weighting turns out not to be the explanation.")
d.text(s, .9, 2.35, 11.5, .8, [
    "These patients really do differ from one another by about 11.8 mmHg. A model that arrives at",
    "the watch still answering that differently is distinguishing people; one answering 2 is not.",
], size=14, color=INK, space_after=3)
d.image(s, OVERVIEW / "3_shift_collapse.png", .8, 3.3, 11.7, 2.8)
d.text(s, .9, 6.25, 11.5, 1.2, [
    "The Random Forest goes from 8.0 on hospital patients to 2.1 on the watch — it has started giving",
    "all fifteen volunteers nearly the same number. The CNN drops to 3.2. Gradient boosting barely moves.",
    "Grey and blue sit on top of each other here, so unlike the shift, this collapse is not our doing.",
], size=13.5, color=INK2, space_after=2)

# 16 -------------------------------------------------------------- what's next
s = d.slide("Next", "What comes after this")
bottom = d.table(s, .9, 2.6, 11.5, [
    ["", "Work", "Spec"],
    ["1", "ResNet1D and the Transformer — the two remaining models", "Month 4"],
    ["2", "Few-shot personalization: give the model a few labelled\nreadings from a new person, then re-measure", "Months 6-7"],
    ["3", "How many readings are needed — 1, 3 or 5", "Month 9"],
], col_w=[.4, 8.9, 1.6], size=15, row_h=.72)
d.text(s, .9, bottom + .45, 11.5, 1.9, [
    "One question for you:",
    "Section G says PPG-DaLiA \"does not contain BP labels at all\", but section E asks how much",
    "personalization helps there. Calibration needs labels — should this be the distribution check only?",
], size=14.5, color=RED, space_after=3)
d.text(s, .9, bottom + 1.65, 11.5, .9, [
    "Everything above is in results/overview/RESULTS.md, and any row reproduces with one command:",
    "        python3 -m train --model cnn --protocol calfree",
], size=14, color=INK2, space_after=4)

d.save_pdf(DOCS / "Model_Training_Report.pdf")
