"""Builds the ISGJ (Gerontechnology journal) manuscript from project results.

Format follows https://journal.gerontechnology.org/Guidelines.aspx:
  - Structure: Introduction, Methods, Results, Discussion(s), Conclusion(s)
  - Structured abstract, <=300 words (Background, Research Aim, Methods,
    Results, Conclusion)
  - 4-5 keywords
  - APA citations, single-spaced, 12pt
  - Tables + illustrations combined: no more than 6
  - Full research article: <=7000 words

Produces two files (the journal requires the title page as a SEPARATE
upload, for blind review):
  docs/Journal_Paper_ISGJ.docx / .pdf        — blinded manuscript body
  docs/Journal_Paper_ISGJ_TitlePage.docx     — title page, fill in brackets

Run: python3 scripts/build_journal_paper.py
"""
import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Pt, Mm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIG_GAP = RESULTS / "overview" / "1_leakage_gap.png"
FIG_LIT = RESULTS / "dataset" / "literature" / "01_calibration_free_comparison.png"
FIG_PERS = RESULTS / "overview" / "5_personalization.png"
FIG_SCHEMATIC = RESULTS / "overview" / "paper_protocol_schematic.png"
FIG_BAND = RESULTS / "overview" / "paper_weighting_all_models.png"
FIG_SHIFT = RESULTS / "overview" / "3_shift_amount.png"

FONT = "Times New Roman"
TABLE_FONT = "Arial"


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #

def set_base_style(doc):
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = Pt(12)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.0
    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), FONT)


def h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(13)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(12)
    return p


def para(doc, text, *, bold=False, italic=False, size=12, align=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    return p


def caption(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    r1 = p.add_run(f"{label}. ")
    r1.bold = True
    r1.font.size = Pt(10)
    r2 = p.add_run(text)
    r2.font.size = Pt(10)
    r2.italic = True
    return p


def make_table(doc, rows, col_widths_mm=None, header=True):
    nrow, ncol = len(rows), len(rows[0])
    t = doc.add_table(rows=nrow, cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.name = TABLE_FONT
            r.font.size = Pt(9)
            r.bold = header and i == 0
            p.paragraph_format.space_after = Pt(2)
    if col_widths_mm:
        for j, w in enumerate(col_widths_mm):
            for i in range(nrow):
                t.cell(i, j).width = Mm(w)
    return t


def add_figure(doc, path, width_mm=140):
    doc.add_picture(str(path), width=Mm(width_mm))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER


def word_count_of(doc):
    n = 0
    for p in doc.paragraphs:
        n += len(p.text.split())
    return n


# --------------------------------------------------------------------------- #
# manuscript body
# --------------------------------------------------------------------------- #

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(1)
set_base_style(doc)

TITLE = ("Evaluation-Protocol Leakage in Cuffless Blood Pressure Estimation from "
         "PPG: How Much It Inflates Accuracy, and How Much Brief Personalization "
         "Recovers, Across Five Model Architectures")

para(doc, TITLE, bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para(doc, "[Author names and affiliations removed for blind review; see separate "
          "title page]", italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER,
     space_after=16)

# --------------------------------------------------------------------------- #
h1(doc, "Abstract")

abstract_parts = [
    ("Background.", "Photoplethysmography (PPG)-based cuffless blood "
     "pressure (BP) estimation is often evaluated with splits that allow "
     "clips from the same patient on both sides of training and testing, "
     "inflating apparent accuracy relative to the scenario that matters for "
     "deployment: a genuinely new user, such as an older adult wearing a "
     "consumer watch for ongoing BP awareness."),
    ("Research aim.", "How much does evaluation-protocol leakage inflate "
     "accuracy; is the inflation architecture-dependent; what does "
     "correcting BP-value class imbalance cost and buy per architecture; "
     "and how much does brief few-shot personalization recover, including "
     "under a calibration/evaluation split genuinely separated in time "
     "rather than only in clip identity?"),
    ("Methods.", "Five models spanning classical (Random Forest, Gradient "
     "Boosting) and deep (1D-CNN, ResNet1D, Transformer) architectures were "
     "trained on PulseDB's VitalDB subset (1,164 subjects) and evaluated "
     "under four protocols differing only in which patients/clips appear "
     "in the test set, with PPG-DaLiA as an external, label-free check. "
     "Personalization fit a per-subject affine correction from k = 1-25 "
     "calibration clips, scored on storage-order and chronological "
     "held-out clips."),
    ("Results.", "The calibration-free accuracy gap, relative to "
     "same-patient testing, ranged from 1.09x (Transformer) to 2.28x "
     "(ResNet1D), confirming architecture-dependence. BP-bin weighting "
     "recovered 4-18 mmHg in the rare, critical high-BP range, costing "
     "0.2-2.8 mmHg mean error depending on architecture. Twenty-five "
     "calibration clips cut diastolic MAE by a fifth to a half, within "
     "0.3 mmHg of each model's physiological ceiling; systolic spread "
     "stayed above the AAMI threshold regardless, and under chronological "
     "calibration no model met that threshold at any size tested."),
    ("Conclusions.", "Evaluation-protocol choice changes cuffless-BP "
     "accuracy by more than published architecture progress on this "
     "benchmark, and that effect is itself architecture-dependent. "
     "Personalization recovers most of the physiologically recoverable "
     "diastolic error but not "
     "systolic spread, and part of its apparent benefit reflects short-term "
     "signal continuity rather than stable calibration."),
]
for label, text in abstract_parts:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r1 = p.add_run(label + " ")
    r1.bold = True
    r2 = p.add_run(text)

para(doc, "Keywords: cuffless blood pressure; photoplethysmography; "
          "evaluation leakage; personalization; wearable sensors",
     italic=True, size=11, space_after=16)

# --------------------------------------------------------------------------- #
h1(doc, "Introduction")

para(doc, "Hypertension affects an estimated 1.3 billion adults worldwide, "
          "yet fewer than a quarter have it adequately controlled (World "
          "Health Organization [WHO], 2023), and its prevalence rises "
          "sharply with age — from roughly 27% below age 60 to over 74% "
          "above age 80 (Oliveros et al., 2020). The population most "
          "burdened by "
          "hypertension is also the population least served by cuff-based "
          "self-monitoring: the number of adults aged 60 and over is "
          "projected to double, from roughly 1 billion in 2020 to 2.1 "
          "billion by 2050 (WHO, 2025), and gerontechnology research "
          "consistently identifies unobtrusive, continuous physiological "
          "monitoring as central to supporting independent aging in place "
          "(Olmedo-Aguirre et al., 2022). A wrist-worn device capable of "
          "estimating blood pressure (BP) from photoplethysmography (PPG) "
          "— the same optical pulse sensor already present in most consumer "
          "smartwatches — would let an older adult track BP trends during "
          "ordinary daily activity, without the burden of a cuff, addressing "
          "a documented gap between measurement frequency and the clinical "
          "value of trend awareness in this population.")

para(doc, "The physiological basis for this approach is well established: "
          "the time it takes a pressure pulse to travel between two arterial "
          "sites (pulse transit time, PTT) is related to arterial stiffness "
          "and, through it, to BP (Mukkamala et al., 2015), and a PPG "
          "waveform's morphology — not only its timing relative to a "
          "second sensor — carries additional information correlated with "
          "BP (Elgendi et al., 2019). Translating this relationship into an "
          "accurate estimator has become predominantly a machine-learning "
          "problem over the past decade, with reported architectures "
          "spanning classical regression on hand-crafted pulse-wave "
          "features, convolutional networks operating on the raw waveform, "
          "U-Net-style encoder-decoders, and most recently transformers "
          "conditioned on patient demographics (Arjomand et al., 2024; Chen "
          "et al., 2024; Huang et al., 2024; Shen et al., 2026). Reported "
          "mean absolute errors (MAE) on large public benchmarks span "
          "roughly 4-13 mmHg depending on the study (Huang et al., 2024; "
          "Moulaeifard et al., 2025; Shen et al., 2026). However, as this "
          "study demonstrates, the evaluation protocol underlying these "
          "numbers varies sharply across studies, and protocol choice alone "
          "can move a single model's reported MAE by more than the apparent "
          "improvement from several years of published architecture "
          "progress on the same benchmark.")

para(doc, "Within this landscape, the five architectures compared here "
          "sit at different points. Tree ensembles on hand-crafted "
          "features remain a competitive, GPU-free baseline in several "
          "studies despite their simplicity. Convolutional and residual "
          "networks on the raw waveform are the most widely benchmarked "
          "deep family: Moulaeifard et al. (2025) compare several such "
          "architectures (including XResNet1d101, used as a reference "
          "point below) under both calibrated and calibration-free "
          "evaluation on this same VitalDB subset, reporting a similar "
          "calibrated-versus-uncalibrated gap for a single architecture "
          "to the one this study measures across five. Encoder-decoder "
          "and attention-augmented hybrids (Chen et al., 2024; Huang et "
          "al., 2024) add cross-scale or cross-channel feature fusion on "
          "top of a convolutional backbone, and Chen et al.'s (2024) "
          "rU-Net additionally reports fine-tuning on a new subject's own "
          "labelled segments — the closest published precedent for the "
          "personalization step evaluated here, though without checking "
          "calibration/evaluation temporal separation. Transformer "
          "architectures conditioned on patient demographics (Arjomand et "
          "al., 2024; Shen et al., 2026) report the strongest published "
          "calibration-free numbers on this benchmark to date, which is "
          "part of why the Transformer's full evaluation across all four "
          "protocols — completed only in the latter stage of this study, "
          "see Discussion — mattered enough to revise this study's own "
          "working conclusion about which architecture has the smallest "
          "leakage gap.")

para(doc, "The central methodological hazard is patient-level leakage: if "
          "clips from a given patient's recording appear on both sides of a "
          "train/test split, a model can partially learn that patient's "
          "individual PPG-to-BP mapping rather than the general "
          "physiological relationship, and its reported accuracy on \"new\" "
          "clips from that same patient does not reflect its accuracy on a "
          "genuinely new patient. This is not unique to blood pressure "
          "estimation: a review of machine-learning-based science found "
          "leakage of this general kind in at least 294 papers across 17 "
          "scientific fields, frequently sufficient to overturn a paper's "
          "central claim once corrected (Kapoor & Narayanan, 2023). Within "
          "cuffless BP specifically, a recent methodological review "
          "recommends that every study report performance stratified by "
          "whether test subjects were seen during training, alongside "
          "several other standardized evaluation steps, as a minimum bar for "
          "cross-study comparability (Elgendi et al., 2024). PulseDB (Wang "
          "et al., 2022), the benchmark used throughout this study, ships "
          "three official protocols precisely to make the seen/unseen "
          "distinction testable: a leaky split in which both patient and "
          "clips may overlap between training and testing, a "
          "calibration-based split in which test patients were seen during "
          "training but the specific test clips were held out, and a "
          "calibration-free split in which the test patients never appear "
          "in training at all. A fourth protocol, built by PulseDB "
          "specifically to meet the Association for the Advancement of "
          "Medical Instrumentation's (AAMI) sampling requirements under the "
          "AAMI/ESH/ISO universal validation standard (Stergiou et al., "
          "2018), selects an additional unseen-patient cohort spanning the "
          "full BP range. Despite this infrastructure, many published "
          "cuffless-BP comparisons report a single accuracy number without "
          "always making clear which of these regimes it reflects, making "
          "it difficult for a reader to judge how much of a reported gain "
          "reflects a genuinely better model rather than a more lenient "
          "test.")

para(doc, "A second, related methodological question concerns "
          "personalization. The motivating use case for this class of device "
          "— an older adult who wants ongoing BP awareness without a cuff — "
          "plausibly tolerates a short calibration step against a reference "
          "device, after which the wearable operates calibration-free. Prior "
          "work fine-tuning on a small number of a new subject's own "
          "labelled segments (Chen et al., 2024) reports substantial error "
          "reduction, but does not test whether the calibration segments "
          "selected are drawn close in time to the segments being scored — "
          "raising the possibility that apparent personalization gains "
          "partly reflect short-term signal continuity (the same "
          "within-session physiological state or sensor-contact condition) "
          "rather than a stable per-patient calibration that would survive "
          "to a later use of the device. This concern echoes a "
          "session-leakage finding independently reported for PPG-based BP "
          "evaluation by Tae et al. (2026), using a change-point-aware "
          "buffered evaluation protocol on a different dataset; the present "
          "study raises the same concern specifically for calibration-based "
          "personalization on PulseDB.")

para(doc, "This study addresses four questions on PulseDB's VitalDB subset, "
          "extended with an external, label-free wearable dataset "
          "(PPG-DaLiA; Reiss et al., 2019) as a distribution-shift check: "
          "(1) how much does calibration-free evaluation reduce apparent "
          "accuracy relative to same-patient testing, across a representative "
          "range of architectures from classical ensembles to a "
          "demographic-conditioned transformer; (2) is this reduction "
          "architecture-dependent; (3) what does correcting for the skewed, "
          "normotensive-heavy distribution of training BP values cost and buy, "
          "per architecture; and (4) how much of the calibration-free gap "
          "does brief few-shot personalization recover, and does that "
          "recovery survive a genuinely chronologically separated "
          "calibration/evaluation split rather than one separated only by "
          "clip identity. Addressing these questions together, on the same "
          "five models and the same benchmark, allows this study to report "
          "not only whether a given accuracy number is favourable, but how "
          "much of it is attributable to evaluation protocol rather than to "
          "the model.")

# --------------------------------------------------------------------------- #
h1(doc, "Methods")

h2(doc, "Datasets")
para(doc, "Models were trained and evaluated on the VitalDB-derived subset of "
          "PulseDB (Wang et al., 2022), comprising 10-second, 125 Hz PPG and "
          "ECG segments with paired systolic and diastolic BP (SBP, DBP) "
          "labels, band-pass filtered and minimum-maximum scaled to [0, 1] by "
          "PulseDB's own release pipeline. The training pool contained "
          "419,040 clips from 1,164 subjects after a 10% subject-disjoint "
          "validation hold-out (46,440 clips) used for best-checkpoint "
          "selection. Four official PulseDB test splits were used, detailed "
          "under Protocols below. An independent wearable dataset, PPG-DaLiA "
          "(Reiss et al., 2019; 15 volunteers, wrist-worn sensor, ordinary "
          "daily activity, no BP labels), served as an external check on "
          "whether PulseDB-trained models produce physiologically plausible "
          "and internally consistent output when applied to a free-living, "
          "consumer-grade recording; PPG-DaLiA's own heart-rate labels were "
          "used as a secondary sanity check on signal quality, since no "
          "comparable sanity check is possible for BP.")

h2(doc, "Protocols")
para(doc, "Four protocols, differing only in which patients and clips were "
          "assigned to the test set, were applied to every model with "
          "training otherwise held fixed: leaky (clips drawn at random from "
          "the full subject pool, so a given patient's clips may appear in "
          "both training and test); calibration-based (test subjects were "
          "seen during training; only specific held-out clips differ); "
          "calibration-free (144 subjects never seen during training); and "
          "AAMI (116 subjects, unseen, selected by PulseDB to span the full "
          "clinical BP range). Calibration-free is the protocol most "
          "representative of a genuinely new user and is treated throughout "
          "as the primary evaluation.")
add_figure(doc, FIG_SCHEMATIC, width_mm=165)
caption(doc, "Figure 1", "Study design: the four evaluation protocols "
             "differ only in which patients and clips are assigned to the "
             "test set. Leaky and calibration-based test on previously "
             "seen patients; calibration-free and AAMI test on genuinely "
             "unseen ones.")

h2(doc, "Models")
para(doc, "Five architectures were compared. Random Forest and Gradient "
          "Boosting were trained on 61 hand-crafted morphological and "
          "demographic features per clip (300 trees / 500 boosting rounds "
          "respectively). 1D-CNN (five convolutional blocks, 381,186 "
          "parameters) and ResNet1D (eight residual blocks across four "
          "stages, ResNet-18 style, 2.19M parameters) operated on the raw "
          "two-channel (ECG + PPG) waveform. The Transformer (1.41M "
          "parameters) followed the patch-10 self-attention architecture "
          "with demographic FiLM conditioning and a morphology-classification "
          "auxiliary head reported by Shen et al. (2026), reproduced as "
          "closely as the published description allows: PPG-only input, "
          "per-clip z-score scaling, Adam optimizer (weight decay 1e-8, "
          "fixed learning rate 2e-5), batch size 32. Two deliberate "
          "deviations from that recipe were made for cross-model budget "
          "parity and simplicity: a single joint network predicting both SBP "
          "and DBP (rather than two independent networks), and 50 training "
          "epochs (rather than 100). The auxiliary morphology label itself "
          "is this project's own composite of three PPG-morphology proxies, "
          "since the source paper cites an aggregation rule from unspecified "
          "prior work. Classical models and CNN/ResNet1D used AdamW "
          "(weight decay 1e-4) with a OneCycle learning-rate schedule; "
          "ResNet1D and 1D-CNN's primary calibration-free result used a "
          "schedule decoupled from total epoch count (fixed 20-epoch anneal, "
          "regardless of the 50-epoch total budget), to avoid confounding "
          "\"more training\" with \"a differently shaped learning-rate "
          "curve\" when comparing training budgets. For every model, the "
          "best-validation-checkpoint was used for test-set scoring; final-"
          "epoch weights were never used unless they coincided with the best "
          "checkpoint.")

caption(doc, "Table 1", "Architecture and training configuration summary.")
t_arch_rows = [
    ["Model", "Input", "Params", "Optimizer", "Epochs / batch"],
    ["Random Forest", "61 hand-crafted features", "300 trees", "n/a", "n/a"],
    ["Gradient Boosting", "61 hand-crafted features", "500 rounds", "n/a", "n/a"],
    ["1D-CNN", "raw ECG+PPG waveform", "381,186", "AdamW", "50 / 256"],
    ["ResNet1D", "raw ECG+PPG waveform", "2.19M", "AdamW", "50 / 256"],
    ["Transformer", "raw PPG waveform (z-score)", "1.41M", "Adam", "50 / 32"],
]
make_table(doc, t_arch_rows, col_widths_mm=[26, 46, 20, 20, 24])
doc.add_paragraph()

h2(doc, "BP-bin weighting")
para(doc, "Training BP values in PulseDB are heavily skewed toward the "
          "normotensive range (71.7% of training clips fall in 90-130 mmHg "
          "systolic; only 0.7% exceed 170 mmHg), which can bias a model "
          "toward predicting near the population mean — accurate on "
          "average, but least accurate exactly where a cuffless monitor's "
          "clinical value is highest. A weighted-loss correction, inversely "
          "proportional to each BP bin's training frequency (up to 24.7x for "
          "the rarest bin), was applied during training for every model and "
          "compared against an otherwise identical unweighted run, so that "
          "its cost and benefit could be measured directly rather than "
          "assumed.")

h2(doc, "Personalization")
para(doc, "Per-subject personalization fit an affine correction, "
          "corrected = a x predicted + b, from k = 1, 3, 5, 10, or 25 of a "
          "held-out subject's own calibration-free test clips, with a and b "
          "fit by least squares on those k clips alone and applied to the "
          "subject's remaining clips. Three ways of selecting which k clips "
          "to use were compared (first in storage order, random, and spread "
          "evenly through the recording); results below use the first-k "
          "selection, matching a literal reading of a short calibration "
          "step. Because PulseDB's released calibration-free test file does "
          "not carry a session or case identifier (confirmed directly by "
          "inspecting its fields), a genuinely cross-session personalization "
          "test is not possible on this data; instead, two readings of "
          "\"first k clips\" were compared. The naive reading takes the "
          "first k clips in the file's storage order; a direct check found "
          "storage order uncorrelated with true recording time (r = 0.01, "
          "using PulseDB's segment-index field), so this selection scatters "
          "calibration clips across the full recording session, placing an "
          "evaluation clip within roughly 3 minutes of a calibration clip on "
          "average — plausibly close enough for a model to exploit "
          "short-term signal continuity rather than a stable physiological "
          "calibration. The chronological reading instead sorts by the "
          "genuine segment-index field, takes the true earliest k clips as "
          "calibration, and scores only clips later in the same session "
          "(approximately 70 minutes later on average) — the strongest "
          "cross-time test this single-session dataset supports, though not "
          "a literal cross-session test.")

h2(doc, "Evaluation metrics and statistical treatment")
para(doc, "MAE, mean error (bias), and error standard deviation (SD) were "
          "computed separately for SBP and DBP on every run. AAMI pass/fail "
          "status required bias within +/-5 mmHg and SD within 8 mmHg jointly; "
          "British Hypertension Society (BHS) grade (O'Brien et al., 1990) "
          "required, for grade A, 60% of predictions within 5 mmHg, 85% "
          "within 10 mmHg, and 95% within 15 mmHg of true BP (grade D "
          "otherwise fails all three). "
          "Three training seeds were run for calibration-free (all five "
          "models) and, to test whether the architecture-dependence finding "
          "below is a seed artefact, for calibration-based and leaky "
          "protocols on the two convolutional architectures specifically. "
          "Where three seeds exist, 95% confidence intervals use a "
          "t-distribution (df = 2); single-seed results are reported "
          "descriptively, without a confidence interval, and are flagged as "
          "such in Results.")

# --------------------------------------------------------------------------- #
h1(doc, "Results")

h2(doc, "The calibration-free gap is large and architecture-dependent")
para(doc, "Table 2 reports MAE for every model under every protocol. Relative "
          "to the average of leaky and calibration-based performance (both "
          "of which test on previously seen patients), calibration-free "
          "accuracy was 1.09 to 2.28 times worse, depending on architecture "
          "(Figure 2). The Transformer showed the smallest gap (1.09x: "
          "14.13 mmHg seen-patient average versus 15.44 mmHg calibration-"
          "free), not because it is uniformly the most accurate architecture "
          "— it is in fact the worst of the five on the calibration-based "
          "protocol specifically (13.80 mmHg) — but because it does not "
          "benefit from already-seen patients to the same degree the other "
          "four architectures do. ResNet1D showed the largest gap (2.28x: "
          "6.12 mmHg seen-patient average versus 13.96 mmHg calibration-"
          "free), the sharpest instance of a pattern visible to a lesser "
          "degree in Random Forest and Gradient Boosting: strong apparent "
          "accuracy on seen patients that does not transfer to new ones. "
          "This architecture-dependence held up under a three-seed check on "
          "the two convolutional architectures' calibration-based and leaky "
          "results specifically (standard deviations 0.25-0.90 mmHg across "
          "seeds, an order of magnitude smaller than the gaps separating "
          "architectures).")

caption(doc, "Table 2", "Mean absolute error (SBP / DBP, mmHg) by model and "
             "protocol. * = three-seed mean. † = single schedule-fixed run "
             "(1D-CNN/ResNet1D calfree-weighted only; see Methods), not "
             "seed-averaged. (w) = BP-bin weighted (the primary number); "
             "(u) = unweighted (comparable to prior published numbers, "
             "which do not apply this correction). Unmarked cells are "
             "single seed-0 runs.")
t1_rows = [
    ["Model", "calfree (w)", "calfree (u)", "calbased", "leaky", "AAMI"],
    ["Random Forest", "12.77/8.39*", "12.81/8.59*", "7.14/4.11", "7.47/4.34", "19.67/12.24"],
    ["Gradient Boosting", "14.67/8.98*", "12.76/8.33*", "10.20/6.03", "10.29/6.13", "17.44/11.14"],
    ["1D-CNN", "14.89/9.04†", "13.01/8.34", "10.15/6.55*", "10.36/6.80*", "16.53/11.20"],
    ["ResNet1D", "13.94/8.94†", "13.21/8.53", "6.71/4.26*", "6.41/4.03*", "17.70/11.40"],
    ["Transformer", "15.81/8.71*", "12.89/7.91*", "13.90/8.06*", "13.90/7.96*", "17.03/10.15"],
]
make_table(doc, t1_rows, col_widths_mm=[30, 20, 20, 20, 20, 20])
doc.add_paragraph()
add_figure(doc, FIG_GAP, width_mm=150)
caption(doc, "Figure 2", "Known-patient (leaky/calibration-based average) "
             "versus calibration-free accuracy, all five models, SBP and "
             "DBP. Orange/yellow bars test on previously seen patients; "
             "blue/aqua bars test on unseen patients.")
para(doc, "Table 3 reports the full clinical metric set — correlation, "
          "AAMI pass/fail, and BHS grade — for the calibration-free "
          "(weighted) result. Pearson r is modest across the board "
          "(0.45-0.57), confirming that none of the five models track an "
          "individual patient's BP up and down reliably on unseen "
          "patients; the Transformer's DBP correlation (0.57) is the "
          "highest of any model/target pair, consistent with it carrying "
          "the least calibration-free penalty overall. Every model fails "
          "BHS grade D (the lowest) and AAMI on calibration-free, for "
          "both targets — the correlation and grade numbers tell the "
          "same story as the MAE numbers above, from a different angle.")
caption(doc, "Table 3", "Full clinical metric set, calibration-free "
             "(weighted), all five models.")
t6_rows = [
    ["Model", "SBP r", "SBP AAMI", "SBP BHS", "DBP r", "DBP AAMI", "DBP BHS"],
    ["Random Forest", "0.518", "FAIL", "D", "0.468", "FAIL", "D"],
    ["Gradient Boosting", "0.497", "FAIL", "D", "0.454", "FAIL", "D"],
    ["1D-CNN", "0.546", "FAIL", "D", "0.479", "FAIL", "D"],
    ["ResNet1D", "0.525", "FAIL", "D", "0.473", "FAIL", "D"],
    ["Transformer", "0.530", "FAIL", "D", "0.570", "FAIL", "D"],
]
make_table(doc, t6_rows, col_widths_mm=[26, 14, 14, 14, 14, 14, 14])
doc.add_paragraph()

h2(doc, "BP-bin weighting trades overall accuracy for high-BP coverage, with no single best answer")
para(doc, "BP-bin weighting's effect was architecture-specific rather than "
          "uniformly beneficial. It cost essentially nothing for Random "
          "Forest (-0.18 mmHg overall, i.e. a negligible net improvement) "
          "while recovering 4.05 mmHg of error in the above-170-mmHg band; "
          "it cost the Transformer the most overall (+2.77 mmHg) while "
          "recovering the second-most in that same high-BP band (16.41 "
          "mmHg, behind only the CNN's 17.52 mmHg). Below 130 mmHg systolic "
          "(72% of test clips), the unweighted model was frequently equal "
          "or better; above 150 mmHg, weighting's benefit was consistent "
          "and clinically meaningful across every architecture tested. No "
          "single weighted/unweighted choice dominated across architectures, "
          "which is why both are reported throughout this study rather than "
          "collapsed to one number (Figure 3): the overall cost is small "
          "and architecture-specific, while the high-BP benefit is an "
          "order of magnitude larger for every architecture, consistent "
          "with weighting's cost being concentrated in the crowded "
          "normotensive middle of the training distribution and its "
          "benefit in the sparse, clinically critical extremes.")
add_figure(doc, FIG_BAND, width_mm=160)
caption(doc, "Figure 3", "BP-bin weighting's cost (overall SBP MAE change) "
             "and benefit (above-170 mmHg SBP MAE change), weighted minus "
             "unweighted training, all five models.")

h2(doc, "Against the literature")
para(doc, "Figure 4 compares this study's calibration-free, unweighted "
          "results (matching how prior published numbers were measured, "
          "since the BP-bin weighting above is not standard practice) "
          "against four prior studies on the same VitalDB subset. The "
          "Transformer's unweighted result (12.67 mmHg SBP, three-seed mean "
          "12.89, SD 0.32) was the closest match to the current best "
          "published number, Shen et al.'s (2026) demographic-conditioned "
          "transformer (12.17 mmHg) — within half a point of it, and ahead "
          "of every other architecture tested here as well as of "
          "Moulaeifard et al.'s (2025) XResNet1d101 benchmark (12.70 mmHg). "
          "The 1D-CNN's unweighted result (13.01 mmHg, three-seed mean "
          "13.04, SD 0.12) was close behind. Taken together with the "
          "architecture-dependence finding above, the gap between four "
          "years of published architecture progress on this benchmark — "
          "roughly 12.17 to 13.67 mmHg across the four prior studies "
          "compared here — is smaller than the 1.09x-2.28x gap this study "
          "measures from evaluation protocol alone on a single model "
          "architecture. This is evidence about this benchmark specifically, "
          "not a general claim about the field, since it reflects four "
          "studies rather than an exhaustive literature search.")

add_figure(doc, FIG_LIT, width_mm=150)
caption(doc, "Figure 4", "Calibration-free SBP MAE, this study (orange, "
             "unweighted) against four prior studies on the same PulseDB "
             "VitalDB subset (blue).")

h2(doc, "An unseen-patient test set does not guarantee AAMI-compliant sampling")
para(doc, "Because nearly every calibration-free AAMI pass/fail claim in "
          "this and comparable studies is computed on the calibration-free "
          "test split, whether that split itself satisfies AAMI/ISO "
          "81060-2's own sampling requirement for reference-BP coverage was "
          "checked directly (Table 4). It does not: the calibration-free "
          "split fails the standard's requirement on three of its four "
          "BP-range criteria for both SBP and DBP (for example, only 1.5% "
          "of subjects reach 160 mmHg systolic, against a 5% requirement). "
          "PulseDB's AAMI-specific split, built explicitly to span the full "
          "clinical range, passes every criterion. This indicates that "
          "AAMI pass/fail claims computed on the calibration-free split "
          "describe performance on a population skewed toward normal-to-low "
          "BP, and that the AAMI-specific split is the methodologically "
          "appropriate one to carry AAMI framing specifically, independent "
          "of which split is otherwise preferred for leakage control.")

caption(doc, "Table 4", "AAMI/ISO 81060-2 sampling-coverage check, "
             "calibration-free versus AAMI-specific test splits.")
t2_rows = [
    ["Requirement", "calfree_test (n=144)", "aami_test (n=116)"],
    ["N >= 85", "PASS (144)", "PASS (116)"],
    ["SBP <=10 mmHg in >=5%", "PASS (21.3%)", "PASS (15.6%)"],
    ["SBP >=160 mmHg in >=5%", "FAIL (1.5%)", "PASS (21.5%)"],
    ["SBP >=140 mmHg in >=20%", "FAIL (10.3%)", "PASS (42.6%)"],
    ["DBP <=60 mmHg in >=5%", "PASS (41.2%)", "PASS (24.5%)"],
    ["DBP >=100 mmHg in >=5%", "FAIL (0.3%)", "PASS (10.5%)"],
    ["DBP >=85 mmHg in >=20%", "FAIL (4.1%)", "PASS (31.5%)"],
]
make_table(doc, t2_rows, col_widths_mm=[55, 40, 40])

h2(doc, "An external, label-free wearable check")
add_figure(doc, FIG_SHIFT, width_mm=150)
caption(doc, "Figure 5", "Predicted SBP shift from PulseDB to PPG-DaLiA, "
             "weighted versus unweighted training, all five models.")
para(doc, "On PPG-DaLiA, every model's mean predicted SBP across the 15 "
          "volunteers exceeded the healthy-adult norm, and models still "
          "separated individuals by 2.1-9.6 mmHg (true between-subject "
          "spread: 11.8 mmHg), indicating the shift onto this external "
          "device-and-population combination does not collapse all subjects "
          "to a single value (Figure 5). The fraction of the apparent shift "
          "attributable to BP-bin weighting itself (rather than to the new "
          "device/population) ranged from 2% (Transformer) to 40% (Gradient "
          "Boosting) across four of the five models; ResNet1D was a "
          "reversed outlier, where weighting reduced rather than enlarged "
          "the apparent shift. Heart-rate MAE, the only metric on this "
          "dataset with ground truth, was 13.06 bpm in low-motion activity "
          "segments (sitting, lunch, working, driving) and 33.35 bpm in "
          "high-motion segments (cycling, stairs, walking, table soccer, "
          "transient) using this study's generic pulse detector, against "
          "8.69 bpm reported by Reiss et al. (2019) using a dedicated "
          "denoiser on the same data — a reminder that the BP numbers above "
          "likely carry additional, unmeasured error from signal-processing "
          "choices beyond the BP-estimation model itself.")

para(doc, "A second, independent label-free check is the fraction of each "
          "model's PPG-DaLiA predictions that fall outside physiologically "
          "plausible clinical range (SBP outside 70-200 mmHg or DBP outside "
          "40-120 mmHg; Table 5). No model ever predicts DBP above SBP for "
          "the same clip. Beyond that, the rate of out-of-range predictions "
          "spans two orders of magnitude by architecture, from 0.5% "
          "(Transformer) to 29.7% (Gradient Boosting); the two classical "
          "ensembles produce an implausible prediction on roughly one clip "
          "in four to one in three, while the three deep models stay under "
          "3.4%. This is a plausibility check rather than an accuracy check "
          "— it does not require true labels and cannot confirm a "
          "prediction is correct, only flag predictions that are certainly "
          "wrong — and it reinforces this study's feature-representation "
          "finding that representation choice affects out-of-distribution "
          "behaviour beyond what calibration-free MAE alone shows.")

caption(doc, "Table 5", "Physiologically implausible predictions on "
             "PPG-DaLiA (% of clips outside SBP 70-200 mmHg or DBP "
             "40-120 mmHg), all five models. DBP exceeding SBP on the same "
             "clip: 0.0% for every model.")
t_plausible_rows = [
    ["Model", "SBP out of range", "DBP out of range", "Any"],
    ["Random Forest", "0.0%", "27.2%", "27.2%"],
    ["Gradient Boosting", "13.8%", "25.7%", "29.7%"],
    ["1D-CNN", "2.0%", "2.2%", "3.4%"],
    ["ResNet1D", "0.0%", "0.8%", "0.8%"],
    ["Transformer", "0.3%", "0.5%", "0.5%"],
]
make_table(doc, t_plausible_rows, col_widths_mm=[40, 35, 35, 25])
doc.add_paragraph()

h2(doc, "Personalization closes most of the diastolic gap, not the systolic one")
para(doc, "Table 7 and Figure 6 report personalization results. Under the "
          "naive (storage-order) calibration split, 25 clips reduced "
          "diastolic MAE by 20-51% across the five models (e.g. Random "
          "Forest: 8.38 to 4.81 mmHg), and every model passed AAMI's "
          "diastolic criterion from roughly five calibration clips onward. "
          "This recovery came within 0.3 mmHg of each architecture's "
          "physiological ceiling (Table 6) — the best possible constant "
          "per-subject offset, computed directly from each subject's true "
          "held-out labels and therefore unreachable in real deployment, "
          "but useful to show the remaining room (e.g. 1D-CNN: raw SBP MAE "
          "15.32 mmHg, oracle ceiling 9.03 mmHg, realised 25-clip result "
          "7.74 mmHg). Across all five models, 33-41% of raw SBP error and "
          "40-46% of raw DBP error was this kind of fixable, constant "
          "per-subject offset; the remainder reflects within-subject BP "
          "variability that a constant correction cannot track, since "
          "systolic BP varies more within one subject's own recording (SD "
          "13.9 mmHg, 1D-CNN) than it does between different subjects "
          "(11.8 mmHg).")

caption(doc, "Table 6", "Personalization ceiling: raw MAE versus the "
             "oracle (true-label) per-subject offset, all five models.")
t5_rows = [
    ["Model", "SBP raw->oracle", "SBP recov.", "DBP raw->oracle", "DBP recov."],
    ["Random Forest", "12.76->8.48", "33%", "8.38->4.98", "41%"],
    ["Gradient Boosting", "14.63->8.77", "40%", "9.00->5.15", "43%"],
    ["1D-CNN", "15.32->9.03", "41%", "9.20->5.00", "46%"],
    ["ResNet1D", "13.96->9.01", "35%", "9.03->5.46", "40%"],
    ["Transformer", "15.44->9.34", "39%", "8.47->4.94", "42%"],
]
make_table(doc, t5_rows, col_widths_mm=[28, 26, 16, 26, 16])
doc.add_paragraph()

para(doc, "Under the chronological calibration split, the picture changed "
          "substantially. Diastolic MAE still improved (e.g. Random Forest: "
          "8.38 to 7.32 mmHg), but no model met AAMI's spread criterion at "
          "any calibration size from 1 to 25 clips, for any of the five "
          "architectures — a consistent finding, not a mixed one. Bias "
          "stayed under 2 mmHg throughout under this split, so the AAMI "
          "shortfall reflects error spread specifically, not a systematic "
          "offset the correction failed to remove. The gap between the two "
          "splits' outcomes is consistent with the storage-order split "
          "allowing the correction to exploit short-term signal continuity "
          "(an average 3-minute gap between a calibration and an evaluation "
          "clip) that does not persist across the roughly 70-minute gap the "
          "chronological split enforces — the same general concern Tae et "
          "al. (2026) raise for PPG-based BP evaluation, observed here in "
          "the specific context of few-shot personalization.")

caption(doc, "Table 7", "Diastolic MAE (mmHg) at k=0 and k=25 calibration "
             "clips, naive (storage-order) versus chronological split, all "
             "five models. Pass/fail is AAMI status at k=25.")
t3_rows = [
    ["Model", "k=0", "k=25 naive", "AAMI (naive)", "k=25 chron.", "AAMI (chron.)"],
    ["Random Forest", "8.38", "4.81", "PASS", "7.32", "FAIL"],
    ["Gradient Boosting", "9.00", "4.74", "PASS", "7.11", "FAIL"],
    ["1D-CNN", "9.20", "4.51", "PASS", "7.25", "FAIL"],
    ["ResNet1D", "9.03", "4.83", "PASS", "7.38", "FAIL"],
    ["Transformer", "8.47", "4.40", "PASS", "6.68", "FAIL"],
]
make_table(doc, t3_rows, col_widths_mm=[28, 14, 18, 18, 18, 18])
doc.add_paragraph()

para(doc, "Table 8 gives the full chronological-split sweep underlying this "
          "claim: SDE never approaches AAMI's 8 mmHg spread bound at any "
          "tested k, for any model, including the largest calibration set "
          "tested (k=25, best case 8.75 mmHg, Transformer).")

caption(doc, "Table 8", "Chronological-split DBP MAE / SDE across the full "
             "calibration-size sweep (mmHg), all five models. AAMI requires "
             "SDE <= 8 mmHg; no cell meets this.")
t_chron_rows = [
    ["Model", "k=1", "k=3", "k=5", "k=10", "k=25"],
    ["Random Forest", "9.58/12.74", "11.97/18.96", "9.46/12.68", "8.65/11.86", "7.32/9.68"],
    ["Gradient Boosting", "9.11/11.84", "13.11/27.74", "9.23/12.32", "7.95/10.53", "7.11/9.47"],
    ["1D-CNN", "10.11/13.30", "11.51/17.35", "9.39/12.89", "8.20/10.99", "7.25/9.75"],
    ["ResNet1D", "10.54/13.59", "11.10/15.61", "9.41/12.68", "8.44/11.17", "7.38/9.47"],
    ["Transformer", "9.60/12.59", "10.38/14.81", "8.38/11.23", "7.60/10.08", "6.68/8.75"],
]
make_table(doc, t_chron_rows, col_widths_mm=[30, 24, 24, 24, 24, 24])
doc.add_paragraph()

add_figure(doc, FIG_PERS, width_mm=150)
caption(doc, "Figure 6", "Error reduction as a function of calibration-clip "
             "count (k = 0-25), all five models. Most recoverable error is "
             "gone by k=5; two models show a brief worsening at k=3, "
             "indicating 2-3 clips is too few to fit a stable correction.")

# --------------------------------------------------------------------------- #
h1(doc, "Discussion")

para(doc, "This study's central finding is that evaluation-protocol choice "
          "is not a minor methodological detail for cuffless BP estimation: "
          "on a single benchmark, moving from same-patient to unseen-patient "
          "testing changed apparent accuracy by 1.09x to 2.28x depending on "
          "architecture, a larger effect than the gap separating the best "
          "and weakest architecture-progress results compared against in "
          "this study. Because this effect is architecture-dependent, a "
          "protocol-blind comparison across studies that use different "
          "architectures and different (or unstated) protocols risks "
          "attributing evaluation-protocol leniency to genuine architectural "
          "improvement. Readers and reviewers of cuffless-BP work would "
          "benefit from comparisons that hold protocol fixed, as this study "
          "attempts, or that at minimum state the protocol explicitly enough "
          "for a reader to judge comparability. This sits within a broader, "
          "field-spanning pattern: leakage of essentially this kind — "
          "information about the test set reaching the model indirectly, "
          "rather than through a frankly mislabelled split — has been "
          "documented across at least 17 scientific disciplines using "
          "machine learning, frequently large enough to overturn a study's "
          "central claim once corrected (Kapoor & Narayanan, 2023). The "
          "present results are consistent with that pattern holding in "
          "cuffless BP estimation specifically, and support the standardized "
          "evaluation reporting Elgendi et al. (2024) recommend for the "
          "field: in particular, their recommendation that studies report "
          "accuracy stratified by whether test subjects were seen during "
          "training, rather than a single pooled number.")

para(doc, "This finding also revises this study's own earlier working "
          "conclusion. An initial analysis, run before the Transformer's "
          "evaluation across all four protocols was complete, suggested the "
          "Transformer had the largest calibration-free gap of the five "
          "architectures. With the complete evaluation, the opposite is "
          "true: the Transformer has the smallest gap, and ResNet1D the "
          "largest. The direction of the underlying finding — that the "
          "model most accurate on previously seen patients is not "
          "necessarily the most honest about new ones — holds under either "
          "data state, but which specific architecture it is true of does "
          "not, which is itself a caution about drawing architecture-level "
          "conclusions before every architecture's evaluation is complete.")

para(doc, "The AAMI sampling-coverage finding (Table 4) has a direct, "
          "practical implication for how this and comparable studies should "
          "report AAMI compliance: the test population most commonly used "
          "for calibration-free AAMI claims is itself skewed toward "
          "normal-to-low BP and does not meet AAMI's own high-BP coverage "
          "requirement, independent of model accuracy. This is a property "
          "of the test-set construction, not of any model evaluated against "
          "it, and would not have been visible from accuracy numbers alone.")

para(doc, "The personalization results have a specific and testable "
          "implication for deployment: a brief calibration step recovers "
          "most of the fixable diastolic error, but only under evaluation "
          "conditions this study's chronological-split analysis suggests "
          "may not represent later use of the device. Diastolic AAMI "
          "compliance achieved under the naive, storage-order split did not "
          "survive a genuinely time-separated evaluation, at any calibration "
          "size tested. Systolic error's spread did not reach AAMI "
          "compliance under either split, and the ceiling analysis indicates "
          "why: within-subject systolic variability exceeds between-subject "
          "variability on this benchmark, so no constant per-subject "
          "correction — however well calibrated — can close that "
          "particular gap. Closing it would require modelling beat-to-beat "
          "variation rather than a per-subject offset, which is beyond this "
          "study's scope.")

para(doc, "Several design choices strengthen confidence in these "
          "conclusions specifically. Every comparison central to this "
          "study's main claims holds the dataset, the clip-level feature "
          "pipeline, and the checkpoint-selection rule fixed, varying only "
          "the one factor each question is about — protocol for the "
          "leakage-gap finding, weighting for the cost/benefit finding, "
          "and calibration-clip timing for the personalization finding — "
          "so observed differences are not confounded by incidental "
          "pipeline changes across comparisons, as can happen when results "
          "are pooled across separately published studies. The "
          "architecture-dependence finding was checked against seed "
          "variance directly (three seeds, two architectures, two "
          "protocols) rather than assumed stable from a single run, and "
          "the personalization ceiling analysis grounds the practical "
          "recommendations in a computable upper bound rather than "
          "intuition about how much calibration \"should\" help. Five "
          "architectures spanning three model families (tree ensembles, "
          "convolutional networks, and attention) is also a wider sweep "
          "than most single-architecture evaluations in this literature, "
          "which is what makes the architecture-dependence finding "
          "possible to state at all.")

para(doc, "Several limitations qualify these conclusions. First, PulseDB's "
          "released calibration-free test file does not carry a session or "
          "case identifier, so the chronological split used here, while the "
          "strongest cross-time test this dataset supports, is a "
          "within-session rather than a literal cross-session test; the "
          "personalization-recovery figures above should be read as an "
          "upper bound on what a genuinely separate-visit calibration would "
          "achieve. Second, neither PulseDB (hospital/ICU patients) nor "
          "PPG-DaLiA (15 healthy, free-living volunteers, none over 60) is a "
          "close demographic match for the motivating use case of an older, "
          "ambulatory, consumer-wearable user; PulseDB's patients are not "
          "wearing a consumer-grade sensor, and PPG-DaLiA's volunteers are "
          "neither old enough nor hypertensive enough to stress-test the "
          "clinically relevant range. This reflects what is publicly "
          "available for this task rather than a flaw specific to this "
          "study, but it means the present results should be read as "
          "evidence about measurement methodology, not as a direct estimate "
          "of real-world accuracy in the target population. Third, several "
          "comparisons reported above (weighted versus unweighted, "
          "naive-split versus chronological-split personalization, and "
          "leaky/calibration-based results for Random Forest and Gradient "
          "Boosting) reflect single training runs rather than seed-averaged "
          "estimates, and are reported descriptively rather than with a "
          "confidence interval; this study makes on the order of hundreds "
          "of such comparisons in total, and none of the confidence "
          "intervals reported are corrected for multiple comparisons. "
          "Fourth, the Transformer's auxiliary morphology label is this "
          "study's own composite proxy, since the source architecture's "
          "published description cites an unspecified prior aggregation "
          "rule for its own label; this and other unavoidable reproduction "
          "choices (a single joint SBP/DBP network, 50 rather than 100 "
          "training epochs) mean the remaining gap to the originally "
          "published Transformer result should not be attributed to "
          "architecture alone.")

para(doc, "Finally, the session-leakage concern this study raises for "
          "personalization is consistent with, rather than a first report "
          "of, independent work: Tae et al. (2026) identify the same "
          "general mechanism — temporal proximity between calibration and "
          "evaluation data inflating apparent accuracy — for PPG-based BP "
          "estimation generally, addressed there through a change-point-"
          "aware buffered evaluation protocol on a different dataset. The "
          "present study's contribution is a direct demonstration of the "
          "same mechanism specifically within few-shot personalization on "
          "PulseDB, using the dataset's own (if imperfect) temporal "
          "ordering field.")

# --------------------------------------------------------------------------- #
h1(doc, "Conclusion")
para(doc, "Across five architectures spanning classical ensembles to a "
          "demographic-conditioned transformer, evaluation-protocol leakage "
          "inflated apparent cuffless-BP accuracy by 1.09x to 2.28x, an "
          "effect that is architecture-dependent and, on this benchmark, "
          "larger than several years of reported architecture progress. "
          "Twenty-five real calibration clips recovered most of the "
          "physiologically fixable diastolic error — but this recovery, "
          "and the AAMI compliance it enabled, did not survive a genuinely "
          "chronologically separated calibration/evaluation split, and "
          "systolic error's spread remained above the AAMI threshold "
          "regardless of calibration. Future work on cuffless BP estimation "
          "for home and wearable use should report evaluation protocol "
          "explicitly enough for cross-study comparison, verify that any "
          "test split used for AAMI compliance claims meets the standard's "
          "own sampling requirements, and test personalization gains under "
          "evaluation conditions separated in time from calibration, not "
          "only in clip identity. For the gerontechnology use case that "
          "motivates this work — unobtrusive, continuous BP awareness "
          "supporting independent aging in place (Olmedo-Aguirre et al., "
          "2022) — these distinctions are not academic: a device whose "
          "validated accuracy depends on a lenient test protocol, or whose "
          "personalization benefit does not survive to a later day's use, "
          "will not deliver the trend awareness older users and their "
          "caregivers would reasonably expect from it.")

# --------------------------------------------------------------------------- #
h1(doc, "Data and Code Availability")
para(doc, "PulseDB is available from the original authors "
          "(Wang et al., 2022); this study uses the VitalDB-derived "
          "subset only. PPG-DaLiA is available from the UCI Machine "
          "Learning Repository (Reiss et al., 2019). Neither dataset is "
          "redistributed here. The processing and training pipeline, "
          "trained model checkpoints, and the exact scripts used to "
          "produce every number, table, and figure in this manuscript are "
          "available at [repository URL to be added on acceptance]. "
          "PulseDB's VitalDB portion is released under a non-commercial, "
          "share-alike licence, which constrains redistribution of "
          "derived data but not of code or of the results reported here; "
          "compatibility with a specific target venue's own data policy "
          "has not yet been checked and should be verified before any "
          "venue-specific submission.")

# --------------------------------------------------------------------------- #
h1(doc, "References")

refs = [
    "Arjomand, A., Boudesh, A., Bayatmakou, F., Kent, K. B., & Mohammadi, A. "
    "(2024). TransfoRhythm: A transformer architecture conductive to blood "
    "pressure estimation via solo PPG signal capturing (arXiv:2404.15352). "
    "arXiv. https://arxiv.org/abs/2404.15352",

    "Chen, J., Zhou, X., Feng, L., Ling, B. W.-K., Han, L., & Zhang, H. "
    "(2024). rU-Net, multi-scale feature fusion and transfer learning: "
    "Unlocking the potential of cuffless blood pressure monitoring with "
    "PPG and ECG. IEEE Journal of Biomedical and Health Informatics. "
    "https://doi.org/10.1109/JBHI.2024.3483301",

    "Elgendi, M., Fletcher, R., Liang, Y., Howard, N., Lovell, N. H., "
    "Abbott, D., Lim, K., & Ward, R. (2019). The use of photoplethysmography "
    "for assessing hypertension. npj Digital Medicine, 2, 60. "
    "https://doi.org/10.1038/s41746-019-0136-7",

    "Elgendi, M., Haugg, F., Fletcher, R. R., Allen, J., Shin, H., Alian, "
    "A., & Menon, C. (2024). Recommendations for evaluating "
    "photoplethysmography-based algorithms for blood pressure assessment. "
    "Communications Medicine, 4, 134. "
    "https://doi.org/10.1038/s43856-024-00555-2",

    "Huang, Y., He, Y., Song, Z., Gao, K., & Zheng, Y. (2024). Validation "
    "of deep learning models for cuffless blood pressure estimation on a "
    "large benchmarking dataset. Connected Health and Telemedicine, 3(1), "
    "300002. https://doi.org/10.20517/chatmed.2023.23",

    "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility "
    "crisis in machine-learning-based science. Patterns, 4(9), 100804. "
    "https://doi.org/10.1016/j.patter.2023.100804",

    "Moulaeifard, M., Charlton, P. H., & Strodthoff, N. (2025). "
    "Generalizable deep learning for photoplethysmography-based blood "
    "pressure estimation: A benchmarking study (arXiv:2502.19167). arXiv. "
    "https://arxiv.org/abs/2502.19167",

    "Mukkamala, R., Hahn, J.-O., Inan, O. T., Mestha, L. K., Kim, C.-S., "
    "Töreyin, H., & Kyal, S. (2015). Toward ubiquitous blood pressure "
    "monitoring via pulse transit time: Theory and practice. IEEE "
    "Transactions on Biomedical Engineering, 62(8), 1879-1901. "
    "https://doi.org/10.1109/TBME.2015.2441951",

    "O'Brien, E., Petrie, J., Littler, W., de Swiet, M., Padfield, P. L., "
    "O'Malley, K., Jamieson, M., Altman, D. G., Bland, M., & Atkins, N. "
    "(1990). The British Hypertension Society protocol for the evaluation "
    "of automated and semi-automated blood pressure measuring devices, "
    "with special reference to ambulatory systems. Journal of "
    "Hypertension, 8(7), 607-619. "
    "https://doi.org/10.1097/00004872-199007000-00004",

    "Oliveros, E., Patel, H., Kyung, S., Fugar, S., Goldberg, A., Madan, "
    "N., & Williams, K. A. (2020). Hypertension in older adults: "
    "Assessment, management, and challenges. Clinical Cardiology, 43(2), "
    "99-107.",

    "Olmedo-Aguirre, J. O., Reyes-Campos, J., Alor-Hernández, G., "
    "Machorro-Cano, I., Rodríguez-Mazahua, L., & Sánchez-Cervantes, J. L. "
    "(2022). Remote healthcare for elderly people using wearables: A "
    "review. Biosensors, 12(2), 73. https://doi.org/10.3390/bios12020073",

    "Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). "
    "Deep PPG: Large-scale heart rate estimation with convolutional neural "
    "networks. Sensors, 19(14), 3079. https://doi.org/10.3390/s19143079",

    "Shen, Y., Mathew, N., Rahimi, M., Dhakal, D., Zouridakis, G., Fu, X., "
    "& Hu, R. (2026). DMT: Demographic conditioning, morphology-enhanced "
    "transformer for cuffless blood pressure estimation from PPG signals "
    "(arXiv:2606.11125). arXiv. https://arxiv.org/abs/2606.11125",

    "Stergiou, G. S., Alpert, B., Mieke, S., Asmar, R., Atkins, N., "
    "Eckert, S., Frick, G., Friedman, B., Graßl, T., Ichikawa, T., "
    "Ioannidis, J. P., Lacy, P., McManus, R., Murray, A., Myers, M., "
    "Palatini, P., Parati, G., Quinn, D., Sarkis, J., ... O'Brien, E. "
    "(2018). A universal standard for the validation of blood pressure "
    "measuring devices. Hypertension, 71(3), 368-374. "
    "https://doi.org/10.1161/HYPERTENSIONAHA.117.10237",

    "Tae, Y., Park, M., Rho, G., Yoo, D., & Joo, S. (2026). Change "
    "point-aware evaluation and re-calibration of PPG-based blood pressure "
    "estimation (arXiv:2608.18639). arXiv. "
    "https://arxiv.org/abs/2608.18639",

    "Wang, W., Mohseni, P., Kilgore, K. L., & Najafizadeh, L. (2022). "
    "PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for "
    "benchmarking cuff-less blood pressure estimation methods. Frontiers "
    "in Digital Health, 4, 1090854. "
    "https://doi.org/10.3389/fdgth.2022.1090854",

    "World Health Organization. (2023). Global report on hypertension: "
    "The race against a silent killer. "
    "https://www.who.int/publications/i/item/9789240081062",

    "World Health Organization. (2025). Ageing and health. "
    "https://www.who.int/news-room/fact-sheets/detail/ageing-and-health",
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Mm(10)
    p.paragraph_format.first_line_indent = Mm(-10)
    p.paragraph_format.space_after = Pt(8)
    p.add_run(r).font.size = Pt(11)

# --------------------------------------------------------------------------- #
DOCS.mkdir(parents=True, exist_ok=True)
docx_path = DOCS / "Journal_Paper_ISGJ.docx"
doc.save(str(docx_path))

body_words = sum(len(p.text.split()) for p in doc.paragraphs)
print(f"wrote {docx_path}  (~{body_words} words incl. abstract/refs)")

# --------------------------------------------------------------------------- #
# separate title page (journal requires this as its own upload)
# --------------------------------------------------------------------------- #
tp = Document()
sec = tp.sections[0]
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(1)
set_base_style(tp)

para(tp, TITLE, bold=True, size=14, space_after=16)
para(tp, "Author(s): [Full name(s), academic degree(s) — not current "
         "position — and affiliation(s)]", space_after=8)
para(tp, "Corresponding author: [Name, email, affiliation, mailing address]",
     space_after=8)
para(tp, "Acknowledgments: [max. 300 characters, e.g. advisor/professor "
         "thanks, compute resources]", space_after=8)
para(tp, "Funding: [funding source, or \"This research received no "
         "external funding\"]", space_after=8)
para(tp, "Conflicts of interest: [none declared, or list]", space_after=8)

tp_path = DOCS / "Journal_Paper_ISGJ_TitlePage.docx"
tp.save(str(tp_path))
print(f"wrote {tp_path}  (fill in [brackets] before submission)")

# --------------------------------------------------------------------------- #
# PDF export via Pages (headless AppleScript). Pages needs to already be
# running before AppleScript can address it the first time this permission
# is granted in a session -- `open -a Pages` + a wait handles that reliably,
# where `tell application "Pages" to activate` alone can race and fail with
# "Application isn't running" (-600) on a cold launch.
# --------------------------------------------------------------------------- #
subprocess.run(["open", "-a", "Pages"])
subprocess.run(["osascript", "-e",
                'repeat 30 times\n'
                '  if application "Pages" is running then exit repeat\n'
                '  delay 0.5\n'
                'end repeat'], timeout=30)

for src in (docx_path, tp_path):
    pdf_path = src.with_suffix(".pdf")
    script = f'''
tell application "Pages"
    set d to open (POSIX file "{src}")
    delay 3
    export d to (POSIX file "{pdf_path}") as PDF
    close d saving no
end tell'''
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
    if r.returncode == 0 and pdf_path.exists():
        print(f"wrote {pdf_path.name}")
    else:
        print(f"PDF conversion failed for {src.name}: {r.stderr[-500:]}")
