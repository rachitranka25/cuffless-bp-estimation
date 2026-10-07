"""Regenerate the Experiment 1 (20-epoch, single-seed) notebooks into the archive.

`scripts/build_notebooks.py` is the only source of truth for these notebooks and
it now generates the Experiment 2 (50-epoch, 3-seed) versions in place — the
original 20-epoch .ipynb files were overwritten with no backup when that script
was edited. This script reconstructs *equivalent* notebooks: same cells `02` and
`04` (those never depended on epoch count or seeds), and a 20-epoch / no-seed-
variance version of `03a/03b/03c`, matching exactly what was actually run to
produce the results in experiment_1_20epoch/results/.

This is a reconstruction, not the original file — cell code is regenerated from
this script rather than recovered, and it carries no execution outputs (the
original notebooks, if run interactively on Colab, would have had output cells;
these do not). The training code itself (`src/train.py`) is unchanged between
Experiment 1 and Experiment 2, so re-running these notebooks reproduces the
same pipeline the 20-epoch results came from.

Run:  BP_ROOT=".../experiment_1_20epoch" python3 scripts/build_experiment1_notebooks.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_notebooks import (   # noqa: E402
    md, code, write, colab_setup, result_cell, baselines, personalization,
)

DEEP_NB_V1 = {
    "cnn": ("03a", "1D-CNN", 20, 256,
            ["The second model in spec section I: *\"1D-CNN on raw PPG (+ECG)",
             "waveform — standard deep baseline\"*.",
             "",
             "This one never sees the 61 hand-crafted features. It gets the waveform",
             "itself — 2 x 1250 numbers per clip — and works out what matters on its",
             "own. Whether that beats measuring the pulse by hand is one of the",
             "ablations the spec asks for.",
             "",
             "About 90 minutes on a T4 at 20 epochs."]),
    "resnet": ("03b", "ResNet1D", 20, 256,
               ["The third model in spec section I: *\"ResNet1D / lightweight",
                "Transformer — stronger deep baseline reported in recent",
                "PulseDB-benchmarking papers\"*.",
                "",
                "Same convolutional idea as the CNN, with skip connections so a much",
                "deeper stack still trains. This is the standard strong baseline",
                "across the PulseDB literature.",
                "",
                "About 2.5 hours on a T4 at 20 epochs."]),
    "transformer": ("03c", "Transformer (lightweight)", 20, 128,
                    ["Spec section I: *\"lightweight Transformer\"*. Our own recipe —",
                     "not a reproduction of any published Transformer baseline — sized",
                     "to train in a comparable budget to the CNN and ResNet above: the",
                     "1250-sample clip becomes 125 patches, six blocks of eight-head",
                     "self-attention, and every block is conditioned on age, sex and",
                     "BMI through FiLM.",
                     "",
                     "**This cannot run on the laptop.** Apple's MPS backend is slower",
                     "than its own CPU for attention at this shape (92.8 ms vs 30.5 ms",
                     "per call), and on CPU a single epoch over just 5,820 clips took",
                     "138 seconds — which extrapolates to weeks for the full set.",
                     "",
                     "Batch size is 128 rather than 256 because attention over 125 tokens",
                     "needs more memory per clip. About 5 hours on a T4 at 20 epochs."]),
}


def deep_v1(model):
    num, title, epochs, bs, blurb = DEEP_NB_V1[model]
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
             "              epochs=EPOCHS, batch_size=BATCH)"),
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
             "                        epochs=EPOCHS, batch_size=BATCH)"),
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
             "                        epochs=EPOCHS, batch_size=BATCH)"),
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
           "shift comes from it too."),
        code("shift_nobalance = evaluate_dalia(model=MODEL, protocol='calfree',",
             "                                 tag='nobalance')"),
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


if __name__ == "__main__":
    baselines()
    for m in DEEP_NB_V1:
        deep_v1(m)
    personalization()
