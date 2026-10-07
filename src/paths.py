"""Canonical project paths, resolved from this file's location.

Import this instead of hardcoding relative paths, so scripts work no matter
which directory they are run from.

    import sys; sys.path.insert(0, "src")   # or add src/ to PYTHONPATH
    from paths import RAW_PULSEDB, PROC_PULSEDB
"""

import os
from pathlib import Path

# On Colab the code is imported from Drive but the repo layout is the same, so a
# single environment variable moves every path at once:
#     os.environ["BP_ROOT"] = "/content/drive/MyDrive/bp"
# Set it before importing anything from this package.
ROOT = Path(os.environ.get("BP_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

RAW_PULSEDB = RAW / "pulsedb"
RAW_DALIA = RAW / "ppg_dalia"
PROC_PULSEDB = PROCESSED / "pulsedb"
PROC_DALIA = PROCESSED / "ppg_dalia"

# everything derived from the datasets lives under data/ too
FEATURES = DATA / "features"
SPLITS = DATA / "splits"

RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

# results/ splits three ways, because three different questions get asked of it:
#
#   dataset/    what the data looks like — true before any model existed
#   models/     one self-contained folder per model, so "show me the CNN's
#               results" is a single directory and not a filename filter
#   overview/   everything that only exists by comparing models to each other
#
DATASET_FIG = RESULTS / "dataset"
FIG_PULSEDB = DATASET_FIG / "pulsedb"          # PulseDB only
FIG_DALIA = DATASET_FIG / "ppg_dalia"          # PPG-DaLiA only
FIG_COMPARE = DATASET_FIG / "both_datasets"    # the two side by side
FIG_PREP = DATASET_FIG / "preprocessing"       # evidence for the pipeline
FIG_LIT = DATASET_FIG / "literature"           # published work, not ours
FIG_PLAN = DATASET_FIG / "plan"                # the training-plan diagrams

MODEL_RESULTS = RESULTS / "models"             # models/<name>/{*.json,*.npz,figures/}
OVERVIEW = RESULTS / "overview"                # cross-model figures and the report


def model_dir(name, sub=None):
    """results/models/<name>/[sub] — created on demand."""
    p = MODEL_RESULTS / name / sub if sub else MODEL_RESULTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p

# PulseDB subset files, keyed by the short name used throughout the project.
PULSEDB_SUBSETS = {
    "train":         RAW_PULSEDB / "VitalDB_Train_Subset.mat",
    "calbased_test": RAW_PULSEDB / "VitalDB_CalBased_Test_Subset.mat",
    "calfree_test":  RAW_PULSEDB / "VitalDB_CalFree_Test_Subset.mat",
    "aami_cal":      RAW_PULSEDB / "VitalDB_AAMI_Cal_Subset.mat",
    "aami_test":     RAW_PULSEDB / "VitalDB_AAMI_Test_Subset.mat",
}
