"""Independent re-verification of the personalization numbers reported in the paper.

Reruns src.train.personalize() for all five models directly from the saved,
released per-clip predictions (no retraining), using the exact offset-only
(k<3) / 30%-shrunk-slope (k>=3) correction of Section III-F
(src/train.py:_affine_recalibration), and prints the k=25 naive-split
diastolic MAE and oracle-ceiling recovery fraction for each model so they can
be diffed by eye against Table 9 (ceiling) and Table 12 / Table 17 (naive
column) of the paper.

Run: python3 scripts/verify_personalization.py
"""
import sys
sys.path.insert(0, ".")
from src.train import personalize

MODELS = ["rf", "gb", "cnn", "resnet", "transformer"]
# CNN/ResNet1D's canonical calfree number uses the schedule-fixed checkpoint.
BASE_TAG = {"cnn": "schedfix", "resnet": "schedfix"}

print("=" * 90)
print("Re-verifying personalization numbers from saved per-clip predictions")
print("(offset-only k<3, 30%-shrunk-slope k>=3 -- src/train.py:_affine_recalibration)")
print("=" * 90)

for model in MODELS:
    result = personalize(
        model=model,
        source="calfree",
        shots=(0, 1, 3, 5, 10, 25),
        picks=("first",),
        save=False,
        base_tag=BASE_TAG.get(model, ""),
    )

print("\nDone. Compare the 'k=25' DBP MAE values above (pick=first) against")
print("Table 12 / Table 17's naive column, and the headroom lines against")
print("Table 9's oracle-ceiling recovery percentages, for an exact match.")
