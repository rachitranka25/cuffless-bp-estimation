"""PPG-DaLiA domain-shift evaluation for the two classical models.

notebooks/02_baselines.ipynb covers RF/GB training on all four protocols and
the calfree_nobalance ablation (results/models/{rf,gb}/*.json), but — unlike
the deep-model notebooks (03a/03b/03c_*_colab.ipynb), which each have their
own evaluate_dalia() cells — never scored RF/GB's calfree checkpoint on
PPG-DaLiA. These two calls were run separately to produce
results/models/{rf,gb}/{rf,gb}_dalia_domain_shift.json and the
*_nobalance_dalia_domain_shift.json ablation.

RF/GB were not seed-varied here, matching results/models/{rf,gb}/ — only
CNN/ResNet1D/Transformer have seed1/seed2 domain-shift checks (their seed
variance was already established on the calfree protocol itself).

Run:  python3 scripts/evaluate_rfgb_dalia.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import evaluate_dalia

for model in ("rf", "gb"):
    for tag in ("", "nobalance"):
        print(f"=== evaluate_dalia {model} tag={tag!r} ===")
        evaluate_dalia(model=model, protocol="calfree", tag=tag)
