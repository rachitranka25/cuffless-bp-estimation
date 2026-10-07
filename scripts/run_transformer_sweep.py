"""Transformer sweep: all four protocols, seeds 0-2 where reported, plus the
calfree_nobalance ablation and the PPG-DaLiA domain-shift check.

Unlike CNN/ResNet1D (run_50epoch_queue.py) the Transformer uses the DMT recipe
(arXiv:2606.11125) rather than the project's CNN/ResNet defaults — PPG-only
input, per-clip z-score scaling, Adam with a fixed learning rate, and the
morphology-classification auxiliary head — so every job below passes DMT_KW
explicitly rather than relying on Config's CNN/ResNet-tuned defaults. The two
deliberate deviations from the published recipe (one joint SBP+DBP network
instead of two, and 50 epochs instead of 100) are explained where they first
matter, in src/train.py's Config docstring.

The 13 training jobs actually run (documented in HANDOFF.md's "professor's
PC" section) were executed by hand, cell by cell, from
notebooks/03c_transformer_colab.ipynb on a separate machine with a real GPU —
not from a standalone queue script, since at the time this was a one-off
rather than a repeated sweep. This script is a batch-script equivalent,
written afterward from the same Config values that notebook used (verified
against the (protocol, seed, tag, balance) recorded in every
results/models/transformer/*.json file's own "config" field — all 13 match
exactly). It exists so the full job list is runnable and inspectable as code,
not only as notebook cells.

run() covers the 13 training jobs; evaluate_dalia() covers the 6 label-free
PPG-DaLiA domain-shift checks of the already-trained calfree checkpoints. The
two further results/models/transformer/*personalization*.json files are not
training jobs — they reuse the calfree checkpoint through
notebooks/04_personalization.ipynb's personalize() call, same as every other
model.

Run:  python3 -u scripts/run_transformer_sweep.py   (from repo root or src/)
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run, evaluate_dalia

DMT_KW = dict(
    epochs=50, batch_size=32, lr=2e-5, weight_decay=1e-8,
    channels=(1,), optimizer="adam", lr_schedule="constant",
    zscore=True, aux_morphology=True,
)

# (protocol, seed, tag, balance) in priority order — calfree (the headline
# protocol) first, its three seeds next, then the nobalance ablation and its
# seeds, then the remaining three protocols.
JOBS = [
    ("calfree",  0, "",               True),
    ("calfree",  1, "seed1",          True),
    ("calfree",  2, "seed2",          True),
    ("calfree",  0, "nobalance",      False),
    ("calfree",  1, "nobalance_seed1", False),
    ("calfree",  2, "nobalance_seed2", False),
    ("leaky",    0, "",               True),
    ("leaky",    1, "seed1",          True),
    ("leaky",    2, "seed2",          True),
    ("calbased", 0, "",               True),
    ("calbased", 1, "seed1",          True),
    ("calbased", 2, "seed2",          True),
    ("aami",     0, "",               True),
]

# evaluate_dalia() tags matching the calfree checkpoints trained above —
# label-free scoring only, no new training.
DALIA_TAGS = ["", "nobalance", "seed1", "seed2", "nobalance_seed1", "nobalance_seed2"]

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_transformer.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    open(LOG, "a").close()
    log(f"=== transformer queue starting, {len(JOBS)} training jobs, "
        f"{len(DALIA_TAGS)} domain-shift evaluations ===")
    done, failed = 0, 0
    for i, (protocol, seed, tag, balance) in enumerate(JOBS, 1):
        label = f"transformer {protocol} seed={seed} tag={tag!r} balance={balance}"
        log(f"[{i}/{len(JOBS)}] START {label}")
        t0 = time.time()
        try:
            run(model="transformer", protocol=protocol, seed=seed, tag=tag,
                balance=balance, num_workers=4, **DMT_KW)
            done += 1
            log(f"[{i}/{len(JOBS)}] OK {label} ({time.time()-t0:.0f}s)")
        except Exception:
            failed += 1
            log(f"[{i}/{len(JOBS)}] FAILED {label}\n{traceback.format_exc()}")

    for tag in DALIA_TAGS:
        label = f"evaluate_dalia transformer tag={tag!r}"
        log(f"START {label}")
        try:
            evaluate_dalia(model="transformer", protocol="calfree", tag=tag)
            log(f"OK {label}")
        except Exception:
            log(f"FAILED {label}\n{traceback.format_exc()}")

    log(f"=== transformer queue finished: {done} ok, {failed} failed ===")


if __name__ == "__main__":
    main()
