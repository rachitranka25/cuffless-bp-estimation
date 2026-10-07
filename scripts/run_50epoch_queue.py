"""Sequential queue: RF/GB seeds 1-2, then CNN and ResNet at 50 epochs,
all protocols, seeds 0-2, in priority order. Runs unattended — each step is
wrapped so one failure doesn't kill the rest of the queue.

Run:  python3 -u scripts/run_50epoch_queue.py   (from repo root or src/)
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run, evaluate_dalia

PROTOS = ["calfree", "calfree_nobalance", "leaky", "calbased", "aami"]

# (model, protocol, seed, tag, kw) in priority order — most valuable first
JOBS = []

# 1. calfree, seed 0, both deep models — the headline number
for m in ("cnn", "resnet"):
    JOBS.append((m, "calfree", 0, "", dict(epochs=50)))

# 2. RF/GB seeds 1,2 (fast, cheap, needed for the 3-seed variance claim)
for seed in (1, 2):
    for m in ("rf", "gb"):
        JOBS.append((m, "calfree", seed, f"seed{seed}", dict()))

# 3. CNN/ResNet calfree seeds 1,2 (variance for the headline claim)
for seed in (1, 2):
    for m in ("cnn", "resnet"):
        JOBS.append((m, "calfree", seed, f"seed{seed}", dict(epochs=50)))

# 4. calfree_nobalance, seed 0 — literature comparison + balancing slide
for m in ("cnn", "resnet"):
    JOBS.append((m, "calfree", 0, "nobalance", dict(epochs=50, balance=False)))

# 5. the other three protocols, seed 0
for p in ("leaky", "calbased", "aami"):
    for m in ("cnn", "resnet"):
        JOBS.append((m, p, 0, "", dict(epochs=50)))

# 6. lower priority: seeds 1,2 for the remaining protocols
for seed in (1, 2):
    for p in ("leaky", "calbased", "aami"):
        for m in ("cnn", "resnet"):
            JOBS.append((m, p, seed, f"seed{seed}", dict(epochs=50)))

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_50epoch.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    open(LOG, "a").close()
    log(f"=== queue starting, {len(JOBS)} training jobs ===")
    done, failed = 0, 0
    for i, (m, p, seed, tag, kw) in enumerate(JOBS, 1):
        label = f"{m} {p} seed={seed} tag={tag!r} {kw}"
        log(f"[{i}/{len(JOBS)}] START {label}")
        t0 = time.time()
        try:
            run(model=m, protocol=p, seed=seed, tag=tag, num_workers=4, **kw)
            done += 1
            log(f"[{i}/{len(JOBS)}] OK {label} ({time.time()-t0:.0f}s)")
        except Exception:
            failed += 1
            log(f"[{i}/{len(JOBS)}] FAILED {label}\n{traceback.format_exc()}")

    # domain-shift scoring for cnn/resnet, seed 0, once their calfree checkpoints exist
    for m in ("cnn", "resnet"):
        for tag in ("", "nobalance"):
            label = f"evaluate_dalia {m} tag={tag!r}"
            log(f"START {label}")
            try:
                evaluate_dalia(model=m, protocol="calfree", tag=tag)
                log(f"OK {label}")
            except Exception:
                log(f"FAILED {label}\n{traceback.format_exc()}")

    log(f"=== queue finished: {done} ok, {failed} failed ===")


if __name__ == "__main__":
    main()
