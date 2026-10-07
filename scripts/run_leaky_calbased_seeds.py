"""Item 2 (professor's 3rd-round feedback): seed variance for leaky/calbased,
CNN and ResNet, seeds 1 and 2 — matches the existing seed-0 config exactly
(standard OneCycleLR, no lr_decay_epochs override; these protocols were never
directly implicated in the schedule-confound finding, only calfree was, so
they're not being schedule-fixed here — see the deck's Item 1/2 slides for why).

Paste into a Kaggle cell after the usual setup (BP_ROOT / sys.path already set).

Run:  python3 scripts/run_leaky_calbased_seeds.py   (or paste main()'s body directly)
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import run

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queue_leaky_calbased_seeds.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    jobs = [(m, p, s) for m in ("cnn", "resnet") for p in ("leaky", "calbased") for s in (1, 2)]
    log(f"=== leaky/calbased seed queue starting, {len(jobs)} jobs ===")
    for i, (m, p, seed) in enumerate(jobs, 1):
        tag = f"seed{seed}"
        label = f"{m} {p} seed={seed}"
        log(f"[{i}/{len(jobs)}] START {label}")
        t0 = time.time()
        try:
            run(model=m, protocol=p, epochs=50, seed=seed, tag=tag, batch_size=256)
            log(f"[{i}/{len(jobs)}] OK {label} ({time.time()-t0:.0f}s)")
        except Exception:
            log(f"[{i}/{len(jobs)}] FAILED {label}\n{traceback.format_exc()}")
    log("=== leaky/calbased seed queue finished ===")


if __name__ == "__main__":
    main()
