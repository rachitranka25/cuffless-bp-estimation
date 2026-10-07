"""One entry point for every model and every evaluation protocol.

The project's central claim is that a cuffless-BP number means nothing without
the protocol it was measured under. That claim only holds if the protocols are
the *only* thing that differs between runs — so every model and every protocol
goes through this one file. Notebooks are launchers: they set a model name and
call `run`. Nothing about the pipeline lives in a notebook, because five copies
of a pipeline drift, and then an architecture difference and a batch-size
difference look identical in the results table.

Five models, two families (spec section I):

    rf, gb              trees on the 61 hand-crafted features (features/*.parquet)
    cnn, resnet         convnets on the raw waveform    (processed/*/signals.npy)
    transformer         DMT-style, additionally conditioned on age / sex / BMI

Four protocols. Only the test set changes:

    leaky       clips split at random inside the training subset, so the same
                patients appear on both sides. Spec month 5 asks for this as a
                controlled illustration of the inflation, not as a result.
    calbased    official calbased_test — same 1,293 patients, unseen clips
    calfree     official calfree_test — 144 patients never seen. The headline.
    aami        official aami_test — 116 unseen patients, no calibration. This is
                the zero-shot point of the personalization curve; `personalize`
                fills in the rest.

Two things the spec asks for that are easy to forget once training starts, so
they are built in here rather than bolted on:

  * BP-bin balancing (section L, "stratified sampling or a weighted loss over BP
    value bins"). 55% of the training clips sit in 90-120 mmHg and 1.8% above
    160; left alone a model learns to answer "about 110" and is worst for the
    patients who matter most.
  * subject-disjointness assertions on every honest protocol. A silent leak
    would inflate exactly the number this project exists to measure.

Usage
    from train import run
    run(model="gb",  protocol="calfree")
    run(model="cnn", protocol="calfree", epochs=15)

    python3 -m train --model gb --protocol calfree
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PROC_PULSEDB, PROC_DALIA, FEATURES, RESULTS, model_dir   # noqa: E402
from evaluation import metrics                             # noqa: E402


def _archive_run(name, *files):
    """Copy this run's output files into results/archive/<timestamp>_<name>/
    — never overwritten, never deleted, independent of whatever the "live"
    results/models/ tree does later.

    Added after Experiment 1's original CNN/ResNet result files were lost to
    an `rm -rf` before the 50-epoch retrain, with no backup taken first. From
    here on, results/ is append-only: every run gets its own timestamped copy
    that nothing else ever touches, so a superseded result stays recoverable
    even after the "live" file it corresponds to is overwritten by a rerun.
    """
    import shutil
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest = RESULTS / "archive" / f"{stamp}_{name}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f is not None and f.exists():
            shutil.copy2(f, dest / f.name)
    return dest

CLASSICAL = ("rf", "gb")
DEEP = ("cnn", "resnet", "transformer")
PROTOCOLS = ("leaky", "calbased", "calfree", "aami")

# Bin edges for the imbalance correction. Chosen on clinical categories rather
# than quantiles so the rare bins are the clinically extreme ones, which is the
# point — quantile bins would be equally populated by construction.
SBP_BINS = (0, 90, 110, 130, 150, 170, 1000)

DROP_COLS = ("subject", "sbp", "dbp")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    model: str = "gb"
    protocol: str = "calfree"
    seed: int = 0

    # data
    cap: int | None = None        # max clips per training subject; None = all
    channels: tuple = (0, 1)      # 0 = ECG, 1 = PPG
    val_frac: float = 0.1         # held-out *subjects* for early stopping
    balance: bool = True          # spec L: weight the BP bins

    # deep only
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    # OneCycleLR only reaches its lowest learning rate in the final epochs, and
    # that is where its last improvement arrives — so by default nothing stops
    # before the schedule has finished. The best epoch is checkpointed either
    # way, so running the full budget costs time and nothing else. Lower this to
    # re-enable early stopping.
    min_epoch_frac: float = 1.0
    width: int = 32
    num_workers: int = 2
    device: str | None = None     # None = pick the best available

    # None (default): OneCycleLR's schedule spans the full `epochs` budget, as
    # before — meaning a 20-epoch run and a 50-epoch run don't just train for
    # different lengths, they anneal the learning rate on different curves, so
    # comparing "20 epochs" against "50 epochs" conflates two changes at once.
    # Set this to a fixed epoch count (e.g. 20) to decouple them: the schedule
    # anneals over exactly that many epochs regardless of `epochs`, then holds
    # flat at the final learning rate for the rest — so the first N epochs of
    # a 20-epoch and a 50-epoch run are trained identically, and any
    # difference beyond that is genuinely "more training", not "a different
    # schedule shape".
    lr_decay_epochs: int | None = None

    # transformer-only, DMT-recipe knobs — no effect on rf/gb/cnn/resnet.
    # Defaults below are arXiv:2606.11125's own reported settings (Adam,
    # betas (0.9, 0.999) — torch's own default so nothing to set explicitly,
    # weight_decay 1e-8, fixed lr, batch 32, z-score, shape-classification aux
    # head with learnable multi-task uncertainty weights). Two deliberate
    # deviations, both by explicit instruction rather than limitation: one
    # joint network for SBP+DBP instead of two, and 50 epochs instead of 100,
    # to keep the training budget comparable to the CNN/ResNet runs.
    optimizer: str = "adamw"          # "adamw" (default) or "adam", DMT's choice
    lr_schedule: str = "onecycle"     # "onecycle" (default) or "constant" — DMT
                                      # trains at a single fixed lr the whole run
    zscore: bool = False              # per-clip z-score instead of the stored min-max
    aux_morphology: bool = False      # DMT's auxiliary morphology *classification*
                                      # head + Kendall-style learnable multi-task
                                      # uncertainty weights — see models.DMT and
                                      # _shape_labels for what this approximates

    # bookkeeping
    tag: str = ""
    out_dir: str = ""
    save_ckpt: bool = True
    extra: dict = field(default_factory=dict)

    def name(self):
        t = f"_{self.tag}" if self.tag else ""
        return f"{self.model}_{self.protocol}{t}"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def _subjects(split):
    return np.load(PROC_PULSEDB / split / "subjects.npy", allow_pickle=True)


def _cap_indices(subjects, cap, seed):
    """At most `cap` clips per subject — for quick runs, not for final numbers."""
    if not cap:
        return np.arange(subjects.size)
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out.append(idx if idx.size <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(out))


def _hold_out_subjects(subjects, indices, frac, seed):
    """Split indices by subject, never by clip. Returns (kept, held)."""
    subs = np.unique(subjects[indices])
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    held = set(subs[:max(1, int(len(subs) * frac))].tolist())
    mask = np.fromiter((s in held for s in subjects[indices]), bool, len(indices))
    return indices[~mask], indices[mask]


def bin_weights(sbp, bins=SBP_BINS):
    """Per-clip weights that flatten the SBP histogram (spec section L).

    Each clip is weighted by the inverse frequency of its BP bin, normalised to
    mean 1 so the effective learning rate does not change with the binning. A
    clip above 170 mmHg ends up worth roughly thirty ordinary clips, which is
    the intent: those patients are the reason the method would exist.
    """
    idx = np.digitize(np.asarray(sbp, dtype=float), bins[1:-1])
    counts = np.bincount(idx, minlength=len(bins) - 1).astype(float)
    counts[counts == 0] = np.inf                  # empty bin -> zero weight
    w = (1.0 / counts)[idx]
    return w / w.mean()


def build_protocol(protocol, seed=0, cap=None, val_frac=0.1):
    """Index bundle for one protocol. Training data is always the train subset.

    Returns train / val / test index arrays plus, for the test side, which split
    on disk they index into. Val subjects are held out of training under *every*
    protocol including leaky — early stopping on a leaky validation set would
    leak a second time, through model selection.
    """
    tr_subj = _subjects("train")
    pool = _cap_indices(tr_subj, cap, seed)

    if protocol == "leaky":
        rng = np.random.default_rng(seed)
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        n_test = int(len(shuffled) * 0.2)
        test_idx = np.sort(shuffled[:n_test])
        rest = np.sort(shuffled[n_test:])
        train_idx, val_idx = _hold_out_subjects(tr_subj, rest, val_frac, seed + 1)
        test_split = "train"
    elif protocol in ("calbased", "calfree", "aami"):
        train_idx, val_idx = _hold_out_subjects(tr_subj, pool, val_frac, seed + 1)
        test_split = {"calbased": "calbased_test",
                      "calfree": "calfree_test",
                      "aami": "aami_test"}[protocol]
        test_idx = np.arange(len(_subjects(test_split)))
    else:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")

    b = dict(protocol=protocol, train_split="train", test_split=test_split,
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    b["n_shared_subjects"] = _assert_split_sanity(b)
    return b


def _assert_split_sanity(b):
    """Guard every honest protocol. `leaky` is expected to overlap by design."""
    tr_subj = _subjects(b["train_split"])
    te_subj = _subjects(b["test_split"])
    tr = set(tr_subj[b["train_idx"]].tolist())
    va = set(tr_subj[b["val_idx"]].tolist())
    te = set(te_subj[b["test_idx"]].tolist())

    if tr & va:
        raise AssertionError(f"{len(tr & va)} subjects shared between train and val")
    shared = len(tr & te)
    if b["protocol"] == "leaky":
        return shared                      # the whole point of this protocol
    if b["protocol"] == "calbased":
        return shared                      # official split, overlap is by design
    if shared:
        raise AssertionError(
            f"LEAKAGE in '{b['protocol']}': {shared} subjects in both train and test")
    return 0


# --------------------------------------------------------------------------- #
# classical: trees on the hand-crafted features
# --------------------------------------------------------------------------- #

def load_features(split):
    df = pd.read_parquet(FEATURES / f"{split}.parquet")
    cols = [c for c in df.columns if c not in DROP_COLS]
    return (df[cols].to_numpy(dtype=np.float32), cols,
            df[["sbp", "dbp"]].to_numpy(dtype=np.float64),
            df["subject"].to_numpy())


def load_dalia_features():
    """PPG-DaLiA in the same 61 columns and the same order as PulseDB.

    Column order matters: a tree indexes features positionally, so a reordered
    frame would silently feed pulse width into the slot the model learned as
    transit time. Reindexing against the training columns makes that impossible
    rather than merely unlikely.
    """
    cols = [c for c in pd.read_parquet(FEATURES / "train.parquet").columns
            if c not in DROP_COLS]
    df = pd.read_parquet(FEATURES / "dalia.parquet")
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"dalia.parquet is missing {sorted(missing)}")
    return df[cols].to_numpy(dtype=np.float32), df["subject"].to_numpy()


def _fit_classical(cfg, b, also_score=()):
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

    Xtr_all, cols, ytr_all, _ = load_features(b["train_split"])
    Xtr, ytr = Xtr_all[b["train_idx"]], ytr_all[b["train_idx"]]

    def take(split, idx):
        if split == "dalia":
            X, s = load_dalia_features()
            return X, None, s                      # no BP labels exist there
        if split == b["train_split"]:
            return Xtr_all[idx], ytr_all[idx], _subjects(split)[idx]
        X, _, y, s = load_features(split)
        return X[idx], y[idx], s[idx]

    Xte, yte, subj_te = take(b["test_split"], b["test_idx"])
    w = bin_weights(ytr[:, 0]) if cfg.balance else None

    if cfg.model == "rf":
        # RandomForest cannot consume NaN; impute with the training median so the
        # imputation itself cannot leak anything from the test subjects.
        med = np.nanmedian(Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        impute = lambda X: np.where(np.isfinite(X), X, med)      # noqa: E731
        Xtr, Xte = impute(Xtr), impute(Xte)
        make = lambda: RandomForestRegressor(       # noqa: E731
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            random_state=cfg.seed, **cfg.extra)
    else:
        impute = lambda X: X                        # noqa: E731
        make = lambda: HistGradientBoostingRegressor(   # noqa: E731
            max_iter=500, learning_rate=0.06, early_stopping=True,
            validation_fraction=0.1, random_state=cfg.seed, **cfg.extra)

    fitted, preds, fit_s = [], [], []
    for k, target in enumerate(("sbp", "dbp")):
        t0 = time.time()
        m = make()
        m.fit(Xtr, ytr[:, k], sample_weight=w)
        fitted.append(m)
        preds.append(m.predict(Xte))
        fit_s.append(round(time.time() - t0, 1))
        print(f"  fitted {cfg.model} for {target} in {fit_s[-1]}s")

    info = dict(n_features=len(cols), fit_s=fit_s, features=cols)
    if also_score:
        info["also"] = {}
        for split, idx in also_score:
            X, y, s = take(split, idx)
            X = impute(X)
            info["also"][split] = (np.stack([m.predict(X) for m in fitted], 1), y, s)
    return np.stack(preds, axis=1), yte, subj_te, info


def dalia_windows():
    """(64682, 2, 1250) float16 and the subject id per window."""
    w = PROC_DALIA / "windows"
    return (np.load(w / "signals.npy", mmap_mode="r"),
            np.load(w / "subjects.npy", allow_pickle=True))


def dalia_demographics(stats):
    """Age / sex / BMI per DaLiA window, standardised with PulseDB's statistics.

    The transformer conditions on these, so they have to be on the same scale
    the model was trained with — standardising DaLiA by its own mean would tell
    the model that a 21-year-old is average when the training population's mean
    age is far higher.
    """
    npy = PROC_DALIA / "windows" / "demographics.npy"
    if npy.exists():                       # see bpdata.demographics for why
        d = np.load(npy).astype(np.float64)
        age, sex, bmi = d[:, 0], d[:, 1], d[:, 2]
    else:
        meta = pd.read_parquet(FEATURES / "dalia.parquet")
        age = meta["demo_age"].to_numpy(dtype=np.float64)
        bmi = meta["demo_bmi"].to_numpy(dtype=np.float64)
        sex = meta["demo_male"].to_numpy(dtype=np.float64)
    out = np.stack([
        np.nan_to_num((age - stats["age"][0]) / stats["age"][1], nan=0.0),
        sex,
        np.nan_to_num((bmi - stats["bmi"][0]) / stats["bmi"][1], nan=0.0),
    ], axis=1).astype(np.float32)
    return out, stats


# DMT (arXiv:2606.11125 §IV.A) labels each clip normotensive- or hypertensive-
# -like from the segment-level augmentation index (AI), b/a ratio (BA, from
# the PPG's second derivative), and normalised dicrotic-notch depth (ND),
# thresholded against an "aggregated morphology score" the paper cites from
# unrestated prior work rather than defining inline — so its exact rule isn't
# reproducible. Standing in for it: the closest already-computed columns
# (features/*.parquet, the same ones the rf/gb baselines use) as proxies for
# AI, BA and ND, combined into one composite and split at its training-set
# median. This is this project's own rule, not DMT's — a documented
# approximation of a component the source paper doesn't fully specify either.
SHAPE_PROXY_COLS = ("ppg_reflection_index", "der_apg_ba_ratio", "ppg_notch_pos")


def _shape_labels(split, idx):
    """Full-length (N,) int64 array of 0/1 shape labels, split at the median
    composite score on `idx` rows only (not the whole file, to keep the
    "median" meaningful for the training population the label is used on).

    Indexed by the row's position in the split file, same convention as
    `signals`/`labels`/`demo` — not by position in `idx`.
    """
    df = pd.read_parquet(FEATURES / f"{split}.parquet", columns=list(SHAPE_PROXY_COLS))
    vals = df.to_numpy(dtype=np.float64, copy=True)   # pandas 3 CoW hands back a read-only view
    for c in range(vals.shape[1]):
        col = vals[:, c]
        col[np.isnan(col)] = np.nanmean(col)
    mean, std = vals[idx].mean(0), vals[idx].std(0) + 1e-8
    z = (vals - mean) / std
    score = z[:, 0] + z[:, 1] - z[:, 2]           # AI + BA - ND, this project's composite
    return (score > np.median(score[idx])).astype(np.int64)


# --------------------------------------------------------------------------- #
# deep: convnets and the transformer on the raw waveform
# --------------------------------------------------------------------------- #

def _pick_device(cfg):
    import torch
    if cfg.device:
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Apple's backend is fast for convolutions and pathologically slow for
        # attention (92.8 ms vs 30.5 ms on CPU at this shape), so the transformer
        # opts out. Measured, not assumed — see src/models.py.
        return torch.device("cpu" if cfg.model == "transformer" else "mps")
    return torch.device("cpu")


def _fit_deep(cfg, b, also_score=()):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from bpdata import SegmentDataset, demographics
    from models import build, n_params

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = _pick_device(cfg)

    tr_dir, te_dir = PROC_PULSEDB / b["train_split"], PROC_PULSEDB / b["test_split"]
    sig_tr = np.load(tr_dir / "signals.npy", mmap_mode="r")
    lab_tr = np.load(tr_dir / "labels.npy")
    sig_te = np.load(te_dir / "signals.npy", mmap_mode="r")
    lab_te = np.load(te_dir / "labels.npy")
    subj_te = _subjects(b["test_split"])[b["test_idx"]]

    # standardise the targets on the training clips only
    y_mean = torch.tensor(lab_tr[b["train_idx"]].mean(0), dtype=torch.float32)
    y_std = torch.tensor(lab_tr[b["train_idx"]].std(0), dtype=torch.float32)

    use_demo = cfg.model == "transformer"
    d_tr = d_te = None
    if use_demo:
        d_tr, stats = demographics(b["train_split"])
        d_te = (d_tr if b["test_split"] == b["train_split"]
                else demographics(b["test_split"], stats)[0])

    ch = list(cfg.channels)
    use_shape = cfg.model == "transformer" and cfg.aux_morphology
    shape_tr = _shape_labels(b["train_split"], b["train_idx"]) if use_shape else None
    ds = lambda s, l, i, d, m=None: SegmentDataset(     # noqa: E731
        s, l, i, y_mean, y_std, demo=d, channels=ch, zscore=cfg.zscore, morph=m)
    tr_ds = ds(sig_tr, lab_tr, b["train_idx"], d_tr, shape_tr)
    va_ds = ds(sig_tr, lab_tr, b["val_idx"], d_tr)
    te_ds = ds(sig_te, lab_te, b["test_idx"], d_te)

    # Spec L, deep side: sample the rare BP bins more often instead of weighting
    # the loss, so batch statistics see the whole BP range too.
    if cfg.balance:
        w = bin_weights(lab_tr[b["train_idx"], 0])
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        shuffle = False
    else:
        sampler, shuffle = None, True

    dl = lambda d, **kw: DataLoader(d, batch_size=cfg.batch_size,             # noqa: E731
                                    num_workers=cfg.num_workers,
                                    pin_memory=(dev.type == "cuda"), **kw)
    tr_dl = dl(tr_ds, sampler=sampler, shuffle=shuffle, drop_last=True)
    va_dl = dl(va_ds, shuffle=False)
    te_dl = dl(te_ds, shuffle=False)

    kw = dict(width=cfg.width) if cfg.model in ("cnn", "resnet") else {}
    if cfg.model == "transformer":
        kw["shape_head"] = use_shape
    net = build({"transformer": "dmt"}.get(cfg.model, cfg.model),
                in_ch=len(ch), **kw, **cfg.extra).to(dev)
    print(f"  {cfg.model} on {dev}, {n_params(net):,} parameters, "
          f"{len(tr_ds):,} train / {len(va_ds):,} val / {len(te_ds):,} test clips")

    opt_cls = torch.optim.Adam if cfg.optimizer == "adam" else torch.optim.AdamW
    opt = opt_cls(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = sched_steps = None
    if cfg.lr_schedule == "onecycle":
        decay_epochs = cfg.lr_decay_epochs or cfg.epochs
        sched_steps = decay_epochs * max(1, len(tr_dl))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=sched_steps)
    lossf = torch.nn.SmoothL1Loss()
    sched_step_count = 0

    def forward(batch):
        # `batch[-1]` is always y regardless of whether morph sits in between
        # demo and y — only the training loop below ever needs the morph slot.
        if use_demo:
            x, d, y = batch[0], batch[1], batch[-1]
            return net(x.to(dev), d.to(dev)), y.to(dev)
        x, y = batch
        return net(x.to(dev)), y.to(dev)

    best, best_state, bad, history = np.inf, None, 0, []
    for ep in range(1, cfg.epochs + 1):
        net.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for batch in tr_dl:
            opt.zero_grad(set_to_none=True)
            out, y = forward(batch)
            if use_shape:
                # Eq. 9 of arXiv:2606.11125: L = (1/s_bp)*L_bp + log(s_bp)
                #                                + (1/s_shape)*L_shape + log(s_shape)
                # with s_bp, s_shape learnable — reparametrised as log(s) here
                # (see models.DMT) so the exp(-log_s) + log_s form is exact.
                shape_y = batch[2].to(dev).long()
                l_bp = torch.nn.functional.l1_loss(out, y)           # paper uses MAE, not SmoothL1
                l_shape = torch.nn.functional.cross_entropy(net.last_shape_logits, shape_y)
                lsb, lss = net.log_sigma_bp, net.log_sigma_shape
                loss = (torch.exp(-lsb) * l_bp + lsb
                        + torch.exp(-lss) * l_shape + lss).squeeze()
            else:
                loss = lossf(out, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {ep} ({loss.item()}) — stopping now "
                    "rather than burning the rest of the budget on a broken run.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if sched is not None and sched_step_count < sched_steps:
                sched.step()
                sched_step_count += 1
            # once the decay budget is spent, the optimizer just keeps using
            # whatever lr OneCycleLR last set — a flat continuation, not a
            # second anneal
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

        net.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in va_dl:
                out, y = forward(batch)
                vl += lossf(out, y).item() * y.size(0)
                vn += y.size(0)
        vl /= max(vn, 1)
        history.append(dict(epoch=ep, train_loss=run_loss / max(seen, 1),
                            val_loss=vl, seconds=round(time.time() - t0, 1)))
        print(f"  epoch {ep:>3}/{cfg.epochs}  train {run_loss / max(seen, 1):.4f}  "
              f"val {vl:.4f}  {history[-1]['seconds']:.0f}s")

        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            # OneCycleLR anneals the learning rate over the *full* epoch budget,
            # and most of the gain arrives in the last third when the rate is
            # lowest. Stopping while the rate is still high reads a noisy plateau
            # as convergence and throws that away — so no stop before the
            # schedule has largely run its course.
            if bad >= cfg.patience and ep >= cfg.min_epoch_frac * cfg.epochs:
                if ep < cfg.epochs:
                    print(f"  early stop at epoch {ep} (best val {best:.4f})")
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()

    def predict(loader):
        out = []
        with torch.no_grad():
            for batch in loader:
                p, _ = forward(batch)
                out.append(p.cpu())
        return (torch.cat(out) * y_std + y_mean).numpy()

    pred = predict(te_dl)

    info_also = {}
    for split, idx in also_score:
        if split == "dalia":
            s2, subj2 = dalia_windows()
            l2 = np.zeros((len(s2), 2), dtype=np.float32)   # placeholder, unused
            idx2 = np.arange(len(s2))
            d2 = dalia_demographics(stats)[0] if use_demo else None
            info_also[split] = (predict(dl(ds(s2, l2, idx2, d2), shuffle=False)),
                                None, subj2)
            continue
        s2 = np.load(PROC_PULSEDB / split / "signals.npy", mmap_mode="r")
        l2 = np.load(PROC_PULSEDB / split / "labels.npy")
        d2 = None
        if use_demo:
            d2 = d_tr if split == b["train_split"] else demographics(split, stats)[0]
        info_also[split] = (predict(dl(ds(s2, l2, idx, d2), shuffle=False)),
                            l2[idx], _subjects(split)[idx])

    if cfg.save_ckpt:
        torch.save(dict(state_dict=net.state_dict(), cfg=asdict(cfg),
                        y_mean=y_mean, y_std=y_std),
                   model_dir(cfg.model, "checkpoints") / f"{cfg.name()}.pt")

    return pred, lab_te[b["test_idx"]], subj_te, dict(
        device=str(dev), n_params=n_params(net), epochs_run=len(history),
        best_val_loss=best, history=history, also=info_also)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def run(model="gb", protocol="calfree", save=True, **kw):
    """Train one model under one protocol, score it, and write the JSON."""
    cfg = Config(model=model, protocol=protocol, **kw)
    if cfg.model not in CLASSICAL + DEEP:
        raise ValueError(f"unknown model {cfg.model!r}; expected {CLASSICAL + DEEP}")

    print(f"\n=== {cfg.name()} ===")
    b = build_protocol(cfg.protocol, seed=cfg.seed, cap=cfg.cap, val_frac=cfg.val_frac)
    print(f"  train {len(b['train_idx']):,} clips / "
          f"{len(np.unique(_subjects('train')[b['train_idx']])):,} subjects")
    print(f"  test  {len(b['test_idx']):,} clips from '{b['test_split']}' / "
          f"{len(np.unique(_subjects(b['test_split'])[b['test_idx']])):,} subjects "
          f"({b['n_shared_subjects']:,} also in train)")
    print(f"  BP-bin balancing: {'on' if cfg.balance else 'off'}")

    t0 = time.time()
    fit = _fit_classical if cfg.model in CLASSICAL else _fit_deep
    pred, y_true, subj, info = fit(cfg, b)

    out = dict(config=asdict(cfg), protocol=cfg.protocol, model=cfg.model,
               train_split=b["train_split"], test_split=b["test_split"],
               n_train=int(len(b["train_idx"])), n_val=int(len(b["val_idx"])),
               n_test=int(len(b["test_idx"])),
               n_train_subjects=int(len(np.unique(_subjects("train")[b["train_idx"]]))),
               n_shared_subjects=int(b["n_shared_subjects"]),
               total_seconds=round(time.time() - t0, 1), info=info, results={})

    for k, target in enumerate(("sbp", "dbp")):
        m = metrics(y_true[:, k], pred[:, k], subj)
        out["results"][target] = m
        print(f"  {target.upper():4s} MAE {m['mae']:6.2f}  ME {m['me']:+6.2f}  "
              f"SD {m['sde']:6.2f}  r {m['r']:5.2f}  "
              f"AAMI {'PASS' if m['aami_pass'] else 'FAIL'}  BHS {m['bhs']}")

    if save:
        d = (__import__("pathlib").Path(cfg.out_dir) if cfg.out_dir
             else model_dir(cfg.model))
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{cfg.name()}.json"
        npz_path = d / f"{cfg.name()}_predictions.npz"
        # predictions go beside the metrics so Bland-Altman plots (spec K) can be
        # redrawn without retraining
        np.savez_compressed(npz_path,
                            y_true=y_true, y_pred=pred, subjects=subj.astype(str))
        path.write_text(json.dumps(out, indent=2, default=float))
        print(f"  wrote {path}")
        ckpt = (model_dir(cfg.model, "checkpoints") / f"{cfg.name()}.pt"
                if cfg.model in DEEP and cfg.save_ckpt else None)
        _archive_run(cfg.name(), path, npz_path, ckpt)
    return out


# --------------------------------------------------------------------------- #
# subject-level k-fold, for hyperparameter choice (spec section L)
# --------------------------------------------------------------------------- #

def tune(model="gb", grid=None, folds=5, cap=100, seed=0, metric="sbp"):
    """Subject-level k-fold inside the training subjects only.

    Folds are formed over subjects, never over clips, so a candidate is never
    rewarded for memorising a patient. Test subsets are not touched at all.
    Intended for the tree models, where a fold costs seconds; deep models pick
    their stopping point on the held-out validation subjects instead.
    """
    from sklearn.model_selection import GroupKFold

    if model not in CLASSICAL:
        raise ValueError("tune() is for the tree models; deep models use early stopping")
    grid = grid or [{}]
    X, _, y, subj = load_features("train")
    idx = _cap_indices(subj, cap, seed)
    X, y, subj = X[idx], y[idx], subj[idx]
    col = 0 if metric == "sbp" else 1
    gkf = GroupKFold(n_splits=folds)

    scored = []
    for params in grid:
        maes = []
        for tr, va in gkf.split(X, y[:, col], groups=subj):
            cfg = Config(model=model, seed=seed, extra=dict(params))
            fake = dict(protocol="honest", train_split="train", test_split="train",
                        train_idx=tr, val_idx=tr[:1], test_idx=va, n_shared_subjects=0)
            pred, yte, _, _ = _fit_classical(cfg, fake)
            maes.append(float(np.mean(np.abs(pred[:, col] - yte[:, col]))))
        scored.append((float(np.mean(maes)), float(np.std(maes)), params))
        print(f"  {params}  {metric} MAE {scored[-1][0]:.3f} +/- {scored[-1][1]:.3f}")
    scored.sort(key=lambda r: r[0])
    print(f"  best: {scored[0][2]}  MAE {scored[0][0]:.3f}")
    return scored


# --------------------------------------------------------------------------- #
# few-shot personalization (spec section J, ablation 2 of section L)
# --------------------------------------------------------------------------- #

def _shot_indices(subjects, k, seed=0, pick="first", labels=None):
    """Which k clips of each subject are used as that subject's calibration.

    Spec section L says "the first small % of that subject's segments", and for
    a chronological recording that is the right instinct — a device calibrates
    when you put it on and cannot sample from your future.

    It does not hold on PulseDB's AAMI subsets, and the mismatch is large enough
    to make calibration actively harmful if ignored. `aami_test` is built to
    span the full BP range on purpose, so its clips average 134.9 mmHg SBP while
    the first five clips of `aami_cal` average 119.7 — a systematic -15.2 mmHg
    offset that a first-k calibration then bakes into every prediction. The
    first five also span only ~13 mmHg, which is far too narrow to fit a slope
    on. Hence three strategies, reported side by side rather than one asserted:

        first    the spec's literal reading; correct for a chronological stream
        random   k drawn uniformly from the subject's calibration clips
        spread   k chosen at even quantiles of the subject's calibration BP, so
                 the readings cover that person's range — what a careful
                 calibration protocol would actually ask a user to do
    """
    rng = np.random.default_rng(seed)
    out = {}
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        if k <= 0 or idx.size == 0:
            out[s] = idx[:0]
        elif pick == "first" or idx.size <= k:
            out[s] = idx[:k]
        elif pick == "random":
            out[s] = np.sort(rng.choice(idx, k, replace=False))
        elif pick == "spread":
            order = idx[np.argsort(labels[idx, 0])]
            q = np.linspace(0, order.size - 1, k).round().astype(int)
            out[s] = np.sort(order[np.unique(q)])
        else:
            raise ValueError(f"unknown pick {pick!r}")
    return out


def _affine_recalibration(cal_pred, cal_true, shrink=0.3):
    """Per-subject (scale, offset) from a handful of labelled clips.

    With one or two shots only the offset is identifiable — a slope fitted to
    two points is noise. From three shots up the slope is fitted but shrunk
    toward 1, because a handful of clips from one sitting spans a narrow BP
    range and an unshrunk slope extrapolates wildly outside it.
    """
    if len(cal_pred) == 0:
        return 1.0, 0.0
    if len(cal_pred) < 3 or np.ptp(cal_pred) < 1e-6:
        return 1.0, float(np.mean(cal_true - cal_pred))
    a, b = np.polyfit(cal_pred, cal_true, 1)
    a = shrink * 1.0 + (1 - shrink) * a
    b = float(np.mean(cal_true) - a * np.mean(cal_pred))
    return float(a), b


def offset_headroom(y_true, y_pred, subjects):
    """Ceiling for any per-subject offset method, and where the error lives.

    Gives each subject the best possible constant offset, fitted on its own test
    clips. Unachievable by construction — that is the point: it says how much of
    the error a calibration step could remove even in principle, so a modest
    real gain can be read against the right scale. Reported alongside the share
    of error variance that sits between subjects rather than within them.
    """
    err = np.asarray(y_pred) - np.asarray(y_true)
    corrected = err.copy()
    for s in np.unique(subjects):
        m = subjects == s
        corrected[m] = err[m] - np.median(err[m])
    between = float(np.var([err[subjects == s].mean() for s in np.unique(subjects)]))
    return dict(mae=float(np.mean(np.abs(err))),
                oracle_offset_mae=float(np.mean(np.abs(corrected))),
                recoverable_frac=float(1 - np.mean(np.abs(corrected)) / np.mean(np.abs(err))),
                between_subject_variance_share=between / float(np.var(err)))


def personalize(model="gb", source="calfree", shots=(0, 1, 3, 5, 10, 25),
                picks=("first", "random", "spread"), seed=0, balance=True, save=True, **kw):
    """How much of the calibration-free error a few labelled clips recover.

    Spec section L: "for a held-out subject, use only the first small % of that
    subject's segments as calibration data and evaluate on the remainder". Two
    ways to satisfy that, and they are not equivalent:

    source="calfree"  the literal reading, and the default. One subset, split
                      per subject: the first k of a subject's 400 calfree_test
                      clips calibrate, the other ~395 are scored. Calibration
                      clips are removed from the metric, so k=0 and k=25 are
                      scored on slightly different clip counts — the same clips
                      are dropped from every arm at a given k, so the arms stay
                      comparable.

    source="aami"     PulseDB's own aami_cal / aami_test pair. Reported as a
                      secondary result because the two files are not drawn the
                      same way: aami_test is built to span the full BP range on
                      purpose, so its clips average 134.9 mmHg SBP while the
                      first five aami_cal clips of the same patients average
                      119.7. A first-k calibration therefore bakes in a -15 mmHg
                      offset and makes the error worse. That is a property of
                      the subsets, not of personalization, and is worth
                      reporting as such.

    k=0 is the uncorrected number, so the curve starts at the honest baseline
    and any drop is readable straight off it.
    """
    if source not in ("calfree", "aami"):
        raise ValueError("source must be 'calfree' or 'aami'")
    protocol = "calfree" if source == "calfree" else "aami"
    cfg = Config(model=model, protocol=protocol, seed=seed, **kw)
    print(f"\n=== personalize: {cfg.model}, source={source} ===")

    if source == "calfree":
        # `run(model, protocol="calfree")` already scored every clip in this
        # pool and saved it — reuse that instead of re-fitting the model.
        # Calibration and scoring clips both come from the same pool, so the
        # saved test-split predictions are all this needs. CPU, no GPU, and it
        # takes seconds instead of the hours a re-fit would cost.
        nb = "" if balance else "_nobalance"
        npz_path = model_dir(cfg.model) / f"{cfg.model}_calfree{nb}_predictions.npz"
        if npz_path.exists():
            d = np.load(npz_path)
            p_pool, y_pool, s_pool = d["y_pred"], d["y_true"], d["subjects"]
            print(f"  loaded {len(p_pool):,} saved predictions from {npz_path.name}")
        else:
            fit = _fit_classical if cfg.model in CLASSICAL else _fit_deep
            b = build_protocol(protocol, seed=cfg.seed, cap=cfg.cap, val_frac=cfg.val_frac)
            p_pool, y_pool, s_pool, _ = fit(cfg, b)
        p_cal, y_cal, s_cal = p_pool, y_pool, s_pool
    else:
        # aami_cal was never scored by run(), so this path still re-fits —
        # the same model must score both aami_test and aami_cal.
        fit = _fit_classical if cfg.model in CLASSICAL else _fit_deep
        b = build_protocol(protocol, seed=cfg.seed, cap=cfg.cap, val_frac=cfg.val_frac)
        cal_idx = np.arange(len(_subjects("aami_cal")))
        p_pool, y_pool, s_pool, info = fit(cfg, b, also_score=[("aami_cal", cal_idx)])
        p_cal, y_cal, s_cal = info["also"]["aami_cal"]
    print(f"  base model scored on {len(p_pool):,} test and {len(p_cal):,} "
          f"calibration clips, {len(np.unique(s_pool))} subjects")

    head = {t: offset_headroom(y_pool[:, i], p_pool[:, i], s_pool)
            for i, t in enumerate(("sbp", "dbp"))}
    for t, h in head.items():
        print(f"  {t.upper()} headroom: MAE {h['mae']:.2f} -> {h['oracle_offset_mae']:.2f} "
              f"with a perfect per-subject offset "
              f"({h['recoverable_frac'] * 100:.0f}% recoverable, "
              f"{h['between_subject_variance_share'] * 100:.0f}% of error variance "
              f"is between-subject)")

    curves = {}
    for pick in picks:
        rows = []
        print(f"  --- calibration clips picked: {pick} ---")
        for k in shots:
            take = _shot_indices(s_cal, k, seed, pick, y_cal)
            corrected = p_pool.copy()
            keep = np.ones(len(p_pool), bool)
            for s in np.unique(s_pool):
                mask = s_pool == s
                cal_idx = take.get(s, np.array([], dtype=int))
                if source == "calfree":
                    keep[cal_idx] = False        # never score a calibration clip
                for t in (0, 1):
                    a, off = _affine_recalibration(p_cal[cal_idx, t], y_cal[cal_idx, t])
                    corrected[mask, t] = a * p_pool[mask, t] + off
            row = dict(shots=int(k), n_scored=int(keep.sum()), results={})
            for t, target in enumerate(("sbp", "dbp")):
                row["results"][target] = metrics(y_pool[keep, t], corrected[keep, t],
                                                 s_pool[keep])
            rows.append(row)
            print(f"    k={k:<3} n={row['n_scored']:>6,}  "
                  f"SBP MAE {row['results']['sbp']['mae']:6.2f}  "
                  f"DBP MAE {row['results']['dbp']['mae']:6.2f}  "
                  f"AAMI {'PASS' if row['results']['sbp']['aami_pass'] else 'FAIL'}")
        curves[pick] = rows

    out = dict(model=cfg.model, source=source, method="affine recalibration",
               config=asdict(cfg), headroom=head,
               n_subjects=int(len(np.unique(s_pool))),
               n_pool_clips=int(len(s_pool)), curves=curves)
    if save:
        nb_suffix = "" if balance else "_nobalance"
        path = (model_dir(cfg.model)
                / f"{cfg.model}_personalization_{source}{nb_suffix}.json")
        path.write_text(json.dumps(out, indent=2, default=float))
        print(f"  wrote {path}")
        _archive_run(f"{cfg.model}_personalization_{source}{nb_suffix}", path)
    return out


# --------------------------------------------------------------------------- #
# the domain-shift test (spec sections E, G, H, L)
# --------------------------------------------------------------------------- #

# What a physiologically possible answer looks like. Wider than any healthy
# range on purpose — the question is whether the model has stopped making sense,
# not whether the volunteer is hypertensive.
PLAUSIBLE_SBP = (70.0, 200.0)
PLAUSIBLE_DBP = (40.0, 120.0)


def _score_checkpoint(cfg, b, ckpt):
    """Run a saved deep model over the PulseDB test split and over PPG-DaLiA.

    Reads the target standardisation out of the checkpoint rather than
    recomputing it, so the predictions are on exactly the scale the model was
    trained to produce.
    """
    import torch
    from torch.utils.data import DataLoader
    from bpdata import SegmentDataset, demographics
    from models import build

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    saved = blob["cfg"]
    y_mean, y_std = blob["y_mean"], blob["y_std"]
    ch = list(saved.get("channels", cfg.channels))
    zsc = saved.get("zscore", False)      # must match training or inputs are off-distribution
    use_demo = cfg.model == "transformer"

    dev = _pick_device(cfg)
    kw = dict(width=saved.get("width", cfg.width)) if cfg.model in ("cnn", "resnet") else {}
    if cfg.model == "transformer":
        kw["shape_head"] = saved.get("aux_morphology", False)
    net = build({"transformer": "dmt"}.get(cfg.model, cfg.model),
                in_ch=len(ch), **kw).to(dev)
    net.load_state_dict(blob["state_dict"])
    net.eval()

    stats = demographics(b["train_split"])[1] if use_demo else None

    def predict(sig, lab, idx, demo):
        ds = SegmentDataset(sig, lab, idx, y_mean, y_std, demo=demo, channels=ch, zscore=zsc)
        dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        shuffle=False, pin_memory=(dev.type == "cuda"))
        out = []
        with torch.no_grad():
            for batch in dl:
                if use_demo:
                    x, d, _ = batch
                    out.append(net(x.to(dev), d.to(dev)).cpu())
                else:
                    x, _ = batch
                    out.append(net(x.to(dev)).cpu())
        return (torch.cat(out) * y_std + y_mean).numpy()

    te = PROC_PULSEDB / b["test_split"]
    sig_te = np.load(te / "signals.npy", mmap_mode="r")
    lab_te = np.load(te / "labels.npy")
    d_te = demographics(b["test_split"], stats)[0] if use_demo else None
    p_pulse = predict(sig_te, lab_te, b["test_idx"], d_te)

    sig_d, subj_d = dalia_windows()
    lab_d = np.zeros((len(sig_d), 2), dtype=np.float32)
    d_d = dalia_demographics(stats)[0] if use_demo else None
    p_dalia = predict(sig_d, lab_d, np.arange(len(sig_d)), d_d)

    return (p_pulse, lab_te[b["test_idx"]],
            _subjects(b["test_split"])[b["test_idx"]], p_dalia, subj_d)


def evaluate_dalia(model="gb", protocol="calfree", tag="", seed=0, save=True, **kw):
    """Score a PulseDB-trained model on PPG-DaLiA, which has no BP labels.

    Spec section G is explicit that PPG-DaLiA "does not contain BP labels at
    all" and is "an unlabeled, real-world input-distribution stress test". So
    there is no MAE to report here, and inventing one would be worse than
    reporting nothing. Four things can be measured without labels:

      1. plausibility — how often the model returns a number a human could not
         have, or a diastolic above its own systolic
      2. distribution shift — how far the predictions move relative to the same
         model's predictions on held-out PulseDB patients
      3. collapse — whether the model still separates people, or has fallen back
         on predicting one number for everyone
      4. motion — whether it degrades with activity, using DaLiA's own labels

    A fifth check the spec offers (section G, "a fully labeled secondary
    sanity-check task for heart-rate estimation") tests the signal pipeline
    rather than the BP model, and is reported separately and labelled as such.
    """
    cfg = Config(model=model, protocol=protocol, tag=tag, seed=seed, **kw)
    if tag == "nobalance":
        # the control: was the shift caused by the domain, or by the BP-bin
        # weighting spreading the predictions apart in the first place?
        cfg.balance = False
    print(f"\n=== domain shift: {cfg.name()} trained on PulseDB, "
          f"tested on PPG-DaLiA ===")

    b = build_protocol(protocol, seed=cfg.seed, cap=cfg.cap, val_frac=cfg.val_frac)
    ckpt = model_dir(cfg.model, "checkpoints") / f"{cfg.name()}.pt"

    if cfg.model in DEEP and ckpt.exists():
        # The deep models are already trained; re-fitting one to score a second
        # test set would cost hours and, worse, would not be the same model.
        print(f"  loading {ckpt.name}")
        p_pulse, y_pulse, s_pulse, p_dalia, s_dalia = _score_checkpoint(cfg, b, ckpt)
    else:
        fit = _fit_classical if cfg.model in CLASSICAL else _fit_deep
        p_pulse, y_pulse, s_pulse, info = fit(cfg, b, also_score=[("dalia", None)])
        p_dalia, _, s_dalia = info["also"]["dalia"]
    print(f"  scored {len(p_pulse):,} PulseDB and {len(p_dalia):,} PPG-DaLiA clips")

    meta = pd.read_parquet(FEATURES / "dalia.parquet")
    out = dict(model=cfg.model, tag=cfg.tag, balanced=cfg.balance,
               trained_on=f"PulseDB {protocol}",
               tested_on="PPG-DaLiA", config=asdict(cfg),
               n_dalia_clips=int(len(p_dalia)),
               n_dalia_subjects=int(len(np.unique(s_dalia))))

    # 1 — plausibility ------------------------------------------------------
    sbp, dbp = p_dalia[:, 0], p_dalia[:, 1]
    bad_sbp = ~((sbp >= PLAUSIBLE_SBP[0]) & (sbp <= PLAUSIBLE_SBP[1]))
    bad_dbp = ~((dbp >= PLAUSIBLE_DBP[0]) & (dbp <= PLAUSIBLE_DBP[1]))
    inverted = dbp >= sbp
    out["plausibility"] = dict(
        sbp_range=list(PLAUSIBLE_SBP), dbp_range=list(PLAUSIBLE_DBP),
        pct_sbp_impossible=float(bad_sbp.mean() * 100),
        pct_dbp_impossible=float(bad_dbp.mean() * 100),
        pct_dbp_above_sbp=float(inverted.mean() * 100),
        pct_any_impossible=float((bad_sbp | bad_dbp | inverted).mean() * 100))
    print(f"  implausible: SBP {bad_sbp.mean() * 100:.2f}%  "
          f"DBP {bad_dbp.mean() * 100:.2f}%  DBP>=SBP {inverted.mean() * 100:.2f}%")

    # 2 — distribution shift ------------------------------------------------
    def dist(v):
        return dict(mean=float(v.mean()), sd=float(v.std()),
                    p5=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                    p95=float(np.percentile(v, 95)))

    out["distribution"] = {}
    for k, t in enumerate(("sbp", "dbp")):
        out["distribution"][t] = dict(
            pulsedb_true=dist(y_pulse[:, k]),
            pulsedb_predicted=dist(p_pulse[:, k]),
            dalia_predicted=dist(p_dalia[:, k]))
        d = out["distribution"][t]
        print(f"  {t.upper()} mean — PulseDB true {d['pulsedb_true']['mean']:.1f}, "
              f"predicted {d['pulsedb_predicted']['mean']:.1f}, "
              f"DaLiA predicted {d['dalia_predicted']['mean']:.1f}")

    # 3 — collapse ----------------------------------------------------------
    # If the model has given up it returns nearly the same number for everyone,
    # so between-subject spread goes to zero while the true spread does not.
    def between_subject_sd(v, s):
        return float(np.std([v[s == u].mean() for u in np.unique(s)]))

    out["collapse"] = {}
    for k, t in enumerate(("sbp", "dbp")):
        out["collapse"][t] = dict(
            pulsedb_true=between_subject_sd(y_pulse[:, k], s_pulse),
            pulsedb_predicted=between_subject_sd(p_pulse[:, k], s_pulse),
            dalia_predicted=between_subject_sd(p_dalia[:, k], s_dalia))
        c = out["collapse"][t]
        print(f"  {t.upper()} between-subject SD — PulseDB true "
              f"{c['pulsedb_true']:.1f}, predicted {c['pulsedb_predicted']:.1f}, "
              f"DaLiA predicted {c['dalia_predicted']:.1f}")

    # 4 — motion ------------------------------------------------------------
    act = meta["activity_name"].to_numpy()
    out["by_activity"] = {}
    for a in sorted(set(act)):
        m = act == a
        out["by_activity"][str(a)] = dict(
            n=int(m.sum()),
            sbp_mean=float(p_dalia[m, 0].mean()), sbp_sd=float(p_dalia[m, 0].std()),
            pct_impossible=float((bad_sbp | bad_dbp | inverted)[m].mean() * 100))
    low = meta["low_motion"].to_numpy().astype(bool)
    out["by_motion"] = {
        "low_motion": dict(n=int(low.sum()),
                           sbp_mean=float(p_dalia[low, 0].mean()),
                           pct_impossible=float((bad_sbp | bad_dbp | inverted)[low].mean() * 100)),
        "high_motion": dict(n=int((~low).sum()),
                            sbp_mean=float(p_dalia[~low, 0].mean()),
                            pct_impossible=float((bad_sbp | bad_dbp | inverted)[~low].mean() * 100)),
    }
    print(f"  implausible under low motion "
          f"{out['by_motion']['low_motion']['pct_impossible']:.2f}% vs high motion "
          f"{out['by_motion']['high_motion']['pct_impossible']:.2f}%")

    # 5 — the labelled side task, which is about the signal pipeline ---------
    hr_ppg = meta["hrv_hr_ppg"].to_numpy(dtype=float)
    hr_ref = meta["hr_ecg_ref"].to_numpy(dtype=float)
    ok = np.isfinite(hr_ppg) & np.isfinite(hr_ref)
    out["heart_rate_crosscheck"] = dict(
        note="tests the PPG pulse detector against the chest-ECG reference, "
             "not the BP model",
        n=int(ok.sum()),
        mae_bpm=float(np.mean(np.abs(hr_ppg[ok] - hr_ref[ok]))),
        low_motion_mae_bpm=float(np.mean(np.abs(hr_ppg[ok & low] - hr_ref[ok & low]))),
        high_motion_mae_bpm=float(np.mean(np.abs(hr_ppg[ok & ~low] - hr_ref[ok & ~low]))))
    h = out["heart_rate_crosscheck"]
    print(f"  heart-rate cross-check: {h['mae_bpm']:.2f} bpm overall "
          f"({h['low_motion_mae_bpm']:.2f} still, {h['high_motion_mae_bpm']:.2f} moving)")

    if save:
        d = model_dir(cfg.model)
        stem = f"{cfg.name().replace('_' + protocol, '')}_dalia"
        json_path = d / f"{stem}_domain_shift.json"
        npz_path = d / f"{stem}_predictions.npz"
        json_path.write_text(json.dumps(out, indent=2, default=float))
        np.savez_compressed(npz_path,
                            y_pred=p_dalia, subjects=s_dalia.astype(str),
                            activity=act.astype(str), low_motion=low)
        print(f"  wrote {json_path}")
        _archive_run(f"{stem}_domain_shift", json_path, npz_path)
    return out


def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="gb", choices=list(CLASSICAL + DEEP))
    p.add_argument("--protocol", default="calfree", choices=list(PROTOCOLS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cap", type=int, default=0, help="max clips per subject, 0 = all")
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="")
    a = p.parse_args(argv)
    run(model=a.model, protocol=a.protocol, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, cap=a.cap or None, balance=not a.no_balance, device=a.device, tag=a.tag)


if __name__ == "__main__":
    _main()
