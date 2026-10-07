"""BP-estimation metrics, in the form the cuffless-BP literature reports them.

AAMI/ANSI SP10 is judged on the *error* distribution: mean error (bias) within
+/-5 mmHg and standard deviation of error within 8 mmHg, over at least 85
subjects. MAE is reported alongside because it is what most papers headline,
but MAE alone cannot decide AAMI compliance — a method can have low MAE and
still fail on bias or spread.

BHS grades on the proportion of absolute errors under 5/10/15 mmHg:
    A: >=60% / >=85% / >=95%
    B: >=50% / >=75% / >=90%
    C: >=40% / >=65% / >=85%
"""

import numpy as np
import pandas as pd

AAMI_MEAN_LIMIT = 5.0
AAMI_SD_LIMIT = 8.0
AAMI_MIN_SUBJECTS = 85


def metrics(y_true, y_pred, subjects=None):
    """Error metrics for one BP target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    me = float(np.mean(err))
    sde = float(np.std(err, ddof=1))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else np.nan

    n_subj = int(pd.Series(subjects).nunique()) if subjects is not None else None
    aami = (abs(me) <= AAMI_MEAN_LIMIT and sde <= AAMI_SD_LIMIT)
    if n_subj is not None and n_subj < AAMI_MIN_SUBJECTS:
        aami_note = f"limits met but only {n_subj} subjects (<{AAMI_MIN_SUBJECTS})"
    else:
        aami_note = ""

    a = np.abs(err)
    p5, p10, p15 = (float(np.mean(a <= t) * 100) for t in (5, 10, 15))

    return dict(n=int(y_true.size), n_subjects=n_subj,
                me=me, sde=sde, mae=mae, rmse=rmse, r=r,
                aami_pass=bool(aami), aami_note=aami_note,
                pct_le5=p5, pct_le10=p10, pct_le15=p15, bhs=bhs_grade(p5, p10, p15))


def bhs_grade(p5, p10, p15):
    for grade, (a, b, c) in (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85))):
        if p5 >= a and p10 >= b and p15 >= c:
            return grade
    return "D"


def report(y_true, y_pred, subjects=None, label=""):
    m = metrics(y_true, y_pred, subjects)
    note = f"  [{m['aami_note']}]" if m["aami_note"] else ""
    print(f"  {label:22s} n={m['n']:>7d} "
          f"MAE={m['mae']:6.2f}  ME={m['me']:+6.2f}  SD={m['sde']:6.2f}  "
          f"r={m['r']:5.2f}  AAMI={'PASS' if m['aami_pass'] else 'FAIL'}  "
          f"BHS={m['bhs']}{note}")
    return m


def subject_level(y_true, y_pred, subjects):
    """Per-subject mean error, as AAMI intends the statistic to be formed."""
    df = pd.DataFrame({"y": y_true, "p": y_pred, "s": subjects})
    g = df.groupby("s").apply(lambda x: pd.Series({
        "me": (x.p - x.y).mean(), "mae": (x.p - x.y).abs().mean(), "n": len(x)
    }), include_groups=False)
    return g
