"""Fiducial point detection on PulseDB's 10 s / 125 Hz ECG and PPG segments.

PulseDB ships ECG_F and PPG_F, already band-pass filtered and min-max normalised
per segment, so this module does detection rather than heavy cleaning.

Two properties of the data drive the design (see results/eda/segments.png):

* ECG polarity is not consistent across segments — some records have inverted R
  waves. Detection therefore runs on the squared derivative (Pan-Tompkins style),
  which responds to QRS steepness regardless of sign, and the peak is only
  refined on the original trace afterwards.
* ECG quality varies a lot, from textbook to unusable, while PPG is uniformly
  clean. Every routine here returns whatever it could find plus a quality score,
  so downstream code can decide what to drop — silently discarding bad segments
  from a test set would inflate the very numbers this project is trying to
  measure honestly.

All returned indices are in samples at FS.
"""

import numpy as np
from scipy import signal as sps

FS = 125
SEG_LEN = 1250

# Physiological bounds used to sanity-check detections.
MIN_HR, MAX_HR = 30.0, 200.0
MIN_RR = int(FS * 60 / MAX_HR)   # samples
MAX_RR = int(FS * 60 / MIN_HR)


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #

def _sos(low, high, fs=FS, order=4):
    nyq = fs / 2
    if high is None:
        return sps.butter(order, low / nyq, btype="highpass", output="sos")
    if low is None:
        return sps.butter(order, high / nyq, btype="lowpass", output="sos")
    return sps.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


_SOS_QRS = _sos(5.0, 15.0)      # QRS emphasis for R-peak detection
_SOS_PPG = _sos(0.5, 8.0)       # PPG pulse band
_SOS_PPG_SMOOTH = _sos(None, 10.0)


def bandpass(x, sos):
    return sps.sosfiltfilt(sos, x)


# --------------------------------------------------------------------------- #
# ECG
# --------------------------------------------------------------------------- #

def detect_rpeaks(ecg, fs=FS):
    """Pan-Tompkins style R-peak detection, polarity independent.

    Returns (peaks, quality) where quality in [0, 1] is the fraction of
    detected RR intervals that are physiologically plausible and regular.
    """
    x = np.asarray(ecg, dtype=np.float64)
    if x.size < fs or not np.isfinite(x).all() or np.ptp(x) == 0:
        return np.array([], dtype=int), 0.0

    f = bandpass(x, _SOS_QRS)
    d = np.diff(f, prepend=f[0])
    sq = d * d
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")

    # Adaptive threshold: peaks must clear a high percentile of the envelope.
    thr = np.percentile(integ, 92) * 0.35
    if thr <= 0:
        return np.array([], dtype=int), 0.0
    cand, _ = sps.find_peaks(integ, height=thr, distance=MIN_RR)
    if cand.size < 2:
        return np.array([], dtype=int), 0.0

    # Refine onto the true R apex: largest absolute deviation from the local
    # baseline, which works for both upright and inverted QRS complexes.
    base = np.median(x)
    half = max(1, int(0.05 * fs))
    peaks = []
    for c in cand:
        lo, hi = max(0, c - half), min(x.size, c + half + 1)
        peaks.append(lo + int(np.argmax(np.abs(x[lo:hi] - base))))
    peaks = np.unique(peaks)

    return peaks, _rr_quality(peaks)


def _rr_quality(peaks):
    """Fraction of RR intervals that are in range and close to the median RR."""
    if peaks.size < 3:
        return 0.0
    rr = np.diff(peaks)
    med = np.median(rr)
    if not (MIN_RR <= med <= MAX_RR):
        return 0.0
    ok = (rr >= MIN_RR) & (rr <= MAX_RR) & (np.abs(rr - med) <= 0.35 * med)
    return float(ok.mean())


# --------------------------------------------------------------------------- #
# PPG
# --------------------------------------------------------------------------- #

def detect_ppg_fiducials(ppg, fs=FS):
    """Locate per-beat PPG landmarks.

    Returns a dict with equal-length arrays (one entry per accepted beat):
        foot    pulse onset (diastolic minimum starting the beat)
        peak    systolic peak
        notch   dicrotic notch, -1 where not detectable
        next_foot  onset of the following beat (beat end)
    plus 'quality', the fraction of beats surviving the plausibility checks.
    """
    x = np.asarray(ppg, dtype=np.float64)
    empty = dict(foot=np.array([], int), peak=np.array([], int),
                 notch=np.array([], int), next_foot=np.array([], int),
                 quality=0.0)
    if x.size < fs or not np.isfinite(x).all() or np.ptp(x) == 0:
        return empty

    f = bandpass(x, _SOS_PPG_SMOOTH)

    # Systolic peaks. Distance is seeded from the dominant spectral peak so it
    # adapts to the subject's heart rate instead of assuming a fixed range.
    dist = _dominant_period(f, fs)
    peaks, _ = sps.find_peaks(f, distance=max(MIN_RR, int(0.6 * dist)),
                              prominence=0.1 * np.ptp(f))
    if peaks.size < 2:
        return empty

    # Feet: minimum of the trough between consecutive peaks.
    feet = []
    for a, b in zip(np.r_[0, peaks[:-1]], peaks):
        lo = a if a == 0 else a
        feet.append(lo + int(np.argmin(f[lo:b + 1])))
    feet = np.array(feet, dtype=int)

    # Keep beats that have a following foot, i.e. a complete pulse.
    keep = []
    for i, p in enumerate(peaks[:-1]):
        nf = feet[i + 1] if i + 1 < feet.size else -1
        if nf > p:
            keep.append(i)
    if not keep:
        return empty
    keep = np.array(keep)
    foot, peak, next_foot = feet[keep], peaks[keep], feet[keep + 1]

    # Plausibility: sane beat length and a real systolic excursion.
    amp = f[peak] - f[foot]
    dur = next_foot - foot
    good = (dur >= MIN_RR) & (dur <= MAX_RR) & (amp > 0.05 * np.ptp(f)) & (peak > foot)
    quality = float(good.mean()) if good.size else 0.0
    foot, peak, next_foot = foot[good], peak[good], next_foot[good]
    if foot.size == 0:
        return empty

    notch = _dicrotic_notch(f, peak, next_foot, fs)
    return dict(foot=foot, peak=peak, notch=notch,
                next_foot=next_foot, quality=quality)


def _dominant_period(x, fs):
    """Beat period in samples from the autocorrelation peak."""
    y = x - x.mean()
    ac = np.correlate(y, y, mode="full")[y.size - 1:]
    lo, hi = MIN_RR, min(MAX_RR, ac.size - 1)
    if hi <= lo:
        return MIN_RR
    return int(lo + np.argmax(ac[lo:hi]))


def _dicrotic_notch(f, peak, next_foot, fs):
    """Notch as the strongest upward inflection on the diastolic downslope.

    The notch is the 'e' wave of the acceleration signal (second derivative);
    where the artery is stiff enough that no notch forms, this returns -1
    rather than inventing a location.
    """
    apg = np.gradient(np.gradient(f))
    out = np.full(peak.size, -1, dtype=int)
    for i, (p, nf) in enumerate(zip(peak, next_foot)):
        lo = p + max(2, int(0.08 * fs))       # skip the peak's own curvature
        hi = min(nf, p + int(0.55 * fs))
        if hi - lo < 3:
            continue
        seg = apg[lo:hi]
        loc, props = sps.find_peaks(seg, prominence=0)
        if loc.size == 0:
            continue
        out[i] = lo + int(loc[np.argmax(props["prominences"])])
    return out
