"""Hand-crafted PTT and PPG-morphology features for cuffless BP estimation.

One feature row per 10 s segment. Landmarks come from `fiducials`; each
landmark-derived quantity is computed per beat and then aggregated across the
beats in the segment with a **median** (and IQR for spread), so a single
mis-detected beat cannot drag a segment's features around.

Feature groups, so ablations can switch them on and off cleanly:

    ppg_    PPG morphology — shape of the pulse wave
    der_    first/second derivative (VPG/APG) landmarks
    pat_    pulse arrival time, the ECG->PPG timing that carries most of the
            classical BP signal
    hrv_    rate and short-term variability from the ECG
    qual_   signal quality, kept as columns rather than used to silently filter
    demo_   age / sex / body habitus, available in PulseDB metadata

`qual_hr_agreement` compares the heart rate implied by the ECG with the one
implied by the PPG. The two are measuring the same heartbeat, so disagreement
means at least one channel is unreliable — in practice this separates good from
bad segments far better than either channel's own quality score.
"""

import numpy as np
from scipy import signal as sps

from fiducials import (FS, detect_rpeaks, detect_ppg_fiducials, bandpass,
                       _SOS_PPG_SMOOTH)

WIDTH_LEVELS = (0.10, 0.25, 0.50, 0.75)


def _agg(v, name, out):
    """Median + IQR of a per-beat quantity."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        out[name] = np.nan
        out[name + "_iqr"] = np.nan
    else:
        out[name] = float(np.median(v))
        out[name + "_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size > 1 else 0.0


def extract(ecg, ppg, demo=None):
    """Return a flat dict of features for one segment.

    ecg, ppg : 1-D arrays, 1250 samples at 125 Hz (PulseDB ECG_F / PPG_F)
    demo     : optional dict with age, gender, height, weight, bmi
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)
    out = {}

    rpeaks, ecg_q = detect_rpeaks(ecg)
    fid = detect_ppg_fiducials(ppg)
    foot, peak, notch, nfoot = fid["foot"], fid["peak"], fid["notch"], fid["next_foot"]

    # ---------------- rate and variability ----------------
    hr_ecg = 60.0 * FS / np.median(np.diff(rpeaks)) if rpeaks.size >= 3 else np.nan
    hr_ppg = 60.0 * FS / np.median(nfoot - foot) if foot.size >= 2 else np.nan
    out["hrv_hr_ecg"] = hr_ecg
    out["hrv_hr_ppg"] = hr_ppg
    if rpeaks.size >= 3:
        rr = np.diff(rpeaks) / FS * 1000.0          # ms
        out["hrv_sdnn"] = float(np.std(rr))
        out["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 1 else np.nan
    else:
        out["hrv_sdnn"] = out["hrv_rmssd"] = np.nan

    # ---------------- quality ----------------
    if np.isfinite(hr_ecg) and np.isfinite(hr_ppg) and hr_ppg > 0:
        agree = 1.0 - min(1.0, abs(hr_ecg - hr_ppg) / hr_ppg)
    else:
        agree = 0.0
    out["qual_ecg"] = float(ecg_q)
    out["qual_ppg"] = float(fid["quality"])
    out["qual_hr_agreement"] = float(agree)
    out["qual_n_beats"] = int(foot.size)
    out["qual_n_rpeaks"] = int(rpeaks.size)

    if foot.size == 0:
        _fill_nan(out)
        _add_demo(out, demo)
        return out

    smooth = bandpass(ppg, _SOS_PPG_SMOOTH)
    vpg = np.gradient(smooth)
    apg = np.gradient(vpg)

    # ---------------- per-beat PPG morphology ----------------
    amp, pi, rise, decay, ct_ratio = [], [], [], [], []
    widths = {lv: [] for lv in WIDTH_LEVELS}
    sys_area, dia_area, area_ratio = [], [], []
    notch_pos, refl_idx = [], []
    vpg_max, vpg_max_t, apg_a, apg_b, apg_ba = [], [], [], [], []

    for k in range(foot.size):
        f0, pk, nf = int(foot[k]), int(peak[k]), int(nfoot[k])
        beat = smooth[f0:nf + 1]
        if beat.size < 5:
            continue
        base = smooth[f0]
        a = smooth[pk] - base
        if a <= 0:
            continue
        n = nf - f0

        amp.append(a)
        pi.append(n / FS)
        rise.append((pk - f0) / FS)
        decay.append((nf - pk) / FS)
        ct_ratio.append((pk - f0) / n)

        for lv in WIDTH_LEVELS:
            widths[lv].append(_width_at(beat, base + lv * a) / FS)

        s_area = np.trapezoid(beat[:pk - f0 + 1] - base) / FS
        d_area = np.trapezoid(beat[pk - f0:] - base) / FS
        sys_area.append(s_area)
        dia_area.append(d_area)
        area_ratio.append(d_area / s_area if s_area > 0 else np.nan)

        nt = int(notch[k])
        if nt > f0:
            notch_pos.append((nt - f0) / n)
            refl_idx.append((smooth[nt] - base) / a)
        else:
            notch_pos.append(np.nan)
            refl_idx.append(np.nan)

        # upstroke steepness and the APG a/b waves (arterial stiffness proxies)
        up = slice(f0, pk + 1)
        if pk > f0:
            j = int(np.argmax(vpg[up]))
            vpg_max.append(vpg[f0 + j] * FS / a)
            vpg_max_t.append(j / FS)
            seg_a = apg[f0:pk + 1]
            ia = int(np.argmax(seg_a))
            apg_a.append(seg_a[ia])
            tail = apg[f0 + ia:nf + 1]
            if tail.size > 2:
                ib = int(np.argmin(tail))
                apg_b.append(tail[ib])
                apg_ba.append(tail[ib] / seg_a[ia] if seg_a[ia] != 0 else np.nan)

    _agg(amp, "ppg_sys_amp", out)
    _agg(pi, "ppg_pulse_interval", out)
    _agg(rise, "ppg_rise_time", out)
    _agg(decay, "ppg_decay_time", out)
    _agg(ct_ratio, "ppg_crest_time_ratio", out)
    for lv in WIDTH_LEVELS:
        _agg(widths[lv], f"ppg_width_{int(lv*100)}", out)
    _agg(sys_area, "ppg_sys_area", out)
    _agg(dia_area, "ppg_dia_area", out)
    _agg(area_ratio, "ppg_area_ratio", out)
    _agg(notch_pos, "ppg_notch_pos", out)
    _agg(refl_idx, "ppg_reflection_index", out)
    out["ppg_notch_frac"] = float(np.mean(notch[:foot.size] > 0)) if foot.size else np.nan

    _agg(vpg_max, "der_vpg_max", out)
    _agg(vpg_max_t, "der_vpg_max_time", out)
    _agg(apg_a, "der_apg_a", out)
    _agg(apg_b, "der_apg_b", out)
    _agg(apg_ba, "der_apg_ba_ratio", out)

    # ---------------- pulse arrival time ----------------
    # READ THIS BEFORE USING pat_* FEATURES.
    #
    # PulseDB's PPG channel is not time-aligned with its ECG/ABP channels. On the
    # VitalDB subsets the R-peak to ABP-foot delay is a tight 0.144 s (p25 0.128,
    # p75 0.160) — textbook pulse transit time — while the R-peak to PPG-foot
    # delay sits at 0.584 s with the PPG/ABP offset varying segment to segment.
    # Circular statistics over 987 clean segments put the phase concentration at
    # R=0.48 across the dataset but 0.83 within a subject: the offset behaves like
    # a per-record property, not a global constant, so it cannot be corrected
    # without a reference — and the only reference available is ABP, which is the
    # label. See scripts/check_alignment.py.
    #
    # Consequences: absolute PAT is NOT a valid calibration-free feature here. It
    # is still computed, for two reasons. It acts as a near-unique subject
    # fingerprint, so it is exactly the kind of feature that inflates results
    # under a leaky split — useful ammunition for the leakage experiment. And
    # because the offset is largely per-subject, a few-shot calibration step
    # should be able to absorb it, which is directly testable.
    #
    # Delays are paired within one beat period rather than a fixed physiological
    # window, so the value stays well defined whatever the offset is.
    rr = np.median(np.diff(rpeaks)) / FS if rpeaks.size >= 3 else np.nan
    pat_foot = _pair_rpeak_to(rpeaks, foot, rr)
    pat_peak = _pair_rpeak_to(rpeaks, peak, rr)
    maxslope = _max_slope_idx(vpg, foot, peak)
    pat_slope = _pair_rpeak_to(rpeaks, maxslope, rr)
    _agg(pat_foot, "pat_foot", out)
    _agg(pat_peak, "pat_peak", out)
    _agg(pat_slope, "pat_maxslope", out)
    # Phase — delay as a fraction of the beat — is the offset-robust form.
    _agg([p / rr for p in pat_foot] if np.isfinite(rr) and rr > 0 else [],
         "pat_foot_phase", out)

    _add_demo(out, demo)
    return out


# --------------------------------------------------------------------------- #

def _pair_rpeak_to(rpeaks, targets, rr):
    """For each R peak, the delay to the next target landmark, in seconds.

    Accepted up to one beat period (rr) rather than a fixed physiological
    window — see the note on channel alignment in `extract`.
    """
    if rpeaks.size == 0 or np.asarray(targets).size == 0 or not np.isfinite(rr):
        return []
    targets = np.asarray(targets)
    targets = targets[targets >= 0]
    if targets.size == 0:
        return []
    out = []
    for r in rpeaks:
        after = targets[targets > r]
        if after.size:
            dt = (after[0] - r) / FS
            if 0 < dt <= rr * 1.05:
                out.append(dt)
    return out


def _max_slope_idx(vpg, foot, peak):
    """Index of steepest upstroke within each beat."""
    idx = []
    for f0, pk in zip(foot, peak):
        if pk > f0:
            idx.append(int(f0) + int(np.argmax(vpg[int(f0):int(pk) + 1])))
    return np.array(idx, dtype=int)


def _width_at(beat, level):
    """Width of the pulse at an absolute amplitude level, in samples."""
    above = np.flatnonzero(beat >= level)
    return float(above[-1] - above[0]) if above.size >= 2 else np.nan


def _fill_nan(out):
    names = (["ppg_sys_amp", "ppg_pulse_interval", "ppg_rise_time", "ppg_decay_time",
              "ppg_crest_time_ratio"]
             + [f"ppg_width_{int(lv*100)}" for lv in WIDTH_LEVELS]
             + ["ppg_sys_area", "ppg_dia_area", "ppg_area_ratio", "ppg_notch_pos",
                "ppg_reflection_index", "der_vpg_max", "der_vpg_max_time",
                "der_apg_a", "der_apg_b", "der_apg_ba_ratio",
                "pat_foot", "pat_peak", "pat_maxslope", "pat_foot_phase"])
    for n in names:
        out[n] = np.nan
        out[n + "_iqr"] = np.nan
    out["ppg_notch_frac"] = np.nan


def _add_demo(out, demo):
    if demo is None:
        return
    out["demo_age"] = float(demo.get("age", np.nan))
    g = demo.get("gender", "")
    out["demo_male"] = 1.0 if str(g).upper().startswith("M") else 0.0
    out["demo_height"] = float(demo.get("height", np.nan))
    out["demo_weight"] = float(demo.get("weight", np.nan))
    out["demo_bmi"] = float(demo.get("bmi", np.nan))


def feature_names(with_demo=True):
    """Column order produced by `extract`, for building empty frames."""
    probe = extract(np.zeros(1250), np.zeros(1250),
                    demo={"age": 0, "gender": "M", "height": 0, "weight": 0, "bmi": 0}
                    if with_demo else None)
    return list(probe.keys())
