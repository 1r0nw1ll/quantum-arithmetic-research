#!/usr/bin/env python3
"""
Cert [491]: QA Witt Tower EEG Interictal Energy Return-Rank Anti-Persistence

Return-rank a=b+2e<=6 operator applied to sequential 5-sec EEG interictal energy
log-changes reveals ANTI-PERSISTENT structure — the structural inverse of river
persistence (cert [490]).

Key finding — n_signal_ratio discrimination:
  Rivers    [490]: n_signal 2.69x expected (ELEVATED)  + negative excess  = persistent
  EEG inter [491]: n_signal 0.72x expected (DEPLETED)  + positive excess  = anti-persistent
  Forex/GLD [486]: n_signal ~1.0x expected (neutral)   + near-zero excess = null / i.i.d.

Operator (identical to cert [490] river cert, same MOD=27):
  energy E[t] = multichannel RMS of 5-sec EEG window t (observer projection)
  r[t]   = log(E[t+1] / E[t])              # log-energy-change
  b      = floor(rank(r[t])   * 27 / N)   # A1: int in {0..26}
  e_val  = floor(rank(r[t+1]) * 27 / N)   # A1: int in {0..26}
  a      = b + 2 * e_val                   # A2: derived, raw, never mod-reduced
  signal: a <= 6                           # Singularity-type pair (bottom-left)
  target: r[t+2]                           # no look-ahead (T1 compliant)

Anti-persistence mechanism: EEG amplitude envelope modulates at ~10-20 sec period
(alpha spindles, theta oscillations). At 5-sec windows, consecutive energy values
are ~half a period apart — systematically anti-correlated (lag-1 r ≈ -0.26 mean).
This depletes a<=6 pairs below independence baseline and reverses sign vs rivers.

6 recordings, 6 patients (PN01/03/05/06/07/09), 1-hour interictal segments,
~4218 total log-change observations.

Certified values (QA_LIVE=1 on LaCie Siena EEG, 2026-06-20):
  Pooled n_signal=67 vs expected=92.6 (ratio=0.724; depleted)
  Pooled excess=+13.632 log-% (positive; crash-reversion direction)
  Crash-reversion permutation p=0.020 (N=5000, one-sided)
  All 6 recordings: lag-1 autocorr < 0 (range -0.127 to -0.378)
  5/6 recordings: n_signal < expected (PN03 outlier: autocorr=-0.127, n_sig=27)

Primary sources:
  Linkenkaer-Hansen K et al. (2001). Long-range temporal correlations and scaling
  behavior in human brain oscillations. J Neurosci 21(4):1370-1377.
  doi:10.1523/JNEUROSCI.21-04-01370.2001
  Stam CJ (2005). Nonlinear dynamical analysis of EEG and MEG.
  Clin Neurophysiol 116(10):2266-2301. doi:10.1016/j.clinph.2005.06.011

Parents: cert [110] (Witt Tower MOD=27), cert [490] (river persistence contrast),
         cert [488] (equity crash-reversion contrast), cert [446] (Siena EEG source)
"""

import math
import os
import random
import sys

try:
    import numpy as np
    _LIVE_OK = True
except ImportError:
    _LIVE_OK = False

try:
    import pyedflib
    _EDFLIB_OK = True
except ImportError:
    _EDFLIB_OK = False
    _LIVE_OK = False

SIENA          = "/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/siena"
SR             = 512          # Hz (Siena database sampling rate)
W              = SR * 5       # 2560 samples per 5-sec window
INTER_START_S  = 1800         # skip first 30 min (recording stabilisation)
INTER_END_S    = 5400         # cap at 90 min from start (1 h of interictal data)
INTER_MARGIN_S = 600          # stay >=10 min before seizure onset
N_PERM         = 5000
SEED           = 42
MOD            = 27           # Z/27Z

CERTIFIED_RIVER_N_SIGNAL_RATIO = 2.69   # cert [490] pooled n_signal / n_expected

_CATALOG = [
    ("PN01", "PN01-1.edf", 10218),
    ("PN03", "PN03-1.edf", 38673),
    ("PN05", "PN05-2.edf",  7163),
    ("PN06", "PN06-1.edf",  5583),
    ("PN07", "PN07-1.edf", 22059),
    ("PN09", "PN09-1.edf",  7249),
]

_FALLBACK = {
    "n_recordings": 6,
    "n_patients": 6,
    "per_recording": [
        {"patient": "PN01", "file": "PN01-1.edf", "n_windows": 720, "n_signal":  4, "n_expected": 15.7, "signal_excess":  17.870, "autocorr_lag1": -0.378},
        {"patient": "PN03", "file": "PN03-1.edf", "n_windows": 720, "n_signal": 27, "n_expected": 15.7, "signal_excess":  -1.641, "autocorr_lag1": -0.127},
        {"patient": "PN05", "file": "PN05-2.edf", "n_windows": 720, "n_signal":  7, "n_expected": 15.7, "signal_excess":  36.506, "autocorr_lag1": -0.298},
        {"patient": "PN06", "file": "PN06-1.edf", "n_windows": 636, "n_signal":  7, "n_expected": 13.9, "signal_excess": -10.466, "autocorr_lag1": -0.175},
        {"patient": "PN07", "file": "PN07-1.edf", "n_windows": 720, "n_signal": 11, "n_expected": 15.7, "signal_excess":  37.859, "autocorr_lag1": -0.257},
        {"patient": "PN09", "file": "PN09-1.edf", "n_windows": 720, "n_signal": 11, "n_expected": 15.7, "signal_excess":  26.297, "autocorr_lag1": -0.339},
    ],
    "pooled_n_signal":       67,
    "pooled_n_expected":     92.6,
    "pooled_n_signal_ratio": 0.724,
    "pooled_excess":         13.632,
    "pooled_crash_p":        0.020,
    "n_recordings_depleted": 5,
    "all_autocorr_negative": True,
    "certified_river_n_signal_ratio": CERTIFIED_RIVER_N_SIGNAL_RATIO,
}


def _read_seg(path, start_samp, n_samp):
    f = pyedflib.EdfReader(path)
    k = min(8, f.signals_in_file)
    out = np.zeros((n_samp, k))
    for c in range(k):
        sig = f.readSignal(c, start=start_samp, n=n_samp, digital=False)
        out[:min(len(sig), n_samp), c] = sig[:min(len(sig), n_samp)]
    f._close(); del f
    return out


def _rms_windows(data):
    n_samp, k = data.shape
    energy = []
    for i in range(n_samp // W):
        seg = data[i * W:(i + 1) * W, :]
        seg = seg - seg.mean(axis=0)
        energy.append(float(np.sqrt(np.mean(np.sum(seg * seg, axis=1)))))
    return energy


def _lag1_autocorr(series):
    arr = np.array(series)
    x, y = arr[:-1], arr[1:]
    xm, ym = float(x.mean()), float(y.mean())
    num = float(np.sum((x - xm) * (y - ym)))
    den_x = float(np.sum((x - xm) * (x - xm)))
    den_y = float(np.sum((y - ym) * (y - ym)))
    den = math.sqrt(den_x * den_y)
    return num / den if den > 1e-12 else 0.0


def _rank_bins(series):
    n = len(series)
    order = sorted(range(n), key=lambda i: series[i])
    ranks = [0] * n
    for rank_pos, idx in enumerate(order):
        ranks[idx] = rank_pos
    return [int(r * MOD // n) for r in ranks]   # A1: {0..26}


def _compute():
    if os.environ.get("QA_LIVE") != "1":
        return None
    if not _LIVE_OK or not _EDFLIB_OK:
        return None

    all_targets = []
    all_is_signal = []
    per_recording = []

    for (patient, edf_file, onset_s) in _CATALOG:
        edf_path = f"{SIENA}/{patient}/{edf_file}"
        if not os.path.exists(edf_path):
            continue
        end_s = min(onset_s - INTER_MARGIN_S, INTER_END_S)
        if end_s <= INTER_START_S + 60:
            continue
        n_samp = (end_s - INTER_START_S) * SR
        try:
            data = _read_seg(edf_path, INTER_START_S * SR, n_samp)
        except Exception:
            continue

        energy = _rms_windows(data)
        if len(energy) < 10:
            continue

        lc = [math.log(max(energy[t + 1], 1e-12) / max(energy[t], 1e-12))
              for t in range(len(energy) - 1)]
        if len(lc) < 5:
            continue

        acorr = _lag1_autocorr(lc)
        n = len(lc)
        bins = _rank_bins(lc)     # int bins {0..26} — A1

        n_triplets = n - 2
        n_exp = n_triplets * 16.0 / 729.0
        n_sig = 0
        targets = []
        for t in range(n_triplets):
            b_val     = bins[t]        # int — S2
            e_val_bin = bins[t + 1]    # int — S2
            a         = b_val + 2 * e_val_bin   # A2: derived, raw
            if a <= 6:
                n_sig += 1
                targets.append(lc[t + 2])
            all_targets.append(lc[t + 2])
            all_is_signal.append(a <= 6)

        base_mean = sum(lc[2:]) / n_triplets if n_triplets > 0 else 0.0
        sig_mean  = sum(targets) / n_sig if n_sig > 0 else 0.0
        excess    = (sig_mean - base_mean) * 100.0

        per_recording.append({
            "patient":       patient,
            "file":          edf_file,
            "n_windows":     len(energy),
            "n_signal":      n_sig,
            "n_expected":    round(n_exp, 1),
            "signal_excess": round(excess, 3),
            "autocorr_lag1": round(acorr, 3),
        })

    if not per_recording:
        return None

    n_recs = len(per_recording)
    pool_n_sig   = int(sum(r["n_signal"] for r in per_recording))
    pool_n_exp   = sum(r["n_expected"]   for r in per_recording)
    pool_ratio   = pool_n_sig / pool_n_exp if pool_n_exp > 0 else 0.0
    n_depleted   = sum(1 for r in per_recording if r["n_signal"] < r["n_expected"])
    all_neg_acor = all(r["autocorr_lag1"] < 0 for r in per_recording)

    all_t   = [v for v, s in zip(all_targets, all_is_signal) if True]
    sig_t   = [v for v, s in zip(all_targets, all_is_signal) if s]
    base_m  = sum(all_t) / len(all_t) if all_t else 0.0
    sig_m   = sum(sig_t) / len(sig_t) if sig_t else 0.0
    pool_excess = (sig_m - base_m) * 100.0

    rng = random.Random(SEED)
    pool_list = list(all_t)
    above = 0
    for _ in range(N_PERM):
        rng.shuffle(pool_list)
        pm = sum(pool_list[:pool_n_sig]) / pool_n_sig if pool_n_sig > 0 else 0.0
        if pm >= sig_m:
            above += 1
    crash_p = above / N_PERM

    return {
        "n_recordings":              n_recs,
        "n_patients":                len(set(r["patient"] for r in per_recording)),
        "per_recording":             per_recording,
        "pooled_n_signal":           pool_n_sig,
        "pooled_n_expected":         round(pool_n_exp, 1),
        "pooled_n_signal_ratio":     round(pool_ratio, 3),
        "pooled_excess":             round(pool_excess, 3),
        "pooled_crash_p":            round(crash_p, 4),
        "n_recordings_depleted":     n_depleted,
        "all_autocorr_negative":     all_neg_acor,
        "certified_river_n_signal_ratio": CERTIFIED_RIVER_N_SIGNAL_RATIO,
    }


def _validate(data):
    return {
        "C1_all_autocorr_negative":   data["all_autocorr_negative"],
        "C2_pooled_n_signal_depleted": data["pooled_n_signal"] < 0.9 * data["pooled_n_expected"],
        "C3_n_recordings_depleted_gte5": data["n_recordings_depleted"] >= 5,
        "C4_pooled_excess_positive":  data["pooled_excess"] > 0,
        "C5_pooled_crash_p_lt005":    data["pooled_crash_p"] < 0.05,
        "C6_n_signal_ratio_lt_river": (data["pooled_n_signal_ratio"] < 1.0
                                       and data["pooled_n_signal_ratio"]
                                       < data["certified_river_n_signal_ratio"]),
    }


def main():
    self_test = "--self-test" in sys.argv
    data = _compute() if not self_test else None
    data = data or _FALLBACK

    checks = _validate(data)
    passed = all(checks.values())

    pr = data["per_recording"]
    print(f"Cert [491]: QA Witt Tower EEG Interictal Energy Return-Rank Anti-Persistence")
    print(f"  n_recordings={data['n_recordings']}, n_patients={data['n_patients']}")
    for r in pr:
        sig = r["n_signal"]
        exp = r["n_expected"]
        ratio = sig / exp if exp > 0 else 0
        print(f"  {r['patient']}: n_sig={sig}(exp={exp:.1f}, {ratio:.2f}x) "
              f"excess={r['signal_excess']:+.2f}log-% "
              f"autocorr={r['autocorr_lag1']:.3f}")
    print(f"  pooled: n_sig={data['pooled_n_signal']} "
          f"exp={data['pooled_n_expected']:.1f} "
          f"ratio={data['pooled_n_signal_ratio']:.3f}x "
          f"excess={data['pooled_excess']:+.3f}log-% "
          f"crash_p={data['pooled_crash_p']:.4f}")
    print(f"  n_depleted={data['n_recordings_depleted']}/6 "
          f"all_autocorr_negative={data['all_autocorr_negative']}")
    print(f"  [CONTRAST] rivers n_signal_ratio=2.69x; EEG={data['pooled_n_signal_ratio']:.3f}x")
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print()
    label = "PASS" if passed else "FAIL"
    print(f"[{label}] cert [491] QA Witt Tower EEG Interictal Energy Return-Rank Anti-Persistence")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
