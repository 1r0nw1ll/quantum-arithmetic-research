#!/usr/bin/env python3
"""Compute: EEG Seizure Pre-Ictal Tier Elevation [479]
For each Siena patient recording, rank-calibrate on interictal energy,
measure T2 rate in pre-ictal window (0-300s before onset) vs interictal baseline.
"""
import math, random, sys
import numpy as np
import pyedflib

SIENA = "/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/siena"
MOD   = 27
W_SEC = 5      # 5-second windows (same as cert [446])
SR    = 512    # sample rate
W     = W_SEC * SR  # 2560 samples
N_PERM = 5000
SEED   = 42

PRERICTAL_SEC = 300   # 5-minute pre-ictal window
INTER_SEC     = 600   # 10-minute interictal sample (after first 120s)
INTER_SKIP    = 120   # skip first 2 minutes (recording artifacts)


def _ts(t):
    """HH.MM.SS or HH:MM.SS → total seconds from midnight."""
    t = t.strip().replace(':', '.').replace(' ', '')
    parts = t.split('.')
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return h*3600 + m*60 + s


def _onset_secs(reg_start_str, seiz_start_str):
    """Seconds from recording start to seizure onset (handles overnight)."""
    t0 = _ts(reg_start_str)
    t1 = _ts(seiz_start_str)
    diff = t1 - t0
    if diff < 0: diff += 24*3600
    return diff


# ── Seizure catalog (from Seizures-list-*.txt) ──────────────────────────────
# Each entry: (patient, edf_file, reg_start, seiz_start)
# Only include seizures with onset >= INTER_SKIP + INTER_SEC + PRERICTAL_SEC + 60 = ~1080s
SEIZURES = [
    # PN00: 4 recordings, 5 seizures; use first 3 with clear times
    ("PN00", "PN00-1.edf", "19.39.33", "19.58.36"),   # onset=1143s ✓
    ("PN00", "PN00-2.edf", "02.18.17", "02.38.37"),   # onset=1220s ✓
    # PN00-3 has suspicious 61-min seizure end — skip
    # PN01: one recording, two seizures; use first
    ("PN01", "PN01-1.edf", "19.00.44", "21.51.02"),   # onset=10218s ✓ (overnight)
    # PN03
    ("PN03", "PN03-1.edf", "22.44.37", "09.29.10"),   # onset=38433s ✓ (overnight)
    ("PN03", "PN03-2.edf", "21.31.04", "07.13.05"),   # onset=35521s ✓ (overnight)
    # PN05
    ("PN05", "PN05-2.edf", "06.46.02", "08.45.25"),   # onset=7163s ✓
    ("PN05", "PN05-3.edf", "06.01.23", "07.55.19"),   # onset=6836s ✓
    ("PN05", "PN05-4.edf", "06.38.35", "07.38.43"),   # onset=3608s ✓
    # PN06
    ("PN06", "PN06-1.edf", "04.21.22", "05.54.25"),   # onset=5583s ✓
    ("PN06", "PN06-2.edf", "21.11.29", "23.39.09"),   # onset=8860s ✓
    ("PN06", "PN06-3.edf", "06.25.51", "08.10.26"),   # onset=6275s ✓
    # PN07
    ("PN07", "PN07-1.edf", "23.18.10", "05.25.49"),   # onset=22059s ✓ (overnight)
    # PN09
    ("PN09", "PN09-1.edf", "14.08.54", "16.09.43"),   # onset=7249s ✓
    # PN13
    ("PN13", "PN13-1.edf", "08.24.28", "10.22.10"),   # onset=6942s ✓
    ("PN13", "PN13-2.edf", "06.55.02", "08.55.51"),   # onset=7249s ✓
    # PN14
    ("PN14", "PN14-1.edf", "11.44.58", "13.46.00"),   # onset=7262s ✓
    ("PN14", "PN14-2.edf", "15.50.13", "17.54.52"),   # onset=7479s ✓
    # PN16 — need to check times
]


def _read_windows(edf_path, start_samp, n_samps, n_chan):
    """Read n_samps samples from start_samp, all channels, return (n_win, n_chan) array."""
    f = pyedflib.EdfReader(edf_path)
    n_channels = f.signals_in_file
    # Use first min(n_chan, available) channels
    k = min(n_chan, n_channels)
    data = np.zeros((n_samps, k))
    for c in range(k):
        sig = f.readSignal(c, start=start_samp, n=n_samps, digital=False)
        data[:min(len(sig), n_samps), c] = sig[:min(len(sig), n_samps)]
    f._close()
    del f
    return data


def _windows_rms(data, W):
    """Compute multi-channel energy RMS per W-sample window."""
    n, k = data.shape
    n_win = n // W
    rms_vals = []
    for i in range(n_win):
        seg = data[i*W:(i+1)*W, :]  # (W, k)
        # DC removal per channel
        seg = seg - seg.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum(seg**2, axis=1))))
        rms_vals.append(rms)
    return rms_vals


def _t2_threshold(inter_rms):
    """66.7th percentile of interictal RMS → T2 boundary."""
    return float(np.percentile(inter_rms, 200.0/3))


def analyze_seizure(patient, edf_file, reg_start, seiz_start):
    edf_path = f"{SIENA}/{patient}/{edf_file}"
    import os
    if not os.path.exists(edf_path):
        print(f"  SKIP {patient}/{edf_file}: file not found")
        return None

    onset_s = _onset_secs(reg_start, seiz_start)
    if onset_s < INTER_SKIP + INTER_SEC + PRERICTAL_SEC + 60:
        print(f"  SKIP {patient}/{edf_file}: onset {onset_s}s too short")
        return None

    # Interictal: INTER_SKIP → INTER_SKIP + INTER_SEC
    inter_start_samp = INTER_SKIP * SR
    inter_n_samp     = INTER_SEC * SR

    # Pre-ictal: onset - PRERICTAL_SEC → onset
    prerictal_start_samp = (onset_s - PRERICTAL_SEC) * SR
    prerictal_n_samp     = PRERICTAL_SEC * SR

    print(f"  {patient}/{edf_file}: onset={onset_s}s "
          f"inter=[{INTER_SKIP},{INTER_SKIP+INTER_SEC}]s "
          f"prerictal=[{onset_s-PRERICTAL_SEC},{onset_s}]s")

    try:
        inter_data = _read_windows(edf_path, inter_start_samp, inter_n_samp, 8)
        pre_data   = _read_windows(edf_path, prerictal_start_samp, prerictal_n_samp, 8)
    except Exception as e:
        print(f"  ERROR reading {edf_file}: {e}")
        return None

    inter_rms = _windows_rms(inter_data, W)
    pre_rms   = _windows_rms(pre_data, W)

    if len(inter_rms) < 10 or len(pre_rms) < 10:
        print(f"  SKIP {patient}/{edf_file}: too few windows")
        return None

    t2_thresh = _t2_threshold(inter_rms)
    t1_thresh = float(np.percentile(inter_rms, 100.0/3))  # 33.3th pct

    # Tier assignment: <t1=T0, t1<=<t2=T1, >=t2=T2
    def _tier_thresh(v, t1, t2):
        if v >= t2: return 2
        if v >= t1: return 1
        return 0

    inter_tiers = [_tier_thresh(v, t1_thresh, t2_thresh) for v in inter_rms]
    pre_tiers   = [_tier_thresh(v, t1_thresh, t2_thresh) for v in pre_rms]

    inter_t2_rate = sum(1 for t in inter_tiers if t == 2) / len(inter_tiers)
    pre_t2_rate   = sum(1 for t in pre_tiers   if t == 2) / len(pre_tiers)

    # Sub-window T2 rates (monotone check)
    n_pre = len(pre_tiers)
    # early = first 60%, late = last 40%
    early_tiers = pre_tiers[:int(n_pre*0.6)]
    late_tiers  = pre_tiers[int(n_pre*0.6):]
    early_t2 = sum(1 for t in early_tiers if t == 2) / len(early_tiers) if early_tiers else 0
    late_t2  = sum(1 for t in late_tiers  if t == 2) / len(late_tiers)  if late_tiers  else 0

    print(f"    inter_t2={inter_t2_rate:.3f} pre_t2={pre_t2_rate:.3f} "
          f"early={early_t2:.3f} late={late_t2:.3f}")

    return {
        "patient": patient, "file": edf_file, "onset_s": onset_s,
        "n_inter": len(inter_tiers), "n_pre": len(pre_tiers),
        "inter_t2_rate": inter_t2_rate,
        "pre_t2_rate": pre_t2_rate,
        "early_t2_rate": early_t2,   # t-300 to t-120
        "late_t2_rate": late_t2,     # t-120 to t-0
        "pre_exceeds_inter": pre_t2_rate > inter_t2_rate,
    }


def pooled_perm_test(results):
    """Permutation test: pool all pre-ictal windows, shuffle inter vs pre labels."""
    all_inter = []
    all_pre   = []
    for r in results:
        all_inter.extend([0]*r["n_inter"])
        all_pre.extend([1]*r["n_pre"])
    # Under null: no difference in T2 rate between inter and pre
    # We shuffle inter/pre labels within each patient
    # Simplified: use T2 counts and do Fisher-style permutation
    obs_pre_t2 = sum(r["pre_t2_rate"] * r["n_pre"] for r in results)
    obs_inter_t2 = sum(r["inter_t2_rate"] * r["n_inter"] for r in results)
    n_total_pre   = sum(r["n_pre"] for r in results)
    n_total_inter = sum(r["n_inter"] for r in results)
    obs_diff = obs_pre_t2/n_total_pre - obs_inter_t2/n_total_inter
    print(f"\n  Pooled: pre_t2={obs_pre_t2/n_total_pre:.4f} inter_t2={obs_inter_t2/n_total_inter:.4f}")
    print(f"  T2 excess (pre-inter): {obs_diff:+.4f}")

    # Permutation within each recording: shuffle inter/pre labels
    # Represent as flat array of tier labels (0=inter, 1=inter, ...) with T2 indicator
    # Use per-patient shuffle for matched permutation
    random.seed(SEED)
    perm_diffs = []
    for _ in range(N_PERM):
        perm_pre_t2 = 0
        perm_total_pre = 0
        perm_inter_t2 = 0
        perm_total_inter = 0
        for r in results:
            # Pool inter+pre T2 counts, then randomly split at n_inter/n_pre
            total_n = r["n_inter"] + r["n_pre"]
            total_t2 = int(r["inter_t2_rate"]*r["n_inter"]) + int(r["pre_t2_rate"]*r["n_pre"])
            # Hypergeometric draw: how many of total_t2 land in n_pre slots?
            # Use shuffle proxy: shuffle binary indicator
            indicators = [1]*total_t2 + [0]*(total_n - total_t2)
            random.shuffle(indicators)
            perm_pre_t2_r   = sum(indicators[:r["n_pre"]])
            perm_inter_t2_r = sum(indicators[r["n_pre"]:])
            perm_pre_t2    += perm_pre_t2_r
            perm_total_pre += r["n_pre"]
            perm_inter_t2  += perm_inter_t2_r
            perm_total_inter += r["n_inter"]
        perm_diff = perm_pre_t2/perm_total_pre - perm_inter_t2/perm_total_inter
        perm_diffs.append(perm_diff)

    perm_p = sum(1 for d in perm_diffs if d >= obs_diff) / N_PERM
    print(f"  Permutation p (one-sided): {perm_p:.4f}")
    return {
        "pre_t2_rate": obs_pre_t2/n_total_pre,
        "inter_t2_rate": obs_inter_t2/n_total_inter,
        "t2_excess": obs_diff,
        "perm_p": perm_p,
        "n_total_pre": n_total_pre,
        "n_total_inter": n_total_inter,
    }


if __name__ == "__main__":
    print("=== EEG Seizure Pre-Ictal Tier Elevation Compute [479] ===\n")
    results = []
    for (patient, edf_file, reg_start, seiz_start) in SEIZURES:
        r = analyze_seizure(patient, edf_file, reg_start, seiz_start)
        if r is not None:
            results.append(r)

    print(f"\n=== Summary: {len(results)} recordings processed ===")
    patients_with_excess = sum(1 for r in results if r["pre_exceeds_inter"])
    print(f"  Pre-ictal T2 > interictal T2 in {patients_with_excess}/{len(results)} recordings")

    for r in results:
        print(f"  {r['patient']}/{r['file']}: inter={r['inter_t2_rate']:.3f} "
              f"pre={r['pre_t2_rate']:.3f} excess={r['pre_t2_rate']-r['inter_t2_rate']:+.3f} "
              f"early={r['early_t2_rate']:.3f} late={r['late_t2_rate']:.3f}")

    if results:
        pool = pooled_perm_test(results)

        # Monotone check: early_t2 < late_t2 in most recordings
        monotone_ok = sum(1 for r in results if r["late_t2_rate"] >= r["early_t2_rate"])
        print(f"\n  Monotone rise (late>=early T2): {monotone_ok}/{len(results)}")

        print("\n=== Fallback values for validator ===")
        print(f"  n_recordings: {len(results)}")
        print(f"  n_patients: {len(set(r['patient'] for r in results))}")
        print(f"  pre_t2_rate: {pool['pre_t2_rate']:.4f}")
        print(f"  inter_t2_rate: {pool['inter_t2_rate']:.4f}")
        print(f"  t2_excess: {pool['t2_excess']:+.4f}")
        print(f"  perm_p: {pool['perm_p']:.4f}")
        print(f"  n_pre_recordings_exceed: {patients_with_excess}")
        print(f"  n_monotone: {monotone_ok}")
