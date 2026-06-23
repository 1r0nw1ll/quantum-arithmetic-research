#!/usr/bin/env python3
"""Compute: EEG Pre-Ictal Bimodal Stratification [480]
Extends cert [479] with full T0/T1/T2 per recording.
Tests: hyperactive (T2-dominant) vs quiescent (T0-dominant) pre-ictal poles.
"""
import random, sys
import numpy as np
import pyedflib

SIENA        = "/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/siena"
SR           = 512
W_SEC        = 5
W            = W_SEC * SR
INTER_SKIP   = 120
INTER_SEC    = 600
PRERICTAL_SEC= 300
N_PERM       = 5000
SEED         = 42

# Same catalog as cert [479]
CATALOG = [
    ("PN00","PN00-1.edf","19.39.33","19.58.36"),
    ("PN00","PN00-2.edf","02.18.17","02.38.37"),
    ("PN01","PN01-1.edf","19.00.44","21.51.02"),
    ("PN03","PN03-1.edf","22.44.37","09.29.10"),
    ("PN03","PN03-2.edf","21.31.04","07.13.05"),
    ("PN05","PN05-2.edf","06.46.02","08.45.25"),
    ("PN05","PN05-3.edf","06.01.23","07.55.19"),
    ("PN05","PN05-4.edf","06.38.35","07.38.43"),
    ("PN06","PN06-1.edf","04.21.22","05.54.25"),
    ("PN06","PN06-2.edf","21.11.29","23.39.09"),
    ("PN06","PN06-3.edf","06.25.51","08.10.26"),
    ("PN07","PN07-1.edf","23.18.10","05.25.49"),
    ("PN09","PN09-1.edf","14.08.54","16.09.43"),
    ("PN13","PN13-1.edf","08.24.28","10.22.10"),
    ("PN13","PN13-2.edf","06.55.02","08.55.51"),
    ("PN14","PN14-1.edf","11.44.58","13.46.00"),
    ("PN14","PN14-2.edf","15.50.13","17.54.52"),
]

HYPER_THRESH  = 0.55   # T2 rate above this -> hyperactive
QUIET_THRESH  = 0.20   # T2 rate below this -> quiescent


def _ts(t):
    t = t.strip().replace(':', '.').replace(' ', '')
    p = t.split('.')
    return int(p[0])*3600 + int(p[1])*60 + int(p[2])


def _onset(reg, seiz):
    diff = _ts(seiz) - _ts(reg)
    if diff < 0: diff += 86400
    return diff


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
    n, k = data.shape
    vals = []
    for i in range(n // W):
        seg = data[i*W:(i+1)*W, :]
        seg = seg - seg.mean(axis=0)
        vals.append(float(np.sqrt(np.mean(np.sum(seg*seg, axis=1)))))
    return vals


def analyze():
    results = []
    for (patient, edf_file, reg_start, seiz_start) in CATALOG:
        edf_path = f"{SIENA}/{patient}/{edf_file}"
        import os
        if not os.path.exists(edf_path):
            print(f"  SKIP {patient}/{edf_file}: not found"); continue
        onset_s = _onset(reg_start, seiz_start)
        if onset_s < INTER_SKIP + INTER_SEC + PRERICTAL_SEC + 60:
            print(f"  SKIP {patient}/{edf_file}: onset too short"); continue
        try:
            inter_data = _read_seg(edf_path, INTER_SKIP*SR, INTER_SEC*SR)
            pre_data   = _read_seg(edf_path, (onset_s - PRERICTAL_SEC)*SR, PRERICTAL_SEC*SR)
        except Exception as e:
            print(f"  ERROR {edf_file}: {e}"); continue

        inter_rms = _rms_windows(inter_data)
        pre_rms   = _rms_windows(pre_data)
        if len(inter_rms) < 10 or len(pre_rms) < 10: continue

        t2_thr = float(np.percentile(inter_rms, 200.0/3))
        t1_thr = float(np.percentile(inter_rms, 100.0/3))

        def _tier(v): return 2 if v >= t2_thr else (1 if v >= t1_thr else 0)

        pre_tiers = [_tier(v) for v in pre_rms]
        n_pre = len(pre_tiers)

        t0_rate = sum(1 for t in pre_tiers if t == 0) / n_pre
        t1_rate = sum(1 for t in pre_tiers if t == 1) / n_pre
        t2_rate = sum(1 for t in pre_tiers if t == 2) / n_pre

        label = ("hyper"   if t2_rate > HYPER_THRESH else
                 "quiet"   if t2_rate < QUIET_THRESH else
                 "mixed")

        print(f"  {patient}/{edf_file}: T0={t0_rate:.3f} T1={t1_rate:.3f} T2={t2_rate:.3f} → {label}")

        results.append({
            "patient": patient, "file": edf_file, "onset_s": onset_s,
            "t0_rate": t0_rate, "t1_rate": t1_rate, "t2_rate": t2_rate,
            "label": label,
        })
    return results


if __name__ == "__main__":
    print("=== EEG Pre-Ictal Bimodal Stratification Compute [480] ===\n")
    results = analyze()

    hyper = [r for r in results if r["label"] == "hyper"]
    quiet = [r for r in results if r["label"] == "quiet"]
    mixed = [r for r in results if r["label"] == "mixed"]

    print(f"\n=== Groups: hyper={len(hyper)} quiet={len(quiet)} mixed={len(mixed)} ===")

    print("\n-- Hyperactive (T2 > 0.55): --")
    for r in hyper:
        print(f"  {r['patient']}/{r['file']}: T0={r['t0_rate']:.3f} T1={r['t1_rate']:.3f} T2={r['t2_rate']:.3f}")

    print("\n-- Quiescent (T2 < 0.20): --")
    for r in quiet:
        print(f"  {r['patient']}/{r['file']}: T0={r['t0_rate']:.3f} T1={r['t1_rate']:.3f} T2={r['t2_rate']:.3f}")

    if hyper:
        mean_hyp_t2 = sum(r["t2_rate"] for r in hyper) / len(hyper)
        mean_hyp_t0 = sum(r["t0_rate"] for r in hyper) / len(hyper)
        print(f"\nHyper group: mean_T2={mean_hyp_t2:.3f} mean_T0={mean_hyp_t0:.3f}")

    if quiet:
        mean_qui_t2 = sum(r["t2_rate"] for r in quiet) / len(quiet)
        mean_qui_t0 = sum(r["t0_rate"] for r in quiet) / len(quiet)
        print(f"Quiet group: mean_T2={mean_qui_t2:.3f} mean_T0={mean_qui_t0:.3f}")

        gap = mean_hyp_t2 - mean_qui_t2 if hyper else 0
        print(f"Bimodal T2 gap: {gap:.3f}")

    # Within-patient heterogeneity
    from collections import defaultdict
    by_patient = defaultdict(list)
    for r in results:
        by_patient[r["patient"]].append(r["label"])
    mixed_patients = [p for p, labels in by_patient.items()
                      if "hyper" in labels and "quiet" in labels]
    print(f"\nPatients with both hyper and quiet: {mixed_patients}")

    print("\n=== Fallback values for validator ===")
    if hyper and quiet:
        print(f"  n_hyper:         {len(hyper)}")
        print(f"  n_quiet:         {len(quiet)}")
        print(f"  mean_hyp_t2:     {mean_hyp_t2:.4f}")
        print(f"  mean_hyp_t0:     {mean_hyp_t0:.4f}")
        print(f"  mean_qui_t2:     {mean_qui_t2:.4f}")
        print(f"  mean_qui_t0:     {mean_qui_t0:.4f}")
        print(f"  gap:             {gap:.4f}")
        print(f"  mixed_patients:  {mixed_patients}")
        n_quiet_t0_enriched = sum(1 for r in quiet if r["t0_rate"] > 0.45)
        print(f"  n_quiet_t0_gt45: {n_quiet_t0_enriched}/{len(quiet)}")
        for r in results:
            print(f"  {r['patient']:4s}|{r['file']:12s}: T0={r['t0_rate']:.3f} T1={r['t1_rate']:.3f} T2={r['t2_rate']:.3f} [{r['label']}]")
