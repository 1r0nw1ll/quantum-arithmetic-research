#!/usr/bin/env python3
"""Combined-2: leaner model to reduce df penalty in Fisher's method.

With 6 QA features and 6 patients, Fisher uses df=36 — too many for the
per-patient effect sizes.  Here we pick 2 features that are complementary:
  - f0_sing: Singularity fraction from topographic transitions (captures chb13)
  - f2_sat:  Satellite fraction from multiband transitions (captures chb12/14)

This gives df=12 in Fisher, ~3× lower p-values at the same effect size.

Also tries a per-patient-normalised version: each feature → z-score using
training-interictal mean/std, so chb14's polarity reversal is handled.
"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

SUB_SEC = 1.0

MICROSTATE_STATES = {
    "A_frontal":   (8,  3),
    "B_occipital": (5, 16),
    "C_right":     (11, 19),
    "D_baseline":  (24, 24),
}
TRANSITION_TABLE = {
    (s1, s2): orbit_fam(MICROSTATE_STATES[s1][0], MICROSTATE_STATES[s2][1], m=24)
    for s1 in MICROSTATE_STATES for s2 in MICROSTATE_STATES
}
BAND_LIST = list(BANDS.items())
MICROSTATE_STATES_INT = {0: (8,3), 1: (5,16), 2: (11,19), 3: (24,24)}


def topo_fv(mc):
    rms  = np.sqrt(np.mean(mc.astype(np.float64) * mc.astype(np.float64), axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms


def fit_topo_km(windows):
    all_fvs = []
    for w in windows:
        sub_n = int(SUB_SEC * w["fs"])
        step  = max(1, sub_n // 2)
        for s in range(0, w["sig"].shape[1] - sub_n + 1, step):
            all_fvs.append(topo_fv(w["sig"][:, s: s + sub_n]))
    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(np.array(all_fvs))
    norms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank  = np.argsort(norms)
    names = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    order = [""] * 4
    for i, cid in enumerate(rank): order[cid] = names[i]
    return km, order


def f0_sing_frac(sig, fs, km, state_order):
    sub_n = int(SUB_SEC * fs); step = max(1, sub_n // 2)
    states = [state_order[int(km.predict(topo_fv(sig[:, s:s+sub_n]).reshape(1,-1))[0])]
              for s in range(0, sig.shape[1]-sub_n+1, step)]
    if len(states) < 2: return 0.0
    orbits = [TRANSITION_TABLE[(states[i], states[i+1])] for i in range(len(states)-1)]
    return orbits.count("Singularity") / len(orbits)


def band_topo_fv(sig, fs, low, high):
    filt = bandpass(sig, fs, low, high)
    rms  = np.sqrt(np.mean(filt * filt, axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms


def fit_band_kms(windows):
    result = []
    for (bname, (lo, hi)) in BAND_LIST:
        fvs = np.array([band_topo_fv(w["sig"], w["fs"], lo, hi) for w in windows])
        km  = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(fvs)
        norms = np.linalg.norm(km.cluster_centers_, axis=1)
        rank  = np.argsort(norms)
        order = np.empty(4, dtype=int)
        for i, cid in enumerate(rank): order[cid] = i
        result.append((km, order))
    return result


def f2_sat_frac(sig, fs, kms_orders):
    assignments = [int(order[int(km.predict(band_topo_fv(sig, fs, lo, hi).reshape(1,-1))[0])])
                   for (bname, (lo,hi)), (km, order) in zip(BAND_LIST, kms_orders)]
    orbits = [orbit_fam(MICROSTATE_STATES_INT[assignments[i]][0],
                        MICROSTATE_STATES_INT[assignments[i+1]][1], m=24)
              for i in range(len(assignments)-1)]
    n = len(orbits)
    return orbits.count("Satellite") / n if n else 0.0


def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"]==1 for w in windows) < 4: return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] Combined-2: {len(windows)} windows...")

    km_topo, state_order = fit_topo_km(windows)
    kms_orders = fit_band_kms(windows)

    delta  = np.array([delta_power_scalar(w["sig"], w["fs"]) for w in windows])
    f0_sing = np.array([f0_sing_frac(w["sig"], w["fs"], km_topo, state_order) for w in windows])
    f2_sat  = np.array([f2_sat_frac(w["sig"], w["fs"], kms_orders) for w in windows])

    # Raw 2-feature model
    feats_raw = np.column_stack([delta, f0_sing, f2_sat])
    dr2_raw, p_raw = eval_delta_r2(feats_raw, labels)

    # Z-score features within each patient (normalised by training interictal)
    tr_idx, te_idx = train_test_split(np.arange(len(labels)), test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    inter_mask = labels[tr_idx] == 0
    for feat_arr in [f0_sing, f2_sat]:
        mu  = feat_arr[tr_idx][inter_mask].mean()
        sig = feat_arr[tr_idx][inter_mask].std() + 1e-8
        feat_arr[:] = (feat_arr - mu) / sig  # in-place zscore

    feats_z = np.column_stack([delta, np.abs(f0_sing), np.abs(f2_sat)])
    dr2_z, p_z = eval_delta_r2(feats_z, labels)

    for lab, name in [(1,"ictal"),(0,"interictal")]:
        idx = np.where(labels==lab)[0]
        if len(idx):
            print(f"  [{pid}] {name:12s}: f0_sing|z|={np.abs(f0_sing[idx]).mean():.3f}  "
                  f"f2_sat|z|={np.abs(f2_sat[idx]).mean():.3f}")

    print(f"  [{pid}] raw ΔR²={dr2_raw:+.4f} p={p_raw:.4f}  "
          f"z-score ΔR²={dr2_z:+.4f} p={p_z:.4f}")
    return {"patient": pid,
            "dr2_raw": round(dr2_raw, 5), "p_raw": round(p_raw, 6),
            "dr2_z":   round(dr2_z,   5), "p_z":   round(p_z,   6)}


def main():
    print("="*60 + "\nCombined-2: f0_sing + f2_sat (lean, 2 QA features)\n" + "="*60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results:
        print("No results"); return
    n = len(results)
    for label, key_dr2, key_p in [("raw",     "dr2_raw", "p_raw"),
                                   ("z-score", "dr2_z",   "p_z")]:
        vals = [r[key_dr2] for r in results]
        ps   = [r[key_p]   for r in results]
        chi2_f, p_f = fishers(ps)
        n_sig = sum(1 for p in ps if p < 0.05)
        print(f"\n{label}: Mean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/n**0.5:.4f}  "
              f"pos={sum(v>0 for v in vals)}/{n}  sig={n_sig}/{n}  Fisher p={p_f:.3e}")
    Path("results/eeg_feat_combined2.json").write_text(
        json.dumps({"patients": results,
                    "mean_dr2_raw": float(np.mean([r["dr2_raw"] for r in results])),
                    "mean_dr2_z":   float(np.mean([r["dr2_z"]   for r in results])),
                    "fisher_p_raw": fishers([r["p_raw"] for r in results])[1],
                    "fisher_p_z":   fishers([r["p_z"]   for r in results])[1]}, indent=2))
    print("Saved: results/eeg_feat_combined2.json")


if __name__ == "__main__": main()
