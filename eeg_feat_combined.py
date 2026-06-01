#!/usr/bin/env python3
"""Feature combined: F0 (transition orbit) + F2 (multiband) stacked model.

Motivation: F0 captures chb09 (Singularity collapse, ΔR²=+0.178) and F2
captures chb14 (inter-band structure, ΔR²=+0.103).  Both use topographic
k-means → orbit-family-of-transition — the shared mechanism is the transition
between microstates rather than the absolute microstate.

Feature vector (col 0 = delta baseline, cols 1+ = QA augmentation):
  [delta, f0_sing, f0_sat, f0_cos, f2_cos, f2_sat, f2_sing]

Three models compared:
  F0:       delta + transition orbit fracs   (3 QA features)
  F2:       delta + multiband orbit fracs    (3 QA features)
  COMBINED: delta + both                     (6 QA features)
"""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

# ── Constants inherited from the two source scripts ───────────────────────────
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
assert sum(1 for f in TRANSITION_TABLE.values() if f == "Singularity") == 1
assert sum(1 for f in TRANSITION_TABLE.values() if f == "Satellite")   == 3

BAND_LIST = list(BANDS.items())   # ordered: delta, theta, alpha, beta, gamma

MICROSTATE_STATES_INT = {
    0: (8,  3),   # rank 0 (weakest) ≈ D_baseline equivalent
    1: (5, 16),
    2: (11, 19),
    3: (24, 24),  # rank 3 (strongest) ≈ A_frontal equivalent
}


# ── F0: topographic k-means transition features ───────────────────────────────

def topo_fv(mc):
    rms  = np.sqrt(np.mean(mc.astype(np.float64) * mc.astype(np.float64), axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms


def fit_topo_km(windows):
    all_fvs = []
    for w in windows:
        sub_n = int(SUB_SEC * w["fs"])
        step  = max(1, sub_n // 2)
        mc    = w["sig"]
        for s in range(0, mc.shape[1] - sub_n + 1, step):
            all_fvs.append(topo_fv(mc[:, s: s + sub_n]))
    X  = np.array(all_fvs)
    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(X)
    norms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank  = np.argsort(norms)
    names = ["D_baseline", "C_right", "B_occipital", "A_frontal"]
    state_order = [""] * 4
    for i, cid in enumerate(rank):
        state_order[cid] = names[i]
    return km, state_order


def f0_fracs(sig, fs, km, state_order):
    sub_n = int(SUB_SEC * fs)
    step  = max(1, sub_n // 2)
    states = []
    for s in range(0, sig.shape[1] - sub_n + 1, step):
        fv  = topo_fv(sig[:, s: s + sub_n]).reshape(1, -1)
        cid = int(km.predict(fv)[0])
        states.append(state_order[cid])
    if len(states) < 2:
        return 0.0, 0.0, 1.0
    orbits = [TRANSITION_TABLE[(states[i], states[i+1])] for i in range(len(states)-1)]
    n = len(orbits)
    return (orbits.count("Singularity")/n,
            orbits.count("Satellite")/n,
            orbits.count("Cosmos")/n)


# ── F2: multiband per-band k-means transition features ────────────────────────

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
        for i, cid in enumerate(rank):
            order[cid] = i
        result.append((km, order))
    return result


def f2_fracs(sig, fs, kms_orders):
    assignments = []
    for (bname, (lo, hi)), (km, order) in zip(BAND_LIST, kms_orders):
        fv  = band_topo_fv(sig, fs, lo, hi).reshape(1, -1)
        cid = int(km.predict(fv)[0])
        assignments.append(int(order[cid]))
    orbits = []
    for i in range(len(assignments) - 1):
        b = MICROSTATE_STATES_INT[assignments[i]][0]
        e = MICROSTATE_STATES_INT[assignments[i+1]][1]
        orbits.append(orbit_fam(b, e, m=24))
    n = len(orbits)
    return (orbits.count("Cosmos")/n,
            orbits.count("Satellite")/n,
            orbits.count("Singularity")/n)


# ── Per-patient analysis ──────────────────────────────────────────────────────

def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"]==1 for w in windows) < 4:
        return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] Combined: fitting k-means ({len(windows)} windows)...")

    km_topo, state_order = fit_topo_km(windows)
    kms_orders = fit_band_kms(windows)

    delta  = np.array([delta_power_scalar(w["sig"], w["fs"]) for w in windows])
    f0_arr = np.array([f0_fracs(w["sig"], w["fs"], km_topo, state_order) for w in windows])
    f2_arr = np.array([f2_fracs(w["sig"], w["fs"], kms_orders) for w in windows])

    # [delta | f0_sing f0_sat f0_cos | f2_cos f2_sat f2_sing]
    feats_f0   = np.hstack([delta.reshape(-1,1), f0_arr])
    feats_f2   = np.hstack([delta.reshape(-1,1), f2_arr])
    feats_comb = np.hstack([delta.reshape(-1,1), f0_arr, f2_arr])

    for lab, name in [(1,"ictal"),(0,"interictal")]:
        idx = np.where(labels==lab)[0]
        if len(idx):
            f0 = f0_arr[idx]
            f2 = f2_arr[idx]
            print(f"  [{pid}] {name:12s}: "
                  f"F0 sing={f0[:,0].mean():.3f} sat={f0[:,1].mean():.3f}  "
                  f"F2 cos={f2[:,0].mean():.3f} sat={f2[:,1].mean():.3f} sing={f2[:,2].mean():.3f}")

    dr2_f0,   p_f0   = eval_delta_r2(feats_f0,   labels)
    dr2_f2,   p_f2   = eval_delta_r2(feats_f2,   labels)
    dr2_comb, p_comb = eval_delta_r2(feats_comb, labels)

    print(f"  [{pid}] F0 ΔR²={dr2_f0:+.4f}  "
          f"F2 ΔR²={dr2_f2:+.4f}  "
          f"COMBINED ΔR²={dr2_comb:+.4f}  p={p_comb:.4f}")

    return {"patient": pid,
            "dr2_f0":   round(dr2_f0,   5), "p_f0":   round(p_f0,   6),
            "dr2_f2":   round(dr2_f2,   5), "p_f2":   round(p_f2,   6),
            "dr2_comb": round(dr2_comb, 5), "p_comb": round(p_comb, 6)}


def main():
    print("="*60 + "\nCombined: F0 (transition) + F2 (multiband) stacked model\n" + "="*60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results:
        print("No results"); return

    n = len(results)
    for label, key_dr2, key_p in [("F0",       "dr2_f0",   "p_f0"),
                                   ("F2",       "dr2_f2",   "p_f2"),
                                   ("COMBINED", "dr2_comb", "p_comb")]:
        vals = [r[key_dr2] for r in results]
        ps   = [r[key_p]   for r in results]
        chi2_f, p_f = fishers(ps)
        print(f"\n{label}: Mean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/n**0.5:.4f}  "
              f"pos={sum(v>0 for v in vals)}/{n}  Fisher p={p_f:.3e}")

    Path("results/eeg_feat_combined.json").write_text(
        json.dumps({"patients": results,
                    "aggregate": {m: {
                        "mean_dr2": round(float(np.mean([r[f"dr2_{m}"] for r in results])), 5),
                        "n_pos":    sum(1 for r in results if r[f"dr2_{m}"] > 0),
                        "fisher_p": round(fishers([r[f"p_{m}"] for r in results])[1], 8),
                    } for m in ["f0","f2","comb"]}
                    }, indent=2))
    print("\nSaved: results/eeg_feat_combined.json")


if __name__ == "__main__": main()
