#!/usr/bin/env python3
"""Feature 2: Multi-band spatial k-means — inter-band transition orbits."""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

K = 4
BAND_NAMES = list(BANDS.keys())   # delta, theta, alpha, beta, gamma

MICROSTATE_STATES = {
    0: (8,  3),   # cluster 0 (weakest) → D_baseline-like
    1: (5, 16),
    2: (11, 19),
    3: (24, 24),  # cluster 3 (strongest) → A_frontal-like
}

def band_topo_fv(sig, fs, low, high):
    """RMS per channel in a frequency band, L2-normalised."""
    filt = bandpass(sig, fs, low, high)
    rms  = np.sqrt(np.mean(filt**2, axis=1))
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 1e-9 else rms

def fit_band_kms(windows, low, high):
    fvs = np.array([band_topo_fv(w["sig"], w["fs"], low, high) for w in windows])
    km  = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(fvs)
    # rank by centroid norm → state order
    norms = np.linalg.norm(km.cluster_centers_, axis=1)
    rank  = np.argsort(norms)
    order = np.empty(K, dtype=int)
    for i, cid in enumerate(rank): order[cid] = i  # 0=weakest, 3=strongest
    return km, order

def window_orbit_frac_multiband(sig, fs, kms_orders):
    """
    5 bands × k=4 → 5 cluster labels per window.
    4 consecutive inter-band transitions → (b,e) pairs → orbit families.
    Returns [delta, cosmos_frac, satellite_frac, singularity_frac].
    """
    assignments = []
    for (bname, (lo, hi)), (km, order) in zip(BANDS.items(), kms_orders):
        fv = band_topo_fv(sig, fs, lo, hi).reshape(1,-1)
        cid = int(km.predict(fv)[0])
        assignments.append(int(order[cid]))  # 0–3 rank

    # Inter-band transitions: delta→theta, theta→alpha, alpha→beta, beta→gamma
    orbits = []
    for i in range(len(assignments)-1):
        b = MICROSTATE_STATES[assignments[i]][0]
        e = MICROSTATE_STATES[assignments[i+1]][1]
        orbits.append(orbit_fam(b, e))
    n = len(orbits)
    return [
        orbits.count("Cosmos")       / n,
        orbits.count("Satellite")    / n,
        orbits.count("Singularity")  / n,
    ]

def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"]==1 for w in windows) < 4: return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] Multiband: fitting 5 band k-means...")
    kms_orders = [fit_band_kms(windows, lo, hi) for (lo, hi) in BANDS.values()]

    orb = np.array([window_orbit_frac_multiband(w["sig"], w["fs"], kms_orders)
                    for w in windows])
    delta = np.array([delta_power_scalar(w["sig"], w["fs"]) for w in windows])
    feats = np.hstack([delta.reshape(-1,1), orb])

    for lab, name in [(1,"ictal"),(0,"interictal")]:
        idx = np.where(labels==lab)[0]
        if len(idx):
            f = orb[idx]
            print(f"  [{pid}] {name:12s}: cos={f[:,0].mean():.3f} "
                  f"sat={f[:,1].mean():.3f} sing={f[:,2].mean():.3f}")

    dr2, p = eval_delta_r2(feats, labels)
    print(f"  [{pid}] ΔR²={dr2:+.4f}  p={p:.4f}")
    return {"patient":pid, "delta_r2":round(dr2,5), "p":round(p,6)}

def main():
    print("="*60 + "\nFeature 2: Multi-band spatial k-means → inter-band orbits\n" + "="*60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results: print("No results"); return
    vals = [r["delta_r2"] for r in results]
    ps   = [r["p"]        for r in results]
    chi2_f, p_f = fishers(ps)
    print(f"\nMean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/len(vals)**0.5:.4f}  "
          f"pos={sum(v>0 for v in vals)}/{len(vals)}  Fisher p={p_f:.3e}")
    Path("results/eeg_feat_multiband.json").write_text(
        json.dumps({"patients":results,"mean_dr2":float(np.mean(vals)),
                    "fisher_p":p_f}, indent=2))
    print("Saved: results/eeg_feat_multiband.json")

if __name__ == "__main__": main()
