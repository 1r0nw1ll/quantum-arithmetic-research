#!/usr/bin/env python3
QA_COMPLIANCE = "EEG seizure: coherence-graph degree to orbit family; integer (b,e) from degree mod 9 plus 1; observer layer only"
"""Feature 4: Coherence graph → algebraic degree spectrum → QA orbit.

For each window:
  1. Compute cross-spectral coherence matrix (n_ch × n_ch)
  2. Threshold at 75th percentile → binary connectivity graph
  3. Per-channel degree = number of significant coherence links
  4. For each connected pair (i,j): b = deg[i]%9+1, e = deg[j]%9+1 → orbit_fam(b,e,m=9)
  5. Orbit family fractions = QA structural fingerprint of the synchrony graph
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.signal import csd, welch
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

COH_THRESH_PCT = 75   # percentile threshold for binary graph
COH_BAND = (1, 50)    # broadband coherence


def coherence_matrix(sig, fs, low=1, high=50):
    """Magnitude-squared coherence matrix [n_ch, n_ch]."""
    nperseg = min(4 * fs, sig.shape[1])
    n_ch = sig.shape[0]
    C = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        fi, Pii = welch(sig[i].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (fi >= low) & (fi <= high)
        for j in range(i + 1, n_ch):
            _, Pjj = welch(sig[j].astype(np.float64), fs=fs, nperseg=nperseg)
            _, Pij = csd(sig[i].astype(np.float64), sig[j].astype(np.float64),
                         fs=fs, nperseg=nperseg)
            pii = Pii[mask]
            pjj = Pjj[mask]
            pij = Pij[mask]
            denom = pii * pjj
            coh = np.mean(np.abs(pij) ** 2 / np.clip(denom, 1e-30, None))
            C[i, j] = C[j, i] = float(np.clip(coh, 0.0, 1.0))  # noqa: A1-1 — observer: coherence in [0,1]
    return C


def window_features(sig, fs):
    """Returns [delta, cosmos_frac, satellite_frac, singularity_frac,
               mean_degree, mean_coh]."""
    delta = delta_power_scalar(sig, fs)

    C = coherence_matrix(sig, fs, *COH_BAND)
    n_ch = C.shape[0]

    # binary adjacency at threshold
    upper = C[np.triu_indices(n_ch, k=1)]
    thresh = float(np.percentile(upper, COH_THRESH_PCT))
    A = (C >= thresh).astype(int)
    np.fill_diagonal(A, 0)

    degrees = A.sum(axis=1)  # [n_ch]
    mean_degree = float(degrees.mean())
    mean_coh = float(upper.mean())

    orbit_families = []
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            if A[i, j]:
                b = int(degrees[i] % 9) + 1
                e = int(degrees[j] % 9) + 1
                orbit_families.append(orbit_fam(b, e, m=9))

    n = len(orbit_families)
    if n == 0:
        return np.array([delta, 1.0, 0.0, 0.0, mean_degree, mean_coh])

    cos_frac  = orbit_families.count("Cosmos")      / n
    sat_frac  = orbit_families.count("Satellite")   / n
    sing_frac = orbit_families.count("Singularity") / n

    return np.array([delta, cos_frac, sat_frac, sing_frac, mean_degree, mean_coh])


def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"] == 1 for w in windows) < 4:
        return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] Coherence AGS: {len(windows)} windows...")

    feats = np.array([window_features(w["sig"], w["fs"]) for w in windows])
    reg_feats = feats[:, :4]   # delta + orbit fracs

    for lab, name in [(1, "ictal"), (0, "interictal")]:
        idx = np.where(labels == lab)[0]
        if len(idx):
            f = feats[idx]
            print(f"  [{pid}] {name:12s}: cos={f[:,1].mean():.3f} "
                  f"sat={f[:,2].mean():.3f} sing={f[:,3].mean():.3f} "
                  f"deg={f[:,4].mean():.1f} coh={f[:,5].mean():.3f}")

    dr2, p = eval_delta_r2(reg_feats, labels)
    print(f"  [{pid}] ΔR²={dr2:+.4f}  p={p:.4f}")
    return {"patient": pid, "delta_r2": round(dr2, 5), "p": round(p, 6)}


def main():
    print("=" * 60 + "\nFeature 4: Coherence graph AGS → degree-orbit encoding\n" + "=" * 60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results:
        print("No results")
        return
    vals = [r["delta_r2"] for r in results]
    ps   = [r["p"]        for r in results]
    chi2_f, p_f = fishers(ps)
    print(f"\nMean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/len(vals)**0.5:.4f}  "
          f"pos={sum(v > 0 for v in vals)}/{len(vals)}  Fisher p={p_f:.3e}")
    Path("results/eeg_feat_coherence_ags.json").write_text(
        json.dumps({"patients": results, "mean_dr2": float(np.mean(vals)),
                    "fisher_p": p_f}, indent=2))
    print("Saved: results/eeg_feat_coherence_ags.json")


if __name__ == "__main__":
    main()
