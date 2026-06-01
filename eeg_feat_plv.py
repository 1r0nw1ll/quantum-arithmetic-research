#!/usr/bin/env python3
"""Feature 1: Cross-channel phase synchrony (PLV) → QA orbit."""
import json, sys
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

def plv_matrix(sig, fs, low=1, high=50):
    """Broadband PLV [n_ch, n_ch] via vectorised Hilbert."""
    filtered = bandpass(sig, fs, low, high)
    c = np.exp(1j * np.angle(hilbert(filtered, axis=1)))  # [n_ch, n_samp]
    n = c.shape[1]
    return np.abs(c @ c.conj().T) / n

def plv_to_feat(plv):
    """Upper triangle → flat vector."""
    n = plv.shape[0]
    idx = np.triu_indices(n, k=1)
    return plv[idx]

def window_features(w, pca_fit=None):
    sig, fs = w["sig"], w["fs"]
    delta = delta_power_scalar(sig, fs)
    plv   = plv_matrix(sig, fs)
    vec   = plv_to_feat(plv)     # 253-dim
    if pca_fit is not None:
        c1c2 = pca_fit.transform(vec.reshape(1,-1))[0]
        b = int(np.floor(np.abs(c1c2[0])) % 9) + 1
        e = int(np.floor(np.abs(c1c2[1])) % 9) + 1
        fam = orbit_fam(b, e)
        return np.array([delta, float(b), float(e),
                         1.0 if fam=="Cosmos" else 0.0,
                         1.0 if fam=="Satellite" else 0.0,
                         1.0 if fam=="Singularity" else 0.0])
    return vec

def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"]==1 for w in windows) < 4: return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] PLV: fitting PCA on {len(windows)} windows...")
    # Fit PCA on all windows (no labels — Theorem NT)
    vecs = np.array([window_features(w) for w in windows])
    pca  = PCA(n_components=2, random_state=SEED).fit(vecs)
    feats = np.array([window_features(w, pca) for w in windows])

    # Per-class mean orbit family fracs
    for lab, name in [(1,"ictal"),(0,"interictal")]:
        idx = np.where(labels==lab)[0]
        if len(idx):
            f = feats[idx]
            print(f"  [{pid}] {name:12s}: b={f[:,1].mean():.2f} e={f[:,2].mean():.2f} "
                  f"cos={f[:,3].mean():.3f} sat={f[:,4].mean():.3f} "
                  f"sing={f[:,5].mean():.3f}")

    dr2, p = eval_delta_r2(feats, labels)
    print(f"  [{pid}] ΔR²={dr2:+.4f}  p={p:.4f}")
    return {"patient":pid, "delta_r2":round(dr2,5), "p":round(p,6),
            "n_sei": int((labels==1).sum()), "n_bas": int((labels==0).sum())}

def main():
    print("="*60 + "\nFeature 1: Cross-channel PLV → PCA → mod-9 orbit\n" + "="*60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results: print("No results"); return
    vals = [r["delta_r2"] for r in results]
    ps   = [r["p"]        for r in results]
    chi2_f, p_f = fishers(ps)
    print(f"\nMean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/len(vals)**0.5:.4f}  "
          f"pos={sum(v>0 for v in vals)}/{len(vals)}  Fisher p={p_f:.3e}")
    Path("results/eeg_feat_plv.json").write_text(
        json.dumps({"patients":results, "mean_dr2":float(np.mean(vals)),
                    "fisher_p":p_f}, indent=2))
    print("Saved: results/eeg_feat_plv.json")

if __name__ == "__main__": main()
