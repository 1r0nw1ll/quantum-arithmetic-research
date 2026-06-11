#!/usr/bin/env python3
QA_COMPLIANCE = "EEG seizure: spectral entropy to orbit family; quantised entropy levels in {1..9}; integer QA state"
"""Feature 5: Spectral entropy baseline-deviation → QA orbit.

For each window:
  1. Compute per-channel spectral entropy H_i (normalised to [0,1])
  2. Quantise H_i to 9 levels → q_i ∈ {1,...,9}
  3. For each consecutive channel pair (sorted by entropy): b=q_i, e=q_{i+1} → orbit_fam(b,e,m=9)
  4. Orbit fracs + mean entropy + entropy variance = feature vector
  5. Baseline deviation: distance from training interictal entropy centroid
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *
from eeg_feat_shared import _null_ll, _model_ll


def spectral_entropy(sig, fs, low=1.0, high=50.0):
    """Per-channel spectral entropy, normalised to [0,1].  Shape: (n_ch,)."""
    nperseg = min(4 * fs, sig.shape[1])
    entropies = []
    for ch in range(sig.shape[0]):
        freqs, Pxx = welch(sig[ch].astype(np.float64), fs=fs, nperseg=nperseg)
        mask = (freqs >= low) & (freqs <= high)
        P = Pxx[mask]
        P = P / (P.sum() + 1e-30)
        H = -float(np.sum(P * np.log(P + 1e-30)))
        H_max = float(np.log(mask.sum()))
        entropies.append(H / H_max if H_max > 0 else 0.0)
    return np.array(entropies)


def entropy_to_orbit_fracs(H_ch):
    """Quantise per-channel entropies → (b,e) pairs → orbit fracs."""
    # q_i in {1,...,9} by equal-width binning over [0,1]
    q = np.clip(np.floor(H_ch * 9).astype(int) + 1, 1, 9)
    order = np.argsort(H_ch)
    q_sorted = q[order]
    families = [orbit_fam(int(q_sorted[i]), int(q_sorted[i + 1]), m=9)
                for i in range(len(q_sorted) - 1)]
    n = len(families)
    if n == 0:
        return 1.0, 0.0, 0.0
    return (families.count("Cosmos") / n,
            families.count("Satellite") / n,
            families.count("Singularity") / n)


def window_features(sig, fs):
    """[delta, cosmos, satellite, singularity, mean_H, std_H]."""
    delta = delta_power_scalar(sig, fs)
    H = spectral_entropy(sig, fs)
    cos_f, sat_f, sing_f = entropy_to_orbit_fracs(H)
    return np.array([delta, cos_f, sat_f, sing_f, float(H.mean()), float(H.std())])


def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"] == 1 for w in windows) < 4:
        return None
    labels = np.array([w["label"] for w in windows])
    print(f"  [{pid}] Spectral entropy: {len(windows)} windows...")

    feats = np.array([window_features(w["sig"], w["fs"]) for w in windows])
    reg_feats = feats[:, :4]  # delta + orbit fracs

    for lab, name in [(1, "ictal"), (0, "interictal")]:
        idx = np.where(labels == lab)[0]
        if len(idx):
            f = feats[idx]
            print(f"  [{pid}] {name:12s}: cos={f[:,1].mean():.3f} "
                  f"sat={f[:,2].mean():.3f} sing={f[:,3].mean():.3f} "
                  f"H={f[:,4].mean():.3f}±{f[:,5].mean():.3f}")

    dr2, p = eval_delta_r2(reg_feats, labels)

    # Baseline-deviation model: entropy vector distance from interictal mean
    tr_idx, te_idx = train_test_split(np.arange(len(labels)), test_size=TEST_FRAC,
                                      stratify=labels, random_state=SEED)
    base_mean = feats[tr_idx][labels[tr_idx] == 0][:, 4].mean()
    dev = np.abs(feats[:, 4] - base_mean).reshape(-1, 1)
    dev_feats = np.hstack([feats[:, :1], dev])

    from scipy.stats import chi2 as chi2_
    ll_null = _null_ll(labels[te_idx])
    lr_b = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_b.fit(dev_feats[tr_idx, :1], labels[tr_idx])
    ll_b = _model_ll(lr_b, dev_feats[te_idx, :1], labels[te_idx])
    lr_a = LogisticRegression(penalty="l2", random_state=SEED, max_iter=2000)
    lr_a.fit(dev_feats[tr_idx], labels[tr_idx])
    ll_a = _model_ll(lr_a, dev_feats[te_idx], labels[te_idx])
    r2_b = 1.0 - ll_b / ll_null if ll_null else 0.0
    r2_a = 1.0 - ll_a / ll_null if ll_null else 0.0
    dr2_dev = float(r2_a - r2_b)

    print(f"  [{pid}] orbit ΔR²={dr2:+.4f}  p={p:.4f}  "
          f"entropy-dev ΔR²={dr2_dev:+.4f}")
    return {"patient": pid, "delta_r2": round(dr2, 5), "p": round(p, 6),
            "dev_dr2": round(dr2_dev, 5)}


def main():
    print("=" * 60 + "\nFeature 5: Spectral entropy → orbit + baseline deviation\n" + "=" * 60)
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results:
        print("No results")
        return
    vals = [r["delta_r2"] for r in results]
    ps   = [r["p"]        for r in results]
    chi2_f, p_f = fishers(ps)
    dev_vals = [r.get("dev_dr2", 0) for r in results]
    print(f"\nOrbit  — Mean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/len(vals)**0.5:.4f}  "
          f"pos={sum(v > 0 for v in vals)}/{len(vals)}  Fisher p={p_f:.3e}")
    print(f"Dev    — Mean ΔR²={np.mean(dev_vals):+.4f}  "
          f"pos={sum(v > 0 for v in dev_vals)}/{len(results)}")
    Path("results/eeg_feat_entropy.json").write_text(
        json.dumps({"patients": results, "mean_dr2": float(np.mean(vals)),
                    "fisher_p": p_f, "mean_dev_dr2": float(np.mean(dev_vals))}, indent=2))
    print("Saved: results/eeg_feat_entropy.json")


if __name__ == "__main__":
    main()
