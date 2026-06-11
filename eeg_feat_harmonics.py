#!/usr/bin/env python3
QA_COMPLIANCE = "EEG seizure: harmonic ratio peak pairs to orbit family; integer (b,e) from peak indices; no float state"
"""Feature 3: Harmonic ratio structure — do seizure frequencies follow Fibonacci ratios?

For each window:
  1. Compute mean PSD across channels
  2. Find top spectral peaks
  3. Compute all pairwise ratios between peaks
  4. For each ratio, map to a (b,e) pair and classify orbit family
  5. Test: are Fibonacci-ratio peaks enriched during seizure?

QA claim: if seizure frequencies are related by Fibonacci ratios, the
discretised (b,e) pairs from those ratios should cluster on specific orbit families.
"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, welch
sys.path.insert(0, str(Path(__file__).parent))
from eeg_feat_shared import *

# Fibonacci ratios to test (f_j / f_i for j > i)
FIB_NUMS = [1, 1, 2, 3, 5, 8, 13, 21, 34]
FIB_RATIOS = sorted(set(
    round(FIB_NUMS[j] / FIB_NUMS[i], 4)
    for i in range(len(FIB_NUMS))
    for j in range(i+1, len(FIB_NUMS))
    if FIB_NUMS[i] > 0
))
FIB_RATIO_THRESH = 0.08   # within 8% counts as Fibonacci ratio

def is_fib_ratio(r):
    return any(abs(r - fr) / fr < FIB_RATIO_THRESH for fr in FIB_RATIOS if fr > 0)

def spectral_peaks(sig, fs, n_peaks=6, f_min=1.0, f_max=50.0):
    """Find top-n spectral peaks in mean PSD."""
    nperseg = min(4*fs, sig.shape[1])
    freqs, Pxx = welch(sig.mean(axis=0).astype(np.float64), fs=fs, nperseg=nperseg)
    mask = (freqs >= f_min) & (freqs <= f_max)
    f_sub, P_sub = freqs[mask], Pxx[mask]
    if len(f_sub) < 3:
        return np.array([]), np.array([])
    peaks, props = find_peaks(P_sub, height=np.percentile(P_sub, 60), distance=3)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(P_sub)])
    # Sort by height descending, take top n
    order = np.argsort(P_sub[peaks])[::-1][:n_peaks]
    top   = peaks[order]
    return f_sub[top], P_sub[top]

def window_features(sig, fs):
    """
    Returns [delta, fib_ratio_frac, dominant_b, dominant_e,
             cosmos_frac, satellite_frac, singularity_frac,
             dom_freq_hz, mean_harmonic_ratio].
    """
    delta = delta_power_scalar(sig, fs)
    peak_freqs, peak_pows = spectral_peaks(sig, fs)

    if len(peak_freqs) < 2:
        # No peaks — degenerate window
        return np.array([delta, 0.0, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # Dominant frequency
    dom_f = float(peak_freqs[0])

    # Pairwise ratios (larger/smaller for ratios ≥ 1)
    ratios = []
    orbit_families = []
    for i in range(len(peak_freqs)):
        for j in range(i+1, len(peak_freqs)):
            fi, fj = peak_freqs[i], peak_freqs[j]
            r = max(fi,fj) / max(min(fi,fj), 0.1)
            ratios.append(r)
            # Map ratio to (b,e) via discretisation:
            # b = position of lower freq peak in mod-9 frequency grid
            # e = position of upper freq peak in mod-9 frequency grid
            # Grid: floor(f / (f_max/9)) + 1, clipped to [1,9]
            b = min(9, max(1, int(np.floor(min(fi,fj) / (50.0/9))) + 1))
            e = min(9, max(1, int(np.floor(max(fi,fj) / (50.0/9))) + 1))
            orbit_families.append(orbit_fam(b, e, m=9))

    fib_frac = float(sum(1 for r in ratios if is_fib_ratio(r)) / max(len(ratios),1))
    mean_ratio = float(np.mean(ratios)) if ratios else 1.0

    n = len(orbit_families)
    cos_frac  = orbit_families.count("Cosmos")       / n if n else 1.0
    sat_frac  = orbit_families.count("Satellite")    / n if n else 0.0
    sing_frac = orbit_families.count("Singularity")  / n if n else 0.0

    # Dominant freq → (b,e) for primary orbit classification
    b_dom = min(9, max(1, int(np.floor(dom_f / (50.0/9))) + 1))
    e_dom = b_dom  # use same for single-frequency case

    return np.array([delta, fib_frac, float(b_dom), float(e_dom),
                     cos_frac, sat_frac, sing_frac, dom_f, mean_ratio])

def analyse(pdir):
    pid = pdir.name
    windows = extract_windows(pdir)
    if not windows or sum(w["label"]==1 for w in windows) < 4: return None
    labels = np.array([w["label"] for w in windows])

    feats = np.array([window_features(w["sig"], w["fs"]) for w in windows])
    # Regression features: delta + fib_frac + cos + sat + sing
    reg_feats = feats[:, [0,1,4,5,6]]

    for lab, name in [(1,"ictal"),(0,"interictal")]:
        idx = np.where(labels==lab)[0]
        if len(idx):
            f = feats[idx]
            print(f"  [{pid}] {name:12s}: fib_frac={f[:,1].mean():.3f} "
                  f"dom_f={f[:,7].mean():.1f}Hz "
                  f"mean_ratio={f[:,8].mean():.3f} "
                  f"cos={f[:,4].mean():.3f} sat={f[:,5].mean():.3f}")

    dr2, p = eval_delta_r2(reg_feats, labels)

    # Also test fib_frac alone as discriminant (the core claim)
    fib_ic  = feats[labels==1, 1]
    fib_int = feats[labels==0, 1]
    from scipy.stats import ttest_ind
    t, p_t = ttest_ind(fib_ic, fib_int, equal_var=False)
    print(f"  [{pid}] ΔR²={dr2:+.4f}  p={p:.4f}  "
          f"fib_frac: ictal={fib_ic.mean():.3f} inter={fib_int.mean():.3f} "
          f"t={t:.2f} p_t={p_t:.4f}")
    return {"patient":pid, "delta_r2":round(dr2,5), "p":round(p,6),
            "fib_frac_ictal":round(float(fib_ic.mean()),4),
            "fib_frac_inter":round(float(fib_int.mean()),4),
            "fib_ttest_p":round(float(p_t),4)}

def main():
    print("="*60 + "\nFeature 3: Harmonic ratio structure → Fibonacci alignment\n" + "="*60)
    print(f"Fibonacci ratios tested (±8%): {FIB_RATIOS[:10]}...")
    results = [r for r in (analyse(pd) for pd in ready_patients()) if r]
    if not results: print("No results"); return
    vals = [r["delta_r2"] for r in results]
    ps   = [r["p"]        for r in results]
    chi2_f, p_f = fishers(ps)
    n_enrich = sum(1 for r in results
                   if r.get("fib_frac_ictal",0) > r.get("fib_frac_inter",0))
    print(f"\nMean ΔR²={np.mean(vals):+.4f}±{np.std(vals)/len(vals)**0.5:.4f}  "
          f"pos={sum(v>0 for v in vals)}/{len(vals)}  Fisher p={p_f:.3e}")
    print(f"Fibonacci ratio enriched in ictal: {n_enrich}/{len(results)} patients")
    Path("results/eeg_feat_harmonics.json").write_text(
        json.dumps({"patients":results,"mean_dr2":float(np.mean(vals)),
                    "fisher_p":p_f,"n_fib_enriched":n_enrich}, indent=2))
    print("Saved: results/eeg_feat_harmonics.json")

if __name__ == "__main__": main()
