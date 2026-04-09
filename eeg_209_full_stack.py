#!/usr/bin/env python3
"""
eeg_209_full_stack.py — Full [209] Stack: Invariants + Bateson + Coupling + CRT

Builds on cert [209] generator inference with four extensions:

1. RAW + MODULAR invariants: 21 QA invariants from raw (b, Δ) AND orbit
   from mod-reduced (b%m, e). Elements use raw d=b+e per hard rule;
   mod reduction is T-operator only.

2. BATESON LEVELS: recursive generator inference per [191].
   Level 0: signal b_t
   Level I: generator e_t from b
   Level II: meta-generator from e
   Each level captures dynamics at a different abstraction.

3. COUPLING MATRIX: pairwise generator correlation between channels.
   Full N×N QA-derived coupling matrix, not just scalar synchrony.

4. DUAL-MODULUS CRT: generators inferred in mod-9 AND mod-24 simultaneously.
   CRT consistency between moduli = error-detection channel per [205].

Runs on CHB-MIT chb01 and reports per-feature ΔR².
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "signal_generator_inference_full_stack",
    "axioms_checked": ["A1", "A2", "S1", "S2", "T1", "T2"],
}

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments/qa_lab")))

from eeg_chbmit_scale import load_patient_dataset
from qa_orbit_rules import orbit_family, norm_f

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
MOD_9 = 9
MOD_24 = 24
DOWNSAMPLE = 4


# ── Quantization ─────────────────────────────────────────────────────────────

def quantize_multichannel(multi_ch, m, downsample=DOWNSAMPLE):
    """Global percentile bins across all channels. Returns list of list[int]."""
    n_ch = multi_ch.shape[0]
    ds = multi_ch[:, ::downsample]
    n_samp = ds.shape[1]
    if n_samp < 2:
        return [[1] for _ in range(n_ch)]
    all_vals = ds.ravel()
    edges = [float(np.percentile(all_vals, 100 * k / m)) for k in range(1, m)]
    result = []
    for ch in range(n_ch):
        q = []
        for t in range(n_samp):
            val = float(ds[ch, t])
            b = 1
            for edge in edges:
                if val > edge:
                    b += 1
            q.append(b)
        result.append(q)
    return result


# ── Core generators ──────────────────────────────────────────────────────────

def infer_generator(b_t, b_next, m):
    """Mod-reduced generator in {1,...,m}."""
    return ((b_next - b_t - 1) % m) + 1


def raw_delta(b_t, b_next):
    """Raw transition (signed integer, no mod reduction)."""
    return b_next - b_t


# ── Extension 1: Raw + Modular Invariants ────────────────────────────────────

def compute_raw_invariants(b, delta):
    """QA invariants from RAW (b, Δ) — no mod reduction. Elements use raw d=b+e."""
    e = delta  # raw transition as generator
    d = b + e
    a = b + 2 * e
    B = b * b  # S1
    E = e * e
    D = d * d
    A = a * a
    C = 2 * d * e
    F = b * a
    G = D + E
    return {
        "raw_B": B, "raw_E": E, "raw_D": D, "raw_A": A,
        "raw_C": C, "raw_F": F, "raw_G": G,
        "raw_f": b * b + b * e - e * e,
    }


# ── Extension 2: Bateson Levels ──────────────────────────────────────────────

def bateson_levels(quantized, m, max_level=3):
    """
    Recursive generator inference per [191].
    Level 0: signal
    Level k: generators inferred from level k-1
    """
    levels = [quantized]
    for level in range(1, max_level + 1):
        prev = levels[-1]
        if len(prev) < 3:
            break
        generators = [infer_generator(prev[t], prev[t + 1], m)
                      for t in range(len(prev) - 1)]
        levels.append(generators)
    return levels


def level_entropy(sequence, m):
    """Generator distribution entropy for a sequence."""
    if len(sequence) < 2:
        return 0.0
    counts = [0] * m
    for v in sequence:
        counts[(v - 1) % m] += 1
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * np.log2(c / total) for c in counts if c > 0)


# ── Extension 3: Coupling Matrix ─────────────────────────────────────────────

def generator_coupling_matrix(all_quantized, m):
    """
    Pairwise generator correlation between channels.
    Returns N×N matrix of Pearson correlations between generator sequences.
    """
    n_ch = len(all_quantized)
    min_len = min(len(q) for q in all_quantized) - 1
    if min_len < 10:
        return np.eye(n_ch)

    # Compute generator sequences
    gen_seqs = []
    for i in range(n_ch):
        gens = [infer_generator(all_quantized[i][t], all_quantized[i][t + 1], m)
                for t in range(min_len)]
        gen_seqs.append(gens)

    # Correlation matrix
    gen_matrix = np.array(gen_seqs, dtype=float)  # (n_ch, min_len)
    corr = np.corrcoef(gen_matrix)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def coupling_features(corr_matrix):
    """Extract features from the generator coupling matrix."""
    n = corr_matrix.shape[0]
    np.fill_diagonal(corr_matrix, 0.0)
    abs_corr = np.abs(corr_matrix)

    # Eigenvalues capture mode structure
    eigvals = np.sort(np.linalg.eigvalsh(corr_matrix + np.eye(n)))[::-1]
    participation_ratio = (eigvals.sum() * eigvals.sum()) / (
        (eigvals * eigvals).sum() * n + 1e-10)

    return {
        "coupling_mean": float(abs_corr.mean()),
        "coupling_max": float(abs_corr.max()),
        "coupling_std": float(abs_corr.std()),
        "coupling_participation": float(participation_ratio),
        "coupling_eig1_frac": float(eigvals[0] / (eigvals.sum() + 1e-10)),
    }


# ── Extension 4: Dual-Modulus CRT ───────────────────────────────────────────

def dual_modulus_generators(quantized_9, quantized_24):
    """
    Infer generators in both mod-9 and mod-24. Check CRT consistency.
    """
    min_len = min(len(quantized_9), len(quantized_24)) - 1
    if min_len < 2:
        return {"crt_rate": 0.0, "mean_e9": 0.0, "mean_e24": 0.0}

    consistent = 0
    e9_vals = []
    e24_vals = []

    for t in range(min_len):
        e9 = infer_generator(quantized_9[t], quantized_9[t + 1], MOD_9)
        e24 = infer_generator(quantized_24[t], quantized_24[t + 1], MOD_24)
        e9_vals.append(e9)
        e24_vals.append(e24)

        # CRT consistency: e24 mod 9 should equal e9
        # (since 9 divides... actually gcd(9,24)=3, not 9|24)
        # Use weaker check: orbit family must be compatible
        o9 = orbit_family(((quantized_9[t] - 1) % MOD_9) + 1, e9, MOD_9)
        o24 = orbit_family(((quantized_24[t] - 1) % MOD_24) + 1, e24, MOD_24)
        # Singularity must pair with singularity
        if o9 == "singularity" and o24 == "singularity":
            consistent += 1
        elif o9 != "singularity" and o24 != "singularity":
            consistent += 1
        # else: inconsistent

    return {
        "crt_rate": consistent / min_len,
        "mean_e9": float(np.mean(e9_vals)),
        "mean_e24": float(np.mean(e24_vals)),
        "e9_entropy": level_entropy(e9_vals, MOD_9),
        "e24_entropy": level_entropy(e24_vals, MOD_24),
    }


# ── Full feature extraction ─────────────────────────────────────────────────

def extract_full_features(multi_ch):
    """Extract ALL [209] stack features from one EEG window."""
    n_ch = multi_ch.shape[0]

    # Quantize at both moduli
    q9 = quantize_multichannel(multi_ch, MOD_9)
    q24 = quantize_multichannel(multi_ch, MOD_24)

    # --- Per-channel stats (mod-9 primary) ---
    all_raw_feats = {k: [] for k in ["raw_C", "raw_F", "raw_G", "raw_f"]}
    all_level_entropies = {f"L{k}_entropy": [] for k in range(4)}
    crt_results = []

    for i in range(n_ch):
        # Raw invariants
        for t in range(len(q9[i]) - 1):
            delta = raw_delta(q9[i][t], q9[i][t + 1])
            inv = compute_raw_invariants(q9[i][t], delta)
            for k in all_raw_feats:
                all_raw_feats[k].append(inv[k])

        # Bateson levels
        levels = bateson_levels(q9[i], MOD_9, max_level=3)
        for k, seq in enumerate(levels):
            ent = level_entropy(seq, MOD_9)
            all_level_entropies[f"L{k}_entropy"].append(ent)

        # Dual-modulus CRT
        crt = dual_modulus_generators(q9[i], q24[i])
        crt_results.append(crt)

    # Aggregate per-channel stats
    features = {}
    for k, vals in all_raw_feats.items():
        features[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
    for k, vals in all_level_entropies.items():
        features[k] = float(np.mean(vals)) if vals else 0.0
    features["crt_rate"] = float(np.mean([c["crt_rate"] for c in crt_results]))
    features["e24_entropy"] = float(np.mean([c["e24_entropy"] for c in crt_results]))

    # --- Coupling matrix (mod-9) ---
    corr = generator_coupling_matrix(q9, MOD_9)
    coup = coupling_features(corr)
    features.update(coup)

    # --- Basic [209] features for comparison ---
    from eeg_signal_dynamics_observer import extract_window_features
    basic = extract_window_features(multi_ch)
    features["basic_sing"] = basic["singularity_frac"]
    features["basic_synch"] = basic["generator_synchrony"]
    features["basic_entropy"] = basic["gen_entropy"]
    features["basic_mean_f"] = basic["mean_f"]

    return features


# ── Logistic regression ─────────────────────────────────────────────────────

def _fit(X, y, lr=0.1, n_iter=3000, l2=1e-4):
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        logits = np.clip(X @ beta, -30, 30)
        probs = expit(logits)
        beta -= lr * (X.T @ (probs - y) / len(y) + l2 * beta)
    return beta

def _ll(X, y, beta):
    logits = np.clip(X @ beta, -30, 30)
    probs = np.clip(expit(logits), 1e-10, 1 - 1e-10)
    return float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

def _std(x):
    return (x - x.mean()) / (x.std() + 1e-9)

def delta_power_ratio(waveform, fs):
    from numpy.fft import rfft, rfftfreq  # noqa: T2-D-5
    freqs = rfftfreq(len(waveform), d=1.0 / fs)
    spectrum = np.abs(rfft(waveform))
    total = float(np.sum(spectrum * spectrum))
    if total < 1e-12:
        return 0.0
    return float(np.sum(spectrum[(freqs >= 0.5) & (freqs <= 4.0)] ** 2)) / total  # noqa: S1-1 — observer layer


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("[209] FULL STACK: Invariants + Bateson + Coupling + CRT")
    print("=" * 72)

    ds = load_patient_dataset(PATIENT_DIR)
    from collections import Counter as Ctr
    modal = Ctr(d["multi_ch"].shape[0] for d in ds).most_common(1)[0][0]
    ds = [d for d in ds if d["multi_ch"].shape[0] == modal]
    y = np.array([1.0 if d["type"] == "seizure" else 0.0 for d in ds])
    n = len(y)
    fs = ds[0]["fs"]
    print(f"  chb01: {int(y.sum())} seizure + {int((1-y).sum())} baseline")

    # Extract features
    print("  Extracting full stack features...")
    all_feats = [extract_full_features(d["multi_ch"]) for d in ds]
    delta = np.array([delta_power_ratio(d["waveform"].astype(np.float64), fs) for d in ds])

    # Feature distributions
    feat_keys = sorted(all_feats[0].keys())
    print(f"\n  {'Feature':<25} {'Seizure':>10} {'Baseline':>10} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for k in feat_keys:
        sei = np.mean([f[k] for f, yi in zip(all_feats, y) if yi == 1])
        bas = np.mean([f[k] for f, yi in zip(all_feats, y) if yi == 0])
        diff = sei - bas
        if abs(diff) > 0.001:
            print(f"  {k:<25} {sei:>10.4f} {bas:>10.4f} {diff:>+10.4f}")

    # Nested models
    print(f"\n{'='*72}")
    print("NESTED MODEL COMPARISON")

    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit(X0, y))
    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit(X1, y))
    r2_delta = 1 - ll1 / ll0

    def test_feature_set(name, keys, df):
        arrs = [np.array([f[k] for f in all_feats]) for k in keys]
        cols = [np.ones(n), _std(delta)] + [_std(a) for a in arrs]
        X = np.column_stack(cols)
        ll = _ll(X, y, _fit(X, y))
        r2 = 1 - ll / ll0
        dr2 = r2 - r2_delta
        lr_stat = 2 * (ll - ll1)
        p = float(chi2.sf(max(0, lr_stat), df=df))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:<40} ΔR²={dr2:+.4f}  p={p:.6f} {sig}")
        return dr2, p

    print(f"\n  {'Model':<40} {'Result'}")
    print(f"  {'-'*40} {'-'*30}")
    print(f"  {'delta only':<40} R²={r2_delta:.4f}")

    # Basic [209]
    test_feature_set("+ basic [209] (sing/synch/ent/f)",
                     ["basic_sing", "basic_synch", "basic_entropy", "basic_mean_f"], 4)

    # Raw invariants
    test_feature_set("+ raw invariants (C/F/G/f)",
                     ["mean_raw_C", "mean_raw_F", "mean_raw_G", "mean_raw_f"], 4)

    # Bateson levels
    test_feature_set("+ Bateson levels (L0-L3 entropy)",
                     ["L0_entropy", "L1_entropy", "L2_entropy", "L3_entropy"], 4)

    # Coupling matrix
    test_feature_set("+ coupling (mean/max/std/PR/eig1)",
                     ["coupling_mean", "coupling_max", "coupling_std",
                      "coupling_participation", "coupling_eig1_frac"], 5)

    # CRT
    test_feature_set("+ CRT (rate/e24_entropy)",
                     ["crt_rate", "e24_entropy"], 2)

    # Full stack
    full_keys = ["mean_raw_C", "mean_raw_f",
                 "L1_entropy", "L2_entropy",
                 "coupling_mean", "coupling_eig1_frac",
                 "crt_rate", "e24_entropy"]
    test_feature_set("+ FULL STACK (8 best features)",
                     full_keys, 8)

    # Save
    results = {"patient": "chb01", "n": n}
    out_path = Path("eeg_209_full_stack_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
