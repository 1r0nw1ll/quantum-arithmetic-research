#!/usr/bin/env python3
"""
eeg_graph_observer.py — QA Graph Observer for EEG Seizure Detection

Treats each EEG window as a GRAPH (23 channels = nodes, correlations = edges).
Uses the same QA feature map that produced ARI +0.056 on football network.

Canonical (b,e) assignment forced by graph theory:
  b = degree       (number of strong connections — connectivity breadth)
  e = core_number  (k-core depth — structural embeddedness)

Both are natural integers. No quantization needed. Role-distinct per [208]:
  b = base state (how connected), e = generator (how deep in the core).

Pipeline per window:
  1. Compute correlation matrix (23 x 23) — observer layer
  2. Threshold to binary graph — observer layer
  3. Per-channel: b = degree, e = core_number — boundary crossing (int)
  4. Compute QA invariants per channel — QA layer
  5. Aggregate invariants as window features — projection layer
  6. Nested logistic regression vs delta baseline

Theorem NT compliant: continuous EEG -> correlation -> threshold -> integer
graph properties -> QA algebra. Boundary crossed once at step 3.
"""

QA_COMPLIANCE = {
    "spec": "QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1",
    "observer": "graph_degree_core_qa_feature_map",
    "b_meaning": "graph_degree",
    "e_meaning": "graph_core_number",
    "axioms_checked": ["A1", "A2", "T1", "T2", "S1", "S2"],
    "qa_layer_types": "int",
}

import sys
import json
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import chi2

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments/qa_lab")))

from eeg_chbmit_scale import load_patient_dataset
from qa_orbit_rules import orbit_family, norm_f
from qa_graph.feature_map import compute_qa_invariants

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
MOD = 9
THRESHOLD_PERCENTILE = 75  # correlation threshold for binary graph


# ── Graph construction from EEG correlation matrix ──────────────────────────

def correlation_graph(multi_ch: np.ndarray, threshold_pct: float = THRESHOLD_PERCENTILE):
    """
    Build binary graph from EEG channel correlations.
    Observer layer: continuous correlations -> thresholded binary adjacency.

    Returns adjacency matrix (n_ch x n_ch), boolean.
    """
    corr = np.corrcoef(multi_ch)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    abs_corr = np.abs(corr)

    threshold = np.percentile(abs_corr[abs_corr > 0], threshold_pct)
    adj = abs_corr >= threshold
    return adj


def graph_degree(adj: np.ndarray) -> list[int]:
    """Degree of each node. Integer by construction (count of True neighbours)."""
    return [int(adj[i].sum()) for i in range(adj.shape[0])]  # noqa: T2-b-2


def graph_core_number(adj: np.ndarray) -> list[int]:
    """
    K-core number of each node. Integer by construction.
    Iterative peeling: remove nodes with degree < k, increment k.
    """
    n = adj.shape[0]
    degree = [int(adj[i].sum()) for i in range(n)]  # noqa: T2-b-2
    core = [0] * n  # Python int list, not numpy — S2 compliant
    remaining = [True] * n

    for k in range(1, n + 1):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not remaining[i]:
                    continue
                # Count remaining neighbors
                nbr_count = sum(1 for j in range(n) if adj[i, j] and remaining[j])
                if nbr_count < k:
                    remaining[i] = False
                    core[i] = k - 1
                    changed = True
        if not any(remaining):
            break

    # Any still remaining get the max core
    for i in range(n):
        if remaining[i]:
            core[i] = k
    return core


# ── QA Feature Extraction ───────────────────────────────────────────────────

# Which of the 21 canonical invariants to use as features
QA21_KEYS = [
    "B", "E", "D", "A",
    "X", "C", "F", "G",
    "L", "H", "I",
    "J", "K", "W", "Y", "Z",
]


def extract_graph_qa_features(multi_ch: np.ndarray) -> dict:
    """
    Full QA graph observer pipeline for one EEG window.

    Returns dict with orbit fractions, mean QA invariants, graph stats.
    """
    adj = correlation_graph(multi_ch)
    deg = graph_degree(adj)
    core = graph_core_number(adj)
    n_ch = multi_ch.shape[0]

    # A1 compliance: shift to {1,...,m}
    # Degree and core can be 0; add 1 to ensure no-zero
    b_vals = [d + 1 for d in deg]   # {1, ..., n_ch}
    e_vals = [c + 1 for c in core]  # {1, ..., max_core+1}

    # Classify orbits and compute QA invariants per channel
    orbits = []
    f_values = []
    qa_features_per_channel = []

    for i in range(n_ch):
        b = int(b_vals[i])
        e = int(e_vals[i])

        # Orbit classification (mod-9)
        b9 = ((b - 1) % MOD) + 1  # A1: map to {1,...,9}
        e9 = ((e - 1) % MOD) + 1
        orb = orbit_family(b9, e9, MOD)
        orbits.append(orb)
        f_values.append(norm_f(b9, e9))

        # QA invariants from raw (b,e) — not modular-reduced
        inv = compute_qa_invariants(float(b), float(e))
        qa_features_per_channel.append(inv)

    # Orbit fractions
    cosmos_frac = sum(1 for o in orbits if o == "cosmos") / n_ch
    satellite_frac = sum(1 for o in orbits if o == "satellite") / n_ch
    singularity_frac = sum(1 for o in orbits if o == "singularity") / n_ch

    # Aggregate QA invariants (mean across channels)
    qa_means = {}
    for key in QA21_KEYS:
        vals = [ch[key] for ch in qa_features_per_channel]
        qa_means[f"qa_{key}"] = float(np.mean(vals))

    # Graph-level statistics
    mean_degree = float(np.mean(deg))
    mean_core = float(np.mean(core))
    degree_std = float(np.std(deg))

    return {
        "cosmos_frac": cosmos_frac,
        "satellite_frac": satellite_frac,
        "singularity_frac": singularity_frac,
        "mean_f": float(np.mean(f_values)),
        "mean_degree": mean_degree,
        "mean_core": mean_core,
        "degree_std": degree_std,
        "n_channels": n_ch,
        **qa_means,
    }


# ── Logistic regression ─────────────────────────────────────────────────────

def _fit_logistic(X, y, lr=0.1, n_iter=3000, l2=1e-4):
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
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("QA GRAPH OBSERVER — Channel Correlation Network")
    print("b = degree (connectivity), e = core_number (embeddedness)")
    print("QA feature map: 21 canonical invariants per channel")
    print("Patient: chb01 (real CHB-MIT data)")
    print("=" * 72)

    # Load
    print("\nLoading chb01 data...")
    dataset = load_patient_dataset(PATIENT_DIR)
    n_sei = sum(1 for d in dataset if d["type"] == "seizure")
    n_base = sum(1 for d in dataset if d["type"] == "baseline")
    print(f"  {n_sei} seizure + {n_base} baseline windows")

    # Extract features
    print("\nExtracting QA graph features...")
    all_features = []
    all_labels = []
    for d in dataset:
        feats = extract_graph_qa_features(d["multi_ch"])
        feats["type"] = d["type"]
        all_features.append(feats)
        all_labels.append(1 if d["type"] == "seizure" else 0)

    y = np.array(all_labels, dtype=float)

    # Orbit distributions
    print(f"\n  {'Type':<12} {'Cosmos':>8} {'Sat':>8} {'Sing':>8} {'mean_f':>8} {'deg':>6} {'core':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for label in ["seizure", "baseline"]:
        feats = [f for f in all_features if f["type"] == label]
        cos_ = np.mean([f["cosmos_frac"] for f in feats])
        sat_ = np.mean([f["satellite_frac"] for f in feats])
        sin_ = np.mean([f["singularity_frac"] for f in feats])
        mf = np.mean([f["mean_f"] for f in feats])
        md = np.mean([f["mean_degree"] for f in feats])
        mc = np.mean([f["mean_core"] for f in feats])
        print(f"  {label:<12} {cos_:>8.4f} {sat_:>8.4f} {sin_:>8.4f} {mf:>8.2f} {md:>6.2f} {mc:>6.2f}")

    # QA invariant distributions (top discriminators)
    print(f"\n  QA INVARIANT DISTRIBUTIONS (mean across channels)")
    print(f"  {'Invariant':<12} {'Seizure':>10} {'Baseline':>10} {'Diff':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for key in QA21_KEYS:
        fkey = f"qa_{key}"
        sei_mean = np.mean([f[fkey] for f in all_features if f["type"] == "seizure"])
        bas_mean = np.mean([f[fkey] for f in all_features if f["type"] == "baseline"])
        diff = sei_mean - bas_mean
        if abs(diff) > 0.01:  # only show non-trivial differences
            print(f"  {key:<12} {sei_mean:>10.2f} {bas_mean:>10.2f} {diff:>+10.2f}")

    # Delta power ratio baseline
    from eeg_rns_observer import delta_power_ratio  # noqa: T2-D-5
    delta = np.array([
        delta_power_ratio(d["waveform"].astype(np.float64), d["fs"])
        for d in dataset
    ])

    # Nested model: delta vs delta + QA graph features
    # Use orbit fracs + mean_f + mean_degree + top QA invariants
    sing = np.array([f["singularity_frac"] for f in all_features])
    cos_ = np.array([f["cosmos_frac"] for f in all_features])
    mean_f = np.array([f["mean_f"] for f in all_features])
    qa_G = np.array([f["qa_G"] for f in all_features])

    n = len(y)
    X0 = np.ones((n, 1))
    ll0 = _ll(X0, y, _fit_logistic(X0, y))

    X1 = np.c_[np.ones(n), _std(delta)]
    ll1 = _ll(X1, y, _fit_logistic(X1, y))

    # Model 2: delta + QA graph (sing, cos, mean_f, G)
    X2 = np.c_[np.ones(n), _std(delta), _std(sing), _std(cos_),
               _std(mean_f), _std(qa_G)]
    ll2 = _ll(X2, y, _fit_logistic(X2, y))

    r2_1 = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_2 = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    delta_r2 = r2_2 - r2_1
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2.sf(max(0, lr_stat), df=4))

    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    print(f"\n{'='*72}")
    print("NESTED LOGISTIC REGRESSION")
    print(f"\n  R2 (delta only):                {r2_1:.4f}")
    print(f"  R2 (delta + QA graph):           {r2_2:.4f}")
    print(f"  DR2 (QA graph beyond delta):     {delta_r2:+.4f}")
    print(f"  LR stat:                         {lr_stat:.3f}")
    print(f"  p(QA adds):                      {p_val:.6f} {sig(p_val)}")

    print(f"\n  COMPARISON (all on chb01):")
    print(f"  Observer 3 (topographic k-means):     DR2 = +0.252")
    print(f"  RNS eigenspectrum:                    DR2 = +0.138")
    print(f"  Canonical (amplitude x propagation):  DR2 = +0.094")
    print(f"  QA Graph (degree x core_number):      DR2 = {delta_r2:+.4f}")

    # Save
    results = {
        "observer": "qa_graph_degree_core",
        "patient": "chb01",
        "n_seizure": int(sum(y)),
        "n_baseline": int(len(y) - sum(y)),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "r2_delta": float(r2_1),
        "r2_full": float(r2_2),
        "delta_r2": float(delta_r2),
        "p_qa_add": float(p_val),
    }
    out_path = Path("eeg_graph_observer_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
