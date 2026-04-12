#!/usr/bin/env python3
"""
eeg_brain_connectivity_graph.py — Community detection on EEG functional connectivity.

Builds a functional connectivity graph from EEG channel correlations:
    1. Load CHB-MIT chb01 multi-channel EEG segments (23 channels)
    2. Compute pairwise Pearson correlation between channels per segment
    3. Threshold to get binary adjacency (|r| > threshold)
    4. Extract QA features per node: (b=degree, e=core_number) → 102 features
       (84 qa_invariants + 18 Diophantine from [133][183][213][214])
    5. Run spectral clustering with baseline vs qa83 vs qa102 (full + Diophantine)
    6. Compare community structure between seizure vs baseline segments

The hypothesis: seizure segments have different functional connectivity
community structure than baseline, and QA features (especially the
Diophantine extensions) help detect this difference.

This is the FIRST brain connectivity graph benchmark in the QA project —
closing the loop between the graph-types initiative and the EEG domain.

QA axiom compliance:
    A1: node features derived from integer degree/core_number
    A2: d, a always derived (never assigned independently)
    T2: correlations and scores are observer-layer measurements
    S1: b*b not b**2
    S2: (b, e) are integers; float features are observer projections

Will Dale + Claude, 2026-04-12.
"""

QA_COMPLIANCE = "observer=eeg_brain_connectivity, state_alphabet=eeg_channel_graph_topology, tier=functional_connectivity_community_detection"

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments/qa_lab")))

from qa_graph.feature_map import qa_feature_vector, compute_qa_invariants
from qa_graph.diophantine_features import diophantine_feature_vector, full_qa_feature_vector

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
CORR_THRESHOLD = 0.3  # |r| > threshold → edge


# ── Evaluation ───────────────────────────────────────────────────────────────

def _comb2(x):
    return 0.0 if x < 2 else x * (x - 1) / 2.0


def compute_ari(pred, gt):
    n = len(pred)
    if n < 2:
        return 0.0
    k = max(pred) + 1
    t = max(gt) + 1
    cm = [[0] * t for _ in range(k)]
    for p, g in zip(pred, gt):
        cm[p][g] += 1
    sum_comb = 0.0
    a_sum = 0.0
    b_sums = [0] * t
    for i in range(k):
        row_sum = sum(cm[i])
        a_sum += _comb2(row_sum)
        for j in range(t):
            sum_comb += _comb2(cm[i][j])
            b_sums[j] += cm[i][j]
    b_sum = sum(_comb2(x) for x in b_sums)
    total = _comb2(n)
    expected = (a_sum * b_sum) / total if total else 0.0
    max_idx = 0.5 * (a_sum + b_sum)
    den = max_idx - expected
    return 0.0 if abs(den) < 1e-12 else (sum_comb - expected) / den


# ── Spectral clustering ─────────────────────────────────────────────────────

def spectral_cluster(W, k, seed=42):
    n = W.shape[0]
    if n < k:
        return np.zeros(n, dtype=int)
    d_vec = W.sum(axis=1)
    d_inv_sqrt = np.where(d_vec > 0, 1.0 / np.sqrt(d_vec), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_norm = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    Z = eigvecs[:, :k]
    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = np.where(row_norms > 1e-10, Z / row_norms, 0.0)
    rng = np.random.RandomState(seed)
    centers = Z[rng.choice(n, min(k, n), replace=False)]
    for _ in range(100):
        dists = np.linalg.norm(Z[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = Z[mask].mean(axis=0)
            else:
                new_centers[c] = centers[c]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


# ── Build connectivity graph ────────────────────────────────────────────────

def build_connectivity_graph(multi_ch, threshold=CORR_THRESHOLD):
    """Build functional connectivity graph from multi-channel EEG.

    Returns:
        adj: (n_ch, n_ch) binary adjacency matrix
        corr: (n_ch, n_ch) correlation matrix (observer-layer float)
        degrees: list of integer degrees per channel
    """
    n_ch = multi_ch.shape[0]
    corr = np.corrcoef(multi_ch)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    adj = (np.abs(corr) > threshold).astype(float)
    degrees = adj.sum(axis=1).astype(int).tolist()
    return adj, corr, degrees


def extract_node_qa_features(adj, mode="full"):
    """Extract QA features per node from the connectivity graph.

    mode: "qa21" (21 features), "qa83" (84 features), "full" (102 features)
    """
    import networkx as nx
    n = adj.shape[0]
    G = nx.from_numpy_array(adj)
    degree = dict(G.degree())
    core = nx.core_number(G)

    features = []
    names = None
    for v in range(n):
        b = max(1, int(degree[v]))  # A1: at least 1
        e = max(1, int(core[v]))    # A1: at least 1

        if mode == "qa21":
            vec, nm = qa_feature_vector(float(b), float(e), mode="qa21")
            vec = list(vec)
        elif mode == "qa83":
            inv = compute_qa_invariants(float(b), float(e))
            nm = sorted(inv.keys())
            vec = [float(inv[k]) for k in nm]
        elif mode == "full":
            vec, nm = full_qa_feature_vector(b, e)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        features.append(vec)
        if names is None:
            names = nm

    return np.array(features), names


# ── Community detection methods ──────────────────────────────────────────────

def detect_communities(adj, features, k=2, mode="baseline"):
    """Run community detection.

    mode:
        'baseline': spectral clustering on raw adjacency
        'qa_kernel': spectral clustering on QA-weighted adjacency
    """
    n = adj.shape[0]
    if mode == "baseline":
        labels = spectral_cluster(adj, k)
        return labels

    if mode == "qa_kernel":
        F_mat = features.copy()
        mu = F_mat.mean(axis=0)
        sigma = F_mat.std(axis=0)
        sigma[sigma < 1e-10] = 1.0
        F_mat = (F_mat - mu) / sigma
        dists = np.sum((F_mat[:, None, :] - F_mat[None, :, :]) *
                       (F_mat[:, None, :] - F_mat[None, :, :]), axis=2)
        tau = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        W = np.exp(-dists / (2 * tau)) * adj
        labels = spectral_cluster(W, k)
        return labels

    raise ValueError(f"unknown mode {mode!r}")


# ── Segment-level analysis ───────────────────────────────────────────────────

def analyze_segment(multi_ch, threshold=CORR_THRESHOLD, k_communities=2):
    """Analyze one EEG segment: build graph, extract features, detect communities.

    Returns dict with graph stats, community structure per method, and
    Diophantine feature summaries.
    """
    adj, corr, degrees = build_connectivity_graph(multi_ch, threshold)
    n_ch = multi_ch.shape[0]
    n_edges = int(adj.sum()) // 2
    mean_corr = float(np.abs(corr).mean())

    result = {
        "n_channels": n_ch,
        "n_edges": n_edges,
        "mean_abs_corr": round(mean_corr, 4),
        "mean_degree": round(sum(degrees) / len(degrees), 2),
        "degree_cv": round(float(np.std(degrees) / np.mean(degrees)), 4) if np.mean(degrees) > 0 else 0,
        "communities": {},
        "diophantine_summary": {},
    }

    for feat_mode in ("qa21", "qa83", "full"):
        features, names = extract_node_qa_features(adj, mode=feat_mode)
        for det_mode in ("baseline", "qa_kernel"):
            key = f"{det_mode}_{feat_mode}" if det_mode == "qa_kernel" else det_mode
            if det_mode == "baseline" and feat_mode != "qa21":
                continue  # baseline doesn't use features
            labels = detect_communities(adj, features, k=k_communities, mode=det_mode)
            comm_sizes = dict(Counter(labels.tolist()))
            result["communities"][key] = {
                "labels": labels.tolist(),
                "sizes": comm_sizes,
                "n_communities": len(comm_sizes),
            }

    # Diophantine feature summary (from full features)
    full_feats, full_names = extract_node_qa_features(adj, mode="full")
    dio_start = 84  # first 84 are qa_invariants, rest are diophantine
    if full_feats.shape[1] > dio_start:
        dio_feats = full_feats[:, dio_start:]
        dio_names = full_names[dio_start:]
        for i, name in enumerate(dio_names):
            vals = dio_feats[:, i]
            result["diophantine_summary"][name] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
            }

    return result


# ── Data loading ─────────────────────────────────────────────────────────────

def load_segments(patient_dir):
    """Load CHB-MIT chb01 segments."""
    try:
        from eeg_chbmit_scale import load_patient_dataset
        return load_patient_dataset(patient_dir, window_sec=4.0)
    except Exception as ex:
        print(f"WARNING: could not load CHB-MIT data: {ex}")
        print("Using synthetic fallback.")
        return _synthetic_segments()


def _synthetic_segments():
    """20 synthetic EEG segments for testing."""
    segments = []
    for i in range(20):
        label = "seizure" if i >= 10 else "baseline"
        n_ch, n_samp = 23, 1024
        if label == "baseline":
            data = np.random.randn(n_ch, n_samp) * 50  # noqa: T2-D-5
        else:
            # Seizure: add correlated component across subset of channels
            data = np.random.randn(n_ch, n_samp) * 50  # noqa: T2-D-5
            shared = np.sin(2 * np.pi * 3 * np.arange(n_samp) / 256) * 80
            for ch in range(0, 12):
                data[ch] += shared
        segments.append({"type": label, "multi_ch": data, "source": f"synth_{i:02d}"})
    return segments


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== EEG Brain Connectivity Graph Benchmark ===")
    print(f"Threshold: |r| > {CORR_THRESHOLD}")
    print(f"Feature modes: qa21 (21), qa83 (84), full (102 = 84 + 18 Diophantine)")
    print()

    segments = load_segments(PATIENT_DIR)
    if not segments:
        print("No data.")
        return

    print(f"Loaded {len(segments)} segments")

    baseline_results = []
    seizure_results = []

    for seg in segments:
        if isinstance(seg, dict):
            multi_ch = np.array(seg.get("multi_ch", []))
            label = seg.get("type", "unknown")
        else:
            continue

        if multi_ch.ndim != 2 or multi_ch.shape[0] < 3:
            continue

        result = analyze_segment(multi_ch, threshold=CORR_THRESHOLD)
        result["label"] = label

        if label == "seizure":
            seizure_results.append(result)
        else:
            baseline_results.append(result)

    print(f"Analyzed: {len(baseline_results)} baseline, {len(seizure_results)} seizure")
    print()

    # Compare graph-level statistics
    print("=== Graph-Level Statistics ===")
    for name, group in [("baseline", baseline_results), ("seizure", seizure_results)]:
        if not group:
            continue
        edges = [r["n_edges"] for r in group]
        corrs = [r["mean_abs_corr"] for r in group]
        degs = [r["mean_degree"] for r in group]
        cvs = [r["degree_cv"] for r in group]
        print(f"  {name:10s}: n={len(group):3d}  edges={np.mean(edges):.1f}±{np.std(edges):.1f}  "
              f"|r|={np.mean(corrs):.3f}  deg={np.mean(degs):.1f}  cv={np.mean(cvs):.3f}")

    # Compare community structure stability across methods
    print()
    print("=== Community Detection: method comparison ===")
    methods = ["baseline", "qa_kernel_qa21", "qa_kernel_qa83", "qa_kernel_full"]
    for method in methods:
        baseline_n_comm = [r["communities"].get(method, {}).get("n_communities", 0) for r in baseline_results if method in r["communities"]]
        seizure_n_comm = [r["communities"].get(method, {}).get("n_communities", 0) for r in seizure_results if method in r["communities"]]

        if baseline_n_comm and seizure_n_comm:
            # Compare community assignments between baseline and qa_kernel methods
            # Use modularity proxy: std of community sizes (more unequal = more structured)
            baseline_size_stds = []
            seizure_size_stds = []
            for r in baseline_results:
                if method in r["communities"]:
                    sizes = list(r["communities"][method]["sizes"].values())
                    baseline_size_stds.append(np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0)
            for r in seizure_results:
                if method in r["communities"]:
                    sizes = list(r["communities"][method]["sizes"].values())
                    seizure_size_stds.append(np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0)

            if baseline_size_stds and seizure_size_stds:
                b_asym = np.mean(baseline_size_stds)
                s_asym = np.mean(seizure_size_stds)
                direction = "seizure>" if s_asym > b_asym else "baseline>"
                print(f"  {method:25s}: community_asymmetry baseline={b_asym:.3f} seizure={s_asym:.3f} {direction}")

    # Compare Diophantine features between seizure and baseline
    print()
    print("=== Diophantine Feature Comparison: seizure vs baseline ===")
    dio_keys = sorted(baseline_results[0]["diophantine_summary"].keys()) if baseline_results else []
    for key in dio_keys:
        b_vals = [r["diophantine_summary"][key]["mean"] for r in baseline_results if key in r["diophantine_summary"]]
        s_vals = [r["diophantine_summary"][key]["mean"] for r in seizure_results if key in r["diophantine_summary"]]
        if b_vals and s_vals:
            b_mean = np.mean(b_vals)
            s_mean = np.mean(s_vals)
            diff = s_mean - b_mean
            direction = "seizure>" if diff > 0 else "baseline>"
            # Simple effect size
            pooled_std = np.sqrt((np.var(b_vals) + np.var(s_vals)) / 2) if len(b_vals) > 1 and len(s_vals) > 1 else 1.0
            cohens_d = abs(diff) / pooled_std if pooled_std > 0.001 else 0.0
            if cohens_d > 0.2:  # Only report medium+ effect sizes
                print(f"  {key:30s}: baseline={b_mean:.4f} seizure={s_mean:.4f} d={cohens_d:.2f} {direction}")

    # Save results
    out = {
        "experiment": "eeg_brain_connectivity_graph",
        "threshold": CORR_THRESHOLD,
        "n_baseline": len(baseline_results),
        "n_seizure": len(seizure_results),
        "feature_modes": {"qa21": 21, "qa83": 84, "full": 102},
    }
    out_path = Path("eeg_brain_connectivity_graph_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
