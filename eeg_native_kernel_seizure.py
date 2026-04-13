#!/usr/bin/env python3
"""EEG seizure detection via QA-native kernels on per-segment connectivity graphs.

Pipeline:
    1. Load CHB-MIT chb01 segments (164 segments: 56 baseline, 108 seizure)
    2. For each segment, build functional connectivity graph (channel correlation)
    3. Apply auto_qa_cluster to each connectivity graph (picks best QA kernel)
    4. Extract community-structure features per segment:
       - number of communities found
       - modularity Q of the partition
       - which kernel was selected
       - within-community vs between-community correlation strength
       - Eisenstein norm distribution per community
    5. Test if these community-derived features discriminate seizure vs baseline

Hypothesis: seizure segments have different community structure than baseline
(e.g., more giant-component domination from hypersynchrony), and the QA-native
kernel exposes this through a different selected kernel + higher modularity.

QA axiom compliance:
    A1: per-channel (b, e) integer-derived
    A2: d, a derived from b, e
    T2: correlation matrices are observer-layer; QA features are integer
"""

QA_COMPLIANCE = "observer=eeg_native_kernel_seizure, state_alphabet=eeg_connectivity_graph_communities, tier=community_structure_seizure_classifier"

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path("/home/player2/signal_experiments")))
sys.path.insert(0, str(Path("/home/player2/wt-papers")))
sys.path.insert(0, str(Path("/home/player2/signal_experiments/qa_lab")))

import networkx as nx

from qa_graph.auto_kernel import (
    auto_qa_cluster, compute_modularity, compute_graph_invariants
)
from qa_graph.signed_temporal import eisenstein_norm

np.random.seed(42)

PATIENT_DIR = Path("/home/player2/wt-papers/archive/phase_artifacts/phase2_data/eeg/chbmit/chb01")
CORR_THRESHOLD = 0.3


def build_connectivity_graph(multi_ch, threshold=CORR_THRESHOLD):
    """Build functional connectivity graph from EEG correlations."""
    n_ch = multi_ch.shape[0]
    corr = np.corrcoef(multi_ch)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    adj = (np.abs(corr) > threshold).astype(float)
    return adj, corr


def extract_community_features(multi_ch, k_communities=2):
    """Extract QA-native community-structure features from EEG segment.

    Returns dict with: kernel_selected, modularity, num_components,
    largest_component_frac, mean_within_corr, mean_between_corr,
    eisenstein_within_var, etc.
    """
    adj, corr = build_connectivity_graph(multi_ch)
    n = adj.shape[0]

    if adj.sum() == 0:
        return {
            "kernel_selected": "baseline", "mapping": "none",
            "modularity": 0.0, "n_components_found": 1,
            "largest_component_frac": 1.0,
            "mean_within_corr": 0.0, "mean_between_corr": 0.0,
            "eisenstein_within_var": 0.0,
            "n_edges": 0, "mean_degree": 0.0,
        }

    G = nx.from_numpy_array(adj)

    # Auto-pick best QA kernel + map
    try:
        labels, info = auto_qa_cluster(G, k=k_communities)
    except Exception:
        labels = np.zeros(n, dtype=int)
        info = {"mapping": "fallback", "kernel": "baseline", "modularity": 0.0}

    # Community-structure features
    sizes = Counter(labels.tolist())
    largest = max(sizes.values()) if sizes else n
    n_comp_found = len(sizes)

    # Within vs between correlation
    within_corrs, between_corrs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr[i, j])
            if labels[i] == labels[j]:
                within_corrs.append(c)
            else:
                between_corrs.append(c)
    mean_within = float(np.mean(within_corrs)) if within_corrs else 0.0
    mean_between = float(np.mean(between_corrs)) if between_corrs else 0.0

    # Eisenstein norm per channel (using b=degree, e=core)
    degree = dict(G.degree())
    core = nx.core_number(G)
    norms = []
    for v in range(n):
        b = max(1, int(degree[v]))
        e = max(1, int(core[v]))
        norms.append(eisenstein_norm(b, e))

    # Within-community Eisenstein variance (lower = more uniform communities)
    eis_within_var = 0.0
    n_within = 0
    for c in sizes:
        members = [norms[i] for i in range(n) if labels[i] == c]
        if len(members) > 1:
            eis_within_var += float(np.var(members)) * len(members)
            n_within += len(members)
    if n_within > 0:
        eis_within_var = eis_within_var / n_within

    return {
        "kernel_selected": info.get("kernel", "unknown"),
        "mapping": info.get("mapping", "unknown"),
        "modularity": float(info.get("modularity", 0.0)),
        "n_components_found": n_comp_found,
        "largest_component_frac": largest / n,
        "mean_within_corr": mean_within,
        "mean_between_corr": mean_between,
        "within_between_ratio": mean_within / (mean_between + 1e-9),
        "eisenstein_within_var": eis_within_var,
        "eisenstein_global_mean": float(np.mean(norms)),
        "eisenstein_global_std": float(np.std(norms)),
        "n_edges": int(adj.sum() // 2),
        "mean_degree": float(adj.sum(axis=1).mean()),
    }


def load_chb01_segments():
    try:
        from eeg_chbmit_scale import load_patient_dataset
        return load_patient_dataset(PATIENT_DIR, window_sec=4.0)
    except Exception as ex:
        print(f"WARN: {ex}")
        return []


def main():
    print("=== EEG Native-Kernel Seizure Detection ===\n")

    segments = load_chb01_segments()
    if not segments:
        print("No data.")
        return

    print(f"Loaded {len(segments)} segments")

    baseline_feats, seizure_feats = [], []
    kernel_log = {"baseline": Counter(), "seizure": Counter()}

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        multi_ch = np.array(seg.get("multi_ch", []))
        if multi_ch.ndim != 2 or multi_ch.shape[0] < 3:
            continue
        label = seg.get("type", "")

        feats = extract_community_features(multi_ch)
        if label == "seizure":
            seizure_feats.append(feats)
            kernel_log["seizure"][feats["kernel_selected"]] += 1
        else:
            baseline_feats.append(feats)
            kernel_log["baseline"][feats["kernel_selected"]] += 1

        if (i + 1) % 30 == 0:
            print(f"  Processed {i+1}/{len(segments)}")

    print(f"\nProcessed: {len(baseline_feats)} baseline, {len(seizure_feats)} seizure")

    # Kernel selection distribution
    print("\n=== Auto-Selected Kernel Distribution ===")
    for label, kc in kernel_log.items():
        total = sum(kc.values())
        print(f"  {label}: {dict(kc)} (n={total})")

    # Per-feature R² and Cohen's d
    print("\n=== Community-Structure Features (seizure vs baseline) ===")
    feature_keys = [k for k in baseline_feats[0]
                    if k not in ("kernel_selected", "mapping")]

    results = {"feature_stats": {}}
    for k in sorted(feature_keys):
        b_vals = np.array([f[k] for f in baseline_feats], dtype=float)
        s_vals = np.array([f[k] for f in seizure_feats], dtype=float)
        b_vals = np.nan_to_num(b_vals, nan=0.0)
        s_vals = np.nan_to_num(s_vals, nan=0.0)

        all_vals = np.concatenate([b_vals, s_vals])
        labels = np.concatenate([np.zeros(len(b_vals)), np.ones(len(s_vals))])
        if all_vals.std() < 1e-10:
            r2 = 0.0
        else:
            r = np.corrcoef(all_vals, labels)[0, 1]
            r2 = r * r if not np.isnan(r) else 0.0

        b_mean, s_mean = float(b_vals.mean()), float(s_vals.mean())
        pooled_std = np.sqrt((b_vals.var() + s_vals.var()) / 2) if len(b_vals) > 1 and len(s_vals) > 1 else 1.0
        d = abs(b_mean - s_mean) / pooled_std if pooled_std > 1e-3 else 0.0
        direction = "seizure>" if s_mean > b_mean else "baseline>"

        results["feature_stats"][k] = {
            "R2": round(r2, 4), "cohens_d": round(d, 4),
            "baseline_mean": round(b_mean, 4), "seizure_mean": round(s_mean, 4),
            "direction": direction,
        }

        if d > 0.2 or r2 > 0.05:
            print(f"  {k:30s}: R²={r2:.4f}  d={d:.3f}  base={b_mean:.4f} seizure={s_mean:.4f}  {direction}")

    # Save
    out_path = Path("eeg_native_kernel_seizure_results.json")
    results["kernel_distribution"] = {k: dict(v) for k, v in kernel_log.items()}
    results["n_baseline"] = len(baseline_feats)
    results["n_seizure"] = len(seizure_feats)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
