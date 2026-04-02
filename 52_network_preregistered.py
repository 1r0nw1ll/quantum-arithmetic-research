#!/usr/bin/env python3
"""
52_network_preregistered.py — Pre-registered Φ(D) test on network traffic data
================================================================================

PRE-REGISTRATION (written BEFORE seeing any results):

  Domain: Network intrusion detection (KDD Cup 1999 / NSL-KDD)
  Classification: DISORDER-STRESS (Φ = -1)
  Rationale: Network attacks disrupt normal traffic patterns.
    Normal = regular request/response patterns, consistent packet sizes.
    Attack = disruption (DoS floods, probes, exploits).
    Analogous to cardiac/EEG: attack disrupts the "sinus rhythm" of the network.
  Prediction: QA orbit features discriminate attack from normal beyond
    byte-count / duration baseline.

  This is the 2nd proper Φ(D) attempt (bearing was ceiling-effect).
  Architecture: Same as all other domains (k-means → QA → orbit stats).

Data: NSL-KDD dataset (cleaned version of KDD99)
      ~125K training connections, 41 features, binary labels (normal/attack)
      Available at: https://www.unb.ca/cic/datasets/nsl.html
      Alternative: fetched from common mirrors.

CORRECTED SURROGATE DESIGN: Real labels + real baseline held fixed,
only QA orbit features surrogated.
"""

from __future__ import annotations

QA_COMPLIANCE = "observer=network_topographic, state_alphabet=traffic_microstate"

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from scipy.stats import chi2 as chi2_dist
import urllib.request

from qa_orbit_rules import orbit_family

HERE = os.path.dirname(os.path.abspath(__file__))
MODULUS = 9
N_CLUSTERS = 4
N_SURROGATES = 200
WINDOW_SIZE = 20

np.random.seed(42)

# NSL-KDD column names (41 features + label + difficulty)
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

# Numeric features for clustering (skip categorical: protocol_type, service, flag)
NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Baseline feature: total bytes (simple traffic volume)
BASELINE_COL = "src_bytes"


def download_nslkdd():
    """Download NSL-KDD training set."""
    cache_path = os.path.join(HERE, ".nslkdd_train.csv")
    if os.path.exists(cache_path):
        print("Loading cached NSL-KDD data...")
        return pd.read_csv(cache_path)

    # Try common mirror
    urls = [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ]

    for url in urls:
        try:
            print(f"Downloading NSL-KDD from {url[:60]}...")
            req = urllib.request.Request(url, headers={"User-Agent": "QA-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            # Parse CSV (no header)
            lines = data.strip().split("\n")
            rows = [line.split(",") for line in lines]
            df = pd.DataFrame(rows, columns=KDD_COLUMNS)
            # Convert numeric columns
            for col in NUMERIC_COLS + [BASELINE_COL]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df.to_csv(cache_path, index=False)
            print(f"  Cached to {cache_path}")
            return df
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    print("Could not download NSL-KDD")
    return None


def nested_model(y, baseline, feat1, feat2):
    """Nested LR using sklearn."""
    scaler = StandardScaler()

    p1 = np.clip(y.mean(), 1e-10, 1 - 1e-10)
    ll0 = float(np.sum(y * np.log(p1) + (1 - y) * np.log(1 - p1)))

    X1 = scaler.fit_transform(baseline.reshape(-1, 1))
    lr1 = LogisticRegression(C=1e4, max_iter=1000, solver='lbfgs')
    lr1.fit(X1, y)
    p1_pred = np.clip(lr1.predict_proba(X1)[:, 1], 1e-10, 1 - 1e-10)
    ll1 = float(np.sum(y * np.log(p1_pred) + (1 - y) * np.log(1 - p1_pred)))

    X2 = scaler.fit_transform(np.c_[baseline, feat1, feat2])
    lr2 = LogisticRegression(C=1e4, max_iter=1000, solver='lbfgs')
    lr2.fit(X2, y)
    p2_pred = np.clip(lr2.predict_proba(X2)[:, 1], 1e-10, 1 - 1e-10)
    ll2 = float(np.sum(y * np.log(p2_pred) + (1 - y) * np.log(1 - p2_pred)))

    r2_base = 1.0 - ll1 / ll0 if ll0 != 0 else 0.0
    r2_full = 1.0 - ll2 / ll0 if ll0 != 0 else 0.0
    lr_stat = 2.0 * (ll2 - ll1)
    p_val = float(chi2_dist.sf(max(0.0, lr_stat), df=2))

    return {"r2_base": r2_base, "r2_full": r2_full,
            "delta_r2": r2_full - r2_base, "p_qa_add": p_val}


def main():
    print("=" * 72)
    print("PRE-REGISTERED Φ(D) TEST: Network Traffic Domain")
    print("Classification: DISORDER-STRESS (Φ = -1)")
    print("Prediction: QA orbit features discriminate attack from normal")
    print("Data: NSL-KDD (cleaned KDD Cup 1999)")
    print("=" * 72)

    df = download_nslkdd()
    if df is None:
        return

    # Binary label: normal vs attack
    df["is_attack"] = (df["label"] != "normal").astype(float)
    n_normal = int((df["is_attack"] == 0).sum())
    n_attack = int((df["is_attack"] == 1).sum())
    print(f"\n{len(df)} connections ({n_normal} normal, {n_attack} attack)")

    # Subsample for speed (use 20K balanced)
    max_per_class = 10000
    normal_idx = df[df["is_attack"] == 0].index[:max_per_class]
    attack_idx = df[df["is_attack"] == 1].index[:max_per_class]
    subset_idx = np.concatenate([normal_idx, attack_idx])
    np.random.shuffle(subset_idx)
    df_sub = df.loc[subset_idx].reset_index(drop=True)
    print(f"Subsampled: {len(df_sub)} ({int((df_sub['is_attack']==0).sum())} normal, "
          f"{int((df_sub['is_attack']==1).sum())} attack)")

    # Prepare numeric features
    feat_df = df_sub[NUMERIC_COLS].fillna(0)
    features = feat_df.values.astype(float)

    y = df_sub["is_attack"].values.astype(float)
    baseline = df_sub[BASELINE_COL].values.astype(float)

    # k-means on features (unsupervised)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    km.fit(features_scaled)
    labels = km.predict(features_scaled)

    # QA orbit statistics from sliding windows
    CMAP = {0: 3, 1: 6, 2: 9, 3: 1}

    sing_fracs, cos_fracs = [], []
    for i in range(len(labels) - WINDOW_SIZE):
        seg = labels[i:i + WINDOW_SIZE]
        orbits = []
        for j in range(len(seg) - 1):
            b = CMAP.get(seg[j], 1)
            e = CMAP.get(seg[j + 1], 1)
            orbits.append(orbit_family(int(b), int(e), MODULUS))
        n_orb = len(orbits)
        sing_fracs.append(sum(1 for o in orbits if o == "singularity") / n_orb)
        cos_fracs.append(sum(1 for o in orbits if o == "cosmos") / n_orb)

    sing_fracs = np.array(sing_fracs)
    cos_fracs = np.array(cos_fracs)

    y_aligned = y[WINDOW_SIZE:][:len(sing_fracs)]
    baseline_aligned = baseline[WINDOW_SIZE:][:len(sing_fracs)]

    print(f"Windowed: {len(y_aligned)} samples, "
          f"{int(y_aligned.sum())} attack, {int((1 - y_aligned).sum())} normal")

    # ====================================================================
    # REAL NESTED MODEL
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("NESTED MODEL: attack ~ src_bytes vs ~ src_bytes + QA_sing + QA_cos")
    print("=" * 70)

    result = nested_model(y_aligned, baseline_aligned, sing_fracs, cos_fracs)
    print(f"  R² (src_bytes only): {result['r2_base']:.4f}")
    print(f"  R² (src_bytes + QA): {result['r2_full']:.4f}")
    print(f"  ΔR²: {result['delta_r2']:+.4f}")
    print(f"  p(QA adds): {result['p_qa_add']:.6f}")

    sig = "***" if result["p_qa_add"] < 0.001 else "**" if result["p_qa_add"] < 0.01 else "*" if result["p_qa_add"] < 0.05 else "ns"
    print(f"  Significance: {sig}")

    # ====================================================================
    # CORRECTED SURROGATES
    # ====================================================================
    print(f"\n{'=' * 70}")
    print(f"SURROGATE VALIDATION ({N_SURROGATES} iterations)")
    print("=" * 70)

    surr_types = ["permuted_segments", "random_fracs"]
    surr_results = {st: {"delta_r2": []} for st in surr_types}

    for st in surr_types:
        print(f"\n  {st}:")
        for i in range(N_SURROGATES):
            rng = np.random.RandomState(9000 + i)

            if st == "permuted_segments":
                idx = rng.permutation(len(y_aligned))
                s_sing = sing_fracs[idx]
                s_cos = cos_fracs[idx]
            elif st == "random_fracs":
                s_sing = rng.uniform(0, 0.5, len(y_aligned))
                s_cos = rng.uniform(0, 0.5, len(y_aligned))

            try:
                r = nested_model(y_aligned, baseline_aligned, s_sing, s_cos)
                surr_results[st]["delta_r2"].append(r["delta_r2"])
            except:
                surr_results[st]["delta_r2"].append(0.0)

            if (i + 1) % 50 == 0:
                sys.stdout.write(f"\r    {i + 1}/{N_SURROGATES}")
                sys.stdout.flush()
        print()

    # Compare
    print(f"\n{'=' * 70}")
    print("SURROGATE COMPARISON")
    print("=" * 70)

    summary = {}
    for st in surr_types:
        vals = np.array(surr_results[st]["delta_r2"])
        vals = vals[np.isfinite(vals)]
        mean_s, std_s = np.mean(vals), np.std(vals)
        rank_p = float(np.mean(vals >= result["delta_r2"]))
        z = (result["delta_r2"] - mean_s) / std_s if std_s > 0 else 0
        beats = "BEATS" if rank_p < 0.05 else "FAILS"
        sig_s = "***" if rank_p < 0.001 else "**" if rank_p < 0.01 else "*" if rank_p < 0.05 else "ns"

        print(f"  {st}: real={result['delta_r2']:+.4f}, surr={mean_s:+.4f}±{std_s:.4f}, "
              f"rank_p={rank_p:.4f} → {beats} {sig_s}")
        summary[st] = {"real": float(result["delta_r2"]), "surr_mean": float(mean_s),
                        "surr_std": float(std_s), "rank_p": float(rank_p), "beats": beats == "BEATS"}

    n_pass = sum(1 for v in summary.values() if v["beats"])

    # ====================================================================
    # Φ(D) VERDICT
    # ====================================================================
    print(f"\n{'=' * 70}")
    print("Φ(D) PRE-REGISTRATION VERDICT")
    print("=" * 70)

    checks = {
        "QA adds beyond baseline": result["delta_r2"] > 0 and result["p_qa_add"] < 0.05,
        "Survives surrogates (≥1/2)": n_pass >= 1,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if all(checks.values()):
        print(f"\n  Φ(D) = -1 CONFIRMED for network traffic domain")
        print(f"  2nd successful Φ(D) pre-registration")
    else:
        print(f"\n  Φ(D) test FAILED for network traffic domain")

    output = {
        "domain": "network_traffic_preregistered",
        "phi_preregistered": -1,
        "classification": "disorder-stress",
        "data": "NSL-KDD",
        "n_connections": len(y_aligned),
        "real_result": result,
        "surrogate_summary": summary,
        "n_surr_pass": n_pass,
        "phi_confirmed": all(checks.values()),
    }
    with open(os.path.join(HERE, "52_network_preregistered_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to 52_network_preregistered_results.json")


if __name__ == "__main__":
    main()
