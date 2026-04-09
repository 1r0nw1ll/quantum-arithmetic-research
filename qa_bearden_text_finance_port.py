#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=topographic_6D_text, state_alphabet=kmeans_k6_mod24_A1"
"""
qa_bearden_text_finance_port.py — Finance-framework port to text domain.

Hypothesis: The attempt-3 text Bearden test (qa_bearden_injection_denial.py) was
FRAMEWORK-LIMITED, not domain-limited. Attempt-3 used raw ord%24 encoding with
window=7 and same-stream global features. The finance framework uses:
  - 6D standardized feature vectors per window
  - K-means k=6 clustering -> QA states
  - QCI window=63
  - Separate global feature stream

This script ports the finance architecture to text and tests whether QCI predicts
injection labels, controlling for baseline text statistics.

Datasets: deepset/prompt-injections (662 rows) + xTRam1/safe-guard (10296 rows).

Pre-registration (locked before run):
  Primary: partial_r(QCI, label | char_entropy, length) sign-locked NEGATIVE
  STRONG: |r| >= 0.15 AND perm_p < 0.01
  WEAK:   |r| >= 0.10 AND perm_p < 0.05
  NULL:   otherwise

QA Axiom Compliance:
  A1: States in {1,...,24}. qa_mod: ((b+e-1) % 24) + 1
  A2: d = b+e, a = b+2e derived
  S1: b*b not b-squared
  S2: b, e are int
  T2: Float features = observer projections only. K-means labels (int) are QA input.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans

# ------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parent
DEEPSET_JSONL = (
    REPO / "qa_alphageometry_ptolemy" / "external_validation_data"
    / "deepset_prompt_injections_full" / "deepset_prompt_injections_full.jsonl"
)
XTRAM1_JSONL = (
    REPO / "qa_alphageometry_ptolemy" / "external_validation_data"
    / "xtram1_safeguard_injection" / "xtram1_safeguard_injection.jsonl"
)
RESULTS_DIR = REPO / "results" / "bearden_text_finance_port"

# ------------------------------------------------------------------ constants
MODULUS = 24
N_CLUSTERS = 6
# Finance cluster-to-state map (from script 35 / qa_observer)
CLUSTER_MAP = {0: 8, 1: 16, 2: 24, 3: 5, 4: 3, 5: 11}
QCI_WINDOW = 63          # Finance standard (attempt-3 used 7)
FEATURE_WINDOW = 63      # Sliding window for 6D text features
FEATURE_STRIDE = 1       # Step between windows
MIN_TEXT_LEN = 128        # Need enough chars for meaningful windows
SEED = 42
N_PERM = 10_000


# ------------------------------------------------------------------ A1-compliant QA
def qa_mod(x: int, m: int) -> int:
    """A1-compliant modular arithmetic: result in {1,...,m}."""
    return ((int(x) - 1) % m) + 1


def qa_step(b: int, e: int, m: int) -> tuple:
    """One QA step. Returns (new_b, new_e). A1-compliant."""
    new_b = e
    new_e = qa_mod(b + e, m)
    return new_b, new_e


def qa_tuple(b: int, e: int, m: int) -> tuple:
    """Compute (b, e, d, a) tuple. A2: d = b+e, a = b+2e (mod m, A1)."""
    d = qa_mod(b + e, m)       # A2
    a = qa_mod(b + 2 * e, m)   # A2
    return (int(b), int(e), int(d), int(a))


# ------------------------------------------------------------------ 6D text features (observer projection)
def char_entropy(chars: str) -> float:
    """Shannon entropy of character distribution (bits/char). OBSERVER PROJECTION."""
    if not chars:
        return 0.0
    n = len(chars)
    counts = Counter(chars)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def bigram_diversity(chars: str) -> float:
    """Fraction of unique bigrams out of total bigrams. OBSERVER PROJECTION."""
    if len(chars) < 2:
        return 0.0
    bigrams = [chars[i:i+2] for i in range(len(chars) - 1)]
    return len(set(bigrams)) / len(bigrams)


def punctuation_rate(chars: str) -> float:
    """Fraction of punctuation characters. OBSERVER PROJECTION."""
    if not chars:
        return 0.0
    punct = sum(1 for c in chars if not c.isalnum() and not c.isspace())
    return punct / len(chars)


def uppercase_rate(chars: str) -> float:
    """Fraction of uppercase characters among alpha chars. OBSERVER PROJECTION."""
    if not chars:
        return 0.0
    alpha = sum(1 for c in chars if c.isalpha())
    if alpha == 0:
        return 0.0
    upper = sum(1 for c in chars if c.isupper())
    return upper / alpha


def digit_rate(chars: str) -> float:
    """Fraction of digit characters. OBSERVER PROJECTION."""
    if not chars:
        return 0.0
    return sum(1 for c in chars if c.isdigit()) / len(chars)


def mean_char_mod24(chars: str) -> float:
    """Mean of (ord(c) % 24 + 1) for all chars. OBSERVER PROJECTION.
    Note: this is a FLOAT average used as a feature, NOT a QA state."""
    if not chars:
        return 0.0
    return np.mean([(ord(c) % MODULUS) + 1 for c in chars])


def extract_6d_features(text: str, window: int, stride: int) -> np.ndarray:
    """Extract 6D feature vectors from sliding character windows.

    Returns (n_windows, 6) array. Each row is an OBSERVER PROJECTION
    that will be clustered (never directly used as QA state).
    """
    n = len(text)
    if n < window:
        return np.empty((0, 6))

    features = []
    for start in range(0, n - window + 1, stride):
        chunk = text[start:start + window]
        feat = [
            char_entropy(chunk),        # F1: entropy
            bigram_diversity(chunk),     # F2: bigram diversity
            punctuation_rate(chunk),     # F3: punctuation rate
            uppercase_rate(chunk),       # F4: uppercase rate
            digit_rate(chunk),           # F5: digit rate
            mean_char_mod24(chunk),      # F6: mean char mod24
        ]
        features.append(feat)

    return np.array(features, dtype=np.float64)


# ------------------------------------------------------------------ global text stats (separate stream)
def global_text_stats(text: str) -> dict:
    """Compute global text statistics — SEPARATE from QA pipeline.
    These are observer projections used as controls, never mixed with QA state."""
    return {
        "global_entropy": char_entropy(text),
        "global_bigram_div": bigram_diversity(text),
        "global_punct_rate": punctuation_rate(text),
        "global_upper_rate": uppercase_rate(text),
        "global_digit_rate": digit_rate(text),
        "global_length": len(text),
    }


# ------------------------------------------------------------------ QCI computation
def compute_qci_from_labels(labels: np.ndarray, cmap: dict, modulus: int,
                            window: int) -> np.ndarray:
    """Compute QCI (T-operator coherence) from integer cluster labels.

    T2 compliant: labels are integer cluster assignments (discrete).
    QA states come from CLUSTER_MAP (integer -> integer).
    No floats enter QA logic.

    Returns array of length len(labels)-2, with NaN where rolling not filled.
    """
    t_match = []
    for t in range(len(labels) - 2):
        b = int(cmap[int(labels[t])])         # S2: int
        e = int(cmap[int(labels[t + 1])])     # S2: int
        actual = int(cmap[int(labels[t + 2])])
        pred = qa_mod(b + e, modulus)          # A1: ((b+e-1) % 24) + 1
        t_match.append(1 if pred == actual else 0)

    series = pd.Series(t_match, dtype=float)
    qci = series.rolling(window, min_periods=window // 2).mean()
    return qci.values


# ------------------------------------------------------------------ statistics
def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on X (with intercept)."""
    X_ = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    return y - X_ @ beta


def partial_r(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> tuple:
    """Partial Pearson r(y, x | Z). Returns (r, p)."""
    yr = residualize(y, Z)
    xr = residualize(x, Z)
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p)


def permutation_partial_r(y: np.ndarray, x: np.ndarray, Z: np.ndarray,
                          n_perm: int, seed: int) -> tuple:
    """Permutation p-value for partial_r by shuffling y."""
    observed, _ = partial_r(y, x, Z)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        null[i], _ = partial_r(y_shuf, x, Z)
    p = float((np.abs(null) >= abs(observed)).sum() + 1) / (n_perm + 1)
    return observed, p, null


# ------------------------------------------------------------------ data
def load_rows(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path) as f:
        return [json.loads(ln) for ln in f]


# ------------------------------------------------------------------ per-dataset pipeline
def process_dataset(name: str, rows: list, km_fit: KMeans = None) -> tuple:
    """Full finance-style pipeline on one text dataset.

    1. Extract 6D features per sliding window
    2. Standardize (z-score)
    3. K-means k=6 -> cluster labels (QA input)
    4. QCI from cluster labels
    5. Aggregate QCI per prompt
    6. Correlate with injection label, controlling for global stats

    Returns (DataFrame of per-prompt results, fitted KMeans, summary dict).
    """
    print(f"\n{'='*60}")
    print(f"[{name}] Processing {len(rows)} prompts (finance-style pipeline)")
    print(f"{'='*60}")

    # --- Step 1: Extract 6D features for ALL prompts (pooled for k-means)
    all_features = []       # pooled for k-means fitting
    prompt_indices = []     # which prompt each window belongs to
    prompt_offsets = []     # start index in all_features for each prompt
    valid_prompts = []      # prompts long enough

    for idx, row in enumerate(rows):
        text = row.get("text", "") or ""
        if len(text) < MIN_TEXT_LEN:
            continue
        feats = extract_6d_features(text, FEATURE_WINDOW, FEATURE_STRIDE)
        if len(feats) < QCI_WINDOW + 2:
            # Need at least QCI_WINDOW + 2 windows for meaningful QCI
            continue
        prompt_offsets.append(len(all_features))
        all_features.extend(feats.tolist())
        prompt_indices.extend([len(valid_prompts)] * len(feats))
        valid_prompts.append(row)

    if not all_features:
        print(f"[{name}] No prompts with enough text. Skipping.")
        return None, km_fit, {}

    X = np.array(all_features, dtype=np.float64)
    print(f"[{name}] Valid prompts: {len(valid_prompts)} / {len(rows)}")
    print(f"[{name}] Total windows: {X.shape[0]} ({X.shape[1]}D features)")

    # --- Step 2: Standardize (z-score per feature, finance-style)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-10
    X_std = (X - mu) / sigma

    # --- Step 3: K-means k=6
    if km_fit is None:
        print(f"[{name}] Fitting K-means k={N_CLUSTERS} ...")
        km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
        km.fit(X_std)
    else:
        km = km_fit
        print(f"[{name}] Using pre-fit K-means")

    labels = km.predict(X_std)  # integer cluster labels -> QA input (T2 compliant)
    print(f"[{name}] Cluster distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Step 4 & 5: QCI per prompt
    records = []
    prompt_indices_arr = np.array(prompt_indices)

    for pidx, row in enumerate(valid_prompts):
        text = row.get("text", "") or ""
        mask = prompt_indices_arr == pidx
        prompt_labels = labels[mask]

        if len(prompt_labels) < QCI_WINDOW + 2:
            continue

        # Compute QCI from cluster labels (finance pipeline)
        qci_series = compute_qci_from_labels(
            prompt_labels, CLUSTER_MAP, MODULUS, QCI_WINDOW
        )

        # Aggregate: mean QCI over the prompt
        valid_qci = qci_series[np.isfinite(qci_series)]
        if len(valid_qci) == 0:
            continue

        mean_qci = float(np.mean(valid_qci))

        # Global text stats (SEPARATE stream — T2 compliant)
        gstats = global_text_stats(text)

        records.append({
            "label": int(row.get("label", 0)),
            "mean_qci": mean_qci,
            "n_windows": int(len(prompt_labels)),
            "n_valid_qci": int(len(valid_qci)),
            **gstats,
        })

    df = pd.DataFrame(records)
    print(f"[{name}] Prompts with valid QCI: {len(df)}")
    if len(df) == 0:
        return df, km, {}

    # --- Label balance
    n_benign = int((df["label"] == 0).sum())
    n_attack = int((df["label"] == 1).sum())
    print(f"[{name}] Labels: benign={n_benign}, attack={n_attack}")

    # --- Step 6: Statistical tests
    y = df["label"].values.astype(np.float64)
    qci = df["mean_qci"].values.astype(np.float64)
    entropy = df["global_entropy"].values.astype(np.float64)
    length = df["global_length"].values.astype(np.float64)

    # Raw correlations
    raw_r, raw_p = stats.pearsonr(qci, y)
    spearman_r, spearman_p = stats.spearmanr(qci, y)
    print(f"[{name}] raw Pearson  r(QCI, label) = {raw_r:+.4f}  p = {raw_p:.4g}")
    print(f"[{name}] raw Spearman r(QCI, label) = {spearman_r:+.4f}  p = {spearman_p:.4g}")

    # Partial correlation controlling for entropy (the obvious confound)
    pr_ent, pp_ent = partial_r(y, qci, entropy.reshape(-1, 1))
    print(f"[{name}] partial r(QCI, label | entropy) = {pr_ent:+.4f}  p = {pp_ent:.4g}")

    # Partial correlation controlling for entropy + length
    controls = np.column_stack([entropy, length])
    pr_full, pp_full = partial_r(y, qci, controls)
    print(f"[{name}] partial r(QCI, label | entropy, length) = {pr_full:+.4f}  p = {pp_full:.4g}  <-- PRIMARY")

    # Permutation test on primary
    print(f"[{name}] Running {N_PERM} permutations ...")
    obs_r, perm_p, null_dist = permutation_partial_r(y, qci, controls, N_PERM, SEED)
    print(f"[{name}] permutation p = {perm_p:.4g} (null mean={null_dist.mean():+.4f}, sd={null_dist.std():.4f})")

    # Group means
    mean_qci_benign = float(df[df["label"] == 0]["mean_qci"].mean())
    mean_qci_attack = float(df[df["label"] == 1]["mean_qci"].mean())
    delta = mean_qci_attack - mean_qci_benign
    print(f"[{name}] mean QCI benign={mean_qci_benign:.4f}, attack={mean_qci_attack:.4f}, delta={delta:+.4f}")

    # Logistic regression: QCI -> label
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        X_lr = qci.reshape(-1, 1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        accs = []
        for tr, te in skf.split(X_lr, y.astype(int)):
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(X_lr[tr], y[tr].astype(int))
            accs.append(clf.score(X_lr[te], y[te].astype(int)))
        logreg_acc = float(np.mean(accs))
        print(f"[{name}] LogReg(QCI->label) 5-fold CV acc = {logreg_acc:.4f}")
    except Exception as e:
        print(f"[{name}] LogReg failed: {e}")
        logreg_acc = None

    summary = {
        "n_total": len(rows),
        "n_valid": len(df),
        "n_benign": n_benign,
        "n_attack": n_attack,
        "raw_pearson_r": float(raw_r),
        "raw_pearson_p": float(raw_p),
        "raw_spearman_r": float(spearman_r),
        "raw_spearman_p": float(spearman_p),
        "partial_r_entropy": float(pr_ent),
        "partial_p_entropy": float(pp_ent),
        "partial_r_entropy_length": float(pr_full),
        "partial_p_entropy_length_analytic": float(pp_full),
        "partial_p_entropy_length_perm": float(perm_p),
        "null_mean": float(null_dist.mean()),
        "null_std": float(null_dist.std()),
        "mean_qci_benign": mean_qci_benign,
        "mean_qci_attack": mean_qci_attack,
        "delta_attack_minus_benign": float(delta),
        "logreg_cv_acc": logreg_acc,
    }

    return df, km, summary


def classify_outcome(res_deep: dict, res_xtram: dict) -> str:
    """Classify using pre-registered thresholds, sign locked NEGATIVE."""
    outcomes = []
    for name, res in [("deepset", res_deep), ("xtram1", res_xtram)]:
        if not res:
            outcomes.append("SKIP")
            continue
        r = res.get("partial_r_entropy_length", 0)
        p = res.get("partial_p_entropy_length_perm", 1)
        sign_neg = r < 0
        if sign_neg and abs(r) >= 0.15 and p < 0.01:
            outcomes.append("STRONG")
        elif sign_neg and abs(r) >= 0.10 and p < 0.05:
            outcomes.append("WEAK")
        else:
            outcomes.append("NULL")

    # Overall
    if "STRONG" in outcomes:
        return "STRONG"
    if "WEAK" in outcomes:
        return "WEAK"
    return "NULL"


# ------------------------------------------------------------------ comparison with attempt-3
def compare_with_attempt3(res_deep: dict, res_xtram: dict):
    """Print comparison table with attempt-3 results."""
    print(f"\n{'='*60}")
    print("COMPARISON WITH ATTEMPT-3 (raw ord%24, window=7)")
    print(f"{'='*60}")

    # Attempt-3 results from memory: deepset r=-0.087, xtram1 r=-0.076
    att3 = {
        "deepset": {"r": -0.087, "description": "raw ord%24, window=7"},
        "xtram1": {"r": -0.076, "description": "raw ord%24, window=7"},
    }
    print(f"{'Metric':<40} {'Attempt-3':>12} {'Finance-Port':>12} {'Delta':>10}")
    print("-" * 76)
    for ds_name, ds_res in [("deepset", res_deep), ("xtram1", res_xtram)]:
        if not ds_res:
            continue
        old_r = att3.get(ds_name, {}).get("r", float("nan"))
        new_r = ds_res.get("partial_r_entropy_length", float("nan"))
        delta = new_r - old_r if not (math.isnan(old_r) or math.isnan(new_r)) else float("nan")
        print(f"  {ds_name} partial_r                     {old_r:>+12.4f} {new_r:>+12.4f} {delta:>+10.4f}")


# ------------------------------------------------------------------ main
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 72)
    print("QA Bearden Phase-Conjugate — Finance-Framework Port to Text")
    print("Architecture: 6D text features -> z-score -> K-means k=6 -> QCI(w=63)")
    print("Primary: partial_r(QCI, label | entropy, length), sign locked NEGATIVE")
    print("=" * 72)

    # --- Load data
    rows_deep = load_rows(DEEPSET_JSONL)
    rows_xtram = load_rows(XTRAM1_JSONL)
    print(f"[data] deepset: {len(rows_deep)} rows, xtram1: {len(rows_xtram)} rows")

    # --- Process each dataset (fit k-means independently per dataset, like finance OOS)
    df_deep, km_deep, res_deep = process_dataset("deepset", rows_deep)
    df_xtram, km_xtram, res_xtram = process_dataset("xtram1", rows_xtram)

    # --- Outcome
    outcome = classify_outcome(res_deep, res_xtram)
    print(f"\n{'='*72}")
    print(f"OUTCOME: {outcome}")
    if res_deep:
        print(f"  deepset: partial_r = {res_deep.get('partial_r_entropy_length', 'N/A'):+.4f}  "
              f"perm_p = {res_deep.get('partial_p_entropy_length_perm', 'N/A'):.4g}")
    if res_xtram:
        print(f"  xtram1:  partial_r = {res_xtram.get('partial_r_entropy_length', 'N/A'):+.4f}  "
              f"perm_p = {res_xtram.get('partial_p_entropy_length_perm', 'N/A'):.4g}")
    print(f"{'='*72}")

    # --- Compare with attempt-3
    compare_with_attempt3(res_deep, res_xtram)

    # --- Save results
    if df_deep is not None and len(df_deep) > 0:
        df_deep.to_csv(RESULTS_DIR / "per_prompt_deepset.csv", index=False)
    if df_xtram is not None and len(df_xtram) > 0:
        df_xtram.to_csv(RESULTS_DIR / "per_prompt_xtram1.csv", index=False)

    summary = {
        "schema": "qa_bearden_text_finance_port.v1",
        "description": "Finance-framework port: 6D text features, k-means k=6, QCI window=63",
        "comparison": "attempt-3 used raw ord%24, window=7, same-stream global",
        "primary_test": "partial_r(QCI, label | char_entropy, length)",
        "sign_locked": "negative (matches [155] finance)",
        "feature_window": FEATURE_WINDOW,
        "qci_window": QCI_WINDOW,
        "n_clusters": N_CLUSTERS,
        "min_text_len": MIN_TEXT_LEN,
        "datasets": {
            "deepset": res_deep,
            "xtram1": res_xtram,
        },
        "thresholds": {
            "STRONG": "|partial_r| >= 0.15 AND perm_p < 0.01, sign negative",
            "WEAK": "|partial_r| >= 0.10 AND perm_p < 0.05, sign negative",
            "NULL": "otherwise",
        },
        "outcome": outcome,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"\n[saved] summary.json")

    # --- Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        datasets = []
        if df_deep is not None and len(df_deep) > 0:
            datasets.append(("deepset", df_deep, res_deep))
        if df_xtram is not None and len(df_xtram) > 0:
            datasets.append(("xtram1", df_xtram, res_xtram))

        if datasets:
            n_plots = len(datasets)
            fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]

            for ax, (dname, df, res) in zip(axes, datasets):
                benign = df[df["label"] == 0]["mean_qci"]
                attack = df[df["label"] == 1]["mean_qci"]
                ax.hist(benign, bins=30, alpha=0.6, label=f"benign (n={len(benign)})", density=True)
                ax.hist(attack, bins=30, alpha=0.6, label=f"attack (n={len(attack)})", density=True)
                r_val = res.get("partial_r_entropy_length", 0)
                p_val = res.get("partial_p_entropy_length_perm", 1)
                ax.set_title(f"{dname}: partial_r={r_val:+.4f}, p={p_val:.3g}")
                ax.set_xlabel("Mean QCI")
                ax.set_ylabel("Density")
                ax.legend()

            fig.suptitle(f"Bearden Finance-Port to Text — {outcome}", fontsize=14)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / "qci_distributions.png", dpi=120)
            plt.close(fig)
            print("[plot] wrote qci_distributions.png")

        # Permutation histogram for xtram1 (larger dataset)
        if res_xtram:
            # Re-run a small permutation just for the plot data
            # Actually we already have it in the null_dist from the function...
            # We'll save a separate diagnostic plot
            pass

    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"[done] results in {RESULTS_DIR}")
    return outcome


if __name__ == "__main__":
    sys.exit(0 if main() in ("STRONG", "WEAK", "NULL") else 1)
