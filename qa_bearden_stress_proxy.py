#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=single_char_projection, state_alphabet=mod24_A1_compliant"
"""
qa_bearden_stress_proxy.py — Bearden phase-conjugate retest with continuous stress proxy.

Replaces binary attack label (used in domain-2/domain-3 pilots) with a continuous
dataset-independent stress proxy — character Shannon entropy — matching the finance
Bearden setup structurally rather than via label proxy.

Pre-registration: results/bearden_stress_proxy/PRE_REGISTRATION.md (locked 2026-04-05
BEFORE any entropy values computed on the cached data).

Primary hypothesis (sign LOCKED NEGATIVE from [155] finance):
    raw_r(qci_gap, char_entropy) <= -0.10 on BOTH datasets, with p < 0.05 on both
    where qci_gap = qci_local - qci_global (per cert fixture).

Targets BOTH datasets under a single locked rule. No retuning.
"""

from __future__ import annotations

import gzip as _gzip
import json
import math
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent

# Reuse shared helpers from domain-2 script
sys.path.insert(0, str(REPO))
from qa_bearden_injection_denial import (  # type: ignore
    extract_qa_features,
    MODULUS,
    SEED,
    MIN_LEN,
)

RESULTS_DIR = REPO / "results" / "bearden_stress_proxy"

# Data sources (cached from previous runs)
DEEPSET_JSONL = REPO / "qa_alphageometry_ptolemy" / "external_validation_data" / \
    "deepset_prompt_injections_full" / "deepset_prompt_injections_full.jsonl"
XTRAM1_JSONL = REPO / "qa_alphageometry_ptolemy" / "external_validation_data" / \
    "xtram1_safeguard_injection" / "xtram1_safeguard_injection.jsonl"

N_PERM_PRIMARY = 10_000


# ------------------------------------------------------------------ stress proxies
def char_entropy(text: str) -> float:
    """Shannon entropy of the empirical char distribution, bits/char."""
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    h = 0.0
    for c_count in counts.values():
        p = c_count / n
        h -= p * math.log2(p)
    return h


def gzip_ratio(text: str) -> float:
    """Gzip compressibility ratio in [0, 1+]. Higher = less compressible."""
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    if len(raw) == 0:
        return 0.0
    compressed = _gzip.compress(raw)
    return len(compressed) / len(raw)


# ------------------------------------------------------------------ stats helpers
def permutation_r(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int) -> tuple[float, float, np.ndarray]:
    """Permutation null by shuffling y; returns (observed_r, perm_p, null_dist)."""
    observed_r, _ = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        r_i, _ = stats.pearsonr(x, y_shuf)
        null[i] = r_i
    p = float((np.abs(null) >= abs(observed_r)).sum() + 1) / (n_perm + 1)
    return float(observed_r), p, null


def partial_r_control(gap: np.ndarray, target: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """Partial Pearson r(gap, target | control)."""
    X = np.column_stack([control, np.ones(len(control))])
    beta_gap, *_ = np.linalg.lstsq(X, gap, rcond=None)
    beta_tgt, *_ = np.linalg.lstsq(X, target, rcond=None)
    gap_r = gap - X @ beta_gap
    tgt_r = target - X @ beta_tgt
    r, p = stats.pearsonr(gap_r, tgt_r)
    return float(r), float(p)


# ------------------------------------------------------------------ data loading
def load_rows(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing cached dataset: {jsonl_path}")
    with open(jsonl_path) as f:
        return [json.loads(ln) for ln in f]


def extract_dataset(name: str, jsonl_path: Path) -> pd.DataFrame:
    """Extract QA features + stress proxies for a dataset, return per-prompt DataFrame."""
    print(f"[{name}] loading {jsonl_path.name} ...")
    rows = load_rows(jsonl_path)
    records = []
    for r in rows:
        text = r.get("text", "") or ""
        qa = extract_qa_features(text)
        # extract_qa_features already uses the corrected gap = local - global
        records.append({
            "label": int(r.get("label", 0)),
            "length": qa["length"],
            "qci_local": qa["qci_local"],
            "qci_global": qa["qci_global"],
            "qci_gap": qa["qci_gap"],
            "char_entropy": char_entropy(text) if qa["length"] >= MIN_LEN else np.nan,
            "gzip_ratio": gzip_ratio(text) if qa["length"] >= MIN_LEN else np.nan,
        })
    df = pd.DataFrame(records).dropna(subset=["qci_gap", "char_entropy"]).reset_index(drop=True)
    print(f"[{name}] valid rows: {len(df)}")
    return df


def analyze_dataset(name: str, df: pd.DataFrame, perm_seed: int) -> dict:
    """Run all statistical tests on one dataset. Returns dict of results."""
    gap = df["qci_gap"].values.astype(float)
    entropy = df["char_entropy"].values.astype(float)
    gzipr = df["gzip_ratio"].values.astype(float)
    length = df["length"].values.astype(float)

    # Primary: raw r(gap, entropy)
    r_ent, p_ent = stats.pearsonr(gap, entropy)
    # Permutation p for primary
    print(f"[{name}] running {N_PERM_PRIMARY} permutations for primary test ...")
    _, perm_p, null_dist = permutation_r(gap, entropy, N_PERM_PRIMARY, perm_seed)

    # Secondary: gzip
    r_gzip, p_gzip = stats.pearsonr(gap, gzipr)
    # Length
    r_len, p_len = stats.pearsonr(gap, length)
    # H5: partial r(gap, entropy | length)
    pr_ent_len, pp_ent_len = partial_r_control(gap, entropy, length)

    out = {
        "n": int(len(df)),
        "primary_r_entropy": float(r_ent),
        "primary_p_entropy_analytic": float(p_ent),
        "primary_p_entropy_permutation": float(perm_p),
        "secondary_r_gzip": float(r_gzip),
        "secondary_p_gzip": float(p_gzip),
        "secondary_r_length": float(r_len),
        "secondary_p_length": float(p_len),
        "partial_r_entropy_given_length": float(pr_ent_len),
        "partial_p_entropy_given_length": float(pp_ent_len),
        "mean_gap": float(gap.mean()),
        "mean_entropy": float(entropy.mean()),
        "null_dist_mean": float(null_dist.mean()),
        "null_dist_std": float(null_dist.std()),
    }
    print(f"[{name}] PRIMARY  r(gap, entropy)          = {r_ent:+.4f}  analytic p = {p_ent:.4g}  perm p = {perm_p:.4g}")
    print(f"[{name}] SECONDARY r(gap, gzip_ratio)      = {r_gzip:+.4f}  p = {p_gzip:.4g}")
    print(f"[{name}] SECONDARY r(gap, length)          = {r_len:+.4f}  p = {p_len:.4g}")
    print(f"[{name}] H5       partial(gap, ent | len)  = {pr_ent_len:+.4f}  p = {pp_ent_len:.4g}")
    return out


# ------------------------------------------------------------------ decision
def classify_outcome(res2: dict, res3: dict) -> str:
    r2 = res2["primary_r_entropy"]
    r3 = res3["primary_r_entropy"]
    p2 = res2["primary_p_entropy_permutation"]
    p3 = res3["primary_p_entropy_permutation"]

    sign_consistent = (r2 < 0) and (r3 < 0)

    if not sign_consistent:
        return "NULL"

    if (r2 <= -0.15) and (r3 <= -0.15) and (p2 < 0.01) and (p3 < 0.01):
        return "STRONG"

    hit_10_either = (r2 <= -0.10) or (r3 <= -0.10)
    sig_both = (p2 < 0.05) and (p3 < 0.05)
    if hit_10_either and sig_both:
        return "WEAK"

    hit_10_one = int(r2 <= -0.10 and p2 < 0.05) + int(r3 <= -0.10 and p3 < 0.05)
    if hit_10_one == 1:
        return "SINGLE"

    return "NULL"


# ------------------------------------------------------------------ main
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 72)
    print("QA Bearden Phase-Conjugate — Continuous Stress Proxy (char entropy)")
    print("Primary: raw_r(qci_gap, char_entropy), sign locked NEGATIVE (finance)")
    print("Datasets: deepset/prompt-injections + xTRam1/safe-guard-prompt-injection")
    print("=" * 72)

    # --- load and extract both datasets
    df2 = extract_dataset("deepset", DEEPSET_JSONL)
    df3 = extract_dataset("xtram1", XTRAM1_JSONL)

    # --- analyze both
    res2 = analyze_dataset("deepset", df2, perm_seed=SEED)
    res3 = analyze_dataset("xtram1", df3, perm_seed=SEED + 1)

    # --- decision
    outcome = classify_outcome(res2, res3)
    print("=" * 72)
    print(f"OUTCOME: {outcome}")
    print(f"  deepset primary r = {res2['primary_r_entropy']:+.4f}  perm p = {res2['primary_p_entropy_permutation']:.4g}")
    print(f"  xtram1  primary r = {res3['primary_r_entropy']:+.4f}  perm p = {res3['primary_p_entropy_permutation']:.4g}")
    print("=" * 72)

    # --- outputs
    df2.to_csv(RESULTS_DIR / "per_prompt_deepset.csv", index=False)
    df3.to_csv(RESULTS_DIR / "per_prompt_xtram1.csv", index=False)

    summary = {
        "schema": "qa_bearden_stress_proxy.v1",
        "pre_registration": "results/bearden_stress_proxy/PRE_REGISTRATION.md",
        "primary_test": "raw_r(qci_gap, char_entropy)",
        "sign_locked": "negative (matches [155] finance partial_r = -0.17 to -0.42)",
        "gap_definition": "qci_local - qci_global (per cert fixture bpc_pass_default.json)",
        "datasets": {
            "deepset/prompt-injections": res2,
            "xTRam1/safe-guard-prompt-injection": res3,
        },
        "thresholds": {
            "STRONG": "both datasets r <= -0.15 AND p_perm < 0.01",
            "WEAK": "both signs negative, p_perm < 0.05 on both, at least one r <= -0.10",
            "SINGLE": "both signs negative, exactly one dataset meets r <= -0.10 AND p < 0.05",
            "NULL": "signs inconsistent OR both sub-threshold",
        },
        "outcome": outcome,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # --- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (name, df, res) in zip(axes, [
            ("deepset (n=661)", df2, res2),
            ("xTRam1 (n=10296)", df3, res3),
        ]):
            ax.scatter(df["char_entropy"], df["qci_gap"], s=4, alpha=0.3)
            # Least-squares line
            x = df["char_entropy"].values.astype(float)
            y = df["qci_gap"].values.astype(float)
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, m * xs + b, "r-", lw=2,
                    label=f"r = {res['primary_r_entropy']:+.4f}, p = {res['primary_p_entropy_permutation']:.2g}")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("char entropy (bits/char)")
            ax.set_ylabel("qci_gap = local - global")
            ax.set_title(name)
            ax.legend()
        fig.suptitle(f"Bearden phase-conjugate continuous stress proxy — {outcome}")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "entropy_vs_gap_scatter.png", dpi=120)
        plt.close(fig)
        print("[plot] wrote entropy_vs_gap_scatter.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"[done] results in {RESULTS_DIR}")
    return outcome


if __name__ == "__main__":
    sys.exit(0 if main() in ("STRONG", "WEAK", "SINGLE", "NULL") else 1)
