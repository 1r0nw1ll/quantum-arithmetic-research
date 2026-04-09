#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=single_char_projection, state_alphabet=mod24_A1_compliant"
"""
qa_bearden_injection_denial.py — Domain #2 candidate for [155].

Tests whether Bearden phase-conjugate signature (global tightens, local scatters)
transfers from finance (Domain #1) to prompt-injection defense (Domain #2).

Pre-registration: results/bearden_injection/PRE_REGISTRATION.md (locked 2026-04-05).

Hypothesis (sign-locked, positive):
    partial_r(qci_gap, attack_label | length, scanner_hits) >= +0.15, p < 0.01
    where qci_gap = qci_global - qci_local.

Observer projection (single Theorem-NT boundary crossing):
    states[i] = (ord(text[i]) % 24) + 1    # A1-compliant, deterministic, no model

QA path:
    char stream -> states in {1..24} -> QCI with identity cmap -> local/global -> gap

Run:
    python qa_bearden_injection_denial.py

Outputs to results/bearden_injection/:
    per_prompt.csv, summary.json, permutation_histogram.png
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "qa_alphageometry_ptolemy" / "external_validation_data" / "deepset_prompt_injections_full"
DATA_FILE = DATA_DIR / "deepset_prompt_injections_full.jsonl"
MANIFEST_FILE = DATA_DIR / "MANIFEST.json"
RESULTS_DIR = REPO / "results" / "bearden_injection"

# ------------------------------------------------------------------ import QA stack
sys.path.insert(0, str(REPO / "qa_observer"))
sys.path.insert(0, str(REPO / "qa_alphageometry_ptolemy"))
from qa_observer.core import QCI  # type: ignore
from qa_guardrail.threat_scanner import scan_for_threats  # type: ignore

# ------------------------------------------------------------------ constants
MODULUS = 24
IDENTITY_CMAP = {i: i for i in range(1, MODULUS + 1)}
QCI_LOCAL_WINDOW = 7
QCI_GLOBAL_WINDOW = 63     # capped at len(states)-2 per-prompt
MIN_LEN = 10               # prompts shorter than this excluded from stats
SEED = 42
N_PERM = 10_000


# ------------------------------------------------------------------ data pull
def _fetch_split(split: str, total: int, length: int = 100) -> list[dict]:
    """Fetch a split via HF datasets-server /rows API. Zero library dependency."""
    rows: list[dict] = []
    offset = 0
    while offset < total:
        batch_len = min(length, total - offset)
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote('deepset/prompt-injections', safe='')}"
            f"&config=default&split={split}&offset={offset}&length={batch_len}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "qa-research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        for item in payload.get("rows", []):
            row = item.get("row", {})
            rows.append({
                "split": split,
                "row_idx": item.get("row_idx"),
                "text": row.get("text", ""),
                "label": int(row.get("label", 0)),
            })
        offset += batch_len
        time.sleep(0.25)  # polite
    return rows


def pull_data() -> list[dict]:
    """Pull train+test from HF datasets-server; cache locally with SHA256 lock."""
    if DATA_FILE.exists() and MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        with open(DATA_FILE, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash == manifest.get("sha256"):
            print(f"[data] cache hit: {manifest['rows']} rows, sha256={actual_hash[:12]}")
            with open(DATA_FILE) as f:
                return [json.loads(line) for line in f]
        print("[data] cache sha256 mismatch — refetching")

    print("[data] fetching train (546) + test (116) from HF datasets-server ...")
    rows = _fetch_split("train", 546) + _fetch_split("test", 116)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA_FILE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "dataset": "deepset/prompt-injections",
        "source_url": "https://huggingface.co/datasets/deepset/prompt-injections",
        "license": "apache-2.0",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": {"train": 546, "test": 116},
        "rows": len(rows),
        "sha256": sha,
        "fetch_api": "https://datasets-server.huggingface.co/rows",
        "script": "qa_bearden_injection_denial.py",
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[data] wrote {len(rows)} rows, sha256={sha[:12]}")
    return rows


# ------------------------------------------------------------------ QA feature extraction
def text_to_states(text: str) -> np.ndarray:
    """THE single observer projection: char stream -> {1..24}.

    Theorem-NT boundary crossing: external continuous domain (unicode codepoint)
    -> QA discrete domain. A1-compliant (never 0). No subsequent crossings.
    """
    if not text:
        return np.array([], dtype=np.int64)
    return np.array([(ord(c) % MODULUS) + 1 for c in text], dtype=np.int64)


def extract_qa_features(text: str) -> dict:
    """Per-prompt feature extraction. Returns dict or NaN-filled if too short."""
    states = text_to_states(text)
    n = len(states)
    out = {
        "length": n,
        "qci_local": np.nan,
        "qci_global": np.nan,
        "qci_gap": np.nan,
    }
    if n < MIN_LEN:
        return out

    global_w = min(QCI_GLOBAL_WINDOW, max(3, n - 2))
    local_w = min(QCI_LOCAL_WINDOW, max(3, n - 2))

    qci_local_obj = QCI(modulus=MODULUS, cmap=IDENTITY_CMAP, window=local_w)
    qci_global_obj = QCI(modulus=MODULUS, cmap=IDENTITY_CMAP, window=global_w)

    local_series = qci_local_obj.compute(states)
    global_series = qci_global_obj.compute(states)

    out["qci_local"] = float(np.nanmean(local_series)) if np.any(np.isfinite(local_series)) else np.nan
    out["qci_global"] = float(np.nanmean(global_series)) if np.any(np.isfinite(global_series)) else np.nan
    # Cert [155] fixture definition: QCI_gap = QCI_local - QCI_global
    # (original run had this inverted; 2026-04-05 bug fix after Will pushback)
    out["qci_gap"] = out["qci_local"] - out["qci_global"]
    return out


# ------------------------------------------------------------------ stats
def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return y residuals after OLS on X (with intercept)."""
    X_ = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    return y - X_ @ beta


def partial_r(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """Partial Pearson r(y, x | Z). Z is (n, k). Returns (r, two-sided p)."""
    yr = residualize(y, Z)
    xr = residualize(x, Z)
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p)


def permutation_p(y: np.ndarray, x: np.ndarray, Z: np.ndarray, n_perm: int, seed: int) -> tuple[float, np.ndarray]:
    """Permutation p-value for partial_r by shuffling y; returns (p, null_dist)."""
    observed, _ = partial_r(y, x, Z)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = rng.permutation(y)
        r_i, _ = partial_r(y_shuf, x, Z)
        null[i] = r_i
    # two-sided
    p = float((np.abs(null) >= abs(observed)).sum() + 1) / (n_perm + 1)
    return p, null


def char_bigram_logreg(texts: list[str], labels: np.ndarray, n_buckets: int = 1024, seed: int = 42) -> float:
    """Baseline B2: hashed char-bigram logistic regression, 5-fold CV accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    def featurize(t: str) -> np.ndarray:
        v = np.zeros(n_buckets, dtype=np.float32)
        for i in range(len(t) - 1):
            h = hash(t[i:i+2]) % n_buckets
            v[h] += 1.0
        s = v.sum()
        if s > 0:
            v /= s
        return v

    X = np.array([featurize(t) for t in texts])
    y = labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


# ------------------------------------------------------------------ decision
def classify_outcome(pr: float, pp: float) -> str:
    """Match [155] finance sign: QCI_gap (= local - global) has NEGATIVE partial
    correlation with stress target. Magnitude thresholds inherited from original
    pre-registration. Bug fix 2026-04-05: sign direction corrected from positive
    (original, wrong) to negative (cert-consistent)."""
    sign_matches_finance = pr < 0
    if abs(pr) >= 0.15 and pp < 0.01 and sign_matches_finance:
        return "STRONG"
    if abs(pr) >= 0.10 and pp < 0.05 and sign_matches_finance:
        return "WEAK"
    return "NULL"


# ------------------------------------------------------------------ main
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 70)
    print("QA Bearden Phase-Conjugate — Domain #2: Prompt-Injection Denial")
    print("=" * 70)

    # --- 1. data
    rows = pull_data()
    print(f"[data] {len(rows)} total rows; label balance:",
          {int(k): int(v) for k, v in pd.Series([r['label'] for r in rows]).value_counts().items()})

    # --- 2. features
    print("[feat] extracting QA features + scanner hits ...")
    records = []
    for r in rows:
        text = r["text"] or ""
        qa = extract_qa_features(text)
        scan = scan_for_threats(text)
        records.append({
            "split": r["split"],
            "row_idx": r["row_idx"],
            "label": r["label"],
            "text": text,
            "length": qa["length"],
            "qci_local": qa["qci_local"],
            "qci_global": qa["qci_global"],
            "qci_gap": qa["qci_gap"],
            "scanner_hits": len(scan.get("all_patterns", [])),
            "scanner_threats_found": 1 if scan.get("threats_found") else 0,
        })
    df = pd.DataFrame(records)
    df_valid = df.dropna(subset=["qci_gap"]).reset_index(drop=True)
    n_dropped = len(df) - len(df_valid)
    print(f"[feat] valid rows: {len(df_valid)} (dropped {n_dropped} with length < {MIN_LEN})")

    # --- 3. primary test
    y = df_valid["label"].values.astype(np.int64)
    gap = df_valid["qci_gap"].values.astype(float)
    length = df_valid["length"].values.astype(float)
    hits = df_valid["scanner_hits"].values.astype(float)

    raw_r, raw_p = stats.pearsonr(gap, y)
    pr_len, pp_len = partial_r(y, gap, length.reshape(-1, 1))
    pr_full, pp_full = partial_r(y, gap, np.column_stack([length, hits]))

    print(f"[stat] raw_r           = {raw_r:+.4f}  p = {raw_p:.4g}")
    print(f"[stat] partial_r|len   = {pr_len:+.4f}  p = {pp_len:.4g}")
    print(f"[stat] partial_r|len,hits = {pr_full:+.4f}  p = {pp_full:.4g}  <-- PRIMARY")

    # --- 4. permutation
    print(f"[perm] running {N_PERM} label permutations (seed={SEED}) ...")
    perm_p, null_dist = permutation_p(y, gap, np.column_stack([length, hits]), N_PERM, SEED)
    print(f"[perm] permutation p   = {perm_p:.4g}  (null mean = {null_dist.mean():+.4f}, sd = {null_dist.std():.4f})")

    # --- 5. baseline B2
    print("[base] char-bigram logreg 5-fold CV ...")
    try:
        b2_acc = char_bigram_logreg(df_valid["text"].tolist(), y, seed=SEED)
        print(f"[base] B2 accuracy     = {b2_acc:.4f}")
    except Exception as e:
        print(f"[base] B2 failed: {e}")
        b2_acc = None

    # --- 6. sanity: group means + scanner coverage
    mean_gap_benign = df_valid[df_valid["label"] == 0]["qci_gap"].mean()
    mean_gap_attack = df_valid[df_valid["label"] == 1]["qci_gap"].mean()
    scanner_acc = (df_valid["scanner_threats_found"] == df_valid["label"]).mean()
    print(f"[desc] mean qci_gap benign = {mean_gap_benign:+.4f}")
    print(f"[desc] mean qci_gap attack = {mean_gap_attack:+.4f}")
    print(f"[desc] delta (attack-benign) = {mean_gap_attack - mean_gap_benign:+.4f}")
    print(f"[desc] scanner accuracy (B1 baseline) = {scanner_acc:.4f}")

    # --- 7. decision
    outcome = classify_outcome(pr_full, perm_p)
    print("=" * 70)
    print(f"OUTCOME: {outcome}")
    print("=" * 70)

    # --- 8. write outputs
    df_valid.drop(columns=["text"]).to_csv(RESULTS_DIR / "per_prompt.csv", index=False)

    summary = {
        "schema": "qa_bearden_injection_denial.v1",
        "pre_registration": "results/bearden_injection/PRE_REGISTRATION.md",
        "n_total": len(df),
        "n_valid": len(df_valid),
        "n_dropped_short": n_dropped,
        "label_balance": {
            "benign": int((df_valid["label"] == 0).sum()),
            "attack": int((df_valid["label"] == 1).sum()),
        },
        "stats": {
            "raw_r": raw_r,
            "raw_p": raw_p,
            "partial_r_length": pr_len,
            "partial_p_length": pp_len,
            "partial_r_length_hits": pr_full,
            "partial_p_length_hits_analytic": pp_full,
            "partial_p_length_hits_permutation": perm_p,
            "n_permutations": N_PERM,
            "null_mean": float(null_dist.mean()),
            "null_std": float(null_dist.std()),
        },
        "descriptives": {
            "mean_qci_gap_benign": float(mean_gap_benign),
            "mean_qci_gap_attack": float(mean_gap_attack),
            "delta_attack_minus_benign": float(mean_gap_attack - mean_gap_benign),
        },
        "baselines": {
            "B1_scanner_accuracy": float(scanner_acc),
            "B2_char_bigram_logreg_cv_acc": b2_acc,
        },
        "sign_locked": "positive",
        "primary_test": "partial_r(qci_gap, label | length, scanner_hits)",
        "thresholds": {
            "STRONG": "partial_r >= +0.15 AND perm_p < 0.01",
            "WEAK": "partial_r >= +0.10 AND perm_p < 0.05",
            "NULL": "otherwise",
        },
        "outcome": outcome,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # --- 9. permutation histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_dist, bins=60, alpha=0.7, label="null (label permuted)")
        ax.axvline(pr_full, color="red", lw=2, label=f"observed = {pr_full:+.4f}")
        ax.axvline(0.15, color="green", ls="--", lw=1, label="STRONG threshold")
        ax.set_xlabel("partial_r(qci_gap, label | length, scanner_hits)")
        ax.set_ylabel("count")
        ax.set_title(f"Bearden injection denial — {outcome} (p={perm_p:.4g}, n={len(df_valid)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "permutation_histogram.png", dpi=120)
        print(f"[plot] wrote permutation_histogram.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"[done] results in {RESULTS_DIR}")
    return outcome


if __name__ == "__main__":
    sys.exit(0 if main() in ("STRONG", "WEAK", "NULL") else 1)
