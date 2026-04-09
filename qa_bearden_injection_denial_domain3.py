#!/usr/bin/env python3
# QA_COMPLIANCE = "observer=single_char_projection, state_alphabet=mod24_A1_compliant"
"""
qa_bearden_injection_denial_domain3.py — Domain #3 replication for [155].

Direct replication of qa_bearden_injection_denial.py (2026-04-05 WEAK pilot)
on xTRam1/safe-guard-prompt-injection (10,296 rows, 15.5x larger than deepset).

All feature extraction, observer projection, windowing, statistics, baselines,
and thresholds are INHERITED UNCHANGED from the domain-2 script via import.
Only the data source and output paths differ.

Pre-registration: results/bearden_injection_domain3/PRE_REGISTRATION.md
                  (locked 2026-04-05, sign-locked positive, no retuning).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent

# Reuse domain-2 helpers by importing the module (no side effects at module scope).
sys.path.insert(0, str(REPO))
from qa_bearden_injection_denial import (  # type: ignore
    extract_qa_features,
    partial_r,
    permutation_p,
    char_bigram_logreg,
    classify_outcome,
    MODULUS,
    SEED,
    N_PERM,
    MIN_LEN,
)
sys.path.insert(0, str(REPO / "qa_alphageometry_ptolemy"))
from qa_guardrail.threat_scanner import scan_for_threats  # type: ignore

# ------------------------------------------------------------------ paths
DATA_DIR = REPO / "qa_alphageometry_ptolemy" / "external_validation_data" / "xtram1_safeguard_injection"
DATA_FILE = DATA_DIR / "xtram1_safeguard_injection.jsonl"
MANIFEST_FILE = DATA_DIR / "MANIFEST.json"
RESULTS_DIR = REPO / "results" / "bearden_injection_domain3"

DATASET = "xTRam1/safe-guard-prompt-injection"
SPLITS = {"train": 8236, "test": 2060}


# ------------------------------------------------------------------ data pull
def _fetch_with_retry(url: str, max_retries: int = 6) -> dict:
    """HTTP GET with exponential backoff on 429/5xx."""
    delay = 5.0
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "qa-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                # Honor Retry-After if present
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"  [retry] HTTP {e.code}, sleeping {wait:.0f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            raise
    raise RuntimeError(f"exhausted retries for {url}")


def _fetch_split(split: str, total: int, length: int = 100) -> list[dict]:
    rows: list[dict] = []
    # Resume support: check if partial file exists
    partial_file = DATA_DIR / f".partial_{split}.jsonl"
    if partial_file.exists():
        with open(partial_file) as f:
            rows = [json.loads(ln) for ln in f]
        print(f"  [resume] {split}: loaded {len(rows)} cached rows from partial file")

    offset = len(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    while offset < total:
        batch_len = min(length, total - offset)
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(DATASET, safe='')}"
            f"&config=default&split={split}&offset={offset}&length={batch_len}"
        )
        payload = _fetch_with_retry(url)
        new_rows = []
        for item in payload.get("rows", []):
            row = item.get("row", {})
            new_rows.append({
                "split": split,
                "row_idx": item.get("row_idx"),
                "text": row.get("text", "") or "",
                "label": int(row.get("label", 0)),
            })
        rows.extend(new_rows)
        # Persist partial progress every batch
        with open(partial_file, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        offset += batch_len
        if offset % 1000 == 0 or offset == total:
            print(f"  [pull] {split}: {offset}/{total}")
        time.sleep(0.6)
    # Remove partial marker on success
    if partial_file.exists():
        partial_file.unlink()
    return rows


def pull_data() -> list[dict]:
    if DATA_FILE.exists() and MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        with open(DATA_FILE, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual == manifest.get("sha256"):
            print(f"[data] cache hit: {manifest['rows']} rows, sha256={actual[:12]}")
            with open(DATA_FILE) as f:
                return [json.loads(line) for line in f]
        print("[data] cache sha256 mismatch — refetching")

    print(f"[data] fetching {DATASET} train ({SPLITS['train']}) + test ({SPLITS['test']}) ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = _fetch_split("train", SPLITS["train"]) + _fetch_split("test", SPLITS["test"])
    with open(DATA_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA_FILE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "dataset": DATASET,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "see HF dataset card (public, no auth)",
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "splits": SPLITS,
        "rows": len(rows),
        "sha256": sha,
        "fetch_api": "https://datasets-server.huggingface.co/rows",
        "script": "qa_bearden_injection_denial_domain3.py",
    }
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[data] wrote {len(rows)} rows, sha256={sha[:12]}")
    return rows


# ------------------------------------------------------------------ main
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("=" * 72)
    print("QA Bearden Phase-Conjugate — Domain #3 Replication")
    print(f"Dataset: {DATASET}")
    print("=" * 72)

    # --- 1. data
    rows = pull_data()
    label_counts = {int(k): int(v) for k, v in pd.Series([r["label"] for r in rows]).value_counts().items()}
    print(f"[data] {len(rows)} rows; label balance: {label_counts}")

    # --- 2. features
    print("[feat] extracting QA features + scanner hits ...")
    records = []
    t0 = time.time()
    for i, r in enumerate(rows):
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
        if (i + 1) % 2000 == 0:
            print(f"  [feat] {i+1}/{len(rows)}  ({time.time()-t0:.1f}s)")
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

    print(f"[stat] raw_r                = {raw_r:+.4f}  p = {raw_p:.4g}")
    print(f"[stat] partial_r|len        = {pr_len:+.4f}  p = {pp_len:.4g}")
    print(f"[stat] partial_r|len,hits   = {pr_full:+.4f}  p = {pp_full:.4g}  <-- PRIMARY")

    # --- 4. permutation
    print(f"[perm] running {N_PERM} label permutations (seed={SEED}) ...")
    perm_p, null_dist = permutation_p(y, gap, np.column_stack([length, hits]), N_PERM, SEED)
    print(f"[perm] permutation p        = {perm_p:.4g}  (null mean={null_dist.mean():+.4f}, sd={null_dist.std():.4f})")

    # --- 5. baseline B2
    print("[base] char-bigram logreg 5-fold CV ...")
    try:
        b2_acc = char_bigram_logreg(df_valid["text"].tolist(), y, seed=SEED)
        print(f"[base] B2 accuracy          = {b2_acc:.4f}")
    except Exception as e:
        print(f"[base] B2 failed: {e}")
        b2_acc = None

    # --- 6. descriptives
    mean_gap_benign = df_valid[df_valid["label"] == 0]["qci_gap"].mean()
    mean_gap_attack = df_valid[df_valid["label"] == 1]["qci_gap"].mean()
    scanner_acc = (df_valid["scanner_threats_found"] == df_valid["label"]).mean()
    print(f"[desc] mean qci_gap benign  = {mean_gap_benign:+.4f}")
    print(f"[desc] mean qci_gap attack  = {mean_gap_attack:+.4f}")
    print(f"[desc] delta (attack-benign) = {mean_gap_attack - mean_gap_benign:+.4f}")
    print(f"[desc] B1 scanner accuracy  = {scanner_acc:.4f}")

    # --- 7. decision
    outcome = classify_outcome(pr_full, perm_p)
    print("=" * 72)
    print(f"OUTCOME: {outcome}")
    print(f"Domain-2 reference: partial_r=+0.1393, perm p=0.0006 (WEAK, n=661)")
    print(f"Domain-3 result:    partial_r={pr_full:+.4f}, perm p={perm_p:.4g}  (n={len(df_valid)})")
    print("=" * 72)

    # --- 8. outputs
    df_valid.drop(columns=["text"]).to_csv(RESULTS_DIR / "per_prompt.csv", index=False)

    summary = {
        "schema": "qa_bearden_injection_denial_domain3.v1",
        "pre_registration": "results/bearden_injection_domain3/PRE_REGISTRATION.md",
        "dataset": DATASET,
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
        "domain2_reference": {
            "dataset": "deepset/prompt-injections",
            "n": 661,
            "partial_r_length_hits": 0.1393,
            "perm_p": 0.0006,
            "outcome": "WEAK",
        },
        "sign_locked": "positive",
        "thresholds": {
            "STRONG": "partial_r >= +0.15 AND perm_p < 0.01",
            "WEAK": "partial_r >= +0.10 AND perm_p < 0.05",
            "NULL": "otherwise",
        },
        "outcome": outcome,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # --- 9. plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(null_dist, bins=80, alpha=0.7, label="null (label permuted)")
        ax.axvline(pr_full, color="red", lw=2, label=f"observed = {pr_full:+.4f}")
        ax.axvline(0.15, color="green", ls="--", lw=1, label="STRONG threshold")
        ax.axvline(0.1393, color="orange", ls=":", lw=1, label="domain-2 reference (+0.1393)")
        ax.set_xlabel("partial_r(qci_gap, label | length, scanner_hits)")
        ax.set_ylabel("count")
        ax.set_title(f"Bearden domain-3 replication — {outcome} (p={perm_p:.4g}, n={len(df_valid)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "permutation_histogram.png", dpi=120)
        plt.close(fig)
        print("[plot] wrote permutation_histogram.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print(f"[done] results in {RESULTS_DIR}")
    return outcome


if __name__ == "__main__":
    sys.exit(0 if main() in ("STRONG", "WEAK", "NULL") else 1)
