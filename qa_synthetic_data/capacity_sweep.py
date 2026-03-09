#!/usr/bin/env python3
"""
QA-ORBIT Minimum-Sufficient-Architecture Study

Question: What is the weakest inductive bias that learns QA structure
across orbit families?

Primary task:   invariant_pred  (onehot | onehot+modular)
Secondary task: shortest_witness (onehot_flat | onehot_flat+oracle)

Ablation: shortest_witness + precomputed norm f(b,e) injected as oracle feature.

Architecture ladder (default alpha=0.0001 unless alpha sweep):
  Capacity:      (64,), (128,), (256,), (512,)
  Depth:         (64,64), (256,256), (256,256,256)
  Wider+deep:    (512,256), (512,256,128)
  Alpha sweep:   MLP(256,256) x alpha in {0.0001, 0.001, 0.01}

Tiers (by test accuracy):
  SOLVED:  >= 0.95
  STRONG:  >= 0.85
  PARTIAL: >= 0.60
  FAILED:  <  0.40

Usage:
    python capacity_sweep.py [--modulus 24]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────

ARCH_LADDER: List[Tuple[str, Tuple[int, ...]]] = [
    ("MLP(64,)",           (64,)),
    ("MLP(128,)",          (128,)),
    ("MLP(256,)",          (256,)),
    ("MLP(512,)",          (512,)),
    ("MLP(64,64)",         (64, 64)),
    ("MLP(256,256)",       (256, 256)),
    ("MLP(256,256,256)",   (256, 256, 256)),
    ("MLP(512,256)",       (512, 256)),
    ("MLP(512,256,128)",   (512, 256, 128)),
]

ALPHA_SWEEP_ARCH = (256, 256)
ALPHA_VALUES = [0.0001, 0.001, 0.01]

TIERS = [
    (0.95, "SOLVED"),
    (0.85, "STRONG"),
    (0.60, "PARTIAL"),
    (0.40, "FAILED"),
]


def classify_tier(acc: float) -> str:
    for threshold, label in TIERS:
        if acc >= threshold:
            return label
    return "FAILED"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_dir: Path, modulus: int, split: str) -> List[Dict[str, Any]]:
    path = data_dir / f"QA_SYNTHETIC_mod{modulus}_{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_task(rows: List[Dict], task_type: str) -> List[Dict]:
    return [r for r in rows if r["task_type"] == task_type]


# ── Feature extractors ────────────────────────────────────────────────────────

def qa_norm(b: int, e: int, m: int) -> int:
    return (b * b + b * e - e * e) % m


def feat_onehot_invariant(row: Dict, m: int) -> np.ndarray:
    """49-dim: b onehot (m) + e onehot (m) + 1/m scalar."""
    inp = row["input"]
    b, e = int(inp["b"]), int(inp["e"])
    vec = np.zeros(2 * m + 1, dtype=np.float64)
    vec[b] = 1.0
    vec[m + e] = 1.0
    vec[2 * m] = 1.0 / m
    return vec


def feat_onehot_modular(row: Dict, m: int) -> np.ndarray:
    """onehot + explicit modular/Fourier features (inductive bias variant).

    Adds:
      - Fourier harmonics of b and e (sin/cos of 2πb/m, 2πe/m)
      - Fourier harmonics of cross terms (b+e, b-e)
      - Unmodded quadratic norm (b²+be-e²)/m² as a scalar
      - Wrapped cross term b·e / m²
    These give the model direct computational scaffolding for modular
    quadratic forms without revealing the answer.
    """
    base = feat_onehot_invariant(row, m)
    inp = row["input"]
    b, e = int(inp["b"]), int(inp["e"])
    tau = 2.0 * math.pi / m
    extras = np.array([
        math.sin(tau * b),
        math.cos(tau * b),
        math.sin(tau * e),
        math.cos(tau * e),
        math.sin(tau * (b + e)),
        math.cos(tau * (b + e)),
        math.sin(tau * (b - e)),
        math.cos(tau * (b - e)),
        float(b * b + b * e - e * e) / float(m * m),  # unmodded norm, scaled
        float(b * e) / float(m * m),                  # cross term
    ], dtype=np.float64)
    return np.concatenate([base, extras])


def feat_onehot_flat(row: Dict, m: int) -> np.ndarray:
    """97-dim: b + e + b_target + e_target onehot (m dims each) + 1/m scalar."""
    inp = row["input"]
    b, e = int(inp["b"]), int(inp["e"])
    bt, et = int(inp["b_target"]), int(inp["e_target"])
    vec = np.zeros(4 * m + 1, dtype=np.float64)
    vec[b] = 1.0
    vec[m + e] = 1.0
    vec[2 * m + bt] = 1.0
    vec[3 * m + et] = 1.0
    vec[4 * m] = 1.0 / m
    return vec


def feat_onehot_oracle(row: Dict, m: int) -> np.ndarray:
    """onehot_flat + precomputed norms f(b,e) and f(b*,e*) as oracle features.

    This ablation injects the algebraic invariant directly, testing whether
    access to f collapses the shortest_witness task. Labeled as ablation/control,
    not a candidate architecture.
    """
    base = feat_onehot_flat(row, m)
    inp = row["input"]
    b, e = int(inp["b"]), int(inp["e"])
    bt, et = int(inp["b_target"]), int(inp["e_target"])
    # One-hot encode source and target norms (m dims each)
    src_norm_oh = np.zeros(m, dtype=np.float64)
    tgt_norm_oh = np.zeros(m, dtype=np.float64)
    src_norm_oh[qa_norm(b, e, m)] = 1.0
    tgt_norm_oh[qa_norm(bt, et, m)] = 1.0
    return np.concatenate([base, src_norm_oh, tgt_norm_oh])


# ── Parameter counting ────────────────────────────────────────────────────────

def count_params(input_size: int, hidden: Tuple[int, ...], n_classes: int) -> int:
    layers = [input_size] + list(hidden) + [n_classes]
    total = 0
    for i in range(len(layers) - 1):
        total += layers[i] * layers[i + 1] + layers[i + 1]  # weights + biases
    return total


# ── Sweep runner ──────────────────────────────────────────────────────────────

def _make_mlp(hidden: Tuple[int, ...], alpha: float) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        alpha=alpha,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=42,
        verbose=False,
    )


def _run_one(
    arch_label: str,
    hidden: Tuple[int, ...],
    alpha: float,
    task: str,
    encoding: str,
    X_tr, y_tr_enc,
    X_dev, y_dev_enc,
    X_te, y_te_enc,
    n_params: int,
) -> Dict:
    t0 = time.time()
    clf = _make_mlp(hidden, alpha)
    clf.fit(X_tr, y_tr_enc)
    elapsed = time.time() - t0

    tr_acc  = clf.score(X_tr,  y_tr_enc)
    dev_acc = clf.score(X_dev, y_dev_enc)
    te_acc  = clf.score(X_te,  y_te_enc)
    gap     = dev_acc - te_acc

    row = {
        "task":         task,
        "encoding":     encoding,
        "architecture": arch_label,
        "alpha":        alpha,
        "n_params":     n_params,
        "train":        round(tr_acc,  4),
        "dev":          round(dev_acc, 4),
        "test":         round(te_acc,  4),
        "iid_ood_gap":  round(gap,     4),
        "tier":         classify_tier(te_acc),
        "elapsed_s":    round(elapsed, 1),
    }
    print(
        f"  {arch_label:<28} α={alpha}  params={n_params:>8,}  "
        f"train={tr_acc:.3f}  dev={dev_acc:.3f}  test={te_acc:.3f}  "
        f"gap={gap:+.3f}  [{classify_tier(te_acc)}]  ({elapsed:.1f}s)"
    )
    return row


def run_sweep(
    task_type: str,
    encoding_label: str,
    feat_fn,
    m: int,
    splits_raw: Dict[str, List[Dict]],
) -> List[Dict]:
    print(f"\n{'='*70}")
    print(f"TASK: {task_type}  |  ENCODING: {encoding_label}")
    print(f"{'='*70}")

    def build(split):
        rows = filter_task(splits_raw[split], task_type)
        X = np.array([feat_fn(r, m) for r in rows], dtype=np.float64)
        y = np.array([r["answer"] for r in rows])
        return X, y

    X_tr, y_tr = build("train")
    X_dev, y_dev = build("dev")
    X_te, y_te = build("test")

    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_dev, y_te]))
    y_tr_enc  = le.transform(y_tr)
    y_dev_enc = le.transform(y_dev)
    y_te_enc  = le.transform(y_te)

    n_classes  = len(le.classes_)
    input_size = X_tr.shape[1]

    print(
        f"Input dims: {input_size}  |  Classes: {n_classes}  |  "
        f"Train: {len(X_tr)}  Dev: {len(X_dev)}  Test: {len(X_te)}"
    )

    results = []

    # ── Architecture ladder (alpha=0.0001) ────────────────────────────────────
    print("\n  -- Architecture ladder (α=0.0001) --")
    for arch_label, hidden in ARCH_LADDER:
        n_params = count_params(input_size, hidden, n_classes)
        row = _run_one(
            arch_label, hidden, 0.0001, task_type, encoding_label,
            X_tr, y_tr_enc, X_dev, y_dev_enc, X_te, y_te_enc, n_params,
        )
        results.append(row)

    # ── Alpha sweep (ALPHA_SWEEP_ARCH) ────────────────────────────────────────
    print(f"\n  -- Alpha sweep (arch={ALPHA_SWEEP_ARCH}) --")
    n_params = count_params(input_size, ALPHA_SWEEP_ARCH, n_classes)
    for alpha in ALPHA_VALUES:
        arch_label = f"MLP(256,256) α={alpha}"
        row = _run_one(
            arch_label, ALPHA_SWEEP_ARCH, alpha, task_type, encoding_label,
            X_tr, y_tr_enc, X_dev, y_dev_enc, X_te, y_te_enc, n_params,
        )
        results.append(row)

    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(all_rows: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY — MINIMUM SUFFICIENT ARCHITECTURE")
    print("=" * 80)
    hdr = f"{'Task':<20} {'Encoding':<22} {'Architecture':<30} {'Params':>8} {'Test':>6} {'IID-OOD':>8} Tier"
    print(hdr)
    print("-" * 80)
    for r in all_rows:
        print(
            f"{r['task']:<20} {r['encoding']:<22} {r['architecture']:<30} "
            f"{r['n_params']:>8,} {r['test']:>6.3f} {r['iid_ood_gap']:>+8.3f} {r['tier']}"
        )

    # Per-task first-model-reaching-tier table
    task_enc_pairs = [
        ("invariant_pred",   "onehot"),
        ("invariant_pred",   "onehot+modular"),
        ("shortest_witness", "onehot_flat"),
        ("shortest_witness", "onehot_flat+oracle"),
    ]
    print()
    for task, enc in task_enc_pairs:
        subset = [r for r in all_rows if r["task"] == task and r["encoding"] == enc]
        if not subset:
            continue
        print(f"\n── {task} / {enc} ──")
        for t_label in ["SOLVED", "STRONG", "PARTIAL"]:
            matches = [r for r in subset if r["tier"] == t_label]
            if matches:
                best = min(matches, key=lambda r: r["n_params"])
                print(
                    f"  {t_label:<8}: {best['architecture']}"
                    f"  ({best['n_params']:,} params)"
                    f"  test={best['test']:.3f}  gap={best['iid_ood_gap']:+.3f}"
                )
            else:
                print(f"  {t_label:<8}: —  (none reached)")

    # Best test by task/encoding (for paper table)
    print("\n── Best test accuracy per (task, encoding) ──")
    for task, enc in task_enc_pairs:
        subset = [r for r in all_rows if r["task"] == task and r["encoding"] == enc]
        if not subset:
            continue
        best = max(subset, key=lambda r: r["test"])
        print(
            f"  {task}/{enc}: best test={best['test']:.3f}"
            f"  [{best['tier']}]  arch={best['architecture']}"
            f"  params={best['n_params']:,}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=24)
    args = parser.parse_args()

    m = args.modulus
    data_dir = Path(__file__).parent / "data"

    print(f"QA-ORBIT Minimum-Sufficient-Architecture Sweep  (mod-{m})")
    print("Loading splits...")
    splits_raw = {s: load_split(data_dir, m, s) for s in ["train", "dev", "test"]}
    total = sum(len(v) for v in splits_raw.values())
    print(f"  {total:,} total tasks loaded")

    all_rows: List[Dict] = []

    # PRIMARY — invariant_pred
    all_rows += run_sweep("invariant_pred",   "onehot",            feat_onehot_invariant, m, splits_raw)
    all_rows += run_sweep("invariant_pred",   "onehot+modular",    feat_onehot_modular,   m, splits_raw)

    # SECONDARY — shortest_witness
    all_rows += run_sweep("shortest_witness", "onehot_flat",       feat_onehot_flat,      m, splits_raw)

    # ABLATION — shortest_witness + oracle norm
    all_rows += run_sweep("shortest_witness", "onehot_flat+oracle", feat_onehot_oracle,   m, splits_raw)

    print_summary(all_rows)

    out = Path(__file__).parent / f"capacity_sweep_results_mod{m}.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
