#!/usr/bin/env python3
"""
QA-ORBIT Reachability Baseline — balanced data (50/50 True/False)

Runs the four encoding × model combinations needed to complete the
four-task baseline table in the paper. Previously reported reachability
results used pre-balance data (~92% True); this reruns on the balanced split.

Encodings:
  float_poly  : 13 normalised + quadratic features
  onehot_flat : b, e, b_target, e_target one-hot (4m dims) + 1/m scalar

Models:
  LogisticRegression (linear baseline)
  MLP(64,64)         (nonlinear baseline)

Usage:
    python reachability_baseline.py [--modulus 24]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

TASK = "reachability"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(data_dir: Path, modulus: int, split: str) -> List[Dict[str, Any]]:
    path = data_dir / f"QA_SYNTHETIC_mod{modulus}_{split}.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["task_type"] == TASK:
                    rows.append(r)
    return rows


# ── Feature extractors ────────────────────────────────────────────────────────

def _poly2(x: float, y: float) -> List[float]:
    return [x, y, x * x, x * y, y * y]


def _wrap_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def feat_float_poly(row: Dict, m: int) -> np.ndarray:
    inp = row["input"]
    b  = inp["b"]  / m
    e  = inp["e"]  / m
    bt = inp["b_target"] / m
    et = inp["e_target"] / m
    feats = _poly2(b, e) + [1.0 / m] + _poly2(bt, et) + [
        _wrap_dist(b, bt),
        _wrap_dist(e, et),
    ]
    return np.array(feats, dtype=np.float64)


def feat_onehot_flat(row: Dict, m: int) -> np.ndarray:
    inp = row["input"]
    b  = int(inp["b"])
    e  = int(inp["e"])
    bt = int(inp["b_target"])
    et = int(inp["e_target"])
    vec = np.zeros(4 * m + 1, dtype=np.float64)
    vec[b] = 1.0
    vec[m + e] = 1.0
    vec[2 * m + bt] = 1.0
    vec[3 * m + et] = 1.0
    vec[4 * m] = 1.0 / m
    return vec


# ── Runner ────────────────────────────────────────────────────────────────────

def run(rows_tr, rows_dev, rows_te, feat_fn, m, model, model_label, enc_label):
    X_tr  = np.array([feat_fn(r, m) for r in rows_tr],  dtype=np.float64)
    X_dev = np.array([feat_fn(r, m) for r in rows_dev], dtype=np.float64)
    X_te  = np.array([feat_fn(r, m) for r in rows_te],  dtype=np.float64)

    y_tr  = np.array([r["answer"] for r in rows_tr])
    y_dev = np.array([r["answer"] for r in rows_dev])
    y_te  = np.array([r["answer"] for r in rows_te])

    model.fit(X_tr, y_tr)
    tr_acc  = model.score(X_tr,  y_tr)
    dev_acc = model.score(X_dev, y_dev)
    te_acc  = model.score(X_te,  y_te)
    gap     = dev_acc - te_acc

    print(
        f"  {enc_label:<16} {model_label:<18} "
        f"train={tr_acc:.3f}  dev={dev_acc:.3f}  test={te_acc:.3f}  gap={gap:+.3f}"
    )
    return {
        "task": TASK, "encoding": enc_label, "model": model_label,
        "train": round(tr_acc, 4), "dev": round(dev_acc, 4),
        "test": round(te_acc, 4), "iid_ood_gap": round(gap, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=24)
    args = parser.parse_args()
    m = args.modulus

    data_dir = Path(__file__).parent / "data"
    rows_tr  = load_split(data_dir, m, "train")
    rows_dev = load_split(data_dir, m, "dev")
    rows_te  = load_split(data_dir, m, "test")

    print(f"Reachability baseline (mod-{m}, balanced 50/50)")
    print(f"  Train: {len(rows_tr)}  Dev: {len(rows_dev)}  Test: {len(rows_te)}")

    # Verify balance
    for name, rows in [("train", rows_tr), ("dev", rows_dev), ("test", rows_te)]:
        n_true = sum(1 for r in rows if r["answer"] is True)
        pct = 100 * n_true / len(rows) if rows else 0
        print(f"  {name}: {n_true}/{len(rows)} True ({pct:.1f}%)")

    print()
    print(f"{'Encoding':<16} {'Model':<18} {'Train':>7} {'Dev':>7} {'Test':>7} {'IID-OOD':>8}")
    print("-" * 65)

    configs = [
        ("float_poly", feat_float_poly),
        ("onehot_flat", feat_onehot_flat),
    ]
    models = [
        ("LR",        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")),
        ("MLP(64,64)", MLPClassifier(
            hidden_layer_sizes=(64, 64), activation="relu", alpha=0.0001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42,
        )),
    ]

    results = []
    for enc_label, feat_fn in configs:
        for model_label, model in models:
            row = run(rows_tr, rows_dev, rows_te, feat_fn, m, model, model_label, enc_label)
            results.append(row)

    # Symbolic ceiling
    print(f"  {'symbolic':<16} {'—':<18} train= 1.000  dev= 1.000  test= 1.000  gap=+0.000")

    out = Path(__file__).parent / f"reachability_results_mod{m}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
