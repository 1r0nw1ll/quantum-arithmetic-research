#!/usr/bin/env python3
"""
QA-ORBIT Sequence/Attention-Inspired Baseline — `shortest_witness`

Standard MLP on flat features cannot represent the pairwise orbit relationship
between source (b,e) and target (b_target,e_target) needed to predict step count.

This script uses cross-product one-hot features, which are equivalent to a
single bilinear attention head: each (source_token, target_token) pair gets
an explicit joint representation, capturing exact orbit identity.

Feature sets compared:
  1. float_poly  : degree-2 normalised floats  [previous best: MLP dev=0.404]
  2. onehot_flat : source + target one-hot, no cross terms
  3. bilinear    : onehot_flat + b⊗b_target + e⊗e_target  [attention-equivalent]

Architecture: wider MLP to handle the higher-dimensional bilinear input.

Usage:
    python sequence_baseline.py [--modulus 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))

TASK = "shortest_witness"
DIFFICULTIES = ["easy", "medium", "hard"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return [r for r in rows if r["task_type"] == TASK]


# ── Feature encodings ─────────────────────────────────────────────────────────

def feat_float_poly(row: Dict[str, Any]) -> np.ndarray:
    inp = row["input"]
    m = float(inp["modulus"])
    b, e = inp["b"] / m, inp["e"] / m
    bt, et = inp["b_target"] / m, inp["e_target"] / m

    def poly2(x, y):
        return [x, y, x*x, x*y, y*y]

    def wrap(a, b_):
        d = abs(a - b_)
        return min(d, 1.0 - d)

    return np.array(
        poly2(b, e) + poly2(bt, et) + [wrap(b, bt), wrap(e, et), 1.0 / m],
        dtype=np.float64,
    )


def feat_onehot_flat(row: Dict[str, Any], m: int) -> np.ndarray:
    """Source + target, each one-hot — no cross terms."""
    inp = row["input"]
    b, e   = int(inp["b"]),        int(inp["e"])
    bt, et = int(inp["b_target"]), int(inp["e_target"])
    vec = np.zeros(4 * m + 1, dtype=np.float64)
    vec[b]          = 1.0
    vec[m + e]      = 1.0
    vec[2*m + bt]   = 1.0
    vec[3*m + et]   = 1.0
    vec[-1] = float(m)
    return vec


def feat_bilinear(row: Dict[str, Any], m: int) -> np.ndarray:
    """
    One-hot flat + b⊗b_target + e⊗e_target outer products.

    The outer product b⊗b_target is a m×m matrix (flattened), where entry
    [i,j]=1 iff b==i AND b_target==j. This is the exact pairwise identity
    captured by a dot-product attention head with one-hot queries/keys.
    """
    inp = row["input"]
    b, e   = int(inp["b"]),        int(inp["e"])
    bt, et = int(inp["b_target"]), int(inp["e_target"])

    # one-hot base (4m+1 dims)
    base = np.zeros(4 * m + 1, dtype=np.float64)
    base[b]          = 1.0
    base[m + e]      = 1.0
    base[2*m + bt]   = 1.0
    base[3*m + et]   = 1.0
    base[-1] = float(m)

    # outer products: b⊗b_target (m²), e⊗e_target (m²)
    bb_outer = np.zeros(m * m, dtype=np.float64)
    ee_outer = np.zeros(m * m, dtype=np.float64)
    bb_outer[b * m + bt] = 1.0
    ee_outer[e * m + et] = 1.0

    return np.concatenate([base, bb_outer, ee_outer])


def build_xy(rows, feat_fn, **kw) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([feat_fn(r, **kw) for r in rows])
    y = np.array([str(r["answer"]) for r in rows])
    return X, y


# ── Model ─────────────────────────────────────────────────────────────────────

def make_mlp(hidden: tuple) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=1e-3,
        )),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def acc(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(yt == yp)) if len(yt) > 0 else float("nan")


def acc_by_diff(rows: List[Dict[str, Any]], y_pred_str: np.ndarray) -> Dict[str, float]:
    result = {}
    for d in DIFFICULTIES:
        idx = [i for i, r in enumerate(rows) if r.get("difficulty") == d]
        if not idx:
            result[d] = float("nan")
            continue
        yt = np.array([str(rows[i]["answer"]) for i in idx])
        yp = y_pred_str[idx]
        result[d] = acc(yt, yp)
    return result


def run_encoding(name: str, hidden: tuple, feat_fn, feat_kw: dict,
                 train_rows, dev_rows, test_rows,
                 le: LabelEncoder) -> Dict[str, Any]:
    X_tr, y_tr = build_xy(train_rows, feat_fn, **feat_kw)
    X_dv, y_dv = build_xy(dev_rows,   feat_fn, **feat_kw)
    X_ts, y_ts = build_xy(test_rows,  feat_fn, **feat_kw)

    y_tr_e = le.transform(y_tr)
    y_dv_e = le.transform(y_dv)
    y_ts_e = le.transform(y_ts)

    model = make_mlp(hidden)
    model.fit(X_tr, y_tr_e)

    pred_tr = model.predict(X_tr)
    pred_dv = model.predict(X_dv)
    pred_ts = model.predict(X_ts)

    acc_tr = acc(y_tr_e, pred_tr)
    acc_dv = acc(y_dv_e, pred_dv)
    acc_ts = acc(y_ts_e, pred_ts)
    gap    = acc_dv - acc_ts

    pred_ts_str = le.inverse_transform(pred_ts)
    diff_ts = acc_by_diff(test_rows, pred_ts_str)

    print(f"\n  [{name}]  hidden={hidden}  input_dim={X_tr.shape[1]}")
    print(f"    train={acc_tr:.3f}  dev={acc_dv:.3f}  test={acc_ts:.3f}  gap={gap:+.3f}")
    print(f"    difficulty (test):  ", end="")
    for d in DIFFICULTIES:
        v = diff_ts[d]
        s = f"{v:.3f}" if v == v else "  —  "
        print(f"{d}={s}  ", end="")
    print()

    return {
        "name": name, "hidden": list(hidden), "input_dim": int(X_tr.shape[1]),
        "acc_train": acc_tr, "acc_dev": acc_dv, "acc_test": acc_ts,
        "iid_ood_gap": gap, "diff_test": diff_ts,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=24)
    args = parser.parse_args()
    m = args.modulus

    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")

    def load(split):
        return load_jsonl(os.path.join(data_dir, f"QA_SYNTHETIC_mod{m}_{split}.jsonl"))

    train_rows = load("train")
    dev_rows   = load("dev")
    test_rows  = load("test")

    print(f"\n{'='*68}")
    print(f"  QA-ORBIT Sequence Baseline — shortest_witness — mod-{m}")
    print(f"  train={len(train_rows)}  dev={len(dev_rows)}  test={len(test_rows)}")
    print(f"{'='*68}")

    # Shared label encoder across all encodings
    all_answers = (
        [str(r["answer"]) for r in train_rows] +
        [str(r["answer"]) for r in dev_rows] +
        [str(r["answer"]) for r in test_rows]
    )
    le = LabelEncoder()
    le.fit(all_answers)
    print(f"\n  Output classes: {len(le.classes_)}  "
          f"({le.classes_[0]} .. {le.classes_[-1]})")

    configs = [
        ("float_poly",  (64, 64),   feat_float_poly,  {}),
        ("onehot_flat", (128, 64),  feat_onehot_flat, {"m": m}),
        ("bilinear",    (256, 128), feat_bilinear,    {"m": m}),
    ]

    results = []
    for name, hidden, feat_fn, feat_kw in configs:
        r = run_encoding(name, hidden, feat_fn, feat_kw,
                         train_rows, dev_rows, test_rows, le)
        results.append(r)

    # Summary table
    print(f"\n{'─'*68}")
    print(f"  Summary — shortest_witness mod-{m}")
    print(f"{'─'*68}")
    print(f"  {'Encoding':<14} {'Train':>7} {'Dev':>7} {'Test':>7} {'Gap':>7}")
    print(f"  {'─'*52}")
    for r in results:
        print(f"  {r['name']:<14} {r['acc_train']:>7.3f} {r['acc_dev']:>7.3f} "
              f"{r['acc_test']:>7.3f} {r['iid_ood_gap']:>+7.3f}")
    print(f"  {'symbolic':<14} {'—':>7} {'1.000':>7} {'1.000':>7} {'0.000':>7}")
    print(f"{'─'*68}")

    # Interpretation
    bilinear_dv  = results[2]["acc_dev"]
    bilinear_ts  = results[2]["acc_test"]
    prev_dv      = results[0]["acc_dev"]  # float_poly
    improvement  = bilinear_dv - prev_dv
    print(f"\n  Bilinear vs float_poly: dev {prev_dv:.3f} → {bilinear_dv:.3f} "
          f"({improvement:+.3f})")
    if improvement > 0.05:
        print("  → Pairwise orbit identity provides meaningful signal over flat features.")
    elif improvement > 0.0:
        print("  → Marginal improvement; most signal already captured by float features.")
    else:
        print("  → No improvement; bilinear cross terms do not help this encoding.")

    out = {
        "modulus": m,
        "n_train": len(train_rows), "n_dev": len(dev_rows), "n_test": len(test_rows),
        "results": results,
    }
    out_path = os.path.join(base, f"sequence_baseline_results_mod{m}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {out_path}\n")


if __name__ == "__main__":
    main()
