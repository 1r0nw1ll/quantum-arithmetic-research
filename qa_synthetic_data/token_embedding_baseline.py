#!/usr/bin/env python3
"""
QA-ORBIT Token Embedding Baseline — `invariant_pred`

Isolates whether the 0% numeric-baseline result is due to:
  (A) Representation failure — floats can't express integer/modular structure
  (B) Task difficulty — the algebraic rule is genuinely hard to learn

Three input encodings tested, all on invariant_pred only:

  1. float_norm    : (b/m, e/m, 1/m)  + degree-2 poly  [previous baseline]
  2. integer       : (b, e, m) as raw integers, no normalisation
  3. onehot        : b one-hot (m dims), e one-hot (m dims), m as scalar
                     Passing one-hot through a linear layer is equivalent
                     to a learned embedding per integer value.

Architecture (for encodings 2 and 3): sklearn MLPClassifier with hidden
layers matched to ChatGPT's suggested design (64×64).

Usage:
    python token_embedding_baseline.py [--modulus 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(__file__))

TASK = "invariant_pred"
DIFFICULTIES = ["easy", "medium", "hard"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if r["task_type"] == TASK]


# ── Feature encodings ─────────────────────────────────────────────────────────

def feat_float_poly(row: Dict[str, Any]) -> np.ndarray:
    """Previous baseline: degree-2 polynomial of normalised floats."""
    inp = row["input"]
    m = float(inp["modulus"])
    b, e = inp["b"] / m, inp["e"] / m
    return np.array([b, e, b*b, b*e, e*e, 1.0/m], dtype=np.float64)


def feat_integer(row: Dict[str, Any]) -> np.ndarray:
    """Raw integers — preserves modular structure, no normalisation."""
    inp = row["input"]
    return np.array([float(inp["b"]), float(inp["e"]), float(inp["modulus"])],
                    dtype=np.float64)


def feat_onehot(row: Dict[str, Any], m_max: int) -> np.ndarray:
    """
    One-hot encode b and e independently (m_max dims each).
    Passing one-hot through a linear layer == learned embedding per integer.
    This exactly matches the token-embedding architecture ChatGPT described.
    """
    inp = row["input"]
    b, e, m = int(inp["b"]), int(inp["e"]), int(inp["modulus"])
    vec = np.zeros(m_max * 2 + 1, dtype=np.float64)
    vec[b] = 1.0
    vec[m_max + e] = 1.0
    vec[-1] = float(m)  # modulus as scalar (constant within one modulus run)
    return vec


def build_xy(rows: List[Dict[str, Any]], feat_fn, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([feat_fn(r, **kwargs) for r in rows])
    y = np.array([str(r["answer"]) for r in rows])
    return X, y


# ── Models ────────────────────────────────────────────────────────────────────

def make_mlp(hidden=(64, 64)) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            learning_rate_init=1e-3,
        )),
    ])


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                   random_state=42)),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def acc(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(yt == yp)) if len(yt) > 0 else float("nan")


def evaluate_encoding(
    name: str,
    train_rows, dev_rows, test_rows,
    feat_fn, feat_kwargs: dict,
) -> Dict[str, Any]:
    X_tr, y_tr = build_xy(train_rows, feat_fn, **feat_kwargs)
    X_dv, y_dv = build_xy(dev_rows,   feat_fn, **feat_kwargs)
    X_ts, y_ts = build_xy(test_rows,  feat_fn, **feat_kwargs)

    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_dv, y_ts]))
    y_tr_e = le.transform(y_tr)
    y_dv_e = le.transform(y_dv)
    y_ts_e = le.transform(y_ts)

    results = {}
    for model_name, model in [("LogReg", make_lr()), ("MLP(64×64)", make_mlp())]:
        model.fit(X_tr, y_tr_e)
        acc_tr = acc(y_tr_e, model.predict(X_tr))
        acc_dv = acc(y_dv_e, model.predict(X_dv))
        acc_ts = acc(y_ts_e, model.predict(X_ts))
        results[model_name] = {
            "train": acc_tr, "dev": acc_dv, "test": acc_ts,
            "gap": acc_dv - acc_ts,
        }
        tag = "(← SOLVED IID)" if acc_dv > 0.5 else ""
        print(f"    [{name}] {model_name:<14}  "
              f"train={acc_tr:.3f}  dev={acc_dv:.3f}  test={acc_ts:.3f}  "
              f"gap={acc_dv-acc_ts:+.3f}  {tag}")

    return results


# ── Symbolic ceiling check ────────────────────────────────────────────────────

def symbolic_check(rows: List[Dict[str, Any]]) -> float:
    """Verify deterministic verifier achieves 100%."""
    from core import qa_norm
    correct = 0
    for r in rows:
        inp = r["input"]
        pred = qa_norm(inp["b"], inp["e"], inp["modulus"])
        if pred == r["answer"]:
            correct += 1
    return correct / len(rows) if rows else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=24)
    args = parser.parse_args()
    m = args.modulus

    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")

    def load(split: str) -> List[Dict[str, Any]]:
        return load_jsonl(os.path.join(data_dir, f"QA_SYNTHETIC_mod{m}_{split}.jsonl"))

    train_rows = load("train")
    dev_rows   = load("dev")
    test_rows  = load("test")

    print(f"\n{'='*68}")
    print(f"  QA-ORBIT Token Embedding Baseline — invariant_pred — mod-{m}")
    print(f"  train={len(train_rows)}  dev={len(dev_rows)}  test={len(test_rows)}")
    print(f"{'='*68}")

    # Confirm symbolic ceiling
    ceil = symbolic_check(test_rows)
    print(f"\n  Symbolic ceiling (test): {ceil:.3f}")

    print(f"\n  ── Encoding comparison ──")
    print(f"  Legend: gap = dev − test  |  Hypothesis A = encoding fixes 0%  "
          f"|  Hypothesis B = task genuinely hard\n")

    encodings = [
        ("float_poly",  feat_float_poly, {}),
        ("integer",     feat_integer,    {}),
        ("onehot",      feat_onehot,     {"m_max": m}),
    ]

    all_results: Dict[str, Any] = {}
    for enc_name, feat_fn, feat_kwargs in encodings:
        res = evaluate_encoding(
            enc_name, train_rows, dev_rows, test_rows, feat_fn, feat_kwargs
        )
        all_results[enc_name] = res

    # Verdict
    print(f"\n  ── Verdict ──")
    onehot_mlp_dev  = all_results["onehot"]["MLP(64×64)"]["dev"]
    onehot_mlp_test = all_results["onehot"]["MLP(64×64)"]["test"]

    if onehot_mlp_dev > 0.5:
        verdict = (
            "HYPOTHESIS A — Representation failure.\n"
            f"  One-hot MLP dev={onehot_mlp_dev:.3f}, test={onehot_mlp_test:.3f}.\n"
            "  The float encoding was the bottleneck; token-level encoding solves the IID task.\n"
            "  The IID/OOD gap then measures algebraic generalisation across orbit families."
        )
    elif onehot_mlp_dev > 0.15:
        verdict = (
            "MIXED — Partial representation gain.\n"
            f"  One-hot MLP dev={onehot_mlp_dev:.3f} vs float 0.000.\n"
            "  Encoding helps but the task remains substantially unsolved."
        )
    else:
        verdict = (
            "HYPOTHESIS B — Task genuinely hard.\n"
            f"  One-hot MLP dev={onehot_mlp_dev:.3f}, test={onehot_mlp_test:.3f}.\n"
            "  Token-level encoding does not rescue invariant_pred.\n"
            "  Conclusion: learning the QA norm is algorithmically hard for standard MLP/LR baselines\n"
            "  regardless of encoding. This strengthens the benchmark claim."
        )
    print(f"  {verdict}")

    # Save
    out = {
        "modulus": m,
        "n_train": len(train_rows), "n_dev": len(dev_rows), "n_test": len(test_rows),
        "encodings": all_results,
        "verdict": verdict.split("\n")[0],
    }
    out_path = os.path.join(base, f"token_baseline_results_mod{m}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {out_path}\n")


if __name__ == "__main__":
    main()
