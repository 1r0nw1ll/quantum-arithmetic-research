#!/usr/bin/env python3
"""
QA-ORBIT ML Baseline Suite

Trains and evaluates:
  1. LogisticRegression   (linear baseline)
  2. MLPClassifier        (nonlinear baseline)

Per-task-type models on flattened numeric inputs.

Reports:
  - train / dev / test accuracy per model × task type
  - per-difficulty breakdown (easy / medium / hard)
  - IID (dev) vs OOD (test) gap per task type

Outputs:
  - printed summary table
  - iid_ood_gap.png  (grouped bar chart)
  - results.json     (machine-readable)

Usage:
    python train_ml_baselines.py [--modulus 24] [--out-dir .]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

TASK_TYPES = ["invariant_pred", "orbit_class", "reachability", "shortest_witness"]
DIFFICULTIES = ["easy", "medium", "hard"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Feature engineering ───────────────────────────────────────────────────────

def _poly2(x: float, y: float) -> List[float]:
    """Degree-2 polynomial expansion: x, y, x², xy, y²."""
    return [x, y, x * x, x * y, y * y]


def _wrap_dist(a: float, b: float) -> float:
    """Circular distance on [0,1)."""
    d = abs(a - b)
    return min(d, 1.0 - d)


def extract_features(row: Dict[str, Any]) -> np.ndarray:
    """Polynomial (degree-2) features normalised by modulus."""
    inp = row["input"]
    m = float(inp["modulus"])
    b = inp["b"] / m
    e = inp["e"] / m
    feats: List[float] = _poly2(b, e) + [1.0 / m]

    if "b_target" in inp:
        bt = inp["b_target"] / m
        et = inp["e_target"] / m
        feats += _poly2(bt, et)
        # wrapped circular distances (orbit geometry signal)
        feats += [_wrap_dist(b, bt), _wrap_dist(e, et)]
    else:
        feats += [0.0] * 7  # pad to uniform width

    return np.array(feats, dtype=np.float64)


def encode_answer(answer: Any) -> str:
    """Stringify answer for LabelEncoder compatibility."""
    return str(answer)


def build_xy(rows: List[Dict[str, Any]], tt: str) -> Tuple[np.ndarray, np.ndarray]:
    subset = [r for r in rows if r["task_type"] == tt]
    X = np.array([extract_features(r) for r in subset])
    y = np.array([encode_answer(r["answer"]) for r in subset])
    return X, y, subset


# ── Model factory ─────────────────────────────────────────────────────────────

def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                   solver="lbfgs", random_state=42)),
    ])


def make_mlp() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])


MODELS = {
    "LogisticRegression": make_lr,
    "MLP(64×64)":         make_mlp,
}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) > 0 else float("nan")


def accuracy_by_difficulty(
    rows: List[Dict[str, Any]],
    y_pred: np.ndarray,
) -> Dict[str, float]:
    result = {}
    for d in DIFFICULTIES:
        mask = [i for i, r in enumerate(rows) if r.get("difficulty") == d]
        if not mask:
            result[d] = float("nan")
            continue
        yt = np.array([encode_answer(rows[i]["answer"]) for i in mask])
        yp = y_pred[mask]
        result[d] = accuracy(yt, yp)
    return result


# ── Main training + evaluation loop ──────────────────────────────────────────

def run(modulus: int, data_dir: str) -> Dict[str, Any]:
    prefix = os.path.join(data_dir, f"QA_SYNTHETIC_mod{modulus}")
    train_rows = load_jsonl(f"{prefix}_train.jsonl")
    dev_rows   = load_jsonl(f"{prefix}_dev.jsonl")
    test_rows  = load_jsonl(f"{prefix}_test.jsonl")
    all_rows   = load_jsonl(f"{prefix}_all.jsonl")

    print(f"\n{'='*64}")
    print(f"  QA-ORBIT ML Baselines — mod-{modulus}")
    print(f"  train={len(train_rows)}  dev={len(dev_rows)}  test={len(test_rows)}")
    print(f"{'='*64}")

    results: Dict[str, Any] = {
        "modulus": modulus,
        "n_train": len(train_rows),
        "n_dev":   len(dev_rows),
        "n_test":  len(test_rows),
        "models":  {},
    }

    for model_name, model_factory in MODELS.items():
        print(f"\n  ── {model_name} ──")
        model_results: Dict[str, Any] = {"task_types": {}}

        for tt in TASK_TYPES:
            X_train, y_train, train_sub = build_xy(train_rows, tt)
            X_dev,   y_dev,   dev_sub   = build_xy(dev_rows, tt)
            X_test,  y_test,  test_sub  = build_xy(test_rows, tt)

            if len(X_train) == 0 or len(X_dev) == 0 or len(X_test) == 0:
                print(f"    {tt}: skipped (empty split)")
                continue

            # Fit
            le = LabelEncoder()
            le.fit(np.concatenate([y_train, y_dev, y_test]))
            y_train_enc = le.transform(y_train)
            y_dev_enc   = le.transform(y_dev)
            y_test_enc  = le.transform(y_test)

            model = model_factory()
            model.fit(X_train, y_train_enc)

            # Predict
            pred_train = model.predict(X_train)
            pred_dev   = model.predict(X_dev)
            pred_test  = model.predict(X_test)

            acc_train = accuracy(y_train_enc, pred_train)
            acc_dev   = accuracy(y_dev_enc,   pred_dev)
            acc_test  = accuracy(y_test_enc,  pred_test)
            gap = acc_dev - acc_test

            # Per-difficulty on test
            pred_test_str = le.inverse_transform(pred_test)
            diff_test = accuracy_by_difficulty(test_sub, pred_test_str)

            print(f"    {tt:<22}  train={acc_train:.3f}  dev={acc_dev:.3f}  "
                  f"test={acc_test:.3f}  gap={gap:+.3f}")
            for d in DIFFICULTIES:
                v = diff_test[d]
                tag = f"{v:.3f}" if not (v != v) else "  —  "
                print(f"      {d:<8} test-acc={tag}")

            model_results["task_types"][tt] = {
                "n_train":   len(X_train),
                "n_dev":     len(X_dev),
                "n_test":    len(X_test),
                "acc_train": acc_train,
                "acc_dev":   acc_dev,
                "acc_test":  acc_test,
                "iid_ood_gap": gap,
                "diff_test": diff_test,
            }

        # Overall
        tt_res = model_results["task_types"]
        if tt_res:
            avg_dev  = float(np.mean([v["acc_dev"]  for v in tt_res.values()]))
            avg_test = float(np.mean([v["acc_test"] for v in tt_res.values()]))
            avg_gap  = avg_dev - avg_test
            model_results["avg_dev"]  = avg_dev
            model_results["avg_test"] = avg_test
            model_results["avg_iid_ood_gap"] = avg_gap
            print(f"\n    {'AVERAGE':<22}  {'':>12}dev={avg_dev:.3f}  "
                  f"test={avg_test:.3f}  gap={avg_gap:+.3f}")

        results["models"][model_name] = model_results

    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(results: Dict[str, Any], out_path: str) -> None:
    models = list(results["models"].keys())
    n_models = len(models)
    n_tasks = len(TASK_TYPES)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    bar_w = 0.35
    x = np.arange(n_tasks)
    colors = {"dev": "#4C72B0", "test": "#DD8452"}

    for ax, model_name in zip(axes, models):
        tt_res = results["models"][model_name].get("task_types", {})
        dev_accs  = [tt_res.get(tt, {}).get("acc_dev",  float("nan")) for tt in TASK_TYPES]
        test_accs = [tt_res.get(tt, {}).get("acc_test", float("nan")) for tt in TASK_TYPES]

        bars_dev  = ax.bar(x - bar_w / 2, dev_accs,  bar_w,
                           label="dev (IID)",  color=colors["dev"],  alpha=0.85)
        bars_test = ax.bar(x + bar_w / 2, test_accs, bar_w,
                           label="test (OOD)", color=colors["test"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in TASK_TYPES], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_title(model_name, fontsize=11)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="symbolic ceil")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # annotate gap
        for xi, (dv, ts) in enumerate(zip(dev_accs, test_accs)):
            if dv == dv and ts == ts:  # not nan
                gap = dv - ts
                ax.text(xi, max(dv, ts) + 0.02, f"Δ{gap:+.2f}",
                        ha="center", va="bottom", fontsize=7, color="black")

    modulus = results["modulus"]
    # Primary tasks marker
    for ax in axes:
        ax.axvspan(-0.5, 0.5, alpha=0.06, color="green", label="primary task")
        ax.axvspan(2.5, 3.5, alpha=0.06, color="green")

    fig.suptitle(
        f"QA-ORBIT mod-{modulus}: IID (dev) vs OOD (test) Accuracy\n"
        f"[green shading = primary algebraic tasks; others: label-prior or underfeatured]",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Figure saved → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Any]) -> None:
    modulus = results["modulus"]
    models = list(results["models"].keys())

    header_w = 22
    col_w = 15
    print(f"\n{'─'*72}")
    print(f"  QA-ORBIT mod-{modulus} — Summary Table")
    print(f"{'─'*72}")
    header = f"  {'Task Type':<{header_w}}"
    for m in models:
        header += f"  {'dev':>5} {'test':>5} {'gap':>5}  "
    print(header)
    print(f"  {'─'*(header_w + len(models) * 22)}")

    for tt in TASK_TYPES:
        row = f"  {tt:<{header_w}}"
        for m in models:
            tt_res = results["models"][m].get("task_types", {}).get(tt, {})
            dv = tt_res.get("acc_dev",  float("nan"))
            ts = tt_res.get("acc_test", float("nan"))
            gp = dv - ts if (dv == dv and ts == ts) else float("nan")
            row += f"  {dv:5.3f} {ts:5.3f} {gp:+5.3f}  "
        print(row)

    print(f"  {'─'*(header_w + len(models) * 22)}")
    row = f"  {'AVERAGE':<{header_w}}"
    for m in models:
        dv = results["models"][m].get("avg_dev",  float("nan"))
        ts = results["models"][m].get("avg_test", float("nan"))
        gp = results["models"][m].get("avg_iid_ood_gap", float("nan"))
        row += f"  {dv:5.3f} {ts:5.3f} {gp:+5.3f}  "
    print(row)
    print(f"{'─'*72}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=24)
    parser.add_argument("--out-dir", default=None,
                        help="Directory for figure + results JSON. Default: qa_synthetic_data/")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, "data")
    out_dir = args.out_dir or base

    results = run(modulus=args.modulus, data_dir=data_dir)
    print_summary_table(results)

    fig_path = os.path.join(out_dir, f"iid_ood_gap_mod{args.modulus}.png")
    make_figure(results, out_path=fig_path)

    json_path = os.path.join(out_dir, f"ml_baseline_results_mod{args.modulus}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON → {json_path}\n")


if __name__ == "__main__":
    main()
