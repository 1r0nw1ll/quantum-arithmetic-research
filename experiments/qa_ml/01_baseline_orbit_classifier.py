"""Sample-efficiency benchmark: do QA-expanded features beat raw (b,e) for orbit prediction?

Setup
-----
  mod-24 full grid minus the unique singularity (24,24): 575 (b,e) pairs,
  567 cosmos / 8 satellite. Singularity is a single-point class, not learnable
  from data and incompatible with StratifiedShuffleSplit, so it is excluded.

  Binary classification using qa_orbit_rules.orbit_family ground truth.
  Class label: 0 = cosmos, 1 = satellite.

  Three feature sets:
    raw     : (b, e)                                — no QA structure
    qa      : (b, e, d, a, C, F, G)                 — QA algebraic packet
              with d=b+e, a=b+2e, C=2de, F=ab, G=d*d+e*e
    qa_full : (b, e, d, a, C, F, G, phi_b, phi_e)   — QA packet + modular phase
              phi_b = b mod (m//3), phi_e = e mod (m//3); m//3 is the satellite divisor

Models
------
  LogisticRegression (lbfgs, class_weight=balanced)
  RandomForestClassifier (class_weight=balanced)

Sweep train counts; multiple stratified shuffle splits per count; report macro F1
plus balanced accuracy.

Metric
------
  Macro F1 (handles class imbalance — counts each class equally).

Theorem NT
----------
  Features cross the boundary INTO the observer (sklearn). Predictions cross
  back OUT to orbit-class labels. The model is float-side; the QA layer is
  int-side; no float feedback into QA state.

QA_COMPLIANCE = "qa_ml_orbit_benchmark — observer-side sklearn classifier on
QA structural features computed by tools.qa_ml (raw int arithmetic)"
"""

from __future__ import annotations

BENCHMARK_PROTOCOL_REF = "benchmark_protocol.json"

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tools.qa_ml import build_dataset  # noqa: E402
from qa_reproducibility import log_run  # noqa: E402

MODULUS = 24
PROTOCOL_PATH = Path(__file__).parent / "benchmark_protocol.json"
N_TRAIN_LIST = [20, 40, 80, 160, 320]
N_SEEDS = 30
OUT_PATH = Path(__file__).parent / "results_orbit_baseline.json"


def _build_models(seed: int) -> dict:
    return {
        "logreg": LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    }


def feature_set_swap(
    x_raw: np.ndarray, x_qa: np.ndarray, x_qa_full: np.ndarray, feat_name: str,
) -> np.ndarray:
    """Ablation callable declared in benchmark_protocol.json.

    Swaps between three feature sets to isolate the contribution of (a) the QA
    algebraic columns (d, a, C, F, G) and (b) the modular-phase columns
    (phi_b, phi_e) on top of the raw (b, e) baseline.
    """
    if feat_name == "raw":
        return x_raw
    if feat_name == "qa":
        return x_qa
    if feat_name == "qa_full":
        return x_qa_full
    raise ValueError(f"unknown feat_name {feat_name!r}; expected 'raw' / 'qa' / 'qa_full'")


def evaluate(
    x_raw: np.ndarray, x_qa: np.ndarray, x_qa_full: np.ndarray, y: np.ndarray,
    n_train: int, seed: int,
) -> list[dict]:
    n_total = len(y)
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_train, random_state=seed,
    )
    train_idx, test_idx = next(sss.split(np.zeros((n_total, 1)), y))

    rows = []
    for feat_name in ("raw", "qa", "qa_full"):
        X = feature_set_swap(x_raw, x_qa, x_qa_full, feat_name)
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        for model_name, model in _build_models(seed).items():
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            rows.append({
                "n_train": n_train,
                "seed": seed,
                "feat": feat_name,
                "model": model_name,
                "macro_f1": float(f1_score(
                    yte, preds, average="macro", zero_division=0,
                )),
                "balanced_acc": float(balanced_accuracy_score(yte, preds)),
            })
    return rows


def main() -> None:
    print(f"Loading mod-{MODULUS} dataset ...")
    x_raw_l, x_qa_l, x_qa_full_l, y_l, _ = build_dataset(MODULUS)
    x_raw = np.asarray(x_raw_l, dtype=np.int64)
    x_qa = np.asarray(x_qa_l, dtype=np.int64)
    x_qa_full = np.asarray(x_qa_full_l, dtype=np.int64)
    y = np.asarray(y_l, dtype=np.int64)

    # Drop singularity (label 0 in ORBIT_LABELS = ('singularity','satellite','cosmos')).
    # Re-map: cosmos -> 0, satellite -> 1.
    keep = y != 0
    x_raw = x_raw[keep]
    x_qa = x_qa[keep]
    x_qa_full = x_qa_full[keep]
    y = y[keep]
    y_binary = np.where(y == 2, 0, 1).astype(np.int64)  # cosmos -> 0, satellite -> 1
    y = y_binary
    class_counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    print(f"  N={len(y)}, class counts (cosmos=0, satellite=1): {class_counts}")

    # StratifiedShuffleSplit needs at least 1 of each class to keep — singularity
    # has only 1 sample, so it will land in train or test depending on seed.
    t0 = time.time()
    results: list[dict] = []
    for n_train in N_TRAIN_LIST:
        for seed in range(N_SEEDS):
            try:
                results.extend(evaluate(x_raw, x_qa, x_qa_full, y, n_train, seed))
            except ValueError as exc:
                print(f"  skip n_train={n_train} seed={seed}: {exc}")
    elapsed = time.time() - t0
    print(f"  trained {len(results)} models in {elapsed:.1f}s")

    # Aggregate
    print("\nResults (mean macro_f1 ± std, n_seeds={n}):".format(n=N_SEEDS))
    header = f"{'n_train':>8} {'feat':>4} {'model':>7}  {'mean':>6}  {'std':>6}"
    print(header)
    print("-" * len(header))

    summary: dict[tuple, dict[str, float]] = {}
    for n_train in N_TRAIN_LIST:
        for feat in ("raw", "qa", "qa_full"):
            for model_name in ("logreg", "rf"):
                scores = [
                    r["macro_f1"] for r in results
                    if r["n_train"] == n_train and r["feat"] == feat
                    and r["model"] == model_name
                ]
                if not scores:
                    continue
                mean, std = float(np.mean(scores)), float(np.std(scores))
                summary[(n_train, feat, model_name)] = {
                    "macro_f1_mean": mean, "macro_f1_std": std,
                }
                print(f"{n_train:>8} {feat:>4} {model_name:>7}  {mean:>6.3f}  {std:>6.3f}")

    print("\nDeltas vs raw baseline (mean macro_f1 - raw_mean):")
    print(f"{'n_train':>8} {'model':>7}  {'qa_delta':>+10}  {'qa_full_delta':>+13}")
    for n_train in N_TRAIN_LIST:
        for model_name in ("logreg", "rf"):
            raw_key = (n_train, "raw", model_name)
            qa_key = (n_train, "qa", model_name)
            full_key = (n_train, "qa_full", model_name)
            if all(k in summary for k in (raw_key, qa_key, full_key)):
                d_qa = summary[qa_key]["macro_f1_mean"] - summary[raw_key]["macro_f1_mean"]
                d_full = summary[full_key]["macro_f1_mean"] - summary[raw_key]["macro_f1_mean"]
                print(f"{n_train:>8} {model_name:>7}  {d_qa:>+10.3f}  {d_full:>+13.3f}")

    OUT_PATH.write_text(json.dumps({
        "modulus": MODULUS,
        "n_train_list": N_TRAIN_LIST,
        "n_seeds": N_SEEDS,
        "class_counts": class_counts,
        "raw_results": results,
        "summary": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary.items()},
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")

    log_run(
        PROTOCOL_PATH,
        status="completed",
        results={
            "elapsed_s": elapsed,
            "n_models_trained": len(results),
            "summary": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary.items()},
            "results_json": str(OUT_PATH.relative_to(PROTOCOL_PATH.parent)),
        },
    )


if __name__ == "__main__":
    main()
