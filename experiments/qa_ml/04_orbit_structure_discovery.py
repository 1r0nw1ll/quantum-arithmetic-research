"""QA-ML v3 pilot — orbit structure discovery (T1 failure-mode + T3 distillation).

Pilot scope per QA_ML_V3_STRUCTURE_DISCOVERY_PLAN.md:

  - Subset of the planned M_train (10 moduli covering all 3 failure regimes)
  - Subset of the planned M_test (5 moduli incl. m=8 boundary, m=75 anomaly)
  - T1 only (3-class shortcut failure mode)
  - Baselines: qa_full_logreg, decision tree (which doubles as T3 distillation
    target since CART is already a symbolic model)
  - T3 rediscovery scoring against [277] and [278]
  - No GCN in pilot (v3.1 if pilot is promising)
  - No T2 in pilot (v3.1)

Success criterion (per the plan):
  T3 rediscover ≥ 0.95 against [277] undercount cluster AND ≥ 0.95 against
  [278] overclaim cluster on the held-out test set => "strong rediscovery"
  signal, proceed to full v3. Below that, report honestly and decide whether
  more features / a non-linear model / a richer extraction method are needed.

QA_COMPLIANCE = "qa_ml_v3_pilot — sklearn observer; T1/T3 only; no cert promotion"
"""

from __future__ import annotations

BENCHMARK_PROTOCOL_REF = "benchmark_protocol_v3.json"

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from tools.qa_ml.qa_dataset_v3 import (  # noqa: E402
    build_v3_dataset,
    T1_CLASSES,
)
from tools.qa_ml.qa_features_v3 import FEATURE_NAMES_V3  # noqa: E402
from qa_reproducibility import log_run  # noqa: E402


M_TRAIN = [9, 10, 12, 15, 18, 20, 21, 24, 25, 30]
M_TEST = [7, 8, 11, 30, 45, 75]
SEED = 0
PROTOCOL_PATH = Path(__file__).parent / "benchmark_protocol_v3.json"
OUT_PATH = Path(__file__).parent / "results_v3_structure_discovery.json"


def _rediscovery_score(predictions, ground_truth):
    """Fraction of states where the model agrees with the cert's classification."""
    if len(predictions) == 0:
        return float("nan")
    matches = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return matches / len(predictions)


def evaluate_predictor(name, model, X_test, y_test, triples_test, ground_truth_per_state):
    """Compute T1 accuracy plus rediscovery scores vs [277] and [278]."""
    y_pred = model.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Per-class recall
    cls_recall = {}
    for cls_idx, cls_name in enumerate(T1_CLASSES):
        mask = np.asarray(y_test) == cls_idx
        if mask.sum() == 0:
            cls_recall[cls_name] = float("nan")
        else:
            cls_recall[cls_name] = float((y_pred[mask] == cls_idx).mean())

    # Rediscovery against [277]: held-out states where ground truth = undercount.
    # The cert says these should be exactly the (gcd-signature) missed pairs at
    # m = 15k. We measure whether the model's class-1 predictions overlap the
    # true class-1 set on the test moduli that meet [277]'s scope.
    cert_277_mask = []  # m = 15k AND ground truth == 1
    cert_278_mask = []  # 3 not divides m AND m >= 7 AND m != 8 AND ground truth == 2
    for (b, e, m), gt in zip(triples_test, y_test):
        cert_277_mask.append(m % 15 == 0 and gt == 1)
        cert_278_mask.append((m % 3 != 0) and (m >= 7) and (m != 8) and gt == 2)

    cert_277_mask = np.asarray(cert_277_mask)
    cert_278_mask = np.asarray(cert_278_mask)

    if cert_277_mask.sum() > 0:
        rediscover_277 = float((y_pred[cert_277_mask] == 1).mean())
    else:
        rediscover_277 = float("nan")
    if cert_278_mask.sum() > 0:
        rediscover_278 = float((y_pred[cert_278_mask] == 2).mean())
    else:
        rediscover_278 = float("nan")

    # m=8 boundary handling: at m=8 the canonical satellite count is 0 but the
    # shortcut over-claims 15. The model should predict class 2 (overclaim) for
    # the 15 false-satellite pairs at m=8 if it has learned the over-claim rule.
    m8_mask = np.asarray([m == 8 and gt == 2 for (b, e, m), gt in zip(triples_test, y_test)])
    if m8_mask.sum() > 0:
        m8_recall = float((y_pred[m8_mask] == 2).mean())
    else:
        m8_recall = float("nan")

    # m=75 boundary: 32 undercount pairs (m=15·5 is in [277] scope).
    m75_mask = np.asarray([m == 75 and gt == 1 for (b, e, m), gt in zip(triples_test, y_test)])
    if m75_mask.sum() > 0:
        m75_recall = float((y_pred[m75_mask] == 1).mean())
    else:
        m75_recall = float("nan")

    return {
        "model": name,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "per_class_recall": cls_recall,
        "rediscover_277_undercount": rediscover_277,
        "rediscover_278_overclaim": rediscover_278,
        "m8_overclaim_recall": m8_recall,
        "m75_undercount_recall": m75_recall,
        "n_test": len(y_test),
        "test_class_counts": dict(Counter(int(y) for y in y_test)),
        "pred_class_counts": dict(Counter(int(y) for y in y_pred)),
    }


def main() -> None:
    print(f"QA-ML v3 pilot — T1 failure-mode prediction + T3 distillation")
    print(f"  M_train = {M_TRAIN}  ({sum(m*m for m in M_TRAIN)} states)")
    print(f"  M_test  = {M_TEST}   ({sum(m*m for m in M_TEST)} states)\n")

    t0 = time.time()
    ds_train = build_v3_dataset(M_TRAIN)
    ds_test = build_v3_dataset(M_TEST)
    t_build = time.time() - t0
    print(f"  built train+test in {t_build:.1f}s")
    print(f"  train T1 class counts: {ds_train['class_counts_t1']}")
    print(f"  test  T1 class counts: {ds_test['class_counts_t1']}\n")

    X_train = np.asarray(ds_train["X"], dtype=np.int64)
    y_train = np.asarray(ds_train["y_t1"], dtype=np.int64)
    X_test = np.asarray(ds_test["X"], dtype=np.int64)
    y_test = np.asarray(ds_test["y_t1"], dtype=np.int64)
    triples_test = ds_test["triples"]

    # ---- Baseline 1: LogisticRegression on qa_full_v3 features ----
    print("Training qa_full_logreg ...")
    t0 = time.time()
    logreg = LogisticRegression(
        max_iter=4000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    logreg.fit(X_train, y_train)
    eval_logreg = evaluate_predictor(
        "qa_full_logreg", logreg, X_test, y_test, triples_test, None,
    )
    print(f"  trained in {time.time() - t0:.1f}s")

    # ---- Baseline 2: DecisionTreeClassifier (also serves as T3 distillation) ----
    print("Training decision_tree (T3 distillation candidate) ...")
    t0 = time.time()
    tree = DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=SEED,
    )
    tree.fit(X_train, y_train)
    eval_tree = evaluate_predictor(
        "decision_tree", tree, X_test, y_test, triples_test, None,
    )
    print(f"  trained in {time.time() - t0:.1f}s")

    # ---- T3: dump the tree's structure for inspection ----
    tree_text = export_text(tree, feature_names=list(FEATURE_NAMES_V3), max_depth=10)
    tree_path = Path(__file__).parent / "results_v3_decision_tree.txt"
    tree_path.write_text(tree_text, encoding="utf-8")
    n_nodes = tree.tree_.node_count
    n_leaves = sum(1 for is_leaf in (tree.tree_.children_left == -1) if is_leaf)
    print(f"  tree depth={tree.get_depth()} nodes={n_nodes} leaves={n_leaves}")

    # Inspect feature importances
    importances = list(zip(FEATURE_NAMES_V3, tree.feature_importances_))
    importances.sort(key=lambda kv: kv[1], reverse=True)
    print("  top 8 tree feature importances:")
    for feat, imp in importances[:8]:
        if imp > 0:
            print(f"    {feat:>15}  {imp:.3f}")

    # ---- Report ----
    print("\nResults summary:")
    print(f"{'model':>20} {'macro_f1':>9} {'bal_acc':>8} "
          f"{'rediscover_277':>15} {'rediscover_278':>15} {'m8_over':>8} {'m75_under':>10}")
    for r in [eval_logreg, eval_tree]:
        print(
            f"{r['model']:>20} "
            f"{r['macro_f1']:>9.3f} "
            f"{r['balanced_acc']:>8.3f} "
            f"{r['rediscover_277_undercount']:>15.3f} "
            f"{r['rediscover_278_overclaim']:>15.3f} "
            f"{r['m8_overclaim_recall']:>8.3f} "
            f"{r['m75_undercount_recall']:>10.3f}"
        )

    print("\nPer-class recall (T1):")
    for r in [eval_logreg, eval_tree]:
        print(f"  {r['model']:>20}: {r['per_class_recall']}")

    elapsed = time.time() - t0
    results = {
        "m_train": M_TRAIN,
        "m_test": M_TEST,
        "seed": SEED,
        "n_train_states": int(len(X_train)),
        "n_test_states": int(len(X_test)),
        "feature_names": list(FEATURE_NAMES_V3),
        "models": [eval_logreg, eval_tree],
        "decision_tree_summary": {
            "depth": int(tree.get_depth()),
            "node_count": int(n_nodes),
            "leaf_count": int(n_leaves),
            "top_importances": importances[:8],
        },
        "decision_tree_path": str(tree_path.relative_to(PROTOCOL_PATH.parent)),
    }
    OUT_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    if PROTOCOL_PATH.exists():
        log_run(
            PROTOCOL_PATH, status="completed",
            results={
                "elapsed_s": elapsed,
                "best_model_by_macro_f1": max([eval_logreg, eval_tree],
                                              key=lambda r: r["macro_f1"])["model"],
                "rediscover_277": eval_tree["rediscover_277_undercount"],
                "rediscover_278": eval_tree["rediscover_278_overclaim"],
                "decision_tree_depth": int(tree.get_depth()),
            },
        )
    else:
        print(f"  (skipped reproducibility ledger — no protocol at {PROTOCOL_PATH})")


if __name__ == "__main__":
    main()
