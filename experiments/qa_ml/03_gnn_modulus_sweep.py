"""QA-ML v2 boundary sweep — does generator-topology advantage hold across moduli?

Hypothesis under test
---------------------
  The mod-24 GCN result generalizes: on any QA orbit grid with a well-defined
  satellite divisor (m // 3, when satellites form a non-trivial period-8 class
  under qa_step), the generator-graph adjacency lifts macro F1 above the
  identity-adjacency ablation by at least +0.10 at a fixed train fraction.

Setup
-----
  For each m in MODULI:
    1. Enumerate all (b, e) in {1,...,m}^2.
    2. Classify by orbit_period (the empirical orbit length under qa_step):
       period 1 -> singularity (masked from supervision/eval)
       period 8 -> satellite   (label 1)
       else     -> cosmos      (label 0)
       Also compute the algebraic rule from qa_orbit_rules.orbit_family
       and report agreement.
    3. If satellite count < 2 (cannot stratify), skip with reason 'too_few_satellites'.
    4. Build reachability graph from sigma/mu/lambda_2/nu, symmetrize, GCN-normalize.
    5. Run four configs (raw_mlp, qa_full_logreg, gcn with_graph, gcn without_graph)
       at TRAIN_FRACTION over N_SEEDS seeds.
    6. Report macro F1 mean/std and graph_delta = with_graph - without_graph.

  PASS criterion per modulus:  graph_delta >= 0.10.

Theorem NT
----------
  Same as v2 (02_gnn_orbit_classifier.py): integer-derived adjacency and node
  features; float-side GCN observer; no float feedback into QA layer.

QA_COMPLIANCE = "qa_ml_gnn_modulus_sweep — boundary hardening for QA-ML v2 cert"
"""

from __future__ import annotations

BENCHMARK_PROTOCOL_REF = "benchmark_protocol_v2_modulus_sweep.json"

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from qa_orbit_rules import orbit_family, orbit_period  # noqa: E402
from tools.qa_ml.qa_dataset import all_pairs  # noqa: E402
from tools.qa_ml.qa_features import qa_packet_full  # noqa: E402
from tools.qa_ml.qa_graph import dense_adjacency, gcn_normalize  # noqa: E402
from qa_reproducibility import log_run  # noqa: E402

MODULI = [9, 12, 15, 18, 21, 24, 27, 30, 36]
TRAIN_FRACTION = 0.30
N_SEEDS = 20
EPOCHS = 300
HIDDEN = 32
LR = 0.01
WEIGHT_DECAY = 5e-4
PASS_THRESHOLD = 0.10
PROTOCOL_PATH = Path(__file__).parent / "benchmark_protocol_v2_modulus_sweep.json"
OUT_PATH = Path(__file__).parent / "results_gnn_modulus_sweep.json"


class GCN(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(n_features, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(a_norm @ x))
        h = F.relu(self.lin2(a_norm @ h))
        return self.head(h)


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(n_features, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        h = F.relu(self.lin2(h))
        return self.head(h)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)  # noqa: T2-D-5  observer-side; deterministic split + torch init
    torch.manual_seed(seed)


def _z_score(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu_tr = x[train_mask].mean(axis=0, keepdims=True)
    sd_tr = x[train_mask].std(axis=0, keepdims=True)
    sd_tr = np.where(sd_tr < 1e-8, 1.0, sd_tr)
    return (x - mu_tr) / sd_tr


def _class_weights(y_train: np.ndarray) -> np.ndarray:
    n = len(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    weights = n / (len(classes) * counts)
    full = np.ones(2, dtype=np.float64)
    for c, w in zip(classes, weights):
        full[int(c)] = w
    return full


def _train_torch(model: nn.Module, forward_fn, y_t: torch.Tensor,
                 train_mask: torch.Tensor, class_weight: torch.Tensor) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = forward_fn(model)
        loss = F.cross_entropy(logits[train_mask], y_t[train_mask], weight=class_weight)
        loss.backward()
        opt.step()


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def feature_set_swap(x_qa_full: np.ndarray, adj_norm: np.ndarray, ablation: str):
    """Ablation callable declared in benchmark_protocol_v2_modulus_sweep.json."""
    if ablation == "with_graph":
        return x_qa_full, adj_norm
    if ablation == "without_graph":
        return x_qa_full, np.eye(adj_norm.shape[0], dtype=adj_norm.dtype)
    raise ValueError(f"unknown ablation {ablation!r}")


def classify_modulus(m: int) -> dict:
    """Compute per-pair orbit periods, labels, and agreement with algebraic rule."""
    pairs = all_pairs(m)
    periods = np.array([orbit_period(b, e, m) for b, e in pairs], dtype=np.int64)

    labels = np.full(len(pairs), -1, dtype=np.int64)
    labels[periods != 1] = 0  # tentative cosmos
    labels[periods == 8] = 1  # satellite
    labels[periods == 1] = -1  # singularity (mask)

    algebraic = np.array(
        [orbit_family(b, e, m) for b, e in pairs], dtype=object,
    )
    alg_to_int = {"singularity": -1, "satellite": 1, "cosmos": 0}
    alg_int = np.array([alg_to_int[a] for a in algebraic], dtype=np.int64)
    agreement = int((alg_int == labels).sum())

    period_dist = {int(p): int((periods == p).sum()) for p in np.unique(periods)}
    return {
        "modulus": m,
        "n_pairs": len(pairs),
        "period_distribution": period_dist,
        "n_singularity": int((labels == -1).sum()),
        "n_satellite": int((labels == 1).sum()),
        "n_cosmos": int((labels == 0).sum()),
        "algebraic_agreement": agreement,
        "labels": labels,
        "pairs": pairs,
    }


def _qa_full_features(pairs: list[tuple[int, int]], m: int) -> np.ndarray:
    return np.asarray([qa_packet_full(b, e, m) for (b, e) in pairs], dtype=np.int64)


def _raw_features(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(pairs, dtype=np.int64)


def evaluate_modulus(m: int, info: dict) -> tuple[list[dict], dict]:
    pairs = info["pairs"]
    labels = info["labels"]
    label_mask = labels >= 0
    n_sat = info["n_satellite"]
    if n_sat < 2:
        return [], {"status": "skipped", "reason": "too_few_satellites", "n_satellite": n_sat}

    n_total = len(pairs)
    x_raw = _raw_features(pairs)
    x_qa_full = _qa_full_features(pairs, m)

    adj = dense_adjacency(m, symmetric=True)
    adj_norm = gcn_normalize(adj)

    n_labeled = int(label_mask.sum())
    n_train = max(2, int(round(TRAIN_FRACTION * n_labeled)))
    n_train = min(n_train, n_labeled - 2)  # always keep at least 2 in test

    rows: list[dict] = []
    for seed in range(N_SEEDS):
        _set_seed(seed)
        labeled_indices = np.where(label_mask)[0]
        y_labeled = labels[labeled_indices]
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
            train_pos, test_pos = next(sss.split(np.zeros((len(y_labeled), 1)), y_labeled))
        except ValueError as exc:
            rows.append({"modulus": m, "seed": seed, "skip_reason": str(exc)})
            continue
        train_idx = labeled_indices[train_pos]
        test_idx = labeled_indices[test_pos]
        train_mask_full = np.zeros(n_total, dtype=bool)
        train_mask_full[train_idx] = True

        y_full = np.where(label_mask, labels, 0).astype(np.int64)
        cw = _class_weights(y_full[train_idx])
        cw_t = torch.from_numpy(cw).float()

        x_qa_full_std = _z_score(x_qa_full.astype(np.float64), train_mask_full)
        x_raw_std = _z_score(x_raw.astype(np.float64), train_mask_full)

        # raw_mlp
        _set_seed(seed)
        mlp = MLP(n_features=x_raw_std.shape[1], hidden=HIDDEN, n_classes=2)
        x_t = torch.from_numpy(x_raw_std).float()
        y_t = torch.from_numpy(y_full).long()
        _train_torch(mlp, lambda m_: m_(x_t), y_t,
                     torch.from_numpy(train_mask_full), cw_t)
        mlp.eval()
        with torch.no_grad():
            preds = mlp(x_t).argmax(dim=1).cpu().numpy()
        rows.append({
            "modulus": m, "n_train": n_train, "seed": seed,
            "model": "raw_mlp", "ablation": "n/a",
            **_scores(y_full[test_idx], preds[test_idx]),
        })

        # qa_full_logreg
        logreg = LogisticRegression(
            max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed,
        )
        logreg.fit(x_qa_full[train_idx], y_full[train_idx])
        preds = logreg.predict(x_qa_full[test_idx])
        rows.append({
            "modulus": m, "n_train": n_train, "seed": seed,
            "model": "qa_full_logreg", "ablation": "n/a",
            **_scores(y_full[test_idx], preds),
        })

        # gcn (with/without graph)
        for ablation in ("with_graph", "without_graph"):
            feats, adj_used = feature_set_swap(x_qa_full_std, adj_norm, ablation)
            _set_seed(seed)
            gcn = GCN(n_features=feats.shape[1], hidden=HIDDEN, n_classes=2)
            x_t = torch.from_numpy(feats).float()
            a_t = torch.from_numpy(adj_used).float()
            _train_torch(gcn, lambda m_: m_(x_t, a_t), y_t,
                         torch.from_numpy(train_mask_full), cw_t)
            gcn.eval()
            with torch.no_grad():
                preds = gcn(x_t, a_t).argmax(dim=1).cpu().numpy()
            rows.append({
                "modulus": m, "n_train": n_train, "seed": seed,
                "model": "gcn_qa_full", "ablation": ablation,
                **_scores(y_full[test_idx], preds[test_idx]),
            })

    summary = _summarize_modulus(m, rows, n_train, info)
    return rows, summary


def _summarize_modulus(m: int, rows: list[dict], n_train: int, info: dict) -> dict:
    by_key: dict[str, list[float]] = {}
    for r in rows:
        if "macro_f1" not in r:
            continue
        key = r["model"] + ("|" + r["ablation"] if r["ablation"] != "n/a" else "")
        by_key.setdefault(key, []).append(r["macro_f1"])
    means = {k: float(np.mean(v)) for k, v in by_key.items()}
    stds = {k: float(np.std(v)) for k, v in by_key.items()}
    with_g = means.get("gcn_qa_full|with_graph")
    no_g = means.get("gcn_qa_full|without_graph")
    raw_m = means.get("raw_mlp")
    lr_m = means.get("qa_full_logreg")
    graph_delta = (with_g - no_g) if (with_g is not None and no_g is not None) else None
    pass_threshold = (graph_delta is not None and graph_delta >= PASS_THRESHOLD)
    return {
        "status": "ran",
        "modulus": m,
        "n_train": n_train,
        "n_pairs": info["n_pairs"],
        "n_satellite": info["n_satellite"],
        "n_cosmos": info["n_cosmos"],
        "n_singularity": info["n_singularity"],
        "algebraic_agreement": info["algebraic_agreement"],
        "period_distribution": info["period_distribution"],
        "macro_f1_mean": means,
        "macro_f1_std": stds,
        "graph_delta": graph_delta,
        "raw_mlp_mean": raw_m,
        "qa_full_logreg_mean": lr_m,
        "passes_threshold": pass_threshold,
    }


def main() -> None:
    print(f"QA-ML v2 modulus sweep — moduli={MODULI}, train_frac={TRAIN_FRACTION}, seeds={N_SEEDS}")
    print(f"PASS threshold: graph_delta >= {PASS_THRESHOLD:.2f}\n")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    skipped: list[dict] = []

    t0 = time.time()
    for m in MODULI:
        info = classify_modulus(m)
        print(f"  m={m:>2}: pairs={info['n_pairs']:>4} sat={info['n_satellite']:>2} "
              f"sing={info['n_singularity']:>1} cos={info['n_cosmos']:>4} "
              f"alg_agree={info['algebraic_agreement']}/{info['n_pairs']} "
              f"periods={info['period_distribution']}")
        if info["n_satellite"] < 2:
            skipped.append({"modulus": m, "reason": "too_few_satellites",
                            "n_satellite": info["n_satellite"]})
            continue
        rows, summary = evaluate_modulus(m, info)
        all_rows.extend(rows)
        summaries.append(summary)
        gd = summary["graph_delta"]
        verdict = "PASS" if summary["passes_threshold"] else "FAIL"
        print(f"        graph_delta={gd:+.3f}  [{verdict}]  "
              f"with_graph={summary['macro_f1_mean'].get('gcn_qa_full|with_graph', 0):.3f}  "
              f"without_graph={summary['macro_f1_mean'].get('gcn_qa_full|without_graph', 0):.3f}")

    elapsed = time.time() - t0
    print(f"\n  total elapsed: {elapsed:.1f}s, rows: {len(all_rows)}")

    # Verdict table
    print("\nFinal table — graph_delta per modulus (PASS = >= +0.10):")
    print(f"{'m':>3}  {'sat':>3} {'pairs':>5}  {'graph_delta':>12}  {'verdict':>7}  "
          f"{'with_graph':>10}  {'no_graph':>10}  {'qa_logreg':>9}  {'raw_mlp':>8}")
    for s in summaries:
        m = s["modulus"]
        verdict = "PASS" if s["passes_threshold"] else "FAIL"
        gd = s["graph_delta"]
        wg = s["macro_f1_mean"].get("gcn_qa_full|with_graph", float("nan"))
        ng = s["macro_f1_mean"].get("gcn_qa_full|without_graph", float("nan"))
        lr = s["macro_f1_mean"].get("qa_full_logreg", float("nan"))
        rm = s["macro_f1_mean"].get("raw_mlp", float("nan"))
        print(f"{m:>3}  {s['n_satellite']:>3} {s['n_pairs']:>5}  "
              f"{gd:>+12.3f}  {verdict:>7}  {wg:>10.3f}  {ng:>10.3f}  {lr:>9.3f}  {rm:>8.3f}")
    if skipped:
        print("\nSkipped moduli:")
        for sk in skipped:
            print(f"  m={sk['modulus']}: {sk['reason']} (n_satellite={sk['n_satellite']})")

    # Cert wording recommendation
    pass_moduli = [s["modulus"] for s in summaries if s["passes_threshold"]]
    fail_moduli = [s["modulus"] for s in summaries if not s["passes_threshold"]]
    print("\nCert wording recommendation:")
    if pass_moduli and not fail_moduli:
        print(f"  Strong: graph topology lifts macro_f1 by >= {PASS_THRESHOLD:.2f} for all "
              f"tested moduli with non-trivial satellite class: {pass_moduli}")
    elif pass_moduli and fail_moduli:
        print(f"  Bounded: passes for {pass_moduli}; fails for {fail_moduli} (boundary).")
    elif not pass_moduli and fail_moduli:
        print(f"  Refute: graph topology does NOT generalize beyond the training modulus; "
              f"passes for none of {fail_moduli}.")
    if skipped:
        print(f"  Excluded (no satellite class to test): {[s['modulus'] for s in skipped]}")

    OUT_PATH.write_text(json.dumps({
        "moduli": MODULI,
        "train_fraction": TRAIN_FRACTION,
        "n_seeds": N_SEEDS,
        "epochs": EPOCHS,
        "pass_threshold": PASS_THRESHOLD,
        "summaries": summaries,
        "skipped": skipped,
        "raw_results": all_rows,
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")

    log_run(
        PROTOCOL_PATH,
        status="completed",
        results={
            "elapsed_s": elapsed,
            "n_models_trained": len(all_rows),
            "pass_moduli": pass_moduli,
            "fail_moduli": fail_moduli,
            "skipped_moduli": [s["modulus"] for s in skipped],
            "summaries": summaries,
            "results_json": str(OUT_PATH.relative_to(PROTOCOL_PATH.parent)),
        },
    )


if __name__ == "__main__":
    main()
