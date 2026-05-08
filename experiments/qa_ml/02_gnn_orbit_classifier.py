"""QA-ML v2 — Reachability-graph GCN vs feature-only baselines.

Hypothesis
----------
  QA generator topology (sigma, mu, lambda_2, nu edges) improves orbit-class
  learning under sparse satellite support, beyond what feature engineering
  alone (qa_full = (b,e,d,a,C,F,G,phi_b,phi_e)) can deliver.

Setup
-----
  Single transductive graph: 576 nodes (full mod-24 grid). Labels are binary
  (cosmos=0, satellite=1) with the lone singularity (24,24) masked out of
  supervision and evaluation. Edges from tools.qa_ml.qa_graph.build_edges,
  symmetrized, with self-loops added by GCN normalization.

  Train mask: n_train labeled nodes selected by StratifiedShuffleSplit per
  seed; test mask: the remaining 575 - n_train labeled nodes.

Models
------
  raw_mlp        : 2-layer MLP on (b, e)                   [no graph, 2 inputs]
  qa_full_logreg : LogisticRegression on qa_full features  [carried from v1]
  gcn_qa_full    : 2-layer GCN on qa_full + adjacency      [graph + features]

  Ablation: gcn_qa_full with adjacency replaced by identity (= MLP on qa_full).

Theorem NT
----------
  Adjacency entries and node features are integer-derived; the GCN itself is
  observer-side float arithmetic. Boundary crossed once on input, once on output
  (predicted class). No float feedback into the QA layer.

QA_COMPLIANCE = "qa_ml_gnn_benchmark — observer-side torch GCN on QA-derived
features and integer reachability adjacency (sigma, mu, lambda_2, nu)"
"""

from __future__ import annotations

BENCHMARK_PROTOCOL_REF = "benchmark_protocol_v2.json"

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

from tools.qa_ml import build_dataset  # noqa: E402
from tools.qa_ml.qa_graph import dense_adjacency, gcn_normalize  # noqa: E402
from qa_reproducibility import log_run  # noqa: E402

MODULUS = 24
N_TRAIN_LIST = [40, 80, 160, 320]
N_SEEDS = 20
EPOCHS = 300
HIDDEN = 32
LR = 0.01
WEIGHT_DECAY = 5e-4
DEVICE = torch.device("cpu")
PROTOCOL_PATH = Path(__file__).parent / "benchmark_protocol_v2.json"
OUT_PATH = Path(__file__).parent / "results_gnn.json"


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
    np.random.seed(seed)  # noqa: T2-D-5  observer-side seed for sklearn split + torch init
    torch.manual_seed(seed)


def _z_score(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu_tr = x[train_mask].mean(axis=0, keepdims=True)
    sd_tr = x[train_mask].std(axis=0, keepdims=True)
    sd_tr = np.where(sd_tr < 1e-8, 1.0, sd_tr)
    return (x - mu_tr) / sd_tr


def _train_torch_classifier(
    model: nn.Module,
    forward_fn,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    class_weight: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = forward_fn(model)
        loss = F.cross_entropy(logits[train_mask], y_tensor[train_mask], weight=class_weight)
        loss.backward()
        opt.step()


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def feature_set_swap(x_qa_full: np.ndarray, adj_norm: np.ndarray, ablation: str):
    """Ablation callable declared in benchmark_protocol_v2.json.

    'with_graph' returns (qa_full, normalized adjacency); 'without_graph'
    replaces adjacency with the identity, collapsing the GCN to a 2-layer MLP
    on the same features. Identical seed/init isolates the graph contribution.
    """
    if ablation == "with_graph":
        return x_qa_full, adj_norm
    if ablation == "without_graph":
        return x_qa_full, np.eye(adj_norm.shape[0], dtype=adj_norm.dtype)
    raise ValueError(f"unknown ablation {ablation!r}")


def evaluate_seed(
    x_raw: np.ndarray,
    x_qa_full: np.ndarray,
    adj_norm: np.ndarray,
    y_full: np.ndarray,
    label_mask: np.ndarray,
    n_train: int,
    seed: int,
) -> list[dict]:
    _set_seed(seed)

    labeled_indices = np.where(label_mask)[0]
    y_labeled = y_full[labeled_indices]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
    train_pos, test_pos = next(sss.split(np.zeros((len(y_labeled), 1)), y_labeled))
    train_idx = labeled_indices[train_pos]
    test_idx = labeled_indices[test_pos]

    n_total = x_qa_full.shape[0]
    train_mask_full = np.zeros(n_total, dtype=bool)
    train_mask_full[train_idx] = True

    rows: list[dict] = []

    # Standardize features per training fold (observer-side preprocessing).
    x_qa_full_std = _z_score(x_qa_full.astype(np.float64), train_mask_full)
    x_raw_std = _z_score(x_raw.astype(np.float64), train_mask_full)

    # ---- raw_mlp (no graph, raw features) ----
    _set_seed(seed)
    mlp = MLP(n_features=x_raw_std.shape[1], hidden=HIDDEN, n_classes=2)
    x_t = torch.from_numpy(x_raw_std).float()
    y_t = torch.from_numpy(y_full).long()
    cw = _class_weights(y_full[train_idx])
    _train_torch_classifier(
        mlp, lambda m: m(x_t), x_t, y_t,
        torch.from_numpy(train_mask_full),
        torch.from_numpy(cw).float(),
    )
    mlp.eval()
    with torch.no_grad():
        preds = mlp(x_t).argmax(dim=1).cpu().numpy()
    rows.append({
        "n_train": n_train, "seed": seed, "model": "raw_mlp", "ablation": "n/a",
        **_scores(y_full[test_idx], preds[test_idx]),
    })

    # ---- qa_full_logreg (no graph, qa_full features) — the v1 SOTA carry-over ----
    logreg = LogisticRegression(
        max_iter=4000, class_weight="balanced", solver="lbfgs", random_state=seed,
    )
    logreg.fit(x_qa_full[train_idx], y_full[train_idx])
    preds = logreg.predict(x_qa_full[test_idx])
    rows.append({
        "n_train": n_train, "seed": seed, "model": "qa_full_logreg", "ablation": "n/a",
        **_scores(y_full[test_idx], preds),
    })

    # ---- gcn_qa_full (with graph) ----
    for ablation in ("with_graph", "without_graph"):
        feats, adj_used = feature_set_swap(x_qa_full_std, adj_norm, ablation)
        _set_seed(seed)
        gcn = GCN(n_features=feats.shape[1], hidden=HIDDEN, n_classes=2)
        x_t = torch.from_numpy(feats).float()
        a_t = torch.from_numpy(adj_used).float()
        y_t = torch.from_numpy(y_full).long()
        _train_torch_classifier(
            gcn, lambda m: m(x_t, a_t), x_t, y_t,
            torch.from_numpy(train_mask_full),
            torch.from_numpy(cw).float(),
        )
        gcn.eval()
        with torch.no_grad():
            preds = gcn(x_t, a_t).argmax(dim=1).cpu().numpy()
        rows.append({
            "n_train": n_train, "seed": seed,
            "model": "gcn_qa_full", "ablation": ablation,
            **_scores(y_full[test_idx], preds[test_idx]),
        })

    return rows


def _class_weights(y_train: np.ndarray) -> np.ndarray:
    """Balanced class weights matching sklearn's class_weight='balanced'."""
    n = len(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    weights = n / (len(classes) * counts)
    full = np.ones(2, dtype=np.float64)
    for c, w in zip(classes, weights):
        full[int(c)] = w
    return full


def main() -> None:
    print(f"Loading mod-{MODULUS} dataset and graph ...")
    x_raw_l, _, x_qa_full_l, y_l, _ = build_dataset(MODULUS)
    x_raw = np.asarray(x_raw_l, dtype=np.int64)
    x_qa_full = np.asarray(x_qa_full_l, dtype=np.int64)
    y = np.asarray(y_l, dtype=np.int64)

    # Re-map labels: cosmos -> 0, satellite -> 1, singularity -> -1 (masked).
    y_binary = np.full_like(y, fill_value=-1)
    y_binary[y == 2] = 0  # cosmos
    y_binary[y == 1] = 1  # satellite
    label_mask = y_binary >= 0
    y_full = np.where(label_mask, y_binary, 0).astype(np.int64)
    print(f"  N={len(y_binary)}, labeled={int(label_mask.sum())}, "
          f"masked (singularity)={int((~label_mask).sum())}")
    print(f"  cosmos={int((y_binary == 0).sum())}, satellite={int((y_binary == 1).sum())}")

    adj = dense_adjacency(MODULUS, symmetric=True)
    adj_norm = gcn_normalize(adj)
    print(f"  adjacency: {adj.shape}, edges (incl. symmetrized) = {int((adj != 0).sum())}")

    t0 = time.time()
    results: list[dict] = []
    for n_train in N_TRAIN_LIST:
        print(f"  n_train={n_train} ...", flush=True)
        for seed in range(N_SEEDS):
            try:
                rows = evaluate_seed(
                    x_raw, x_qa_full, adj_norm, y_full, label_mask, n_train, seed,
                )
                results.extend(rows)
            except ValueError as exc:
                print(f"    skip seed={seed}: {exc}")
    elapsed = time.time() - t0
    print(f"  trained {len(results)} models in {elapsed:.1f}s")

    # Aggregate: macro_f1 mean +/- std over seeds, per (n_train, model+ablation).
    summary: dict[str, dict[str, float]] = {}
    for n_train in N_TRAIN_LIST:
        for key in ("raw_mlp", "qa_full_logreg",
                    "gcn_qa_full|with_graph", "gcn_qa_full|without_graph"):
            if "|" in key:
                model_name, ablation = key.split("|")
            else:
                model_name, ablation = key, "n/a"
            scores = [
                r["macro_f1"] for r in results
                if r["n_train"] == n_train
                and r["model"] == model_name
                and r["ablation"] == ablation
            ]
            if not scores:
                continue
            mean, std = float(np.mean(scores)), float(np.std(scores))
            summary[f"{n_train}|{key}"] = {"macro_f1_mean": mean, "macro_f1_std": std}

    print("\nMacro F1 (mean +/- std):")
    print(f"{'n_train':>8} {'model':>20} {'ablation':>16}  {'mean':>6}  {'std':>6}")
    print("-" * 64)
    for n_train in N_TRAIN_LIST:
        for key in ("raw_mlp", "qa_full_logreg",
                    "gcn_qa_full|with_graph", "gcn_qa_full|without_graph"):
            full_key = f"{n_train}|{key}"
            if full_key not in summary:
                continue
            if "|" in key:
                model_name, ablation = key.split("|")
            else:
                model_name, ablation = key, "n/a"
            v = summary[full_key]
            print(f"{n_train:>8} {model_name:>20} {ablation:>16}  "
                  f"{v['macro_f1_mean']:>6.3f}  {v['macro_f1_std']:>6.3f}")

    print("\nGCN-with-graph delta vs ablations (macro_f1 mean):")
    print(f"{'n_train':>8}  {'vs raw_mlp':>+12}  {'vs qa_logreg':>+14}  {'vs no_graph':>+14}")
    for n_train in N_TRAIN_LIST:
        with_g = summary.get(f"{n_train}|gcn_qa_full|with_graph")
        if not with_g:
            continue
        m_with = with_g["macro_f1_mean"]
        d_raw = m_with - summary.get(f"{n_train}|raw_mlp", {}).get("macro_f1_mean", 0.0)
        d_lr = m_with - summary.get(f"{n_train}|qa_full_logreg", {}).get("macro_f1_mean", 0.0)
        d_ng = m_with - summary.get(f"{n_train}|gcn_qa_full|without_graph", {}).get("macro_f1_mean", 0.0)
        print(f"{n_train:>8}  {d_raw:>+12.3f}  {d_lr:>+14.3f}  {d_ng:>+14.3f}")

    OUT_PATH.write_text(json.dumps({
        "modulus": MODULUS,
        "n_train_list": N_TRAIN_LIST,
        "n_seeds": N_SEEDS,
        "epochs": EPOCHS,
        "hidden": HIDDEN,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "raw_results": results,
        "summary": summary,
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")

    log_run(
        PROTOCOL_PATH,
        status="completed",
        results={
            "elapsed_s": elapsed,
            "n_models_trained": len(results),
            "summary": summary,
            "results_json": str(OUT_PATH.relative_to(PROTOCOL_PATH.parent)),
        },
    )


if __name__ == "__main__":
    main()
