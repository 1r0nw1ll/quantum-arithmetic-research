#!/usr/bin/env python3
"""Empirical κ experiment — GNN message passing (family [93]).

Design:
  - Synthetic stochastic-block-model graph: N=200 nodes, 2 communities
  - 2-layer GCN (pure numpy): H^(l+1) = ReLU(A_norm @ H^(l) @ W^(l))
  - Node binary classification, BCE loss
  - 8 conditions: 7 QA substrates spanning H_QA ∈ [0.20,0.71] + plain SGD
  - QA-modulated update: W -= lr * gain * H_QA * dW
      gain = min(||[dW1,dW2]_flat||_2, 2.0)
      κ_t  = 1 - |1 - lr * gain * H_QA|
  - Results written to empirical_kappa_gnn_results.json

Prediction: r(mean_κ, final_loss) < -0.70 — same pattern as MLP [89].
"""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path

# ── Reproducibility ─────────────────────────────────────────────────────────
RNG_SEED   = 42
np.random.seed(RNG_SEED)

# ── Hyperparameters ──────────────────────────────────────────────────────────
N_NODES    = 200
N_FEATURES = 16
N_HIDDEN   = 32
N_EPOCHS   = 300
BASE_LR    = 0.05
EPS        = 1e-12
GAIN_CAP   = 2.0
THRESH_ACC = 0.80      # node classification is harder → lower threshold

# ── Graph construction (stochastic block model) ──────────────────────────────

def make_sbm_graph(n: int = N_NODES, p_in: float = 0.30, p_out: float = 0.05,
                   rng: np.random.Generator = None) -> tuple:
    """Return (A_norm, X, y) for a 2-community SBM node classification task."""
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    half = n // 2
    labels = np.array([0] * half + [1] * half)

    # Adjacency
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            same = (labels[i] == labels[j])
            if rng.random() < (p_in if same else p_out):
                A[i, j] = A[j, i] = 1.0

    # Symmetric normalised adjacency with self-loops: D^{-1/2}(A+I)D^{-1/2}
    A_hat  = A + np.eye(n)
    deg    = A_hat.sum(axis=1)
    D_inv  = np.diag(1.0 / np.sqrt(np.maximum(deg, EPS)))
    A_norm = D_inv @ A_hat @ D_inv

    # Node features: Gaussian with class-shifted mean
    mu0 = rng.normal(0, 1, N_FEATURES)
    mu1 = rng.normal(1, 1, N_FEATURES)
    X0  = rng.normal(mu0, 1.0, (half, N_FEATURES))
    X1  = rng.normal(mu1, 1.0, (half, N_FEATURES))
    X   = np.vstack([X0, X1]).astype(np.float64)
    mu  = X.mean(0); sd = X.std(0) + EPS
    X   = (X - mu) / sd

    y = labels.reshape(-1, 1).astype(np.float64)
    return A_norm, X, y


# ── QA helpers ───────────────────────────────────────────────────────────────

def qa_step(b: float, e: float, m: int = 9):
    d = (int(b) + int(e)) % m or m
    a = (int(b) + 2 * int(e)) % m or m
    return float(d), float(a)


def compute_h_qa(b: float, e: float) -> float:
    d, a = qa_step(b, e)
    G    = e * e + d * d
    F    = b * a
    h    = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h) / (1.0 + abs(h))


def grad_gain(grad_flat: np.ndarray) -> float:
    return min(float(np.sqrt((grad_flat ** 2).sum())), GAIN_CAP)


def kappa(lr: float, gain: float, h_qa: float) -> float:
    return 1.0 - abs(1.0 - lr * gain * h_qa)


# ── GCN forward / backward (2 layers, pure numpy) ────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x):
    return np.maximum(0.0, x)


def bce(pred, y):
    p = np.clip(pred, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gcn_forward(A_norm, X, W1, W2):
    """Returns (Z1, H1, logits, pred)."""
    AX   = A_norm @ X          # N×d
    Z1   = AX @ W1             # N×H
    H1   = relu(Z1)
    AH1  = A_norm @ H1         # N×H
    logits = AH1 @ W2          # N×1
    pred   = sigmoid(logits)
    return Z1, H1, AH1, logits, pred


def gcn_backward(A_norm, X, W1, W2, Z1, H1, AH1, pred, y):
    """Returns (dW1, dW2) gradients."""
    n    = X.shape[0]
    dlog = (pred - y) / n          # N×1
    dW2  = AH1.T @ dlog            # H×1
    dAH1 = dlog @ W2.T             # N×H
    dH1  = A_norm.T @ dAH1        # N×H  (A_norm symmetric, so A_norm.T=A_norm)
    dZ1  = dH1 * (Z1 > 0)         # N×H
    AX   = A_norm @ X
    dW1  = AX.T @ dZ1             # d×H
    return dW1, dW2


def init_gcn(seed_offset: int = 0):
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    s1  = math.sqrt(2.0 / N_FEATURES)
    s2  = math.sqrt(2.0 / N_HIDDEN)
    W1  = rng.normal(0, s1, (N_FEATURES, N_HIDDEN))
    W2  = rng.normal(0, s2, (N_HIDDEN, 1))
    return W1, W2


# ── Training ─────────────────────────────────────────────────────────────────

def train_gcn(A_norm, X, y, h_qa: float, seed_offset: int = 0) -> dict:
    plain_sgd = (h_qa == 0.0)
    W1, W2    = init_gcn(seed_offset)

    epoch_losses: list[float] = []
    epoch_accs:  list[float]  = []
    kappas:      list[float]  = []
    delta_losses: list[float] = []
    epochs_to_thresh          = None
    prev_loss: float | None   = None

    for epoch in range(N_EPOCHS):
        Z1, H1, AH1, logits, pred = gcn_forward(A_norm, X, W1, W2)
        loss = bce(pred, y)

        dW1, dW2 = gcn_backward(A_norm, X, W1, W2, Z1, H1, AH1, pred, y)
        grad_flat = np.concatenate([dW1.ravel(), dW2.ravel()])

        if plain_sgd:
            eff_gain = 1.0; eff_hqa = 1.0
        else:
            eff_gain = grad_gain(grad_flat); eff_hqa = h_qa

        k = kappa(BASE_LR, eff_gain, eff_hqa)
        kappas.append(k)

        if prev_loss is not None:
            delta_losses.append(prev_loss - loss)
        prev_loss = loss

        scale = BASE_LR * eff_gain * eff_hqa
        W1 -= scale * dW1
        W2 -= scale * dW2

        # End-of-epoch metrics (same weights, re-evaluate)
        _, _, _, _, pred_all = gcn_forward(A_norm, X, W1, W2)
        loss_e = bce(pred_all, y)
        acc_e  = float(np.mean((pred_all.ravel() > 0.5) == y.ravel()))
        epoch_losses.append(loss_e)
        epoch_accs.append(acc_e)
        if epochs_to_thresh is None and acc_e >= THRESH_ACC:
            epochs_to_thresh = epoch + 1

    mean_kappa = float(np.mean(kappas))
    ks = np.array(kappas[1:])
    dl = np.array(delta_losses)
    mn = min(len(ks), len(dl))
    r  = float(np.corrcoef(ks[:mn], dl[:mn])[0, 1]) if mn > 2 else 0.0

    return {
        "mean_kappa":           mean_kappa,
        "pearson_r_kappa_dloss": r,
        "final_loss":           epoch_losses[-1],
        "final_acc":            epoch_accs[-1],
        "min_loss":             min(epoch_losses),
        "epochs_to_thresh":     epochs_to_thresh if epochs_to_thresh is not None else -1,
    }


# ── Conditions ───────────────────────────────────────────────────────────────

SUBSTRATES = [
    ("plain_SGD",  0, 0),
    ("(2,8) low",  2, 8),
    ("(4,7)",      4, 7),
    ("(1,4)",      1, 4),
    ("(5,3) mid",  5, 3),
    ("(9,8)",      9, 8),
    ("(3,5)",      3, 5),
    ("(1,5) high", 1, 5),
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RNG_SEED)
    A_norm, X, y = make_sbm_graph(rng=rng)
    print(f"Graph: {N_NODES} nodes, 2 communities, 2-layer GCN")
    print(f"Base LR: {BASE_LR}, Epochs: {N_EPOCHS}")
    print(f"QA update: W -= lr * gain * H_QA * dW;  gain = min(||dW||, {GAIN_CAP})")
    print()

    results = []
    for i, (label, b, e) in enumerate(SUBSTRATES):
        if b == 0 and e == 0:
            h_qa_val   = 0.0
            display_hqa = "1.0 (fixed)"
        else:
            h_qa_val   = compute_h_qa(float(b), float(e))
            display_hqa = f"{h_qa_val:.6f}"

        print(f"[{i+1}/{len(SUBSTRATES)}] {label}  H_QA={display_hqa} ...", end=" ", flush=True)
        res = train_gcn(A_norm, X, y, h_qa_val, seed_offset=i)
        print(f"loss={res['final_loss']:.4f}  acc={res['final_acc']:.3f}  "
              f"mean_κ={res['mean_kappa']:.4f}  r={res['pearson_r_kappa_dloss']:.4f}  "
              f"ep@{int(THRESH_ACC*100)}%={res['epochs_to_thresh']}")

        results.append({
            "label":     label, "b": b, "e": e,
            "H_QA":      h_qa_val if h_qa_val != 0.0 else 1.0,
            "plain_sgd": (h_qa_val == 0.0),
            **res,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"{'Condition':<20} {'H_QA':>8} {'mean_κ':>8} {'r(κ,Δl)':>9} "
          f"{'final_loss':>11} {'final_acc':>10} {'ep@80%':>7}")
    print("-" * 90)
    for r in results:
        ep = r["epochs_to_thresh"] if r["epochs_to_thresh"] != -1 else "N/A"
        print(f"{r['label']:<20} {r['H_QA']:>8.4f} {r['mean_kappa']:>8.4f} "
              f"{r['pearson_r_kappa_dloss']:>9.4f} {r['final_loss']:>11.6f} "
              f"{r['final_acc']:>10.4f} {str(ep):>7}")
    print("=" * 90)

    hqas      = np.array([r["H_QA"]       for r in results])
    mean_kaps = np.array([r["mean_kappa"]  for r in results])
    fin_loss  = np.array([r["final_loss"]  for r in results])
    fin_acc   = np.array([r["final_acc"]   for r in results])

    r_kap_loss = float(np.corrcoef(mean_kaps, fin_loss)[0, 1])
    r_kap_acc  = float(np.corrcoef(mean_kaps, fin_acc)[0, 1])
    r_hqa_acc  = float(np.corrcoef(hqas, fin_acc)[0, 1])
    print()
    print("Cross-condition Pearson correlations:")
    print(f"  r(H_QA,   final_acc) = {r_hqa_acc:+.4f}")
    print(f"  r(mean_κ, final_acc) = {r_kap_acc:+.4f}")
    print(f"  r(mean_κ, final_loss)= {r_kap_loss:+.4f}  (target: < -0.70)")

    out = {
        "experiment": "empirical_kappa_gnn_v1",
        "architecture": "2-layer GCN, node binary classification",
        "n_nodes": N_NODES, "n_features": N_FEATURES, "n_hidden": N_HIDDEN,
        "n_epochs": N_EPOCHS, "base_lr": BASE_LR, "gain_cap": GAIN_CAP,
        "thresh_acc": THRESH_ACC, "rng_seed": RNG_SEED,
        "r_kappa_loss": r_kap_loss,
        "r_kappa_acc":  r_kap_acc,
        "r_hqa_acc":    r_hqa_acc,
        "conditions": results,
    }
    out_path = Path(__file__).parent / "empirical_kappa_gnn_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
