#!/usr/bin/env python3
"""Empirical κ experiment — QA-modulated SGD on synthetic classification.

Design:
  - Synthetic 2-class Gaussian data (n=800, d=20)
  - Tiny MLP: 20 → 32 (ReLU) → 1 (sigmoid), binary cross-entropy
  - 8 conditions: 7 QA substrates spanning H_QA ∈ [0.20, 0.71] + plain SGD baseline
  - Base lr fixed across all conditions; QA-modulated update:
      p_after = p_before - lr * gain * H_QA * grad
      gain = min(||grad_flat||_2, 2.0)   [family 101 derived gain]
      κ_t  = 1 - |1 - lr * gain * H_QA|
  - Metrics per condition:
      - Epochs to 90% train accuracy (or inf if never reached)
      - Final loss at epoch 300
      - Mean κ over all steps
      - Pearson r(κ_t, Δloss_t) within-run
  - Results written to empirical_kappa_results.json + printed table
"""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ── Experiment hyperparameters ─────────────────────────────────────────────────
N_SAMPLES   = 800
N_FEATURES  = 20
N_HIDDEN    = 32
BATCH_SIZE  = 64
N_EPOCHS    = 300
BASE_LR     = 0.1        # fixed across all conditions
EPS         = 1e-12
GAIN_CAP    = 2.0
THRESH_ACC  = 0.90       # convergence threshold

# ── QA substrate helpers ───────────────────────────────────────────────────────

def qa_step(b: float, e: float, m: int = 9):
    d = (int(b) + int(e)) % m or m
    a = (int(b) + 2 * int(e)) % m or m
    return float(d), float(a)


def compute_h_qa(b: float, e: float) -> float:
    d, a = qa_step(b, e)
    G     = e * e + d * d
    F     = b * a
    h_raw = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h_raw) / (1.0 + abs(h_raw))


def grad_gain(grad_flat: np.ndarray) -> float:
    norm = float(np.sqrt(np.sum(grad_flat ** 2)))
    return min(norm, GAIN_CAP)


def kappa(lr: float, gain: float, h_qa: float) -> float:
    return 1.0 - abs(1.0 - lr * gain * h_qa)


# ── MLP in pure numpy ──────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def bce_loss(pred: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(pred, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def forward(W1, b1, W2, b2, X):
    z1   = X @ W1 + b1       # (n, H)
    a1   = relu(z1)           # (n, H)
    z2   = a1 @ W2 + b2       # (n, 1)
    pred = sigmoid(z2)        # (n, 1)
    return z1, a1, z2, pred


def backward(W1, b1, W2, b2, X, y, z1, a1, z2, pred):
    n    = X.shape[0]
    # output layer
    dz2  = (pred - y) / n                # (n, 1)
    dW2  = a1.T @ dz2                    # (H, 1)
    db2  = dz2.sum(axis=0, keepdims=True)  # (1, 1)
    # hidden layer
    da1  = dz2 @ W2.T                    # (n, H)
    dz1  = da1 * (z1 > 0).astype(float)  # (n, H)
    dW1  = X.T @ dz1                     # (d, H)
    db1  = dz1.sum(axis=0, keepdims=True)  # (1, H)
    return dW1, db1, dW2, db2


def init_weights(seed_offset: int = 0):
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    scale1 = math.sqrt(2.0 / N_FEATURES)
    scale2 = math.sqrt(2.0 / N_HIDDEN)
    W1 = rng.normal(0, scale1, (N_FEATURES, N_HIDDEN))
    b1 = np.zeros((1, N_HIDDEN))
    W2 = rng.normal(0, scale2, (N_HIDDEN, 1))
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


# ── Training loop ──────────────────────────────────────────────────────────────

def train_condition(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    h_qa: float,          # 0.0 means plain SGD (H_QA=1, gain=1 fixed)
    seed_offset: int = 0,
) -> dict:
    """Train one condition. h_qa=0.0 is a sentinel for plain SGD."""
    plain_sgd = (h_qa == 0.0)

    W1, b1, W2, b2 = init_weights(seed_offset)
    n_tr   = X_tr.shape[0]
    n_steps = math.ceil(n_tr / BATCH_SIZE)

    kappas:     list[float] = []
    delta_losses: list[float] = []
    epoch_losses: list[float] = []
    epoch_accs:  list[float]  = []
    epochs_to_thresh          = None

    prev_loss: float | None = None

    for epoch in range(N_EPOCHS):
        # shuffle
        idx = np.random.permutation(n_tr)
        X_sh, y_sh = X_tr[idx], y_tr[idx]

        for step in range(n_steps):
            sl   = slice(step * BATCH_SIZE, (step + 1) * BATCH_SIZE)
            Xb   = X_sh[sl]
            yb   = y_sh[sl]

            z1, a1, z2, pred = forward(W1, b1, W2, b2, Xb)
            loss_b = bce_loss(pred, yb)

            dW1, db1, dW2, db2 = backward(W1, b1, W2, b2, Xb, yb, z1, a1, z2, pred)

            # collect flat gradient for gain
            grad_flat = np.concatenate([
                dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()
            ])

            if plain_sgd:
                effective_gain = 1.0
                effective_hqa  = 1.0
            else:
                effective_gain = grad_gain(grad_flat)
                effective_hqa  = h_qa

            k = kappa(BASE_LR, effective_gain, effective_hqa)
            kappas.append(k)

            if prev_loss is not None:
                delta_losses.append(prev_loss - loss_b)
            prev_loss = loss_b

            scale = BASE_LR * effective_gain * effective_hqa
            W1 -= scale * dW1
            b1 -= scale * db1
            W2 -= scale * dW2
            b2 -= scale * db2

        # end-of-epoch metrics
        _, _, _, pred_all = forward(W1, b1, W2, b2, X_tr)
        loss_e = bce_loss(pred_all, y_tr)
        acc_e  = float(np.mean((pred_all.ravel() > 0.5) == y_tr.ravel()))
        epoch_losses.append(loss_e)
        epoch_accs.append(acc_e)

        if epochs_to_thresh is None and acc_e >= THRESH_ACC:
            epochs_to_thresh = epoch + 1

    mean_kappa  = float(np.mean(kappas)) if kappas else 0.0
    # Pearson correlation of κ_t with Δloss_t (align lengths)
    ks = np.array(kappas[1:])          # drop first step (no delta yet)
    dl = np.array(delta_losses)
    min_len = min(len(ks), len(dl))
    if min_len > 2:
        r = float(np.corrcoef(ks[:min_len], dl[:min_len])[0, 1])
    else:
        r = 0.0

    return {
        "mean_kappa":        mean_kappa,
        "pearson_r_kappa_dloss": r,
        "final_loss":        epoch_losses[-1],
        "final_acc":         epoch_accs[-1],
        "epochs_to_thresh":  epochs_to_thresh if epochs_to_thresh is not None else -1,
        "epoch_losses":      epoch_losses,
        "epoch_accs":        epoch_accs,
    }


# ── Conditions ─────────────────────────────────────────────────────────────────

SUBSTRATES = [
    # label,        b,  e
    ("plain_SGD",   0,  0),   # sentinel: plain SGD
    ("(2,8) low",   2,  8),
    ("(4,7)",       4,  7),
    ("(1,4)",       1,  4),
    ("(5,3) mid",   5,  3),
    ("(9,8)",       9,  8),
    ("(3,5)",       3,  5),
    ("(1,5) high",  1,  5),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Generate synthetic data
    rng = np.random.default_rng(RNG_SEED)
    mu0 = rng.normal(0, 1, N_FEATURES)
    mu1 = rng.normal(1, 1, N_FEATURES)
    X0  = rng.normal(mu0, 1.5, (N_SAMPLES // 2, N_FEATURES))
    X1  = rng.normal(mu1, 1.5, (N_SAMPLES // 2, N_FEATURES))
    X   = np.vstack([X0, X1]).astype(np.float64)
    y   = np.concatenate([np.zeros(N_SAMPLES // 2), np.ones(N_SAMPLES // 2)]).reshape(-1, 1)
    # Standardize
    mu  = X.mean(axis=0); sd = X.std(axis=0) + EPS
    X   = (X - mu) / sd

    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features, binary")
    print(f"MLP: {N_FEATURES} -> {N_HIDDEN} (ReLU) -> 1 (sigmoid)")
    print(f"Base LR: {BASE_LR}, Epochs: {N_EPOCHS}, Batch: {BATCH_SIZE}")
    print(f"QA update: p -= lr * gain * H_QA * grad;  gain = min(||grad||, {GAIN_CAP})")
    print(f"κ = 1 - |1 - lr * gain * H_QA|")
    print()

    results = []
    for i, (label, b, e) in enumerate(SUBSTRATES):
        if b == 0 and e == 0:
            h_qa_val = 0.0   # sentinel for plain SGD
            display_hqa = "1.0 (fixed)"
        else:
            h_qa_val = compute_h_qa(float(b), float(e))
            display_hqa = f"{h_qa_val:.6f}"

        print(f"Running [{i+1}/{len(SUBSTRATES)}]: {label}  H_QA={display_hqa} ...", end=" ", flush=True)
        res = train_condition(X, y, h_qa_val, seed_offset=i)
        print(f"loss={res['final_loss']:.4f}  acc={res['final_acc']:.3f}  "
              f"mean_κ={res['mean_kappa']:.4f}  r={res['pearson_r_kappa_dloss']:.4f}  "
              f"thresh_epoch={res['epochs_to_thresh']}")

        results.append({
            "label":      label,
            "b":          b, "e": e,
            "H_QA":       h_qa_val if h_qa_val != 0.0 else 1.0,
            "plain_sgd":  (h_qa_val == 0.0),
            **res,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 88)
    print(f"{'Condition':<20} {'H_QA':>8} {'mean_κ':>8} {'r(κ,Δl)':>9} "
          f"{'final_loss':>11} {'final_acc':>10} {'ep@90%':>7}")
    print("-" * 88)
    for r in results:
        thresh = r["epochs_to_thresh"] if r["epochs_to_thresh"] != -1 else "N/A"
        print(f"{r['label']:<20} {r['H_QA']:>8.4f} {r['mean_kappa']:>8.4f} "
              f"{r['pearson_r_kappa_dloss']:>9.4f} {r['final_loss']:>11.6f} "
              f"{r['final_acc']:>10.4f} {str(thresh):>7}")
    print("=" * 88)

    # ── Correlation analysis ───────────────────────────────────────────────────
    hqas       = np.array([r["H_QA"] for r in results])
    mean_kaps  = np.array([r["mean_kappa"] for r in results])
    final_accs = np.array([r["final_acc"] for r in results])
    final_loss = np.array([r["final_loss"] for r in results])
    thresh_eps = np.array([r["epochs_to_thresh"] for r in results], dtype=float)
    thresh_eps[thresh_eps < 0] = N_EPOCHS  # never-converged → max

    if len(results) > 2:
        r_hqa_acc   = float(np.corrcoef(hqas, final_accs)[0, 1])
        r_kap_acc   = float(np.corrcoef(mean_kaps, final_accs)[0, 1])
        r_kap_loss  = float(np.corrcoef(mean_kaps, final_loss)[0, 1])
        r_kap_epoch = float(np.corrcoef(mean_kaps, thresh_eps)[0, 1])
        print()
        print("Cross-condition Pearson correlations:")
        print(f"  r(H_QA,    final_acc) = {r_hqa_acc:+.4f}")
        print(f"  r(mean_κ,  final_acc) = {r_kap_acc:+.4f}")
        print(f"  r(mean_κ,  final_loss)= {r_kap_loss:+.4f}")
        print(f"  r(mean_κ,  ep@90%)   = {r_kap_epoch:+.4f}  (negative=faster)")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out = {
        "experiment":  "empirical_kappa_v1",
        "n_samples":   N_SAMPLES, "n_features": N_FEATURES,
        "n_hidden":    N_HIDDEN,  "batch_size":  BATCH_SIZE,
        "n_epochs":    N_EPOCHS,  "base_lr":     BASE_LR,
        "gain_cap":    GAIN_CAP,  "thresh_acc":  THRESH_ACC,
        "rng_seed":    RNG_SEED,
        "conditions":  results,
    }
    # strip epoch curves from JSON to keep it small
    for r in out["conditions"]:
        del r["epoch_losses"]
        del r["epoch_accs"]

    out_path = Path(__file__).parent / "empirical_kappa_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
