#!/usr/bin/env python3
"""Empirical κ experiment 3 — corrected gain (gain=1, normalized).

Diagnosis from Exp 1: gain = min(||grad||, 2.0) ≈ 0.03–0.08 for typical
MLP gradients, making η_eff = lr * gain * H_QA ≈ 0.003–0.008 regardless
of substrate. The H_QA ordering is masked by the tiny effective step.

Fix: set gain = 1.0 (normalized). Now η_eff = lr * H_QA directly, and the
substrate ordering is cleanly exposed.

Design:
  - Same synthetic dataset as Exp 1 (n=800, d=20, binary Gaussian)
  - Same MLP architecture (20→32→1)
  - Same substrates
  - gain = 1.0 (not gradient-norm-based)
  - BASE_LR swept over [0.5, 1.0, 1.5] to probe η_eff regimes
  - 5 independent seeds per condition
  - Primary metric: r(mean_κ, final_loss) across substrates

Comparison baseline: plain SGD with matched η_eff = BASE_LR (gain=H_QA=1).

Outputs:
  empirical_kappa_exp3_results.json
  empirical_kappa_exp3_curves.png
  empirical_kappa_exp3_correlation.png
"""
from __future__ import annotations

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
RNG_SEED = 42
np.random.seed(RNG_SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_SAMPLES  = 800
N_FEATURES = 20
N_HIDDEN   = 32
BATCH_SIZE = 64
N_EPOCHS   = 300
EPS        = 1e-12
THRESH_ACC = 0.90
N_SEEDS    = 5

# Base LR values to sweep — η_eff = BASE_LR * H_QA (gain=1)
# High H_QA ≈ 0.71, low H_QA ≈ 0.20
# TARGET: η_eff ∈ (0, 1) for all substrates → BASE_LR < 1/0.71 ≈ 1.41
LR_SWEEP = [0.5, 1.0, 1.4]

# ── QA substrate helpers ───────────────────────────────────────────────────────

def qa_step(b: float, e: float, m: int = 9):
    d = (int(b) + int(e)) % m or m
    a = (int(b) + 2 * int(e)) % m or m
    return float(d), float(a)


def compute_h_qa(b: float, e: float) -> float:
    d, a = qa_step(b, e)
    G    = e * e + d * d
    F    = b * a
    h_raw = 0.25 * (F / (G + EPS) + (e * d) / (a + b + EPS))
    return abs(h_raw) / (1.0 + abs(h_raw))


def kappa(lr: float, h_qa: float) -> float:
    """κ_t = 1 - |1 - η_eff| where η_eff = lr * 1.0 * H_QA (gain=1)."""
    return 1.0 - abs(1.0 - lr * h_qa)


# ── MLP ────────────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def relu(x):
    return np.maximum(0.0, x)

def bce_loss(pred, y):
    p = np.clip(pred, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def forward(W1, b1, W2, b2, X):
    z1   = X @ W1 + b1
    a1   = relu(z1)
    z2   = a1 @ W2 + b2
    pred = sigmoid(z2)
    return z1, a1, z2, pred

def backward(W1, b1, W2, b2, X, y, z1, a1, z2, pred):
    n   = X.shape[0]
    dz2 = (pred - y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).astype(float)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def init_weights(seed):
    rng = np.random.default_rng(seed)
    W1  = rng.normal(0, math.sqrt(2.0 / N_FEATURES), (N_FEATURES, N_HIDDEN))
    b1  = np.zeros((1, N_HIDDEN))
    W2  = rng.normal(0, math.sqrt(2.0 / N_HIDDEN), (N_HIDDEN, 1))
    b2  = np.zeros((1, 1))
    return W1, b1, W2, b2


# ── Training ───────────────────────────────────────────────────────────────────

def train_one(X_tr, y_tr, h_qa: float, base_lr: float,
              plain_sgd: bool, seed: int) -> dict:
    """Train one (substrate, lr, seed) triple. Returns metrics dict."""
    W1, b1, W2, b2 = init_weights(seed)
    n_tr    = X_tr.shape[0]
    n_steps = math.ceil(n_tr / BATCH_SIZE)

    rng          = np.random.default_rng(seed + 1000)
    kappas       = []
    epoch_losses = []
    epoch_accs   = []
    epochs_to_thresh = None

    for epoch in range(N_EPOCHS):
        idx = rng.permutation(n_tr)
        X_sh, y_sh = X_tr[idx], y_tr[idx]

        for step in range(n_steps):
            sl   = slice(step * BATCH_SIZE, (step + 1) * BATCH_SIZE)
            Xb, yb = X_sh[sl], y_sh[sl]

            z1, a1, z2, pred = forward(W1, b1, W2, b2, Xb)
            dW1, db1, dW2, db2 = backward(W1, b1, W2, b2, Xb, yb, z1, a1, z2, pred)

            if plain_sgd:
                scale = base_lr          # η_eff = base_lr (baseline)
                k     = 1.0 - abs(1.0 - base_lr)
            else:
                scale = base_lr * h_qa   # η_eff = base_lr * H_QA (gain=1)
                k     = kappa(base_lr, h_qa)

            kappas.append(k)
            W1 -= scale * dW1
            b1 -= scale * db1
            W2 -= scale * dW2
            b2 -= scale * db2

        _, _, _, pred_all = forward(W1, b1, W2, b2, X_tr)
        loss_e = bce_loss(pred_all, y_tr)
        acc_e  = float(np.mean((pred_all.ravel() > 0.5) == y_tr.ravel()))
        epoch_losses.append(loss_e)
        epoch_accs.append(acc_e)
        if epochs_to_thresh is None and acc_e >= THRESH_ACC:
            epochs_to_thresh = epoch + 1

    return {
        "mean_kappa":       float(np.mean(kappas)),
        "final_loss":       epoch_losses[-1],
        "final_acc":        epoch_accs[-1],
        "epochs_to_thresh": epochs_to_thresh if epochs_to_thresh is not None else -1,
        "epoch_losses":     epoch_losses,
        "epoch_accs":       epoch_accs,
    }


# ── Substrates ─────────────────────────────────────────────────────────────────

SUBSTRATES = [
    # label,         b,  e,   plain_sgd
    ("plain_SGD",    0,  0,   True),
    ("(2,8) low",    2,  8,   False),
    ("(4,7)",        4,  7,   False),
    ("(1,4)",        1,  4,   False),
    ("(5,3) mid",    5,  3,   False),
    ("(9,8)",        9,  8,   False),
    ("(3,5)",        3,  5,   False),
    ("(1,5) high",   1,  5,   False),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def build_dataset() -> tuple:
    rng = np.random.default_rng(RNG_SEED)
    mu0 = rng.normal(0, 1, N_FEATURES)
    mu1 = rng.normal(1, 1, N_FEATURES)
    X0  = rng.normal(mu0, 1.5, (N_SAMPLES // 2, N_FEATURES))
    X1  = rng.normal(mu1, 1.5, (N_SAMPLES // 2, N_FEATURES))
    X   = np.vstack([X0, X1]).astype(np.float64)
    y   = np.concatenate([np.zeros(N_SAMPLES//2), np.ones(N_SAMPLES//2)]).reshape(-1, 1)
    mu  = X.mean(0); sd = X.std(0) + EPS
    return (X - mu) / sd, y


def main():
    X_tr, y_tr = build_dataset()
    print(f"Dataset: {N_SAMPLES} samples, {N_FEATURES} features  |  "
          f"{N_SEEDS} seeds  |  LR sweep: {LR_SWEEP}")
    print(f"Update rule: p -= lr * H_QA * grad  (gain=1, corrected from Exp 1)\n")

    all_results = {}  # base_lr -> list of condition dicts

    for base_lr in LR_SWEEP:
        print(f"{'='*72}")
        print(f"BASE_LR = {base_lr}  (η_eff range: "
              f"[{base_lr*0.20:.3f}, {base_lr*0.71:.3f}])")
        print(f"{'='*72}")
        cond_results = []

        for label, b, e, plain in SUBSTRATES:
            h_qa = 0.0 if plain else compute_h_qa(float(b), float(e))
            eta_eff = base_lr if plain else base_lr * h_qa

            seed_losses = []
            seed_accs   = []
            seed_kappas = []
            seed_thresh = []
            seed_curves = []

            for seed in range(N_SEEDS):
                res = train_one(X_tr, y_tr, h_qa, base_lr, plain, seed=RNG_SEED + seed * 100)
                seed_losses.append(res["final_loss"])
                seed_accs.append(res["final_acc"])
                seed_kappas.append(res["mean_kappa"])
                seed_thresh.append(res["epochs_to_thresh"])
                seed_curves.append(res["epoch_losses"])

            mean_loss  = float(np.mean(seed_losses))
            std_loss   = float(np.std(seed_losses))
            mean_acc   = float(np.mean(seed_accs))
            mean_kappa = float(np.mean(seed_kappas))
            thresh_arr = [t for t in seed_thresh if t > 0]
            mean_thresh = float(np.mean(thresh_arr)) if thresh_arr else -1.0

            print(f"  {label:<18} H_QA={h_qa:.4f}  η_eff={eta_eff:.4f}  "
                  f"κ={mean_kappa:.4f}  loss={mean_loss:.4f}±{std_loss:.4f}  "
                  f"acc={mean_acc:.3f}  ep@90%={mean_thresh:.1f}")

            cond_results.append({
                "label":         label,
                "b": b, "e": e,
                "H_QA":          h_qa if not plain else 1.0,
                "eta_eff":       eta_eff,
                "plain_sgd":     plain,
                "mean_kappa":    mean_kappa,
                "mean_loss":     mean_loss,
                "std_loss":      std_loss,
                "mean_acc":      mean_acc,
                "mean_thresh":   mean_thresh,
                "seed_curves":   seed_curves,
            })

        # Cross-condition correlations
        hqas   = np.array([c["H_QA"]     for c in cond_results])
        kaps   = np.array([c["mean_kappa"] for c in cond_results])
        losses = np.array([c["mean_loss"]  for c in cond_results])
        accs   = np.array([c["mean_acc"]   for c in cond_results])

        r_kap_loss = float(np.corrcoef(kaps, losses)[0, 1])
        r_hqa_acc  = float(np.corrcoef(hqas, accs)[0, 1])
        print(f"\n  r(mean_κ,  final_loss) = {r_kap_loss:+.4f}  "
              f"(Exp 1 had -0.843 with gain=||g||)")
        print(f"  r(H_QA,    final_acc)  = {r_hqa_acc:+.4f}")

        all_results[str(base_lr)] = {
            "base_lr":     base_lr,
            "r_kap_loss":  r_kap_loss,
            "r_hqa_acc":   r_hqa_acc,
            "conditions":  cond_results,
        }

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_curves(all_results)
    _plot_correlation(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "empirical_kappa_exp3_results.json"
    # strip curves to keep JSON small
    for lr_key, block in all_results.items():
        for c in block["conditions"]:
            del c["seed_curves"]
    out_path.write_text(json.dumps({
        "experiment": "empirical_kappa_exp3_gain1",
        "description": "Corrected gain=1; η_eff=lr*H_QA; LR sweep",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_hidden": N_HIDDEN, "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS, "n_seeds": N_SEEDS,
        "lr_sweep": LR_SWEEP,
        "results": all_results,
    }, indent=2))
    print(f"\nResults saved to {out_path}")


def _plot_curves(all_results):
    fig, axes = plt.subplots(1, len(LR_SWEEP), figsize=(5 * len(LR_SWEEP), 4), sharey=True)
    if len(LR_SWEEP) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SUBSTRATES)))

    for ax, (lr_key, block) in zip(axes, all_results.items()):
        for i, cond in enumerate(block["conditions"]):
            curves = cond["seed_curves"]
            mean_c = np.mean(curves, axis=0)
            label  = f"{cond['label']} (κ={cond['mean_kappa']:.2f})"
            ls     = "--" if cond["plain_sgd"] else "-"
            ax.plot(mean_c, color=colors[i], ls=ls, lw=1.5, label=label, alpha=0.85)
        ax.set_title(f"lr={block['base_lr']}  r(κ,loss)={block['r_kap_loss']:+.3f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=6)
        ax.set_yscale("log")

    fig.suptitle("Exp 3: corrected gain=1 (η_eff = lr × H_QA)", fontsize=11)
    plt.tight_layout()
    out = Path(__file__).parent / "empirical_kappa_exp3_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Loss curves saved to {out}")


def _plot_correlation(all_results):
    n_lr = len(LR_SWEEP)
    fig, axes = plt.subplots(1, n_lr, figsize=(4 * n_lr, 4))
    if n_lr == 1:
        axes = [axes]

    for ax, (lr_key, block) in zip(axes, all_results.items()):
        kaps   = [c["mean_kappa"] for c in block["conditions"]]
        losses = [c["mean_loss"]  for c in block["conditions"]]
        plain  = [c["plain_sgd"]  for c in block["conditions"]]
        for k, l, p in zip(kaps, losses, plain):
            marker = "D" if p else "o"
            ax.scatter(k, l, marker=marker, s=60, zorder=3)
        ax.set_xlabel("mean κ")
        ax.set_ylabel("final loss (mean over seeds)")
        ax.set_title(f"lr={block['base_lr']}  r={block['r_kap_loss']:+.3f}")

    fig.suptitle("Exp 3: κ vs final loss (corrected gain=1)", fontsize=11)
    plt.tight_layout()
    out = Path(__file__).parent / "empirical_kappa_exp3_correlation.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Correlation plot saved to {out}")


if __name__ == "__main__":
    main()
