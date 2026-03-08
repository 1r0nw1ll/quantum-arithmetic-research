#!/usr/bin/env python3
"""Empirical κ experiment — single-head self-attention layer (family [94]).

Design:
  - Synthetic sequence classification: B=128 sequences, L=8 tokens, d=16 dims
  - Single-head self-attention + mean-pool + linear classifier (pure numpy)
      Q=XW_Q, K=XW_K, V=XW_V
      Attn = softmax(QK^T / sqrt(d_k))
      ctx  = Attn @ V
      out  = ctx @ W_O          (output projection)
      logit = mean(out, axis=L) @ W_cls
      pred  = sigmoid(logit)
  - BCE loss, 8 conditions: 7 QA substrates + plain SGD
  - QA-modulated update: W -= lr * gain * H_QA * dW
      gain = min(||[dW_Q,dW_K,dW_V,dW_O,dW_cls]_flat||_2, 2.0)
      κ_t  = 1 - |1 - lr * gain * H_QA|
  - Results written to empirical_kappa_attention_results.json

Prediction: r(mean_κ, final_loss) < -0.70.
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
B          = 128     # batch (all sequences)
SEQ_LEN    = 8
D_MODEL    = 16
D_K        = 16      # key/query/value dim = D_MODEL (single head)
N_EPOCHS   = 300
BASE_LR    = 0.10
EPS        = 1e-12
GAIN_CAP   = 2.0
THRESH_ACC = 0.80

# ── Dataset ──────────────────────────────────────────────────────────────────

def make_dataset(rng: np.random.Generator) -> tuple:
    """Synthetic sequence classification: class determined by mean token sign."""
    half = B // 2
    # Class 0: tokens drawn from N(−0.5, 1), class 1: N(+0.5, 1)
    X0 = rng.normal(-0.5, 1.0, (half, SEQ_LEN, D_MODEL))
    X1 = rng.normal( 0.5, 1.0, (half, SEQ_LEN, D_MODEL))
    X  = np.concatenate([X0, X1], axis=0).astype(np.float64)
    # Normalise each feature across batch
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True) + EPS
    X  = (X - mu) / sd
    y  = np.array([0.0] * half + [1.0] * half).reshape(-1, 1)
    return X, y


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


# ── Attention forward / backward (pure numpy) ─────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce(pred, y):
    p = np.clip(pred, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def softmax(x):
    """Numerically stable softmax over last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=-1, keepdims=True) + EPS)


def attn_forward(X, W_Q, W_K, W_V, W_O, W_cls):
    """
    X:     B×L×D_MODEL
    Returns (Q, K, V, scores, attn, ctx, out, pooled, logits, pred)
    """
    scale = math.sqrt(D_K)
    Q     = X @ W_Q                              # B×L×D_K
    K     = X @ W_K                              # B×L×D_K
    V     = X @ W_V                              # B×L×D_K
    scores = Q @ K.transpose(0, 2, 1) / scale   # B×L×L
    attn   = softmax(scores)                     # B×L×L
    ctx    = attn @ V                            # B×L×D_K
    out    = ctx @ W_O                           # B×L×D_MODEL
    pooled = out.mean(axis=1)                    # B×D_MODEL
    logits = pooled @ W_cls                      # B×1
    pred   = sigmoid(logits)
    return Q, K, V, scores, attn, ctx, out, pooled, logits, pred


def attn_backward(X, W_Q, W_K, W_V, W_O, W_cls,
                  Q, K, V, scores, attn, ctx, out, pooled, pred, y):
    """Returns gradients (dW_Q, dW_K, dW_V, dW_O, dW_cls)."""
    n     = X.shape[0]
    scale = math.sqrt(D_K)

    # Classifier
    dlogits  = (pred - y) / n                            # B×1
    dW_cls   = pooled.T @ dlogits                        # D_MODEL×1
    dpooled  = dlogits @ W_cls.T                         # B×D_MODEL

    # Mean-pool backward: each of the L positions receives equal gradient
    dout = np.tile(dpooled[:, None, :], (1, SEQ_LEN, 1)) / SEQ_LEN  # B×L×D_MODEL

    # Output projection
    dW_O     = ctx.reshape(n * SEQ_LEN, D_K).T @ dout.reshape(n * SEQ_LEN, D_MODEL)
    dctx     = dout @ W_O.T                              # B×L×D_K

    # Attention
    dV       = attn.transpose(0, 2, 1) @ dctx           # B×L×D_K
    dattn    = dctx @ V.transpose(0, 2, 1)              # B×L×L
    # Softmax backward: dscores = attn * (dattn - (attn*dattn).sum(-1,k=True))
    dscores  = attn * (dattn - (attn * dattn).sum(axis=-1, keepdims=True))
    dscores  /= scale                                    # B×L×L

    dQ       = dscores @ K                              # B×L×D_K
    dK       = dscores.transpose(0, 2, 1) @ Q          # B×L×D_K

    Xf = X.reshape(n * SEQ_LEN, D_MODEL)
    dW_Q = Xf.T @ dQ.reshape(n * SEQ_LEN, D_K)         # D_MODEL×D_K
    dW_K = Xf.T @ dK.reshape(n * SEQ_LEN, D_K)         # D_MODEL×D_K
    dW_V = Xf.T @ dV.reshape(n * SEQ_LEN, D_K)         # D_MODEL×D_K

    return dW_Q, dW_K, dW_V, dW_O, dW_cls


def init_attn(seed_offset: int = 0):
    rng = np.random.default_rng(RNG_SEED + seed_offset)
    s   = math.sqrt(2.0 / D_MODEL)
    W_Q   = rng.normal(0, s, (D_MODEL, D_K))
    W_K   = rng.normal(0, s, (D_MODEL, D_K))
    W_V   = rng.normal(0, s, (D_MODEL, D_K))
    W_O   = rng.normal(0, s, (D_K, D_MODEL))
    W_cls = rng.normal(0, s, (D_MODEL, 1))
    return W_Q, W_K, W_V, W_O, W_cls


# ── Training ─────────────────────────────────────────────────────────────────

def train_attn(X, y, h_qa: float, seed_offset: int = 0) -> dict:
    plain_sgd = (h_qa == 0.0)
    W_Q, W_K, W_V, W_O, W_cls = init_attn(seed_offset)

    epoch_losses: list[float] = []
    epoch_accs:  list[float]  = []
    kappas:      list[float]  = []
    delta_losses: list[float] = []
    epochs_to_thresh          = None
    prev_loss: float | None   = None

    for epoch in range(N_EPOCHS):
        Q, K, V, scores, attn, ctx, out, pooled, logits, pred = \
            attn_forward(X, W_Q, W_K, W_V, W_O, W_cls)
        loss = bce(pred, y)

        dW_Q, dW_K, dW_V, dW_O, dW_cls = attn_backward(
            X, W_Q, W_K, W_V, W_O, W_cls,
            Q, K, V, scores, attn, ctx, out, pooled, pred, y)

        grad_flat = np.concatenate([
            dW_Q.ravel(), dW_K.ravel(), dW_V.ravel(),
            dW_O.ravel(), dW_cls.ravel()
        ])

        if plain_sgd:
            eff_gain = 1.0; eff_hqa = 1.0
        else:
            eff_gain = grad_gain(grad_flat); eff_hqa = h_qa

        k = kappa(BASE_LR, eff_gain, eff_hqa)
        kappas.append(k)
        if prev_loss is not None:
            delta_losses.append(prev_loss - loss)
        prev_loss = loss

        scale  = BASE_LR * eff_gain * eff_hqa
        W_Q   -= scale * dW_Q
        W_K   -= scale * dW_K
        W_V   -= scale * dW_V
        W_O   -= scale * dW_O
        W_cls -= scale * dW_cls

        # End-of-epoch evaluation
        _, _, _, _, _, _, _, _, _, pred_all = \
            attn_forward(X, W_Q, W_K, W_V, W_O, W_cls)
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
        "mean_kappa":            mean_kappa,
        "pearson_r_kappa_dloss": r,
        "final_loss":            epoch_losses[-1],
        "final_acc":             epoch_accs[-1],
        "min_loss":              min(epoch_losses),
        "epochs_to_thresh":      epochs_to_thresh if epochs_to_thresh is not None else -1,
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
    rng  = np.random.default_rng(RNG_SEED)
    X, y = make_dataset(rng)
    print(f"Dataset: {B} sequences × {SEQ_LEN} tokens × {D_MODEL} dims, binary")
    print(f"Architecture: single-head self-attention (d_k={D_K}) + mean-pool + linear")
    print(f"Base LR: {BASE_LR}, Epochs: {N_EPOCHS}")
    print(f"QA update: W -= lr * gain * H_QA * dW;  gain = min(||dW||, {GAIN_CAP})")
    print()

    results = []
    for i, (label, b, e) in enumerate(SUBSTRATES):
        if b == 0 and e == 0:
            h_qa_val    = 0.0
            display_hqa = "1.0 (fixed)"
        else:
            h_qa_val    = compute_h_qa(float(b), float(e))
            display_hqa = f"{h_qa_val:.6f}"

        print(f"[{i+1}/{len(SUBSTRATES)}] {label}  H_QA={display_hqa} ...", end=" ", flush=True)
        res = train_attn(X, y, h_qa_val, seed_offset=i)
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
        "experiment": "empirical_kappa_attention_v1",
        "architecture": "single-head self-attention + mean-pool + linear classifier",
        "batch_size": B, "seq_len": SEQ_LEN, "d_model": D_MODEL, "d_k": D_K,
        "n_epochs": N_EPOCHS, "base_lr": BASE_LR, "gain_cap": GAIN_CAP,
        "thresh_acc": THRESH_ACC, "rng_seed": RNG_SEED,
        "r_kappa_loss": r_kap_loss,
        "r_kappa_acc":  r_kap_acc,
        "r_hqa_acc":    r_hqa_acc,
        "conditions": results,
    }
    out_path = Path(__file__).parent / "empirical_kappa_attention_results.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
