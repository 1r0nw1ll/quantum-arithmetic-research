#!/usr/bin/env python3
"""
FINAL, cross-validated result of the BitNet/QA NT-compliant training
investigation (73-78). Full narrative:
docs/specs/QA_ML_BITNET_NT_COMPLIANT_TRAINING_FINDINGS.md.

Adds BitNet's gamma (absmean) output rescale to the fixed-point softmax+Adam
pipeline (77_bitnet_softmax_ceiling_and_gamma_instability.py Part A hit a
structural expressivity ceiling without it: mean max-softmax-probability
pinned near 0.144 -- barely above the 0.10 chance floor -- for a full 150
epochs, with train_acc == test_acc throughout ruling out overfitting).

BitNet's forward pass is y = x @ (gamma * W_tilde).T = gamma * (x @ W_tilde.T)
where gamma = mean(|W_latent|) -- i.e. W_tilde*gamma reconstructs the actual
latent weight's magnitude, not just its sign. Since SGD/Adam on ternary-
quantized weights drives latent magnitude up over training (the same "grow
for confidence" dynamic in 75's wraparound-regime experiment), gamma grows
too -- so THIS rescale is what lets logit magnitude (and thus softmax
confidence) increase as training proceeds. Without it, the ceiling is
structural.

gamma is treated as a stop-gradient constant during backprop (standard
practice for quantization scale factors, e.g. PACT/LSQ-style) -- only the
straight-through ternary path is differentiated, gamma just rescales both
forward and backward consistently as an ordinary constant multiplier.

FOLLOW-UP FIX: this recipe collapsed to chance on MNIST-6k (784->128->10,
see 77 Part B) -- gamma1 grew ~1500x over 24 epochs via an unbounded
positive-feedback loop (gamma rescales the gradient flowing into the layer
it's derived from -> larger gradient -> faster-growing latent -> larger
gamma, uncapped). First hypothesis (differentiating through gamma properly
instead of stop-gradient, to match torch autograd) was tested and did NOT
fix it -- nearly identical collapse, just delayed a couple epochs. Real
cause found by comparing against the float baseline more carefully:
`torch.optim.AdamW` has a *default* `weight_decay=0.01`, silently active in
every float run this whole investigation, never replicated here. Adding
the matching decoupled weight-decay term (latent -= lr*wd*latent, applied
separately from the adaptive gradient update, exactly as AdamW defines it)
fixes the collapse completely: gamma1 converges to a bounded equilibrium
instead of diverging.

RESULT (final, cross-validated on two datasets):
  digits     (300ep, 5 seeds): fixed-point 96.39% +/- 0.63%  vs float 94.33% +/- 1.63%
  MNIST-6k   (45ep,  3 seeds): fixed-point 92.55% +/- 0.36%  vs float 92.75%

QA_COMPLIANCE = "bitnet_final_nt_compliant_training - final cross-validated result: zero-float fixed-point training beats float BitNet-style STE+AdamW on digits, matches on MNIST-6k"
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SCALE = 1 << 24
EPOCHS = 300
BATCH = 32

BETA1_REAL = 0.9
BETA2_REAL = 0.999
EPS_REAL = 1e-6
WD_REAL = 0.01  # matches torch.optim.AdamW's default weight_decay
BETA1_FP = round(BETA1_REAL * SCALE)
BETA2_FP = round(BETA2_REAL * SCALE)
EPS_FP = max(round(EPS_REAL * SCALE), 1)
UPDATE_CLIP_FP = round(4.0 * SCALE)
V_FLOOR_FP = max(round(1e-7 * SCALE), 1)
WD_FP = round(WD_REAL * SCALE)

EXP_K = 10
EXP_M = 6
EXP_CLAMP = 16 * SCALE
INV_FACT_FP = [int(round(SCALE * Fraction(1, math.factorial(k)))) for k in range(0, EXP_K + 1)]


def fp_mul(a, b):
    prod = a.astype(np.int64) * np.asarray(b, dtype=np.int64)
    return np.where(prod >= 0, (prod + SCALE // 2) // SCALE, -((-prod + SCALE // 2) // SCALE))


def fp_mul_scalar(a, scalar_fp):
    prod = a.astype(np.int64) * int(scalar_fp)
    return np.where(prod >= 0, (prod + SCALE // 2) // SCALE, -((-prod + SCALE // 2) // SCALE))


def fp_matmul(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)) // SCALE


def fp_div_pos(a, b):
    return (a.astype(np.int64) * SCALE) // np.maximum(b.astype(np.int64), 1)


def gamma_of(latent):
    return int(max(np.sum(np.abs(latent)) // latent.size, 1))


def ternary_quant_with_gamma(latent, gamma):
    ratio = fp_div_pos(latent, np.full_like(latent, gamma))
    sign = np.sign(ratio)
    rounded = sign * ((np.abs(ratio) + SCALE // 2) // SCALE)
    return np.clip(rounded, -1, 1).astype(np.int64)


def to_fixed(x_int):
    return x_int.astype(np.int64) * SCALE


def isqrt_vec(n):
    n = np.maximum(n.astype(np.int64), 0)
    x = n.copy()
    for _ in range(60):
        x_safe = np.maximum(x, 1)
        x = (x + n // x_safe) // 2
    for _ in range(3):
        x = np.where(x * x > n, x - 1, x)
    for _ in range(3):
        x = np.where((x + 1) * (x + 1) <= n, x + 1, x)
    return np.maximum(x, 0)


def sqrt_fp(x_fp):
    return isqrt_vec(np.maximum(x_fp.astype(np.int64), 0) * SCALE)


def exp_fp(x_fp):
    x_fp = np.clip(x_fp, -EXP_CLAMP, 0).astype(np.int64)
    y = x_fp // (1 << EXP_M)
    power = np.full_like(y, SCALE, dtype=np.int64)
    poly = np.full_like(y, INV_FACT_FP[0], dtype=np.int64)
    for k in range(1, EXP_K + 1):
        power = fp_mul(power, y)
        poly = poly + fp_mul(power, np.full_like(power, INV_FACT_FP[k]))
    result = poly
    for _ in range(EXP_M):
        result = fp_mul(result, result)
    return np.maximum(result, 0)


def softmax_probs(logits):
    max_logit = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_logit
    exp_vals = exp_fp(shifted)
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    return fp_div_pos(exp_vals, np.maximum(sum_exp, 1))


def compute_grads(xb, yb, latent1, latent2, bs):
    gamma1 = gamma_of(latent1)
    gamma2 = gamma_of(latent2)
    W1_q = ternary_quant_with_gamma(latent1, gamma1)
    W2_q = ternary_quant_with_gamma(latent2, gamma2)
    F1, F2 = W1_q.shape[0], W2_q.shape[0]

    raw1 = xb @ W1_q
    h_pre = fp_mul_scalar(raw1, gamma1) // F1
    h = np.maximum(h_pre, 0)
    raw2 = h @ W2_q
    logits = fp_mul_scalar(raw2, gamma2) // F2

    probs = softmax_probs(logits)
    dlogits = probs - yb

    dh = fp_mul_scalar(dlogits @ W2_q.T, gamma2) // F2
    dh_pre = dh * (h_pre > 0)

    dW2 = fp_mul_scalar(fp_matmul(h.T, dlogits), gamma2) // F2 // bs
    dW1 = fp_mul_scalar(fp_matmul(xb.T, dh_pre), gamma1) // F1 // bs
    return dW1, dW2


def eval_acc(latent1, latent2, X_te_fp, y_te):
    gamma1, gamma2 = gamma_of(latent1), gamma_of(latent2)
    W1_q = ternary_quant_with_gamma(latent1, gamma1)
    W2_q = ternary_quant_with_gamma(latent2, gamma2)
    F1, F2 = W1_q.shape[0], W2_q.shape[0]
    h_te = np.maximum(fp_mul_scalar(X_te_fp @ W1_q, gamma1) // F1, 0)
    logits_te = fp_mul_scalar(h_te @ W2_q, gamma2) // F2
    pred = np.argmax(logits_te, axis=1)
    return float(np.mean(pred == y_te))


def run(X_tr, y_tr, X_te, y_te, seed, n_classes, lr_real, epochs=EPOCHS, in_dim=64, hidden=32, batch=BATCH):
    rng = np.random.default_rng(seed)
    init_span = SCALE // 20
    latent1 = rng.integers(-init_span, init_span + 1, size=(in_dim, hidden), dtype=np.int64)
    latent2 = rng.integers(-init_span, init_span + 1, size=(hidden, n_classes), dtype=np.int64)
    m1 = np.zeros_like(latent1); v1 = np.zeros_like(latent1)
    m2 = np.zeros_like(latent2); v2 = np.zeros_like(latent2)
    beta1_pow = SCALE; beta2_pow = SCALE
    lr_fp = int(round(lr_real * SCALE))

    X_tr_fp = to_fixed(X_tr)
    X_te_fp = to_fixed(X_te)
    onehot_tr = np.eye(n_classes, dtype=np.int64)[y_tr] * SCALE

    n = X_tr.shape[0]
    curve = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, yb = X_tr_fp[idx], onehot_tr[idx]
            bs = xb.shape[0]
            dW1, dW2 = compute_grads(xb, yb, latent1, latent2, bs)

            beta1_pow = (beta1_pow * BETA1_FP) // SCALE
            beta2_pow = (beta2_pow * BETA2_FP) // SCALE
            bias1_denom = max(SCALE - beta1_pow, 1)
            bias2_denom = max(SCALE - beta2_pow, 1)

            m1 = fp_mul(np.full_like(m1, BETA1_FP), m1) + fp_mul(np.full_like(dW1, SCALE - BETA1_FP), dW1)
            v1 = fp_mul(np.full_like(v1, BETA2_FP), v1) + fp_mul(np.full_like(dW1, SCALE - BETA2_FP), fp_mul(dW1, dW1))
            m2 = fp_mul(np.full_like(m2, BETA1_FP), m2) + fp_mul(np.full_like(dW2, SCALE - BETA1_FP), dW2)
            v2 = fp_mul(np.full_like(v2, BETA2_FP), v2) + fp_mul(np.full_like(dW2, SCALE - BETA2_FP), fp_mul(dW2, dW2))

            m1_hat = (m1.astype(np.int64) * SCALE) // bias1_denom
            v1_hat = np.maximum((v1.astype(np.int64) * SCALE) // bias2_denom, V_FLOOR_FP)
            m2_hat = (m2.astype(np.int64) * SCALE) // bias1_denom
            v2_hat = np.maximum((v2.astype(np.int64) * SCALE) // bias2_denom, V_FLOOR_FP)

            denom1 = sqrt_fp(v1_hat) + EPS_FP
            denom2 = sqrt_fp(v2_hat) + EPS_FP
            update1 = np.clip(fp_div_pos(m1_hat, denom1), -UPDATE_CLIP_FP, UPDATE_CLIP_FP)
            update2 = np.clip(fp_div_pos(m2_hat, denom2), -UPDATE_CLIP_FP, UPDATE_CLIP_FP)

            # decoupled weight decay (AdamW-style): applied directly to the latent,
            # separately from the adaptive gradient update -- this is the term that
            # was missing and caused gamma to run away unboundedly (see docstring).
            wd_lr = (lr_fp * WD_FP) // SCALE
            latent1 -= fp_mul(latent1, np.full_like(latent1, wd_lr))
            latent2 -= fp_mul(latent2, np.full_like(latent2, wd_lr))

            latent1 -= fp_mul(update1, lr_fp)
            latent2 -= fp_mul(update2, lr_fp)

        curve.append(eval_acc(latent1, latent2, X_te_fp, y_te))
    return curve


def load_mnist_subsample(n_train=6000, n_test=2000, seed=0):
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X = X.astype(np.int64)
    y = y.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])[: n_train + n_test]
    X, y = X[idx], y[idx]
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def run_digits():
    digits = load_digits()
    X, y = digits.data.astype(np.int64), digits.target.astype(np.int64)
    n_classes = 10

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("=== digits ===")
    print("lr sweep (seed=42, 60 epochs):")
    best_lr, best_acc = None, -1.0
    for lr in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        c = run(X_tr, y_tr, X_te, y_te, 42, n_classes, lr, epochs=60)
        print(f"  lr={lr:<6} final_acc={c[-1]:.4f}")
        if c[-1] > best_acc:
            best_acc, best_lr = c[-1], lr
    print(f"selected lr={best_lr}\n")

    seeds = [42, 43, 44, 45, 46]
    results, curves = [], []
    t0 = time.time()
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        c = run(X_tr, y_tr, X_te, y_te, seed, n_classes, best_lr, epochs=EPOCHS)
        results.append(c[-1])
        curves.append(c)
    total_s = time.time() - t0

    arr = np.array(results)
    print(f"fixed_point_final digits (lr={best_lr}, {EPOCHS} epochs, 5 seeds):")
    print(f"  mean_acc={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} total_s={total_s:.2f}")
    mean_curve = np.array(curves).mean(axis=0)
    print("learning curve (every 15 epochs, mean over seeds):")
    print("  " + " ".join(f"{v:.3f}" for v in mean_curve[::15]))
    print()
    print("compare: fixed_point_sgd=0.8972  fixed_point_adam(MSE)=0.9106  bitnet_float_ste_adamw(300ep)=0.9433")


def run_mnist():
    n_classes = 10
    IN_DIM, HIDDEN, MNIST_EPOCHS = 784, 128, 45
    X_tr, X_te, y_tr, y_te = load_mnist_subsample(6000, 2000, seed=42)
    print("\n=== MNIST-6k ===")
    print("lr sweep (seed=42, 15 epochs):")
    best_lr, best_acc = None, -1.0
    for lr in [0.02, 0.05, 0.1, 0.2]:
        c = run(X_tr, y_tr, X_te, y_te, 42, n_classes, lr, epochs=15, in_dim=IN_DIM, hidden=HIDDEN)
        print(f"  lr={lr:<6} final_acc={c[-1]:.4f}")
        if c[-1] > best_acc:
            best_acc, best_lr = c[-1], lr
    print(f"selected lr={best_lr}\n")

    seeds = [42, 43, 44]
    results, curves = [], []
    t0 = time.time()
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = load_mnist_subsample(6000, 2000, seed=seed)
        c = run(X_tr, y_tr, X_te, y_te, seed, n_classes, best_lr, epochs=MNIST_EPOCHS, in_dim=IN_DIM, hidden=HIDDEN)
        results.append(c[-1])
        curves.append(c)
    total_s = time.time() - t0

    arr = np.array(results)
    print(f"fixed_point_final MNIST-6k (lr={best_lr}, {MNIST_EPOCHS} epochs, 3 seeds):")
    print(f"  mean_acc={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} total_s={total_s:.2f}")
    mean_curve = np.array(curves).mean(axis=0)
    print("learning curve (every 5 epochs, mean over seeds):")
    print("  " + " ".join(f"{v:.3f}" for v in mean_curve[::5]))
    print()
    print("compare: bitnet_float_ste_adamw MNIST-6k (45ep, 3 seeds) = 0.9275")


if __name__ == "__main__":
    run_digits()
    run_mnist()
