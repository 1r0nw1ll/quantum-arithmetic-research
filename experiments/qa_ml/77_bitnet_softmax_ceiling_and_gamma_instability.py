#!/usr/bin/env python3
"""
Two sequential negative-but-diagnosed findings on top of fixed-point Adam
(76_bitnet_fixed_point_adam.py, 91.1% with MSE loss). See
docs/specs/QA_ML_BITNET_NT_COMPLIANT_TRAINING_FINDINGS.md sections 7-8 for
full narrative. The final, fixed, cross-validated result is in
78_bitnet_final_nt_compliant_training.py -- this file documents the two
problems found on the way there.

PART A -- real cross-entropy hits a structural expressivity ceiling.
Implements exp() via range-reduction (divide by 2^6) + 10-term truncated
Taylor series with exact 1/k! coefficients (Fraction-computed once) +
repeated squaring -- accurate to ~1e-6 relative error over the range
softmax logits produce. Only forward exp() is needed since the softmax +
cross-entropy gradient has the closed form probs - onehot (no fixed-point
log() required). Swapping MSE for this real cross-entropy DROPS accuracy to
~69%. Diagnosis: train_acc tracks test_acc throughout (rules out
overfitting); mean max-softmax-probability stays pinned near 0.144 --
barely above the 0.10 chance floor -- for the full run. The ternary output
layer's logit range is structurally too coarse (bounded fan-in-normalized
sum of +-1/0 terms) to satisfy cross-entropy's preference for confident,
sharply-peaked predictions.

PART B -- adding BitNet's gamma (absmean) output rescale fixes the ceiling
on digits (reaches ~96.6%, beating the float baseline) but is UNSTABLE: a
generalization check on MNIST-6k (784->128->10) shows gamma1 growing
~1500x over ~24 epochs via an unbounded positive-feedback loop (gamma
rescales the gradient flowing into the layer it's derived from -> larger
gradient -> faster-growing latent -> larger gamma, uncapped). Accuracy
peaks ~93% then collapses to chance within a single epoch. Left
unresolved here on purpose -- 78 diagnoses and fixes it (spoiler: it
wasn't the gamma-gradient path, it was a missing AdamW-style weight decay
term).

QA_COMPLIANCE = "bitnet_softmax_ceiling_and_gamma_instability - real softmax-CE via integer exp() diagnosed expressivity ceiling; gamma rescale fixes it on digits but is unstable, documented before the fix"
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SCALE = 1 << 24
BATCH = 32

BETA1_REAL = 0.9
BETA2_REAL = 0.999
EPS_REAL = 1e-6
BETA1_FP = round(BETA1_REAL * SCALE)
BETA2_FP = round(BETA2_REAL * SCALE)
EPS_FP = max(round(EPS_REAL * SCALE), 1)
UPDATE_CLIP_FP = round(4.0 * SCALE)
V_FLOOR_FP = max(round(1e-7 * SCALE), 1)

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


def ternary_quant(latent):
    gamma = gamma_of(latent)
    ratio = fp_div_pos(latent, np.full_like(latent, gamma))
    sign = np.sign(ratio)
    rounded = sign * ((np.abs(ratio) + SCALE // 2) // SCALE)
    return np.clip(rounded, -1, 1).astype(np.int64)


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


def adam_step(latent, m, v, dW, beta1_pow, beta2_pow, lr_fp):
    bias1_denom = max(SCALE - beta1_pow, 1)
    bias2_denom = max(SCALE - beta2_pow, 1)
    m = fp_mul(np.full_like(m, BETA1_FP), m) + fp_mul(np.full_like(dW, SCALE - BETA1_FP), dW)
    v = fp_mul(np.full_like(v, BETA2_FP), v) + fp_mul(np.full_like(dW, SCALE - BETA2_FP), fp_mul(dW, dW))
    m_hat = (m.astype(np.int64) * SCALE) // bias1_denom
    v_hat = np.maximum((v.astype(np.int64) * SCALE) // bias2_denom, V_FLOOR_FP)
    denom = sqrt_fp(v_hat) + EPS_FP
    update = np.clip(fp_div_pos(m_hat, denom), -UPDATE_CLIP_FP, UPDATE_CLIP_FP)
    latent = latent - fp_mul(update, lr_fp)
    return latent, m, v


# ---------- Part A: no-gamma ceiling ----------

def compute_grads_no_gamma(xb, yb, latent1, latent2, bs):
    W1_q = ternary_quant(latent1)
    W2_q = ternary_quant(latent2)
    F1, F2 = W1_q.shape[0], W2_q.shape[0]
    h_pre = (xb @ W1_q) // F1
    h = np.maximum(h_pre, 0)
    logits = (h @ W2_q) // F2
    probs = softmax_probs(logits)
    dlogits = probs - yb
    dh = (dlogits @ W2_q.T) // F2
    dh_pre = dh * (h_pre > 0)
    dW2 = (fp_matmul(h.T, dlogits) // F2) // bs
    dW1 = (fp_matmul(xb.T, dh_pre) // F1) // bs
    return dW1, dW2, h_pre, logits


def run_no_gamma(X_tr, y_tr, X_te, y_te, seed, n_classes, lr_real, epochs):
    rng = np.random.default_rng(seed)
    init_span = SCALE // 20
    latent1 = rng.integers(-init_span, init_span + 1, size=(64, 32), dtype=np.int64)
    latent2 = rng.integers(-init_span, init_span + 1, size=(32, n_classes), dtype=np.int64)
    m1 = np.zeros_like(latent1); v1 = np.zeros_like(latent1)
    m2 = np.zeros_like(latent2); v2 = np.zeros_like(latent2)
    beta1_pow = SCALE; beta2_pow = SCALE
    lr_fp = int(round(lr_real * SCALE))
    X_tr_fp, X_te_fp = to_fixed(X_tr), to_fixed(X_te)
    onehot_tr = np.eye(n_classes, dtype=np.int64)[y_tr] * SCALE
    n = X_tr.shape[0]
    curve, max_probs = [], []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            xb, yb = X_tr_fp[idx], onehot_tr[idx]
            bs = xb.shape[0]
            dW1, dW2, _, _ = compute_grads_no_gamma(xb, yb, latent1, latent2, bs)
            beta1_pow = (beta1_pow * BETA1_FP) // SCALE
            beta2_pow = (beta2_pow * BETA2_FP) // SCALE
            latent1, m1, v1 = adam_step(latent1, m1, v1, dW1, beta1_pow, beta2_pow, lr_fp)
            latent2, m2, v2 = adam_step(latent2, m2, v2, dW2, beta1_pow, beta2_pow, lr_fp)
        W1_q, W2_q = ternary_quant(latent1), ternary_quant(latent2)
        h_te = np.maximum((X_te_fp @ W1_q) // 64, 0)
        logits_te = (h_te @ W2_q) // 32
        probs_te = softmax_probs(logits_te)
        pred = np.argmax(logits_te, axis=1)
        curve.append(float(np.mean(pred == y_te)))
        max_probs.append(float(np.mean(np.max(probs_te, axis=1))) / SCALE)
    return curve, max_probs


# ---------- Part B: gamma added, unstable (stop-gradient gamma) ----------

def compute_grads_gamma(xb, yb, latent1, latent2, bs):
    gamma1, gamma2 = gamma_of(latent1), gamma_of(latent2)
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
    return dW1, dW2, gamma1, gamma2


def run_gamma_unstable(X_tr, y_tr, X_te, y_te, seed, n_classes, lr_real, epochs, in_dim, hidden):
    rng = np.random.default_rng(seed)
    init_span = SCALE // 20
    latent1 = rng.integers(-init_span, init_span + 1, size=(in_dim, hidden), dtype=np.int64)
    latent2 = rng.integers(-init_span, init_span + 1, size=(hidden, n_classes), dtype=np.int64)
    m1 = np.zeros_like(latent1); v1 = np.zeros_like(latent1)
    m2 = np.zeros_like(latent2); v2 = np.zeros_like(latent2)
    beta1_pow = SCALE; beta2_pow = SCALE
    lr_fp = int(round(lr_real * SCALE))
    X_tr_fp, X_te_fp = to_fixed(X_tr), to_fixed(X_te)
    onehot_tr = np.eye(n_classes, dtype=np.int64)[y_tr] * SCALE
    n = X_tr.shape[0]
    curve, gamma1_trace = [], []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            xb, yb = X_tr_fp[idx], onehot_tr[idx]
            bs = xb.shape[0]
            dW1, dW2, g1, g2 = compute_grads_gamma(xb, yb, latent1, latent2, bs)
            beta1_pow = (beta1_pow * BETA1_FP) // SCALE
            beta2_pow = (beta2_pow * BETA2_FP) // SCALE
            latent1, m1, v1 = adam_step(latent1, m1, v1, dW1, beta1_pow, beta2_pow, lr_fp)
            latent2, m2, v2 = adam_step(latent2, m2, v2, dW2, beta1_pow, beta2_pow, lr_fp)
        g1 = gamma_of(latent1)
        W1_q = ternary_quant_with_gamma(latent1, g1)
        g2 = gamma_of(latent2)
        W2_q = ternary_quant_with_gamma(latent2, g2)
        h_te = np.maximum(fp_mul_scalar(X_te_fp @ W1_q, g1) // in_dim, 0)
        logits_te = fp_mul_scalar(h_te @ W2_q, g2) // hidden
        pred = np.argmax(logits_te, axis=1)
        curve.append(float(np.mean(pred == y_te)))
        gamma1_trace.append(g1 / SCALE)
    return curve, gamma1_trace


def load_mnist_subsample(n_train=6000, n_test=2000, seed=0):
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    X, y = X.astype(np.int64), y.astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])[: n_train + n_test]
    X, y = X[idx], y[idx]
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def main():
    digits = load_digits()
    X, y = digits.data.astype(np.int64), digits.target.astype(np.int64)
    n_classes = 10
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("=== Part A: no-gamma cross-entropy ceiling (digits, single seed) ===")
    curve, max_probs = run_no_gamma(X_tr, y_tr, X_te, y_te, 42, n_classes, 0.0005, epochs=90)
    print("  test_acc every 15 epochs:  " + " ".join(f"{v:.3f}" for v in curve[::15]))
    print("  mean_max_prob every 15ep:  " + " ".join(f"{v:.3f}" for v in max_probs[::15]))
    print("  (pinned near 0.10-0.15 chance floor throughout -- expressivity ceiling, not underfitting)\n")

    print("=== Part B: gamma added, unstable (MNIST-6k, single seed, lr=0.1) ===")
    Xm_tr, Xm_te, ym_tr, ym_te = load_mnist_subsample(6000, 2000, seed=42)
    curve, gamma1_trace = run_gamma_unstable(Xm_tr, ym_tr, Xm_te, ym_te, 42, n_classes, 0.1, epochs=25, in_dim=784, hidden=128)
    print("  test_acc every epoch (0-24): " + " ".join(f"{v:.3f}" for v in curve))
    print("  gamma1 every epoch (0-24):   " + " ".join(f"{v:.2f}" for v in gamma1_trace))
    print("  (peaks ~epoch 8-11 then collapses as gamma1 runs away -- see 78 for the fix)")


if __name__ == "__main__":
    main()
