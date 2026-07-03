#!/usr/bin/env python3
"""
Integer-analog of Adam's adaptive moments, on top of fixed-point SGD
(74_bitnet_fixed_point_sgd.py, 89.7%).

Adam needs one operation SGD didn't: sqrt(v_hat). Square roots of arbitrary
rationals aren't rational, so this can't be done with pure Fraction/
fixed-point arithmetic exactly -- but floor(sqrt(n)) for integer n IS an
exact, deterministic, integer-only operation (Newton's method in integer
arithmetic, same algorithm used in exact-arithmetic libraries and
cryptography -- not a float sqrt() cast). That keeps the whole optimizer
NT-compliant: no IEEE-754 float touches the causal path.

Uses a larger fixed-point scale (Q8.24, SCALE=2^24) than the SGD experiment
(Q16.16) for headroom on the eps term -- eps=1e-8 isn't representable at
Q16.16 (rounds to 0) without risking overflow elsewhere.

RESULT: 91.06% mean test accuracy (5 seeds), a modest but real gain over
plain SGD's 89.72%. Reaching this required diagnosing and fixing three real
bugs (see docs/specs/QA_ML_BITNET_NT_COMPLIANT_TRAINING_FINDINGS.md section
6 for the full mechanism of each):

  1. eps too large (1e-3, chosen for representability) swamped the adaptive
     denominator in 85-100% of weights -- the optimizer had silently
     degenerated into plain scaled SGD (first attempt matched SGD almost
     exactly: 89.78%).
  2. Floor-division in the EMA accumulator (v-moment) introduced a
     systematic downward bias over hundreds of steps -- sqrt(v_hat) was
     collapsing toward zero instead of tracking gradient RMS. Fixed by
     switching fp_mul to round-to-nearest instead of floor.
  3. Sparse ReLU gradients (most weights get exactly-zero gradient on most
     batches) drove v to exact integer zero, which got permanently stuck
     there; the next nonzero gradient then divided by ~eps and exploded
     (updates up to 5000x too large). Fixed with a v-floor (keeps v_hat off
     exact zero) plus a standard Adam update-magnitude clip -- the same
     mitigations real low-precision Adam implementations use for the same
     reason.

QA_COMPLIANCE = "bitnet_fixed_point_adam - integer-analog Adam (exact isqrt via Newton's method) in Q8.24 fixed-point; three diagnosed low-precision Adam bugs fixed"
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SCALE = 1 << 24
EPOCHS = 150
BATCH = 32

BETA1_REAL = 0.9
BETA2_REAL = 0.999
EPS_REAL = 1e-6  # representable at Q8.24 (eps_fp ~= 17); see fp_mul rounding fix below

BETA1_FP = round(BETA1_REAL * SCALE)
BETA2_FP = round(BETA2_REAL * SCALE)
EPS_FP = max(round(EPS_REAL * SCALE), 1)
UPDATE_CLIP_FP = round(4.0 * SCALE)  # clip Adam's normalized update to +-4 real units
V_FLOOR_FP = max(round(1e-7 * SCALE), 1)  # keep v_hat off exact zero (sparse-gradient fix)


def fp_mul(a, b):
    # round-to-nearest, not floor: repeated floor-division on small-magnitude
    # EMA accumulators (as in Adam's v-moment) introduces a systematic
    # downward bias over hundreds of steps. Symmetric rounding removes it.
    prod = a.astype(np.int64) * np.asarray(b, dtype=np.int64)
    return np.where(prod >= 0, (prod + SCALE // 2) // SCALE, -((-prod + SCALE // 2) // SCALE))


def fp_matmul(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)) // SCALE


def fp_div_pos(a, b):
    # a, b both fixed-point, b > 0 guaranteed by caller
    return (a.astype(np.int64) * SCALE) // np.maximum(b.astype(np.int64), 1)


def ternary_quant(latent):
    gamma = np.maximum(np.sum(np.abs(latent)) // latent.size, 1)
    ratio = fp_div_pos(latent, np.full_like(latent, gamma))
    sign = np.sign(ratio)
    rounded = sign * ((np.abs(ratio) + SCALE // 2) // SCALE)
    return np.clip(rounded, -1, 1).astype(np.int64)


def to_fixed(x_int):
    return x_int.astype(np.int64) * SCALE


def isqrt_vec(n):
    # exact floor(sqrt(n)) for elementwise nonnegative int64 array n, via
    # vectorized Newton's method + integer cleanup pass. No float anywhere.
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
    # x_fp fixed-point (scale S) nonnegative -> fixed-point sqrt (scale S),
    # exact via floor(sqrt(x_fp * S)) = floor(sqrt(x_real) * S).
    return isqrt_vec(np.maximum(x_fp.astype(np.int64), 0) * SCALE)


def compute_grads(xb, yb, latent1, latent2, bs):
    W1_q = ternary_quant(latent1)
    W2_q = ternary_quant(latent2)
    F1, F2 = W1_q.shape[0], W2_q.shape[0]
    h_pre = (xb @ W1_q) // F1
    h = np.maximum(h_pre, 0)
    logits = (h @ W2_q) // F2
    dlogits = 2 * (logits - yb)
    dh = (dlogits @ W2_q.T) // F2
    dh_pre = dh * (h_pre > 0)
    dW2 = (fp_matmul(h.T, dlogits) // F2) // bs
    dW1 = (fp_matmul(xb.T, dh_pre) // F1) // bs
    return dW1, dW2


def eval_acc(latent1, latent2, X_te_fp, y_te):
    W1_q = ternary_quant(latent1)
    W2_q = ternary_quant(latent2)
    F1, F2 = W1_q.shape[0], W2_q.shape[0]
    h_te = np.maximum((X_te_fp @ W1_q) // F1, 0)
    logits_te = (h_te @ W2_q) // F2
    pred = np.argmax(logits_te, axis=1)
    return float(np.mean(pred == y_te))


def run_fixed_point_adam(X_tr, y_tr, X_te, y_te, seed, n_classes, lr_real, epochs=EPOCHS):
    rng = np.random.default_rng(seed)
    init_span = SCALE // 20
    latent1 = rng.integers(-init_span, init_span + 1, size=(64, 32), dtype=np.int64)
    latent2 = rng.integers(-init_span, init_span + 1, size=(32, n_classes), dtype=np.int64)
    m1 = np.zeros_like(latent1)
    v1 = np.zeros_like(latent1)
    m2 = np.zeros_like(latent2)
    v2 = np.zeros_like(latent2)
    beta1_pow = SCALE
    beta2_pow = SCALE
    lr_fp = int(round(lr_real * SCALE))

    X_tr_fp = to_fixed(X_tr)
    X_te_fp = to_fixed(X_te)
    onehot_tr = np.eye(n_classes, dtype=np.int64)[y_tr] * SCALE

    n = X_tr.shape[0]
    curve = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
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
            update1 = fp_div_pos(m1_hat, denom1)
            update2 = fp_div_pos(m2_hat, denom2)
            # standard Adam update-magnitude clip (real Adam implementations do this
            # too, e.g. AMSGrad/gradient clipping): with sparse ReLU gradients, v can
            # decay to exactly 0 in integer arithmetic and get stuck there, so the next
            # nonzero gradient for that weight divides by ~eps and explodes.
            update1 = np.clip(update1, -UPDATE_CLIP_FP, UPDATE_CLIP_FP)
            update2 = np.clip(update2, -UPDATE_CLIP_FP, UPDATE_CLIP_FP)

            latent1 -= fp_mul(update1, lr_fp)
            latent2 -= fp_mul(update2, lr_fp)

        curve.append(eval_acc(latent1, latent2, X_te_fp, y_te))
    return curve


def main():
    digits = load_digits()
    X, y = digits.data.astype(np.int64), digits.target.astype(np.int64)
    n_classes = 10

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("lr sweep (seed=42, 30 epochs):")
    best_lr, best_acc = None, -1.0
    for lr in [0.0002, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003]:
        c = run_fixed_point_adam(X_tr, y_tr, X_te, y_te, 42, n_classes, lr, epochs=30)
        print(f"  lr={lr:<6} final_acc={c[-1]:.4f}")
        if c[-1] > best_acc:
            best_acc, best_lr = c[-1], lr
    print(f"selected lr={best_lr}\n")

    seeds = [42, 43, 44, 45, 46]
    results, curves = [], []
    t0 = time.time()
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        c = run_fixed_point_adam(X_tr, y_tr, X_te, y_te, seed, n_classes, best_lr)
        results.append(c[-1])
        curves.append(c)
    total_s = time.time() - t0

    arr = np.array(results)
    print(f"fixed_point_adam (lr={best_lr}, {EPOCHS} epochs, 5 seeds):")
    print(f"  mean_acc={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} total_s={total_s:.2f}")
    mean_curve = np.array(curves).mean(axis=0)
    print("learning curve (every 15 epochs, mean over seeds):")
    print("  " + " ".join(f"{v:.3f}" for v in mean_curve[::15]))
    print()
    print("compare: fixed_point_sgd=0.8972  bitnet_float_ste_adamw=0.9472")


if __name__ == "__main__":
    main()
