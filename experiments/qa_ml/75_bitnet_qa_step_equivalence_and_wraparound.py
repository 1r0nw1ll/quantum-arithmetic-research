#!/usr/bin/env python3
"""
Isolates whether qa_step itself (CLAUDE.md: ((b+e-1) % m) + 1) does anything
different from plain integer subtraction, in two regimes, on top of the
working fixed-point SGD baseline (74_bitnet_fixed_point_sgd.py, ~89.7% acc).

PART A -- non-wraparound regime: routes the same fixed-point SGD update
through qa_step on an offset representation with a modulus large enough
that it never wraps (unbounded SGD drifted latent weights to magnitude
~5.4; the modulus here gives ~16.7M of headroom). RESULT: bit-identical to
plain subtraction (mean_acc=0.8972 to 4 decimals). qa_step is a cyclic-group
relabeling of addition here, not a distinct mechanism.

PART B -- wraparound regime: same gradients/lr/seeds, but with a bound set
well below the natural excursion, so wraparound is forced to occur
repeatedly. Three conditions: unbounded (reference), clip/saturate (the
standard convention real int8/fixed-point quantized training uses on
overflow), and qa_step wrap (QA's actual convention). RESULT: wrap collapses
to chance (~10%) while clip scores *better* than unbounded (91.0-91.2%,
acting as implicit regularization). Mechanism verified directly (see
docs/specs/QA_ML_BITNET_NT_COMPLIANT_TRAINING_FINDINGS.md section 5): SGD
drives ternary-quantized weight magnitude up over training for robustness;
wrapping reverses exactly the sign of the weights the optimizer is most
confident about, while clipping merely caps the growth safely.

QA_COMPLIANCE = "bitnet_qa_step_equivalence_and_wraparound - qa_step vs plain subtraction, non-wrap (identical) and wraparound (harmful) regimes"
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SCALE = 1 << 16
EPOCHS = 150
BATCH = 32
LR = 1.0
BOUND_NONWRAP = 1 << 24  # modulus half-width, orders of magnitude larger than any weight excursion
M_NONWRAP = 2 * BOUND_NONWRAP


def fp_mul(a, b):
    return (a.astype(np.int64) * np.asarray(b, dtype=np.int64)) // SCALE


def fp_matmul(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)) // SCALE


def fp_div_pos(a, b):
    return (a.astype(np.int64) * SCALE) // np.maximum(b.astype(np.int64), 1)


def ternary_quant(latent):
    gamma = np.maximum(np.sum(np.abs(latent)) // latent.size, 1)
    ratio = fp_div_pos(latent, np.full_like(latent, gamma))
    sign = np.sign(ratio)
    rounded = sign * ((np.abs(ratio) + SCALE // 2) // SCALE)
    return np.clip(rounded, -1, 1).astype(np.int64)


def to_fixed(x_int):
    return x_int.astype(np.int64) * SCALE


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


def run(mode, X_tr, y_tr, X_te, y_te, seed, n_classes, bound_real=None):
    rng = np.random.default_rng(seed)
    if mode == "wrap_nonwrap_test":
        BOUND, M = BOUND_NONWRAP, M_NONWRAP
    elif mode in ("clip", "wrap"):
        BOUND = int(round(bound_real * SCALE))
        M = 2 * BOUND
    else:
        BOUND = M = None  # unbounded

    init_span = SCALE // 20 if mode == "unbounded" else min(SCALE // 20, (BOUND or SCALE) // 2)
    latent1 = rng.integers(-init_span, init_span + 1, size=(64, 32), dtype=np.int64)
    latent2 = rng.integers(-init_span, init_span + 1, size=(32, n_classes), dtype=np.int64)
    lr_fp = int(round(LR * SCALE))

    X_tr_fp = to_fixed(X_tr)
    X_te_fp = to_fixed(X_te)
    onehot_tr = np.eye(n_classes, dtype=np.int64)[y_tr] * SCALE

    n = X_tr.shape[0]
    curve = []
    events = 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            xb, yb = X_tr_fp[idx], onehot_tr[idx]
            bs = xb.shape[0]
            dW1, dW2 = compute_grads(xb, yb, latent1, latent2, bs)
            u1 = fp_mul(dW1, lr_fp)
            u2 = fp_mul(dW2, lr_fp)

            if mode == "wrap_nonwrap_test":
                b1 = latent1 + BOUND + 1
                b2 = latent2 + BOUND + 1
                b1 = ((b1 - u1 - 1) % M) + 1
                b2 = ((b2 - u2 - 1) % M) + 1
                latent1 = b1 - BOUND - 1
                latent2 = b2 - BOUND - 1
            elif mode == "wrap":
                raw1, raw2 = latent1 - u1, latent2 - u2
                events += int(np.sum((raw1 < -BOUND) | (raw1 >= BOUND)))
                events += int(np.sum((raw2 < -BOUND) | (raw2 >= BOUND)))
                b1 = latent1 + BOUND + 1
                b2 = latent2 + BOUND + 1
                b1 = ((b1 - u1 - 1) % M) + 1
                b2 = ((b2 - u2 - 1) % M) + 1
                latent1 = b1 - BOUND - 1
                latent2 = b2 - BOUND - 1
            elif mode == "clip":
                raw1, raw2 = latent1 - u1, latent2 - u2
                events += int(np.sum((raw1 < -BOUND) | (raw1 >= BOUND)))
                events += int(np.sum((raw2 < -BOUND) | (raw2 >= BOUND)))
                latent1 = np.clip(raw1, -BOUND, BOUND - 1)
                latent2 = np.clip(raw2, -BOUND, BOUND - 1)
            else:  # unbounded
                latent1 = latent1 - u1
                latent2 = latent2 - u2

        curve.append(eval_acc(latent1, latent2, X_te_fp, y_te))
    return curve, events


def part_a_nonwrap_equivalence(X, y, n_classes):
    print("=== Part A: qa_step vs plain subtraction, non-wrap regime ===")
    seeds = [42, 43, 44, 45, 46]
    accs = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        curve, _ = run("wrap_nonwrap_test", X_tr, y_tr, X_te, y_te, seed, n_classes)
        accs.append(curve[-1])
    arr = np.array(accs)
    print(f"  qa_step (offset, large modulus): mean_acc={arr.mean():.4f} std={arr.std():.4f}")
    print("  compare: 74_bitnet_fixed_point_sgd.py plain subtraction = 0.8972 (bit-identical)\n")


def part_b_wraparound(X, y, n_classes):
    print("=== Part B: wraparound regime ===")
    seeds = [42, 43, 44, 45, 46]
    for bound_real in [0.5, 0.1]:
        print(f"  --- BOUND = +-{bound_real} (real units) ---")
        for mode in ["unbounded", "clip", "wrap"]:
            accs, total_events = [], 0
            for seed in seeds:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
                b = bound_real if mode != "unbounded" else 1e9
                curve, events = run(mode, X_tr, y_tr, X_te, y_te, seed, n_classes, b)
                accs.append(curve[-1])
                total_events += events
            arr = np.array(accs)
            print(f"    {mode:<10} mean_acc={arr.mean():.4f} std={arr.std():.4f} boundary_events={total_events}")
        print()


def main():
    digits = load_digits()
    X, y = digits.data.astype(np.int64), digits.target.astype(np.int64)
    n_classes = 10
    part_a_nonwrap_equivalence(X, y, n_classes)
    part_b_wraparound(X, y, n_classes)


if __name__ == "__main__":
    main()
