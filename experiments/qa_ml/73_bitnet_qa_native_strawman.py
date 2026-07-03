#!/usr/bin/env python3
"""
Tests the one untested claim from the BitNet/QA synthesis: can a QA-native
discrete update rule (qa_step generator + a-mod-3 ternary readout, integers
only, no float ever causally drives weight state) train a ternary-weight MLP
competitively with a standard BitNet-style STE + float-gradient optimizer?

Four conditions on sklearn digits (64 -> 32 -> 10, ternary weights, no bias):

  A. QA-native   : per-weight integer position b in {1..24} (qa_core modulus),
                    updated via qa_step(b, e) = ((b+e-1) % 24) + 1 where e is
                    derived from an integer vote count (never a float). Ternary
                    readout = f((b-1) % 3) -- the a-mod-3 red/green/blue
                    partition. NT-compliant: no float touches weight state.
  B. BitNet-style: float latent weights, absmean ternary quantization,
                    straight-through estimator, AdamW. This is what the
                    literature actually does -- float gradients ARE causal
                    to the ternary weight, which is the T2 violation QA's
                    firewall would flag. Included as the real baseline to beat.
  C. Random      : random ternary weights, untrained. Floor.
  D. Naive vote  : same integer vote signal as A, but read out via a raw
                    thresholded accumulator (sign with dead-zone) instead of
                    qa_step + mod-3 routing. Isolates whether the QA machinery
                    (generator step, mod-3 partition) adds anything over a
                    generic discrete perceptron, or whether it's cosmetic.

5 seeds each. Reports mean/std test accuracy and wall-clock time.

RESULT: both A and D collapse to chance (~10%) -- this is a strawman, not a
fair test of "no float causally enters QA state" (see qa_native_ordered for
a partial correction, and 74_bitnet_fixed_point_sgd.py for the real fix: the
issue was using a weak vote-counting heuristic instead of real gradient
descent, not discreteness itself). Kept as a documented negative result --
see docs/specs/QA_ML_BITNET_NT_COMPLIANT_TRAINING_FINDINGS.md section 3.

QA_COMPLIANCE = "bitnet_qa_native_strawman - integer vote-counting update heuristic; diagnosed negative result, not a discreteness claim"
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

M = 24  # QA applied modulus (CLAUDE.md)
MOD3_MAP = {0: 0, 1: 1, 2: -1}  # a-mod-3 residue -> ternary symbol (fixed, arbitrary assignment)
EPOCHS = 60
BATCH = 32
VOTE_CLIP = 3
DEADZONE_T = 6  # for condition D, tuned to roughly match A's update magnitude scale


def qa_step(b: np.ndarray, e: np.ndarray) -> np.ndarray:
    return ((b + e - 1) % M) + 1


def ternary_from_b(b: np.ndarray) -> np.ndarray:
    residue = (b - 1) % 3
    out = np.zeros_like(b)
    out[residue == 1] = 1
    out[residue == 2] = -1
    return out


def ternary_from_b_ordered(b: np.ndarray) -> np.ndarray:
    # order-preserving readout over the same b in {1..M}: thirds instead of mod-3 residue.
    out = np.zeros_like(b)
    out[b <= M // 3] = -1
    out[b > 2 * (M // 3)] = 1
    return out


def forward(X, W1, W2):
    h_pre = X @ W1
    h = np.maximum(h_pre, 0)
    logits = h @ W2
    return h, logits


def accuracy(X, y, W1, W2):
    _, logits = forward(X, W1, W2)
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def sign_i(x):
    return np.sign(x).astype(np.int64)


def compute_votes(X_batch, y_batch, W1, W2, n_classes):
    h, logits = forward(X_batch, W1, W2)
    pred = np.argmax(logits, axis=1)
    vote1 = np.zeros_like(W1, dtype=np.int64)
    vote2 = np.zeros_like(W2, dtype=np.int64)
    for i in range(X_batch.shape[0]):
        t, p = y_batch[i], pred[i]
        if t == p:
            continue
        sh = sign_i(h[i, :])
        vote2[:, t] += sh
        vote2[:, p] -= sh
        err_hidden = sign_i(W2[:, t] - W2[:, p])
        sx = sign_i(X_batch[i, :])
        vote1 += np.outer(sx, err_hidden)
    return vote1, vote2


def run_qa_native(X_tr, y_tr, X_te, y_te, seed, n_classes):
    rng = np.random.default_rng(seed)
    b1 = rng.integers(1, M + 1, size=(64, 32), dtype=np.int64)
    b2 = rng.integers(1, M + 1, size=(32, n_classes), dtype=np.int64)
    curve = []
    n = X_tr.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            W1, W2 = ternary_from_b(b1), ternary_from_b(b2)
            vote1, vote2 = compute_votes(X_tr[idx], y_tr[idx], W1, W2, n_classes)
            vote1 = np.clip(vote1, -VOTE_CLIP, VOTE_CLIP)
            vote2 = np.clip(vote2, -VOTE_CLIP, VOTE_CLIP)
            nz1 = vote1 != 0
            e1 = vote1[nz1] % M
            b1[nz1] = qa_step(b1[nz1], e1)
            nz2 = vote2 != 0
            e2 = vote2[nz2] % M
            b2[nz2] = qa_step(b2[nz2], e2)
        W1, W2 = ternary_from_b(b1), ternary_from_b(b2)
        curve.append(accuracy(X_te, y_te, W1, W2))
    return curve


def run_qa_native_ordered(X_tr, y_tr, X_te, y_te, seed, n_classes):
    rng = np.random.default_rng(seed)
    b1 = rng.integers(1, M + 1, size=(64, 32), dtype=np.int64)
    b2 = rng.integers(1, M + 1, size=(32, n_classes), dtype=np.int64)
    curve = []
    n = X_tr.shape[0]
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            W1, W2 = ternary_from_b_ordered(b1), ternary_from_b_ordered(b2)
            vote1, vote2 = compute_votes(X_tr[idx], y_tr[idx], W1, W2, n_classes)
            vote1 = np.clip(vote1, -VOTE_CLIP, VOTE_CLIP)
            vote2 = np.clip(vote2, -VOTE_CLIP, VOTE_CLIP)
            # clamp b to [1, M] instead of wrapping, so the order-preserving
            # readout doesn't cycle at the boundary (a genuine qa_step
            # deviation, made explicit rather than silently wrapping).
            nz1 = vote1 != 0
            b1[nz1] = np.clip(b1[nz1] + vote1[nz1], 1, M)
            nz2 = vote2 != 0
            b2[nz2] = np.clip(b2[nz2] + vote2[nz2], 1, M)
        W1, W2 = ternary_from_b_ordered(b1), ternary_from_b_ordered(b2)
        curve.append(accuracy(X_te, y_te, W1, W2))
    return curve


def run_naive_vote(X_tr, y_tr, X_te, y_te, seed, n_classes):
    rng = np.random.default_rng(seed)
    acc1 = rng.integers(-2, 3, size=(64, 32), dtype=np.int64)
    acc2 = rng.integers(-2, 3, size=(32, n_classes), dtype=np.int64)
    curve = []
    n = X_tr.shape[0]

    def readout(acc):
        out = np.zeros_like(acc)
        out[acc >= DEADZONE_T] = 1
        out[acc <= -DEADZONE_T] = -1
        return out

    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            W1, W2 = readout(acc1), readout(acc2)
            vote1, vote2 = compute_votes(X_tr[idx], y_tr[idx], W1, W2, n_classes)
            acc1 += np.clip(vote1, -VOTE_CLIP, VOTE_CLIP)
            acc2 += np.clip(vote2, -VOTE_CLIP, VOTE_CLIP)
        W1, W2 = readout(acc1), readout(acc2)
        curve.append(accuracy(X_te, y_te, W1, W2))
    return curve


def run_random(X_te, y_te, seed, n_classes):
    rng = np.random.default_rng(seed)
    b1 = rng.integers(1, M + 1, size=(64, 32), dtype=np.int64)
    b2 = rng.integers(1, M + 1, size=(32, n_classes), dtype=np.int64)
    W1, W2 = ternary_from_b(b1), ternary_from_b(b2)
    return accuracy(X_te, y_te, W1, W2)


def run_bitnet(X_tr, y_tr, X_te, y_te, seed, n_classes):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class TernaryLinear(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(d_out, d_in) * 0.1)

        def forward(self, x):
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_q = torch.clamp(torch.round(self.weight / gamma), -1, 1)
            w_ste = self.weight + (w_q - self.weight).detach()
            return x @ w_ste.t() * gamma

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = TernaryLinear(64, 32)
            self.l2 = TernaryLinear(32, n_classes)

        def forward(self, x):
            h = torch.relu(self.l1(x))
            return self.l2(h)

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.long)

    n = Xt.shape[0]
    rng = np.random.default_rng(seed)
    curve = []
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            opt.zero_grad()
            out = net(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = net(Xte).argmax(dim=1)
            curve.append(float((pred == yte).float().mean()))
    return curve


def main():
    digits = load_digits()
    X, y = digits.data.astype(np.int64), digits.target.astype(np.int64)
    n_classes = 10

    seeds = [42, 43, 44, 45, 46]
    conds = ["qa_native", "qa_native_ordered", "naive_vote", "bitnet_ste", "random"]
    results = {c: [] for c in conds}
    times = {c: 0.0 for c in conds}
    curves = {c: [] for c in conds if c != "random"}

    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        t0 = time.time()
        c = run_qa_native(X_tr, y_tr, X_te, y_te, seed, n_classes)
        times["qa_native"] += time.time() - t0
        results["qa_native"].append(c[-1])
        curves["qa_native"].append(c)

        t0 = time.time()
        c = run_qa_native_ordered(X_tr, y_tr, X_te, y_te, seed, n_classes)
        times["qa_native_ordered"] += time.time() - t0
        results["qa_native_ordered"].append(c[-1])
        curves["qa_native_ordered"].append(c)

        t0 = time.time()
        c = run_naive_vote(X_tr, y_tr, X_te, y_te, seed, n_classes)
        times["naive_vote"] += time.time() - t0
        results["naive_vote"].append(c[-1])
        curves["naive_vote"].append(c)

        t0 = time.time()
        c = run_bitnet(X_tr, y_tr, X_te, y_te, seed, n_classes)
        times["bitnet_ste"] += time.time() - t0
        results["bitnet_ste"].append(c[-1])
        curves["bitnet_ste"].append(c)

        t0 = time.time()
        r = run_random(X_te, y_te, seed, n_classes)
        times["random"] += time.time() - t0
        results["random"].append(r)

    print(f"{'condition':<20} {'mean_acc':>10} {'std':>8} {'min':>8} {'max':>8} {'total_s':>10}")
    for cond in ["random", "naive_vote", "qa_native", "qa_native_ordered", "bitnet_ste"]:
        arr = np.array(results[cond])
        print(f"{cond:<20} {arr.mean():>10.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {times[cond]:>10.2f}")

    print()
    print("Learning curves (test acc every 10 epochs, mean over seeds):")
    for cond in ["naive_vote", "qa_native", "qa_native_ordered", "bitnet_ste"]:
        arr = np.array(curves[cond])  # (seeds, epochs)
        mean_curve = arr.mean(axis=0)
        sampled = mean_curve[::10]
        print(f"  {cond:<20}: " + " ".join(f"{v:.3f}" for v in sampled))


if __name__ == "__main__":
    main()
