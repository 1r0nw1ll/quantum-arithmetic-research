#!/usr/bin/env python3
"""
analyze_coherence_trajectory.py

Per-epoch orbit coherence tracker for QA-native orbit-reg RBM.

Runs a single seed, computing orbit coherence scores (c_real) after every
epoch — no permutation test (too slow per epoch) — to map the rise/peak/decay
of the coherence transient.

Usage:
    python qa_kona_ebm_qa_native_orbit_reg_v1/analyze_coherence_trajectory.py
    python qa_kona_ebm_qa_native_orbit_reg_v1/analyze_coherence_trajectory.py \
        --lambda-orbit 10.0 --n-epochs 50 --seeds 0 1 2 --json-out /tmp/traj.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import struct
from typing import Dict, List

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_FAMILY63 = os.path.join(os.path.dirname(_DIR), "qa_kona_ebm_qa_native_v1")
sys.path.insert(0, _FAMILY63)
sys.path.insert(0, _DIR)

from qa_orbit_map import build_orbit_map

MNIST_PATH = "/home/player2/signal_experiments/data/MNIST/raw"
BATCH_SIZE = 100
GRAD_EXPLOSION_THRESHOLD = 1000.0
N_HIDDEN = 81
N_VISIBLE = 784


def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _load_images(n_samples):
    fpath = os.path.join(MNIST_PATH, "train-images-idx3-ubyte")
    with open(fpath, "rb") as f:
        _magic, _n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(n_samples * rows * cols), dtype=np.uint8)
    return (data.reshape(n_samples, rows * cols) / 255.0 > 0.5).astype(np.float64)


def _load_labels(n_samples):
    fpath = os.path.join(MNIST_PATH, "train-labels-idx1-ubyte")
    with open(fpath, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(n_samples), dtype=np.uint8)
    return labels.astype(np.int32)


def _pearson_corr_matrix(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = Xc / norms
    return Xn.T @ Xn / X.shape[0]


def _orbit_coherence(h_probs, unit_idxs):
    if len(unit_idxs) < 2:
        return 1.0
    sub = h_probs[:, unit_idxs]
    C = _pearson_corr_matrix(sub)
    n = len(unit_idxs)
    off = C.sum() - np.trace(C)
    return float(off / (n * n - n))


def _build_epoch_lrs(n_epochs, lr, lr_schedule):
    """Return list of per-epoch learning rates (1-indexed epoch -> 0-indexed list)."""
    if lr_schedule is None:
        return [lr] * n_epochs
    steps = sorted(lr_schedule["steps"], key=lambda s: s["epoch"])
    result = []
    for ep in range(1, n_epochs + 1):
        cur_lr = lr
        for step in steps:
            if step["epoch"] <= ep:
                cur_lr = step["lr"]
        result.append(cur_lr)
    return result


def run_trajectory(n_samples, n_epochs, lr, lambda_orbit, seed, lr_schedule=None):
    rng = np.random.default_rng(seed)

    images = _load_images(n_samples)
    _load_labels(n_samples)  # not used per-epoch but keeps parity with trainer

    idx = rng.permutation(n_samples)
    images = images[idx]

    _states, orbit_labels, _orbit_ids = build_orbit_map()
    orbit_unit_indices: Dict[str, List[int]] = {
        "COSMOS": [], "SATELLITE": [], "SINGULARITY": []
    }
    for i, lbl in enumerate(orbit_labels):
        orbit_unit_indices[lbl].append(i)

    W = rng.normal(0.0, 0.01, size=(N_HIDDEN, N_VISIBLE))
    b_vis = np.zeros(N_VISIBLE)
    c_hid = np.zeros(N_HIDDEN)

    n_batches = n_samples // BATCH_SIZE
    epoch_lrs = _build_epoch_lrs(n_epochs, lr, lr_schedule)

    trajectory = []  # list of per-epoch dicts

    for epoch in range(n_epochs):
        current_lr = epoch_lrs[epoch]
        epoch_recon, epoch_grad, epoch_reg = [], [], []
        broke = False

        for b_idx in range(n_batches):
            v0 = images[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]

            h0_prob = _sigmoid(v0 @ W.T + c_hid)
            h0 = (rng.random(h0_prob.shape) < h0_prob).astype(np.float64)
            v1_prob = _sigmoid(h0 @ W + b_vis)
            h1_prob = _sigmoid(v1_prob @ W.T + c_hid)

            dW = (h0_prob.T @ v0 - h1_prob.T @ v1_prob) / BATCH_SIZE
            db_update = np.mean(v0 - v1_prob, axis=0)
            dc_update = np.mean(h0_prob - h1_prob, axis=0)

            grad_norm = float(np.linalg.norm(dW, ord="fro"))
            epoch_grad.append(grad_norm)
            epoch_recon.append(float(np.mean((v0 - v1_prob) ** 2)))

            if current_lr * grad_norm > GRAD_EXPLOSION_THRESHOLD:
                broke = True
                break

            # Orbit-coherence regularizer (COSMOS + SATELLITE only)
            batch_reg_sq = 0.0
            for otype in ["COSMOS", "SATELLITE"]:
                idxs = orbit_unit_indices[otype]
                W_orbit = W[idxs, :]
                mu_orbit = W_orbit.mean(axis=0)
                reg_grad = W_orbit - mu_orbit
                batch_reg_sq += float(np.sum(reg_grad * reg_grad))
                dW[idxs, :] -= current_lr * lambda_orbit * 2.0 * reg_grad

            epoch_reg.append(batch_reg_sq ** 0.5)

            W += current_lr * dW
            b_vis += current_lr * db_update
            c_hid += current_lr * dc_update

            if not (np.isfinite(W).all() and np.isfinite(b_vis).all() and
                    np.isfinite(c_hid).all()):
                broke = True
                break

        if broke:
            trajectory.append({
                "epoch": epoch + 1,
                "status": "STOPPED",
                "cosmos_c": None, "sat_c": None,
                "energy": None, "recon": None, "reg_norm": None,
            })
            break

        # Per-epoch orbit coherence (no perm test — fast)
        h_probs = _sigmoid(images @ W.T + c_hid)
        cosmos_c = round(_orbit_coherence(h_probs, orbit_unit_indices["COSMOS"]), 6)
        sat_c    = round(_orbit_coherence(h_probs, orbit_unit_indices["SATELLITE"]), 6)

        bias_term  = images @ b_vis
        hidden_pre = images @ W.T + c_hid
        energy     = round(float(np.mean(-bias_term - np.sum(np.log1p(np.exp(hidden_pre)), axis=1))), 4)
        recon      = round(float(np.mean(epoch_recon)), 6)
        reg_norm   = round(float(np.mean(epoch_reg)), 6) if epoch_reg else 0.0

        trajectory.append({
            "epoch":    epoch + 1,
            "status":   "OK",
            "cosmos_c": cosmos_c,
            "sat_c":    sat_c,
            "energy":   energy,
            "recon":    recon,
            "reg_norm": reg_norm,
        })

    return trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Per-epoch orbit coherence trajectory for QA-native orbit-reg RBM."
    )
    parser.add_argument("--lambda-orbit", type=float, default=10.0)
    parser.add_argument("--n-epochs",     type=int,   default=50)
    parser.add_argument("--n-samples",    type=int,   default=1000)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--seeds",        type=int,   nargs="+", default=[0, 1, 2])
    parser.add_argument("--lr-schedule", type=str, default=None, dest="lr_schedule",
                        help="JSON string for step LR schedule, e.g. {\"type\":\"step\",\"steps\":[{\"epoch\":1,\"lr\":0.01}]}")
    parser.add_argument("--json-out",     type=str,   default=None)
    args = parser.parse_args()
    
    lr_schedule = json.loads(args.lr_schedule) if args.lr_schedule else None

    print(f"Coherence trajectory: lambda={args.lambda_orbit}, "
          f"n_epochs={args.n_epochs}, lr={args.lr}, seeds={args.seeds}, "
          f"lr_schedule={lr_schedule}")
    print()

    # Header
    header = (f"{'ep':>4}  {'COSMOS_c':>10}  {'SAT_c':>10}  "
              f"{'energy':>10}  {'recon':>8}  {'reg_norm':>10}")
    all_results = {}

    for seed in args.seeds:
        print(f"--- seed={seed} ---")
        print(header)
        print("-" * len(header))

        traj = run_trajectory(
            n_samples=args.n_samples,
            n_epochs=args.n_epochs,
            lr=args.lr,
            lambda_orbit=args.lambda_orbit,
            seed=seed,
            lr_schedule=lr_schedule,
        )

        for row in traj:
            if row["status"] == "STOPPED":
                print(f"{row['epoch']:>4}  STOPPED")
                break
            print(f"{row['epoch']:>4}  {row['cosmos_c']:>10.6f}  {row['sat_c']:>10.6f}  "
                  f"{row['energy']:>10.2f}  {row['recon']:>8.6f}  {row['reg_norm']:>10.6f}")

        all_results[str(seed)] = traj

        # Find peak epoch for COSMOS
        ok_rows = [r for r in traj if r["status"] == "OK"]
        if ok_rows:
            peak = max(ok_rows, key=lambda r: r["cosmos_c"])
            print(f"  -> COSMOS peak: epoch={peak['epoch']}, c={peak['cosmos_c']:.6f}")
        print()

    if args.json_out:
        body = {
            "config": {
                "lambda_orbit": args.lambda_orbit,
                "n_epochs":     args.n_epochs,
                "lr":           args.lr,
                "n_samples":    args.n_samples,
                "seeds":        args.seeds,
                "lr_schedule":  lr_schedule,
            },
            "trajectories": all_results,
        }
        body_bytes = json.dumps(body, sort_keys=True,
                                separators=(",", ":"), ensure_ascii=False).encode()
        body["sha256"] = hashlib.sha256(body_bytes).hexdigest()
        with open(args.json_out, "w") as fh:
            json.dump(body, fh, indent=2)
        print(f"JSON written to: {args.json_out}")


if __name__ == "__main__":
    main()
