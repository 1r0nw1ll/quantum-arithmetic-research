"""
rbm_qa_native_train.py

QA-indexed RBM trainer for qa_kona_ebm_qa_native_v1.

n_hidden = 81, fixed: one unit per QA state (b,e) in canonical sorted order.
Training: CD-1, numpy.random.default_rng(seed), batch_size=100, single shuffle.
After training: orbit analysis linking hidden unit activations to QA orbit structure.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

MNIST_PATH = "/home/player2/signal_experiments/data/MNIST/raw"
BATCH_SIZE = 100
GRAD_EXPLOSION_THRESHOLD = 1000.0
N_HIDDEN = 81   # fixed: one per QA state
N_VISIBLE = 784  # MNIST


def load_mnist_images(n_samples: int) -> np.ndarray:
    fpath = os.path.join(MNIST_PATH, "train-images-idx3-ubyte")
    with open(fpath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(n_samples * rows * cols), dtype=np.uint8)
    return (data.reshape(n_samples, rows * cols) / 255.0 > 0.5).astype(np.float64)


def load_mnist_labels(n_samples: int) -> np.ndarray:
    fpath = os.path.join(MNIST_PATH, "train-labels-idx1-ubyte")
    with open(fpath, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(n_samples), dtype=np.uint8)
    return labels.astype(np.int32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _free_energy(v: np.ndarray, W: np.ndarray, b_vis: np.ndarray, c: np.ndarray) -> float:
    bias_term = v @ b_vis
    hidden_pre = v @ W.T + c
    log_term = np.sum(np.log1p(np.exp(hidden_pre)), axis=1)
    return float(np.mean(-bias_term - log_term))


def _pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of columns of X (shape: n_samples x n_units)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    # Avoid division by zero for constant units
    norms = np.where(norms == 0, 1.0, norms)
    Xn = Xc / norms
    return Xn.T @ Xn / X.shape[0]

def compute_coherence_permutation_gap(
    h_probs: np.ndarray,
    orbit_unit_indices: dict,
    n_perm: int = 500,
    perm_seed: int = 0,
) -> dict:
    """
    Permutation gap test for orbit coherence.

    For each orbit type, compute the mean pairwise Pearson correlation among real
    orbit units (C_real), then compare to C_perm from 100 random permutations of
    unit->orbit assignment (preserving orbit sizes).

    Returns dict with keys COSMOS, SATELLITE, SINGULARITY, each containing:
      c_real: float
      c_perm_mean: float
      c_perm_std: float
      z_score: float  (= (c_real - c_perm_mean) / c_perm_std if std > 0 else 0.0)
      p_value: float  (fraction of permutations where C_perm >= C_real)
    """
    rng = np.random.default_rng(perm_seed)
    n_units = h_probs.shape[1]  # 81

    # Compute C_real per orbit type
    def mean_pairwise_r(idxs):
        if len(idxs) < 2:
            return 1.0
        sub = h_probs[:, idxs]
        C = _pearson_corr_matrix(sub)
        n = len(idxs)
        off = C.sum() - np.trace(C)
        return float(off / (n * n - n))

    orbit_types = ["COSMOS", "SATELLITE", "SINGULARITY"]
    sizes = {ot: len(orbit_unit_indices[ot]) for ot in orbit_types}
    c_real = {ot: mean_pairwise_r(orbit_unit_indices[ot]) for ot in orbit_types}

    # Permutation loop
    perm_scores = {ot: [] for ot in orbit_types}
    all_units = list(range(n_units))
    for _ in range(n_perm):
        perm = rng.permutation(all_units).tolist()
        # Re-assign: COSMOS gets first 72, SATELLITE next 8, SINGULARITY last 1
        perm_idxs = {
            "COSMOS":      perm[:sizes["COSMOS"]],
            "SATELLITE":   perm[sizes["COSMOS"]: sizes["COSMOS"] + sizes["SATELLITE"]],
            "SINGULARITY": perm[sizes["COSMOS"] + sizes["SATELLITE"]:],
        }
        for ot in orbit_types:
            perm_scores[ot].append(mean_pairwise_r(perm_idxs[ot]))

    result = {}
    for ot in orbit_types:
        cr = c_real[ot]
        ps = perm_scores[ot]
        pm = float(np.mean(ps))
        ps_std = float(np.std(ps))
        z = round((cr - pm) / ps_std, 6) if ps_std > 1e-12 else 0.0
        k = sum(p >= cr for p in ps)
        pv = round((k + 1) / (n_perm + 1), 6)
        result[ot] = {
            "c_real":      round(cr, 6),
            "c_perm_mean": round(pm, 6),
            "c_perm_std":  round(ps_std, 6),
            "z_score":     z,
            "p_value":     pv,
        }
    return result

def train_qa_rbm(
    n_samples: int,
    n_epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """
    Train QA-indexed RBM (n_hidden=81, n_visible=784) with CD-1.

    Returns result dict including orbit_analysis.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qa_orbit_map import build_orbit_map, orbit_map_hash as compute_orbit_map_hash

    rng = np.random.default_rng(seed)

    # Load data + labels (same n_samples for orbit analysis)
    images = load_mnist_images(n_samples)   # (n_samples, 784)
    labels = load_mnist_labels(n_samples)   # (n_samples,)

    idx = rng.permutation(n_samples)
    images = images[idx]
    labels = labels[idx]

    # Orbit map
    states, orbit_labels, orbit_ids = build_orbit_map()
    omhash = compute_orbit_map_hash()

    # Group hidden unit indices by orbit type
    orbit_unit_indices: Dict[str, List[int]] = {"COSMOS": [], "SATELLITE": [], "SINGULARITY": []}
    for i, lbl in enumerate(orbit_labels):
        orbit_unit_indices[lbl].append(i)

    # Initialise RBM parameters
    W = rng.normal(0.0, 0.01, size=(N_HIDDEN, N_VISIBLE))
    b_vis = np.zeros(N_VISIBLE)
    c_hid = np.zeros(N_HIDDEN)

    n_batches = n_samples // BATCH_SIZE

    energy_per_epoch: list = []
    recon_error_per_epoch: list = []
    grad_norm_per_epoch: list = []

    exploded = False
    explosion_update_norm = 0.0

    for epoch in range(n_epochs):
        epoch_recon_errors: list = []
        epoch_grad_norms: list = []

        for b_idx in range(n_batches):
            v0 = images[b_idx * BATCH_SIZE: (b_idx + 1) * BATCH_SIZE]

            # Positive phase
            h0_prob = _sigmoid(v0 @ W.T + c_hid)
            h0 = (rng.random(h0_prob.shape) < h0_prob).astype(np.float64)

            # Negative phase
            v1_prob = _sigmoid(h0 @ W + b_vis)
            h1_prob = _sigmoid(v1_prob @ W.T + c_hid)

            # Gradients
            dW = (h0_prob.T @ v0 - h1_prob.T @ v1_prob) / BATCH_SIZE
            db_update = np.mean(v0 - v1_prob, axis=0)
            dc_update = np.mean(h0_prob - h1_prob, axis=0)

            grad_norm = float(np.linalg.norm(dW, ord="fro"))
            update_norm = lr * grad_norm
            epoch_grad_norms.append(grad_norm)

            recon_err = float(np.mean((v0 - v1_prob) ** 2))
            epoch_recon_errors.append(recon_err)

            if update_norm > GRAD_EXPLOSION_THRESHOLD:
                exploded = True
                explosion_update_norm = update_norm
                break

            W += lr * dW
            b_vis += lr * db_update
            c_hid += lr * dc_update

        if exploded:
            break

        energy = _free_energy(images, W, b_vis, c_hid)
        energy_per_epoch.append(round(energy, 6))
        recon_error_per_epoch.append(round(float(np.mean(epoch_recon_errors)), 6))
        grad_norm_per_epoch.append(round(float(np.mean(epoch_grad_norms)), 6))

    if exploded:
        energy_per_epoch.append(round(_free_energy(images, W, b_vis, c_hid), 6))
        recon_error_per_epoch.append(
            round(float(np.mean(epoch_recon_errors)) if epoch_recon_errors else 0.0, 6)
        )
        grad_norm_per_epoch.append(round(explosion_update_norm, 6))
        status = "GRADIENT_EXPLOSION"
    else:
        if len(energy_per_epoch) >= 2 and energy_per_epoch[-1] < energy_per_epoch[0]:
            status = "CONVERGED"
        else:
            status = "STALLED"

    trace_payload = json.dumps(energy_per_epoch, separators=(",", ":")).encode()
    trace_hash = hashlib.sha256(trace_payload).hexdigest()
    final_weights_norm = round(float(np.linalg.norm(W, ord="fro")), 6)

    # ------------------------------------------------------------------
    # Orbit analysis: use same 1000 samples (after shuffle, first n_samples)
    # Use images as-is (already shuffled)
    # ------------------------------------------------------------------
    n_analysis = min(1000, n_samples)
    analysis_images = images[:n_analysis]
    analysis_labels = labels[:n_analysis]

    # Hidden activation probabilities: P(h=1|v) = sigmoid(W^T v + c)
    # W shape: (n_hidden, n_visible), so h_prob = images @ W.T + c_hid
    h_probs = _sigmoid(analysis_images @ W.T + c_hid)  # (n_analysis, 81)

    # orbit_class_alignment: for each orbit type, mean activation per digit class
    orbit_class_alignment: Dict[str, List[float]] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        unit_idxs = orbit_unit_indices[otype]
        # Mean activation over units in this orbit type: shape (n_analysis,)
        orbit_h = h_probs[:, unit_idxs]  # (n_analysis, n_units_in_orbit)
        # Mean per sample across units
        orbit_h_mean = orbit_h.mean(axis=1)  # (n_analysis,)
        # Mean per digit class
        class_means = []
        for digit in range(10):
            mask = analysis_labels == digit
            if mask.sum() == 0:
                class_means.append(0.0)
            else:
                class_means.append(round(float(orbit_h_mean[mask].mean()), 6))
        orbit_class_alignment[otype] = class_means

    # orbit_coherence_score: mean pairwise Pearson correlation among units in each orbit type
    orbit_coherence_score: Dict[str, float] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        unit_idxs = orbit_unit_indices[otype]
        orbit_h = h_probs[:, unit_idxs]  # (n_analysis, n_units)
        if len(unit_idxs) < 2:
            # Only 1 unit (SINGULARITY): self-correlation = 1.0
            orbit_coherence_score[otype] = round(1.0, 6)
        else:
            corr_matrix = _pearson_corr_matrix(orbit_h)  # (n_units, n_units)
            n = len(unit_idxs)
            # Mean of off-diagonal elements
            off_diag_sum = corr_matrix.sum() - np.trace(corr_matrix)
            n_pairs = n * n - n  # n*(n-1) off-diagonal elements
            mean_corr = float(off_diag_sum / n_pairs)
            orbit_coherence_score[otype] = round(mean_corr, 6)

    # orbit_dominant_class: which digit class has highest mean activation per orbit type
    orbit_dominant_class: Dict[str, int] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        orbit_dominant_class[otype] = int(np.argmax(orbit_class_alignment[otype]))

    # ------------------------------------------------------------------
    # Permutation gap test for orbit coherence
    # ------------------------------------------------------------------
    coherence_gap_stats = compute_coherence_permutation_gap(
        h_probs, orbit_unit_indices, n_perm=500, perm_seed=seed
    )

    orbit_analysis = {
        "orbit_class_alignment": orbit_class_alignment,
        "orbit_coherence_score": orbit_coherence_score,
        "orbit_dominant_class": orbit_dominant_class,
        "orbit_map_hash": omhash,
        "coherence_gap_stats": coherence_gap_stats,
    }

    return {
        "n_visible": N_VISIBLE,
        "n_hidden": N_HIDDEN,
        "n_samples": n_samples,
        "n_epochs": n_epochs,
        "lr": lr,
        "seed": seed,
        "energy_per_epoch": energy_per_epoch,
        "reconstruction_error_per_epoch": recon_error_per_epoch,
        "grad_norm_per_epoch": grad_norm_per_epoch,
        "trace_hash": trace_hash,
        "final_weights_norm": final_weights_norm,
        "status": status,
        "orbit_analysis": orbit_analysis,
    }


if __name__ == "__main__":
    result = train_qa_rbm(n_samples=1000, n_epochs=5, lr=0.01, seed=42)
    print(json.dumps(result, indent=2))
