"""
rbm_qa_orbit_reg_train.py

QA-indexed RBM trainer with orbit-coherence regularizer for
qa_kona_ebm_qa_native_orbit_reg_v1 (family [64]).

n_hidden = 81, fixed: one unit per QA state (b,e) in canonical sorted order.
Training: CD-1, numpy.random.default_rng(seed), batch_size=100, single shuffle.

Regularizer: R(W) = sum_O sum_{i in O} ||W_i - mu_O||^2
  where mu_O = mean weight vector of orbit O.
  Gradient: dR/dW_i = 2*(W_i - mu_O)
  Applied per batch: dW[idxs] -= lr * lambda_orbit * 2.0 * (W_orbit - mu_orbit)

NaN/Inf check after every batch update. If triggered, status =
REGULARIZER_NUMERIC_INSTABILITY and training halts.

Additional outputs vs family [63]:
  lambda_orbit       : float
  reg_norm_per_epoch : list[float] Frobenius norm of reg gradient across orbit units
  reg_trace_hash     : sha256 of json.dumps(reg_norm_per_epoch, ...)
  trace_hash         : sha256 of json.dumps(energy_per_epoch, ...) only
  lr_per_epoch       : list[float] actual lr used each epoch (always present)

LR schedule (optional):
  lr_schedule = {"type": "step", "steps": [{"epoch": 1, "lr": 0.01}, ...]}
  steps sorted by epoch ascending; first step must have epoch == 1.
  At each epoch (1-indexed), the lr from the most recent step whose epoch <=
  current epoch is used.  If lr_schedule is None, all epochs use the fixed lr.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FAMILY63 = os.path.join(os.path.dirname(_HERE), "qa_kona_ebm_qa_native_v1")
sys.path.insert(0, _FAMILY63)
sys.path.insert(0, _HERE)

from qa_orbit_map import build_orbit_map, orbit_map_hash as compute_orbit_map_hash

MNIST_PATH = "/home/player2/signal_experiments/data/MNIST/raw"
BATCH_SIZE = 100
GRAD_EXPLOSION_THRESHOLD = 1000.0
N_HIDDEN = 81
N_VISIBLE = 784


def load_mnist_images(n_samples: int) -> np.ndarray:
    fpath = os.path.join(MNIST_PATH, "train-images-idx3-ubyte")
    with open(fpath, "rb") as f:
        _magic, _n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(n_samples * rows * cols), dtype=np.uint8)
    return (data.reshape(n_samples, rows * cols) / 255.0 > 0.5).astype(np.float64)


def load_mnist_labels(n_samples: int) -> np.ndarray:
    fpath = os.path.join(MNIST_PATH, "train-labels-idx1-ubyte")
    with open(fpath, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(n_samples), dtype=np.uint8)
    return labels.astype(np.int32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _free_energy(v: np.ndarray, W: np.ndarray,
                 b_vis: np.ndarray, c: np.ndarray) -> float:
    bias_term = v @ b_vis
    hidden_pre = v @ W.T + c
    log_term = np.sum(np.log1p(np.exp(hidden_pre)), axis=1)
    return float(np.mean(-bias_term - log_term))


def _pearson_corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Xc, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = Xc / norms
    return Xn.T @ Xn / X.shape[0]


def compute_coherence_permutation_gap(
    h_probs: np.ndarray,
    orbit_unit_indices: dict,
    n_perm: int = 500,
    perm_seed: int = 0,
) -> dict:
    rng = np.random.default_rng(perm_seed)
    n_units = h_probs.shape[1]

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

    perm_scores = {ot: [] for ot in orbit_types}
    all_units = list(range(n_units))
    for _ in range(n_perm):
        perm = rng.permutation(all_units).tolist()
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


def _validate_lr_schedule(lr_schedule: dict) -> None:
    """Raise ValueError if lr_schedule is malformed."""
    if not isinstance(lr_schedule, dict):
        raise ValueError("lr_schedule must be a dict")
    if lr_schedule.get("type") != "step":
        raise ValueError(
            f"lr_schedule.type must be 'step', got {lr_schedule.get('type')!r}"
        )
    steps = lr_schedule.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("lr_schedule.steps must be a non-empty list")
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"lr_schedule.steps[{i}] must be a dict")
        if "epoch" not in step or "lr" not in step:
            raise ValueError(
                f"lr_schedule.steps[{i}] must have 'epoch' and 'lr' keys"
            )
        if not isinstance(step["epoch"], int) or step["epoch"] < 1:
            raise ValueError(
                f"lr_schedule.steps[{i}].epoch must be a positive integer"
            )
        if not isinstance(step["lr"], (int, float)) or step["lr"] <= 0:
            raise ValueError(
                f"lr_schedule.steps[{i}].lr must be a positive number"
            )
    if steps[0]["epoch"] != 1:
        raise ValueError(
            f"lr_schedule.steps[0].epoch must be 1, got {steps[0]['epoch']}"
        )


def _build_lr_lookup(lr_schedule: dict, n_epochs: int, default_lr: float) -> List[float]:
    """
    Return a list of length n_epochs with the lr to use for each epoch
    (epoch 1 = index 0).  Steps are sorted ascending by epoch.
    """
    steps = sorted(lr_schedule["steps"], key=lambda s: s["epoch"])
    result = []
    for ep in range(1, n_epochs + 1):
        chosen_lr = default_lr
        for step in steps:
            if step["epoch"] <= ep:
                chosen_lr = float(step["lr"])
            else:
                break
        result.append(round(chosen_lr, 8))
    return result


def train_qa_orbit_reg_rbm(
    n_samples: int,
    n_epochs: int,
    lr: float,
    lambda_orbit: float,
    seed: int,
    lr_schedule: Optional[dict] = None,
) -> dict:
    """
    Train QA-indexed RBM with orbit-coherence regularizer.

    Parameters
    ----------
    n_samples     : number of MNIST training images to use
    n_epochs      : number of full passes over the data
    lr            : base learning rate (used when lr_schedule is None)
    lambda_orbit  : orbit-coherence regularization strength
    seed          : RNG seed for reproducibility
    lr_schedule   : optional step-wise LR schedule dict (see module docstring)

    Returns
    -------
    dict with keys including lr_per_epoch (always present).
    Status: CONVERGED | STALLED | GRADIENT_EXPLOSION | REGULARIZER_NUMERIC_INSTABILITY
    """
    # Validate lr_schedule if provided
    if lr_schedule is not None:
        _validate_lr_schedule(lr_schedule)
        epoch_lrs = _build_lr_lookup(lr_schedule, n_epochs, lr)
    else:
        epoch_lrs = [round(lr, 8)] * n_epochs

    rng = np.random.default_rng(seed)

    images = load_mnist_images(n_samples)
    labels = load_mnist_labels(n_samples)

    idx = rng.permutation(n_samples)
    images = images[idx]
    labels = labels[idx]

    _states, orbit_labels, _orbit_ids = build_orbit_map()
    omhash = compute_orbit_map_hash()

    orbit_unit_indices: Dict[str, List[int]] = {
        "COSMOS": [], "SATELLITE": [], "SINGULARITY": []
    }
    for i, lbl in enumerate(orbit_labels):
        orbit_unit_indices[lbl].append(i)

    W = rng.normal(0.0, 0.01, size=(N_HIDDEN, N_VISIBLE))
    b_vis = np.zeros(N_VISIBLE)
    c_hid = np.zeros(N_HIDDEN)

    n_batches = n_samples // BATCH_SIZE

    energy_per_epoch: List[float] = []
    recon_error_per_epoch: List[float] = []
    grad_norm_per_epoch: List[float] = []
    reg_norm_per_epoch: List[float] = []
    lr_per_epoch: List[float] = []

    exploded = False
    reg_instability = False
    explosion_update_norm = 0.0
    _last_recon: List[float] = []
    _last_grad: List[float] = []
    _last_reg: List[float] = []
    _last_epoch_lr: float = epoch_lrs[0] if epoch_lrs else lr

    for epoch in range(n_epochs):
        current_lr = epoch_lrs[epoch]
        epoch_recon: List[float] = []
        epoch_grad: List[float] = []
        epoch_reg: List[float] = []
        batch_broke = False
        _last_epoch_lr = current_lr

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
            update_norm = current_lr * grad_norm
            epoch_grad.append(grad_norm)
            epoch_recon.append(float(np.mean((v0 - v1_prob) ** 2)))

            if update_norm > GRAD_EXPLOSION_THRESHOLD:
                exploded = True
                explosion_update_norm = update_norm
                _last_recon = epoch_recon
                _last_grad = epoch_grad
                _last_reg = epoch_reg
                batch_broke = True
                break

            # Orbit-coherence regularizer
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

            if not (np.isfinite(W).all() and
                    np.isfinite(b_vis).all() and
                    np.isfinite(c_hid).all()):
                reg_instability = True
                _last_recon = epoch_recon
                _last_grad = epoch_grad
                _last_reg = epoch_reg
                batch_broke = True
                break

        if batch_broke:
            break

        energy_per_epoch.append(round(_free_energy(images, W, b_vis, c_hid), 6))
        recon_error_per_epoch.append(round(float(np.mean(epoch_recon)), 6))
        grad_norm_per_epoch.append(round(float(np.mean(epoch_grad)), 6))
        reg_norm_per_epoch.append(round(float(np.mean(epoch_reg)) if epoch_reg else 0.0, 6))
        lr_per_epoch.append(current_lr)

    # Status and partial-epoch records on early exit
    if reg_instability:
        try:
            e_val = round(_free_energy(images, W, b_vis, c_hid), 6)
        except Exception:
            e_val = float("nan")
        energy_per_epoch.append(e_val)
        recon_error_per_epoch.append(round(float(np.mean(_last_recon)) if _last_recon else 0.0, 6))
        grad_norm_per_epoch.append(round(float(np.mean(_last_grad)) if _last_grad else 0.0, 6))
        reg_norm_per_epoch.append(round(float(np.mean(_last_reg)) if _last_reg else 0.0, 6))
        lr_per_epoch.append(_last_epoch_lr)
        status = "REGULARIZER_NUMERIC_INSTABILITY"
    elif exploded:
        try:
            e_val = round(_free_energy(images, W, b_vis, c_hid), 6)
        except Exception:
            e_val = float("nan")
        energy_per_epoch.append(e_val)
        recon_error_per_epoch.append(round(float(np.mean(_last_recon)) if _last_recon else 0.0, 6))
        grad_norm_per_epoch.append(round(explosion_update_norm, 6))
        reg_norm_per_epoch.append(round(float(np.mean(_last_reg)) if _last_reg else 0.0, 6))
        lr_per_epoch.append(_last_epoch_lr)
        status = "GRADIENT_EXPLOSION"
    else:
        if len(energy_per_epoch) >= 2 and energy_per_epoch[-1] < energy_per_epoch[0]:
            status = "CONVERGED"
        else:
            status = "STALLED"

    trace_payload = json.dumps(energy_per_epoch, separators=(",", ":")).encode()
    trace_hash = hashlib.sha256(trace_payload).hexdigest()
    reg_payload = json.dumps(reg_norm_per_epoch, separators=(",", ":")).encode()
    reg_trace_hash = hashlib.sha256(reg_payload).hexdigest()
    final_weights_norm = round(float(np.linalg.norm(W, ord="fro")), 6)

    # Orbit analysis
    n_analysis = min(1000, n_samples)
    analysis_images = images[:n_analysis]
    analysis_labels = labels[:n_analysis]
    h_probs = _sigmoid(analysis_images @ W.T + c_hid)

    orbit_class_alignment: Dict[str, List[float]] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        unit_idxs = orbit_unit_indices[otype]
        orbit_h = h_probs[:, unit_idxs]
        orbit_h_mean = orbit_h.mean(axis=1)
        class_means = []
        for digit in range(10):
            mask = analysis_labels == digit
            if mask.sum() == 0:
                class_means.append(0.0)
            else:
                class_means.append(round(float(orbit_h_mean[mask].mean()), 6))
        orbit_class_alignment[otype] = class_means

    orbit_coherence_score: Dict[str, float] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        unit_idxs = orbit_unit_indices[otype]
        orbit_h = h_probs[:, unit_idxs]
        if len(unit_idxs) < 2:
            orbit_coherence_score[otype] = round(1.0, 6)
        else:
            corr_matrix = _pearson_corr_matrix(orbit_h)
            n = len(unit_idxs)
            off_diag_sum = corr_matrix.sum() - np.trace(corr_matrix)
            n_pairs = n * n - n
            orbit_coherence_score[otype] = round(float(off_diag_sum / n_pairs), 6)

    orbit_dominant_class: Dict[str, int] = {}
    for otype in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        orbit_dominant_class[otype] = int(np.argmax(orbit_class_alignment[otype]))

    coherence_gap_stats = compute_coherence_permutation_gap(
        h_probs, orbit_unit_indices, n_perm=500, perm_seed=seed
    )

    orbit_analysis = {
        "orbit_class_alignment": orbit_class_alignment,
        "orbit_coherence_score": orbit_coherence_score,
        "orbit_dominant_class":  orbit_dominant_class,
        "orbit_map_hash":        omhash,
        "coherence_gap_stats":   coherence_gap_stats,
    }

    return {
        "n_visible":                      N_VISIBLE,
        "n_hidden":                       N_HIDDEN,
        "n_samples":                      n_samples,
        "n_epochs":                       n_epochs,
        "lr":                             lr,
        "lambda_orbit":                   lambda_orbit,
        "seed":                           seed,
        "energy_per_epoch":               energy_per_epoch,
        "reconstruction_error_per_epoch": recon_error_per_epoch,
        "grad_norm_per_epoch":            grad_norm_per_epoch,
        "reg_norm_per_epoch":             reg_norm_per_epoch,
        "lr_per_epoch":                   lr_per_epoch,
        "trace_hash":                     trace_hash,
        "reg_trace_hash":                 reg_trace_hash,
        "final_weights_norm":             final_weights_norm,
        "status":                         status,
        "orbit_analysis":                 orbit_analysis,
    }


if __name__ == "__main__":
    result = train_qa_orbit_reg_rbm(
        n_samples=1000, n_epochs=5, lr=0.01, lambda_orbit=1e-3, seed=42
    )
    print(json.dumps(result, indent=2))
