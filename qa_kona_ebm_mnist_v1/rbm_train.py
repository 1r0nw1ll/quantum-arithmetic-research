"""
rbm_train.py

Standalone numpy-only RBM trainer for qa_kona_ebm_mnist_v1.

Training algorithm: Contrastive Divergence CD-1.
Determinism contract: numpy.random.default_rng(seed), fixed batch_size=100,
single shuffle at start, no BLAS parallelism (pure numpy matmul).
"""
from __future__ import annotations

import hashlib
import gzip
import json
import os
import struct
from pathlib import Path

import numpy as np

BATCH_SIZE = 100
GRAD_EXPLOSION_THRESHOLD = 1000.0


def _default_mnist_path() -> Path:
    # Repo layout: <repo>/qa_kona_ebm_mnist_v1/rbm_train.py -> <repo>/data/MNIST/raw
    return Path(__file__).resolve().parent.parent / "data" / "MNIST" / "raw"


def _resolve_mnist_image_path() -> Path:
    base = Path(os.environ.get("MNIST_PATH", _default_mnist_path()))
    raw_path = base / "train-images-idx3-ubyte"
    gz_path = base / "train-images-idx3-ubyte.gz"
    if raw_path.exists():
        return raw_path
    if gz_path.exists():
        return gz_path
    raise FileNotFoundError(
        f"MNIST train images not found. Checked: '{raw_path}' and '{gz_path}'. "
        "Set MNIST_PATH to override."
    )


def load_mnist_images(n_samples: int) -> np.ndarray:
    """Load first n_samples binarised MNIST training images (784-dim)."""
    fpath = _resolve_mnist_image_path()
    opener = gzip.open if fpath.suffix == ".gz" else open
    with opener(fpath, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(n_samples * rows * cols), dtype=np.uint8)
    return (data.reshape(n_samples, rows * cols) / 255.0 > 0.5).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _free_energy(v: np.ndarray, W: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Mean free energy over a batch of visible vectors.

    F(v) = -b^T v - sum_j log(1 + exp(c_j + W_j . v))
    Mean over all samples in v.
    """
    bias_term = v @ b                           # (n,)
    hidden_pre = v @ W.T + c                    # (n, n_hidden)
    log_term = np.sum(np.log1p(np.exp(hidden_pre)), axis=1)  # (n,)
    return float(np.mean(-bias_term - log_term))


def train_rbm(
    n_visible: int,
    n_hidden: int,
    n_samples: int,
    n_epochs: int,
    lr: float,
    seed: int,
) -> dict:
    """
    Train a binary RBM with CD-1 on MNIST.

    Explosion detection: if lr * grad_norm > GRAD_EXPLOSION_THRESHOLD,
    the weight update would cause gradient explosion; stop and report.

    Returns a result dict conforming to the qa_kona_ebm_mnist_v1 cert schema.
    """
    rng = np.random.default_rng(seed)

    # Load and shuffle data once
    data = load_mnist_images(n_samples)                # (n_samples, 784)
    idx = rng.permutation(n_samples)
    data = data[idx]

    # Initialise weights and biases
    W = rng.normal(0.0, 0.01, size=(n_hidden, n_visible))   # (n_hidden, n_visible)
    b = np.zeros(n_visible)                                  # visible bias
    c = np.zeros(n_hidden)                                   # hidden bias

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
            v0 = data[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]  # (bs, n_visible)

            # Positive phase
            h0_prob = _sigmoid(v0 @ W.T + c)          # (bs, n_hidden)
            h0 = (rng.random(h0_prob.shape) < h0_prob).astype(np.float64)

            # Negative phase - reconstruct using probabilities
            v1_prob = _sigmoid(h0 @ W + b)             # (bs, n_visible)
            h1_prob = _sigmoid(v1_prob @ W.T + c)      # (bs, n_hidden)

            # Gradients (use h0_prob for positive, h1_prob for negative)
            dW = (h0_prob.T @ v0 - h1_prob.T @ v1_prob) / BATCH_SIZE  # (n_hidden, n_visible)
            db_update = np.mean(v0 - v1_prob, axis=0)
            dc_update = np.mean(h0_prob - h1_prob, axis=0)

            grad_norm = float(np.linalg.norm(dW, ord="fro"))
            # The effective update magnitude scaled by lr is what causes explosion
            update_norm = lr * grad_norm
            epoch_grad_norms.append(grad_norm)

            # Reconstruction error (MSE between v0 and v1 probabilities)
            recon_err = float(np.mean((v0 - v1_prob) ** 2))
            epoch_recon_errors.append(recon_err)

            # Check for explosion: if the weight update norm exceeds threshold
            if update_norm > GRAD_EXPLOSION_THRESHOLD:
                exploded = True
                explosion_update_norm = update_norm
                break

            # Apply updates
            W += lr * dW
            b += lr * db_update
            c += lr * dc_update

        if exploded:
            break

        energy = _free_energy(data, W, b, c)
        energy_per_epoch.append(round(energy, 6))
        recon_error_per_epoch.append(round(float(np.mean(epoch_recon_errors)), 6))
        grad_norm_per_epoch.append(round(float(np.mean(epoch_grad_norms)), 6))

    if exploded:
        # Record one entry for the epoch that exploded
        energy_per_epoch.append(round(_free_energy(data, W, b, c), 6))
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

    # Trace hash over energy_per_epoch rounded to 6dp
    trace_payload = json.dumps(energy_per_epoch, separators=(",", ":")).encode()
    trace_hash = hashlib.sha256(trace_payload).hexdigest()

    final_weights_norm = round(float(np.linalg.norm(W, ord="fro")), 6)

    return {
        "n_visible": n_visible,
        "n_hidden": n_hidden,
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
    }


if __name__ == "__main__":
    import json
    result = train_rbm(784, 64, 1000, 5, 0.01, 42)
    print(json.dumps(result, indent=2))
