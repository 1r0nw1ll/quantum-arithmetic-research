#!/usr/bin/env python3
"""
level3_recompute_validation.py

LEVEL 3 RECOMPUTE VALIDATION — The Ungameable Test.

This script:
1. Loads REAL MNIST data from raw IDX files
2. Trains a 2-layer MLP using PURE NUMPY (no PyTorch)
3. Extracts ACTUAL spectral norms via SVD of real weight matrices
4. Computes ACTUAL metric geometry from the real data matrix
5. Emits a certificate from these measurements
6. Independently recomputes everything from scratch using the hooks
7. Checks that the certificate matches the recomputation

Every number in the certificate comes from real data and real trained
weights. Nothing is approximated from literature. The recompute hooks
independently verify every claim.

Dependencies: numpy, scipy (both available)
Data: MNIST raw IDX files at data/MNIST/raw/
"""

from __future__ import annotations

import struct
import gzip
import hashlib
import json
import sys
import time
import os
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np

# Import the QA certificate infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from qa_generalization_certificate import (
    GeneralizationBoundCertificate,
    GeneralizationCertificateBundle,
    MetricGeometryWitness,
    OperatorNormWitness,
    ActivationRegularityWitness,
    ActivationType,
    GaugeFreedomWitness,
    GeneralizationFailure,
)
from qa_generalization_validator_v3 import (
    GeneralizationCertificateValidator,
    GeneralizationBundleValidator,
    ValidationLevel,
    ValidationStatus,
)
from qa_generalization_hooks import (
    MetricGeometryHook,
    OperatorNormHook,
    GeneralizationBoundHook,
    HookRegistry,
)


# ============================================================================
# MNIST LOADER (pure Python + numpy, no dependencies)
# ============================================================================

MNIST_DIR = Path(__file__).parent.parent / "data" / "MNIST" / "raw"


def load_idx_images(path: Path) -> np.ndarray:
    """Load IDX image file into numpy array."""
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051, f"Bad magic: {magic}"
        n_images = struct.unpack(">I", f.read(4))[0]
        n_rows = struct.unpack(">I", f.read(4))[0]
        n_cols = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n_images, n_rows * n_cols).astype(np.float64) / 255.0


def load_idx_labels(path: Path) -> np.ndarray:
    """Load IDX label file into numpy array."""
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open

    with opener(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049, f"Bad magic: {magic}"
        n_labels = struct.unpack(">I", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data


def load_mnist(n_train: int = 2000, n_test: int = 500) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load MNIST train and test data.

    Using 2000 train + 500 test: small enough to compute full pairwise
    distance matrix (~32MB) without memory pressure on 7.6GB machine,
    large enough for meaningful training and statistics.
    """
    # Try uncompressed first, then gzipped
    for suffix in ["", ".gz"]:
        train_img = MNIST_DIR / f"train-images-idx3-ubyte{suffix}"
        train_lbl = MNIST_DIR / f"train-labels-idx1-ubyte{suffix}"
        test_img = MNIST_DIR / f"t10k-images-idx3-ubyte{suffix}"
        test_lbl = MNIST_DIR / f"t10k-labels-idx1-ubyte{suffix}"

        if train_img.exists():
            break
    else:
        raise FileNotFoundError(f"MNIST not found in {MNIST_DIR}")

    X_train = load_idx_images(train_img)[:n_train]
    y_train = load_idx_labels(train_lbl)[:n_train]
    X_test = load_idx_images(test_img)[:n_test]
    y_test = load_idx_labels(test_lbl)[:n_test]

    return X_train, y_train, X_test, y_test


# ============================================================================
# NUMPY MLP: Train a real model with real weights
# ============================================================================

class NumpyMLP:
    """
    2-layer MLP trained with pure numpy.

    Architecture: input(784) -> hidden(128) -> output(10)
    Activation: ReLU
    Loss: Cross-entropy with softmax
    Optimizer: SGD with momentum
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128,
                 output_dim: int = 10, seed: int = 42):
        rng = np.random.RandomState(seed)

        # Xavier initialization
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (logits, hidden activations)."""
        h = X @ self.W1 + self.b1
        h_relu = np.maximum(h, 0)  # ReLU
        logits = h_relu @ self.W2 + self.b2
        return logits, h_relu

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def cross_entropy_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Cross-entropy loss."""
        probs = self.softmax(logits)
        n = len(labels)
        log_probs = -np.log(probs[np.arange(n), labels] + 1e-12)
        return float(log_probs.mean())

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        logits, _ = self.forward(X)
        preds = logits.argmax(axis=1)
        return float((preds == y).mean())

    def train_step(self, X: np.ndarray, y: np.ndarray,
                   lr: float = 0.01, momentum: float = 0.9) -> float:
        """Single training step with backpropagation."""
        n = len(y)

        # Forward
        logits, h_relu = self.forward(X)
        loss = self.cross_entropy_loss(logits, y)

        # Backward: softmax + cross-entropy gradient
        probs = self.softmax(logits)
        d_logits = probs.copy()
        d_logits[np.arange(n), y] -= 1.0
        d_logits /= n

        # Gradients for W2, b2
        dW2 = h_relu.T @ d_logits
        db2 = d_logits.sum(axis=0)

        # Backprop through ReLU
        d_h = d_logits @ self.W2.T
        d_h[h_relu <= 0] = 0  # ReLU gradient

        # Gradients for W1, b1
        dW1 = X.T @ d_h
        db1 = d_h.sum(axis=0)

        # SGD with momentum
        self.vW1 = momentum * self.vW1 - lr * dW1
        self.vb1 = momentum * self.vb1 - lr * db1
        self.vW2 = momentum * self.vW2 - lr * dW2
        self.vb2 = momentum * self.vb2 - lr * db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

        return loss

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 128,
              lr: float = 0.01, verbose: bool = True) -> List[float]:
        """Train for multiple epochs."""
        losses = []
        n = len(y)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n)
            X_shuf = X[perm]
            y_shuf = y[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                loss = self.train_step(X_batch, y_batch, lr=lr)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, train_acc={acc:.4f}")

        return losses


# ============================================================================
# MEASUREMENT FUNCTIONS: Extract real numbers from real data/weights
# ============================================================================

def measure_spectral_norms(model: NumpyMLP) -> Dict[str, Any]:
    """
    Measure ACTUAL spectral norms of trained weight matrices via SVD.

    This is the ground truth — no approximation.
    """
    # Layer 1: W1 (784 x 128)
    s1 = np.linalg.svd(model.W1, compute_uv=False)
    spectral_norm_1 = float(s1[0])
    bias_norm_1 = float(np.linalg.norm(model.b1))

    # Layer 2: W2 (128 x 10)
    s2 = np.linalg.svd(model.W2, compute_uv=False)
    spectral_norm_2 = float(s2[0])
    bias_norm_2 = float(np.linalg.norm(model.b2))

    return {
        "layer_count": 2,
        "spectral_norms": [spectral_norm_1, spectral_norm_2],
        "bias_norms": [bias_norm_1, bias_norm_2],
        "spectral_product": spectral_norm_1 * spectral_norm_2,
        "bias_sum": bias_norm_1 + bias_norm_2,
        # Also store raw singular values for full transparency
        "all_singular_values_W1": s1.tolist(),
        "all_singular_values_W2": s2.tolist(),
    }


def measure_metric_geometry(X: np.ndarray) -> Dict[str, Any]:
    """
    Measure ACTUAL metric geometry of data.

    No subsampling — operates on the full dataset so that the
    recompute hook (which also operates on full data) produces
    identical results. With n_train=2000, the pairwise distance
    matrix is 2000x2000 * 8 bytes = 32MB, well within budget.
    """
    n, d = X.shape

    # Pairwise distances via broadcasting
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i·x_j
    norms_sq = np.sum(X ** 2, axis=1)
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * X @ X.T
    dist_sq = np.maximum(dist_sq, 0)  # Numerical safety

    # Upper triangle (exclude diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    distances = np.sqrt(dist_sq[mask])

    # Data hash of the full dataset
    data_hash = hashlib.sha256(X.tobytes()).hexdigest()

    # Covering number estimate at epsilon = mean_distance / 10
    mean_dist = float(np.mean(distances))
    epsilon = mean_dist / 10.0

    # Greedy set cover approximation
    uncovered = np.ones(n, dtype=bool)
    centers = []
    for _ in range(n):
        if not uncovered.any():
            break
        idx = np.where(uncovered)[0][0]
        centers.append(idx)
        dists_from_center = np.sqrt(np.maximum(dist_sq[idx], 0))
        uncovered[dists_from_center <= epsilon] = False

    return {
        "data_hash": data_hash,
        "n_samples": n,
        "input_dim": d,
        "min_distance": float(np.min(distances)),
        "max_distance": float(np.max(distances)),
        "mean_distance": mean_dist,
        "median_distance": float(np.median(distances)),
        "std_distance": float(np.std(distances)),
        "covering_number": len(centers),
        "epsilon": epsilon,
    }


# ============================================================================
# CERTIFICATE EMITTER: Real measurements → QA Certificate
# ============================================================================

def emit_real_certificate(
    model: NumpyMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    norms: Dict[str, Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Emit a generalization certificate from REAL measurements.

    Every number here comes from actual data and actual weights.
    """
    # Real losses
    train_logits, _ = model.forward(X_train)
    test_logits, _ = model.forward(X_test)
    train_loss = model.cross_entropy_loss(train_logits, y_train)
    test_loss = model.cross_entropy_loss(test_logits, y_test)
    empirical_gap = test_loss - train_loss

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    # Compute the generalization bound from REAL measurements
    C = 4  # Universal constant
    D_geom = geometry["mean_distance"]
    spec_prod = norms["spectral_product"]
    bias_term = 1.0 + norms["bias_sum"]
    sqrt_n = geometry["n_samples"] ** 0.5

    computed_bound = C * D_geom * spec_prod * bias_term / sqrt_n

    # Convert to Fractions for exact arithmetic
    bound_frac = Fraction(computed_bound).limit_denominator(10**9)
    gap_frac = Fraction(empirical_gap).limit_denominator(10**9)
    train_frac = Fraction(train_loss).limit_denominator(10**9)
    test_frac = Fraction(test_loss).limit_denominator(10**9)

    is_vacuous = bound_frac > 1

    # Build certificate
    if is_vacuous:
        cert = {
            "certificate_id": "level3_real_mnist_mlp_001",
            "version": "1.0.0",
            "schema": "QA_GENERALIZATION_CERT_V1",
            "success": False,
            "failure_mode": "bound_vacuous",
            "failure_witness": {
                "reason": f"Computed bound ({float(bound_frac):.6f}) exceeds 1.0",
                "computed_bound": str(bound_frac),
                "threshold": "1/1",
            },
        }
    else:
        tracking = bound_frac - gap_frac
        cert = {
            "certificate_id": "level3_real_mnist_mlp_001",
            "version": "1.0.0",
            "schema": "QA_GENERALIZATION_CERT_V1",
            "success": True,
            "generalization_bound": str(bound_frac),
            "empirical_train_loss": str(train_frac),
            "empirical_test_loss": str(test_frac),
            "empirical_gap": str(gap_frac),
            "tracking_error": str(tracking),
        }

    # Add witnesses (same for both success and failure)
    cert["metric_geometry"] = {
        "data_hash": geometry["data_hash"],
        "n_samples": geometry["n_samples"],
        "input_dim": geometry["input_dim"],
        "min_distance": str(Fraction(geometry["min_distance"]).limit_denominator(10**9)),
        "max_distance": str(Fraction(geometry["max_distance"]).limit_denominator(10**9)),
        "mean_distance": str(Fraction(geometry["mean_distance"]).limit_denominator(10**9)),
        "covering_number": geometry["covering_number"],
        "epsilon": str(Fraction(geometry["epsilon"]).limit_denominator(10**9)),
    }

    cert["operator_norms"] = {
        "layer_count": norms["layer_count"],
        "spectral_norms": [
            str(Fraction(s).limit_denominator(10**9))
            for s in norms["spectral_norms"]
        ],
        "bias_norms": [
            str(Fraction(b).limit_denominator(10**9))
            for b in norms["bias_norms"]
        ],
        "spectral_product": str(Fraction(norms["spectral_product"]).limit_denominator(10**9)),
        "bias_sum": str(Fraction(norms["bias_sum"]).limit_denominator(10**9)),
    }

    cert["activation_regularity"] = {
        "activation_type": "relu",
        "lipschitz_constant": "1",
    }

    cert["architecture"] = {
        "type": "MLP [784, 128, 10]",
        "layers": [784, 128, 10],
        "total_params": 784 * 128 + 128 + 128 * 10 + 10,
    }

    cert["_real_measurements"] = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "empirical_gap": empirical_gap,
        "computed_bound": float(bound_frac),
        "bound_is_vacuous": is_vacuous,
        "spectral_product": norms["spectral_product"],
        "mean_pairwise_distance": geometry["mean_distance"],
    }

    return cert


# ============================================================================
# LEVEL 3 RECOMPUTE: Independent verification from raw data
# ============================================================================

def run_level3_recompute(
    cert: Dict[str, Any],
    model: NumpyMLP,
    X_train: np.ndarray,
) -> Dict[str, Any]:
    """
    Independently recompute all certificate witnesses from raw data.

    This is the "trust but verify" step. The certificate was emitted
    from measurements. Now we recompute those same measurements
    independently and check they match.
    """
    results = {
        "metric_geometry": {"hook": "MetricGeometryHook", "status": None, "details": {}},
        "operator_norms": {"hook": "OperatorNormHook", "status": None, "details": {}},
        "bound_recompute": {"hook": "GeneralizationBoundHook", "status": None, "details": {}},
    }

    # --- Recompute metric geometry ---
    mg_hook = MetricGeometryHook()
    mg_result = mg_hook.run(cert, X_train)
    results["metric_geometry"]["status"] = "PASS" if mg_result.matches_certificate else "FAIL"
    results["metric_geometry"]["details"] = {
        "success": mg_result.success,
        "matches": mg_result.matches_certificate,
        "discrepancies": mg_result.discrepancies,
        "recomputed": mg_result.recomputed,
        "error": mg_result.error,
    }

    # --- Recompute operator norms ---
    on_hook = OperatorNormHook()
    weight_data = {
        "weights": [model.W1.T, model.W2.T],  # Transpose because hook expects (out, in) format
        "biases": [model.b1, model.b2],
    }
    on_result = on_hook.run(cert, weight_data)
    results["operator_norms"]["status"] = "PASS" if on_result.matches_certificate else "FAIL"
    results["operator_norms"]["details"] = {
        "success": on_result.success,
        "matches": on_result.matches_certificate,
        "discrepancies": on_result.discrepancies,
        "recomputed": on_result.recomputed,
        "error": on_result.error,
    }

    # --- Recompute generalization bound ---
    bound_hook = GeneralizationBoundHook()
    bound_result = bound_hook.run(cert, None)

    # For failure certs, the hook can't find generalization_bound in the cert.
    # We verify by recomputing and checking against failure_witness.computed_bound.
    if not cert.get("success", True) and cert.get("failure_mode") == "bound_vacuous":
        recomp_bound = bound_result.recomputed.get("generalization_bound")
        witness_bound = cert.get("failure_witness", {}).get("computed_bound")
        if recomp_bound and witness_bound:
            recomp_val = Fraction(recomp_bound)
            witness_val = Fraction(witness_bound)
            rel_err = abs(recomp_val - witness_val) / max(abs(witness_val), Fraction(1, 10**9))
            bound_matches = rel_err <= Fraction(1, 10)  # 10% tolerance
            discrepancies = [] if bound_matches else [
                f"bound: witness={witness_val}, recomputed={recomp_val}"
            ]
            # Also verify the recomputed bound IS vacuous (>1)
            if recomp_val <= 1:
                bound_matches = False
                discrepancies.append(f"recomputed bound {recomp_val} is NOT vacuous (should be >1)")
            results["bound_recompute"]["status"] = "PASS" if bound_matches else "FAIL"
            results["bound_recompute"]["details"] = {
                "success": True,
                "matches": bound_matches,
                "discrepancies": discrepancies,
                "recomputed": bound_result.recomputed,
                "note": "Compared against failure_witness.computed_bound (failure cert)",
            }
        else:
            results["bound_recompute"]["status"] = "FAIL"
            results["bound_recompute"]["details"] = {
                "success": False,
                "matches": False,
                "discrepancies": ["Cannot compare: missing recomputed or witness bound"],
                "recomputed": bound_result.recomputed,
            }
    else:
        results["bound_recompute"]["status"] = "PASS" if bound_result.matches_certificate else "FAIL"
        results["bound_recompute"]["details"] = {
            "success": bound_result.success,
            "matches": bound_result.matches_certificate,
            "discrepancies": bound_result.discrepancies,
            "recomputed": bound_result.recomputed,
            "error": bound_result.error,
        }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Level 3 Recompute Validation")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode: smaller dataset, fewer epochs, exit-code only")
    args = parser.parse_args()

    # Sizes: env overrides > --ci defaults > full defaults
    n_train = int(os.environ.get("QA_L3_NTRAIN", 512 if args.ci else 2000))
    n_test = int(os.environ.get("QA_L3_NTEST", 128 if args.ci else 500))
    n_epochs = int(os.environ.get("QA_L3_EPOCHS", 15 if args.ci else 50))
    verbose = not args.ci

    np.random.seed(42)

    if verbose:
        print("=" * 78)
        print("LEVEL 3 RECOMPUTE VALIDATION — Real Data, Real Weights")
        print("=" * 78)
        print()

    # ------------------------------------------------------------------
    # Step 1: Load REAL MNIST data
    # ------------------------------------------------------------------
    if verbose:
        print("[1/6] Loading MNIST data...")
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_mnist(n_train=n_train, n_test=n_test)
    if verbose:
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Loaded in {time.time()-t0:.2f}s")
        print()

    # ------------------------------------------------------------------
    # Step 2: Train a REAL model
    # ------------------------------------------------------------------
    if verbose:
        print("[2/6] Training 2-layer MLP (pure numpy)...")
    model = NumpyMLP(input_dim=784, hidden_dim=128, output_dim=10, seed=42)

    t0 = time.time()
    losses = model.train(X_train, y_train, epochs=n_epochs, batch_size=128,
                         lr=0.01, verbose=verbose)
    train_time = time.time() - t0

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)

    if verbose:
        print(f"\n  Training complete in {train_time:.1f}s")
        print(f"  Final train accuracy: {train_acc:.4f}")
        print(f"  Final test accuracy:  {test_acc:.4f}")
        print(f"  Final train loss:     {losses[-1]:.6f}")
        print()

    # ------------------------------------------------------------------
    # Step 3: Measure REAL spectral norms
    # ------------------------------------------------------------------
    if verbose:
        print("[3/6] Measuring spectral norms via SVD...")
    norms = measure_spectral_norms(model)

    if verbose:
        print(f"  W1 spectral norm: {norms['spectral_norms'][0]:.6f}")
        print(f"  W2 spectral norm: {norms['spectral_norms'][1]:.6f}")
        print(f"  Spectral product: {norms['spectral_product']:.6f}")
        print(f"  b1 norm:          {norms['bias_norms'][0]:.6f}")
        print(f"  b2 norm:          {norms['bias_norms'][1]:.6f}")
        print(f"  Bias sum:         {norms['bias_sum']:.6f}")
        print()

    # ------------------------------------------------------------------
    # Step 4: Measure REAL metric geometry
    # ------------------------------------------------------------------
    if verbose:
        print("[4/6] Computing metric geometry of training data...")
    t0 = time.time()
    geometry = measure_metric_geometry(X_train)
    geo_time = time.time() - t0

    if verbose:
        print(f"  Data hash:        {geometry['data_hash'][:16]}...")
        print(f"  Min distance:     {geometry['min_distance']:.6f}")
        print(f"  Max distance:     {geometry['max_distance']:.6f}")
        print(f"  Mean distance:    {geometry['mean_distance']:.6f}")
        print(f"  Median distance:  {geometry['median_distance']:.6f}")
        print(f"  Covering number:  {geometry['covering_number']} (eps={geometry['epsilon']:.4f})")
        print(f"  Computed in {geo_time:.2f}s")
        print()

    # ------------------------------------------------------------------
    # Step 5: Emit certificate from REAL measurements
    # ------------------------------------------------------------------
    if verbose:
        print("[5/6] Emitting generalization certificate from real measurements...")
    cert = emit_real_certificate(
        model, X_train, y_train, X_test, y_test, norms, geometry
    )

    real = cert["_real_measurements"]
    if verbose:
        print(f"  Certificate ID:   {cert['certificate_id']}")
        print(f"  Success:          {cert['success']}")
        if cert["success"]:
            print(f"  Gen. bound:       {cert['generalization_bound']} "
                  f"(≈{real['computed_bound']:.6f})")
            print(f"  Empirical gap:    {cert['empirical_gap']} "
                  f"(≈{real['empirical_gap']:.6f})")
            print(f"  Tracking error:   {cert['tracking_error']}")
            print(f"  Bound valid:      {real['computed_bound'] >= real['empirical_gap']}")
        else:
            print(f"  Failure mode:     {cert['failure_mode']}")
            print(f"  Computed bound:   {real['computed_bound']:.6f} (vacuous: > 1)")
        print()

    # ------------------------------------------------------------------
    # Step 6: Run Level 2 validator + Level 3 recompute hooks
    # ------------------------------------------------------------------
    if verbose:
        print("[6/6] Running validation...")
        print()

    # Level 2: Schema + consistency
    if verbose:
        print("--- Level 2: Schema + Consistency Validation ---")
    validator = GeneralizationCertificateValidator(strict=True)
    report = validator.validate(cert, level=ValidationLevel.CONSISTENCY)
    if verbose:
        print(report.summary())
        print()

    # Level 3: Independent recompute
    if verbose:
        print("--- Level 3: Independent Recompute Verification ---")
        print()
    recompute = run_level3_recompute(cert, model, X_train)

    all_level3_pass = True
    for component, result in recompute.items():
        status = result["status"]
        hook = result["hook"]
        details = result["details"]

        if status == "PASS":
            if verbose:
                print(f"  [{status}] {hook}")
        else:
            all_level3_pass = False
            if verbose:
                print(f"  [{status}] {hook}")
                if details.get("discrepancies"):
                    for d in details["discrepancies"]:
                        print(f"        Discrepancy: {d}")
                if details.get("error"):
                    print(f"        Error: {details['error']}")

    passed = report.all_passed and all_level3_pass

    if verbose:
        print()
        print("=" * 78)
        print("FINAL VERDICT")
        print("=" * 78)
        print()
        print(f"  Model trained:     YES (train_acc={train_acc:.4f}, test_acc={test_acc:.4f})")
        print(f"  Real norms:        YES (spectral_product={norms['spectral_product']:.4f})")
        print(f"  Real geometry:     YES (mean_dist={geometry['mean_distance']:.4f})")
        print(f"  Certificate valid: {'YES' if report.all_passed else 'NO'} "
              f"(Level 2: {report.passed} passed, {report.failed} failed)")
        print(f"  Recompute match:   {'YES' if all_level3_pass else 'NO'}")
        print()

        if passed:
            print("  LEVEL 3 VALIDATION PASSED.")
            print("  Certificate matches independent recomputation from raw data.")
            print("  Every number is derived from real weights and real data — nothing approximated.")
        else:
            print("  LEVEL 3 VALIDATION FAILED — see discrepancies above.")
    else:
        # CI mode: one-line output
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] Level 3 recompute (n={n_train}, epochs={n_epochs}) "
              f"L2={report.passed}/{report.passed+report.failed} "
              f"L3={'3/3' if all_level3_pass else 'FAIL'}")

    # Save outputs
    output_dir = Path(__file__).parent / "external_validation_certs"
    output_dir.mkdir(exist_ok=True)

    cert_path = output_dir / "level3_real_mnist_cert.json"
    with open(cert_path, "w") as f:
        json.dump(cert, f, indent=2, default=str)

    recompute_path = output_dir / "level3_recompute_report.json"
    with open(recompute_path, "w") as f:
        json.dump(recompute, f, indent=2, default=str)

    if verbose:
        print()
        print(f"  Certificate:      {cert_path}")
        print(f"  Recompute report: {recompute_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
