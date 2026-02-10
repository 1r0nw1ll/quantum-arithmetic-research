"""
qa_generalization_hooks.py

Recomputation hooks for QA Generalization Bound Certificates.
These hooks allow independent verification of certificate witnesses
by recomputing from raw data.

Based on arXiv:2504.05695: "Architecture independent generalization bounds
for overparametrized deep ReLU networks"

Hooks implemented:
1. MetricGeometryHook: Recompute D_geom from data
2. OperatorNormHook: Recompute spectral/bias norms from weights
3. GeneralizationBoundHook: Recompute bound from witnesses
4. ZeroLossConstructorHook: Verify zero-loss construction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Any, List, Optional, Tuple, Protocol
import hashlib
import json

# Try to import numpy, but make it optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============================================================================
# HOOK PROTOCOL
# ============================================================================

class RecomputeHook(Protocol):
    """Protocol for recompute hooks."""

    @property
    def hook_id(self) -> str:
        """Unique identifier for this hook."""
        ...

    def recompute(self, certificate: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """Recompute witness from raw data."""
        ...

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> bool:
        """Verify certificate matches recomputed values."""
        ...


# ============================================================================
# BASE HOOK CLASS
# ============================================================================

@dataclass
class HookResult:
    """Result of a hook computation."""
    hook_id: str
    success: bool
    recomputed: Dict[str, Any]
    matches_certificate: bool
    discrepancies: List[str]
    error: Optional[str] = None


class BaseRecomputeHook(ABC):
    """Base class for recompute hooks."""

    @property
    @abstractmethod
    def hook_id(self) -> str:
        pass

    @abstractmethod
    def recompute(self, certificate: Dict[str, Any], data: Any) -> Dict[str, Any]:
        pass

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Default verification: compare all fields."""
        discrepancies = []
        for key, value in recomputed.items():
            if key not in certificate:
                discrepancies.append(f"Missing field in certificate: {key}")
            elif str(certificate[key]) != str(value):
                discrepancies.append(f"Mismatch in {key}: cert={certificate[key]}, recomputed={value}")
        return len(discrepancies) == 0, discrepancies

    def run(self, certificate: Dict[str, Any], data: Any) -> HookResult:
        """Run the hook and return result."""
        try:
            recomputed = self.recompute(certificate, data)
            matches, discrepancies = self.verify(certificate, recomputed)
            return HookResult(
                hook_id=self.hook_id,
                success=True,
                recomputed=recomputed,
                matches_certificate=matches,
                discrepancies=discrepancies
            )
        except Exception as e:
            return HookResult(
                hook_id=self.hook_id,
                success=False,
                recomputed={},
                matches_certificate=False,
                discrepancies=[],
                error=str(e)
            )


# ============================================================================
# METRIC GEOMETRY HOOK
# ============================================================================

class MetricGeometryHook(BaseRecomputeHook):
    """
    Recompute metric geometry (D_geom) from raw data.

    Requires numpy for efficient distance computation.
    """

    @property
    def hook_id(self) -> str:
        return "metric_geometry_v1"

    def recompute(self, certificate: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """
        Recompute metric geometry from data matrix.

        Args:
            certificate: The certificate to verify
            data: numpy array of shape (n_samples, input_dim)

        Returns:
            Dict with recomputed metric geometry fields
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for metric geometry recompute")

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        n_samples, input_dim = data.shape

        # Compute pairwise distances
        # Using broadcasting: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i.x_j
        norms_sq = np.sum(data ** 2, axis=1)
        distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * data @ data.T

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n_samples, n_samples), dtype=bool), k=1)
        distances = np.sqrt(np.maximum(distances_sq[mask], 0))

        # Compute statistics
        min_dist = float(np.min(distances))
        max_dist = float(np.max(distances))
        mean_dist = float(np.mean(distances))

        # Compute data hash
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()

        # Convert to exact rationals
        return {
            "data_hash": data_hash,
            "n_samples": n_samples,
            "input_dim": input_dim,
            "min_distance": str(Fraction(min_dist).limit_denominator(10**9)),
            "max_distance": str(Fraction(max_dist).limit_denominator(10**9)),
            "mean_distance": str(Fraction(mean_dist).limit_denominator(10**9)),
        }

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify metric geometry fields."""
        mg = certificate.get("metric_geometry", {})
        discrepancies = []

        # Check data hash
        if mg.get("data_hash") != recomputed.get("data_hash"):
            discrepancies.append(f"data_hash mismatch")

        # Check dimensions
        if mg.get("n_samples") != recomputed.get("n_samples"):
            discrepancies.append(f"n_samples: cert={mg.get('n_samples')}, recomputed={recomputed.get('n_samples')}")

        if mg.get("input_dim") != recomputed.get("input_dim"):
            discrepancies.append(f"input_dim: cert={mg.get('input_dim')}, recomputed={recomputed.get('input_dim')}")

        # Check distance bounds (allow small tolerance due to float->rational conversion)
        for field in ["min_distance", "max_distance", "mean_distance"]:
            cert_val = Fraction(mg.get(field, "0"))
            recomp_val = Fraction(recomputed.get(field, "0"))
            rel_error = abs(cert_val - recomp_val) / max(abs(cert_val), Fraction(1, 10**9))
            if rel_error > Fraction(1, 1000):  # 0.1% tolerance
                discrepancies.append(f"{field}: cert={cert_val}, recomputed={recomp_val}")

        return len(discrepancies) == 0, discrepancies


# ============================================================================
# OPERATOR NORM HOOK
# ============================================================================

class OperatorNormHook(BaseRecomputeHook):
    """
    Recompute operator norms from network weights.

    Requires numpy for SVD computation.
    """

    @property
    def hook_id(self) -> str:
        return "operator_norm_v1"

    def recompute(self, certificate: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """
        Recompute operator norms from weight matrices.

        Args:
            certificate: The certificate to verify
            data: Dict with 'weights' (list of weight matrices) and 'biases' (list of bias vectors)

        Returns:
            Dict with recomputed operator norm fields
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for operator norm recompute")

        weights = data.get("weights", [])
        biases = data.get("biases", [])

        if not weights:
            raise ValueError("No weights provided")

        spectral_norms = []
        bias_norms = []

        for i, W in enumerate(weights):
            W = np.array(W)
            # Spectral norm = largest singular value
            s = np.linalg.svd(W, compute_uv=False)
            spectral_norms.append(float(s[0]))

            # Bias norm
            if i < len(biases):
                b = np.array(biases[i])
                bias_norms.append(float(np.linalg.norm(b)))
            else:
                bias_norms.append(0.0)

        # Compute aggregates
        spectral_product = 1.0
        for s in spectral_norms:
            spectral_product *= s

        bias_sum = sum(bias_norms)

        return {
            "layer_count": len(weights),
            "spectral_norms": [str(Fraction(s).limit_denominator(10**9)) for s in spectral_norms],
            "bias_norms": [str(Fraction(b).limit_denominator(10**9)) for b in bias_norms],
            "spectral_product": str(Fraction(spectral_product).limit_denominator(10**9)),
            "bias_sum": str(Fraction(bias_sum).limit_denominator(10**9)),
        }

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify operator norm fields."""
        on = certificate.get("operator_norms", {})
        discrepancies = []

        # Check layer count
        if on.get("layer_count") != recomputed.get("layer_count"):
            discrepancies.append(f"layer_count: cert={on.get('layer_count')}, recomputed={recomputed.get('layer_count')}")

        # Check spectral norms
        cert_spectral = on.get("spectral_norms", [])
        recomp_spectral = recomputed.get("spectral_norms", [])

        if len(cert_spectral) != len(recomp_spectral):
            discrepancies.append(f"spectral_norms length mismatch")
        else:
            for i, (c, r) in enumerate(zip(cert_spectral, recomp_spectral)):
                cert_val = Fraction(c)
                recomp_val = Fraction(r)
                rel_error = abs(cert_val - recomp_val) / max(abs(cert_val), Fraction(1, 10**9))
                if rel_error > Fraction(1, 100):  # 1% tolerance
                    discrepancies.append(f"spectral_norms[{i}]: cert={cert_val}, recomputed={recomp_val}")

        return len(discrepancies) == 0, discrepancies


# ============================================================================
# GENERALIZATION BOUND HOOK
# ============================================================================

class GeneralizationBoundHook(BaseRecomputeHook):
    """
    Recompute generalization bound from witnesses.

    The paper's bound:
        gap ≤ C * D_geom * (prod ||W_l||_2) * (1 + sum ||b_l||_2) / sqrt(n)
    """

    # Universal constant from the paper (approximate)
    C_CONSTANT = Fraction(4, 1)

    @property
    def hook_id(self) -> str:
        return "generalization_bound_v1"

    def recompute(self, certificate: Dict[str, Any], data: Any = None) -> Dict[str, Any]:
        """
        Recompute generalization bound from certificate witnesses.

        Args:
            certificate: The certificate containing metric_geometry and operator_norms
            data: Not used (all data is in certificate)

        Returns:
            Dict with recomputed bound
        """
        mg = certificate.get("metric_geometry", {})
        on = certificate.get("operator_norms", {})

        # Extract values
        n_samples = mg.get("n_samples", 1)
        mean_distance = Fraction(mg.get("mean_distance", "1"))

        spectral_product = Fraction(on.get("spectral_product", "1"))
        bias_sum = Fraction(on.get("bias_sum", "0"))

        # Approximate D_geom as mean_distance (simplified)
        D_geom = mean_distance

        # Compute bound
        # gap ≤ C * D_geom * spectral_product * (1 + bias_sum) / sqrt(n)
        sqrt_n = Fraction(int(n_samples ** 0.5))  # Approximate sqrt as integer

        bound = self.C_CONSTANT * D_geom * spectral_product * (1 + bias_sum) / sqrt_n

        return {
            "generalization_bound": str(bound),
            "components": {
                "C": str(self.C_CONSTANT),
                "D_geom": str(D_geom),
                "spectral_product": str(spectral_product),
                "bias_term": str(1 + bias_sum),
                "sqrt_n": str(sqrt_n),
            }
        }

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify generalization bound."""
        discrepancies = []

        cert_bound = certificate.get("generalization_bound")
        recomp_bound = recomputed.get("generalization_bound")

        if cert_bound is None or recomp_bound is None:
            discrepancies.append("Missing generalization_bound")
            return False, discrepancies

        cert_val = Fraction(cert_bound)
        recomp_val = Fraction(recomp_bound)

        # Allow 10% tolerance due to approximations
        rel_error = abs(cert_val - recomp_val) / max(abs(recomp_val), Fraction(1, 10**9))
        if rel_error > Fraction(1, 10):
            discrepancies.append(f"generalization_bound: cert={cert_val}, recomputed={recomp_val}")

        return len(discrepancies) == 0, discrepancies


# ============================================================================
# ZERO-LOSS CONSTRUCTOR HOOK
# ============================================================================

class ZeroLossConstructorHook(BaseRecomputeHook):
    """
    Verify zero-loss construction by evaluating network on training data.

    For n ≤ d, the paper provides explicit constructors that achieve
    exactly zero training loss.
    """

    @property
    def hook_id(self) -> str:
        return "zero_loss_constructor_v1"

    def recompute(self, certificate: Dict[str, Any], data: Any) -> Dict[str, Any]:
        """
        Verify zero-loss by evaluating network.

        Args:
            certificate: The certificate with weights/biases
            data: Dict with 'X' (inputs) and 'y' (targets)

        Returns:
            Dict with residuals and verification status
        """
        if not HAS_NUMPY:
            raise RuntimeError("numpy required for zero-loss verification")

        zlc = certificate.get("zero_loss_constructor", {})

        X = np.array(data.get("X", []))
        y = np.array(data.get("y", []))

        weights = zlc.get("weights", [])
        biases = zlc.get("biases", [])

        if not weights or len(X) == 0:
            raise ValueError("Missing weights or data")

        # Forward pass through network
        h = X
        for i, (W, b) in enumerate(zip(weights, biases)):
            W = np.array([[Fraction(w) for w in row] for row in W], dtype=float)
            b = np.array([Fraction(bi) for bi in b], dtype=float)
            h = h @ W.T + b
            # ReLU (except last layer)
            if i < len(weights) - 1:
                h = np.maximum(h, 0)

        # Compute residuals
        residuals = (h.flatten() - y.flatten()).tolist()

        return {
            "residuals": [str(Fraction(r).limit_denominator(10**9)) for r in residuals],
            "max_residual": str(Fraction(max(abs(r) for r in residuals)).limit_denominator(10**9)),
            "verified_zero_loss": all(abs(r) < 1e-9 for r in residuals),
        }

    def verify(self, certificate: Dict[str, Any], recomputed: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify zero-loss construction."""
        discrepancies = []

        if not recomputed.get("verified_zero_loss", False):
            max_res = recomputed.get("max_residual", "unknown")
            discrepancies.append(f"Non-zero residuals found (max: {max_res})")

        return len(discrepancies) == 0, discrepancies


# ============================================================================
# HOOK REGISTRY
# ============================================================================

class HookRegistry:
    """Registry of available recompute hooks."""

    _hooks: Dict[str, BaseRecomputeHook] = {}

    @classmethod
    def register(cls, hook: BaseRecomputeHook) -> None:
        """Register a hook."""
        cls._hooks[hook.hook_id] = hook

    @classmethod
    def get(cls, hook_id: str) -> Optional[BaseRecomputeHook]:
        """Get a hook by ID."""
        return cls._hooks.get(hook_id)

    @classmethod
    def list_hooks(cls) -> List[str]:
        """List all registered hook IDs."""
        return list(cls._hooks.keys())

    @classmethod
    def run_all(cls, certificate: Dict[str, Any], data: Dict[str, Any]) -> List[HookResult]:
        """Run all registered hooks."""
        results = []
        for hook in cls._hooks.values():
            hook_data = data.get(hook.hook_id, data)
            results.append(hook.run(certificate, hook_data))
        return results


# Register default hooks
HookRegistry.register(MetricGeometryHook())
HookRegistry.register(OperatorNormHook())
HookRegistry.register(GeneralizationBoundHook())
HookRegistry.register(ZeroLossConstructorHook())


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def recompute_metric_geometry(certificate: Dict[str, Any], data: Any) -> HookResult:
    """Recompute and verify metric geometry."""
    hook = MetricGeometryHook()
    return hook.run(certificate, data)


def recompute_operator_norms(certificate: Dict[str, Any], weights_and_biases: Dict[str, Any]) -> HookResult:
    """Recompute and verify operator norms."""
    hook = OperatorNormHook()
    return hook.run(certificate, weights_and_biases)


def recompute_generalization_bound(certificate: Dict[str, Any]) -> HookResult:
    """Recompute and verify generalization bound."""
    hook = GeneralizationBoundHook()
    return hook.run(certificate, None)


def verify_zero_loss_constructor(certificate: Dict[str, Any], data: Dict[str, Any]) -> HookResult:
    """Verify zero-loss constructor."""
    hook = ZeroLossConstructorHook()
    return hook.run(certificate, data)


# ============================================================================
# CLI FOR TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=== QA Generalization Hooks ===\n")

    print("Registered hooks:")
    for hook_id in HookRegistry.list_hooks():
        print(f"  - {hook_id}")

    print("\n--- Testing GeneralizationBoundHook ---")

    # Create a test certificate
    test_cert = {
        "metric_geometry": {
            "n_samples": 60000,
            "mean_distance": "15/1",
        },
        "operator_norms": {
            "spectral_product": "3/1",
            "bias_sum": "3/10",
        },
        "generalization_bound": "1/4",  # Approximate
    }

    result = recompute_generalization_bound(test_cert)

    print(f"\nHook ID: {result.hook_id}")
    print(f"Success: {result.success}")
    print(f"Recomputed: {json.dumps(result.recomputed, indent=2)}")
    print(f"Matches certificate: {result.matches_certificate}")
    if result.discrepancies:
        print(f"Discrepancies: {result.discrepancies}")
    if result.error:
        print(f"Error: {result.error}")

    print("\n--- Hook system ready ---")
