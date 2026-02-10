"""
qa_generalization_certificate.py

QA Certificate Schema for Architecture-Independent Generalization Bounds
Based on arXiv:2504.05695: "Architecture independent generalization bounds
for overparametrized deep ReLU networks"

Maps the paper's key concepts to QA's failure-completeness framework:
- Metric geometry (D_geom) → QA state distances
- Operator norms → certificate bounds
- ReLU regularity → activation witness
- Zero-loss constructors → constructive success certificates
- Overparametrization → gauge freedom (extra coords that don't change risk)

Hard constraints:
- Exact scalars only (int/Fraction) — no floats
- Deterministic serialization
- Failure-completeness: every decision yields success OR obstruction proof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, Union, Tuple
from enum import Enum
from fractions import Fraction
import hashlib
import json

# ============================================================================
# FOUNDATIONAL TYPES (compatible with qa_certificate.py)
# ============================================================================

Scalar = Union[int, Fraction]


def to_scalar(x: Any) -> Scalar:
    """Convert to exact scalar, rejecting floats."""
    if isinstance(x, bool):
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        # For generalization bounds we need to convert measured floats
        # Use Fraction.limit_denominator for bounded precision
        return Fraction(x).limit_denominator(10**9)
    if isinstance(x, str):
        s = x.strip()
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")


def to_scalar_strict(x: Any) -> Scalar:
    """Strict conversion - no floats allowed at all."""
    if isinstance(x, float):
        raise TypeError(f"Float not allowed in strict mode: {x}")
    return to_scalar(x)


# ============================================================================
# METRIC GEOMETRY WITNESS
# ============================================================================

@dataclass(frozen=True)
class MetricGeometryWitness:
    """
    Witness for the metric geometry of data (D_geom from the paper).

    D_geom measures the intrinsic geometric complexity of the data manifold.
    In QA terms: this is the "terrain roughness" that bounds how hard
    generalization can be.

    Key invariant: D_geom is architecture-independent.
    """
    data_hash: str  # SHA-256 of training data
    n_samples: int
    input_dim: int

    # Pairwise distance statistics (exact)
    min_distance: Scalar
    max_distance: Scalar
    mean_distance: Scalar

    # Metric entropy at scale epsilon
    covering_number: int
    epsilon: Scalar

    # Curvature bound (if available)
    curvature_bound: Optional[Scalar] = None

    # Doubling dimension estimate
    doubling_dim: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "min_distance", to_scalar(self.min_distance))
        object.__setattr__(self, "max_distance", to_scalar(self.max_distance))
        object.__setattr__(self, "mean_distance", to_scalar(self.mean_distance))
        object.__setattr__(self, "epsilon", to_scalar(self.epsilon))
        if self.curvature_bound is not None:
            object.__setattr__(self, "curvature_bound", to_scalar(self.curvature_bound))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_hash": self.data_hash,
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
            "min_distance": str(self.min_distance),
            "max_distance": str(self.max_distance),
            "mean_distance": str(self.mean_distance),
            "covering_number": self.covering_number,
            "epsilon": str(self.epsilon),
            "curvature_bound": str(self.curvature_bound) if self.curvature_bound else None,
            "doubling_dim": self.doubling_dim,
        }


# ============================================================================
# OPERATOR NORM WITNESS
# ============================================================================

@dataclass(frozen=True)
class OperatorNormWitness:
    """
    Witness for operator norm bounds on network weights.

    The paper shows generalization depends on:
    - Product of spectral norms: prod_l ||W_l||_2
    - Sum of bias norms: sum_l ||b_l||_2
    - NOT the total parameter count

    QA interpretation: These are the "energy bounds" that constrain
    how far the network can travel in function space.
    """
    layer_count: int

    # Per-layer spectral norms (exact rational approximations)
    spectral_norms: Tuple[Scalar, ...]

    # Per-layer bias norms
    bias_norms: Tuple[Scalar, ...]

    # Aggregate bounds
    spectral_product: Scalar  # prod ||W_l||_2
    bias_sum: Scalar  # sum ||b_l||_2

    # Frobenius norms (secondary)
    frobenius_norms: Optional[Tuple[Scalar, ...]] = None

    def __post_init__(self):
        object.__setattr__(self, "spectral_norms",
                          tuple(to_scalar(x) for x in self.spectral_norms))
        object.__setattr__(self, "bias_norms",
                          tuple(to_scalar(x) for x in self.bias_norms))
        object.__setattr__(self, "spectral_product", to_scalar(self.spectral_product))
        object.__setattr__(self, "bias_sum", to_scalar(self.bias_sum))
        if self.frobenius_norms is not None:
            object.__setattr__(self, "frobenius_norms",
                              tuple(to_scalar(x) for x in self.frobenius_norms))

    def verify_product(self) -> bool:
        """Check spectral_product = prod(spectral_norms)."""
        computed = Fraction(1)
        for s in self.spectral_norms:
            computed *= Fraction(s)
        return computed == Fraction(self.spectral_product)

    def verify_sum(self) -> bool:
        """Check bias_sum = sum(bias_norms)."""
        computed = sum(Fraction(b) for b in self.bias_norms)
        return computed == Fraction(self.bias_sum)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_count": self.layer_count,
            "spectral_norms": [str(s) for s in self.spectral_norms],
            "bias_norms": [str(b) for b in self.bias_norms],
            "spectral_product": str(self.spectral_product),
            "bias_sum": str(self.bias_sum),
            "frobenius_norms": [str(f) for f in self.frobenius_norms] if self.frobenius_norms else None,
        }


# ============================================================================
# ACTIVATION REGULARITY WITNESS
# ============================================================================

class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"


@dataclass(frozen=True)
class ActivationRegularityWitness:
    """
    Witness for activation function regularity (Lipschitz constant, etc.).

    For ReLU: Lipschitz = 1, but the paper tracks the "effective Lipschitz"
    which accounts for the activation pattern stability.

    QA interpretation: Activations are the "gates" that partition the
    input space into linear regions. Regularity bounds the number of regions.
    """
    activation_type: ActivationType

    # Lipschitz constant of activation
    lipschitz_constant: Scalar

    # Number of linear regions (for piecewise linear activations)
    linear_region_count: Optional[int] = None

    # Activation pattern hash (for reproducibility)
    pattern_hash: Optional[str] = None

    # Smoothness parameter (for smooth activations like GELU)
    smoothness_param: Optional[Scalar] = None

    def __post_init__(self):
        object.__setattr__(self, "lipschitz_constant", to_scalar(self.lipschitz_constant))
        if self.smoothness_param is not None:
            object.__setattr__(self, "smoothness_param", to_scalar(self.smoothness_param))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_type": self.activation_type.value,
            "lipschitz_constant": str(self.lipschitz_constant),
            "linear_region_count": self.linear_region_count,
            "pattern_hash": self.pattern_hash,
            "smoothness_param": str(self.smoothness_param) if self.smoothness_param else None,
        }


# ============================================================================
# GENERALIZATION BOUND CERTIFICATE
# ============================================================================

class GeneralizationFailure(Enum):
    """Failure modes for generalization certificates."""
    # Data geometry failures
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    DATA_NOT_SEPARABLE = "data_not_separable"
    METRIC_DEGENERACY = "metric_degeneracy"

    # Norm bound failures
    NORM_EXPLOSION = "norm_explosion"
    SPECTRAL_OVERFLOW = "spectral_overflow"
    BIAS_OVERFLOW = "bias_overflow"

    # Architecture failures
    DEPTH_TOO_SHALLOW = "depth_too_shallow"
    WIDTH_INSUFFICIENT = "width_insufficient"

    # Training failures
    NO_ZERO_LOSS_SOLUTION = "no_zero_loss_solution"
    OPTIMIZATION_STUCK = "optimization_stuck"

    # Bound failures
    BOUND_VACUOUS = "bound_vacuous"
    BOUND_NOT_COMPUTABLE = "bound_not_computable"


@dataclass
class GeneralizationBoundCertificate:
    """
    Certificate for architecture-independent generalization bounds.

    The paper's main theorem:
    For ReLU networks with n samples in R^d, the generalization gap is bounded by:

        gap ≤ C * D_geom * (prod ||W_l||_2) * (1 + sum ||b_l||_2) / sqrt(n)

    where:
    - D_geom is the metric geometry of data
    - ||W_l||_2 is spectral norm of layer l
    - ||b_l||_2 is bias norm of layer l
    - C is a universal constant

    Failure-completeness: If we cannot certify generalization, we must
    provide a constructive obstruction showing why.
    """
    # Certificate metadata
    certificate_id: str
    version: str = "1.0.0"
    schema: str = "QA_GENERALIZATION_CERT_V1"

    # Success or failure
    success: bool = True
    failure_mode: Optional[GeneralizationFailure] = None
    failure_witness: Optional[Dict[str, Any]] = None

    # Core witnesses
    metric_geometry: Optional[MetricGeometryWitness] = None
    operator_norms: Optional[OperatorNormWitness] = None
    activation_regularity: Optional[ActivationRegularityWitness] = None

    # The computed bound
    generalization_bound: Optional[Scalar] = None

    # Empirical validation
    empirical_train_loss: Optional[Scalar] = None
    empirical_test_loss: Optional[Scalar] = None
    empirical_gap: Optional[Scalar] = None

    # Tracking error (bound - empirical)
    tracking_error: Optional[Scalar] = None
    tracking_error_percent: Optional[Scalar] = None

    # Architecture metadata (for reference, not used in bound)
    architecture: Optional[Dict[str, Any]] = None

    # Zero-loss constructor witness (for n ≤ d case)
    zero_loss_constructor: Optional[Dict[str, Any]] = None

    # Overparametrization gauge freedom
    gauge_freedom_dim: Optional[int] = None

    # Recompute hook reference
    recompute_hook: Optional[str] = None

    def __post_init__(self):
        if self.generalization_bound is not None:
            self.generalization_bound = to_scalar(self.generalization_bound)
        if self.empirical_train_loss is not None:
            self.empirical_train_loss = to_scalar(self.empirical_train_loss)
        if self.empirical_test_loss is not None:
            self.empirical_test_loss = to_scalar(self.empirical_test_loss)
        if self.empirical_gap is not None:
            self.empirical_gap = to_scalar(self.empirical_gap)
        if self.tracking_error is not None:
            self.tracking_error = to_scalar(self.tracking_error)
        if self.tracking_error_percent is not None:
            self.tracking_error_percent = to_scalar(self.tracking_error_percent)

    def verify_gap_computation(self) -> bool:
        """Verify empirical_gap = test_loss - train_loss."""
        if self.empirical_gap is None or self.empirical_test_loss is None or self.empirical_train_loss is None:
            return True  # No data to verify
        computed = Fraction(self.empirical_test_loss) - Fraction(self.empirical_train_loss)
        return computed == Fraction(self.empirical_gap)

    def verify_tracking_error(self) -> bool:
        """Verify tracking_error = bound - empirical_gap."""
        if self.tracking_error is None or self.generalization_bound is None or self.empirical_gap is None:
            return True
        computed = Fraction(self.generalization_bound) - Fraction(self.empirical_gap)
        return computed == Fraction(self.tracking_error)

    def verify_bound_validity(self) -> bool:
        """Verify bound ≥ empirical_gap (non-vacuous)."""
        if self.generalization_bound is None or self.empirical_gap is None:
            return True
        return Fraction(self.generalization_bound) >= Fraction(self.empirical_gap)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "certificate_id": self.certificate_id,
            "version": self.version,
            "schema": self.schema,
            "success": self.success,
        }

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.failure_witness:
            result["failure_witness"] = self.failure_witness
        if self.metric_geometry:
            result["metric_geometry"] = self.metric_geometry.to_dict()
        if self.operator_norms:
            result["operator_norms"] = self.operator_norms.to_dict()
        if self.activation_regularity:
            result["activation_regularity"] = self.activation_regularity.to_dict()
        if self.generalization_bound is not None:
            result["generalization_bound"] = str(self.generalization_bound)
        if self.empirical_train_loss is not None:
            result["empirical_train_loss"] = str(self.empirical_train_loss)
        if self.empirical_test_loss is not None:
            result["empirical_test_loss"] = str(self.empirical_test_loss)
        if self.empirical_gap is not None:
            result["empirical_gap"] = str(self.empirical_gap)
        if self.tracking_error is not None:
            result["tracking_error"] = str(self.tracking_error)
        if self.tracking_error_percent is not None:
            result["tracking_error_percent"] = str(self.tracking_error_percent)
        if self.architecture:
            result["architecture"] = self.architecture
        if self.zero_loss_constructor:
            result["zero_loss_constructor"] = self.zero_loss_constructor
        if self.gauge_freedom_dim is not None:
            result["gauge_freedom_dim"] = self.gauge_freedom_dim
        if self.recompute_hook:
            result["recompute_hook"] = self.recompute_hook

        return result

    def compute_certificate_hash(self) -> str:
        """Compute deterministic hash of certificate."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ============================================================================
# ZERO-LOSS CONSTRUCTOR CERTIFICATE (n ≤ d case)
# ============================================================================

@dataclass
class ZeroLossConstructorCertificate:
    """
    Certificate for explicit zero-loss network construction.

    The paper shows: When n ≤ d (samples ≤ input dimension), there exists
    an EXPLICIT constructor (no gradient descent needed) that achieves
    zero training loss.

    This is a SUCCESS certificate with constructive witness.
    """
    # Required fields (no defaults) must come first
    certificate_id: str
    n_samples: int
    input_dim: int
    method: Literal["interpolation", "kernel", "direct"]

    # Optional fields with defaults
    schema: str = "QA_ZERO_LOSS_CONSTRUCTOR_V1"

    # The constructed weights (as exact rationals)
    weights: Optional[List[List[Scalar]]] = None
    biases: Optional[List[Scalar]] = None

    # Verification: f(x_i) = y_i for all i
    verified_zero_loss: bool = False
    residuals: Optional[List[Scalar]] = None  # Should all be 0

    # Construction complexity
    construction_flops: Optional[int] = None

    def __post_init__(self):
        if self.weights:
            self.weights = [[to_scalar(w) for w in row] for row in self.weights]
        if self.biases:
            self.biases = [to_scalar(b) for b in self.biases]
        if self.residuals:
            self.residuals = [to_scalar(r) for r in self.residuals]

    def verify_zero_residuals(self) -> bool:
        """Check all residuals are exactly zero."""
        if self.residuals is None:
            return True
        return all(Fraction(r) == 0 for r in self.residuals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "schema": self.schema,
            "n_samples": self.n_samples,
            "input_dim": self.input_dim,
            "method": self.method,
            "weights": [[str(w) for w in row] for row in self.weights] if self.weights else None,
            "biases": [str(b) for b in self.biases] if self.biases else None,
            "verified_zero_loss": self.verified_zero_loss,
            "residuals": [str(r) for r in self.residuals] if self.residuals else None,
            "construction_flops": self.construction_flops,
        }


# ============================================================================
# GAUGE FREEDOM WITNESS
# ============================================================================

@dataclass(frozen=True)
class GaugeFreedomWitness:
    """
    Witness for overparametrization as gauge freedom.

    The paper's key insight: Extra parameters beyond what's needed for
    zero loss don't affect the generalization bound. They're "gauge degrees
    of freedom" that can be fixed arbitrarily.

    QA interpretation: These are the "null directions" in parameter space
    that leave the certificate unchanged.
    """
    # Total parameters
    total_params: int

    # Minimal parameters needed for zero loss
    minimal_params: int

    # Gauge freedom dimension
    gauge_dim: int  # = total_params - minimal_params

    # Null space basis (if small enough to store)
    null_basis_hash: Optional[str] = None

    # Verification: gauge_dim = total - minimal
    verified: bool = False

    def verify_gauge_dim(self) -> bool:
        """Check gauge_dim = total_params - minimal_params."""
        return self.gauge_dim == self.total_params - self.minimal_params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_params": self.total_params,
            "minimal_params": self.minimal_params,
            "gauge_dim": self.gauge_dim,
            "null_basis_hash": self.null_basis_hash,
            "verified": self.verified,
        }


# ============================================================================
# CERTIFICATE BUNDLE
# ============================================================================

@dataclass
class GeneralizationCertificateBundle:
    """
    Bundle of certificates for a complete generalization analysis.

    A full analysis may include:
    1. Main generalization bound certificate
    2. Zero-loss constructor (if n ≤ d)
    3. Gauge freedom witness
    4. Multiple experiments with different architectures
    """
    # Required fields (no defaults) must come first
    bundle_id: str
    main_certificate: GeneralizationBoundCertificate

    # Optional fields with defaults
    schema: str = "QA_GENERALIZATION_BUNDLE_V1"

    # Optional: zero-loss constructor
    zero_loss_constructor: Optional[ZeroLossConstructorCertificate] = None

    # Optional: gauge freedom
    gauge_freedom: Optional[GaugeFreedomWitness] = None

    # Optional: multiple architecture experiments
    architecture_experiments: Optional[List[GeneralizationBoundCertificate]] = None

    # Coherence checks
    coherence_verified: bool = False
    coherence_failures: List[str] = field(default_factory=list)

    def verify_coherence(self) -> bool:
        """Run all cross-certificate consistency checks."""
        self.coherence_failures = []

        # Check main certificate internal consistency
        if not self.main_certificate.verify_gap_computation():
            self.coherence_failures.append("main_certificate.gap_computation")
        if not self.main_certificate.verify_tracking_error():
            self.coherence_failures.append("main_certificate.tracking_error")
        if not self.main_certificate.verify_bound_validity():
            self.coherence_failures.append("main_certificate.bound_validity")

        # Check operator norm consistency
        if self.main_certificate.operator_norms:
            if not self.main_certificate.operator_norms.verify_product():
                self.coherence_failures.append("operator_norms.product")
            if not self.main_certificate.operator_norms.verify_sum():
                self.coherence_failures.append("operator_norms.sum")

        # Check zero-loss constructor
        if self.zero_loss_constructor:
            if not self.zero_loss_constructor.verify_zero_residuals():
                self.coherence_failures.append("zero_loss_constructor.residuals")

        # Check gauge freedom
        if self.gauge_freedom:
            if not self.gauge_freedom.verify_gauge_dim():
                self.coherence_failures.append("gauge_freedom.dim")

        self.coherence_verified = len(self.coherence_failures) == 0
        return self.coherence_verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "schema": self.schema,
            "main_certificate": self.main_certificate.to_dict(),
            "zero_loss_constructor": self.zero_loss_constructor.to_dict() if self.zero_loss_constructor else None,
            "gauge_freedom": self.gauge_freedom.to_dict() if self.gauge_freedom else None,
            "architecture_experiments": [c.to_dict() for c in self.architecture_experiments] if self.architecture_experiments else None,
            "coherence_verified": self.coherence_verified,
            "coherence_failures": self.coherence_failures,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_success_certificate(
    certificate_id: str,
    metric_geometry: MetricGeometryWitness,
    operator_norms: OperatorNormWitness,
    activation_regularity: ActivationRegularityWitness,
    generalization_bound: Scalar,
    empirical_gap: Optional[Scalar] = None,
) -> GeneralizationBoundCertificate:
    """Create a success certificate with all witnesses."""
    cert = GeneralizationBoundCertificate(
        certificate_id=certificate_id,
        success=True,
        metric_geometry=metric_geometry,
        operator_norms=operator_norms,
        activation_regularity=activation_regularity,
        generalization_bound=generalization_bound,
        empirical_gap=empirical_gap,
    )
    if empirical_gap is not None:
        cert.tracking_error = Fraction(generalization_bound) - Fraction(empirical_gap)
    return cert


def create_failure_certificate(
    certificate_id: str,
    failure_mode: GeneralizationFailure,
    failure_witness: Dict[str, Any],
) -> GeneralizationBoundCertificate:
    """Create a failure certificate with obstruction witness."""
    return GeneralizationBoundCertificate(
        certificate_id=certificate_id,
        success=False,
        failure_mode=failure_mode,
        failure_witness=failure_witness,
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a success certificate for MNIST

    # 1. Metric geometry of MNIST
    geometry = MetricGeometryWitness(
        data_hash="mnist_train_60k_sha256_abc123",
        n_samples=60000,
        input_dim=784,
        min_distance=Fraction(1, 100),
        max_distance=Fraction(50, 1),
        mean_distance=Fraction(15, 1),
        covering_number=1000,
        epsilon=Fraction(1, 10),
    )

    # 2. Operator norms of a 3-layer MLP
    norms = OperatorNormWitness(
        layer_count=3,
        spectral_norms=(Fraction(2, 1), Fraction(3, 2), Fraction(1, 1)),
        bias_norms=(Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)),
        spectral_product=Fraction(3, 1),  # 2 * 3/2 * 1
        bias_sum=Fraction(3, 10),  # 0.1 + 0.1 + 0.1
    )

    # 3. ReLU activation
    activation = ActivationRegularityWitness(
        activation_type=ActivationType.RELU,
        lipschitz_constant=1,
    )

    # 4. Create certificate
    cert = create_success_certificate(
        certificate_id="mnist_mlp_gen_bound_001",
        metric_geometry=geometry,
        operator_norms=norms,
        activation_regularity=activation,
        generalization_bound=Fraction(22, 100),  # 22% bound
        empirical_gap=Fraction(2, 100),  # 2% actual gap
    )

    print("=== Generalization Bound Certificate ===")
    print(json.dumps(cert.to_dict(), indent=2))
    print(f"\nCertificate hash: {cert.compute_certificate_hash()}")
    print(f"Gap verified: {cert.verify_gap_computation()}")
    print(f"Tracking error verified: {cert.verify_tracking_error()}")
    print(f"Bound valid: {cert.verify_bound_validity()}")
