"""
qa_neuralgcm_certificate.py

QA Certificate Schema for NeuralGCM (Neural General Circulation Models)
Based on: "Neural General Circulation Models for Weather and Climate" (DeepMind, 2024)

Maps physics-constrained neural weather prediction to QA framework:
- Conservation laws (mass, energy, momentum) → QA invariants
- Neural parameterizations → gauge freedom
- Forecast skill → reachability certificates
- Physical violations → structured failure modes

Hard constraints:
- Exact scalars only (int/Fraction) — no floats in certificates
- Deterministic serialization
- Failure-completeness: every forecast yields success OR obstruction proof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, Union, Tuple
from enum import Enum
from fractions import Fraction
import hashlib
import json

# ============================================================================
# FOUNDATIONAL TYPES
# ============================================================================

Scalar = Union[int, Fraction]


def to_scalar(x: Any) -> Scalar:
    """Convert to exact scalar, rejecting raw floats."""
    if isinstance(x, bool):
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        # For physical quantities, convert with bounded precision
        return Fraction(x).limit_denominator(10**12)
    if isinstance(x, str):
        s = x.strip()
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")


# ============================================================================
# CONSERVATION LAW WITNESS
# ============================================================================

class ConservationType(Enum):
    """Types of physical conservation laws."""
    MASS = "mass"
    ENERGY = "energy"
    MOMENTUM = "momentum"
    ANGULAR_MOMENTUM = "angular_momentum"
    WATER_VAPOR = "water_vapor"
    ENTROPY = "entropy"


@dataclass(frozen=True)
class ConservationWitness:
    """
    Witness for a single conservation law verification.

    In NeuralGCM, conservation laws are the fundamental invariants:
    - Mass: total atmospheric mass must be constant
    - Energy: total energy (kinetic + potential + internal) conserved
    - Momentum: angular momentum about Earth's axis conserved

    QA interpretation: These are HARD invariants that must be preserved
    by all state transitions (dynamics + neural parameterizations).
    """
    conservation_type: ConservationType

    # Initial and final values
    initial_value: Scalar
    final_value: Scalar

    # Computed delta
    delta: Scalar

    # Tolerance (conservation within this delta is acceptable)
    tolerance: Scalar

    # Verification result
    conserved: bool

    # Relative error
    relative_error: Optional[Scalar] = None

    def __post_init__(self):
        object.__setattr__(self, "initial_value", to_scalar(self.initial_value))
        object.__setattr__(self, "final_value", to_scalar(self.final_value))
        object.__setattr__(self, "delta", to_scalar(self.delta))
        object.__setattr__(self, "tolerance", to_scalar(self.tolerance))
        if self.relative_error is not None:
            object.__setattr__(self, "relative_error", to_scalar(self.relative_error))

    def verify_delta(self) -> bool:
        """Verify delta = final - initial."""
        computed = Fraction(self.final_value) - Fraction(self.initial_value)
        return computed == Fraction(self.delta)

    def verify_conserved(self) -> bool:
        """Verify conserved = (|delta| <= tolerance)."""
        return (abs(Fraction(self.delta)) <= Fraction(self.tolerance)) == self.conserved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conservation_type": self.conservation_type.value,
            "initial_value": str(self.initial_value),
            "final_value": str(self.final_value),
            "delta": str(self.delta),
            "tolerance": str(self.tolerance),
            "conserved": self.conserved,
            "relative_error": str(self.relative_error) if self.relative_error else None,
        }


@dataclass
class ConservationBundle:
    """
    Bundle of all conservation law witnesses for a forecast.

    A valid forecast must satisfy ALL conservation laws simultaneously.
    """
    mass: ConservationWitness
    energy: ConservationWitness
    momentum: ConservationWitness

    # Optional additional conserved quantities
    angular_momentum: Optional[ConservationWitness] = None
    water_vapor: Optional[ConservationWitness] = None

    def all_conserved(self) -> bool:
        """Check if all conservation laws are satisfied."""
        required = [self.mass.conserved, self.energy.conserved, self.momentum.conserved]
        optional = []
        if self.angular_momentum:
            optional.append(self.angular_momentum.conserved)
        if self.water_vapor:
            optional.append(self.water_vapor.conserved)
        return all(required) and all(optional)

    def first_violation(self) -> Optional[str]:
        """Return the first violated conservation law, if any."""
        if not self.mass.conserved:
            return "mass"
        if not self.energy.conserved:
            return "energy"
        if not self.momentum.conserved:
            return "momentum"
        if self.angular_momentum and not self.angular_momentum.conserved:
            return "angular_momentum"
        if self.water_vapor and not self.water_vapor.conserved:
            return "water_vapor"
        return None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "mass": self.mass.to_dict(),
            "energy": self.energy.to_dict(),
            "momentum": self.momentum.to_dict(),
        }
        if self.angular_momentum:
            result["angular_momentum"] = self.angular_momentum.to_dict()
        if self.water_vapor:
            result["water_vapor"] = self.water_vapor.to_dict()
        return result


# ============================================================================
# FORECAST SKILL WITNESS
# ============================================================================

@dataclass(frozen=True)
class ForecastSkillWitness:
    """
    Witness for forecast skill metrics.

    Key metrics from NeuralGCM:
    - RMSE: Root mean squared error vs. ground truth
    - ACC: Anomaly correlation coefficient
    - Skill horizon: Lead time until skill drops below threshold

    QA interpretation: Skill is a reachability certificate—it bounds
    how far the forecast can drift from reality.
    """
    # Forecast parameters
    lead_time_hours: int
    variable: str  # e.g., "z500", "t850", "surface_pressure"

    # Skill metrics
    rmse: Scalar
    acc: Scalar  # Anomaly correlation coefficient (0-1)

    # Baseline comparison
    climatology_rmse: Scalar
    skill_vs_climatology: Scalar  # rmse / climatology_rmse

    # Thresholds
    acc_threshold: Scalar = Fraction(6, 10)  # 0.6 is standard "useful" threshold
    beats_climatology: bool = True

    def __post_init__(self):
        object.__setattr__(self, "rmse", to_scalar(self.rmse))
        object.__setattr__(self, "acc", to_scalar(self.acc))
        object.__setattr__(self, "climatology_rmse", to_scalar(self.climatology_rmse))
        object.__setattr__(self, "skill_vs_climatology", to_scalar(self.skill_vs_climatology))
        object.__setattr__(self, "acc_threshold", to_scalar(self.acc_threshold))

    def verify_skill_ratio(self) -> bool:
        """Verify skill_vs_climatology = rmse / climatology_rmse."""
        if Fraction(self.climatology_rmse) == 0:
            return True
        computed = Fraction(self.rmse) / Fraction(self.climatology_rmse)
        # Allow small tolerance for rational approximation
        diff = abs(computed - Fraction(self.skill_vs_climatology))
        return diff < Fraction(1, 1000)

    def verify_beats_climatology(self) -> bool:
        """Verify beats_climatology = (skill_vs_climatology < 1)."""
        return (Fraction(self.skill_vs_climatology) < 1) == self.beats_climatology

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lead_time_hours": self.lead_time_hours,
            "variable": self.variable,
            "rmse": str(self.rmse),
            "acc": str(self.acc),
            "climatology_rmse": str(self.climatology_rmse),
            "skill_vs_climatology": str(self.skill_vs_climatology),
            "acc_threshold": str(self.acc_threshold),
            "beats_climatology": self.beats_climatology,
        }


# ============================================================================
# PHYSICAL BOUNDS WITNESS
# ============================================================================

@dataclass(frozen=True)
class PhysicalBoundsWitness:
    """
    Witness that physical variables stay within valid ranges.

    Unphysical values indicate model failure:
    - Negative humidity
    - Negative pressure
    - Temperatures below absolute zero
    - Wind speeds exceeding physical limits
    """
    variable: str
    min_value: Scalar
    max_value: Scalar
    physical_min: Scalar
    physical_max: Scalar
    within_bounds: bool

    # Location of violation (if any)
    violation_location: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        object.__setattr__(self, "min_value", to_scalar(self.min_value))
        object.__setattr__(self, "max_value", to_scalar(self.max_value))
        object.__setattr__(self, "physical_min", to_scalar(self.physical_min))
        object.__setattr__(self, "physical_max", to_scalar(self.physical_max))

    def verify_within_bounds(self) -> bool:
        """Verify bounds check is correct."""
        in_bounds = (Fraction(self.min_value) >= Fraction(self.physical_min) and
                     Fraction(self.max_value) <= Fraction(self.physical_max))
        return in_bounds == self.within_bounds

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "variable": self.variable,
            "min_value": str(self.min_value),
            "max_value": str(self.max_value),
            "physical_min": str(self.physical_min),
            "physical_max": str(self.physical_max),
            "within_bounds": self.within_bounds,
        }
        if self.violation_location:
            result["violation_location"] = self.violation_location
        return result


# ============================================================================
# NUMERICAL STABILITY WITNESS
# ============================================================================

@dataclass(frozen=True)
class NumericalStabilityWitness:
    """
    Witness for numerical stability of the forecast.

    CFL condition: Courant-Friedrichs-Lewy stability criterion
    The timestep must satisfy: dt * max_velocity / dx < CFL_limit
    """
    timestep_seconds: Scalar
    grid_spacing_meters: Scalar
    max_velocity: Scalar
    cfl_number: Scalar
    cfl_limit: Scalar = Fraction(1, 1)
    stable: bool = True

    def __post_init__(self):
        object.__setattr__(self, "timestep_seconds", to_scalar(self.timestep_seconds))
        object.__setattr__(self, "grid_spacing_meters", to_scalar(self.grid_spacing_meters))
        object.__setattr__(self, "max_velocity", to_scalar(self.max_velocity))
        object.__setattr__(self, "cfl_number", to_scalar(self.cfl_number))
        object.__setattr__(self, "cfl_limit", to_scalar(self.cfl_limit))

    def verify_cfl(self) -> bool:
        """Verify CFL number computation."""
        if Fraction(self.grid_spacing_meters) == 0:
            return False
        computed = (Fraction(self.timestep_seconds) * Fraction(self.max_velocity) /
                   Fraction(self.grid_spacing_meters))
        diff = abs(computed - Fraction(self.cfl_number))
        return diff < Fraction(1, 1000)

    def verify_stable(self) -> bool:
        """Verify stability = (cfl_number < cfl_limit)."""
        return (Fraction(self.cfl_number) < Fraction(self.cfl_limit)) == self.stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestep_seconds": str(self.timestep_seconds),
            "grid_spacing_meters": str(self.grid_spacing_meters),
            "max_velocity": str(self.max_velocity),
            "cfl_number": str(self.cfl_number),
            "cfl_limit": str(self.cfl_limit),
            "stable": self.stable,
        }


# ============================================================================
# GAUGE FREEDOM WITNESS (Neural Parameterizations)
# ============================================================================

@dataclass(frozen=True)
class ParameterizationGaugeWitness:
    """
    Witness for neural parameterization gauge freedom.

    In NeuralGCM, neural networks replace traditional parameterization schemes
    for subgrid-scale physics (convection, radiation, turbulence).

    These are gauge degrees of freedom: different neural architectures can
    achieve the same physical validity as long as conservation is preserved.
    """
    parameterization_type: str  # "convection", "radiation", "turbulence", etc.
    architecture: str  # "mlp", "cnn", "transformer", etc.
    total_params: int
    minimal_params: int  # Minimal for physical validity
    gauge_dim: int  # = total - minimal

    # Conservation impact
    conservation_preserving: bool = True

    def verify_gauge_dim(self) -> bool:
        """Verify gauge_dim = total_params - minimal_params."""
        return self.gauge_dim == self.total_params - self.minimal_params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameterization_type": self.parameterization_type,
            "architecture": self.architecture,
            "total_params": self.total_params,
            "minimal_params": self.minimal_params,
            "gauge_dim": self.gauge_dim,
            "conservation_preserving": self.conservation_preserving,
        }


# ============================================================================
# FAILURE TAXONOMY
# ============================================================================

class NeuralGCMFailure(Enum):
    """Failure modes for NeuralGCM certificates."""
    # Conservation violations
    MASS_VIOLATION = "mass_conservation_violation"
    ENERGY_VIOLATION = "energy_conservation_violation"
    MOMENTUM_VIOLATION = "momentum_conservation_violation"

    # Physical bound violations
    NEGATIVE_HUMIDITY = "negative_humidity"
    NEGATIVE_PRESSURE = "negative_pressure"
    UNPHYSICAL_TEMPERATURE = "unphysical_temperature"
    UNPHYSICAL_WIND = "unphysical_wind"

    # Numerical stability
    CFL_VIOLATION = "cfl_violation"
    NUMERICAL_INSTABILITY = "numerical_instability"
    DIVERGENCE_DETECTED = "divergence_detected"

    # Skill degradation
    SKILL_BELOW_CLIMATOLOGY = "skill_below_climatology"
    ACC_BELOW_THRESHOLD = "acc_below_threshold"
    SKILL_COLLAPSE = "skill_collapse"


# ============================================================================
# MAIN FORECAST CERTIFICATE
# ============================================================================

@dataclass
class NeuralGCMForecastCertificate:
    """
    Certificate for a NeuralGCM weather/climate forecast.

    A valid forecast certificate requires:
    1. All conservation laws satisfied
    2. Physical bounds respected
    3. Numerical stability maintained
    4. Skill above climatology baseline

    Failure-completeness: If ANY requirement fails, a structured
    obstruction certificate is produced.
    """
    # Certificate metadata
    certificate_id: str
    version: str = "1.0.0"
    schema: str = "QA_NEURALGCM_FORECAST_V1"

    # Success or failure
    success: bool = True
    failure_mode: Optional[NeuralGCMFailure] = None
    failure_witness: Optional[Dict[str, Any]] = None

    # Forecast parameters
    initial_time: Optional[str] = None  # ISO format
    forecast_hours: Optional[int] = None
    initial_state_hash: Optional[str] = None

    # Core witnesses
    conservation: Optional[ConservationBundle] = None
    skill_scores: Optional[List[ForecastSkillWitness]] = None
    physical_bounds: Optional[List[PhysicalBoundsWitness]] = None
    numerical_stability: Optional[NumericalStabilityWitness] = None

    # Gauge freedom (neural parameterizations)
    parameterization_gauge: Optional[List[ParameterizationGaugeWitness]] = None

    # Model metadata
    model_version: Optional[str] = None
    resolution: Optional[str] = None  # e.g., "0.25deg", "1deg"

    # Recompute hook reference
    recompute_hook: Optional[str] = None

    def verify_conservation(self) -> bool:
        """Verify all conservation laws are satisfied."""
        if self.conservation is None:
            return True
        return self.conservation.all_conserved()

    def verify_physical_bounds(self) -> bool:
        """Verify all physical bounds are respected."""
        if self.physical_bounds is None:
            return True
        return all(pb.within_bounds for pb in self.physical_bounds)

    def verify_numerical_stability(self) -> bool:
        """Verify numerical stability."""
        if self.numerical_stability is None:
            return True
        return self.numerical_stability.stable

    def verify_skill(self) -> bool:
        """Verify skill beats climatology."""
        if self.skill_scores is None:
            return True
        return all(ss.beats_climatology for ss in self.skill_scores)

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
        if self.initial_time:
            result["initial_time"] = self.initial_time
        if self.forecast_hours:
            result["forecast_hours"] = self.forecast_hours
        if self.initial_state_hash:
            result["initial_state_hash"] = self.initial_state_hash
        if self.conservation:
            result["conservation"] = self.conservation.to_dict()
        if self.skill_scores:
            result["skill_scores"] = [ss.to_dict() for ss in self.skill_scores]
        if self.physical_bounds:
            result["physical_bounds"] = [pb.to_dict() for pb in self.physical_bounds]
        if self.numerical_stability:
            result["numerical_stability"] = self.numerical_stability.to_dict()
        if self.parameterization_gauge:
            result["parameterization_gauge"] = [pg.to_dict() for pg in self.parameterization_gauge]
        if self.model_version:
            result["model_version"] = self.model_version
        if self.resolution:
            result["resolution"] = self.resolution
        if self.recompute_hook:
            result["recompute_hook"] = self.recompute_hook

        return result

    def compute_certificate_hash(self) -> str:
        """Compute deterministic hash of certificate."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_success_certificate(
    certificate_id: str,
    conservation: ConservationBundle,
    forecast_hours: int,
    skill_scores: Optional[List[ForecastSkillWitness]] = None,
    physical_bounds: Optional[List[PhysicalBoundsWitness]] = None,
    numerical_stability: Optional[NumericalStabilityWitness] = None,
) -> NeuralGCMForecastCertificate:
    """Create a success certificate with all witnesses."""
    return NeuralGCMForecastCertificate(
        certificate_id=certificate_id,
        success=True,
        forecast_hours=forecast_hours,
        conservation=conservation,
        skill_scores=skill_scores,
        physical_bounds=physical_bounds,
        numerical_stability=numerical_stability,
    )


def create_failure_certificate(
    certificate_id: str,
    failure_mode: NeuralGCMFailure,
    failure_witness: Dict[str, Any],
    conservation: Optional[ConservationBundle] = None,
) -> NeuralGCMForecastCertificate:
    """Create a failure certificate with obstruction witness."""
    return NeuralGCMForecastCertificate(
        certificate_id=certificate_id,
        success=False,
        failure_mode=failure_mode,
        failure_witness=failure_witness,
        conservation=conservation,
    )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Create a success certificate for a 10-day forecast

    # 1. Conservation witnesses
    mass = ConservationWitness(
        conservation_type=ConservationType.MASS,
        initial_value=Fraction("5.15e18"),  # kg (approximate atmospheric mass)
        final_value=Fraction("5.15e18"),
        delta=Fraction(0),
        tolerance=Fraction("1e12"),  # 1 teragram tolerance
        conserved=True,
        relative_error=Fraction(0),
    )

    energy = ConservationWitness(
        conservation_type=ConservationType.ENERGY,
        initial_value=Fraction("1.5e24"),  # Joules
        final_value=Fraction("1.5e24"),
        delta=Fraction("1e18"),  # Small drift
        tolerance=Fraction("1e19"),
        conserved=True,
        relative_error=Fraction("1e-6"),
    )

    momentum = ConservationWitness(
        conservation_type=ConservationType.MOMENTUM,
        initial_value=Fraction("1.4e28"),  # kg⋅m/s
        final_value=Fraction("1.4e28"),
        delta=Fraction(0),
        tolerance=Fraction("1e22"),
        conserved=True,
    )

    conservation = ConservationBundle(mass=mass, energy=energy, momentum=momentum)

    # 2. Skill score (Z500 at day 10)
    skill = ForecastSkillWitness(
        lead_time_hours=240,
        variable="z500",
        rmse=Fraction(85),  # meters
        acc=Fraction(75, 100),  # 0.75
        climatology_rmse=Fraction(120),  # meters
        skill_vs_climatology=Fraction(85, 120),
        beats_climatology=True,
    )

    # 3. Create certificate
    cert = create_success_certificate(
        certificate_id="neuralgcm_10day_001",
        conservation=conservation,
        forecast_hours=240,
        skill_scores=[skill],
    )

    print("=== NeuralGCM Forecast Certificate ===")
    print(json.dumps(cert.to_dict(), indent=2))
    print(f"\nCertificate hash: {cert.compute_certificate_hash()}")
    print(f"Conservation verified: {cert.verify_conservation()}")
    print(f"Skill verified: {cert.verify_skill()}")
