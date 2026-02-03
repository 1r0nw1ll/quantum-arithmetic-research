"""
qa_field_computation_certificate.py

QA Certificate Schema for Field Computation (In-Physics Computing)

Based on:
- WISE: Disaggregated ML via In-Physics Computing at RF
  (doi:10.1126/sciadv.adz0817, github:functions-lab/WISE)

Core QA Principle:
    Computation can be realized by PHYSICAL FIELD dynamics.
    Arithmetic operations = wave transformations.
    The medium IS the computer.

    Generator Injection showed: adding tools expands reach.
    Diversity Collapse showed: losing invariants contracts reach.
    Field Computation shows: reach can be realized by PHYSICS,
    not just symbol manipulation.

This certificate proves:
    A target operator L can be realized within error epsilon
    by a sequence of field generators (propagation, mixing,
    phase shift, measurement) while preserving field invariants
    (power budget, bandwidth, synchronization, linearity).

    When a field invariant (e.g., sync_lock) is violated,
    a DESYNC/NOISE/NONLINEARITY barrier is erected --
    the same obstruction algebra as the other certificates.

Imports qa_cert_core for shared plumbing.
"""

from __future__ import annotations

import sys
import os

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, FrozenSet
from enum import Enum
from fractions import Fraction

try:
    # When run as module: python -m qa_alphageometry_ptolemy.qa_field_computation_certificate
    from .qa_cert_core import (
        Scalar, to_scalar, scalar_to_str,
        canonical_json, certificate_hash, state_hash,
        cert_id, utc_now_iso,
        ValidationResult,
    )
except ImportError:
    # When run directly: python qa_field_computation_certificate.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qa_cert_core import (
        Scalar, to_scalar, scalar_to_str,
        canonical_json, certificate_hash, state_hash,
        cert_id, utc_now_iso,
        ValidationResult,
    )


# ============================================================================
# FIELD DOMAIN
# ============================================================================

class FieldDomain(Enum):
    """Physical domain in which computation occurs."""
    RF_INPHYSICS = "rf_inphysics"       # WISE: RF electromagnetic fields
    PHOTONIC = "photonic"                # Optical neural networks
    ANALOG_CIM = "analog_cim"           # Coherent Ising machines
    NEUROMORPHIC = "neuromorphic"        # Spiking dynamics
    ACOUSTIC = "acoustic"               # Acoustic wave computing
    SIMULATED = "simulated"             # Software simulation of field


class GeneratorCategory(Enum):
    """Categories of field generators."""
    FIELD_DYNAMICS = "field_dynamics"       # Propagation, superposition
    CONTROL = "control"                     # Phase shift, gain, attenuation
    MEASUREMENT = "measurement"             # Sample, quantize, decode
    SYNCHRONIZATION = "synchronization"     # Pilot alignment, clock recovery
    CALIBRATION = "calibration"             # Channel estimation, correction
    ENCODING = "encoding"                   # Map data to field representation


class BarrierType(Enum):
    """Barriers specific to field computation."""
    DESYNC = "desync"                       # Synchronization lost
    NOISE_FLOOR = "noise_floor"             # SNR below threshold
    NONLINEARITY = "nonlinearity"           # Left linear operating region
    CHANNEL_MISMATCH = "channel_mismatch"   # Estimated != actual channel
    BANDWIDTH_EXCEEDED = "bandwidth_exceeded"
    POWER_EXCEEDED = "power_exceeded"
    DECODE_FAILURE = "decode_failure"        # Cannot recover data
    NO_CALIBRATION = "no_calibration"       # Channel unknown
    NO_MEASUREMENT = "no_measurement"       # Cannot observe output
    BUDGET_EXHAUSTED = "budget_exhausted"


# ============================================================================
# FIELD GENERATOR
# ============================================================================

@dataclass(frozen=True)
class FieldGenerator:
    """
    A generator that operates on the physical field.

    In WISE: these are operations the EM field performs naturally
    (superposition, multipath mixing) or that hardware controls
    (phase shift, gain, sampling).
    """
    name: str
    category: GeneratorCategory

    # What this generator does physically
    physical_action: str  # e.g., "RF superposition of broadcast signals"

    # Formal signature
    input_signature: Tuple[str, ...]
    output_signature: Tuple[str, ...]

    # Invariants this generator preserves
    preserves_invariants: FrozenSet[str] = frozenset()

    # Invariants this generator may violate
    may_violate: FrozenSet[str] = frozenset()

    # Cost (energy, time, etc.)
    cost: Optional[Scalar] = None

    def __post_init__(self):
        if self.cost is not None:
            object.__setattr__(self, "cost", to_scalar(self.cost))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "physical_action": self.physical_action,
            "input_signature": list(self.input_signature),
            "output_signature": list(self.output_signature),
            "preserves_invariants": sorted(self.preserves_invariants),
            "may_violate": sorted(self.may_violate),
            "cost": scalar_to_str(self.cost) if self.cost is not None else None,
        }


@dataclass(frozen=True)
class FieldGeneratorSet:
    """Set of field generators defining available physical operations."""
    name: str
    generators: FrozenSet[FieldGenerator]
    domain: FieldDomain
    description: Optional[str] = None

    def generator_names(self) -> FrozenSet[str]:
        return frozenset(g.name for g in self.generators)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "generators": [g.to_dict() for g in sorted(self.generators, key=lambda x: x.name)],
            "description": self.description,
        }


# ============================================================================
# TARGET OPERATOR
# ============================================================================

@dataclass(frozen=True)
class OperatorSpec:
    """
    Specification of the computation to be realized by the field.

    In WISE: this is a matrix-vector multiply, convolution, or
    classifier inference step that the EM field performs via
    superposition and mixing.
    """
    operator_kind: str  # "linear_map", "conv1d", "classifier", "beamform"
    shape_in: int
    shape_out: int
    representation: str  # "matrix", "impulse_response", "fourier_mask"
    operator_hash: str   # SHA-256 of canonical representation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator_kind": self.operator_kind,
            "shape_in": self.shape_in,
            "shape_out": self.shape_out,
            "representation": self.representation,
            "operator_hash": self.operator_hash,
        }


# ============================================================================
# FIELD INVARIANTS
# ============================================================================

@dataclass(frozen=True)
class FieldInvariant:
    """
    An invariant that must hold during field computation.

    Hard invariants: violation = barrier (computation fails).
    Soft invariants: violation = degradation (quality drops).
    """
    name: str
    predicate: str       # Human-readable contract
    threshold: Optional[Scalar] = None
    units: Optional[str] = None
    hard: bool = True    # Hard = barrier on violation; Soft = degradation

    def __post_init__(self):
        if self.threshold is not None:
            object.__setattr__(self, "threshold", to_scalar(self.threshold))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "predicate": self.predicate,
            "threshold": scalar_to_str(self.threshold) if self.threshold is not None else None,
            "units": self.units,
            "hard": self.hard,
        }


@dataclass(frozen=True)
class FieldInvariantSet:
    """Collection of field invariants with an oracle specification."""
    invariants: Tuple[FieldInvariant, ...]
    oracle_type: str  # "simulator_check", "hardware_measurement", "replay_trace"

    def hard_invariants(self) -> Tuple[FieldInvariant, ...]:
        return tuple(i for i in self.invariants if i.hard)

    def soft_invariants(self) -> Tuple[FieldInvariant, ...]:
        return tuple(i for i in self.invariants if not i.hard)

    def names(self) -> FrozenSet[str]:
        return frozenset(i.name for i in self.invariants)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard": [i.to_dict() for i in self.hard_invariants()],
            "soft": [i.to_dict() for i in self.soft_invariants()],
            "oracle_type": self.oracle_type,
        }


# ============================================================================
# FIELD TRACE (trajectory through field state space)
# ============================================================================

@dataclass(frozen=True)
class TraceStep:
    """One step in a field computation trajectory."""
    t: int                    # Time step / symbol interval
    generator: str            # Which generator fired
    state_hash: str           # Hash of field state after this step
    observations: Optional[Dict[str, str]] = None  # Optional measurements

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "t": self.t,
            "generator": self.generator,
            "state_hash": self.state_hash,
        }
        if self.observations:
            d["observations"] = self.observations
        return d


@dataclass(frozen=True)
class FieldReachabilityWitness:
    """
    Constructive proof that a target operator is realized by field dynamics.

    The trajectory shows: start state → sequence of field generators → end state
    where end state realizes the target operator within epsilon.
    """
    start_state_hash: str
    end_state_hash: str
    trajectory: Tuple[TraceStep, ...]
    path_length: int

    # Metrics at completion
    operator_error: Scalar     # Distance from target operator
    epsilon: Scalar            # Tolerance
    snr: Optional[Scalar] = None
    energy: Optional[Scalar] = None
    latency: Optional[Scalar] = None

    # Invariants verified along trajectory
    invariants_checked: FrozenSet[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "operator_error", to_scalar(self.operator_error))
        object.__setattr__(self, "epsilon", to_scalar(self.epsilon))
        if self.snr is not None:
            object.__setattr__(self, "snr", to_scalar(self.snr))
        if self.energy is not None:
            object.__setattr__(self, "energy", to_scalar(self.energy))
        if self.latency is not None:
            object.__setattr__(self, "latency", to_scalar(self.latency))
        if len(self.trajectory) != self.path_length:
            raise ValueError(f"Path length mismatch: {len(self.trajectory)} != {self.path_length}")

    @property
    def operator_realized(self) -> bool:
        """Did the field realize the operator within tolerance?"""
        return self.operator_error <= self.epsilon

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "start_state_hash": self.start_state_hash,
            "end_state_hash": self.end_state_hash,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "path_length": self.path_length,
            "operator_error": scalar_to_str(self.operator_error),
            "epsilon": scalar_to_str(self.epsilon),
            "operator_realized": self.operator_realized,
            "invariants_checked": sorted(self.invariants_checked),
        }
        if self.snr is not None:
            d["snr"] = scalar_to_str(self.snr)
        if self.energy is not None:
            d["energy"] = scalar_to_str(self.energy)
        if self.latency is not None:
            d["latency"] = scalar_to_str(self.latency)
        return d


@dataclass(frozen=True)
class FieldBarrier:
    """A barrier preventing field computation from reaching its target."""
    barrier_type: BarrierType
    source_state_class: str
    target_state_class: str
    required_generator: Optional[str] = None
    required_invariant: Optional[str] = None
    obstruction_proof: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "barrier_type": self.barrier_type.value,
            "source_state_class": self.source_state_class,
            "target_state_class": self.target_state_class,
            "required_generator": self.required_generator,
            "required_invariant": self.required_invariant,
            "obstruction_proof": self.obstruction_proof,
        }


@dataclass(frozen=True)
class FieldUnreachabilityWitness:
    """Proof that a target is unreachable under baseline field generators."""
    target_state_class: str
    generator_set_name: str
    barrier: FieldBarrier
    obstruction_argument: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_state_class": self.target_state_class,
            "generator_set_name": self.generator_set_name,
            "barrier": self.barrier.to_dict(),
            "obstruction_argument": self.obstruction_argument,
        }


# ============================================================================
# FIELD COMPUTATION CERTIFICATE
# ============================================================================

class FieldResult(Enum):
    """Result of field computation analysis."""
    BARRIER_CROSSED = "barrier_crossed"
    NO_NEW_REACH = "no_new_reach"
    INVARIANT_BROKEN = "invariant_broken"
    PENDING = "pending"


@dataclass(frozen=True)
class FieldComputationCertificate:
    """
    Certificate proving that physical field dynamics realize a target operator.

    This is the third leg of the QA certificate triad:

        1. Generator Injection:  Reach expands (add tools)
        2. Diversity Collapse:   Reach contracts (lose invariant)
        3. Field Computation:    Reach realized by PHYSICS

    The same obstruction algebra applies:
    - DESYNC barrier = sync generator missing
    - NOISE_FLOOR barrier = calibration generator missing
    - NONLINEARITY barrier = power invariant violated

    All are instances of: barrier crossed (or erected) by
    the generator-invariant pair (G, I).
    """
    certificate_id: str
    timestamp: str

    # What we're computing
    claim: str
    task_class: str  # "matvec", "conv1d", "classifier_inference", "beamforming"
    target_operator: OperatorSpec

    # Physical domain
    domain: FieldDomain

    # Generator sets (same injection pattern as ROI #1)
    baseline_generators: FieldGeneratorSet
    extended_generators: FieldGeneratorSet
    injected_generators: FrozenSet[FieldGenerator]

    # Invariants
    invariants: FieldInvariantSet

    # Evidence
    barrier: FieldBarrier
    before_witness: FieldUnreachabilityWitness
    after_witness: FieldReachabilityWitness

    # Result
    result: FieldResult

    # Source reference
    source_refs: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        # Verify injection = extended - baseline by name
        baseline_names = self.baseline_generators.generator_names()
        extended_names = self.extended_generators.generator_names()
        injected_names = frozenset(g.name for g in self.injected_generators)
        expected = extended_names - baseline_names

        if injected_names != expected:
            raise ValueError(
                f"Injected generators mismatch: got {injected_names}, expected {expected}"
            )
        if not baseline_names.issubset(extended_names):
            raise ValueError("Baseline generators must be subset of extended generators")

    def compute_hash(self) -> str:
        return certificate_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "certificate_id": self.certificate_id,
            "certificate_type": "FIELD_COMPUTATION_CERT",
            "timestamp": self.timestamp,
            "claim": self.claim,
            "task_class": self.task_class,
            "target_operator": self.target_operator.to_dict(),
            "domain": self.domain.value,
            "baseline_generators": self.baseline_generators.to_dict(),
            "extended_generators": self.extended_generators.to_dict(),
            "injected_generators": [
                g.to_dict() for g in sorted(self.injected_generators, key=lambda x: x.name)
            ],
            "invariants": self.invariants.to_dict(),
            "barrier": self.barrier.to_dict(),
            "before_witness": self.before_witness.to_dict(),
            "after_witness": self.after_witness.to_dict(),
            "result": self.result.value,
        }
        if self.source_refs:
            d["source_refs"] = self.source_refs
        return d

    def to_json(self, indent: int = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_certificate(cert: FieldComputationCertificate) -> ValidationResult:
    """
    Validate a field computation certificate.

    Uses qa_cert_core.ValidationResult for failure-complete output.
    """
    v = ValidationResult()

    # Subset check
    baseline_names = cert.baseline_generators.generator_names()
    extended_names = cert.extended_generators.generator_names()
    v.check(
        baseline_names.issubset(extended_names),
        "Baseline generators not subset of extended generators"
    )

    # Injection check
    injected_names = frozenset(g.name for g in cert.injected_generators)
    expected = extended_names - baseline_names
    v.check(
        injected_names == expected,
        f"Injected mismatch: got {injected_names}, expected {expected}"
    )

    # Trajectory uses only extended generators
    for step in cert.after_witness.trajectory:
        v.check(
            step.generator in extended_names,
            f"Trajectory step uses unknown generator: {step.generator}"
        )

    # Operator realization check
    if cert.result == FieldResult.BARRIER_CROSSED:
        v.check(
            cert.after_witness.operator_realized,
            "BARRIER_CROSSED but operator_error > epsilon"
        )
        v.check(
            cert.after_witness.path_length > 0,
            "BARRIER_CROSSED but empty trajectory"
        )

    # Barrier consistency
    v.check(
        cert.barrier.barrier_type in BarrierType,
        f"Unknown barrier type: {cert.barrier.barrier_type}"
    )

    # Domain consistency
    v.check(
        cert.baseline_generators.domain == cert.domain,
        "Baseline generator domain doesn't match certificate domain"
    )
    v.check(
        cert.extended_generators.domain == cert.domain,
        "Extended generator domain doesn't match certificate domain"
    )

    return v


# ============================================================================
# FACTORY: WISE DESYNC OBSTRUCTION EXAMPLE
# ============================================================================

def create_wise_desync_certificate() -> FieldComputationCertificate:
    """
    Create the 'killer example': DESYNC_OBSTRUCTION_WITNESS.

    Scenario:
        A WISE RF network attempts to realize a 4x4 matrix-vector multiply
        using electromagnetic field superposition. Without synchronization
        (pilot alignment), receivers cannot decode the result -- DESYNC barrier.
        Adding the sync_align generator crosses the barrier.

    This demonstrates:
        - Barrier erected: no sync → desync obstruction
        - Generator injected: sync_align
        - Barrier crossed: operator realized within epsilon
        - Invariants preserved: power_budget, bandwidth, linearity
    """
    from fractions import Fraction as F

    # ---- Target operator: 4x4 linear map via RF superposition ----
    target_op = OperatorSpec(
        operator_kind="linear_map",
        shape_in=4,
        shape_out=4,
        representation="matrix",
        operator_hash=state_hash("wise_4x4_matvec_example"),
    )

    # ---- Baseline generators: RF dynamics without sync ----
    propagate = FieldGenerator(
        name="propagate",
        category=GeneratorCategory.FIELD_DYNAMICS,
        physical_action="EM wave propagation through free space (multipath channel)",
        input_signature=("tx_signal",),
        output_signature=("rx_signal_raw",),
        preserves_invariants=frozenset(["power_budget", "bandwidth_limit"]),
        may_violate=frozenset(["sync_lock"]),
    )
    mix = FieldGenerator(
        name="mix",
        category=GeneratorCategory.FIELD_DYNAMICS,
        physical_action="Superposition of multiple broadcast signals at receiver",
        input_signature=("rx_signal_raw",),
        output_signature=("rx_mixed",),
        preserves_invariants=frozenset(["linearity", "power_budget"]),
    )
    phase_shift = FieldGenerator(
        name="phase_shift",
        category=GeneratorCategory.CONTROL,
        physical_action="Controlled phase rotation at transmitter antenna",
        input_signature=("tx_signal",),
        output_signature=("tx_signal_shifted",),
        preserves_invariants=frozenset(["power_budget", "bandwidth_limit", "linearity"]),
    )
    gain = FieldGenerator(
        name="gain",
        category=GeneratorCategory.CONTROL,
        physical_action="Amplification/attenuation at transmitter",
        input_signature=("tx_signal",),
        output_signature=("tx_signal_scaled",),
        preserves_invariants=frozenset(["bandwidth_limit", "linearity"]),
        may_violate=frozenset(["power_budget"]),
    )
    measure = FieldGenerator(
        name="measure",
        category=GeneratorCategory.MEASUREMENT,
        physical_action="ADC sampling at receiver, quantize RF to digital",
        input_signature=("rx_mixed",),
        output_signature=("rx_digital",),
        preserves_invariants=frozenset(["power_budget"]),
    )
    encode = FieldGenerator(
        name="encode",
        category=GeneratorCategory.ENCODING,
        physical_action="Map data vector to RF symbol sequence",
        input_signature=("data_vector",),
        output_signature=("tx_signal",),
        preserves_invariants=frozenset(["bandwidth_limit"]),
    )

    baseline_gens = frozenset([propagate, mix, phase_shift, gain, measure, encode])

    # ---- Injected generator: synchronization ----
    sync_align = FieldGenerator(
        name="sync_align",
        category=GeneratorCategory.SYNCHRONIZATION,
        physical_action="Pilot-based timing and phase alignment at receiver",
        input_signature=("rx_signal_raw", "pilot_sequence"),
        output_signature=("rx_signal_aligned",),
        preserves_invariants=frozenset(["power_budget", "bandwidth_limit", "sync_lock"]),
        cost=F(1, 10),  # 10% overhead for pilot symbols
    )

    extended_gens = baseline_gens | {sync_align}
    injected = frozenset([sync_align])

    baseline_set = FieldGeneratorSet(
        name="wise_rf_nosync",
        generators=baseline_gens,
        domain=FieldDomain.RF_INPHYSICS,
        description="RF field computation without synchronization",
    )
    extended_set = FieldGeneratorSet(
        name="wise_rf_synced",
        generators=extended_gens,
        domain=FieldDomain.RF_INPHYSICS,
        description="RF field computation with pilot-based synchronization",
    )

    # ---- Field invariants ----
    invariants = FieldInvariantSet(
        invariants=(
            FieldInvariant(
                name="power_budget",
                predicate="Total transmit power <= P_max",
                threshold=F(1, 1),  # Normalized to 1
                units="watts_normalized",
                hard=True,
            ),
            FieldInvariant(
                name="bandwidth_limit",
                predicate="Signal bandwidth <= B_max",
                threshold=F(20, 1),  # 20 MHz
                units="MHz",
                hard=True,
            ),
            FieldInvariant(
                name="sync_lock",
                predicate="Receiver timing offset < T_sym/4",
                threshold=F(1, 4),  # Quarter symbol
                units="symbol_fraction",
                hard=True,
            ),
            FieldInvariant(
                name="linearity",
                predicate="Operating within linear region of amplifier",
                hard=True,
            ),
            FieldInvariant(
                name="decode_fidelity",
                predicate="BER < target threshold",
                threshold=F(1, 1000),  # 0.1%
                units="bit_error_rate",
                hard=False,
            ),
        ),
        oracle_type="simulator_check",
    )

    # ---- Barrier: DESYNC ----
    barrier = FieldBarrier(
        barrier_type=BarrierType.DESYNC,
        source_state_class="encoded_rf_broadcast",
        target_state_class="operator_realization_within_epsilon",
        required_generator="sync_align",
        required_invariant="sync_lock",
        obstruction_proof=(
            "Without pilot-based synchronization, receiver timing offset "
            "exceeds T_sym/4. Mixed signal phases are incoherent. "
            "Superposition does not realize target matrix -- output is noise."
        ),
    )

    # ---- Before witness: unreachable without sync ----
    before_witness = FieldUnreachabilityWitness(
        target_state_class="operator_realization_within_epsilon",
        generator_set_name="wise_rf_nosync",
        barrier=barrier,
        obstruction_argument=(
            "Propagation introduces unknown phase offsets per path. "
            "Without sync_align, receiver cannot coherently combine "
            "multipath components. Measured operator error > 0.9 "
            "(effectively random). sync_lock invariant violated."
        ),
    )

    # ---- After witness: reachable with sync ----
    trajectory = (
        TraceStep(t=0, generator="encode",
                  state_hash=state_hash("data_encoded_to_rf"),
                  observations={"action": "4D vector -> 4 RF symbols"}),
        TraceStep(t=1, generator="phase_shift",
                  state_hash=state_hash("phases_set_for_matvec"),
                  observations={"action": "Set antenna phases for row 0-3 of target matrix"}),
        TraceStep(t=2, generator="gain",
                  state_hash=state_hash("amplitudes_set"),
                  observations={"action": "Set antenna gains for matrix coefficients"}),
        TraceStep(t=3, generator="propagate",
                  state_hash=state_hash("rf_in_flight"),
                  observations={"action": "EM waves propagate, multipath superposition occurs"}),
        TraceStep(t=4, generator="mix",
                  state_hash=state_hash("rf_superposed_at_rx"),
                  observations={"action": "Signals from all TX antennas combine at each RX antenna"}),
        TraceStep(t=5, generator="sync_align",
                  state_hash=state_hash("rf_synchronized"),
                  observations={"action": "Pilot-based alignment restores phase coherence"}),
        TraceStep(t=6, generator="measure",
                  state_hash=state_hash("digital_output"),
                  observations={"action": "ADC samples aligned signal, recovers y = Lx + noise"}),
    )

    after_witness = FieldReachabilityWitness(
        start_state_hash=state_hash("initial_data_vector"),
        end_state_hash=state_hash("digital_output"),
        trajectory=trajectory,
        path_length=7,
        operator_error=F(3, 100),   # 3% error
        epsilon=F(5, 100),           # 5% tolerance
        snr=F(20, 1),               # 20 dB
        energy=F(1, 10),            # 0.1 joules
        latency=F(7, 1),            # 7 symbol intervals
        invariants_checked=frozenset([
            "power_budget", "bandwidth_limit", "sync_lock", "linearity"
        ]),
    )

    # ---- Assemble certificate ----
    return FieldComputationCertificate(
        certificate_id=cert_id("FIELD-WISE-DESYNC"),
        timestamp=utc_now_iso(),
        claim=(
            "4x4 matrix-vector multiply realized by RF field superposition "
            "within 5% operator error, after injecting sync_align generator "
            "to cross DESYNC barrier."
        ),
        task_class="matvec",
        target_operator=target_op,
        domain=FieldDomain.RF_INPHYSICS,
        baseline_generators=baseline_set,
        extended_generators=extended_set,
        injected_generators=injected,
        invariants=invariants,
        barrier=barrier,
        before_witness=before_witness,
        after_witness=after_witness,
        result=FieldResult.BARRIER_CROSSED,
        source_refs=[
            {"name": "WISE", "ref": "doi:10.1126/sciadv.adz0817"},
            {"name": "WISE code", "ref": "github:functions-lab/WISE"},
        ],
    )


# ============================================================================
# TRIAD SUMMARY
# ============================================================================

def explain_triad() -> str:
    """The QA certificate triad: injection, collapse, field computation."""
    return """
THE QA CERTIFICATE TRIAD

All three certificates prove the same theorem from different angles:
    Reachability is controlled by the pair (G, I).

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. GENERATOR INJECTION              Reach EXPANDS                  │
│     G₁ ⊂ G₂, I preserved            Add tools → cross barriers     │
│     Domain: agents, proofs            LLM-in-Sandbox, Axiom         │
│                                                                     │
│  2. DIVERSITY COLLAPSE               Reach CONTRACTS                │
│     G fixed, I_div violated           Lose diversity → erect barrier │
│     Domain: search, RL                Execution-Grounded Research    │
│                                                                     │
│  3. FIELD COMPUTATION                Reach REALIZED BY PHYSICS      │
│     G = physical operators            Air is the computer            │
│     I = power, sync, linearity        WISE RF computing             │
│     Barrier: DESYNC, NOISE            Same obstruction algebra       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Shared structure:                                                   │
│  - Generator sets (before/after)                                    │
│  - Invariants (preserved/violated)                                  │
│  - Barriers (obstruction objects, not bugs)                         │
│  - Witnesses (constructive reachability / unreachability proofs)    │
│  - Deterministic hashing + validation (qa_cert_core)                │
│                                                                     │
│  General theorem:                                                    │
│  Capability = Reachability(S, G, I)                                 │
│  This holds whether G operates on bits, proofs, populations,        │
│  or electromagnetic fields.                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    # Create the WISE desync certificate
    cert = create_wise_desync_certificate()

    # Validate
    result = validate_certificate(cert)
    print(f"WISE Desync Certificate Valid: {result.is_valid}")
    if not result.is_valid:
        for issue in result.issues:
            print(f"  Issue: {issue}")
    print()

    # Key metrics
    w = cert.after_witness
    print("=== FIELD COMPUTATION RESULT ===")
    print(f"Task:              {cert.task_class} ({cert.target_operator.shape_in}x{cert.target_operator.shape_out})")
    print(f"Domain:            {cert.domain.value}")
    print(f"Barrier crossed:   {cert.barrier.barrier_type.value}")
    print(f"Injected:          {[g.name for g in cert.injected_generators]}")
    print(f"Operator error:    {w.operator_error} (tolerance: {w.epsilon})")
    print(f"Operator realized: {w.operator_realized}")
    print(f"SNR:               {w.snr} dB")
    print(f"Energy:            {w.energy} J")
    print(f"Path length:       {w.path_length} steps")
    print(f"Invariants held:   {sorted(w.invariants_checked)}")
    print(f"Certificate hash:  {cert.compute_hash()}")
    print()

    # Print triad
    print(explain_triad())

    # Print full JSON
    print("=== FULL CERTIFICATE ===")
    print(cert.to_json())
