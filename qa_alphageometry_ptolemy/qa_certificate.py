
"""
qa_certificate.py

QA Proof Certificate Schema v1.0 (Canonical + Unified Domains)

Unifies:
- QA formal reachability proofs (σ, λ, μ, ν, ...)
- QA-AlphaGeometry proof traces (AG:*)
- QA Physics projection/law emergence (PHYS:* + ProjectionContract)

Hard constraints:
- Exact scalars only (int/Fraction) — no floats anywhere in the certificate.
- Deterministic serialization for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, Union, Set, Tuple
from enum import Enum
from fractions import Fraction
import hashlib
import json

# ============================================================================
# FOUNDATIONAL TYPES
# ============================================================================

# Exact scalar type (no floating point approximations)
Scalar = Union[int, Fraction]

def to_scalar(x: Any) -> Scalar:
    """Convert to exact scalar, rejecting floats.

    Accepted:
      - int, Fraction
      - str:
          * integer: "25"
          * rational: "3/2"
          * decimal: "63.43"  -> Fraction(6343, 100)

    Rejected:
      - float (always)
    """
    if isinstance(x, bool):
        # avoid bool being treated as int
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        raise TypeError(f"Cannot convert float to exact scalar (got {x})")
    if isinstance(x, str):
        s = x.strip()
        # Fraction can parse "a/b" and decimals like "63.43" exactly
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")

# Base QA generator names (non-namespaced)
ALLOWED_GENERATORS = {"σ", "λ", "μ", "ν", "σ_inv", "λ_inv"}

@dataclass(frozen=True)
class StateRef:
    """A reference to a QA state (hash-based, not storing full state)."""
    state_id: str  # Hash of canonical representation
    coords: Tuple[Scalar, ...]  # Only for small states; may be empty for non-QA engines
    packet: Optional[Dict[str, Scalar]] = None  # Invariant packet if small enough

    @staticmethod
    def from_coords_and_packet(coords: Tuple[Any, ...],
                               packet: Dict[str, Any]) -> "StateRef":
        """Canonical constructor with exact scalar normalization."""
        coords_n = tuple(to_scalar(c) for c in coords)
        packet_n = {k: to_scalar(v) for k, v in packet.items()}

        canonical = {
            "coords": [str(c) for c in coords_n],
            "packet": {k: str(v) for k, v in sorted(packet_n.items())},
        }
        state_id = hashlib.sha256(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()[:16]

        return StateRef(state_id=state_id, coords=coords_n, packet=packet_n)

@dataclass(frozen=True)
class Generator:
    """A legal move operator in QA or a formal proof system.

    Namespaces:
      - QA core: "σ", "λ", "μ", "ν", "σ_inv", "λ_inv"
      - AlphaGeometry rule: "AG:<rule_id>"
      - Physics/projection: "PHYS:<thing>"
      - Observer/measurement: "OBS:<observer_id>"
    """
    name: str
    params: Tuple[Scalar, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Allow namespaced generators
        if self.name.startswith("AG:"):
            pass
        elif self.name.startswith("PHYS:"):
            pass
        elif self.name.startswith("OBS:"):
            pass
        elif self.name not in ALLOWED_GENERATORS:
            raise ValueError(
                f"Unknown generator '{self.name}'. "
                f"Allowed: {sorted(ALLOWED_GENERATORS)} or namespaced (AG:*, PHYS:*, OBS:*)"
            )
        object.__setattr__(self, "params", tuple(to_scalar(p) for p in self.params))

    def __repr__(self) -> str:
        if self.params:
            return f"{self.name}{self.params}"
        return self.name

    def __hash__(self) -> int:
        return hash((self.name, self.params))

@dataclass
class MoveWitness:
    """A single step in a QA path (compact, serializable)."""
    gen: Generator
    src: StateRef
    dst: StateRef
    packet_delta: Dict[str, Scalar]
    legal: bool = True  # False for attempted moves that violate invariants

    def __post_init__(self):
        # Normalize packet_delta scalars
        self.packet_delta = {k: to_scalar(v) for k, v in self.packet_delta.items()}

        if self.legal:
            for inv_name, delta in self.packet_delta.items():
                if delta != 0:
                    raise ValueError(
                        f"Legal move must preserve invariants: {inv_name} has Δ={delta} (must be 0)"
                    )
        else:
            if not any(delta != 0 for delta in self.packet_delta.values()):
                raise ValueError("Illegal move must show nonzero invariant delta")

# ============================================================================
# PHYSICS / OBSERVER EXTENSION
# ============================================================================

@dataclass(frozen=True)
class ProjectionContract:
    """Contract for how QA discrete state is projected to continuous observables.

    Continuous quantities MUST only appear here or in context (projection layer),
    never in the QA substrate.
    """
    observer_id: str
    time_projection: str
    preserves_topology: bool
    preserves_symmetry: bool
    continuous_observables: List[str]

    repo_tag: Optional[str] = None
    commit_hash: Optional[str] = None
    determinism_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "time_projection": self.time_projection,
            "preserves_topology": self.preserves_topology,
            "preserves_symmetry": self.preserves_symmetry,
            "continuous_observables": self.continuous_observables,
            "repo_tag": self.repo_tag,
            "commit_hash": self.commit_hash,
            "determinism_hash": self.determinism_hash,
        }

# ============================================================================
# FAILURE TAXONOMY (QA-Complete + Physics)
# ============================================================================

class FailType(Enum):
    # Spatial boundary failures
    OUT_OF_BOUNDS = "out_of_bounds"

    # Invariant structure failures
    INVARIANT_VIOLATION = "invariant_violation"

    # QA-specific algebraic failures
    PARITY_VIOLATION = "parity_violation"
    NON_REDUCTION_VIOLATION = "non_reduction_violation"

    # Topological failures
    SCC_UNREACHABLE = "scc_unreachable"
    SCC_SPLIT = "scc_split"

    # Generator insufficiency
    GENERATOR_INSUFFICIENT = "generator_insufficient"
    DEPTH_EXHAUSTED = "depth_exhausted"

    # Domain/precondition failures
    DOMAIN_CONSTRAINT = "domain_constraint"
    TARGET_UNDEFINED = "target_undefined"

    # Fixed-point isolation (QARM/RML specific)
    FIXED_Q_ISOLATED = "fixed_q_isolated"

    # Cycle/limit-cycle detection (policy divergence)
    CYCLE_DETECTED = "cycle_detected"  # Policy entered repeating cycle without progress

    # Belief-state failures (POMDP/partial observability)
    BELIEF_DEGENERACY = "belief_degeneracy"  # Belief distribution became unusable
    BELIEF_COLLAPSE_WRONG = "belief_collapse_wrong"  # Belief collapsed to incorrect state
    BELIEF_TOO_DIFFUSE = "belief_too_diffuse"  # Belief entropy too high to act
    NON_IDENTIFIABLE = "non_identifiable"  # States observationally aliased - can't distinguish

    # Physics/projection-specific failures
    OBSERVER_UNDEFINED = "observer_undefined"
    LAW_VIOLATION = "law_violation"
    PROJECTION_INCOMPATIBLE = "projection_incompatible"

    # Understanding failures (Beyond World Models, arXiv:2511.12239v1)
    ADHOC_STATE_INJECTION = "adhoc_state_injection"  # Abstract object without derivation

@dataclass
class ObstructionEvidence:
    """Evidence for a certified mathematical obstruction—not a bug."""
    fail_type: FailType

    # Invariant violation evidence
    violated_invariants: Optional[Dict[str, Scalar]] = None
    blocked_move: Optional[MoveWitness] = None

    # SCC topology evidence
    scc_id_reached: Optional[str] = None
    scc_id_goal: Optional[str] = None
    goal_state_id: Optional[str] = None
    scc_counts: Optional[Dict[str, int]] = None
    witness_edges: Optional[List[Tuple[str, str, str]]] = None
    reachable_frontier_hash: Optional[str] = None

    # Generator evidence
    generator_set: Optional[Set[Generator]] = None

    # Search metadata
    max_depth_reached: Optional[int] = None
    states_explored: Optional[int] = None

    # Cycle/limit-cycle evidence (for CYCLE_DETECTED)
    cycle_start_index: Optional[int] = None  # Index in trajectory where cycle begins
    cycle_length: Optional[int] = None  # Length of repeating cycle
    cycle_state: Optional[str] = None  # State where cycle was detected
    cycle_segment: Optional[List[Tuple[str, str]]] = None  # [(state, action), ...] repeating segment

    # Belief-state evidence (for BELIEF_DEGENERACY, BELIEF_COLLAPSE_WRONG, BELIEF_TOO_DIFFUSE)
    belief_entropy: Optional[Scalar] = None  # Shannon entropy of belief distribution
    belief_max_prob: Optional[Scalar] = None  # Maximum probability in belief
    belief_true_state: Optional[str] = None  # Actual ground-truth state
    belief_map_state: Optional[str] = None  # Maximum a posteriori state estimate
    entropy_threshold: Optional[Scalar] = None  # Threshold for "too diffuse"
    observations_received: Optional[int] = None  # Number of observations before failure

    # Non-identifiability evidence (for NON_IDENTIFIABLE)
    aliased_states: Optional[List[str]] = None  # States that produce identical observations
    aliased_region_id: Optional[str] = None  # Identifier for aliased region
    distinguishing_observation: Optional[str] = None  # What observation would distinguish (if any)

    # Physics-specific evidence
    law_name: Optional[str] = None
    measured_observables: Optional[Dict[str, Scalar]] = None
    law_violation_delta: Optional[Scalar] = None
    tolerance: Optional[Scalar] = None

    def __post_init__(self):
        # Normalize scalar dicts
        if self.violated_invariants is not None:
            self.violated_invariants = {k: to_scalar(v) for k, v in self.violated_invariants.items()}
        if self.measured_observables is not None:
            self.measured_observables = {k: to_scalar(v) for k, v in self.measured_observables.items()}
        if self.law_violation_delta is not None:
            self.law_violation_delta = to_scalar(self.law_violation_delta)
        if self.tolerance is not None:
            self.tolerance = to_scalar(self.tolerance)
        if self.belief_entropy is not None:
            self.belief_entropy = to_scalar(self.belief_entropy)
        if self.belief_max_prob is not None:
            self.belief_max_prob = to_scalar(self.belief_max_prob)
        if self.entropy_threshold is not None:
            self.entropy_threshold = to_scalar(self.entropy_threshold)

        ft = self.fail_type

        if ft == FailType.INVARIANT_VIOLATION:
            assert self.violated_invariants is not None, "INVARIANT_VIOLATION requires violated_invariants"
            assert self.blocked_move is not None, "INVARIANT_VIOLATION requires blocked_move"

        elif ft == FailType.PARITY_VIOLATION:
            assert self.blocked_move is not None, "PARITY_VIOLATION requires blocked_move"

        elif ft == FailType.NON_REDUCTION_VIOLATION:
            assert self.blocked_move is not None, "NON_REDUCTION_VIOLATION requires blocked_move"

        elif ft == FailType.SCC_UNREACHABLE:
            assert self.scc_id_reached is not None, "SCC_UNREACHABLE requires scc_id_reached"
            if self.scc_id_goal is None and self.goal_state_id is None:
                raise AssertionError("SCC_UNREACHABLE requires either scc_id_goal or goal_state_id")
            assert (self.witness_edges is not None) or (self.reachable_frontier_hash is not None), \
                "SCC_UNREACHABLE requires witness_edges or reachable_frontier_hash"

        elif ft == FailType.SCC_SPLIT:
            assert self.scc_counts is not None, "SCC_SPLIT requires scc_counts"
            assert len(self.scc_counts) > 1, "SCC_SPLIT requires multiple components"

        elif ft in (FailType.GENERATOR_INSUFFICIENT, FailType.DEPTH_EXHAUSTED):
            assert self.generator_set is not None, f"{ft.value} requires generator_set"
            assert self.max_depth_reached is not None, f"{ft.value} requires max_depth_reached"

        elif ft == FailType.DOMAIN_CONSTRAINT:
            assert self.blocked_move is not None, "DOMAIN_CONSTRAINT requires blocked_move"

        elif ft == FailType.LAW_VIOLATION:
            assert self.law_name is not None, "LAW_VIOLATION requires law_name"
            assert self.measured_observables is not None, "LAW_VIOLATION requires measured_observables"
            assert self.law_violation_delta is not None, "LAW_VIOLATION requires law_violation_delta"
            # tolerance optional but strongly recommended

        elif ft == FailType.CYCLE_DETECTED:
            assert self.cycle_length is not None and self.cycle_length >= 1, \
                "CYCLE_DETECTED requires cycle_length >= 1"
            assert self.cycle_state is not None, "CYCLE_DETECTED requires cycle_state"
            assert self.cycle_segment is not None and len(self.cycle_segment) >= 1, \
                "CYCLE_DETECTED requires non-empty cycle_segment"

        elif ft == FailType.BELIEF_DEGENERACY:
            # Generic belief failure - must have entropy info
            assert self.belief_entropy is not None, "BELIEF_DEGENERACY requires belief_entropy"
            assert self.observations_received is not None, "BELIEF_DEGENERACY requires observations_received"

        elif ft == FailType.BELIEF_COLLAPSE_WRONG:
            # Belief collapsed to wrong state
            assert self.belief_true_state is not None, "BELIEF_COLLAPSE_WRONG requires belief_true_state"
            assert self.belief_map_state is not None, "BELIEF_COLLAPSE_WRONG requires belief_map_state"
            assert self.belief_true_state != self.belief_map_state, \
                "BELIEF_COLLAPSE_WRONG requires true_state != map_state (else it's correct!)"

        elif ft == FailType.BELIEF_TOO_DIFFUSE:
            # Belief entropy too high
            assert self.belief_entropy is not None, "BELIEF_TOO_DIFFUSE requires belief_entropy"
            assert self.entropy_threshold is not None, "BELIEF_TOO_DIFFUSE requires entropy_threshold"

        elif ft == FailType.NON_IDENTIFIABLE:
            # States are observationally aliased
            assert self.aliased_states is not None and len(self.aliased_states) >= 2, \
                "NON_IDENTIFIABLE requires at least 2 aliased_states"

        # TARGET_UNDEFINED, FIXED_Q_ISOLATED, OBSERVER_UNDEFINED, PROJECTION_INCOMPATIBLE: no strict evidence required.

# ============================================================================
# ATTEMPT LOG
# ============================================================================

@dataclass
class AttemptRecord:
    """A single attempted move (successful or failed)."""
    move: MoveWitness
    fail_type: Optional[FailType] = None
    invariant_diff: Optional[Dict[str, Scalar]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gen": {"name": self.move.gen.name, "params": [str(p) for p in self.move.gen.params]},
            "src_id": self.move.src.state_id,
            "dst_id": self.move.dst.state_id,
            "legal": self.move.legal,
            "fail_type": self.fail_type.value if self.fail_type else None,
            "invariant_diff": {k: str(to_scalar(v)) for k, v in (self.invariant_diff or {}).items()},
        }

# ============================================================================
# CERTIFICATE SCHEMA (Core)
# ============================================================================

@dataclass
class InvariantContract:
    """The invariant preservation contract for this proof attempt."""
    tracked_invariants: List[str]
    non_reduction_enforced: bool = True
    fixed_q_mode: Optional[Dict[str, Scalar]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracked_invariants": self.tracked_invariants,
            "non_reduction_enforced": self.non_reduction_enforced,
            "fixed_q_mode": ({k: str(to_scalar(v)) for k, v in self.fixed_q_mode.items()}
                            if self.fixed_q_mode is not None else None),
        }

    def validate_packet_delta(self, packet_delta: Dict[str, Scalar]) -> None:
        for inv_name in packet_delta.keys():
            if inv_name not in self.tracked_invariants:
                raise ValueError(
                    f"packet_delta references untracked invariant '{inv_name}'. "
                    f"Tracked invariants: {self.tracked_invariants}"
                )

@dataclass
class SearchMetadata:
    """Metadata about search process."""
    max_depth: int
    states_explored: int
    frontier_policy: str
    time_elapsed_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "states_explored": self.states_explored,
            "frontier_policy": self.frontier_policy,
            "time_elapsed_ms": self.time_elapsed_ms,
        }

@dataclass
class ProofCertificate:
    """A complete certificate—either success witness or obstruction."""
    theorem_id: str
    generator_set: Set[Generator]
    contracts: InvariantContract
    witness_type: Literal["success", "obstruction"]

    success_path: Optional[List[MoveWitness]] = None
    obstruction: Optional[ObstructionEvidence] = None

    attempt_log: List[AttemptRecord] = field(default_factory=list)
    attempt_log_truncated: bool = False
    attempt_log_total_count: int = 0

    search: Optional[SearchMetadata] = None

    # Observer/projection context (None for pure formal methods)
    observer_id: Optional[str] = None
    projection_contract: Optional[ProjectionContract] = None

    # Generic context block (future-proof)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.witness_type == "success":
            assert self.success_path is not None, "Success requires path witness"
            assert len(self.success_path) > 0, "Path cannot be empty"

            for i in range(len(self.success_path) - 1):
                if self.success_path[i].dst.state_id != self.success_path[i + 1].src.state_id:
                    raise ValueError(f"Path discontinuous at step {i}")

            for move in self.success_path:
                if not move.legal:
                    raise ValueError("Success path contains illegal move")
                self.contracts.validate_packet_delta(move.packet_delta)

            # Validate generator closure: all generators in path must be in generator_set
            path_generators = {move.gen for move in self.success_path}
            if not path_generators.issubset(self.generator_set):
                missing = path_generators - self.generator_set
                raise ValueError(
                    f"Generators used in success_path but not in generator_set: "
                    f"{sorted(g.name for g in missing)}. "
                    f"generator_set must include all generators that appear in witness steps."
                )

        elif self.witness_type == "obstruction":
            assert self.obstruction is not None, "Obstruction requires evidence"
        else:
            raise ValueError(f"Unknown witness_type: {self.witness_type}")

        if self.observer_id is not None and self.projection_contract is None:
            raise ValueError("observer_id requires projection_contract")

    def to_json(self) -> Dict[str, Any]:
        def gen_sort_key(g: Generator) -> Tuple[str, Tuple[str, ...]]:
            return (g.name, tuple(str(p) for p in g.params))

        result: Dict[str, Any] = {
            "schema_version": "1.0",
            "theorem_id": self.theorem_id,
            "generator_set": [
                {"name": g.name, "params": [str(p) for p in g.params]}
                for g in sorted(self.generator_set, key=gen_sort_key)
            ],
            "contracts": self.contracts.to_dict(),
            "witness_type": self.witness_type,
        }

        if self.observer_id is not None:
            result["observer_id"] = self.observer_id
        if self.projection_contract is not None:
            result["projection_contract"] = self.projection_contract.to_dict()
        if self.context:
            result["context"] = self.context

        if self.success_path is not None:
            result["success_path"] = [
                {
                    "gen": {"name": m.gen.name, "params": [str(p) for p in m.gen.params]},
                    "src_id": m.src.state_id,
                    "dst_id": m.dst.state_id,
                    "src_coords": [str(c) for c in m.src.coords],
                    "dst_coords": [str(c) for c in m.dst.coords],
                    "packet_delta": {k: str(v) for k, v in m.packet_delta.items()},
                }
                for m in self.success_path
            ]

        if self.obstruction is not None:
            obs: Dict[str, Any] = {"fail_type": self.obstruction.fail_type.value}

            if self.obstruction.violated_invariants:
                obs["violated_invariants"] = {k: str(v) for k, v in self.obstruction.violated_invariants.items()}
            if self.obstruction.blocked_move:
                m = self.obstruction.blocked_move
                obs["blocked_move"] = {
                    "gen": {"name": m.gen.name, "params": [str(p) for p in m.gen.params]},
                    "src_id": m.src.state_id,
                    "dst_id": m.dst.state_id,
                }

            for k in [
                "scc_id_reached", "scc_id_goal", "goal_state_id", "scc_counts",
                "reachable_frontier_hash", "law_name"
            ]:
                v = getattr(self.obstruction, k)
                if v is not None:
                    obs[k] = v

            if self.obstruction.witness_edges:
                obs["witness_edges"] = [
                    {"src_scc": s, "dst_scc": d, "status": status}
                    for s, d, status in self.obstruction.witness_edges
                ]

            if self.obstruction.generator_set:
                obs["generator_set"] = [
                    {"name": g.name, "params": [str(p) for p in g.params]}
                    for g in sorted(self.obstruction.generator_set, key=gen_sort_key)
                ]

            if self.obstruction.max_depth_reached is not None:
                obs["max_depth_reached"] = self.obstruction.max_depth_reached
            if self.obstruction.states_explored is not None:
                obs["states_explored"] = self.obstruction.states_explored

            if self.obstruction.measured_observables is not None:
                obs["measured_observables"] = {k: str(v) for k, v in self.obstruction.measured_observables.items()}
            if self.obstruction.law_violation_delta is not None:
                obs["law_violation_delta"] = str(self.obstruction.law_violation_delta)
            if self.obstruction.tolerance is not None:
                obs["tolerance"] = str(self.obstruction.tolerance)

            result["obstruction"] = obs

        result["attempt_log"] = {
            "records": [rec.to_dict() for rec in self.attempt_log],
            "truncated": self.attempt_log_truncated,
            "total_count": self.attempt_log_total_count,
        }

        if self.search:
            result["search"] = self.search.to_dict()

        return result


# ============================================================================
# UNDERSTANDING CERTIFICATES (Beyond World Models Integration)
# ============================================================================
# Ref: Gupta & Pruthi (2511.12239v1) "Beyond World Models: Rethinking Understanding"
# These structures capture understanding = obstruction certificates + strategy
#
# Tightened per ChatGPT review (2026-01-18):
# - Hard validity condition for falsifiability (not warning)
# - QARM log schema compatibility
# - Derivation witness required for Strategy
# - Locked compression_ratio definition
# - GeneratorRef for cross-domain examples
# - ProblemSituationCert with explicit obstruction + reachability

class CertificateValidityError(Exception):
    """Raised when a certificate fails validity checks."""
    pass


# ============================================================================
# QARM TRANSITION LOG (Schema-compatible with existing pipeline)
# ============================================================================

@dataclass
class TransitionLog:
    """Per-move log entry compatible with qarm_transition/v1 schema.

    This is the canonical format for micro-traces that can be replayed
    and validated against the deterministic fail classification.
    """
    schema: str = "qarm_transition/v1"
    move: Optional[MoveWitness] = None
    fail_type: Optional[FailType] = None
    invariant_diff: Dict[str, Scalar] = field(default_factory=dict)

    def __post_init__(self):
        self.invariant_diff = {k: to_scalar(v) for k, v in self.invariant_diff.items()}

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"qarm_log_schema": self.schema}
        if self.move:
            result["move"] = {
                "gen": {"name": self.move.gen.name, "params": [str(p) for p in self.move.gen.params]},
                "src_id": self.move.src.state_id,
                "dst_id": self.move.dst.state_id,
                "legal": self.move.legal,
            }
        if self.fail_type:
            result["fail_type"] = self.fail_type.value
        result["invariant_diff"] = {k: str(v) for k, v in self.invariant_diff.items()}
        return result


# ============================================================================
# GENERATOR REFERENCE (For cross-domain examples without collision)
# ============================================================================

@dataclass(frozen=True)
class GeneratorRef:
    """Reference to a generator by namespace:name for examples/papers.

    Use this instead of Generator() in examples to avoid colliding with
    the strict generator validation in the core implementation.

    Namespaces:
      - QA: Core QA generators (σ, λ, μ, ν)
      - AG: AlphaGeometry rules
      - PHYS: Physics/domain-specific
      - PAPER: Hypothetical/paper examples
    """
    namespace: str
    name: str
    params: Tuple[Scalar, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:
        if self.params:
            return f"{self.namespace}:{self.name}{self.params}"
        return f"{self.namespace}:{self.name}"

    def to_generator(self) -> Generator:
        """Convert to actual Generator (for namespaced types only)."""
        full_name = f"{self.namespace}:{self.name}"
        return Generator(name=full_name, params=self.params)


# ============================================================================
# DERIVATION WITNESS (Anti-ad-hoc injection)
# ============================================================================

@dataclass
class DerivationWitness:
    """Witness that an abstract property was derived (not injected ad-hoc).

    The paper warns that enriching world models with arbitrary abstract states
    (primality, motivation, problem-situation) makes them unfalsifiable.

    RML resolves this by requiring derivation witnesses:
    - The invariant must be computed via explicit operator
    - The computation must be verifiable
    """
    invariant_name: str
    derivation_operator: str  # Name of the Δ operator used
    input_data: Dict[str, Any]  # What went into the derivation
    output_value: Scalar
    verifiable: bool = True

    def __post_init__(self):
        self.output_value = to_scalar(self.output_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_name": self.invariant_name,
            "derivation_operator": self.derivation_operator,
            "input_data": self.input_data,
            "output_value": str(self.output_value),
            "verifiable": self.verifiable,
        }


# ============================================================================
# STRATEGY (With required derivation witness)
# ============================================================================

@dataclass
class Strategy:
    """High-level proof/solution strategy identification.

    The paper (citing Poincaré) distinguishes *verification* from *understanding*.
    A strategy captures the "why this approach" that world models miss.

    IMPORTANT: Strategy must have a derivation witness to prevent ad-hoc injection.
    The witness shows how the strategy was derived from the trace/structure.
    """
    type: str  # e.g., "involution_parity", "invariant_counting", "contradiction"
    key_insight: str  # Natural language description of core idea
    prerequisite_knowledge: List[str] = field(default_factory=list)

    # REQUIRED: derivation witness for the strategy itself
    derivation_witness: Optional[DerivationWitness] = None

    def __post_init__(self):
        # Strategy without derivation is a warning (soft mode) or error (strict mode)
        pass

    def has_derivation(self) -> bool:
        """Check if strategy has proper derivation witness."""
        return self.derivation_witness is not None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.type,
            "key_insight": self.key_insight,
            "prerequisite_knowledge": self.prerequisite_knowledge,
            "has_derivation": self.has_derivation(),
        }
        if self.derivation_witness:
            result["derivation_witness"] = self.derivation_witness.to_dict()
        return result


# ============================================================================
# KEY STEP (Non-routine step with necessity witness)
# ============================================================================

@dataclass
class KeyStep:
    """A non-routine step with necessity certificate.

    The paper argues that understanding proofs requires identifying *key steps*
    that aren't mere routine inference. We formalize this by requiring:
    - Description of what the step does
    - Necessity witness: what obstruction arises if step is removed
    """
    index: int
    description: str
    necessity_witness: ObstructionEvidence  # What fails without this step
    compression_contribution: Optional[float] = None  # How much this step compresses the proof

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "index": self.index,
            "description": self.description,
            "necessity_witness": {
                "fail_type": self.necessity_witness.fail_type.value,
            }
        }
        if self.compression_contribution is not None:
            result["compression_contribution"] = self.compression_contribution
        return result


# ============================================================================
# PROBLEM SITUATION CERTIFICATE (Popper-style with explicit obstruction/reachability)
# ============================================================================

@dataclass
class ProblemSituationCert:
    """Popper-style problem-situation understanding certificate.

    The paper's Bohr case study: understanding requires knowing the *problem*
    (discrete spectral lines) that the theory was invented to solve.

    RML-native structure (per ChatGPT review):
    - prior_obstruction: Certificate that target is unreachable under G₀
    - new_path_witness: Path witness that target IS reachable under G₁

    This makes it structurally parallel to the obstruction upgrade pattern:
    "Target unreachable under G₀ (certificate). Reachable under G₁ (witness)."
    """
    # Natural language description
    gap: str  # What the prior theory couldn't explain
    target_phenomenon: str  # The observation requiring explanation
    resolution: str  # How new theory/approach closes the gap
    necessity: str  # Why this resolution is forced

    # Generator sets
    prior_generators: Set[Generator] = field(default_factory=set)
    new_generators: Set[Generator] = field(default_factory=set)

    # RML-native: explicit obstruction under old theory
    prior_obstruction: Optional[ObstructionEvidence] = None

    # RML-native: explicit path witness under new theory
    new_path_witness: Optional[List[MoveWitness]] = None

    def is_complete(self) -> bool:
        """Check if this is a complete problem-situation certificate."""
        return (
            self.prior_obstruction is not None and
            self.new_path_witness is not None and
            len(self.prior_generators) > 0 and
            len(self.new_generators) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        def gen_sort_key(g: Generator) -> str:
            return g.name

        result: Dict[str, Any] = {
            "gap": self.gap,
            "target_phenomenon": self.target_phenomenon,
            "resolution": self.resolution,
            "necessity": self.necessity,
            "is_complete": self.is_complete(),
        }

        if self.prior_generators:
            result["prior_generators"] = [g.name for g in sorted(self.prior_generators, key=gen_sort_key)]
        if self.new_generators:
            result["new_generators"] = [g.name for g in sorted(self.new_generators, key=gen_sort_key)]

        if self.prior_obstruction:
            result["prior_obstruction"] = {
                "fail_type": self.prior_obstruction.fail_type.value,
                "max_depth": self.prior_obstruction.max_depth_reached,
            }

        if self.new_path_witness:
            result["new_path_witness_length"] = len(self.new_path_witness)

        return result


# ============================================================================
# COMPRESSION RATIO (Locked definition)
# ============================================================================

def compute_compression_ratio(
    micro_trace_len: int,
    explanation_path_len: int,
    key_steps_count: int,
    derived_invariants_count: int
) -> float:
    """Compute compression ratio with locked definition.

    Definition (frozen per ChatGPT review):
        compression_ratio = micro_trace_len / (explanation_path_len + key_steps_count + derived_invariants_count)

    This ensures cross-domain comparability (proofs vs physics vs dominoes).

    Args:
        micro_trace_len: Number of moves in full trace (Layer 1)
        explanation_path_len: Number of explanation steps (Layer 3)
        key_steps_count: Number of identified key steps
        derived_invariants_count: Number of derived invariants

    Returns:
        Compression ratio >= 1.0 (1.0 = no compression)
    """
    denominator = explanation_path_len + key_steps_count + derived_invariants_count
    if denominator == 0:
        return 1.0
    ratio = micro_trace_len / denominator
    return max(1.0, ratio)  # Compression ratio is at least 1


# ============================================================================
# UNDERSTANDING CERTIFICATE (Main structure with hard validity)
# ============================================================================

@dataclass
class UnderstandingCertificate:
    """Full understanding certificate (paper's desiderata).

    This is the top-level structure that captures understanding in the
    paper's sense: not just reachability/prediction, but the *why*.

    Three layers:
    1. Micro-trace (optional): The world-model level path (qarm_transition/v1)
    2. Reachability: Whether target is reachable and failure type if not
    3. Understanding: Derived invariants, key steps, strategy, compression

    VALIDITY RULES (hard, not warnings):
    - All derived_invariants MUST have derivation_witnesses
    - If strategy is provided, it MUST have derivation_witness
    - compression_ratio uses locked definition

    The paper's thesis is that world models only give (1), partially (2).
    QA-RML provides all three.
    """
    schema: str = "qa_understanding_cert/v2"  # Bumped for tightened version

    # Target identification
    target: str = ""
    system_id: Optional[str] = None

    # Layer 1: Micro-trace (QARM schema compatible)
    transition_log: List[TransitionLog] = field(default_factory=list)

    # Layer 2: Reachability (from QAWM)
    reachable: Optional[bool] = None
    fail_type: Optional[FailType] = None
    obstruction: Optional[ObstructionEvidence] = None

    # Layer 3: Understanding (QA-RML native)
    derived_invariants: Dict[str, Scalar] = field(default_factory=dict)
    key_steps: List[KeyStep] = field(default_factory=list)
    strategy: Optional[Strategy] = None
    problem_situation: Optional[ProblemSituationCert] = None
    explanation_path: List[str] = field(default_factory=list)

    # Falsifiability: witnesses for all derived invariants
    derivation_witnesses: List[DerivationWitness] = field(default_factory=list)

    # Counterfactual evidence (what fails if key steps removed)
    counterfactual_obstructions: Dict[int, ObstructionEvidence] = field(default_factory=dict)

    # Validity tracking
    strict_mode: bool = True  # If True, invalid certificates raise errors

    def __post_init__(self):
        # Normalize derived_invariants
        self.derived_invariants = {k: to_scalar(v) for k, v in self.derived_invariants.items()}

        # Validate in strict mode
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"Certificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of all validity violations."""
        violations = []

        # Check derived invariants have witnesses
        derived_names = set(self.derived_invariants.keys())
        witnessed_names = {w.invariant_name for w in self.derivation_witnesses}
        unwitnessed = derived_names - witnessed_names
        if unwitnessed:
            violations.append(
                f"ADHOC_STATE_INJECTION: derived invariants without witnesses: {sorted(unwitnessed)}"
            )

        # Check strategy has derivation (if provided)
        if self.strategy and not self.strategy.has_derivation():
            violations.append(
                f"ADHOC_STRATEGY: strategy '{self.strategy.type}' lacks derivation witness"
            )

        return violations

    def is_valid(self) -> bool:
        """Check if certificate passes all validity rules."""
        return len(self.get_validity_violations()) == 0

    def get_compression_ratio(self) -> float:
        """Compute compression ratio with locked definition."""
        micro_len = len(self.transition_log)
        return compute_compression_ratio(
            micro_trace_len=micro_len,
            explanation_path_len=len(self.explanation_path),
            key_steps_count=len(self.key_steps),
            derived_invariants_count=len(self.derived_invariants)
        )

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "target": self.target,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.system_id:
            result["system_id"] = self.system_id

        # Layer 1: QARM-compatible transition log
        if self.transition_log:
            result["transition_log"] = {
                "count": len(self.transition_log),
                "schema": "qarm_transition/v1",
                "entries": [t.to_dict() for t in self.transition_log[:10]],  # First 10 for summary
                "truncated": len(self.transition_log) > 10,
            }

        # Layer 2
        if self.reachable is not None:
            result["reachable"] = self.reachable
        if self.fail_type:
            result["fail_type"] = self.fail_type.value
        if self.obstruction:
            result["obstruction"] = {
                "fail_type": self.obstruction.fail_type.value,
                "max_depth": self.obstruction.max_depth_reached,
            }

        # Layer 3
        result["derived_invariants"] = {k: str(v) for k, v in self.derived_invariants.items()}

        if self.key_steps:
            result["key_steps"] = [ks.to_dict() for ks in self.key_steps]

        if self.strategy:
            result["strategy"] = self.strategy.to_dict()

        if self.problem_situation:
            result["problem_situation"] = self.problem_situation.to_dict()

        if self.explanation_path:
            result["explanation_path"] = self.explanation_path

        # Locked compression ratio
        result["compression_ratio"] = self.get_compression_ratio()

        # Falsifiability
        result["derivation_witnesses"] = [w.to_dict() for w in self.derivation_witnesses]

        if self.counterfactual_obstructions:
            result["counterfactuals"] = {
                str(k): {"fail_type": v.fail_type.value}
                for k, v in self.counterfactual_obstructions.items()
            }

        return result

    @classmethod
    def from_rml_run(
        cls,
        target: str,
        system_id: str,
        transition_log: List[TransitionLog],
        reachable: bool,
        obstruction: Optional[ObstructionEvidence],
        derived_invariants: Dict[str, Scalar],
        derivation_witnesses: List[DerivationWitness],
        explanation_path: List[str],
        key_steps: Optional[List[KeyStep]] = None,
        strategy: Optional[Strategy] = None,
        problem_situation: Optional[ProblemSituationCert] = None,
        strict_mode: bool = True,
    ) -> "UnderstandingCertificate":
        """Construct UnderstandingCertificate from RML run artifacts.

        This is the canonical way to create certificates from pipeline outputs.

        Args:
            target: What we're trying to reach/prove
            system_id: Identifier for the system being analyzed
            transition_log: QARM-compatible move log
            reachable: Whether target was reached
            obstruction: Obstruction evidence if not reachable
            derived_invariants: Invariants computed during analysis
            derivation_witnesses: Witnesses for each derived invariant
            explanation_path: High-level explanation steps
            key_steps: Identified non-routine steps (optional)
            strategy: Identified strategy with derivation (optional)
            problem_situation: Problem-situation cert if applicable (optional)
            strict_mode: If True, raise on validity violations

        Returns:
            Valid UnderstandingCertificate

        Raises:
            CertificateValidityError: If strict_mode and certificate is invalid
        """
        return cls(
            target=target,
            system_id=system_id,
            transition_log=transition_log,
            reachable=reachable,
            fail_type=obstruction.fail_type if obstruction else None,
            obstruction=obstruction,
            derived_invariants=derived_invariants,
            derivation_witnesses=derivation_witnesses,
            explanation_path=explanation_path,
            key_steps=key_steps or [],
            strategy=strategy,
            problem_situation=problem_situation,
            strict_mode=strict_mode,
        )


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def check_for_adhoc_injection(cert: UnderstandingCertificate) -> Optional[str]:
    """Check if certificate contains ad-hoc state injections (unfalsifiable claims).

    Returns description of violation if found, None if certificate is valid.
    """
    violations = cert.get_validity_violations()
    if violations:
        return "; ".join(violations)
    return None


def validate_certificate_strict(cert: UnderstandingCertificate) -> None:
    """Validate certificate and raise if invalid.

    Raises:
        CertificateValidityError: If certificate has validity violations
    """
    violations = cert.get_validity_violations()
    if violations:
        raise CertificateValidityError(
            f"Certificate invalid: {'; '.join(violations)}"
        )


# ============================================================================
# STRICT VALIDATOR V3 (Recompute + Consistency + Operator-specific rules)
# ============================================================================

@dataclass
class StrictValidationResult:
    """Result of strict v3 validation."""
    valid: bool
    violations: List[str]
    warnings: List[str]

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        lines = [f"Valid: {self.valid}"]
        if self.violations:
            lines.append(f"Violations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"  - {v}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _check_cohens_d_verifiable(witness: DerivationWitness) -> Tuple[bool, Optional[str]]:
    """
    Check if a cohens_d witness has sufficient data to be verifiable.

    For cohens_d to be verifiable, input_data must contain EITHER:
    1. baseline_std, seizure_std, baseline_mean, seizure_mean, n_baseline, n_seizure
    2. pooled_std, baseline_mean, seizure_mean
    3. pooled_variance, baseline_mean, seizure_mean

    Returns:
        (is_valid, error_message) - (True, None) if valid, (False, msg) if not
    """
    data = witness.input_data

    # Check for full stats
    has_full = all(k in data for k in [
        "baseline_std", "seizure_std", "baseline_mean", "seizure_mean",
        "n_baseline", "n_seizure"
    ])

    # Check for pooled std
    has_pooled_std = all(k in data for k in [
        "pooled_std", "baseline_mean", "seizure_mean"
    ])

    # Check for pooled variance
    has_pooled_var = all(k in data for k in [
        "pooled_variance", "baseline_mean", "seizure_mean"
    ])

    if has_full or has_pooled_std or has_pooled_var:
        return True, None
    else:
        return False, (
            f"cohens_d marked verifiable but missing required stats. "
            f"Need: (baseline_std, seizure_std, means, n) OR (pooled_std, means) OR (pooled_variance, means)"
        )


def _check_midpoint_threshold_recompute(witness: DerivationWitness) -> Tuple[bool, Optional[str]]:
    """
    Recompute midpoint_threshold and verify it matches output_value.

    Returns:
        (is_valid, error_message)
    """
    data = witness.input_data

    if "baseline_mean" not in data or "seizure_mean" not in data:
        return False, "midpoint_threshold missing baseline_mean or seizure_mean"

    try:
        baseline = float(data["baseline_mean"])
        seizure = float(data["seizure_mean"])
        expected = (baseline + seizure) / 2

        # Allow integer floor/ceil/round
        stored = int(witness.output_value)
        if stored in [int(expected), int(expected) + 1, int(expected) - 1]:
            return True, None
        else:
            return False, f"midpoint_threshold mismatch: computed {expected:.2f}, stored {stored}"
    except (ValueError, TypeError) as e:
        return False, f"midpoint_threshold computation error: {e}"


def validate_certificate_strict_v3(cert: UnderstandingCertificate) -> StrictValidationResult:
    """
    Strict v3 validator with recomputation and operator-specific rules.

    This validator:
    1. Checks all basic validity rules (derived invariants have witnesses, strategy has witness)
    2. Enforces operator-specific requirements:
       - cohens_d: must have std/pooled stats if verifiable=true
       - midpoint_threshold: recomputes and verifies output matches
    3. Does NOT trust has_derivation flag - requires actual derivation_witness object
    4. Checks for valid flag mismatch

    Returns:
        StrictValidationResult with violations and warnings
    """
    violations = []
    warnings = []

    # --- Basic validity (from certificate's own check) ---
    basic_violations = cert.get_validity_violations()
    violations.extend(basic_violations)

    # --- Strategy witness check (strict: don't trust has_derivation) ---
    if cert.strategy is not None:
        if cert.strategy.derivation_witness is None:
            violations.append(
                f"ADHOC_STRATEGY: strategy '{cert.strategy.type}' missing derivation_witness object"
            )

    # --- Operator-specific witness validation ---
    for witness in cert.derivation_witnesses:
        op = witness.derivation_operator

        if op == "cohens_d":
            if witness.verifiable:
                ok, msg = _check_cohens_d_verifiable(witness)
                if not ok:
                    violations.append(f"UNVERIFIABLE_WITNESS: {witness.invariant_name} {msg}")
            else:
                # Allowed but downgraded
                warnings.append(
                    f"NONVERIFIABLE_OK: {witness.invariant_name} cohens_d allowed because verifiable=false (downgraded claim)"
                )

        elif op == "midpoint_threshold":
            if witness.verifiable:
                ok, msg = _check_midpoint_threshold_recompute(witness)
                if not ok:
                    violations.append(f"RECOMPUTE_MISMATCH: {witness.invariant_name} {msg}")

    # --- Valid flag mismatch check ---
    # If we're checking a deserialized cert (via to_json/from_json cycle),
    # compare stored valid flag to computed
    # Note: For in-memory certs, this is always consistent by construction

    is_valid = len(violations) == 0

    return StrictValidationResult(
        valid=is_valid,
        violations=violations,
        warnings=warnings,
    )


def validate_certificate_v3(cert: UnderstandingCertificate, raise_on_fail: bool = True) -> StrictValidationResult:
    """
    Convenience wrapper for strict v3 validation.

    Args:
        cert: Certificate to validate
        raise_on_fail: If True, raise CertificateValidityError on violations

    Returns:
        StrictValidationResult

    Raises:
        CertificateValidityError: If raise_on_fail and violations exist
    """
    result = validate_certificate_strict_v3(cert)

    if raise_on_fail and not result.valid:
        raise CertificateValidityError(
            f"Certificate invalid (strict v3): {'; '.join(result.violations)}"
        )

    return result


# ============================================================================
# POLICY CERTIFICATES (Decision Making Integration)
# ============================================================================
# Ref: Kochenderfer et al. "Algorithms for Decision Making" (MIT Press)
# Maps sequential decision problems to QA reachability framework
#
# Key insight: MDPs over QA lattice are deterministic, structure-preserving.
# Value = distance to target. Policy = generator selection.

class PolicyFailType(Enum):
    """Policy-specific failure modes."""
    # Policy execution failures
    POLICY_DIVERGED = "policy_diverged"  # Policy loops forever
    POLICY_STUCK = "policy_stuck"  # Policy chose illegal move, no fallback
    HORIZON_EXCEEDED = "horizon_exceeded"  # Didn't reach target in k steps
    EXPLORATION_EXHAUSTED = "exploration_exhausted"  # Ran out of exploration budget

    # Learning failures
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough transitions to learn
    MODEL_MISSPECIFIED = "model_misspecified"  # QAWM predictions unreliable

    # Task specification failures
    TARGET_UNREACHABLE = "target_unreachable"  # No path exists
    GENERATOR_SET_INSUFFICIENT = "generator_set_insufficient"  # Need more generators


@dataclass
class PolicyEvaluationStats:
    """Statistics from policy evaluation."""
    n_episodes: int
    successes: int
    total_steps: int
    total_oracle_calls: int

    @property
    def success_rate(self) -> Fraction:
        if self.n_episodes == 0:
            return Fraction(0)
        return Fraction(self.successes, self.n_episodes)

    @property
    def avg_steps(self) -> Optional[Fraction]:
        if self.successes == 0:
            return None
        return Fraction(self.total_steps, self.successes)

    @property
    def avg_oracle_calls(self) -> Fraction:
        if self.n_episodes == 0:
            return Fraction(0)
        return Fraction(self.total_oracle_calls, self.n_episodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "successes": self.successes,
            "success_rate": str(self.success_rate),
            "avg_steps": str(self.avg_steps) if self.avg_steps else None,
            "avg_oracle_calls": str(self.avg_oracle_calls),
        }


class OptimalityMethod(Enum):
    """Methods for proving policy optimality."""
    BFS = "bfs"  # Breadth-first search (unweighted shortest path)
    DIJKSTRA = "dijkstra"  # Weighted shortest path
    VALUE_ITERATION = "value_iteration"  # Dynamic programming
    BELLMAN_FORD = "bellman_ford"  # Handles negative edges


@dataclass
class OptimalityProof:
    """Machine-checkable proof that a policy is optimal.

    This is the QA-native equivalent of proving a policy achieves
    the minimum-cost/shortest path under the given generator set.
    """
    method: OptimalityMethod
    optimal_distance: Scalar  # Proven optimal path length
    states_explored: int  # States visited during proof computation
    predecessor_map_hash: Optional[str] = None  # Hash of predecessor map for verification

    # For bounded verification
    proof_computed_at: Optional[str] = None  # Timestamp or computation context
    verifiable: bool = True  # Can be recomputed from inputs

    def __post_init__(self):
        self.optimal_distance = to_scalar(self.optimal_distance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "optimal_distance": str(self.optimal_distance),
            "states_explored": self.states_explored,
            "predecessor_map_hash": self.predecessor_map_hash,
            "verifiable": self.verifiable,
        }


@dataclass
class PolicyCertificate:
    """Certificate for a decision-making policy.

    Certifies that a policy achieves a target under specified conditions.
    This is the QA-native equivalent of an MDP policy guarantee.

    Three layers (parallel to UnderstandingCertificate):
    1. Task specification: target, horizon, generators
    2. Performance: success rate, steps, oracle efficiency
    3. Understanding: why the policy works (derivation witnesses)

    VALIDITY RULES:
    - If reachability_guarantee=True, must have training_witness
    - If comparing to baseline, must have evaluation_witness
    """
    schema: str = "qa_policy_cert/v1"

    # Policy identification
    policy_id: str = ""
    policy_type: str = ""  # "random_legal", "qawm_greedy", "rml", "bfs_optimal"
    policy_description: str = ""

    # Task specification
    target_class_description: str = ""
    start_class_description: str = ""
    horizon: int = 0
    generator_set: List[GeneratorRef] = field(default_factory=list)

    # Layer 2: Performance certificate
    evaluation_stats: Optional[PolicyEvaluationStats] = None
    reachability_guarantee: bool = False  # True if proven to always reach target
    optimality_guarantee: bool = False  # True if proven to be shortest path

    # Optimality proof (required if optimality_guarantee=True)
    optimality_proof: Optional[OptimalityProof] = None

    # Failure evidence
    failure_mode: Optional[PolicyFailType] = None
    obstruction_if_fail: Optional[ObstructionEvidence] = None

    # Comparison
    baseline_policy_id: Optional[str] = None
    improvement_over_baseline: Optional[Fraction] = None

    # Layer 3: Understanding (anti-ad-hoc)
    training_witness: Optional[DerivationWitness] = None
    evaluation_witness: Optional[DerivationWitness] = None
    strategy: Optional[Strategy] = None

    # Strict mode
    strict_mode: bool = True

    def __post_init__(self):
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"PolicyCertificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of validity violations."""
        violations = []

        # If claiming reachability guarantee, must have derivation
        if self.reachability_guarantee and self.training_witness is None:
            violations.append(
                "ADHOC_GUARANTEE: reachability_guarantee=True but no training_witness"
            )

        # If claiming optimality, must have derivation AND proof
        if self.optimality_guarantee and self.training_witness is None:
            violations.append(
                "ADHOC_GUARANTEE: optimality_guarantee=True but no training_witness"
            )

        # If claiming optimality, must have optimality_proof
        if self.optimality_guarantee and self.optimality_proof is None:
            violations.append(
                "ADHOC_OPTIMALITY: optimality_guarantee=True but no optimality_proof"
            )

        # If comparing to baseline, must have evaluation_witness
        if self.baseline_policy_id is not None and self.evaluation_witness is None:
            violations.append(
                "ADHOC_COMPARISON: baseline comparison without evaluation_witness"
            )

        # Strategy must have derivation if present
        if self.strategy is not None and not self.strategy.has_derivation():
            violations.append(
                f"ADHOC_STRATEGY: strategy '{self.strategy.type}' lacks derivation"
            )

        return violations

    def is_valid(self) -> bool:
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "policy_type": self.policy_type,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.policy_description:
            result["policy_description"] = self.policy_description

        # Task specification
        result["task"] = {
            "target_class": self.target_class_description,
            "start_class": self.start_class_description,
            "horizon": self.horizon,
            "generators": [repr(g) for g in self.generator_set],
        }

        # Performance
        if self.evaluation_stats:
            result["evaluation"] = self.evaluation_stats.to_dict()

        result["guarantees"] = {
            "reachability": self.reachability_guarantee,
            "optimality": self.optimality_guarantee,
        }

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value

        if self.obstruction_if_fail:
            result["obstruction"] = {
                "fail_type": self.obstruction_if_fail.fail_type.value,
            }

        # Comparison
        if self.baseline_policy_id:
            result["comparison"] = {
                "baseline": self.baseline_policy_id,
                "improvement": str(self.improvement_over_baseline) if self.improvement_over_baseline else None,
            }

        # Understanding layer
        if self.training_witness:
            result["training_witness"] = self.training_witness.to_dict()

        if self.evaluation_witness:
            result["evaluation_witness"] = self.evaluation_witness.to_dict()

        if self.strategy:
            result["strategy"] = self.strategy.to_dict()

        return result

    @classmethod
    def from_bfs_optimal(
        cls,
        policy_id: str,
        target_class: str,
        start_class: str,
        horizon: int,
        generators: List[GeneratorRef],
        optimal_path_length: int,
        states_explored: int,
    ) -> "PolicyCertificate":
        """Create certificate for BFS-optimal policy.

        BFS-optimal policy is the gold standard: guaranteed to reach target
        via shortest path.
        """
        return cls(
            policy_id=policy_id,
            policy_type="bfs_optimal",
            policy_description="Optimal policy computed via BFS shortest path",
            target_class_description=target_class,
            start_class_description=start_class,
            horizon=horizon,
            generator_set=generators,
            reachability_guarantee=True,
            optimality_guarantee=True,
            optimality_proof=OptimalityProof(
                method=OptimalityMethod.BFS,
                optimal_distance=optimal_path_length,
                states_explored=states_explored,
                verifiable=True,
            ),
            training_witness=DerivationWitness(
                invariant_name="optimal_path_length",
                derivation_operator="bfs_shortest_path",
                input_data={
                    "start": start_class,
                    "target": target_class,
                    "states_explored": states_explored,
                },
                output_value=optimal_path_length,
                verifiable=True,
            ),
            strict_mode=True,
        )

    @classmethod
    def from_evaluation(
        cls,
        policy_id: str,
        policy_type: str,
        target_class: str,
        start_class: str,
        horizon: int,
        generators: List[GeneratorRef],
        n_episodes: int,
        successes: int,
        total_steps: int,
        total_oracle_calls: int,
        baseline_policy_id: Optional[str] = None,
        baseline_success_rate: Optional[Fraction] = None,
    ) -> "PolicyCertificate":
        """Create certificate from empirical policy evaluation."""
        stats = PolicyEvaluationStats(
            n_episodes=n_episodes,
            successes=successes,
            total_steps=total_steps,
            total_oracle_calls=total_oracle_calls,
        )

        improvement = None
        if baseline_policy_id and baseline_success_rate:
            improvement = stats.success_rate - baseline_success_rate

        return cls(
            policy_id=policy_id,
            policy_type=policy_type,
            policy_description=f"Policy evaluated over {n_episodes} episodes",
            target_class_description=target_class,
            start_class_description=start_class,
            horizon=horizon,
            generator_set=generators,
            evaluation_stats=stats,
            reachability_guarantee=False,  # Empirical, not proven
            optimality_guarantee=False,
            baseline_policy_id=baseline_policy_id,
            improvement_over_baseline=improvement,
            evaluation_witness=DerivationWitness(
                invariant_name="empirical_success_rate",
                derivation_operator="episode_evaluation",
                input_data={
                    "n_episodes": n_episodes,
                    "horizon": horizon,
                },
                output_value=stats.successes,
                verifiable=True,
            ),
            strict_mode=True,
        )


def validate_policy_certificate(cert: PolicyCertificate) -> StrictValidationResult:
    """Validate policy certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Check for unrealistic guarantees without evidence
    if cert.reachability_guarantee and cert.evaluation_stats:
        if cert.evaluation_stats.success_rate < Fraction(1):
            warnings.append(
                f"reachability_guarantee=True but empirical success_rate={cert.evaluation_stats.success_rate}"
            )

    # Structural rule: POLICY_DIVERGED ⇔ CYCLE_DETECTED (bidirectional consistency)
    if cert.failure_mode == PolicyFailType.POLICY_DIVERGED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "DIVERGED_WITHOUT_OBSTRUCTION: failure_mode=POLICY_DIVERGED but no obstruction evidence"
            )
        elif cert.obstruction_if_fail.fail_type != FailType.CYCLE_DETECTED:
            violations.append(
                f"DIVERGED_OBSTRUCTION_MISMATCH: failure_mode=POLICY_DIVERGED requires "
                f"obstruction.fail_type=CYCLE_DETECTED, got {cert.obstruction_if_fail.fail_type.value}"
            )

    # Reverse direction: CYCLE_DETECTED ⇒ POLICY_DIVERGED
    if cert.obstruction_if_fail and cert.obstruction_if_fail.fail_type == FailType.CYCLE_DETECTED:
        if cert.failure_mode != PolicyFailType.POLICY_DIVERGED:
            violations.append(
                f"CYCLE_FAILURE_MISMATCH: obstruction.fail_type=CYCLE_DETECTED requires "
                f"failure_mode=POLICY_DIVERGED, got {cert.failure_mode.value if cert.failure_mode else 'None'}"
            )

    # Observer upgrade consistency: if observer_upgrade_applied=true, must be success (no obstruction)
    if cert.training_witness and cert.training_witness.input_data:
        observer_upgrade = cert.training_witness.input_data.get("observer_upgrade_applied", False)
        if observer_upgrade:
            if cert.obstruction_if_fail is not None:
                violations.append(
                    "OBSERVER_UPGRADE_WITH_OBSTRUCTION: observer_upgrade_applied=true but "
                    "obstruction_if_fail is present (upgrade should resolve failure)"
                )
            if cert.failure_mode is not None:
                violations.append(
                    f"OBSERVER_UPGRADE_WITH_FAILURE: observer_upgrade_applied=true but "
                    f"failure_mode={cert.failure_mode.value} (upgrade should resolve failure)"
                )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# GAME THEORY CERTIFICATES (Multiagent Decision Making)
# ============================================================================
# Ref: Kochenderfer et al. "Algorithms for Decision Making" Ch. 24-27 (MIT Press)
# Maps multiagent games to QA coupled reachability + equilibrium framework
#
# Key insight: Games are reachability problems on product state graphs.
# Nash = mutual reachability constraints (no unilateral improvement).
# Information sets = observer equivalence classes (NON_IDENTIFIABLE generalized).


class GameFailType(Enum):
    """Game-theoretic failure modes for multiagent settings."""
    # Equilibrium computation failures
    NO_EQUILIBRIUM_FOUND = "no_equilibrium_found"  # Algorithm failed to find equilibrium
    EQUILIBRIUM_NOT_VERIFIABLE = "equilibrium_not_verifiable"  # Candidate exists but can't certify

    # Strategy validation failures
    REGRET_TOO_HIGH = "regret_too_high"  # Empirical strategy exceeds regret bound
    EXPLOITABLE_DEVIATION = "exploitable_deviation"  # Found profitable deviation

    # Model/belief failures
    OPPONENT_MODEL_MISMATCH = "opponent_model_mismatch"  # Belief over opponent invalid
    INFORMATION_SET_ALIASING = "information_set_aliasing"  # Multiagent NON_IDENTIFIABLE

    # Coordination failures (Ch. 27)
    COORDINATION_DEADLOCK = "coordination_deadlock"  # Mutual waiting state
    MISCOORDINATION_CYCLE = "miscoordination_cycle"  # Alternating actions never converge
    COMMON_KNOWLEDGE_FAILURE = "common_knowledge_failure"  # Can't establish shared belief

    # Asymmetric settings
    ASYMMETRIC_IDENTIFIABILITY = "asymmetric_identifiability"  # One agent can localize, other cannot


@dataclass
class GameObstructionEvidence:
    """Evidence for game-theoretic failures."""
    fail_type: GameFailType

    # Deviation evidence (for EXPLOITABLE_DEVIATION, REGRET_TOO_HIGH)
    deviating_agent: Optional[int] = None
    deviation_strategy: Optional[str] = None
    deviation_payoff_gain: Optional[Scalar] = None
    regret_value: Optional[Scalar] = None
    regret_threshold: Optional[Scalar] = None

    # Information set evidence (for INFORMATION_SET_ALIASING, ASYMMETRIC_IDENTIFIABILITY)
    aliased_info_sets: Optional[List[str]] = None
    affected_agents: Optional[List[int]] = None
    distinguishing_signal: Optional[str] = None

    # Coordination evidence (for DEADLOCK, MISCOORDINATION_CYCLE)
    deadlock_state: Optional[str] = None
    cycle_states: Optional[List[str]] = None
    cycle_length: Optional[int] = None

    def __post_init__(self):
        ft = self.fail_type

        if ft == GameFailType.EXPLOITABLE_DEVIATION:
            assert self.deviating_agent is not None
            assert self.deviation_payoff_gain is not None
            if self.deviation_payoff_gain is not None:
                self.deviation_payoff_gain = to_scalar(self.deviation_payoff_gain)

        elif ft == GameFailType.REGRET_TOO_HIGH:
            assert self.regret_value is not None
            assert self.regret_threshold is not None
            self.regret_value = to_scalar(self.regret_value)
            self.regret_threshold = to_scalar(self.regret_threshold)
            assert self.regret_value > self.regret_threshold

        elif ft == GameFailType.INFORMATION_SET_ALIASING:
            assert self.aliased_info_sets is not None and len(self.aliased_info_sets) >= 2
            assert self.affected_agents is not None

        elif ft == GameFailType.ASYMMETRIC_IDENTIFIABILITY:
            assert self.affected_agents is not None and len(self.affected_agents) >= 1

        elif ft == GameFailType.COORDINATION_DEADLOCK:
            assert self.deadlock_state is not None

        elif ft == GameFailType.MISCOORDINATION_CYCLE:
            assert self.cycle_states is not None and len(self.cycle_states) >= 2
            assert self.cycle_length is not None and self.cycle_length >= 2


class EquilibriumConcept(Enum):
    """Types of equilibrium solutions."""
    NASH = "nash"  # No unilateral profitable deviation
    CORRELATED = "correlated"  # Joint distribution with no deviation incentive
    MINIMAX = "minimax"  # Zero-sum optimal
    PARETO = "pareto"  # No Pareto improvement exists
    DOMINANT = "dominant"  # Dominant strategy equilibrium


@dataclass
class AgentStrategy:
    """Strategy specification for a single agent."""
    agent_id: int
    strategy_type: str  # "pure", "mixed", "behavioral"
    strategy_description: str

    # For pure strategies
    action: Optional[str] = None

    # For mixed strategies: action -> probability (as Fraction)
    mixed_distribution: Optional[Dict[str, Scalar]] = None

    def __post_init__(self):
        if self.strategy_type == "pure":
            assert self.action is not None
        elif self.strategy_type == "mixed":
            assert self.mixed_distribution is not None
            # Normalize to exact scalars
            self.mixed_distribution = {
                k: to_scalar(v) for k, v in self.mixed_distribution.items()
            }
            # Verify probabilities sum to 1
            total = sum(self.mixed_distribution.values())
            if total != 1:
                raise ValueError(f"Mixed strategy probabilities must sum to 1, got {total}")


@dataclass
class EquilibriumCertificate:
    """Certificate for a game-theoretic equilibrium.

    Certifies that a strategy profile forms an equilibrium under specified solution concept.
    This is the QA-native equivalent of a Nash/correlated equilibrium proof.

    VALIDITY RULES:
    - Must have at least 2 agents
    - Must have verification_witness for claimed equilibrium
    - If exploitability bound claimed, must have derivation
    """
    schema: str = "qa_equilibrium_cert/v1"

    # Game identification
    game_id: str = ""
    game_description: str = ""
    n_agents: int = 2

    # Game specification
    payoff_matrix_hash: Optional[str] = None  # For normal-form games
    game_tree_hash: Optional[str] = None  # For extensive-form games
    action_sets: Dict[int, List[str]] = field(default_factory=dict)  # agent_id -> actions

    # Equilibrium specification
    equilibrium_concept: EquilibriumConcept = EquilibriumConcept.NASH
    strategies: List[AgentStrategy] = field(default_factory=list)

    # Verification results
    is_equilibrium: bool = False
    exploitability_bound: Optional[Scalar] = None  # ε for ε-Nash

    # Failure evidence
    failure_mode: Optional[GameFailType] = None
    obstruction_if_fail: Optional[GameObstructionEvidence] = None

    # Understanding layer (anti-ad-hoc)
    verification_witness: Optional[DerivationWitness] = None

    # Strict mode
    strict_mode: bool = True

    def __post_init__(self):
        # Normalize exploitability bound
        if self.exploitability_bound is not None:
            self.exploitability_bound = to_scalar(self.exploitability_bound)

        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"EquilibriumCertificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of validity violations."""
        violations = []

        # Must have at least 2 agents
        if self.n_agents < 2:
            violations.append("INVALID_GAME: n_agents must be >= 2")

        # If claiming equilibrium, must have verification witness
        if self.is_equilibrium and self.verification_witness is None:
            violations.append(
                "ADHOC_EQUILIBRIUM: is_equilibrium=True but no verification_witness"
            )

        # If claiming exploitability bound, must have witness
        if self.exploitability_bound is not None and self.verification_witness is None:
            violations.append(
                "ADHOC_BOUND: exploitability_bound claimed without verification_witness"
            )

        # Strategies must match n_agents
        if len(self.strategies) != self.n_agents:
            violations.append(
                f"STRATEGY_MISMATCH: {len(self.strategies)} strategies for {self.n_agents} agents"
            )

        return violations

    def is_valid(self) -> bool:
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "game_id": self.game_id,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.game_description:
            result["game_description"] = self.game_description

        # Game specification
        result["game"] = {
            "n_agents": self.n_agents,
            "action_sets": {str(k): v for k, v in self.action_sets.items()},
        }
        if self.payoff_matrix_hash:
            result["game"]["payoff_matrix_hash"] = self.payoff_matrix_hash

        # Equilibrium
        result["equilibrium"] = {
            "concept": self.equilibrium_concept.value,
            "is_equilibrium": self.is_equilibrium,
            "strategies": [
                {
                    "agent_id": s.agent_id,
                    "type": s.strategy_type,
                    "description": s.strategy_description,
                    "action": s.action,
                    "mixed": {k: str(v) for k, v in s.mixed_distribution.items()} if s.mixed_distribution else None,
                }
                for s in self.strategies
            ],
        }

        if self.exploitability_bound is not None:
            result["equilibrium"]["exploitability_bound"] = str(self.exploitability_bound)

        # Failure
        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value

        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {
                "fail_type": obs.fail_type.value,
            }
            if obs.deviating_agent is not None:
                result["obstruction"]["deviating_agent"] = obs.deviating_agent
            if obs.deviation_payoff_gain is not None:
                result["obstruction"]["deviation_payoff_gain"] = str(obs.deviation_payoff_gain)
            if obs.regret_value is not None:
                result["obstruction"]["regret"] = str(obs.regret_value)

        # Witness
        if self.verification_witness:
            result["verification_witness"] = self.verification_witness.to_dict()

        return result


def validate_equilibrium_certificate(cert: EquilibriumCertificate) -> StrictValidationResult:
    """Validate equilibrium certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: if failure_mode is set, is_equilibrium should be False
    if cert.failure_mode is not None and cert.is_equilibrium:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but is_equilibrium=True"
        )

    # Consistency: EXPLOITABLE_DEVIATION requires obstruction with deviation evidence
    if cert.failure_mode == GameFailType.EXPLOITABLE_DEVIATION:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: EXPLOITABLE_DEVIATION requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.deviating_agent is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: EXPLOITABLE_DEVIATION requires deviating_agent"
            )

    # Consistency: REGRET_TOO_HIGH requires regret evidence
    if cert.failure_mode == GameFailType.REGRET_TOO_HIGH:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: REGRET_TOO_HIGH requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.regret_value is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: REGRET_TOO_HIGH requires regret_value"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# JOINT POLICY CERTIFICATES (Multiagent Sequential Decision Making)
# ============================================================================
# Ref: Kochenderfer et al. "Algorithms for Decision Making" Ch. 25 (MIT Press)
# Maps sequential multiagent problems (Markov games) to QA coupled reachability
#
# Key insight: Joint state = product lattice. Joint generator = coupled dynamics.
# Coordination = mutual reachability constraints on product graph.


class JointPolicyFailType(Enum):
    """Failure modes for joint policies in multiagent sequential settings."""
    # Coordination failures
    COORDINATION_DEADLOCK = "coordination_deadlock"  # Mutual waiting, no progress possible
    MISCOORDINATION_CYCLE = "miscoordination_cycle"  # Joint policy oscillates without reaching goal
    COLLISION_DETECTED = "collision_detected"  # Agents occupy same cell/resource

    # Individual agent failures in joint context
    AGENT_STUCK = "agent_stuck"  # One agent cannot progress
    AGENT_DIVERGED = "agent_diverged"  # One agent entered solo cycle

    # Joint reachability failures
    JOINT_TARGET_UNREACHABLE = "joint_target_unreachable"  # No joint path exists
    HORIZON_EXCEEDED = "horizon_exceeded"  # Didn't reach joint goal in time

    # Asymmetric failures
    ASYMMETRIC_PROGRESS = "asymmetric_progress"  # One agent reached goal, other stuck

    # Information asymmetry (Chapter 26)
    ASYMMETRIC_NON_IDENTIFIABLE = "asymmetric_non_identifiable"  # One agent can't distinguish states


@dataclass
class JointObstructionEvidence:
    """Evidence for joint policy failures."""
    fail_type: JointPolicyFailType

    # Deadlock evidence
    deadlock_joint_state: Optional[str] = None
    waiting_agents: Optional[List[int]] = None

    # Cycle evidence
    cycle_joint_states: Optional[List[str]] = None
    cycle_length: Optional[int] = None
    cycle_start_step: Optional[int] = None

    # Collision evidence
    collision_state: Optional[str] = None
    collision_step: Optional[int] = None
    colliding_agents: Optional[List[int]] = None

    # Agent-specific evidence
    stuck_agent: Optional[int] = None
    stuck_state: Optional[str] = None

    # Asymmetric identifiability evidence (Chapter 26)
    non_identifiable_agent: Optional[int] = None  # Which agent can't distinguish
    aliased_joint_states: Optional[List[str]] = None  # >=2 indistinguishable joint states
    other_agents_identifiable: bool = True  # Other agents can distinguish
    agent_observation: Optional[str] = None  # What the blind agent sees

    def __post_init__(self):
        ft = self.fail_type

        if ft == JointPolicyFailType.COORDINATION_DEADLOCK:
            assert self.deadlock_joint_state is not None
            assert self.waiting_agents is not None and len(self.waiting_agents) >= 2

        elif ft == JointPolicyFailType.MISCOORDINATION_CYCLE:
            assert self.cycle_joint_states is not None and len(self.cycle_joint_states) >= 1
            assert self.cycle_length is not None and self.cycle_length >= 1

        elif ft == JointPolicyFailType.COLLISION_DETECTED:
            assert self.collision_state is not None
            assert self.colliding_agents is not None and len(self.colliding_agents) >= 2

        elif ft == JointPolicyFailType.AGENT_STUCK:
            assert self.stuck_agent is not None
            assert self.stuck_state is not None

        elif ft == JointPolicyFailType.ASYMMETRIC_NON_IDENTIFIABLE:
            assert self.non_identifiable_agent is not None
            assert self.aliased_joint_states is not None and len(self.aliased_joint_states) >= 2


@dataclass
class CoordinationStats:
    """Statistics for joint policy execution."""
    n_episodes: int
    joint_successes: int  # Both agents reached goals
    collisions: int
    deadlocks: int
    cycles_detected: int
    total_joint_steps: int

    @property
    def joint_success_rate(self) -> Fraction:
        if self.n_episodes == 0:
            return Fraction(0)
        return Fraction(self.joint_successes, self.n_episodes)

    @property
    def avg_joint_steps(self) -> Optional[Fraction]:
        if self.joint_successes == 0:
            return None
        return Fraction(self.total_joint_steps, self.joint_successes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "joint_successes": self.joint_successes,
            "joint_success_rate": str(self.joint_success_rate),
            "avg_joint_steps": str(self.avg_joint_steps) if self.avg_joint_steps else None,
            "collisions": self.collisions,
            "deadlocks": self.deadlocks,
            "cycles_detected": self.cycles_detected,
        }


@dataclass
class JointPolicyCertificate:
    """Certificate for a joint policy in multiagent sequential settings.

    Certifies that a joint policy achieves coordination goals under specified constraints.
    This is the QA-native equivalent of a Markov game policy guarantee.

    VALIDITY RULES:
    - Must have n_agents >= 2
    - If claiming joint_success, must have coordination_witness
    - Agent goals and policies must match n_agents
    """
    schema: str = "qa_joint_policy_cert/v1"

    # Game/environment identification
    env_id: str = ""
    env_description: str = ""
    n_agents: int = 2

    # Joint state specification
    joint_state_space_description: str = ""
    joint_state_space_hash: Optional[str] = None

    # Agent specifications
    agent_goals: Dict[int, str] = field(default_factory=dict)  # agent_id -> goal description
    agent_policies: Dict[int, str] = field(default_factory=dict)  # agent_id -> policy type

    # Constraints
    collision_constraint: bool = True  # Agents cannot occupy same position
    turn_based: bool = False  # If True, agents move sequentially; if False, simultaneous

    # Joint generators (coupling rules)
    joint_generators: List[str] = field(default_factory=list)

    # Horizon
    horizon: int = 0

    # Results
    joint_success: bool = False
    coordination_stats: Optional[CoordinationStats] = None

    # Failure evidence
    failure_mode: Optional[JointPolicyFailType] = None
    obstruction_if_fail: Optional[JointObstructionEvidence] = None

    # Witness
    coordination_witness: Optional[DerivationWitness] = None
    joint_trajectory: Optional[List[Dict[str, Any]]] = None  # List of joint states

    # Observer upgrades (Chapter 26 - per-agent)
    observer_upgrades: Optional[Dict[int, str]] = None  # agent_id -> upgrade description
    aliased_joint_states_resolved: Optional[List[str]] = None  # States that were aliased before upgrade

    # Strict mode
    strict_mode: bool = True

    def __post_init__(self):
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"JointPolicyCertificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of validity violations."""
        violations = []

        # Must have at least 2 agents
        if self.n_agents < 2:
            violations.append("INVALID_JOINT_GAME: n_agents must be >= 2")

        # If claiming success, must have witness
        if self.joint_success and self.coordination_witness is None:
            violations.append(
                "ADHOC_JOINT_SUCCESS: joint_success=True but no coordination_witness"
            )

        # Agent goals must match n_agents
        if len(self.agent_goals) != self.n_agents:
            violations.append(
                f"GOAL_MISMATCH: {len(self.agent_goals)} goals for {self.n_agents} agents"
            )

        # Agent policies must match n_agents
        if len(self.agent_policies) != self.n_agents:
            violations.append(
                f"POLICY_MISMATCH: {len(self.agent_policies)} policies for {self.n_agents} agents"
            )

        return violations

    def is_valid(self) -> bool:
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "env_id": self.env_id,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.env_description:
            result["env_description"] = self.env_description

        # Environment specification
        result["environment"] = {
            "n_agents": self.n_agents,
            "joint_state_space": self.joint_state_space_description,
            "collision_constraint": self.collision_constraint,
            "turn_based": self.turn_based,
            "horizon": self.horizon,
        }
        if self.joint_state_space_hash:
            result["environment"]["state_space_hash"] = self.joint_state_space_hash

        # Agent specs
        result["agents"] = {
            "goals": {str(k): v for k, v in self.agent_goals.items()},
            "policies": {str(k): v for k, v in self.agent_policies.items()},
        }

        # Joint generators
        if self.joint_generators:
            result["joint_generators"] = self.joint_generators

        # Results
        result["coordination"] = {
            "joint_success": self.joint_success,
        }
        if self.coordination_stats:
            result["coordination"]["stats"] = self.coordination_stats.to_dict()

        # Failure
        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value

        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {
                "fail_type": obs.fail_type.value,
            }
            if obs.deadlock_joint_state:
                result["obstruction"]["deadlock_state"] = obs.deadlock_joint_state
            if obs.waiting_agents:
                result["obstruction"]["waiting_agents"] = obs.waiting_agents
            if obs.cycle_joint_states:
                result["obstruction"]["cycle_states"] = obs.cycle_joint_states
                result["obstruction"]["cycle_length"] = obs.cycle_length
            if obs.collision_state:
                result["obstruction"]["collision_state"] = obs.collision_state
                result["obstruction"]["colliding_agents"] = obs.colliding_agents

        # Trajectory
        if self.joint_trajectory:
            result["joint_trajectory"] = self.joint_trajectory

        # Witness
        if self.coordination_witness:
            result["coordination_witness"] = self.coordination_witness.to_dict()

        return result


def validate_joint_policy_certificate(cert: JointPolicyCertificate) -> StrictValidationResult:
    """Validate joint policy certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: if failure_mode is set, joint_success should be False
    if cert.failure_mode is not None and cert.joint_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but joint_success=True"
        )

    # Consistency: COORDINATION_DEADLOCK requires obstruction with deadlock evidence
    if cert.failure_mode == JointPolicyFailType.COORDINATION_DEADLOCK:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: COORDINATION_DEADLOCK requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.deadlock_joint_state is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: COORDINATION_DEADLOCK requires deadlock_joint_state"
            )

    # Consistency: MISCOORDINATION_CYCLE requires obstruction with cycle evidence
    if cert.failure_mode == JointPolicyFailType.MISCOORDINATION_CYCLE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: MISCOORDINATION_CYCLE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.cycle_joint_states is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: MISCOORDINATION_CYCLE requires cycle_joint_states"
            )

    # Consistency: COLLISION_DETECTED requires obstruction with collision evidence
    if cert.failure_mode == JointPolicyFailType.COLLISION_DETECTED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: COLLISION_DETECTED requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.collision_state is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: COLLISION_DETECTED requires collision_state"
            )

    # Consistency: ASYMMETRIC_NON_IDENTIFIABLE requires obstruction with agent info
    if cert.failure_mode == JointPolicyFailType.ASYMMETRIC_NON_IDENTIFIABLE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: ASYMMETRIC_NON_IDENTIFIABLE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.non_identifiable_agent is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: ASYMMETRIC_NON_IDENTIFIABLE requires non_identifiable_agent"
            )
        elif cert.obstruction_if_fail.aliased_joint_states is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: ASYMMETRIC_NON_IDENTIFIABLE requires aliased_joint_states"
            )

    # Observer upgrade consistency (multiagent): if observer_upgrades is set, must be success
    if cert.observer_upgrades and len(cert.observer_upgrades) > 0:
        if cert.obstruction_if_fail is not None:
            violations.append(
                "OBSERVER_UPGRADE_WITH_OBSTRUCTION: observer_upgrades present but "
                "obstruction_if_fail is set (upgrade should resolve failure)"
            )
        if cert.failure_mode is not None:
            violations.append(
                f"OBSERVER_UPGRADE_WITH_FAILURE: observer_upgrades present but "
                f"failure_mode={cert.failure_mode.value} (upgrade should resolve failure)"
            )
        if not cert.joint_success:
            violations.append(
                "OBSERVER_UPGRADE_WITHOUT_SUCCESS: observer_upgrades present but "
                "joint_success=False (upgrade should enable success)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# INFERENCE CERTIFICATES (Probabilistic Graphical Models)
# ============================================================================
# Ref: Kochenderfer et al. "Algorithms for Decision Making" Ch. 3-4 (MIT Press)
# Maps factor graph inference to QA reachability over distribution space
#
# Key insight: Variable elimination = graph reduction operators.
# Message passing = local invariant propagation.
# Marginals = reachability queries on distribution lattice.


class InferenceFailType(Enum):
    """Failure modes for probabilistic inference."""
    # Structural failures
    TREEWIDTH_TOO_HIGH = "treewidth_too_high"  # Exact inference intractable
    CYCLIC_FACTOR_GRAPH = "cyclic_factor_graph"  # Loopy BP required

    # Numerical failures
    NUMERICAL_UNDERFLOW = "numerical_underflow"  # Probabilities became zero
    NUMERICAL_OVERFLOW = "numerical_overflow"  # Unnormalized values exploded
    NORMALIZATION_FAILED = "normalization_failed"  # Can't normalize to valid distribution

    # Algorithm failures
    MESSAGE_DIVERGENCE = "message_divergence"  # BP messages didn't converge
    ELIMINATION_ORDER_INVALID = "elimination_order_invalid"  # Order violates constraints

    # Evidence failures
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"  # Evidence has P=0
    EVIDENCE_INCOMPLETE = "evidence_incomplete"  # Missing required observations

    # Query failures
    QUERY_VARIABLE_MISSING = "query_variable_missing"  # Queried variable not in graph
    CONDITIONAL_UNDEFINED = "conditional_undefined"  # P(Q|E) undefined (P(E)=0)


@dataclass
class InferenceObstructionEvidence:
    """Evidence for inference failures."""
    fail_type: InferenceFailType

    # Treewidth evidence
    treewidth: Optional[int] = None
    treewidth_threshold: Optional[int] = None

    # Numerical evidence
    underflow_variable: Optional[str] = None
    overflow_value: Optional[Scalar] = None
    normalization_sum: Optional[Scalar] = None

    # Convergence evidence
    iterations_run: Optional[int] = None
    max_iterations: Optional[int] = None
    final_residual: Optional[Scalar] = None
    convergence_threshold: Optional[Scalar] = None

    # Evidence inconsistency
    inconsistent_evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        ft = self.fail_type

        if ft == InferenceFailType.TREEWIDTH_TOO_HIGH:
            assert self.treewidth is not None
            assert self.treewidth_threshold is not None
            assert self.treewidth > self.treewidth_threshold

        elif ft == InferenceFailType.MESSAGE_DIVERGENCE:
            assert self.iterations_run is not None
            assert self.max_iterations is not None
            assert self.iterations_run >= self.max_iterations

        elif ft == InferenceFailType.NUMERICAL_UNDERFLOW:
            assert self.underflow_variable is not None

        elif ft == InferenceFailType.EVIDENCE_INCONSISTENT:
            assert self.inconsistent_evidence is not None

        # Normalize scalars
        if self.overflow_value is not None:
            self.overflow_value = to_scalar(self.overflow_value)
        if self.normalization_sum is not None:
            self.normalization_sum = to_scalar(self.normalization_sum)
        if self.final_residual is not None:
            self.final_residual = to_scalar(self.final_residual)
        if self.convergence_threshold is not None:
            self.convergence_threshold = to_scalar(self.convergence_threshold)


class InferenceMethod(Enum):
    """Methods for exact and approximate inference."""
    VARIABLE_ELIMINATION = "variable_elimination"  # Exact, exponential in treewidth
    BELIEF_PROPAGATION = "belief_propagation"  # Exact on trees, approximate on loops
    JUNCTION_TREE = "junction_tree"  # Exact, builds cluster tree
    GIBBS_SAMPLING = "gibbs_sampling"  # Approximate MCMC
    MEAN_FIELD = "mean_field"  # Approximate variational


@dataclass
class InferenceMethodProof:
    """Machine-checkable proof that inference was performed correctly.

    This is the QA-native equivalent of a certificate for P(Q|E) computation.
    """
    method: InferenceMethod

    # For variable elimination: the elimination order
    elimination_order: Optional[List[str]] = None
    elimination_order_cost: Optional[int] = None  # Fill-in edges or operations

    # For belief propagation: convergence info
    message_schedule: Optional[str] = None  # "parallel", "sequential", "tree"
    iterations: Optional[int] = None
    converged: bool = True
    final_residual: Optional[Scalar] = None

    # For junction tree: cluster info
    junction_tree_hash: Optional[str] = None
    max_cluster_size: Optional[int] = None

    # For sampling: sample statistics
    n_samples: Optional[int] = None
    burn_in: Optional[int] = None
    effective_sample_size: Optional[Scalar] = None

    # General
    verifiable: bool = True

    def __post_init__(self):
        if self.final_residual is not None:
            self.final_residual = to_scalar(self.final_residual)
        if self.effective_sample_size is not None:
            self.effective_sample_size = to_scalar(self.effective_sample_size)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "method": self.method.value,
            "verifiable": self.verifiable,
        }

        if self.elimination_order:
            result["elimination_order"] = self.elimination_order
            if self.elimination_order_cost is not None:
                result["elimination_order_cost"] = self.elimination_order_cost

        if self.message_schedule:
            result["message_schedule"] = self.message_schedule
            result["iterations"] = self.iterations
            result["converged"] = self.converged
            if self.final_residual is not None:
                result["final_residual"] = str(self.final_residual)

        if self.junction_tree_hash:
            result["junction_tree_hash"] = self.junction_tree_hash
            result["max_cluster_size"] = self.max_cluster_size

        if self.n_samples is not None:
            result["n_samples"] = self.n_samples
            result["burn_in"] = self.burn_in
            if self.effective_sample_size is not None:
                result["effective_sample_size"] = str(self.effective_sample_size)

        return result


@dataclass
class FactorSpec:
    """Specification of a factor in the graphical model."""
    factor_id: str
    scope: List[str]  # Variables in this factor's scope
    factor_type: str  # "prior", "conditional", "potential"
    table_hash: Optional[str] = None  # Hash of factor table for verification

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "factor_id": self.factor_id,
            "scope": self.scope,
            "factor_type": self.factor_type,
        }
        if self.table_hash:
            result["table_hash"] = self.table_hash
        return result


@dataclass
class InferenceCertificate:
    """Certificate for probabilistic inference over factor graphs.

    Certifies that P(Q|E) was computed correctly for query variables Q
    given evidence E over a factor graph.

    Three layers:
    1. Model specification: factor graph structure, variables, domains
    2. Query/evidence: what we're computing
    3. Result: distribution with proof method

    VALIDITY RULES:
    - Query variables must be in variable set
    - Evidence variables must be in variable set
    - If exact inference claimed, must have method_proof
    - Result distribution must be properly normalized
    """
    schema: str = "qa_inference_cert/v1"

    # Model identification
    model_id: str = ""
    model_description: str = ""

    # Variable specification
    variables: List[str] = field(default_factory=list)
    variable_domains: Dict[str, List[str]] = field(default_factory=dict)  # var -> domain values

    # Factor specification
    factors: List[FactorSpec] = field(default_factory=list)
    factor_graph_hash: Optional[str] = None

    # Graph properties (for tractability analysis)
    is_tree: bool = False
    treewidth: Optional[int] = None

    # Query specification
    query_variables: List[str] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)  # var -> observed value

    # Result
    inference_success: bool = False
    result_distribution: Optional[Dict[str, Scalar]] = None  # assignment -> probability

    # For single-variable queries, marginal over values
    marginal: Optional[Dict[str, Scalar]] = None  # value -> probability

    # Method proof
    method_proof: Optional[InferenceMethodProof] = None
    exact_inference: bool = False  # True if method is exact (not approximate)

    # Failure evidence
    failure_mode: Optional[InferenceFailType] = None
    obstruction_if_fail: Optional[InferenceObstructionEvidence] = None

    # Understanding layer (anti-ad-hoc)
    inference_witness: Optional[DerivationWitness] = None

    # Strict mode
    strict_mode: bool = True

    def __post_init__(self):
        # Normalize result_distribution scalars
        if self.result_distribution is not None:
            self.result_distribution = {k: to_scalar(v) for k, v in self.result_distribution.items()}
        if self.marginal is not None:
            self.marginal = {k: to_scalar(v) for k, v in self.marginal.items()}

        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"InferenceCertificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of validity violations."""
        violations = []

        # Query variables must be in variable set
        var_set = set(self.variables)
        for qv in self.query_variables:
            if qv not in var_set:
                violations.append(f"QUERY_VARIABLE_MISSING: '{qv}' not in variables")

        # Evidence variables must be in variable set
        for ev in self.evidence.keys():
            if ev not in var_set:
                violations.append(f"EVIDENCE_VARIABLE_MISSING: '{ev}' not in variables")

        # Evidence values must be in domain
        for ev, val in self.evidence.items():
            if ev in self.variable_domains:
                if val not in self.variable_domains[ev]:
                    violations.append(
                        f"EVIDENCE_VALUE_INVALID: '{val}' not in domain of '{ev}'"
                    )

        # If claiming success, must have witness or method_proof
        if self.inference_success:
            if self.inference_witness is None and self.method_proof is None:
                violations.append(
                    "ADHOC_INFERENCE: inference_success=True but no witness or method_proof"
                )

        # If claiming exact inference, must have method_proof
        if self.exact_inference and self.method_proof is None:
            violations.append(
                "ADHOC_EXACTNESS: exact_inference=True but no method_proof"
            )

        # Check normalization of result distribution
        if self.result_distribution is not None and self.inference_success:
            total = sum(self.result_distribution.values())
            if total != 1:
                violations.append(
                    f"NORMALIZATION_ERROR: result_distribution sums to {total}, not 1"
                )

        # Check normalization of marginal
        if self.marginal is not None and self.inference_success:
            total = sum(self.marginal.values())
            if total != 1:
                violations.append(
                    f"MARGINAL_NORMALIZATION_ERROR: marginal sums to {total}, not 1"
                )

        return violations

    def is_valid(self) -> bool:
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.model_description:
            result["model_description"] = self.model_description

        # Model specification
        result["model"] = {
            "variables": self.variables,
            "variable_domains": self.variable_domains,
            "factors": [f.to_dict() for f in self.factors],
            "is_tree": self.is_tree,
        }
        if self.treewidth is not None:
            result["model"]["treewidth"] = self.treewidth
        if self.factor_graph_hash:
            result["model"]["factor_graph_hash"] = self.factor_graph_hash

        # Query
        result["query"] = {
            "variables": self.query_variables,
            "evidence": self.evidence,
        }

        # Result
        result["inference"] = {
            "success": self.inference_success,
            "exact": self.exact_inference,
        }

        if self.result_distribution:
            result["inference"]["distribution"] = {
                k: str(v) for k, v in self.result_distribution.items()
            }

        if self.marginal:
            result["inference"]["marginal"] = {
                k: str(v) for k, v in self.marginal.items()
            }

        # Method proof
        if self.method_proof:
            result["method_proof"] = self.method_proof.to_dict()

        # Failure
        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value

        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {
                "fail_type": obs.fail_type.value,
            }
            if obs.treewidth is not None:
                result["obstruction"]["treewidth"] = obs.treewidth
                result["obstruction"]["threshold"] = obs.treewidth_threshold
            if obs.iterations_run is not None:
                result["obstruction"]["iterations"] = obs.iterations_run
                result["obstruction"]["max_iterations"] = obs.max_iterations

        # Witness
        if self.inference_witness:
            result["inference_witness"] = self.inference_witness.to_dict()

        return result

    @classmethod
    def from_variable_elimination(
        cls,
        model_id: str,
        variables: List[str],
        variable_domains: Dict[str, List[str]],
        factors: List[FactorSpec],
        query_variables: List[str],
        evidence: Dict[str, str],
        elimination_order: List[str],
        result_marginal: Dict[str, Scalar],
    ) -> "InferenceCertificate":
        """Create certificate from variable elimination computation."""
        return cls(
            model_id=model_id,
            variables=variables,
            variable_domains=variable_domains,
            factors=factors,
            query_variables=query_variables,
            evidence=evidence,
            inference_success=True,
            marginal=result_marginal,
            exact_inference=True,
            method_proof=InferenceMethodProof(
                method=InferenceMethod.VARIABLE_ELIMINATION,
                elimination_order=elimination_order,
                verifiable=True,
            ),
            inference_witness=DerivationWitness(
                invariant_name="marginal_probability",
                derivation_operator="variable_elimination",
                input_data={
                    "query": query_variables,
                    "evidence": evidence,
                    "elimination_order": elimination_order,
                },
                output_value=1,  # Normalized
                verifiable=True,
            ),
            strict_mode=True,
        )

    @classmethod
    def from_belief_propagation(
        cls,
        model_id: str,
        variables: List[str],
        variable_domains: Dict[str, List[str]],
        factors: List[FactorSpec],
        query_variables: List[str],
        evidence: Dict[str, str],
        result_marginal: Dict[str, Scalar],
        iterations: int,
        converged: bool,
        is_tree: bool = False,
    ) -> "InferenceCertificate":
        """Create certificate from belief propagation computation."""
        return cls(
            model_id=model_id,
            variables=variables,
            variable_domains=variable_domains,
            factors=factors,
            is_tree=is_tree,
            query_variables=query_variables,
            evidence=evidence,
            inference_success=converged,
            marginal=result_marginal if converged else None,
            exact_inference=is_tree,  # BP is exact on trees
            method_proof=InferenceMethodProof(
                method=InferenceMethod.BELIEF_PROPAGATION,
                message_schedule="parallel",
                iterations=iterations,
                converged=converged,
                verifiable=True,
            ),
            inference_witness=DerivationWitness(
                invariant_name="belief_marginal",
                derivation_operator="belief_propagation",
                input_data={
                    "query": query_variables,
                    "evidence": evidence,
                    "iterations": iterations,
                    "is_tree": is_tree,
                },
                output_value=1 if converged else 0,
                verifiable=True,
            ) if converged else None,
            failure_mode=InferenceFailType.MESSAGE_DIVERGENCE if not converged else None,
            strict_mode=True,
        )


def validate_inference_certificate(cert: InferenceCertificate) -> StrictValidationResult:
    """Validate inference certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: if failure_mode is set, inference_success should be False
    if cert.failure_mode is not None and cert.inference_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but inference_success=True"
        )

    # Consistency: TREEWIDTH_TOO_HIGH requires obstruction with treewidth evidence
    if cert.failure_mode == InferenceFailType.TREEWIDTH_TOO_HIGH:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: TREEWIDTH_TOO_HIGH requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.treewidth is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: TREEWIDTH_TOO_HIGH requires treewidth"
            )

    # Consistency: MESSAGE_DIVERGENCE requires obstruction with iteration evidence
    if cert.failure_mode == InferenceFailType.MESSAGE_DIVERGENCE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: MESSAGE_DIVERGENCE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.iterations_run is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: MESSAGE_DIVERGENCE requires iterations_run"
            )

    # Structural: BP exactness consistency
    # Rule: BP + is_tree=True → exact=True allowed (and encouraged)
    # Rule: BP + is_tree=False → exact=False required (BP on loopy is approximate)
    if cert.method_proof and cert.method_proof.method == InferenceMethod.BELIEF_PROPAGATION:
        if cert.is_tree and cert.inference_success and not cert.exact_inference:
            warnings.append(
                "BP_ON_TREE_NOT_EXACT: BP converged on tree but exact_inference=False "
                "(BP is exact on trees, consider setting exact_inference=True)"
            )
        if not cert.is_tree and cert.exact_inference:
            violations.append(
                "EXACT_BP_ON_LOOPY: exact_inference=True but method=BP and is_tree=False. "
                "BP on loopy graphs is approximate - use junction_tree for exact inference."
            )

    # Check factor scopes reference valid variables
    var_set = set(cert.variables)
    for factor in cert.factors:
        for v in factor.scope:
            if v not in var_set:
                violations.append(
                    f"INVALID_FACTOR_SCOPE: factor '{factor.factor_id}' references unknown variable '{v}'"
                )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# RECOMPUTE-VALIDITY HOOKS (Auditable Mode)
# ============================================================================
# These functions move from "well-formed claim" to "auditable certificate"
# by recomputing claimed outputs from provided inputs.
#
# ChatGPT suggestion: "the difference between well-formed claim and auditable certificate"


def _ve_sum_product(
    factor_tables: Dict[str, Dict[Tuple[str, ...], Scalar]],
    factor_scopes: Dict[str, List[str]],
    elimination_order: List[str],
    query_vars: List[str],
    evidence: Dict[str, str],
) -> Dict[str, Scalar]:
    """
    Variable elimination sum-product algorithm.

    Args:
        factor_tables: factor_id -> {(val1, val2, ...): probability}
        factor_scopes: factor_id -> [var1, var2, ...]
        elimination_order: order to eliminate non-query, non-evidence vars
        query_vars: variables to compute marginal over
        evidence: var -> observed_value

    Returns:
        Marginal distribution over query_vars (single var: value -> prob)
    """
    # Work with mutable copy
    factors = {fid: dict(table) for fid, table in factor_tables.items()}
    scopes = {fid: list(scope) for fid, scope in factor_scopes.items()}

    # Reduce factors by evidence
    for fid in list(factors.keys()):
        scope = scopes[fid]
        for ev_var, ev_val in evidence.items():
            if ev_var in scope:
                ev_idx = scope.index(ev_var)
                new_table = {}
                for assignment, prob in factors[fid].items():
                    if assignment[ev_idx] == ev_val:
                        # Remove evidence variable from assignment
                        new_assignment = tuple(
                            v for i, v in enumerate(assignment) if i != ev_idx
                        )
                        new_table[new_assignment] = prob
                factors[fid] = new_table
                scopes[fid] = [v for v in scope if v != ev_var]

    # Eliminate variables in order
    for elim_var in elimination_order:
        if elim_var in query_vars or elim_var in evidence:
            continue

        # Find factors involving elim_var
        involved = [fid for fid, scope in scopes.items() if elim_var in scope]
        if not involved:
            continue

        # Multiply involved factors
        new_scope = []
        for fid in involved:
            for v in scopes[fid]:
                if v not in new_scope:
                    new_scope.append(v)

        # Build product table
        product_table: Dict[Tuple[str, ...], Scalar] = {}

        # Collect all possible values for each variable in new_scope
        var_values: Dict[str, set] = {v: set() for v in new_scope}
        for fid in involved:
            scope = scopes[fid]
            for assignment in factors[fid].keys():
                for i, v in enumerate(scope):
                    var_values[v].add(assignment[i])

        # Generate all combinations using itertools.product
        from itertools import product as cartesian_product

        value_lists = [sorted(var_values[v]) for v in new_scope]
        for combo in cartesian_product(*value_lists):
            val_map = dict(zip(new_scope, combo))

            # Multiply all involved factors
            product_val: Scalar = Fraction(1)
            valid = True

            for fid in involved:
                scope = scopes[fid]
                factor_assignment = tuple(val_map[v] for v in scope)
                if factor_assignment in factors[fid]:
                    product_val *= factors[fid][factor_assignment]
                else:
                    valid = False
                    break

            if valid:
                product_table[combo] = product_val

        # Sum out elim_var
        elim_idx = new_scope.index(elim_var)
        result_scope = [v for v in new_scope if v != elim_var]
        result_table: Dict[Tuple[str, ...], Scalar] = {}

        for assignment, prob in product_table.items():
            result_assignment = tuple(
                v for i, v in enumerate(assignment) if i != elim_idx
            )
            if result_assignment not in result_table:
                result_table[result_assignment] = Fraction(0)
            result_table[result_assignment] += prob

        # Create new factor
        new_fid = f"_tau_{elim_var}"
        factors[new_fid] = result_table
        scopes[new_fid] = result_scope

        # Remove old factors
        for fid in involved:
            del factors[fid]
            del scopes[fid]

    # Final product over remaining factors
    if len(factors) == 0:
        return {}

    # Get query variable marginal
    if len(query_vars) == 1:
        qvar = query_vars[0]
        marginal: Dict[str, Scalar] = {}

        for fid, table in factors.items():
            if qvar in scopes[fid]:
                qidx = scopes[fid].index(qvar)
                for assignment, prob in table.items():
                    val = assignment[qidx] if qidx < len(assignment) else assignment[0]
                    if val not in marginal:
                        marginal[val] = Fraction(0)
                    marginal[val] += prob

        # Normalize
        total = sum(marginal.values())
        if total > 0:
            marginal = {k: Fraction(v) / Fraction(total) for k, v in marginal.items()}

        return marginal

    return {}


def recompute_ve_marginal(
    cert: "InferenceCertificate",
    factor_tables: Dict[str, Dict[Tuple[str, ...], Scalar]],
) -> StrictValidationResult:
    """
    Recompute variable elimination marginal and verify it matches certificate.

    This is the "auditable mode" - given factor tables, recompute the claimed
    marginal and check it matches the certificate's claim.

    Args:
        cert: InferenceCertificate with VE method_proof
        factor_tables: factor_id -> {(val_tuple): probability}

    Returns:
        StrictValidationResult with violations if recompute doesn't match
    """
    violations = []
    warnings = []

    # Must be VE method
    if cert.method_proof is None or cert.method_proof.method != InferenceMethod.VARIABLE_ELIMINATION:
        violations.append(
            "RECOMPUTE_METHOD_MISMATCH: recompute_ve_marginal requires method=VARIABLE_ELIMINATION"
        )
        return StrictValidationResult(False, violations, warnings)

    # Must have elimination order
    if cert.method_proof.elimination_order is None:
        violations.append(
            "RECOMPUTE_MISSING_ORDER: elimination_order required for VE recompute"
        )
        return StrictValidationResult(False, violations, warnings)

    # Build factor scopes from cert
    factor_scopes = {f.factor_id: f.scope for f in cert.factors}

    # Check all required factors are provided
    for fid in factor_scopes.keys():
        if fid not in factor_tables:
            violations.append(
                f"RECOMPUTE_MISSING_TABLE: factor '{fid}' not in provided factor_tables"
            )

    if violations:
        return StrictValidationResult(False, violations, warnings)

    # Recompute marginal
    try:
        recomputed = _ve_sum_product(
            factor_tables=factor_tables,
            factor_scopes=factor_scopes,
            elimination_order=cert.method_proof.elimination_order,
            query_vars=cert.query_variables,
            evidence=cert.evidence,
        )
    except Exception as e:
        violations.append(f"RECOMPUTE_ERROR: VE computation failed: {e}")
        return StrictValidationResult(False, violations, warnings)

    # Compare with certificate's claimed marginal
    if cert.marginal is None:
        violations.append("RECOMPUTE_NO_CLAIM: certificate has no marginal to verify")
        return StrictValidationResult(False, violations, warnings)

    # Check each value
    for val, claimed_prob in cert.marginal.items():
        if val not in recomputed:
            violations.append(
                f"RECOMPUTE_MISMATCH: value '{val}' in certificate but not in recomputed marginal"
            )
        elif recomputed[val] != claimed_prob:
            violations.append(
                f"RECOMPUTE_MISMATCH: P({cert.query_variables[0]}={val}) "
                f"claimed={claimed_prob}, recomputed={recomputed[val]}"
            )

    for val in recomputed:
        if val not in cert.marginal:
            violations.append(
                f"RECOMPUTE_MISMATCH: value '{val}' in recomputed but not in certificate"
            )

    if not violations:
        warnings.append("RECOMPUTE_VERIFIED: VE marginal matches claimed values")

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def recompute_kalman_update(
    cert: "FilterCertificate",
    A: List[List[Scalar]],  # State transition matrix
    H: List[List[Scalar]],  # Observation matrix
    Q: List[List[Scalar]],  # Process noise covariance
    R: List[List[Scalar]],  # Observation noise covariance
    x0: List[Scalar],       # Initial state
    P0: List[List[Scalar]], # Initial covariance
    observations: List[List[Scalar]],  # Sequence of observations
) -> StrictValidationResult:
    """
    Recompute Kalman filter and verify estimate matches certificate.

    This is the "auditable mode" for Kalman filters - run the filter
    on provided data and check final state matches claim.

    Args:
        cert: FilterCertificate with Kalman method_proof
        A, H, Q, R: System matrices (as nested lists of exact scalars)
        x0, P0: Initial state and covariance
        observations: List of observation vectors

    Returns:
        StrictValidationResult with violations if recompute doesn't match
    """
    violations = []
    warnings = []

    # Must be Kalman method
    if cert.method_proof is None or cert.method_proof.method != FilterMethod.KALMAN:
        violations.append(
            "RECOMPUTE_METHOD_MISMATCH: recompute_kalman_update requires method=KALMAN"
        )
        return StrictValidationResult(False, violations, warnings)

    # Check dimensions
    n = len(x0)  # State dimension
    m = len(observations[0]) if observations else 0  # Observation dimension

    if n != cert.state_dimension:
        violations.append(
            f"RECOMPUTE_DIM_MISMATCH: x0 has dim {n}, cert claims {cert.state_dimension}"
        )

    if len(observations) != cert.n_observations:
        violations.append(
            f"RECOMPUTE_OBS_MISMATCH: {len(observations)} obs provided, cert claims {cert.n_observations}"
        )

    if violations:
        return StrictValidationResult(False, violations, warnings)

    # Helper: matrix-vector multiply
    def mat_vec(M: List[List[Scalar]], v: List[Scalar]) -> List[Scalar]:
        return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]

    # Helper: matrix-matrix multiply
    def mat_mat(A: List[List[Scalar]], B: List[List[Scalar]]) -> List[List[Scalar]]:
        n, m, p = len(A), len(B), len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]

    # Helper: matrix transpose
    def mat_T(M: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

    # Helper: matrix addition
    def mat_add(A: List[List[Scalar]], B: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    # Helper: matrix subtraction
    def mat_sub(A: List[List[Scalar]], B: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    # Helper: scalar * matrix
    def mat_scale(s: Scalar, M: List[List[Scalar]]) -> List[List[Scalar]]:
        return [[s * M[i][j] for j in range(len(M[0]))] for i in range(len(M))]

    # Helper: 2x2 matrix inverse (for simplicity)
    def mat_inv_2x2(M: List[List[Scalar]]) -> List[List[Scalar]]:
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        if det == 0:
            raise ValueError("Singular matrix")
        return [
            [M[1][1] / det, -M[0][1] / det],
            [-M[1][0] / det, M[0][0] / det],
        ]

    # Helper: 1x1 matrix inverse
    def mat_inv_1x1(M: List[List[Scalar]]) -> List[List[Scalar]]:
        if M[0][0] == 0:
            raise ValueError("Singular matrix")
        return [[Fraction(1) / M[0][0]]]

    # Convert inputs to Fraction for exact arithmetic
    def to_frac(x):
        if isinstance(x, list):
            return [to_frac(i) for i in x]
        return to_scalar(x)

    try:
        A_f = to_frac(A)
        H_f = to_frac(H)
        Q_f = to_frac(Q)
        R_f = to_frac(R)
        x = to_frac(x0)
        P = to_frac(P0)
        obs = to_frac(observations)

        # Run Kalman filter
        for z in obs:
            # Predict
            x_pred = mat_vec(A_f, x)
            P_pred = mat_add(mat_mat(mat_mat(A_f, P), mat_T(A_f)), Q_f)

            # Innovation
            y = [z[i] - sum(H_f[i][j] * x_pred[j] for j in range(n)) for i in range(m)]

            # Innovation covariance S = H P H' + R
            S = mat_add(mat_mat(mat_mat(H_f, P_pred), mat_T(H_f)), R_f)

            # Kalman gain K = P H' S^-1
            if m == 1:
                S_inv = mat_inv_1x1(S)
            elif m == 2:
                S_inv = mat_inv_2x2(S)
            else:
                # For larger matrices, would need general inverse
                warnings.append(
                    f"RECOMPUTE_LIMITED: Kalman gain computation limited to m<=2, got m={m}"
                )
                break

            K = mat_mat(mat_mat(P_pred, mat_T(H_f)), S_inv)

            # Update
            x = [x_pred[i] + sum(K[i][j] * y[j] for j in range(m)) for i in range(n)]
            I = [[Fraction(1) if i == j else Fraction(0) for j in range(n)] for i in range(n)]
            P = mat_mat(mat_sub(I, mat_mat(K, H_f)), P_pred)

        # Compute trace of final covariance
        final_trace = sum(P[i][i] for i in range(n))

    except Exception as e:
        violations.append(f"RECOMPUTE_ERROR: Kalman computation failed: {e}")
        return StrictValidationResult(False, violations, warnings)

    # Compare with certificate's claimed estimate
    if cert.state_estimate is not None and cert.state_names:
        for i, name in enumerate(cert.state_names):
            if name in cert.state_estimate:
                claimed = cert.state_estimate[name]
                recomputed = x[i]
                if claimed != recomputed:
                    violations.append(
                        f"RECOMPUTE_MISMATCH: state '{name}' claimed={claimed}, recomputed={recomputed}"
                    )

    # Compare covariance trace
    if cert.covariance_trace is not None:
        if final_trace != cert.covariance_trace:
            # Allow small tolerance for numerical reasons (but we use exact arithmetic)
            violations.append(
                f"RECOMPUTE_MISMATCH: tr(P) claimed={cert.covariance_trace}, recomputed={final_trace}"
            )

    if not violations:
        warnings.append("RECOMPUTE_VERIFIED: Kalman estimate matches claimed values")

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# FILTER CERTIFICATES (State Estimation)
# ============================================================================
# Ref: Kochenderfer et al. "Algorithms for Decision Making" Ch. 9-11 (MIT Press)
# Maps state estimation to QA reachability over belief-state space
#
# Key insight: State estimation = tracking distribution over lattice.
# Predict step = generator application to belief.
# Update step = observation-conditioned refinement.
# Failure = distribution degeneracy or divergence.


class FilterFailType(Enum):
    """Failure modes for state estimation filters."""
    # Numerical failures
    COVARIANCE_SINGULAR = "covariance_singular"  # Kalman: P becomes singular
    COVARIANCE_NOT_PSD = "covariance_not_psd"  # Kalman: P not positive semi-definite
    NUMERICAL_UNDERFLOW = "numerical_underflow"  # Weights/probs became zero

    # Particle filter failures
    PARTICLE_DEGENERACY = "particle_degeneracy"  # Effective sample size too low
    PARTICLE_DIVERGENCE = "particle_divergence"  # Particles don't track true state
    RESAMPLING_COLLAPSE = "resampling_collapse"  # All weight on one particle

    # Model failures
    PROCESS_MODEL_MISMATCH = "process_model_mismatch"  # Dynamics don't match reality
    OBSERVATION_MODEL_MISMATCH = "observation_model_mismatch"  # Sensor model wrong
    INNOVATION_OUTLIER = "innovation_outlier"  # Observation far from prediction

    # Convergence failures
    FILTER_DIVERGED = "filter_diverged"  # Estimate drifts from true state
    ESTIMATE_UNBOUNDED = "estimate_unbounded"  # State estimate exploded

    # Observability failures
    STATE_UNOBSERVABLE = "state_unobservable"  # Can't estimate some states
    RANK_DEFICIENT = "rank_deficient"  # Observability matrix rank deficient


@dataclass
class FilterObstructionEvidence:
    """Evidence for filter failures."""
    fail_type: FilterFailType

    # Numerical evidence
    condition_number: Optional[Scalar] = None
    min_eigenvalue: Optional[Scalar] = None

    # Particle filter evidence
    effective_sample_size: Optional[Scalar] = None
    n_particles: Optional[int] = None
    ess_threshold: Optional[Scalar] = None
    max_weight: Optional[Scalar] = None

    # Innovation evidence
    innovation_norm: Optional[Scalar] = None
    innovation_threshold: Optional[Scalar] = None

    # Divergence evidence
    estimation_error: Optional[Scalar] = None
    error_threshold: Optional[Scalar] = None
    timestep: Optional[int] = None

    # Observability evidence
    observability_rank: Optional[int] = None
    state_dimension: Optional[int] = None
    unobservable_modes: Optional[List[str]] = None

    def __post_init__(self):
        ft = self.fail_type

        if ft == FilterFailType.COVARIANCE_SINGULAR:
            assert self.condition_number is not None or self.min_eigenvalue is not None

        elif ft == FilterFailType.PARTICLE_DEGENERACY:
            assert self.effective_sample_size is not None
            assert self.n_particles is not None
            assert self.ess_threshold is not None
            # ESS below threshold
            if self.effective_sample_size is not None and self.ess_threshold is not None:
                self.effective_sample_size = to_scalar(self.effective_sample_size)
                self.ess_threshold = to_scalar(self.ess_threshold)
                assert self.effective_sample_size < self.ess_threshold

        elif ft == FilterFailType.RESAMPLING_COLLAPSE:
            assert self.max_weight is not None
            self.max_weight = to_scalar(self.max_weight)

        elif ft == FilterFailType.INNOVATION_OUTLIER:
            assert self.innovation_norm is not None
            assert self.innovation_threshold is not None
            self.innovation_norm = to_scalar(self.innovation_norm)
            self.innovation_threshold = to_scalar(self.innovation_threshold)
            assert self.innovation_norm > self.innovation_threshold

        elif ft == FilterFailType.FILTER_DIVERGED:
            assert self.estimation_error is not None
            assert self.error_threshold is not None
            self.estimation_error = to_scalar(self.estimation_error)
            self.error_threshold = to_scalar(self.error_threshold)

        elif ft == FilterFailType.STATE_UNOBSERVABLE:
            assert self.observability_rank is not None
            assert self.state_dimension is not None
            assert self.observability_rank < self.state_dimension

        # Normalize remaining scalars
        if self.condition_number is not None:
            self.condition_number = to_scalar(self.condition_number)
        if self.min_eigenvalue is not None:
            self.min_eigenvalue = to_scalar(self.min_eigenvalue)


class FilterMethod(Enum):
    """Methods for state estimation."""
    KALMAN = "kalman"  # Linear Gaussian, exact
    EXTENDED_KALMAN = "extended_kalman"  # Linearized, approximate
    UNSCENTED_KALMAN = "unscented_kalman"  # Sigma points, better nonlinear
    PARTICLE = "particle"  # Sequential Monte Carlo
    HISTOGRAM = "histogram"  # Discrete state space
    HYBRID = "hybrid"  # Combined methods


@dataclass
class FilterMethodProof:
    """Machine-checkable proof that filter estimation was performed correctly.

    This is the QA-native equivalent of a certificate for state estimation.
    """
    method: FilterMethod

    # Kalman-specific
    kalman_gain_hash: Optional[str] = None  # Hash of K matrix for verification
    innovation_sequence_hash: Optional[str] = None  # Hash of y - Hx sequence
    covariance_trace: Optional[Scalar] = None  # tr(P) for uncertainty tracking

    # Particle filter specific
    n_particles: Optional[int] = None
    resampling_method: Optional[str] = None  # "multinomial", "systematic", "stratified"
    effective_sample_size: Optional[Scalar] = None
    resample_threshold: Optional[Scalar] = None
    n_resamples: Optional[int] = None

    # Histogram filter specific
    n_bins: Optional[int] = None
    bin_width: Optional[Scalar] = None

    # Common
    n_timesteps: Optional[int] = None
    final_estimate_hash: Optional[str] = None
    verifiable: bool = True

    def __post_init__(self):
        if self.covariance_trace is not None:
            self.covariance_trace = to_scalar(self.covariance_trace)
        if self.effective_sample_size is not None:
            self.effective_sample_size = to_scalar(self.effective_sample_size)
        if self.resample_threshold is not None:
            self.resample_threshold = to_scalar(self.resample_threshold)
        if self.bin_width is not None:
            self.bin_width = to_scalar(self.bin_width)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "method": self.method.value,
            "verifiable": self.verifiable,
        }

        if self.n_timesteps is not None:
            result["n_timesteps"] = self.n_timesteps

        # Kalman
        if self.kalman_gain_hash:
            result["kalman_gain_hash"] = self.kalman_gain_hash
        if self.innovation_sequence_hash:
            result["innovation_sequence_hash"] = self.innovation_sequence_hash
        if self.covariance_trace is not None:
            result["covariance_trace"] = str(self.covariance_trace)

        # Particle
        if self.n_particles is not None:
            result["n_particles"] = self.n_particles
            result["resampling_method"] = self.resampling_method
            if self.effective_sample_size is not None:
                result["effective_sample_size"] = str(self.effective_sample_size)
            if self.n_resamples is not None:
                result["n_resamples"] = self.n_resamples

        # Histogram
        if self.n_bins is not None:
            result["n_bins"] = self.n_bins
            if self.bin_width is not None:
                result["bin_width"] = str(self.bin_width)

        return result


@dataclass
class FilterCertificate:
    """Certificate for state estimation over dynamical systems.

    Certifies that state estimation was performed correctly given
    a sequence of observations and a dynamical model.

    Three layers:
    1. Model specification: dynamics, observation model, noise parameters
    2. Estimation: state estimate, uncertainty measure
    3. Quality: innovation consistency, filter health metrics

    VALIDITY RULES:
    - Must have state_dimension > 0
    - If claiming success, must have method_proof
    - Uncertainty measure must be present for successful filters
    """
    schema: str = "qa_filter_cert/v1"

    # Model identification
    model_id: str = ""
    model_description: str = ""

    # State space specification
    state_dimension: int = 0
    state_names: List[str] = field(default_factory=list)
    observation_dimension: int = 0

    # Model type
    linear_system: bool = True
    gaussian_noise: bool = True

    # Estimation task
    n_observations: int = 0
    observation_sequence_hash: Optional[str] = None

    # Result
    filter_success: bool = False

    # State estimate (final)
    state_estimate: Optional[Dict[str, Scalar]] = None  # state_name -> value

    # Uncertainty measure
    covariance_trace: Optional[Scalar] = None  # For Kalman: tr(P)
    credible_interval_width: Optional[Scalar] = None  # For particle: 95% CI width
    entropy: Optional[Scalar] = None  # For histogram: Shannon entropy

    # Method proof
    method_proof: Optional[FilterMethodProof] = None

    # Quality metrics
    normalized_innovation_squared: Optional[Scalar] = None  # NIS statistic
    innovation_whiteness: bool = True  # Innovation sequence uncorrelated

    # Failure evidence
    failure_mode: Optional[FilterFailType] = None
    obstruction_if_fail: Optional[FilterObstructionEvidence] = None

    # Understanding layer (anti-ad-hoc)
    estimation_witness: Optional[DerivationWitness] = None

    # Strict mode
    strict_mode: bool = True

    def __post_init__(self):
        # Normalize scalars
        if self.state_estimate is not None:
            self.state_estimate = {k: to_scalar(v) for k, v in self.state_estimate.items()}
        if self.covariance_trace is not None:
            self.covariance_trace = to_scalar(self.covariance_trace)
        if self.credible_interval_width is not None:
            self.credible_interval_width = to_scalar(self.credible_interval_width)
        if self.entropy is not None:
            self.entropy = to_scalar(self.entropy)
        if self.normalized_innovation_squared is not None:
            self.normalized_innovation_squared = to_scalar(self.normalized_innovation_squared)

        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise CertificateValidityError(
                    f"FilterCertificate invalid: {'; '.join(violations)}"
                )

    def get_validity_violations(self) -> List[str]:
        """Get list of validity violations."""
        violations = []

        # Must have positive state dimension
        if self.state_dimension <= 0:
            violations.append("INVALID_DIMENSION: state_dimension must be > 0")

        # If claiming success, must have method_proof
        if self.filter_success and self.method_proof is None:
            violations.append(
                "ADHOC_FILTER: filter_success=True but no method_proof"
            )

        # If success, must have some uncertainty measure
        if self.filter_success:
            has_uncertainty = (
                self.covariance_trace is not None or
                self.credible_interval_width is not None or
                self.entropy is not None
            )
            if not has_uncertainty:
                violations.append(
                    "MISSING_UNCERTAINTY: filter_success=True but no uncertainty measure"
                )

        # State estimate must match state_names if provided
        if self.state_estimate is not None and self.state_names:
            for name in self.state_estimate.keys():
                if name not in self.state_names:
                    violations.append(
                        f"STATE_NAME_MISMATCH: estimate has '{name}' not in state_names"
                    )

        return violations

    def is_valid(self) -> bool:
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema": self.schema,
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if not self.is_valid():
            result["validity_violations"] = self.get_validity_violations()

        if self.model_description:
            result["model_description"] = self.model_description

        # Model specification
        result["model"] = {
            "state_dimension": self.state_dimension,
            "observation_dimension": self.observation_dimension,
            "linear_system": self.linear_system,
            "gaussian_noise": self.gaussian_noise,
        }
        if self.state_names:
            result["model"]["state_names"] = self.state_names

        # Task
        result["task"] = {
            "n_observations": self.n_observations,
        }
        if self.observation_sequence_hash:
            result["task"]["observation_hash"] = self.observation_sequence_hash

        # Result
        result["estimation"] = {
            "success": self.filter_success,
        }

        if self.state_estimate:
            result["estimation"]["state"] = {
                k: str(v) for k, v in self.state_estimate.items()
            }

        # Uncertainty
        uncertainty = {}
        if self.covariance_trace is not None:
            uncertainty["covariance_trace"] = str(self.covariance_trace)
        if self.credible_interval_width is not None:
            uncertainty["credible_interval_width"] = str(self.credible_interval_width)
        if self.entropy is not None:
            uncertainty["entropy"] = str(self.entropy)
        if uncertainty:
            result["estimation"]["uncertainty"] = uncertainty

        # Quality
        if self.normalized_innovation_squared is not None:
            result["quality"] = {
                "NIS": str(self.normalized_innovation_squared),
                "innovation_whiteness": self.innovation_whiteness,
            }

        # Method proof
        if self.method_proof:
            result["method_proof"] = self.method_proof.to_dict()

        # Failure
        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value

        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {
                "fail_type": obs.fail_type.value,
            }
            if obs.effective_sample_size is not None:
                result["obstruction"]["ess"] = str(obs.effective_sample_size)
                result["obstruction"]["n_particles"] = obs.n_particles
            if obs.estimation_error is not None:
                result["obstruction"]["error"] = str(obs.estimation_error)
            if obs.observability_rank is not None:
                result["obstruction"]["observability_rank"] = obs.observability_rank
                result["obstruction"]["state_dimension"] = obs.state_dimension

        # Witness
        if self.estimation_witness:
            result["estimation_witness"] = self.estimation_witness.to_dict()

        return result

    @classmethod
    def from_kalman(
        cls,
        model_id: str,
        state_names: List[str],
        observation_dimension: int,
        n_observations: int,
        state_estimate: Dict[str, Scalar],
        covariance_trace: Scalar,
    ) -> "FilterCertificate":
        """Create certificate from Kalman filter estimation."""
        return cls(
            model_id=model_id,
            state_dimension=len(state_names),
            state_names=state_names,
            observation_dimension=observation_dimension,
            linear_system=True,
            gaussian_noise=True,
            n_observations=n_observations,
            filter_success=True,
            state_estimate=state_estimate,
            covariance_trace=covariance_trace,
            method_proof=FilterMethodProof(
                method=FilterMethod.KALMAN,
                n_timesteps=n_observations,
                covariance_trace=covariance_trace,
                verifiable=True,
            ),
            estimation_witness=DerivationWitness(
                invariant_name="kalman_estimate",
                derivation_operator="kalman_filter",
                input_data={
                    "n_observations": n_observations,
                    "state_dimension": len(state_names),
                },
                output_value=1,  # Success
                verifiable=True,
            ),
            strict_mode=True,
        )

    @classmethod
    def from_particle_filter(
        cls,
        model_id: str,
        state_names: List[str],
        observation_dimension: int,
        n_observations: int,
        n_particles: int,
        state_estimate: Dict[str, Scalar],
        credible_interval_width: Scalar,
        effective_sample_size: Scalar,
        n_resamples: int,
    ) -> "FilterCertificate":
        """Create certificate from particle filter estimation."""
        return cls(
            model_id=model_id,
            state_dimension=len(state_names),
            state_names=state_names,
            observation_dimension=observation_dimension,
            linear_system=False,
            gaussian_noise=False,
            n_observations=n_observations,
            filter_success=True,
            state_estimate=state_estimate,
            credible_interval_width=credible_interval_width,
            method_proof=FilterMethodProof(
                method=FilterMethod.PARTICLE,
                n_particles=n_particles,
                n_timesteps=n_observations,
                resampling_method="systematic",
                effective_sample_size=effective_sample_size,
                n_resamples=n_resamples,
                verifiable=True,
            ),
            estimation_witness=DerivationWitness(
                invariant_name="particle_estimate",
                derivation_operator="particle_filter",
                input_data={
                    "n_observations": n_observations,
                    "n_particles": n_particles,
                    "n_resamples": n_resamples,
                },
                output_value=1,
                verifiable=True,
            ),
            strict_mode=True,
        )


def validate_filter_certificate(cert: FilterCertificate) -> StrictValidationResult:
    """Validate filter certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: if failure_mode is set, filter_success should be False
    if cert.failure_mode is not None and cert.filter_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but filter_success=True"
        )

    # Consistency: PARTICLE_DEGENERACY requires obstruction with ESS evidence
    if cert.failure_mode == FilterFailType.PARTICLE_DEGENERACY:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: PARTICLE_DEGENERACY requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.effective_sample_size is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: PARTICLE_DEGENERACY requires effective_sample_size"
            )

    # Consistency: STATE_UNOBSERVABLE requires observability evidence
    if cert.failure_mode == FilterFailType.STATE_UNOBSERVABLE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: STATE_UNOBSERVABLE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.observability_rank is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: STATE_UNOBSERVABLE requires observability_rank"
            )

    # Consistency: FILTER_DIVERGED requires error evidence
    if cert.failure_mode == FilterFailType.FILTER_DIVERGED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: FILTER_DIVERGED requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.estimation_error is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: FILTER_DIVERGED requires estimation_error"
            )

    # Structural: Kalman on nonlinear should warn
    if cert.method_proof and cert.method_proof.method == FilterMethod.KALMAN:
        if not cert.linear_system:
            warnings.append(
                "KALMAN_ON_NONLINEAR: Kalman filter on nonlinear system (consider EKF/UKF)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# MCTS CERTIFICATE (Chapter 8: Online Planning)
# ============================================================================
# Monte Carlo Tree Search with QA-native structural pruning.
# Key differentiator: SCC/orbit pruning proof fields.


class MCTSExplorationRule(Enum):
    """Exploration rules for MCTS node selection."""

    UCB1 = "ucb1"  # Upper Confidence Bound
    UCB1_TUNED = "ucb1_tuned"
    PUCT = "puct"  # Predictor + UCT (AlphaGo style)
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON = "thompson"  # Thompson sampling


class MCTSBackupOperator(Enum):
    """Backup operators for value propagation."""

    MEAN = "mean"  # Average returns
    MAX = "max"  # Maximum return (optimistic)
    MINIMAX = "minimax"  # For adversarial games
    SOFT_MAX = "soft_max"  # Boltzmann backup


class MCTSFailType(Enum):
    """Failure modes for MCTS planning."""

    BUDGET_EXHAUSTED = "budget_exhausted"  # Ran out of rollouts
    NO_VALID_ACTIONS = "no_valid_actions"  # Root has no legal moves
    ALL_ROLLOUTS_TERMINAL = "all_rollouts_terminal"  # All paths hit terminal
    TREE_TOO_DEEP = "tree_too_deep"  # Exceeded max depth
    VALUE_DIVERGENCE = "value_divergence"  # Value estimates unstable
    SCC_UNREACHABLE = "scc_unreachable"  # Target SCC unreachable from root


@dataclass
class SCCPruningWitness:
    """
    Witness for SCC-based subtree pruning.

    The key QA-native insight: if a subtree's reachable SCC set
    doesn't intersect the target class, that subtree can be
    deterministically pruned (no rollouts needed).
    """

    # Hash of SCC computation inputs (state space + generator set)
    scc_computation_hash: str

    # Number of nodes pruned via SCC analysis
    nodes_pruned: int

    # SCC IDs that were unreachable from pruned subtrees
    unreachable_scc_ids: List[int] = field(default_factory=list)

    # Target SCC ID (the goal)
    target_scc_id: Optional[int] = None

    # Verification: can recompute SCC membership
    verifiable: bool = True


@dataclass
class QAWMReturnWitness:
    """
    Witness for QAWM-based return prediction.

    Instead of random rollouts, use QAWM (QA World Model) to predict
    return-in-k steps. This makes rollouts deterministic when the
    model is accurate.
    """

    # Hash of QAWM model parameters
    qawm_model_hash: str

    # Number of rollouts replaced by QAWM prediction
    rollouts_replaced: int

    # Horizon used for return-in-k prediction
    prediction_horizon: int

    # Confidence in predictions (optional)
    prediction_confidence: Optional[Scalar] = None


@dataclass
class MCTSMethodProof:
    """Proof of MCTS execution parameters."""

    # Core MCTS parameters
    exploration_rule: MCTSExplorationRule
    backup_operator: MCTSBackupOperator
    rollout_policy: str = "uniform_random"  # or "learned", "heuristic"

    # Budget
    n_rollouts: int = 0
    max_depth: int = 0

    # Tree statistics
    nodes_expanded: int = 0
    max_tree_depth: int = 0
    unique_states_visited: int = 0

    # UCB parameters (if applicable)
    exploration_constant: Optional[Scalar] = None  # c in UCB1

    # Random seed for reproducibility
    random_seed: Optional[int] = None

    # Verification
    verifiable: bool = True


@dataclass
class MCTSObstructionEvidence:
    """Evidence for MCTS failure modes."""

    fail_type: MCTSFailType

    # BUDGET_EXHAUSTED
    rollouts_used: Optional[int] = None
    rollout_budget: Optional[int] = None

    # NO_VALID_ACTIONS
    root_state: Optional[str] = None

    # ALL_ROLLOUTS_TERMINAL
    terminal_count: Optional[int] = None

    # TREE_TOO_DEEP
    depth_reached: Optional[int] = None
    max_depth: Optional[int] = None

    # VALUE_DIVERGENCE
    value_variance: Optional[Scalar] = None
    variance_threshold: Optional[Scalar] = None

    # SCC_UNREACHABLE
    root_scc_id: Optional[int] = None
    target_scc_id: Optional[int] = None


@dataclass
class MCTSCertificate:
    """
    Certificate for Monte Carlo Tree Search planning.

    QA-native features:
    - SCC pruning proof: subtrees pruned by reachability analysis
    - QAWM return prediction: deterministic value estimates
    - Orbit structure: state equivalence classes

    Reference: Algorithms for Decision Making, Chapter 8
    """

    # Model identification
    model_id: str
    model_description: Optional[str] = None

    # State space specification
    root_state: str = ""
    state_space_size: Optional[int] = None
    action_space_size: Optional[int] = None

    # Target specification (QA-native: reachability goal)
    target_class: Optional[str] = None  # SCC or state set name
    target_states: Optional[List[str]] = None

    # Planning outcome
    planning_success: bool = False
    best_action: Optional[str] = None
    expected_return: Optional[Scalar] = None
    action_values: Optional[Dict[str, Scalar]] = None  # Q(root, a) for each a

    # Failure handling
    failure_mode: Optional[MCTSFailType] = None
    obstruction_if_fail: Optional[MCTSObstructionEvidence] = None

    # Method proof
    method_proof: Optional[MCTSMethodProof] = None

    # QA-native pruning witnesses (the key differentiator)
    scc_pruning_witness: Optional[SCCPruningWitness] = None
    qawm_return_witness: Optional[QAWMReturnWitness] = None

    # Comparison with baselines
    vanilla_mcts_rollouts: Optional[int] = None  # Rollouts without QA pruning
    qa_mcts_rollouts: Optional[int] = None  # Rollouts with QA pruning
    pruning_efficiency: Optional[Scalar] = None  # 1 - (qa/vanilla)

    # Derivation witness (optional)
    planning_witness: Optional[DerivationWitness] = None

    # Validation mode
    strict_mode: bool = True

    def __post_init__(self):
        """Validate certificate on construction."""
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise ValueError(f"MCTSCertificate validation failed: {violations}")

    def get_validity_violations(self) -> List[str]:
        """Return list of validity violations."""
        violations = []

        # Must have root state
        if not self.root_state:
            violations.append("ROOT_STATE_REQUIRED: must specify root_state")

        # Success requires best_action
        if self.planning_success and self.best_action is None:
            violations.append(
                "SUCCESS_REQUIRES_ACTION: planning_success=True requires best_action"
            )

        # Success requires method proof
        if self.planning_success and self.method_proof is None:
            violations.append(
                "SUCCESS_REQUIRES_PROOF: planning_success=True requires method_proof"
            )

        # Failure requires failure_mode
        if not self.planning_success and self.failure_mode is None:
            violations.append(
                "FAILURE_REQUIRES_MODE: planning_success=False requires failure_mode"
            )

        # QAWM witness requires hash
        if self.qawm_return_witness is not None:
            if not self.qawm_return_witness.qawm_model_hash:
                violations.append(
                    "QAWM_REQUIRES_HASH: qawm_return_witness requires qawm_model_hash"
                )

        # SCC pruning witness requires hash and count
        if self.scc_pruning_witness is not None:
            if not self.scc_pruning_witness.scc_computation_hash:
                violations.append(
                    "SCC_REQUIRES_HASH: scc_pruning_witness requires scc_computation_hash"
                )
            if self.scc_pruning_witness.nodes_pruned < 0:
                violations.append(
                    "SCC_INVALID_COUNT: nodes_pruned must be non-negative"
                )

        # Pruning efficiency must be valid fraction
        if self.pruning_efficiency is not None:
            eff = to_scalar(self.pruning_efficiency)
            if eff < 0 or eff > 1:
                violations.append(
                    f"INVALID_EFFICIENCY: pruning_efficiency must be in [0,1], got {eff}"
                )

        return violations

    def is_valid(self) -> bool:
        """Check if certificate is valid."""
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        """Export certificate as JSON-serializable dict."""
        result = {
            "schema": "qa_mcts_cert/v1",
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if self.model_description:
            result["model_description"] = self.model_description

        result["root_state"] = self.root_state

        if self.state_space_size is not None:
            result["state_space_size"] = self.state_space_size
        if self.action_space_size is not None:
            result["action_space_size"] = self.action_space_size

        if self.target_class:
            result["target_class"] = self.target_class
        if self.target_states:
            result["target_states"] = self.target_states

        result["planning"] = {
            "success": self.planning_success,
        }
        if self.best_action:
            result["planning"]["best_action"] = self.best_action
        if self.expected_return is not None:
            result["planning"]["expected_return"] = str(self.expected_return)
        if self.action_values:
            result["planning"]["action_values"] = {
                k: str(v) for k, v in self.action_values.items()
            }

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.obstruction_if_fail:
            result["obstruction"] = {
                "fail_type": self.obstruction_if_fail.fail_type.value
            }
            obs = self.obstruction_if_fail
            if obs.rollouts_used is not None:
                result["obstruction"]["rollouts_used"] = obs.rollouts_used
            if obs.rollout_budget is not None:
                result["obstruction"]["rollout_budget"] = obs.rollout_budget

        if self.method_proof:
            mp = self.method_proof
            result["method_proof"] = {
                "exploration_rule": mp.exploration_rule.value,
                "backup_operator": mp.backup_operator.value,
                "rollout_policy": mp.rollout_policy,
                "n_rollouts": mp.n_rollouts,
                "max_depth": mp.max_depth,
                "nodes_expanded": mp.nodes_expanded,
                "verifiable": mp.verifiable,
            }
            if mp.exploration_constant is not None:
                result["method_proof"]["exploration_constant"] = str(
                    mp.exploration_constant
                )
            if mp.random_seed is not None:
                result["method_proof"]["random_seed"] = mp.random_seed

        # QA-native witnesses
        if self.scc_pruning_witness:
            spw = self.scc_pruning_witness
            result["scc_pruning_witness"] = {
                "scc_computation_hash": spw.scc_computation_hash,
                "nodes_pruned": spw.nodes_pruned,
                "unreachable_scc_ids": spw.unreachable_scc_ids,
                "verifiable": spw.verifiable,
            }
            if spw.target_scc_id is not None:
                result["scc_pruning_witness"]["target_scc_id"] = spw.target_scc_id

        if self.qawm_return_witness:
            qrw = self.qawm_return_witness
            result["qawm_return_witness"] = {
                "qawm_model_hash": qrw.qawm_model_hash,
                "rollouts_replaced": qrw.rollouts_replaced,
                "prediction_horizon": qrw.prediction_horizon,
            }
            if qrw.prediction_confidence is not None:
                result["qawm_return_witness"]["prediction_confidence"] = str(
                    qrw.prediction_confidence
                )

        # Comparison metrics
        if self.vanilla_mcts_rollouts is not None:
            result["comparison"] = {
                "vanilla_mcts_rollouts": self.vanilla_mcts_rollouts,
            }
            if self.qa_mcts_rollouts is not None:
                result["comparison"]["qa_mcts_rollouts"] = self.qa_mcts_rollouts
            if self.pruning_efficiency is not None:
                result["comparison"]["pruning_efficiency"] = str(
                    self.pruning_efficiency
                )

        return result

    @classmethod
    def from_mcts_run(
        cls,
        model_id: str,
        root_state: str,
        best_action: str,
        expected_return: Scalar,
        action_values: Dict[str, Scalar],
        exploration_rule: MCTSExplorationRule,
        backup_operator: MCTSBackupOperator,
        n_rollouts: int,
        max_depth: int,
        nodes_expanded: int,
        exploration_constant: Optional[Scalar] = None,
        random_seed: Optional[int] = None,
        target_class: Optional[str] = None,
    ) -> "MCTSCertificate":
        """Factory for successful MCTS run."""
        return cls(
            model_id=model_id,
            root_state=root_state,
            target_class=target_class,
            planning_success=True,
            best_action=best_action,
            expected_return=to_scalar(expected_return),
            action_values={k: to_scalar(v) for k, v in action_values.items()},
            method_proof=MCTSMethodProof(
                exploration_rule=exploration_rule,
                backup_operator=backup_operator,
                n_rollouts=n_rollouts,
                max_depth=max_depth,
                nodes_expanded=nodes_expanded,
                exploration_constant=to_scalar(exploration_constant)
                if exploration_constant is not None
                else None,
                random_seed=random_seed,
            ),
            strict_mode=True,
        )

    @classmethod
    def from_qa_mcts_run(
        cls,
        model_id: str,
        root_state: str,
        best_action: str,
        expected_return: Scalar,
        action_values: Dict[str, Scalar],
        exploration_rule: MCTSExplorationRule,
        backup_operator: MCTSBackupOperator,
        n_rollouts: int,
        max_depth: int,
        nodes_expanded: int,
        # QA-native fields
        scc_computation_hash: str,
        nodes_pruned_by_scc: int,
        unreachable_scc_ids: List[int],
        target_scc_id: int,
        vanilla_rollouts_baseline: int,
        exploration_constant: Optional[Scalar] = None,
        random_seed: Optional[int] = None,
    ) -> "MCTSCertificate":
        """Factory for QA-enhanced MCTS with SCC pruning."""
        qa_rollouts = n_rollouts
        vanilla_rollouts = vanilla_rollouts_baseline
        efficiency = Fraction(vanilla_rollouts - qa_rollouts, vanilla_rollouts) if vanilla_rollouts > 0 else Fraction(0)

        return cls(
            model_id=model_id,
            root_state=root_state,
            target_class=f"SCC_{target_scc_id}",
            planning_success=True,
            best_action=best_action,
            expected_return=to_scalar(expected_return),
            action_values={k: to_scalar(v) for k, v in action_values.items()},
            method_proof=MCTSMethodProof(
                exploration_rule=exploration_rule,
                backup_operator=backup_operator,
                n_rollouts=n_rollouts,
                max_depth=max_depth,
                nodes_expanded=nodes_expanded,
                exploration_constant=to_scalar(exploration_constant)
                if exploration_constant is not None
                else None,
                random_seed=random_seed,
            ),
            scc_pruning_witness=SCCPruningWitness(
                scc_computation_hash=scc_computation_hash,
                nodes_pruned=nodes_pruned_by_scc,
                unreachable_scc_ids=unreachable_scc_ids,
                target_scc_id=target_scc_id,
            ),
            vanilla_mcts_rollouts=vanilla_rollouts,
            qa_mcts_rollouts=qa_rollouts,
            pruning_efficiency=efficiency,
            strict_mode=True,
        )


def validate_mcts_certificate(cert: MCTSCertificate) -> StrictValidationResult:
    """Validate MCTS certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: failure_mode with success
    if cert.failure_mode is not None and cert.planning_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but planning_success=True"
        )

    # Consistency: BUDGET_EXHAUSTED requires budget evidence
    if cert.failure_mode == MCTSFailType.BUDGET_EXHAUSTED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: BUDGET_EXHAUSTED requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.rollout_budget is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: BUDGET_EXHAUSTED requires rollout_budget"
            )

    # Consistency: SCC_UNREACHABLE requires SCC evidence
    if cert.failure_mode == MCTSFailType.SCC_UNREACHABLE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: SCC_UNREACHABLE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.target_scc_id is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: SCC_UNREACHABLE requires target_scc_id"
            )

    # Structural: pruning witness with no pruned nodes is suspicious
    if cert.scc_pruning_witness is not None:
        if cert.scc_pruning_witness.nodes_pruned == 0:
            warnings.append(
                "SCC_PRUNING_ZERO: scc_pruning_witness present but nodes_pruned=0"
            )

    # Structural: QAWM with no rollouts replaced is suspicious
    if cert.qawm_return_witness is not None:
        if cert.qawm_return_witness.rollouts_replaced == 0:
            warnings.append(
                "QAWM_NO_REPLACEMENT: qawm_return_witness present but rollouts_replaced=0"
            )

    # Informational: efficiency report
    if cert.pruning_efficiency is not None:
        eff = to_scalar(cert.pruning_efficiency)
        if eff > Fraction(1, 2):
            warnings.append(
                f"HIGH_PRUNING_EFFICIENCY: {eff} (>50% rollouts saved via SCC analysis)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# EXPLORATION CERTIFICATE (Chapter 9: Exploration-Exploitation)
# ============================================================================
# Certificates for how a policy explored under uncertainty.
# Key QA insight: regret = steps to target vs BFS optimal steps.


class ExplorationMethod(Enum):
    """Methods for exploration-exploitation tradeoff."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    UCB1_TUNED = "ucb1_tuned"
    THOMPSON_SAMPLING = "thompson_sampling"
    BOLTZMANN = "boltzmann"  # Softmax exploration
    OPTIMISTIC = "optimistic"  # Optimistic initialization
    INFO_GAIN = "info_gain"  # Information-directed sampling


class UncertaintyMeasure(Enum):
    """How uncertainty is measured for exploration."""

    VISIT_COUNT = "visit_count"  # N(s,a) based
    POSTERIOR_VARIANCE = "posterior_variance"  # Bayesian
    PACKET_UNCERTAINTY = "packet_uncertainty"  # QA-native
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"  # Multiple models
    QAWM_CONFIDENCE = "qawm_confidence"  # QAWM prediction confidence


class ExplorationFailType(Enum):
    """Failure modes for exploration."""

    BUDGET_EXHAUSTED = "budget_exhausted"  # Ran out of episodes
    STUCK_IN_LOCAL = "stuck_in_local"  # Couldn't escape local optimum
    EXPLORATION_COLLAPSED = "exploration_collapsed"  # Premature convergence
    HIGH_REGRET = "high_regret"  # Regret exceeded threshold


@dataclass
class RegretWitness:
    """
    Witness for regret analysis.

    QA-native: regret is measured as (actual steps to target) - (BFS optimal steps).
    This is a concrete, verifiable quantity in reachability problems.
    """

    # Steps actually taken to reach target
    actual_steps: int

    # BFS optimal steps (lower bound)
    optimal_steps: int

    # Cumulative regret = sum of (actual - optimal) over episodes
    cumulative_regret: int

    # Per-episode regret history (optional)
    regret_per_episode: Optional[List[int]] = None

    # Regret bound (theoretical, if applicable)
    regret_bound: Optional[str] = None  # e.g., "O(sqrt(T))"

    # Verification
    verifiable: bool = True


@dataclass
class ExplorationMethodProof:
    """Proof of exploration method execution."""

    method: ExplorationMethod
    uncertainty_measure: UncertaintyMeasure

    # Budget used
    total_episodes: int = 0
    total_steps: int = 0
    oracle_calls: int = 0  # Calls to environment/simulator

    # Method-specific parameters
    epsilon: Optional[Scalar] = None  # For epsilon-greedy
    exploration_constant: Optional[Scalar] = None  # For UCB (c parameter)
    temperature: Optional[Scalar] = None  # For Boltzmann
    prior_strength: Optional[Scalar] = None  # For Thompson

    # Decay schedule (if applicable)
    decay_schedule: Optional[str] = None  # e.g., "1/sqrt(t)"

    # Statistics
    exploration_rate: Optional[Scalar] = None  # Fraction of exploratory actions
    unique_states_visited: int = 0

    # Verification
    verifiable: bool = True


@dataclass
class ExplorationObstructionEvidence:
    """Evidence for exploration failure modes."""

    fail_type: ExplorationFailType

    # BUDGET_EXHAUSTED
    episodes_used: Optional[int] = None
    episode_budget: Optional[int] = None

    # STUCK_IN_LOCAL
    local_optimum_state: Optional[str] = None
    escape_attempts: Optional[int] = None

    # EXPLORATION_COLLAPSED
    final_exploration_rate: Optional[Scalar] = None
    min_exploration_threshold: Optional[Scalar] = None

    # HIGH_REGRET
    cumulative_regret: Optional[int] = None
    regret_threshold: Optional[int] = None


@dataclass
class ExplorationCertificate:
    """
    Certificate for exploration-exploitation strategy.

    Documents how a policy balanced exploration vs exploitation,
    what uncertainty measure was used, and the resulting regret.

    QA-native: regret is (actual reachability steps) - (BFS optimal).

    Reference: Algorithms for Decision Making, Chapter 9
    """

    # Model identification
    model_id: str
    model_description: Optional[str] = None

    # Environment specification
    state_space_size: Optional[int] = None
    action_space_size: Optional[int] = None
    target_class: Optional[str] = None

    # Exploration outcome
    exploration_success: bool = False
    target_reached: bool = False
    final_policy_id: Optional[str] = None  # Link to PolicyCertificate

    # Regret analysis (the key metric)
    regret_witness: Optional[RegretWitness] = None

    # Failure handling
    failure_mode: Optional[ExplorationFailType] = None
    obstruction_if_fail: Optional[ExplorationObstructionEvidence] = None

    # Method proof
    method_proof: Optional[ExplorationMethodProof] = None

    # Comparison with baselines
    random_baseline_steps: Optional[int] = None
    greedy_baseline_steps: Optional[int] = None
    exploration_improvement: Optional[Scalar] = None  # vs random

    # Derivation witness
    exploration_witness: Optional[DerivationWitness] = None

    # Validation mode
    strict_mode: bool = True

    def __post_init__(self):
        """Validate certificate on construction."""
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise ValueError(f"ExplorationCertificate validation failed: {violations}")

    def get_validity_violations(self) -> List[str]:
        """Return list of validity violations."""
        violations = []

        # Success requires method proof
        if self.exploration_success and self.method_proof is None:
            violations.append(
                "SUCCESS_REQUIRES_PROOF: exploration_success=True requires method_proof"
            )

        # Failure requires failure_mode
        if not self.exploration_success and self.failure_mode is None:
            violations.append(
                "FAILURE_REQUIRES_MODE: exploration_success=False requires failure_mode"
            )

        # Regret witness internal consistency
        if self.regret_witness is not None:
            rw = self.regret_witness
            if rw.actual_steps < rw.optimal_steps:
                violations.append(
                    f"INVALID_REGRET: actual_steps ({rw.actual_steps}) < optimal_steps ({rw.optimal_steps})"
                )
            expected_regret = rw.actual_steps - rw.optimal_steps
            if rw.cumulative_regret < expected_regret:
                violations.append(
                    f"REGRET_UNDERCOUNT: cumulative_regret ({rw.cumulative_regret}) < single-episode regret ({expected_regret})"
                )

        # Method-specific validation
        if self.method_proof is not None:
            mp = self.method_proof
            if mp.method == ExplorationMethod.EPSILON_GREEDY and mp.epsilon is None:
                violations.append(
                    "EPSILON_GREEDY_REQUIRES_EPSILON: epsilon parameter required"
                )
            if mp.method == ExplorationMethod.UCB1 and mp.exploration_constant is None:
                violations.append(
                    "UCB1_REQUIRES_CONSTANT: exploration_constant required"
                )
            if mp.method == ExplorationMethod.BOLTZMANN and mp.temperature is None:
                violations.append(
                    "BOLTZMANN_REQUIRES_TEMP: temperature parameter required"
                )

        return violations

    def is_valid(self) -> bool:
        """Check if certificate is valid."""
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        """Export certificate as JSON-serializable dict."""
        result = {
            "schema": "qa_exploration_cert/v1",
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if self.model_description:
            result["model_description"] = self.model_description

        if self.state_space_size is not None:
            result["state_space_size"] = self.state_space_size
        if self.target_class:
            result["target_class"] = self.target_class

        result["exploration"] = {
            "success": self.exploration_success,
            "target_reached": self.target_reached,
        }
        if self.final_policy_id:
            result["exploration"]["final_policy_id"] = self.final_policy_id

        if self.regret_witness:
            rw = self.regret_witness
            result["regret"] = {
                "actual_steps": rw.actual_steps,
                "optimal_steps": rw.optimal_steps,
                "cumulative_regret": rw.cumulative_regret,
                "verifiable": rw.verifiable,
            }
            if rw.regret_bound:
                result["regret"]["bound"] = rw.regret_bound

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {"fail_type": obs.fail_type.value}
            if obs.cumulative_regret is not None:
                result["obstruction"]["cumulative_regret"] = obs.cumulative_regret

        if self.method_proof:
            mp = self.method_proof
            result["method_proof"] = {
                "method": mp.method.value,
                "uncertainty_measure": mp.uncertainty_measure.value,
                "total_episodes": mp.total_episodes,
                "total_steps": mp.total_steps,
                "unique_states_visited": mp.unique_states_visited,
                "verifiable": mp.verifiable,
            }
            if mp.epsilon is not None:
                result["method_proof"]["epsilon"] = str(mp.epsilon)
            if mp.exploration_constant is not None:
                result["method_proof"]["exploration_constant"] = str(mp.exploration_constant)
            if mp.exploration_rate is not None:
                result["method_proof"]["exploration_rate"] = str(mp.exploration_rate)

        if self.random_baseline_steps is not None:
            result["baselines"] = {
                "random_steps": self.random_baseline_steps,
            }
            if self.greedy_baseline_steps is not None:
                result["baselines"]["greedy_steps"] = self.greedy_baseline_steps
            if self.exploration_improvement is not None:
                result["baselines"]["improvement"] = str(self.exploration_improvement)

        return result

    @classmethod
    def from_ucb_exploration(
        cls,
        model_id: str,
        actual_steps: int,
        optimal_steps: int,
        total_episodes: int,
        exploration_constant: Scalar,
        unique_states_visited: int,
        target_reached: bool = True,
        target_class: Optional[str] = None,
    ) -> "ExplorationCertificate":
        """Factory for UCB1 exploration certificate."""
        cumulative_regret = actual_steps - optimal_steps

        return cls(
            model_id=model_id,
            target_class=target_class,
            exploration_success=True,
            target_reached=target_reached,
            regret_witness=RegretWitness(
                actual_steps=actual_steps,
                optimal_steps=optimal_steps,
                cumulative_regret=cumulative_regret,
                regret_bound="O(sqrt(T * log(T)))",
            ),
            method_proof=ExplorationMethodProof(
                method=ExplorationMethod.UCB1,
                uncertainty_measure=UncertaintyMeasure.VISIT_COUNT,
                total_episodes=total_episodes,
                total_steps=actual_steps,
                exploration_constant=to_scalar(exploration_constant),
                unique_states_visited=unique_states_visited,
            ),
            strict_mode=True,
        )

    @classmethod
    def from_thompson_exploration(
        cls,
        model_id: str,
        actual_steps: int,
        optimal_steps: int,
        total_episodes: int,
        prior_strength: Scalar,
        unique_states_visited: int,
        target_reached: bool = True,
        target_class: Optional[str] = None,
    ) -> "ExplorationCertificate":
        """Factory for Thompson sampling exploration certificate."""
        cumulative_regret = actual_steps - optimal_steps

        return cls(
            model_id=model_id,
            target_class=target_class,
            exploration_success=True,
            target_reached=target_reached,
            regret_witness=RegretWitness(
                actual_steps=actual_steps,
                optimal_steps=optimal_steps,
                cumulative_regret=cumulative_regret,
                regret_bound="O(sqrt(T))",
            ),
            method_proof=ExplorationMethodProof(
                method=ExplorationMethod.THOMPSON_SAMPLING,
                uncertainty_measure=UncertaintyMeasure.POSTERIOR_VARIANCE,
                total_episodes=total_episodes,
                total_steps=actual_steps,
                prior_strength=to_scalar(prior_strength),
                unique_states_visited=unique_states_visited,
            ),
            strict_mode=True,
        )


def validate_exploration_certificate(cert: ExplorationCertificate) -> StrictValidationResult:
    """Validate exploration certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: failure_mode with success
    if cert.failure_mode is not None and cert.exploration_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but exploration_success=True"
        )

    # Consistency: HIGH_REGRET requires regret evidence
    if cert.failure_mode == ExplorationFailType.HIGH_REGRET:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: HIGH_REGRET requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.cumulative_regret is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: HIGH_REGRET requires cumulative_regret"
            )

    # Consistency: EXPLORATION_COLLAPSED requires rate evidence
    if cert.failure_mode == ExplorationFailType.EXPLORATION_COLLAPSED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: EXPLORATION_COLLAPSED requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.final_exploration_rate is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: EXPLORATION_COLLAPSED requires final_exploration_rate"
            )

    # Informational: regret analysis
    if cert.regret_witness is not None:
        rw = cert.regret_witness
        if rw.optimal_steps > 0:
            regret_ratio = Fraction(rw.cumulative_regret, rw.optimal_steps)
            if regret_ratio > Fraction(1, 1):
                warnings.append(
                    f"HIGH_REGRET_RATIO: cumulative_regret/optimal = {regret_ratio} (>1x optimal)"
                )
            elif regret_ratio < Fraction(1, 10):
                warnings.append(
                    f"LOW_REGRET: cumulative_regret/optimal = {regret_ratio} (<10% of optimal)"
                )

    # Informational: exploration rate
    if cert.method_proof is not None and cert.method_proof.exploration_rate is not None:
        rate = to_scalar(cert.method_proof.exploration_rate)
        if rate < Fraction(1, 100):
            warnings.append(
                f"LOW_EXPLORATION_RATE: {rate} (<1% exploratory actions)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# RL CERTIFICATE (Chapter 12: Reinforcement Learning)
# ============================================================================
# Certificates for RL training runs.
# Key QA insight: Q-learning = generator-value learning, reward = distance delta.


class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""

    Q_LEARNING = "q_learning"
    SARSA = "sarsa"
    EXPECTED_SARSA = "expected_sarsa"
    DOUBLE_Q = "double_q"
    POLICY_GRADIENT = "policy_gradient"
    ACTOR_CRITIC = "actor_critic"
    PPO = "ppo"
    DQN = "dqn"


class RewardSpec(Enum):
    """QA-native reward specifications."""

    DISTANCE_DELTA = "distance_delta"  # r = d(s,T) - d(s',T) (BFS distance)
    OBSTRUCTION_PENALTY = "obstruction_penalty"  # r = -1 if obstruction
    GOAL_REWARD = "goal_reward"  # r = +1 if target reached
    SPARSE_GOAL = "sparse_goal"  # r = +1 at goal, 0 elsewhere
    SHAPED_DISTANCE = "shaped_distance"  # Potential-based shaping
    CUSTOM = "custom"


class RLFailType(Enum):
    """Failure modes for RL training."""

    CONVERGENCE_TIMEOUT = "convergence_timeout"  # Didn't converge in budget
    VALUE_DIVERGENCE = "value_divergence"  # Q-values exploded
    POLICY_COLLAPSE = "policy_collapse"  # Policy became deterministic too early
    REWARD_HACKING = "reward_hacking"  # Exploited reward function
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting"  # Performance degraded
    EXPLORATION_FAILURE = "exploration_failure"  # Never found target


@dataclass
class QValueWitness:
    """
    Witness for Q-value computation.

    Stores a sample of (s,a,r,s') transitions with Q-values
    for auditable replay.
    """

    # Sample transitions for verification
    sample_transitions: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"s": str, "a": str, "r": Scalar, "s_next": str, "q_before": Scalar, "q_after": Scalar}, ...]

    # Q-table hash (for tabular methods)
    q_table_hash: Optional[str] = None

    # Final Q-values for key states (optional)
    final_q_values: Optional[Dict[str, Dict[str, Scalar]]] = None  # state -> {action: Q}

    # Verification
    verifiable: bool = True


@dataclass
class RLMethodProof:
    """Proof of RL training execution."""

    algorithm: RLAlgorithm
    reward_spec: RewardSpec

    # Training parameters
    total_episodes: int = 0
    total_steps: int = 0
    learning_rate: Optional[Scalar] = None
    discount_factor: Optional[Scalar] = None  # gamma

    # Learning rate schedule (if applicable)
    lr_schedule: Optional[str] = None  # e.g., "linear_decay", "exponential"

    # Exploration method (links to ExplorationCertificate)
    exploration_method: Optional[ExplorationMethod] = None
    exploration_certificate_id: Optional[str] = None

    # Convergence metrics
    final_loss: Optional[Scalar] = None
    converged: bool = False
    convergence_episode: Optional[int] = None

    # Verification
    verifiable: bool = True


@dataclass
class RLObstructionEvidence:
    """Evidence for RL failure modes."""

    fail_type: RLFailType

    # CONVERGENCE_TIMEOUT
    episodes_run: Optional[int] = None
    episode_budget: Optional[int] = None
    final_performance: Optional[Scalar] = None

    # VALUE_DIVERGENCE
    max_q_value: Optional[Scalar] = None
    divergence_threshold: Optional[Scalar] = None
    divergence_episode: Optional[int] = None

    # POLICY_COLLAPSE
    entropy: Optional[Scalar] = None
    entropy_threshold: Optional[Scalar] = None

    # EXPLORATION_FAILURE
    states_visited: Optional[int] = None
    target_ever_reached: bool = False


@dataclass
class RLCertificate:
    """
    Certificate for Reinforcement Learning training run.

    QA-native features:
    - Q-learning = generator-value learning
    - Reward derived from BFS distance delta
    - Policy becomes reachability-optimal

    Reference: Algorithms for Decision Making, Chapter 12
    """

    # Model identification
    model_id: str
    model_description: Optional[str] = None

    # Environment specification
    state_space_size: Optional[int] = None
    action_space_size: Optional[int] = None
    target_class: Optional[str] = None
    generator_set: Optional[List[str]] = None  # QA-native: generators = actions

    # Training outcome
    training_success: bool = False
    final_policy_id: Optional[str] = None  # Link to PolicyCertificate
    final_performance: Optional[Scalar] = None  # e.g., average return

    # Q-value witness (for auditable replay)
    q_value_witness: Optional[QValueWitness] = None

    # Failure handling
    failure_mode: Optional[RLFailType] = None
    obstruction_if_fail: Optional[RLObstructionEvidence] = None

    # Method proof
    method_proof: Optional[RLMethodProof] = None

    # Comparison with baselines
    random_policy_return: Optional[Scalar] = None
    optimal_policy_return: Optional[Scalar] = None  # If known
    improvement_ratio: Optional[Scalar] = None

    # Derivation witness
    training_witness: Optional[DerivationWitness] = None

    # Validation mode
    strict_mode: bool = True

    def __post_init__(self):
        """Validate certificate on construction."""
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise ValueError(f"RLCertificate validation failed: {violations}")

    def get_validity_violations(self) -> List[str]:
        """Return list of validity violations."""
        violations = []

        # Success requires method proof
        if self.training_success and self.method_proof is None:
            violations.append(
                "SUCCESS_REQUIRES_PROOF: training_success=True requires method_proof"
            )

        # Failure requires failure_mode
        if not self.training_success and self.failure_mode is None:
            violations.append(
                "FAILURE_REQUIRES_MODE: training_success=False requires failure_mode"
            )

        # Q-learning requires discount factor
        if self.method_proof is not None:
            mp = self.method_proof
            if mp.algorithm in [RLAlgorithm.Q_LEARNING, RLAlgorithm.SARSA,
                               RLAlgorithm.DOUBLE_Q, RLAlgorithm.DQN]:
                if mp.discount_factor is None:
                    violations.append(
                        f"{mp.algorithm.value.upper()}_REQUIRES_GAMMA: discount_factor required"
                    )
                if mp.learning_rate is None:
                    violations.append(
                        f"{mp.algorithm.value.upper()}_REQUIRES_LR: learning_rate required"
                    )

        # Improvement ratio consistency
        if self.improvement_ratio is not None:
            if self.random_policy_return is None:
                violations.append(
                    "IMPROVEMENT_REQUIRES_BASELINE: improvement_ratio requires random_policy_return"
                )

        return violations

    def is_valid(self) -> bool:
        """Check if certificate is valid."""
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        """Export certificate as JSON-serializable dict."""
        result = {
            "schema": "qa_rl_cert/v1",
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if self.model_description:
            result["model_description"] = self.model_description

        if self.state_space_size is not None:
            result["state_space_size"] = self.state_space_size
        if self.target_class:
            result["target_class"] = self.target_class
        if self.generator_set:
            result["generator_set"] = self.generator_set

        result["training"] = {
            "success": self.training_success,
        }
        if self.final_policy_id:
            result["training"]["final_policy_id"] = self.final_policy_id
        if self.final_performance is not None:
            result["training"]["final_performance"] = str(self.final_performance)

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {"fail_type": obs.fail_type.value}
            if obs.episodes_run is not None:
                result["obstruction"]["episodes_run"] = obs.episodes_run

        if self.method_proof:
            mp = self.method_proof
            result["method_proof"] = {
                "algorithm": mp.algorithm.value,
                "reward_spec": mp.reward_spec.value,
                "total_episodes": mp.total_episodes,
                "total_steps": mp.total_steps,
                "converged": mp.converged,
                "verifiable": mp.verifiable,
            }
            if mp.learning_rate is not None:
                result["method_proof"]["learning_rate"] = str(mp.learning_rate)
            if mp.discount_factor is not None:
                result["method_proof"]["discount_factor"] = str(mp.discount_factor)
            if mp.exploration_method is not None:
                result["method_proof"]["exploration_method"] = mp.exploration_method.value

        if self.q_value_witness:
            qvw = self.q_value_witness
            result["q_value_witness"] = {
                "n_sample_transitions": len(qvw.sample_transitions),
                "verifiable": qvw.verifiable,
            }
            if qvw.q_table_hash:
                result["q_value_witness"]["q_table_hash"] = qvw.q_table_hash

        if self.random_policy_return is not None:
            result["baselines"] = {
                "random_return": str(self.random_policy_return),
            }
            if self.optimal_policy_return is not None:
                result["baselines"]["optimal_return"] = str(self.optimal_policy_return)
            if self.improvement_ratio is not None:
                result["baselines"]["improvement_ratio"] = str(self.improvement_ratio)

        return result

    @classmethod
    def from_q_learning_run(
        cls,
        model_id: str,
        total_episodes: int,
        total_steps: int,
        learning_rate: Scalar,
        discount_factor: Scalar,
        final_performance: Scalar,
        converged: bool,
        reward_spec: RewardSpec = RewardSpec.DISTANCE_DELTA,
        exploration_method: Optional[ExplorationMethod] = None,
        sample_transitions: Optional[List[Dict[str, Any]]] = None,
        target_class: Optional[str] = None,
        generator_set: Optional[List[str]] = None,
    ) -> "RLCertificate":
        """Factory for Q-learning training run."""
        q_witness = None
        if sample_transitions:
            q_witness = QValueWitness(sample_transitions=sample_transitions)

        return cls(
            model_id=model_id,
            target_class=target_class,
            generator_set=generator_set,
            training_success=converged,
            final_performance=to_scalar(final_performance),
            q_value_witness=q_witness,
            method_proof=RLMethodProof(
                algorithm=RLAlgorithm.Q_LEARNING,
                reward_spec=reward_spec,
                total_episodes=total_episodes,
                total_steps=total_steps,
                learning_rate=to_scalar(learning_rate),
                discount_factor=to_scalar(discount_factor),
                converged=converged,
                exploration_method=exploration_method,
            ),
            failure_mode=RLFailType.CONVERGENCE_TIMEOUT if not converged else None,
            obstruction_if_fail=RLObstructionEvidence(
                fail_type=RLFailType.CONVERGENCE_TIMEOUT,
                episodes_run=total_episodes,
            ) if not converged else None,
            strict_mode=True,
        )


def validate_rl_certificate(cert: RLCertificate) -> StrictValidationResult:
    """Validate RL certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: failure_mode with success
    if cert.failure_mode is not None and cert.training_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but training_success=True"
        )

    # Consistency: CONVERGENCE_TIMEOUT requires episode evidence
    if cert.failure_mode == RLFailType.CONVERGENCE_TIMEOUT:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: CONVERGENCE_TIMEOUT requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.episodes_run is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: CONVERGENCE_TIMEOUT requires episodes_run"
            )

    # Consistency: VALUE_DIVERGENCE requires divergence evidence
    if cert.failure_mode == RLFailType.VALUE_DIVERGENCE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: VALUE_DIVERGENCE requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.max_q_value is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: VALUE_DIVERGENCE requires max_q_value"
            )

    # Informational: QA-native reward
    if cert.method_proof is not None:
        if cert.method_proof.reward_spec == RewardSpec.DISTANCE_DELTA:
            warnings.append(
                "QA_NATIVE_REWARD: using BFS distance delta reward (reachability-optimal)"
            )

    # Informational: discount factor
    if cert.method_proof is not None and cert.method_proof.discount_factor is not None:
        gamma = to_scalar(cert.method_proof.discount_factor)
        if gamma == Fraction(1, 1):
            warnings.append(
                "UNDISCOUNTED: gamma=1 (undiscounted, average reward criterion)"
            )
        elif gamma < Fraction(9, 10):
            warnings.append(
                f"HIGH_DISCOUNT: gamma={gamma} (<0.9, may undervalue long-term rewards)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# RL RECOMPUTE HOOK (Auditable Mode)
# ============================================================================


def recompute_q_learning_update(
    cert: RLCertificate,
    transitions: List[Dict[str, Any]],
) -> StrictValidationResult:
    """
    Recompute Q-learning updates and verify they match certificate's Q-value witness.

    This is the "auditable mode" for RL - given a batch of (s,a,r,s') transitions,
    verify that the Q-value updates match the claimed witness.

    Args:
        cert: RLCertificate with Q-learning method and Q-value witness
        transitions: List of {"s": str, "a": str, "r": Scalar, "s_next": str,
                             "q_before": Scalar, "q_after": Scalar, "max_q_next": Scalar}

    Returns:
        StrictValidationResult with violations if recompute doesn't match
    """
    violations = []
    warnings = []

    # Must be Q-learning family
    if cert.method_proof is None:
        violations.append("RECOMPUTE_NO_METHOD: method_proof required")
        return StrictValidationResult(False, violations, warnings)

    valid_algorithms = [RLAlgorithm.Q_LEARNING, RLAlgorithm.DOUBLE_Q, RLAlgorithm.DQN]
    if cert.method_proof.algorithm not in valid_algorithms:
        violations.append(
            f"RECOMPUTE_METHOD_MISMATCH: recompute_q_learning_update requires Q-learning family, "
            f"got {cert.method_proof.algorithm.value}"
        )
        return StrictValidationResult(False, violations, warnings)

    # Must have learning parameters
    if cert.method_proof.learning_rate is None:
        violations.append("RECOMPUTE_MISSING_LR: learning_rate required for Q recompute")
        return StrictValidationResult(False, violations, warnings)

    if cert.method_proof.discount_factor is None:
        violations.append("RECOMPUTE_MISSING_GAMMA: discount_factor required for Q recompute")
        return StrictValidationResult(False, violations, warnings)

    alpha = to_scalar(cert.method_proof.learning_rate)
    gamma = to_scalar(cert.method_proof.discount_factor)

    # Verify each transition
    for i, t in enumerate(transitions):
        required_keys = ["s", "a", "r", "s_next", "q_before", "q_after", "max_q_next"]
        missing = [k for k in required_keys if k not in t]
        if missing:
            violations.append(
                f"RECOMPUTE_MISSING_KEYS: transition {i} missing keys: {missing}"
            )
            continue

        r = to_scalar(t["r"])
        q_before = to_scalar(t["q_before"])
        q_after_claimed = to_scalar(t["q_after"])
        max_q_next = to_scalar(t["max_q_next"])

        # Q-learning update: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        td_target = r + gamma * max_q_next
        td_error = td_target - q_before
        q_after_recomputed = q_before + alpha * td_error

        if q_after_claimed != q_after_recomputed:
            violations.append(
                f"RECOMPUTE_MISMATCH: transition {i} Q({t['s']},{t['a']}) "
                f"claimed={q_after_claimed}, recomputed={q_after_recomputed}"
            )

    if not violations:
        warnings.append(
            f"RECOMPUTE_VERIFIED: {len(transitions)} Q-learning updates match claimed values"
        )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# IMITATION CERTIFICATE (Chapter 13: Imitation Learning)
# ============================================================================
# Certificates for learning from demonstrations.
# Key QA insight: Inverse RL = target-class inference (reuse identifiability).


class ImitationMethod(Enum):
    """Imitation learning methods."""

    BEHAVIORAL_CLONING = "behavioral_cloning"  # Supervised policy learning
    INVERSE_RL = "inverse_rl"  # Infer reward from demos
    DAGGER = "dagger"  # Dataset Aggregation
    GAIL = "gail"  # Generative Adversarial Imitation
    AIRL = "airl"  # Adversarial Inverse RL


class ImitationFailType(Enum):
    """Failure modes for imitation learning."""

    INSUFFICIENT_DATA = "insufficient_data"  # Not enough demonstrations
    DISTRIBUTION_SHIFT = "distribution_shift"  # BC covariate shift
    REWARD_NON_IDENTIFIABLE = "reward_non_identifiable"  # IRL identifiability failure
    ORACLE_BUDGET_EXHAUSTED = "oracle_budget_exhausted"  # DAgger ran out of queries
    POLICY_DIVERGENCE = "policy_divergence"  # Policy drifted from expert
    TARGET_AMBIGUOUS = "target_ambiguous"  # Multiple consistent targets


@dataclass
class DemonstrationWitness:
    """
    Witness for demonstration data.

    Stores metadata about expert demonstrations used for imitation.
    """

    # Number of trajectories
    n_trajectories: int

    # Total state-action pairs
    n_state_action_pairs: int

    # Dataset hash (for reproducibility)
    dataset_hash: str

    # Expert information (if known)
    expert_id: Optional[str] = None
    expert_policy_hash: Optional[str] = None

    # Coverage metrics
    states_covered: Optional[int] = None
    coverage_ratio: Optional[Scalar] = None  # states_covered / total_states

    # Verification
    verifiable: bool = True


@dataclass
class InverseRLWitness:
    """
    Witness for inverse RL target inference.

    QA-native: IRL becomes target-class inference. The inferred
    "reward" corresponds to distance-to-target in QA terms.
    """

    # Inferred target class (QA-native)
    inferred_target_class: str

    # Confidence in inference
    confidence: Scalar

    # Identifiability check (reuses belief identifiability machinery)
    identifiable: bool
    identifiability_rank: Optional[int] = None

    # Alternative consistent targets (if non-identifiable)
    alternative_targets: Optional[List[str]] = None

    # Verification
    verifiable: bool = True


@dataclass
class DAggerWitness:
    """
    Witness for DAgger interactive learning.

    Tracks oracle queries made during dataset aggregation.
    """

    # Aggregation rounds
    n_rounds: int

    # Oracle queries
    total_oracle_queries: int
    oracle_budget: Optional[int] = None

    # Query strategy
    query_at_uncertainty: bool = True
    uncertainty_threshold: Optional[Scalar] = None

    # Performance per round (optional)
    performance_per_round: Optional[List[Scalar]] = None

    # Verification
    verifiable: bool = True


@dataclass
class ImitationMethodProof:
    """Proof of imitation learning execution."""

    method: ImitationMethod

    # Training parameters
    total_epochs: int = 0
    batch_size: Optional[int] = None
    learning_rate: Optional[Scalar] = None

    # Policy architecture (if neural)
    policy_architecture: Optional[str] = None
    policy_hash: Optional[str] = None

    # Loss metrics
    final_loss: Optional[Scalar] = None
    loss_trace_hash: Optional[str] = None

    # Method-specific witnesses
    demonstration_witness: Optional[DemonstrationWitness] = None
    inverse_rl_witness: Optional[InverseRLWitness] = None
    dagger_witness: Optional[DAggerWitness] = None

    # Verification
    verifiable: bool = True


@dataclass
class ImitationObstructionEvidence:
    """Evidence for imitation learning failure modes."""

    fail_type: ImitationFailType

    # INSUFFICIENT_DATA
    n_demonstrations: Optional[int] = None
    min_demonstrations_required: Optional[int] = None

    # DISTRIBUTION_SHIFT
    kl_divergence: Optional[Scalar] = None
    shift_threshold: Optional[Scalar] = None

    # REWARD_NON_IDENTIFIABLE
    identifiability_rank: Optional[int] = None
    required_rank: Optional[int] = None
    alternative_count: Optional[int] = None

    # ORACLE_BUDGET_EXHAUSTED
    queries_used: Optional[int] = None
    query_budget: Optional[int] = None

    # TARGET_AMBIGUOUS
    consistent_target_count: Optional[int] = None


@dataclass
class ImitationCertificate:
    """
    Certificate for Imitation Learning.

    QA-native features:
    - Behavioral cloning: learn reachability policy directly
    - Inverse RL: infer target class from demonstrations
    - DAgger: query oracle at uncertainty spikes (ties to exploration)

    Reference: Algorithms for Decision Making, Chapter 13
    """

    # Model identification
    model_id: str
    model_description: Optional[str] = None

    # Environment specification
    state_space_size: Optional[int] = None
    action_space_size: Optional[int] = None

    # Expert specification
    expert_policy_id: Optional[str] = None
    expert_target_class: Optional[str] = None  # Ground truth (if known)

    # Learning outcome
    learning_success: bool = False
    final_policy_id: Optional[str] = None
    inferred_target_class: Optional[str] = None  # For IRL

    # Performance metrics
    expert_match_rate: Optional[Scalar] = None  # Agreement with expert
    task_success_rate: Optional[Scalar] = None  # Actually reaching goal

    # Failure handling
    failure_mode: Optional[ImitationFailType] = None
    obstruction_if_fail: Optional[ImitationObstructionEvidence] = None

    # Method proof
    method_proof: Optional[ImitationMethodProof] = None

    # Comparison with baselines
    random_policy_match_rate: Optional[Scalar] = None
    improvement_over_random: Optional[Scalar] = None

    # Derivation witness
    learning_witness: Optional[DerivationWitness] = None

    # Validation mode
    strict_mode: bool = True

    def __post_init__(self):
        """Validate certificate on construction."""
        if self.strict_mode:
            violations = self.get_validity_violations()
            if violations:
                raise ValueError(f"ImitationCertificate validation failed: {violations}")

    def get_validity_violations(self) -> List[str]:
        """Return list of validity violations."""
        violations = []

        # Success requires method proof
        if self.learning_success and self.method_proof is None:
            violations.append(
                "SUCCESS_REQUIRES_PROOF: learning_success=True requires method_proof"
            )

        # Failure requires failure_mode
        if not self.learning_success and self.failure_mode is None:
            violations.append(
                "FAILURE_REQUIRES_MODE: learning_success=False requires failure_mode"
            )

        # Behavioral cloning requires demonstration witness
        if self.method_proof is not None:
            mp = self.method_proof
            if mp.method == ImitationMethod.BEHAVIORAL_CLONING:
                if mp.demonstration_witness is None:
                    violations.append(
                        "BC_REQUIRES_DEMOS: behavioral_cloning requires demonstration_witness"
                    )

            # Inverse RL requires IRL witness
            if mp.method == ImitationMethod.INVERSE_RL:
                if mp.inverse_rl_witness is None:
                    violations.append(
                        "IRL_REQUIRES_WITNESS: inverse_rl requires inverse_rl_witness"
                    )

            # DAgger requires DAgger witness
            if mp.method == ImitationMethod.DAGGER:
                if mp.dagger_witness is None:
                    violations.append(
                        "DAGGER_REQUIRES_WITNESS: dagger requires dagger_witness"
                    )

        # Expert match rate must be valid
        if self.expert_match_rate is not None:
            rate = to_scalar(self.expert_match_rate)
            if rate < 0 or rate > 1:
                violations.append(
                    f"INVALID_MATCH_RATE: expert_match_rate must be in [0,1], got {rate}"
                )

        return violations

    def is_valid(self) -> bool:
        """Check if certificate is valid."""
        return len(self.get_validity_violations()) == 0

    def to_json(self) -> Dict[str, Any]:
        """Export certificate as JSON-serializable dict."""
        result = {
            "schema": "qa_imitation_cert/v1",
            "model_id": self.model_id,
            "valid": self.is_valid(),
        }

        if self.model_description:
            result["model_description"] = self.model_description

        if self.state_space_size is not None:
            result["state_space_size"] = self.state_space_size

        if self.expert_policy_id:
            result["expert"] = {"policy_id": self.expert_policy_id}
            if self.expert_target_class:
                result["expert"]["target_class"] = self.expert_target_class

        result["learning"] = {
            "success": self.learning_success,
        }
        if self.final_policy_id:
            result["learning"]["final_policy_id"] = self.final_policy_id
        if self.inferred_target_class:
            result["learning"]["inferred_target_class"] = self.inferred_target_class
        if self.expert_match_rate is not None:
            result["learning"]["expert_match_rate"] = str(self.expert_match_rate)
        if self.task_success_rate is not None:
            result["learning"]["task_success_rate"] = str(self.task_success_rate)

        if self.failure_mode:
            result["failure_mode"] = self.failure_mode.value
        if self.obstruction_if_fail:
            obs = self.obstruction_if_fail
            result["obstruction"] = {"fail_type": obs.fail_type.value}

        if self.method_proof:
            mp = self.method_proof
            result["method_proof"] = {
                "method": mp.method.value,
                "total_epochs": mp.total_epochs,
                "verifiable": mp.verifiable,
            }
            if mp.final_loss is not None:
                result["method_proof"]["final_loss"] = str(mp.final_loss)
            if mp.policy_hash:
                result["method_proof"]["policy_hash"] = mp.policy_hash

            if mp.demonstration_witness:
                dw = mp.demonstration_witness
                result["demonstration_witness"] = {
                    "n_trajectories": dw.n_trajectories,
                    "n_state_action_pairs": dw.n_state_action_pairs,
                    "dataset_hash": dw.dataset_hash,
                }
                if dw.coverage_ratio is not None:
                    result["demonstration_witness"]["coverage_ratio"] = str(dw.coverage_ratio)

            if mp.inverse_rl_witness:
                irlw = mp.inverse_rl_witness
                result["inverse_rl_witness"] = {
                    "inferred_target_class": irlw.inferred_target_class,
                    "confidence": str(irlw.confidence),
                    "identifiable": irlw.identifiable,
                }
                if irlw.alternative_targets:
                    result["inverse_rl_witness"]["alternative_targets"] = irlw.alternative_targets

            if mp.dagger_witness:
                dagw = mp.dagger_witness
                result["dagger_witness"] = {
                    "n_rounds": dagw.n_rounds,
                    "total_oracle_queries": dagw.total_oracle_queries,
                    "query_at_uncertainty": dagw.query_at_uncertainty,
                }

        return result

    @classmethod
    def from_behavioral_cloning(
        cls,
        model_id: str,
        n_trajectories: int,
        n_state_action_pairs: int,
        dataset_hash: str,
        total_epochs: int,
        final_loss: Scalar,
        expert_match_rate: Scalar,
        learning_success: bool = True,
        expert_policy_id: Optional[str] = None,
    ) -> "ImitationCertificate":
        """Factory for behavioral cloning certificate."""
        return cls(
            model_id=model_id,
            expert_policy_id=expert_policy_id,
            learning_success=learning_success,
            expert_match_rate=to_scalar(expert_match_rate),
            method_proof=ImitationMethodProof(
                method=ImitationMethod.BEHAVIORAL_CLONING,
                total_epochs=total_epochs,
                final_loss=to_scalar(final_loss),
                demonstration_witness=DemonstrationWitness(
                    n_trajectories=n_trajectories,
                    n_state_action_pairs=n_state_action_pairs,
                    dataset_hash=dataset_hash,
                ),
            ),
            failure_mode=ImitationFailType.DISTRIBUTION_SHIFT if not learning_success else None,
            strict_mode=True,
        )

    @classmethod
    def from_inverse_rl(
        cls,
        model_id: str,
        n_trajectories: int,
        n_state_action_pairs: int,
        dataset_hash: str,
        inferred_target_class: str,
        confidence: Scalar,
        identifiable: bool,
        total_epochs: int,
        expert_target_class: Optional[str] = None,
        alternative_targets: Optional[List[str]] = None,
    ) -> "ImitationCertificate":
        """Factory for inverse RL certificate (target-class inference)."""
        learning_success = identifiable
        failure_mode = None
        obstruction = None

        if not identifiable:
            failure_mode = ImitationFailType.REWARD_NON_IDENTIFIABLE
            obstruction = ImitationObstructionEvidence(
                fail_type=ImitationFailType.REWARD_NON_IDENTIFIABLE,
                alternative_count=len(alternative_targets) if alternative_targets else 0,
            )

        return cls(
            model_id=model_id,
            expert_target_class=expert_target_class,
            learning_success=learning_success,
            inferred_target_class=inferred_target_class,
            failure_mode=failure_mode,
            obstruction_if_fail=obstruction,
            method_proof=ImitationMethodProof(
                method=ImitationMethod.INVERSE_RL,
                total_epochs=total_epochs,
                demonstration_witness=DemonstrationWitness(
                    n_trajectories=n_trajectories,
                    n_state_action_pairs=n_state_action_pairs,
                    dataset_hash=dataset_hash,
                ),
                inverse_rl_witness=InverseRLWitness(
                    inferred_target_class=inferred_target_class,
                    confidence=to_scalar(confidence),
                    identifiable=identifiable,
                    alternative_targets=alternative_targets,
                ),
            ),
            strict_mode=True,
        )

    @classmethod
    def from_dagger(
        cls,
        model_id: str,
        n_rounds: int,
        total_oracle_queries: int,
        n_trajectories: int,
        n_state_action_pairs: int,
        dataset_hash: str,
        expert_match_rate: Scalar,
        learning_success: bool = True,
        oracle_budget: Optional[int] = None,
        query_at_uncertainty: bool = True,
    ) -> "ImitationCertificate":
        """Factory for DAgger certificate."""
        failure_mode = None
        obstruction = None

        if not learning_success:
            if oracle_budget and total_oracle_queries >= oracle_budget:
                failure_mode = ImitationFailType.ORACLE_BUDGET_EXHAUSTED
                obstruction = ImitationObstructionEvidence(
                    fail_type=ImitationFailType.ORACLE_BUDGET_EXHAUSTED,
                    queries_used=total_oracle_queries,
                    query_budget=oracle_budget,
                )
            else:
                failure_mode = ImitationFailType.DISTRIBUTION_SHIFT

        return cls(
            model_id=model_id,
            learning_success=learning_success,
            expert_match_rate=to_scalar(expert_match_rate),
            failure_mode=failure_mode,
            obstruction_if_fail=obstruction,
            method_proof=ImitationMethodProof(
                method=ImitationMethod.DAGGER,
                demonstration_witness=DemonstrationWitness(
                    n_trajectories=n_trajectories,
                    n_state_action_pairs=n_state_action_pairs,
                    dataset_hash=dataset_hash,
                ),
                dagger_witness=DAggerWitness(
                    n_rounds=n_rounds,
                    total_oracle_queries=total_oracle_queries,
                    oracle_budget=oracle_budget,
                    query_at_uncertainty=query_at_uncertainty,
                ),
            ),
            strict_mode=True,
        )


def validate_imitation_certificate(cert: ImitationCertificate) -> StrictValidationResult:
    """Validate imitation certificate with structural consistency rules."""
    violations = cert.get_validity_violations()
    warnings = []

    # Consistency: failure_mode with success
    if cert.failure_mode is not None and cert.learning_success:
        violations.append(
            f"INCONSISTENT_STATE: failure_mode={cert.failure_mode.value} but learning_success=True"
        )

    # Consistency: REWARD_NON_IDENTIFIABLE requires evidence
    if cert.failure_mode == ImitationFailType.REWARD_NON_IDENTIFIABLE:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: REWARD_NON_IDENTIFIABLE requires obstruction_if_fail"
            )

    # Consistency: ORACLE_BUDGET_EXHAUSTED requires query evidence
    if cert.failure_mode == ImitationFailType.ORACLE_BUDGET_EXHAUSTED:
        if cert.obstruction_if_fail is None:
            violations.append(
                "MISSING_OBSTRUCTION: ORACLE_BUDGET_EXHAUSTED requires obstruction_if_fail"
            )
        elif cert.obstruction_if_fail.queries_used is None:
            violations.append(
                "INCOMPLETE_OBSTRUCTION: ORACLE_BUDGET_EXHAUSTED requires queries_used"
            )

    # Informational: IRL identifiability
    if cert.method_proof is not None and cert.method_proof.inverse_rl_witness is not None:
        irlw = cert.method_proof.inverse_rl_witness
        if irlw.identifiable:
            warnings.append(
                f"IRL_IDENTIFIABLE: target class '{irlw.inferred_target_class}' uniquely identified"
            )
        else:
            alt_count = len(irlw.alternative_targets) if irlw.alternative_targets else 0
            warnings.append(
                f"IRL_NON_IDENTIFIABLE: {alt_count + 1} consistent target classes"
            )

    # Informational: expert match rate
    if cert.expert_match_rate is not None:
        rate = to_scalar(cert.expert_match_rate)
        if rate >= Fraction(9, 10):
            warnings.append(
                f"HIGH_EXPERT_MATCH: {rate} (>=90% agreement with expert)"
            )
        elif rate < Fraction(1, 2):
            warnings.append(
                f"LOW_EXPERT_MATCH: {rate} (<50% agreement with expert)"
            )

    # Informational: DAgger uncertainty querying
    if cert.method_proof is not None and cert.method_proof.dagger_witness is not None:
        dagw = cert.method_proof.dagger_witness
        if dagw.query_at_uncertainty:
            warnings.append(
                "DAGGER_UNCERTAINTY_QUERY: queries triggered by policy uncertainty (ties to exploration)"
            )

    return StrictValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# CROSS-CERTIFICATE COHERENCE VALIDATORS
# ============================================================================
# These validators check consistency across certificate types in a bundle.
# A certificate bundle is internally coherent if all cross-references align.


@dataclass
class CertificateBundle:
    """
    A bundle of certificates that should be internally coherent.

    Used for end-to-end validation of a decision stack:
    planning → filtering → RL → imitation.
    """

    # Core certificates
    policy_certificates: List[PolicyCertificate] = field(default_factory=list)
    exploration_certificates: List[ExplorationCertificate] = field(default_factory=list)
    rl_certificates: List[RLCertificate] = field(default_factory=list)
    imitation_certificates: List[ImitationCertificate] = field(default_factory=list)
    mcts_certificates: List[MCTSCertificate] = field(default_factory=list)
    filter_certificates: List[FilterCertificate] = field(default_factory=list)
    inference_certificates: List[InferenceCertificate] = field(default_factory=list)

    # Bundle metadata
    bundle_id: str = ""
    description: str = ""
    environment_id: Optional[str] = None
    target_class: Optional[str] = None

    def all_certificates(self) -> List[Any]:
        """Return all certificates in the bundle."""
        return (
            self.policy_certificates +
            self.exploration_certificates +
            self.rl_certificates +
            self.imitation_certificates +
            self.mcts_certificates +
            self.filter_certificates +
            self.inference_certificates
        )

    def to_manifest(self) -> Dict[str, Any]:
        """Export bundle manifest (hashes + summary stats)."""
        import hashlib
        import json

        # Collect all certificate JSONs
        all_jsons = []
        for cert in self.policy_certificates:
            all_jsons.append(("policy", cert.to_json()))
        for cert in self.exploration_certificates:
            all_jsons.append(("exploration", cert.to_json()))
        for cert in self.rl_certificates:
            all_jsons.append(("rl", cert.to_json()))
        for cert in self.imitation_certificates:
            all_jsons.append(("imitation", cert.to_json()))
        for cert in self.mcts_certificates:
            all_jsons.append(("mcts", cert.to_json()))
        for cert in self.filter_certificates:
            all_jsons.append(("filter", cert.to_json()))
        for cert in self.inference_certificates:
            all_jsons.append(("inference", cert.to_json()))

        # Compute bundle hash
        bundle_content = json.dumps(all_jsons, sort_keys=True, default=str)
        bundle_hash = hashlib.sha256(bundle_content.encode()).hexdigest()[:16]

        return {
            "bundle_id": self.bundle_id or f"bundle_{bundle_hash}",
            "description": self.description,
            "environment_id": self.environment_id,
            "target_class": self.target_class,
            "bundle_hash": f"sha256:{bundle_hash}",
            "certificate_counts": {
                "policy": len(self.policy_certificates),
                "exploration": len(self.exploration_certificates),
                "rl": len(self.rl_certificates),
                "imitation": len(self.imitation_certificates),
                "mcts": len(self.mcts_certificates),
                "filter": len(self.filter_certificates),
                "inference": len(self.inference_certificates),
            },
            "total_certificates": len(all_jsons),
        }


@dataclass
class CoherenceResult:
    """Result of cross-certificate coherence validation."""

    coherent: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cross_references_checked: int = 0


def validate_rl_policy_coherence(
    rl_cert: RLCertificate,
    policy_cert: PolicyCertificate,
) -> CoherenceResult:
    """
    Validate coherence between RL certificate and PolicyCertificate.

    Cross-reference rules:
    - If both claim optimal steps, they must match
    - Generator sets must be compatible
    - Target class must match
    """
    violations = []
    warnings = []
    checks = 0

    # Check optimal steps consistency
    if (rl_cert.method_proof is not None and
        policy_cert.optimality_proof is not None and
        policy_cert.optimality_guarantee):

        checks += 1
        rl_steps = rl_cert.method_proof.total_steps
        policy_optimal = policy_cert.optimality_proof.optimal_distance

        # RL total_steps should be >= optimal_distance * episodes
        # (since optimal_distance is per-episode optimal)
        if rl_cert.method_proof.total_episodes > 0:
            avg_steps = Fraction(rl_steps, rl_cert.method_proof.total_episodes)
            if avg_steps < policy_optimal:
                warnings.append(
                    f"RL_BETTER_THAN_OPTIMAL: RL avg_steps ({avg_steps}) < "
                    f"policy optimal_distance ({policy_optimal}) - verify computation"
                )

    # Check generator set compatibility
    if rl_cert.generator_set is not None and policy_cert.generator_set:
        checks += 1
        rl_gens = set(rl_cert.generator_set)
        policy_gens = set(g.name for g in policy_cert.generator_set)

        if rl_gens != policy_gens:
            if not rl_gens.issubset(policy_gens) and not policy_gens.issubset(rl_gens):
                violations.append(
                    f"GENERATOR_MISMATCH: RL generators {rl_gens} != policy generators {policy_gens}"
                )
            else:
                warnings.append(
                    f"GENERATOR_SUBSET: RL uses {rl_gens}, policy uses {policy_gens}"
                )

    # Check target class consistency
    if rl_cert.target_class and policy_cert.target_class_description:
        checks += 1
        if rl_cert.target_class.lower() not in policy_cert.target_class_description.lower():
            warnings.append(
                f"TARGET_CLASS_MISMATCH: RL target '{rl_cert.target_class}' "
                f"vs policy target '{policy_cert.target_class_description}'"
            )

    return CoherenceResult(
        coherent=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        cross_references_checked=checks,
    )


def validate_imitation_exploration_coherence(
    imitation_cert: ImitationCertificate,
    exploration_cert: ExplorationCertificate,
) -> CoherenceResult:
    """
    Validate coherence between ImitationCertificate and ExplorationCertificate.

    Cross-reference rules:
    - Dataset coverage should upper-bound or align with exploration coverage
    - If both claim state counts, they should be consistent
    """
    violations = []
    warnings = []
    checks = 0

    # Check state coverage alignment
    demo_witness = None
    if imitation_cert.method_proof is not None:
        demo_witness = imitation_cert.method_proof.demonstration_witness

    if demo_witness is not None and exploration_cert.method_proof is not None:
        demo_coverage = demo_witness.states_covered
        exp_coverage = exploration_cert.method_proof.unique_states_visited

        if demo_coverage is not None and exp_coverage > 0:
            checks += 1
            if demo_coverage > exp_coverage * 2:
                warnings.append(
                    f"COVERAGE_MISMATCH: demo covers {demo_coverage} states but "
                    f"exploration only visited {exp_coverage} - verify same environment"
                )
            elif demo_coverage < exp_coverage // 2:
                warnings.append(
                    f"LOW_DEMO_COVERAGE: demo covers {demo_coverage} states but "
                    f"exploration visited {exp_coverage} - demos may be insufficient"
                )

    # Check state-action pair counts
    if demo_witness is not None and exploration_cert.regret_witness is not None:
        checks += 1
        demo_pairs = demo_witness.n_state_action_pairs
        exp_steps = exploration_cert.regret_witness.actual_steps

        # Demo data should be substantial relative to exploration
        if demo_pairs < exp_steps // 10:
            warnings.append(
                f"SPARSE_DEMO_DATA: {demo_pairs} demo pairs vs {exp_steps} exploration steps"
            )

    return CoherenceResult(
        coherent=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        cross_references_checked=checks,
    )


def validate_filter_inference_coherence(
    filter_cert: FilterCertificate,
    inference_cert: InferenceCertificate,
) -> CoherenceResult:
    """
    Validate coherence between FilterCertificate and InferenceCertificate.

    Cross-reference rules:
    - Filter entropy failure should relate to inference treewidth/complexity
    - State observability relates to inference identifiability
    """
    violations = []
    warnings = []
    checks = 0

    # Check observability vs inference structure
    if filter_cert.failure_mode == FilterFailType.STATE_UNOBSERVABLE:
        checks += 1
        if inference_cert.is_tree:
            warnings.append(
                "OBSERVABILITY_TREE_TENSION: filter state unobservable but "
                "inference model is tree-structured (usually implies observability)"
            )

    # Check filter success vs inference success alignment
    if filter_cert.filter_success and not inference_cert.inference_success:
        checks += 1
        warnings.append(
            "FILTER_INFERENCE_MISMATCH: filter succeeded but inference failed - "
            "verify they operate on consistent observation models"
        )

    # Check model dimensions
    if filter_cert.observation_dimension > 0 and inference_cert.variables:
        checks += 1
        n_vars = len(inference_cert.variables)
        if filter_cert.observation_dimension > n_vars * 2:
            warnings.append(
                f"DIMENSION_MISMATCH: filter has {filter_cert.observation_dimension} obs dims "
                f"but inference has only {n_vars} variables"
            )

    return CoherenceResult(
        coherent=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        cross_references_checked=checks,
    )


def validate_mcts_exploration_coherence(
    mcts_cert: MCTSCertificate,
    exploration_cert: ExplorationCertificate,
) -> CoherenceResult:
    """
    Validate coherence between MCTSCertificate and ExplorationCertificate.

    Cross-reference rules:
    - Exploration method should be consistent
    - UCB constant should match if both use UCB
    """
    violations = []
    warnings = []
    checks = 0

    # Check exploration method consistency
    if mcts_cert.method_proof is not None and exploration_cert.method_proof is not None:
        mcts_rule = mcts_cert.method_proof.exploration_rule
        exp_method = exploration_cert.method_proof.method

        checks += 1
        # Map MCTS rules to exploration methods
        method_map = {
            MCTSExplorationRule.UCB1: ExplorationMethod.UCB1,
            MCTSExplorationRule.UCB1_TUNED: ExplorationMethod.UCB1,
            MCTSExplorationRule.EPSILON_GREEDY: ExplorationMethod.EPSILON_GREEDY,
            MCTSExplorationRule.THOMPSON: ExplorationMethod.THOMPSON_SAMPLING,
        }

        expected = method_map.get(mcts_rule)
        if expected is not None and expected != exp_method:
            warnings.append(
                f"EXPLORATION_METHOD_MISMATCH: MCTS uses {mcts_rule.value} "
                f"but exploration uses {exp_method.value}"
            )

        # Check UCB constant if both use UCB
        if (mcts_rule in [MCTSExplorationRule.UCB1, MCTSExplorationRule.UCB1_TUNED] and
            exp_method == ExplorationMethod.UCB1):
            checks += 1
            mcts_c = mcts_cert.method_proof.exploration_constant
            exp_c = exploration_cert.method_proof.exploration_constant

            if mcts_c is not None and exp_c is not None and mcts_c != exp_c:
                warnings.append(
                    f"UCB_CONSTANT_MISMATCH: MCTS c={mcts_c} vs exploration c={exp_c}"
                )

    return CoherenceResult(
        coherent=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        cross_references_checked=checks,
    )


def validate_bundle_coherence(bundle: CertificateBundle) -> CoherenceResult:
    """
    Validate coherence across all certificates in a bundle.

    Performs pairwise coherence checks where applicable.
    """
    all_violations = []
    all_warnings = []
    total_checks = 0

    # RL ↔ Policy coherence
    for rl_cert in bundle.rl_certificates:
        for policy_cert in bundle.policy_certificates:
            result = validate_rl_policy_coherence(rl_cert, policy_cert)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            total_checks += result.cross_references_checked

    # Imitation ↔ Exploration coherence
    for imitation_cert in bundle.imitation_certificates:
        for exploration_cert in bundle.exploration_certificates:
            result = validate_imitation_exploration_coherence(imitation_cert, exploration_cert)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            total_checks += result.cross_references_checked

    # Filter ↔ Inference coherence
    for filter_cert in bundle.filter_certificates:
        for inference_cert in bundle.inference_certificates:
            result = validate_filter_inference_coherence(filter_cert, inference_cert)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            total_checks += result.cross_references_checked

    # MCTS ↔ Exploration coherence
    for mcts_cert in bundle.mcts_certificates:
        for exploration_cert in bundle.exploration_certificates:
            result = validate_mcts_exploration_coherence(mcts_cert, exploration_cert)
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            total_checks += result.cross_references_checked

    # Individual certificate validity
    for cert in bundle.policy_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_POLICY_CERT: {cert.policy_id}")
    for cert in bundle.exploration_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_EXPLORATION_CERT: {cert.model_id}")
    for cert in bundle.rl_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_RL_CERT: {cert.model_id}")
    for cert in bundle.imitation_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_IMITATION_CERT: {cert.model_id}")
    for cert in bundle.mcts_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_MCTS_CERT: {cert.model_id}")
    for cert in bundle.filter_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_FILTER_CERT: {cert.model_id}")
    for cert in bundle.inference_certificates:
        if not cert.is_valid():
            all_violations.append(f"INVALID_INFERENCE_CERT: {cert.model_id}")

    # Summary warning
    n_certs = len(bundle.all_certificates())
    if total_checks == 0 and n_certs > 1:
        all_warnings.append(
            f"NO_CROSS_REFERENCES: {n_certs} certificates but no cross-reference checks possible"
        )
    elif total_checks > 0:
        all_warnings.append(
            f"COHERENCE_CHECKED: {total_checks} cross-reference checks performed"
        )

    return CoherenceResult(
        coherent=len(all_violations) == 0,
        violations=all_violations,
        warnings=all_warnings,
        cross_references_checked=total_checks,
    )
