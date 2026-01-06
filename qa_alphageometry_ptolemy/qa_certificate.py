
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

    # Physics/projection-specific failures
    OBSERVER_UNDEFINED = "observer_undefined"
    LAW_VIOLATION = "law_violation"
    PROJECTION_INCOMPATIBLE = "projection_incompatible"

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
            "fixed_q_mode": {k: str(to_scalar(v)) for k, v in (self.fixed_q_mode or {}).items()},
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
