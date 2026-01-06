# qa_physics/projection/qa_observer.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
import math


# ----------------------------
# Core data containers
# ----------------------------

JSONDict = Dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """
    A projection output. Immutable, JSON-serializable.
    - observables: numeric + structured outputs (angles, distances, events, etc.)
    - units: explicit unit tags (observer-defined; may be dimensionless)
    - metadata: projection-specific info (e.g., which invariants used)
    """
    observables: JSONDict
    units: Dict[str, str] = field(default_factory=dict)
    metadata: JSONDict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProjectionReport:
    """
    A structured report about a projection's claimed properties and validations.
    Treat this as your 'observer compliance' record.
    """
    observer_name: str
    version: str
    preserves_symmetry: bool
    preserves_topology: bool
    preserves_failure_semantics: bool
    time_model: str
    unit_system: str
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TransitionLogEvent:
    """
    Deterministic log record for projection-aware runs.
    Mirrors your QARM logging spirit:
      {"move","fail_type","invariant_diff"} plus projection hooks.
    """
    step: int
    move: str
    src_state_id: str
    dst_state_id: str
    is_legal: bool
    fail_type: Optional[str]              # e.g., "OOB", "PARITY", "INVARIANT", ...
    invariant_diff: Optional[JSONDict]    # observer-independent diff summary if available
    obs_diff: Optional[JSONDict]          # observer-dependent diff summary
    meta: JSONDict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


# ----------------------------
# Observer interface
# ----------------------------

class QAObserver(ABC):
    """
    QAObserver is the *physics interface*:
    it defines how discrete QA states/paths yield continuous (or other) observables.

    Key commitments:
    - QA time is discrete (path length); continuous time can ONLY appear here.
    - Projection must be deterministic and explicit.
    - Projection must declare what it preserves and what it destroys.
    """

    name: str = "QAObserver"
    version: str = "0.0"

    # ----- Required declarations -----

    @abstractmethod
    def unit_system(self) -> str:
        """
        Human-readable unit system label. Example:
        - "dimensionless"
        - "observer_affine_seconds"
        - "scaled_length_units"
        """
        raise NotImplementedError

    @abstractmethod
    def time_model(self) -> str:
        """
        Human-readable time projection model label. Example:
        - "identity_on_N (no real time)"
        - "affine t = alpha*k + beta"
        - "phase-modulated affine"
        """
        raise NotImplementedError

    # ----- Core projection APIs -----

    @abstractmethod
    def project_state(self, qa_state: Any) -> Observation:
        """
        Map a QA state to observables.
        Must be deterministic, no randomness, no hidden caches that change outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def project_path(self, qa_path: List[Any]) -> Observation:
        """
        Map an ordered QA path to observables.
        MUST respect order (time as sequence).
        """
        raise NotImplementedError

    # ----- Time projection (PRIMARY) -----

    def qa_duration(self, qa_path: List[Any]) -> int:
        """
        QA duration = path length in edges = len(path)-1.
        This is Axiom T1 in operational form.
        """
        if len(qa_path) < 1:
            return 0
        return max(0, len(qa_path) - 1)

    @abstractmethod
    def project_time(self, k_edges: int, context: Optional[JSONDict] = None) -> float:
        """
        Introduce continuous time as an observer projection:
          M_time: k ∈ N -> t_obs ∈ R

        Theorem NT says this cannot be identity into a continuous embedding
        that preserves all legality/non-reduction/irreversibility properties.
        So this is intentionally an observer-dependent approximation layer.
        """
        raise NotImplementedError

    # ----- Preservation claims (must be explicit) -----

    @abstractmethod
    def preserves_symmetry(self) -> bool:
        """
        Should scaling symmetries (e.g., λ) preserve the measured law?
        For reflection, angles should be scale-invariant.
        """
        raise NotImplementedError

    @abstractmethod
    def preserves_topology(self) -> bool:
        """
        Does the projection preserve SCC/component structure in any sense,
        or does it collapse distinct QA states to identical observations?
        (Collapse may be intentional; must be declared.)
        """
        raise NotImplementedError

    @abstractmethod
    def preserves_failure_semantics(self) -> bool:
        """
        Do QA failure types map to stable observational 'impossibility' classes?
        """
        raise NotImplementedError

    # ----- Validation & reporting -----

    def report(self) -> ProjectionReport:
        return ProjectionReport(
            observer_name=self.name,
            version=self.version,
            preserves_symmetry=self.preserves_symmetry(),
            preserves_topology=self.preserves_topology(),
            preserves_failure_semantics=self.preserves_failure_semantics(),
            time_model=self.time_model(),
            unit_system=self.unit_system(),
            notes=[],
        )

    def validate(self) -> None:
        """
        Override for stronger, observer-specific checks.
        Base checks ensure the interface is self-consistent.
        """
        r = self.report()
        assert isinstance(r.observer_name, str) and r.observer_name, "observer_name missing"
        assert isinstance(r.version, str) and r.version, "version missing"
        assert isinstance(r.time_model, str) and r.time_model, "time_model missing"
        assert isinstance(r.unit_system, str) and r.unit_system, "unit_system missing"

    # ----- Optional helpers -----

    def obs_distance(self, obs_a: Observation, obs_b: Observation) -> float:
        """
        A generic (optional) distance between observations, for debugging.
        Default is 0 unless user overrides.
        """
        return 0.0


# ----------------------------
# Example observers (v0)
# ----------------------------

class NullObserver(QAObserver):
    """
    Intentionally naive projection:
    - time is just k cast to float
    - observables are mostly raw invariants/state fields
    Expected to FAIL to reproduce classical laws robustly.
    Useful as a null model.
    """
    name = "NullObserver"
    version = "0.1"

    def unit_system(self) -> str:
        return "dimensionless"

    def time_model(self) -> str:
        return "t = float(k)  (null model)"

    def project_state(self, qa_state: Any) -> Observation:
        # Assume qa_state can provide a stable dict; otherwise repr() fallback.
        if hasattr(qa_state, "to_dict"):
            d = qa_state.to_dict()
        elif isinstance(qa_state, dict):
            d = dict(qa_state)
        else:
            d = {"repr": repr(qa_state)}
        return Observation(
            observables={"state": d},
            units={},
            metadata={"projection": "null_raw"},
        )

    def project_path(self, qa_path: List[Any]) -> Observation:
        k = self.qa_duration(qa_path)
        return Observation(
            observables={
                "k_edges": k,
                "t_obs": self.project_time(k),
                "path_len_nodes": len(qa_path),
            },
            units={"t_obs": "arb"},
            metadata={"projection": "null_path"},
        )

    def project_time(self, k_edges: int, context: Optional[JSONDict] = None) -> float:
        return float(k_edges)

    def preserves_symmetry(self) -> bool:
        return False  # null model makes no guarantee

    def preserves_topology(self) -> bool:
        return False  # likely collapses many states

    def preserves_failure_semantics(self) -> bool:
        return False  # no mapping defined


class AffineTimeGeometricObserver(QAObserver):
    """
    A minimal "physics-like" observer:
    - continuous time t = alpha*k + beta
    - expects to compute a small set of geometric observables from invariants
      (you will wire the exact invariants once your geometry objects exist)
    """
    name = "AffineTimeGeometricObserver"
    version = "0.1"

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def unit_system(self) -> str:
        return "observer_affine_time + dimensionless_angles"

    def time_model(self) -> str:
        # FIXED: Include "affine" keyword so test can detect it
        return f"affine: t = {self.alpha}*k + {self.beta}"

    def project_time(self, k_edges: int, context: Optional[JSONDict] = None) -> float:
        # Deterministic, monotone if alpha>0
        return self.alpha * float(k_edges) + self.beta

    def project_state(self, qa_state: Any) -> Observation:
        """
        Placeholder logic: we keep this minimal and explicit.
        In your reflection demo, qa_state will likely include:
          - points (S, M, T), mirror geometry
          - derived invariants (quadrances, spreads) in exact form
        """
        d: JSONDict
        if hasattr(qa_state, "to_invariants"):
            d = qa_state.to_invariants()  # should be exact rationals/ints
        elif hasattr(qa_state, "to_dict"):
            d = qa_state.to_dict()
        elif isinstance(qa_state, dict):
            d = dict(qa_state)
        else:
            d = {"repr": repr(qa_state)}

        # Example: if state exposes dimensionless "spread" quantities, keep them.
        observables: JSONDict = {"invariants": d}

        return Observation(
            observables=observables,
            units={},  # invariants should carry their own meaning; angles dimensionless
            metadata={"projection": "affine_time_geometric"},
        )

    def project_path(self, qa_path: List[Any]) -> Observation:
        k = self.qa_duration(qa_path)
        t = self.project_time(k)
        return Observation(
            observables={
                "k_edges": k,
                "t_obs": t,
                "path_len_nodes": len(qa_path),
            },
            units={"t_obs": "arb_seconds"},
            metadata={"projection": "affine_time_geometric_path"},
        )

    def preserves_symmetry(self) -> bool:
        # Goal: geometric observables like angles are scale-invariant under λ
        return True

    def preserves_topology(self) -> bool:
        # Many-to-one is allowed, but for v0 we claim we do not *intentionally* collapse.
        return False  # be conservative until proven

    def preserves_failure_semantics(self) -> bool:
        # Once we map fail types -> observational impossibility categories, this can become True.
        return False

    def validate(self) -> None:
        super().validate()
        assert math.isfinite(self.alpha), "alpha must be finite"
        assert math.isfinite(self.beta), "beta must be finite"
        # Optional: enforce monotonicity if you want clock-like behavior
        # assert self.alpha > 0, "alpha must be > 0 for monotone clock projection"
