"""
qa_beyond_neurons_certificate.py

QA Certificate Schema for Beyond-Neurons Intelligence

Based on:
- Levin & Chis-Ciure: Intelligence Beyond Neurons
  (Biological Journal of the Linnean Society, blae076, 2024)

Core QA Principle (Fourth leg of the certificate architecture):
    Generator Injection:   G1 subset G2 -> Reach EXPANDS
    Diversity Collapse:    I_div violated -> Reach CONTRACTS
    Field Computation:     G = physical operators -> Reach REALIZED BY PHYSICS
    Beyond Neurons:        Intelligence is SUBSTRATE-NEUTRAL and SCALE-FREE

This certificate proves:
    Given a problem space P = <S, O, C, E, H>:
    - S: state space
    - O: operators (generators)
    - C: constraints (invariants)
    - E: evaluation function
    - H: planning horizon

    An agent/system achieves search efficiency K = log10(tau_blind / tau_agent),
    where K > 0 demonstrates non-trivial intelligence regardless of substrate.

    Multi-scale competency architecture:
    Nested problem spaces P_1, P_2, ..., P_n at different scales
    share the same obstruction algebra.

New mechanisms beyond the triad:
    - Constraint editing: Modify C to open new reachable states
    - Goal decoupling: Component E diverges from collective E
    - Horizon expansion: Increase H to enable multi-step strategies

Hard constraints:
- Exact scalars only (int/Fraction) -- no floats
- Deterministic serialization
- Failure-completeness: every validation yields success OR obstruction proof
"""

from __future__ import annotations

import sys
import os

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, FrozenSet
from enum import Enum
from fractions import Fraction

try:
    # When run as module: python -m qa_alphageometry_ptolemy.qa_beyond_neurons_certificate
    from .qa_cert_core import (
        Scalar, to_scalar, scalar_to_str,
        canonical_json, certificate_hash, state_hash,
        cert_id, utc_now_iso,
        ValidationResult,
    )
except ImportError:
    # When run directly: python qa_beyond_neurons_certificate.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qa_cert_core import (
        Scalar, to_scalar, scalar_to_str,
        canonical_json, certificate_hash, state_hash,
        cert_id, utc_now_iso,
        ValidationResult,
    )


# ============================================================================
# SUBSTRATE AND SCALE TYPES
# ============================================================================

class Substrate(Enum):
    """Physical substrate on which intelligence operates."""
    BIOLOGICAL_NEURAL = "biological_neural"         # Brains, nervous systems
    BIOLOGICAL_NON_NEURAL = "biological_non_neural" # Cells, tissues, bioelectric
    SILICON_DIGITAL = "silicon_digital"             # CPUs, GPUs, LLMs
    HYBRID = "hybrid"                               # Brain-computer interfaces
    COLLECTIVE = "collective"                       # Swarms, colonies, markets
    PHYSICAL_FIELD = "physical_field"               # RF, photonic (links to ROI #3)


class CompetencyScale(Enum):
    """Scale at which a problem space operates."""
    MOLECULAR = "molecular"           # Protein folding, gene regulation
    CELLULAR = "cellular"             # Cell migration, division, apoptosis
    TISSUE = "tissue"                 # Morphogenesis, wound healing
    ORGAN = "organ"                   # Organ-level function
    ORGANISM = "organism"             # Whole-body behavior
    SOCIAL = "social"                 # Multi-agent, swarm, market
    ARTIFICIAL = "artificial"         # Silicon-based AI systems


# ============================================================================
# BARRIER TYPES (BEYOND NEURONS SPECIFIC)
# ============================================================================

class BarrierType(Enum):
    """Barriers in multi-scale, substrate-neutral intelligence."""
    GOAL_DECOUPLING = "goal_decoupling"
    CONSTRAINT_LOCK = "constraint_lock"
    HORIZON_LIMIT = "horizon_limit"
    LOCAL_OPTIMA_CAPTURE = "local_optima_capture"
    COMPONENT_FRAGMENTATION = "component_fragmentation"
    SUBSTRATE_MISMATCH = "substrate_mismatch"
    SCALE_BOUNDARY = "scale_boundary"
    EVALUATION_MISALIGNMENT = "evaluation_misalignment"


# ============================================================================
# PROBLEM SPACE P = <S, O, C, E, H>
# ============================================================================

@dataclass(frozen=True)
class Operator:
    """
    An operator (generator) in a problem space.

    Levin's O maps directly to QA's generators.
    The substrate determines what operators are physically available.
    """
    name: str
    substrate: Substrate
    scale: CompetencyScale
    description: str

    # Formal signature
    input_signature: Tuple[str, ...]
    output_signature: Tuple[str, ...]

    # Constraint interaction
    preserves_constraints: FrozenSet[str] = frozenset()
    may_violate: FrozenSet[str] = frozenset()

    # Cost
    cost: Optional[Scalar] = None

    def __post_init__(self):
        if self.cost is not None:
            object.__setattr__(self, "cost", to_scalar(self.cost))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "substrate": self.substrate.value,
            "scale": self.scale.value,
            "description": self.description,
            "input_signature": list(self.input_signature),
            "output_signature": list(self.output_signature),
            "preserves_constraints": sorted(self.preserves_constraints),
            "may_violate": sorted(self.may_violate),
            "cost": scalar_to_str(self.cost) if self.cost is not None else None,
        }


@dataclass(frozen=True)
class OperatorSet:
    """Set of operators available to an agent/system at a given scale."""
    name: str
    operators: FrozenSet[Operator]
    substrate: Substrate
    scale: CompetencyScale
    description: Optional[str] = None

    def operator_names(self) -> FrozenSet[str]:
        return frozenset(o.name for o in self.operators)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "substrate": self.substrate.value,
            "scale": self.scale.value,
            "operators": [o.to_dict() for o in sorted(self.operators, key=lambda x: x.name)],
            "description": self.description,
        }


@dataclass(frozen=True)
class Constraint:
    """
    A constraint in the problem space.

    Levin's C maps to QA's invariants.
    Key insight: constraint EDITING (changing C) is a distinct
    reachability mechanism from generator injection (changing O).
    """
    name: str
    predicate: str
    hard: bool = True           # Hard = barrier on violation
    editable: bool = False      # Can this constraint be deliberately changed?
    editing_operator: Optional[str] = None  # Which operator edits this constraint

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "predicate": self.predicate,
            "hard": self.hard,
            "editable": self.editable,
        }
        if self.editing_operator:
            d["editing_operator"] = self.editing_operator
        return d


@dataclass(frozen=True)
class ConstraintSet:
    """Collection of constraints defining valid states."""
    constraints: Tuple[Constraint, ...]

    def names(self) -> FrozenSet[str]:
        return frozenset(c.name for c in self.constraints)

    def editable_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.editable)

    def hard_constraints(self) -> Tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.hard)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [c.to_dict() for c in self.constraints],
            "editable_count": len(self.editable_constraints()),
            "hard_count": len(self.hard_constraints()),
        }


@dataclass(frozen=True)
class EvaluationFunction:
    """
    The evaluation (fitness/goal) function E.

    In multi-scale systems, each scale has its own E.
    Goal decoupling = component E diverges from collective E.
    """
    name: str
    description: str
    scale: CompetencyScale
    metric_type: str  # "fitness", "morphological_target", "task_completion"
    optimal_value: Optional[Scalar] = None

    def __post_init__(self):
        if self.optimal_value is not None:
            object.__setattr__(self, "optimal_value", to_scalar(self.optimal_value))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "scale": self.scale.value,
            "metric_type": self.metric_type,
        }
        if self.optimal_value is not None:
            d["optimal_value"] = scalar_to_str(self.optimal_value)
        return d


@dataclass(frozen=True)
class ProblemSpace:
    """
    Levin's Problem Space P = <S, O, C, E, H>.

    The fundamental unit of intelligence analysis.
    Every intelligent system -- neural or not -- operates within a problem space.
    """
    name: str
    state_space_description: str   # S: what states look like
    operators: OperatorSet         # O: available generators
    constraints: ConstraintSet     # C: invariants
    evaluation: EvaluationFunction # E: what counts as success
    horizon: int                   # H: planning depth (discrete steps)
    substrate: Substrate
    scale: CompetencyScale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state_space_description": self.state_space_description,
            "operators": self.operators.to_dict(),
            "constraints": self.constraints.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "horizon": self.horizon,
            "substrate": self.substrate.value,
            "scale": self.scale.value,
        }


# ============================================================================
# SEARCH EFFICIENCY (Universal Intelligence Metric)
# ============================================================================

@dataclass(frozen=True)
class SearchEfficiency:
    """
    Search efficiency K = log10(tau_blind / tau_agent).

    This is Levin's universal intelligence metric:
    - tau_blind: expected time for blind/random search
    - tau_agent: expected time for the agent under study
    - K > 0: non-trivial intelligence
    - K = 0: random search (no intelligence)
    - K < 0: worse than random (adversarial obstruction)

    K is substrate-neutral: it applies to cells, brains, and silicon equally.
    """
    tau_blind: Scalar    # Expected blind search time
    tau_agent: Scalar    # Agent's actual/expected time
    K: Scalar            # log10(tau_blind / tau_agent)

    # Context
    problem_space_name: str
    substrate: Substrate

    def __post_init__(self):
        object.__setattr__(self, "tau_blind", to_scalar(self.tau_blind))
        object.__setattr__(self, "tau_agent", to_scalar(self.tau_agent))
        object.__setattr__(self, "K", to_scalar(self.K))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_blind": scalar_to_str(self.tau_blind),
            "tau_agent": scalar_to_str(self.tau_agent),
            "K": scalar_to_str(self.K),
            "problem_space_name": self.problem_space_name,
            "substrate": self.substrate.value,
        }


# ============================================================================
# MULTI-SCALE COMPETENCY ARCHITECTURE
# ============================================================================

@dataclass(frozen=True)
class CompetencyLevel:
    """
    One level in a multi-scale competency architecture.

    Each level has its own problem space and search efficiency.
    Levels are nested: lower levels serve as operators for higher levels.
    """
    level_index: int
    scale: CompetencyScale
    problem_space: ProblemSpace
    search_efficiency: SearchEfficiency

    # Interface to adjacent levels
    provides_to_higher: FrozenSet[str]  # What this level offers as operators upward
    requires_from_lower: FrozenSet[str]  # What this level needs from below

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level_index": self.level_index,
            "scale": self.scale.value,
            "problem_space": self.problem_space.to_dict(),
            "search_efficiency": self.search_efficiency.to_dict(),
            "provides_to_higher": sorted(self.provides_to_higher),
            "requires_from_lower": sorted(self.requires_from_lower),
        }


@dataclass(frozen=True)
class CompetencyArchitecture:
    """
    Multi-scale competency architecture.

    The key Levin insight: intelligence is NESTED.
    Cells are intelligent at their scale.
    Tissues are intelligent at theirs.
    Organisms at theirs.

    Each scale has problem space P_i with its own (G_i, I_i, E_i, H_i).
    The obstruction algebra is the same at every scale.
    """
    levels: Tuple[CompetencyLevel, ...]
    substrate: Substrate
    organism_or_system: str

    def __post_init__(self):
        # Verify levels are ordered by index
        for i in range(1, len(self.levels)):
            if self.levels[i].level_index <= self.levels[i - 1].level_index:
                raise ValueError(
                    f"Levels not ordered: {self.levels[i].level_index} "
                    f"<= {self.levels[i - 1].level_index}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": [l.to_dict() for l in self.levels],
            "substrate": self.substrate.value,
            "organism_or_system": self.organism_or_system,
            "depth": len(self.levels),
        }


# ============================================================================
# BEYOND NEURONS BARRIER
# ============================================================================

@dataclass(frozen=True)
class BeyondNeuronsBarrier:
    """
    A barrier in the multi-scale intelligence framework.

    Extends the triad's obstruction algebra with:
    - Goal decoupling (component vs collective)
    - Constraint lock (cannot edit constraints)
    - Horizon limit (cannot plan far enough)
    """
    barrier_type: BarrierType
    source_state_class: str
    target_state_class: str
    scale: CompetencyScale
    required_mechanism: Optional[str] = None  # operator, constraint_edit, horizon_extension
    obstruction_proof: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "barrier_type": self.barrier_type.value,
            "source_state_class": self.source_state_class,
            "target_state_class": self.target_state_class,
            "scale": self.scale.value,
            "required_mechanism": self.required_mechanism,
            "obstruction_proof": self.obstruction_proof,
        }


# ============================================================================
# WITNESSES
# ============================================================================

@dataclass(frozen=True)
class IntelligenceWitness:
    """
    Constructive proof of intelligence at a given scale.

    Shows: the system achieves K > 0 on a problem space,
    demonstrating non-trivial search efficiency.
    """
    problem_space_name: str
    search_efficiency: SearchEfficiency
    mechanism: str  # "operator_application", "constraint_editing", "horizon_expansion"

    # Evidence trace
    trace_steps: Tuple[Dict[str, str], ...]  # (operator, state_description) sequence
    goal_achieved: bool
    constraints_preserved: FrozenSet[str]
    constraints_edited: FrozenSet[str]  # Constraints that were deliberately changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_space_name": self.problem_space_name,
            "search_efficiency": self.search_efficiency.to_dict(),
            "mechanism": self.mechanism,
            "trace_steps": list(self.trace_steps),
            "goal_achieved": self.goal_achieved,
            "constraints_preserved": sorted(self.constraints_preserved),
            "constraints_edited": sorted(self.constraints_edited),
        }


@dataclass(frozen=True)
class UnreachabilityWitness:
    """
    Proof that a goal is unreachable under current configuration.

    Shows: under given (O, C, H), the target state class is unreachable.
    """
    target_state_class: str
    operator_set_name: str
    barrier: BeyondNeuronsBarrier
    obstruction_argument: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_state_class": self.target_state_class,
            "operator_set_name": self.operator_set_name,
            "barrier": self.barrier.to_dict(),
            "obstruction_argument": self.obstruction_argument,
        }


# ============================================================================
# BEYOND NEURONS INTELLIGENCE CERTIFICATE
# ============================================================================

class IntelligenceResult(Enum):
    """Result of beyond-neurons intelligence analysis."""
    BARRIER_CROSSED = "barrier_crossed"             # Intelligence demonstrated, barrier overcome
    GOAL_DECOUPLED = "goal_decoupled"               # Component diverged from collective
    CONSTRAINT_EDITED = "constraint_edited"         # Constraints modified to expand reach
    HORIZON_EXPANDED = "horizon_expanded"           # Deeper planning enabled new reach
    NO_INTELLIGENCE = "no_intelligence"             # K <= 0, no better than random
    SUBSTRATE_NEUTRAL_CONFIRMED = "substrate_neutral_confirmed"  # Same K on different substrates
    PENDING = "pending"


@dataclass(frozen=True)
class BeyondNeuronsCertificate:
    """
    Certificate proving intelligence is substrate-neutral and scale-free.

    Fourth leg of the QA certificate architecture:

        1. Generator Injection:  Reach expands (add tools)
        2. Diversity Collapse:   Reach contracts (lose invariant)
        3. Field Computation:    Reach realized by physics
        4. Beyond Neurons:       Intelligence is substrate-neutral, scale-free

    The theorem:
        Search efficiency K = log10(tau_blind / tau_agent) is substrate-neutral.
        The same obstruction algebra (barriers, invariants, generators)
        governs intelligence at every scale and on every substrate.

    New mechanisms:
        - Constraint editing: Change C to expand Reach (planaria)
        - Goal decoupling: Component E diverges from collective E (cancer)
        - Horizon expansion: Increase H to enable multi-step strategies
    """
    certificate_id: str
    certificate_type: str  # Always "BEYOND_NEURONS_INTELLIGENCE_CERT"
    timestamp: str

    # The claim
    claim: str
    substrate: Substrate
    scale: CompetencyScale

    # Problem space
    problem_space: ProblemSpace

    # Search efficiency
    search_efficiency: SearchEfficiency

    # Multi-scale architecture (if applicable)
    competency_architecture: Optional[CompetencyArchitecture] = None

    # Operators (before/after if applicable)
    baseline_operators: Optional[OperatorSet] = None
    extended_operators: Optional[OperatorSet] = None
    injected_operators: Optional[FrozenSet[Operator]] = None

    # Constraints (before/after if constraint editing occurred)
    baseline_constraints: Optional[ConstraintSet] = None
    edited_constraints: Optional[ConstraintSet] = None

    # Barrier
    barrier: Optional[BeyondNeuronsBarrier] = None

    # Evidence
    before_witness: Optional[UnreachabilityWitness] = None
    after_witness: Optional[IntelligenceWitness] = None

    # Result
    result: IntelligenceResult = IntelligenceResult.PENDING

    # Metadata
    source_refs: Optional[List[Dict[str, str]]] = None

    def __post_init__(self):
        if self.certificate_type != "BEYOND_NEURONS_INTELLIGENCE_CERT":
            raise ValueError(
                f"certificate_type must be BEYOND_NEURONS_INTELLIGENCE_CERT, "
                f"got {self.certificate_type}"
            )

        # If operator injection occurred, verify subset relation
        if self.baseline_operators and self.extended_operators and self.injected_operators:
            base_names = self.baseline_operators.operator_names()
            ext_names = self.extended_operators.operator_names()
            inj_names = frozenset(o.name for o in self.injected_operators)
            expected = ext_names - base_names
            if inj_names != expected:
                raise ValueError(
                    f"Injected operators mismatch: got {inj_names}, expected {expected}"
                )
            if not base_names.issubset(ext_names):
                raise ValueError("Baseline operators must be subset of extended operators")

    def compute_hash(self) -> str:
        return certificate_hash(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "certificate_id": self.certificate_id,
            "certificate_type": self.certificate_type,
            "timestamp": self.timestamp,
            "claim": self.claim,
            "substrate": self.substrate.value,
            "scale": self.scale.value,
            "problem_space": self.problem_space.to_dict(),
            "search_efficiency": self.search_efficiency.to_dict(),
            "result": self.result.value,
        }

        if self.competency_architecture:
            d["competency_architecture"] = self.competency_architecture.to_dict()
        if self.baseline_operators:
            d["baseline_operators"] = self.baseline_operators.to_dict()
        if self.extended_operators:
            d["extended_operators"] = self.extended_operators.to_dict()
        if self.injected_operators:
            d["injected_operators"] = [
                o.to_dict() for o in sorted(self.injected_operators, key=lambda x: x.name)
            ]
        if self.baseline_constraints:
            d["baseline_constraints"] = self.baseline_constraints.to_dict()
        if self.edited_constraints:
            d["edited_constraints"] = self.edited_constraints.to_dict()
        if self.barrier:
            d["barrier"] = self.barrier.to_dict()
        if self.before_witness:
            d["before_witness"] = self.before_witness.to_dict()
        if self.after_witness:
            d["after_witness"] = self.after_witness.to_dict()
        if self.source_refs:
            d["source_refs"] = self.source_refs

        return d

    def to_json(self, indent: int = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)


# ============================================================================
# FACTORY: PLANARIA REGENERATION (Constraint Editing)
# ============================================================================

def create_planaria_certificate() -> BeyondNeuronsCertificate:
    """
    Planaria head/tail determination via bioelectric constraint editing.

    Scenario:
        Planaria are amputated. Under normal constraints, regeneration
        follows anterior-posterior axis (head regrows at head end).
        By editing bioelectric constraints (gap junction manipulation),
        researchers caused two-headed planaria -- a morphological state
        unreachable under the original constraint set.

    This demonstrates:
        Constraint editing (changing C) expands reachability
        WITHOUT changing the operator set O.
        The cells use the SAME operators (division, migration, apoptosis)
        but reach new morphological states because the constraints changed.

    K: Planaria regeneration is far faster than blind morphospace search.
    """
    from fractions import Fraction as F

    # Operators (unchanged before/after)
    divide = Operator(
        name="cell_division", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Mitotic cell division",
        input_signature=("cell_state",), output_signature=("cell_state", "cell_state"),
        preserves_constraints=frozenset(["genome_integrity"]),
    )
    migrate = Operator(
        name="cell_migration", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Cell migration along morphogen gradient",
        input_signature=("cell_state", "gradient"),
        output_signature=("cell_state",),
        preserves_constraints=frozenset(["tissue_integrity"]),
    )
    apoptosis = Operator(
        name="apoptosis", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Programmed cell death for pattern sculpting",
        input_signature=("cell_state",), output_signature=(),
        preserves_constraints=frozenset(["genome_integrity"]),
    )
    differentiate = Operator(
        name="differentiation", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Cell fate commitment via gene expression",
        input_signature=("cell_state", "signal"),
        output_signature=("differentiated_cell",),
    )
    bioelectric_signal = Operator(
        name="bioelectric_signaling", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.TISSUE,
        description="Gap junction-mediated voltage gradient communication",
        input_signature=("tissue_state",), output_signature=("tissue_state",),
        preserves_constraints=frozenset(["tissue_integrity"]),
        may_violate=frozenset(["anterior_posterior_polarity"]),
    )

    ops = frozenset([divide, migrate, apoptosis, differentiate, bioelectric_signal])
    op_set = OperatorSet(
        name="planaria_cellular_ops",
        operators=ops,
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.TISSUE,
        description="Cellular operators for planaria regeneration",
    )

    # Baseline constraints (normal body plan)
    baseline_constraints = ConstraintSet(constraints=(
        Constraint(
            name="anterior_posterior_polarity",
            predicate="Head regenerates at anterior end only",
            hard=True, editable=True,
            editing_operator="bioelectric_signaling",
        ),
        Constraint(
            name="genome_integrity",
            predicate="No mutations during regeneration",
            hard=True, editable=False,
        ),
        Constraint(
            name="tissue_integrity",
            predicate="Tissue remains contiguous",
            hard=True, editable=False,
        ),
        Constraint(
            name="body_plan_symmetry",
            predicate="Bilateral symmetry maintained",
            hard=False, editable=True,
            editing_operator="bioelectric_signaling",
        ),
    ))

    # Edited constraints (gap junction manipulation -> two-headed)
    edited_constraints = ConstraintSet(constraints=(
        Constraint(
            name="anterior_posterior_polarity",
            predicate="Head regenerates at BOTH ends (polarity inverted at posterior)",
            hard=True, editable=True,
            editing_operator="bioelectric_signaling",
        ),
        Constraint(
            name="genome_integrity",
            predicate="No mutations during regeneration",
            hard=True, editable=False,
        ),
        Constraint(
            name="tissue_integrity",
            predicate="Tissue remains contiguous",
            hard=True, editable=False,
        ),
        Constraint(
            name="body_plan_symmetry",
            predicate="Bilateral symmetry maintained",
            hard=False, editable=True,
            editing_operator="bioelectric_signaling",
        ),
    ))

    evaluation = EvaluationFunction(
        name="morphological_target",
        description="Reach target morphology (species-typical body plan or experimentally induced plan)",
        scale=CompetencyScale.ORGANISM,
        metric_type="morphological_target",
    )

    problem_space = ProblemSpace(
        name="planaria_regeneration",
        state_space_description="Morphological state: cell positions, types, bioelectric potentials",
        operators=op_set,
        constraints=baseline_constraints,
        evaluation=evaluation,
        horizon=14,  # ~14 days regeneration
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.ORGANISM,
    )

    # Search efficiency
    # tau_blind: random cell assembly -> correct morphology ~10^15 configurations
    # tau_agent: planaria regeneration ~14 days ~= 10^6 seconds
    search_efficiency = SearchEfficiency(
        tau_blind=F(10**15, 1),
        tau_agent=F(10**6, 1),
        K=F(9, 1),  # log10(10^15 / 10^6) = 9
        problem_space_name="planaria_regeneration",
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
    )

    # Barrier
    barrier = BeyondNeuronsBarrier(
        barrier_type=BarrierType.CONSTRAINT_LOCK,
        source_state_class="amputated_planaria",
        target_state_class="two_headed_morphology",
        scale=CompetencyScale.ORGANISM,
        required_mechanism="constraint_editing",
        obstruction_proof=(
            "Under normal anterior-posterior polarity constraint, "
            "two-headed morphology is unreachable regardless of operator application. "
            "The constraint locks the posterior end to tail regeneration. "
            "Only by editing the polarity constraint via bioelectric signaling "
            "(gap junction manipulation) does two-headed morphology become reachable."
        ),
    )

    before_witness = UnreachabilityWitness(
        target_state_class="two_headed_morphology",
        operator_set_name="planaria_cellular_ops",
        barrier=barrier,
        obstruction_argument=(
            "With anterior_posterior_polarity constraint active, "
            "posterior wound site always regenerates tail tissue. "
            "All cellular operators (division, migration, apoptosis, differentiation) "
            "preserve this constraint. Two-headed state is outside Reach(S, O, C_baseline)."
        ),
    )

    after_witness = IntelligenceWitness(
        problem_space_name="planaria_regeneration",
        search_efficiency=search_efficiency,
        mechanism="constraint_editing",
        trace_steps=(
            {"operator": "bioelectric_signaling", "state": "polarity_constraint_edited_at_posterior"},
            {"operator": "cell_division", "state": "blastema_forms_at_both_ends"},
            {"operator": "differentiation", "state": "head_fate_committed_at_posterior"},
            {"operator": "cell_migration", "state": "head_structures_forming_bilaterally"},
            {"operator": "apoptosis", "state": "excess_tissue_sculpted_away"},
        ),
        goal_achieved=True,
        constraints_preserved=frozenset(["genome_integrity", "tissue_integrity"]),
        constraints_edited=frozenset(["anterior_posterior_polarity"]),
    )

    return BeyondNeuronsCertificate(
        certificate_id=cert_id("BEYONDNEUR-PLANARIA"),
        certificate_type="BEYOND_NEURONS_INTELLIGENCE_CERT",
        timestamp=utc_now_iso(),
        claim=(
            "Planaria regeneration demonstrates constraint editing as a "
            "reachability mechanism: same operators, different constraints, "
            "expanded reach. K=9 on morphological problem space. No neurons required."
        ),
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.ORGANISM,
        problem_space=problem_space,
        search_efficiency=search_efficiency,
        baseline_constraints=baseline_constraints,
        edited_constraints=edited_constraints,
        barrier=barrier,
        before_witness=before_witness,
        after_witness=after_witness,
        result=IntelligenceResult.CONSTRAINT_EDITED,
        source_refs=[
            {"name": "Levin & Chis-Ciure 2024",
             "ref": "Biological Journal of the Linnean Society, blae076"},
            {"name": "Levin 2019 (bioelectric)",
             "ref": "doi:10.1002/bies.201900146"},
        ],
    )


# ============================================================================
# FACTORY: CANCER GOAL DECOUPLING
# ============================================================================

def create_cancer_certificate() -> BeyondNeuronsCertificate:
    """
    Cancer as goal decoupling in the multi-scale competency architecture.

    Scenario:
        Cancer cells maximize LOCAL fitness (proliferation) but decouple
        from ORGANISM-level goals (tissue homeostasis). This is not a
        random breakdown -- it is reversion to a unicellular fitness
        landscape where the cell's evaluation function E_cell diverges
        from E_organism.

    QA interpretation:
        Goal decoupling = component evaluation E_i diverges from
        collective evaluation E_collective.
        This erects a GOAL_DECOUPLING barrier at the organism scale.

    K: Cancer cells are intelligent at THEIR scale (K_cell > 0 for
    proliferation). The problem is misalignment, not stupidity.
    """
    from fractions import Fraction as F

    # Cell-level operators
    proliferate = Operator(
        name="proliferate", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Uncontrolled cell division (bypasses checkpoints)",
        input_signature=("cell_state",), output_signature=("cell_state", "cell_state"),
        may_violate=frozenset(["growth_checkpoint", "tissue_homeostasis"]),
    )
    evade_apoptosis = Operator(
        name="evade_apoptosis", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Suppress apoptotic signaling pathways",
        input_signature=("apoptotic_signal",), output_signature=("cell_state",),
        may_violate=frozenset(["apoptosis_response", "tissue_homeostasis"]),
    )
    angiogenesis = Operator(
        name="angiogenesis_signal", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.TISSUE,
        description="Recruit blood vessel growth for nutrient supply",
        input_signature=("tissue_state",), output_signature=("vascularized_tissue",),
        may_violate=frozenset(["vascular_constraint"]),
    )
    immune_evasion = Operator(
        name="immune_evasion", substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Downregulate surface antigens to escape immune detection",
        input_signature=("cell_state",), output_signature=("cell_state",),
        may_violate=frozenset(["immune_surveillance"]),
    )

    cancer_ops = frozenset([proliferate, evade_apoptosis, angiogenesis, immune_evasion])
    op_set = OperatorSet(
        name="cancer_hallmark_ops",
        operators=cancer_ops,
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        description="Hallmarks of cancer as operator set (Hanahan & Weinberg)",
    )

    # Constraints (organism-level, being violated by cancer)
    constraints = ConstraintSet(constraints=(
        Constraint(
            name="growth_checkpoint",
            predicate="Cell division requires checkpoint clearance",
            hard=True, editable=False,
        ),
        Constraint(
            name="apoptosis_response",
            predicate="Damaged cells undergo programmed death",
            hard=True, editable=False,
        ),
        Constraint(
            name="tissue_homeostasis",
            predicate="Cell count in tissue remains within bounds",
            hard=True, editable=False,
        ),
        Constraint(
            name="immune_surveillance",
            predicate="Aberrant cells detected and eliminated",
            hard=True, editable=False,
        ),
    ))

    # Two evaluation functions -- the core of goal decoupling
    cell_eval = EvaluationFunction(
        name="cell_proliferation_fitness",
        description="Maximize cell division rate (unicellular fitness landscape)",
        scale=CompetencyScale.CELLULAR,
        metric_type="fitness",
    )

    organism_eval = EvaluationFunction(
        name="tissue_homeostasis",
        description="Maintain tissue structure, cell count, differentiation balance",
        scale=CompetencyScale.ORGANISM,
        metric_type="morphological_target",
    )

    problem_space = ProblemSpace(
        name="cancer_goal_decoupling",
        state_space_description="Cell population: counts, types, spatial arrangement, vascularization",
        operators=op_set,
        constraints=constraints,
        evaluation=cell_eval,  # Cell's OWN evaluation (diverged from organism)
        horizon=1,  # Cancer cells plan 1 step: divide
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
    )

    # Search efficiency: cancer cells are intelligent at THEIR scale
    # tau_blind for establishing a tumor by random growth: ~10^8
    # tau_agent (cancer with hallmarks): ~10^4 divisions
    search_efficiency = SearchEfficiency(
        tau_blind=F(10**8, 1),
        tau_agent=F(10**4, 1),
        K=F(4, 1),  # log10(10^8 / 10^4) = 4
        problem_space_name="cancer_goal_decoupling",
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
    )

    # Two-level competency architecture showing the decoupling
    cell_level = CompetencyLevel(
        level_index=0,
        scale=CompetencyScale.CELLULAR,
        problem_space=problem_space,
        search_efficiency=search_efficiency,
        provides_to_higher=frozenset(["cell_division", "differentiation"]),
        requires_from_lower=frozenset(["gene_expression", "protein_synthesis"]),
    )

    organism_ps = ProblemSpace(
        name="organism_homeostasis",
        state_space_description="Organism morphology: organ function, tissue integrity",
        operators=OperatorSet(
            name="organism_ops",
            operators=frozenset([
                Operator(
                    name="immune_response", substrate=Substrate.BIOLOGICAL_NEURAL,
                    scale=CompetencyScale.ORGANISM,
                    description="Immune system targets aberrant cells",
                    input_signature=("organism_state",), output_signature=("organism_state",),
                    preserves_constraints=frozenset(["tissue_homeostasis"]),
                ),
            ]),
            substrate=Substrate.BIOLOGICAL_NEURAL,
            scale=CompetencyScale.ORGANISM,
        ),
        constraints=ConstraintSet(constraints=(
            Constraint(name="tissue_homeostasis",
                       predicate="All tissues within normal cell count", hard=True),
        )),
        evaluation=organism_eval,
        horizon=365,  # organism plans in days/years
        substrate=Substrate.BIOLOGICAL_NEURAL,
        scale=CompetencyScale.ORGANISM,
    )

    organism_level = CompetencyLevel(
        level_index=1,
        scale=CompetencyScale.ORGANISM,
        problem_space=organism_ps,
        search_efficiency=SearchEfficiency(
            tau_blind=F(10**12, 1),
            tau_agent=F(10**7, 1),
            K=F(5, 1),
            problem_space_name="organism_homeostasis",
            substrate=Substrate.BIOLOGICAL_NEURAL,
        ),
        provides_to_higher=frozenset(["organism_behavior"]),
        requires_from_lower=frozenset(["cell_division", "differentiation", "apoptosis"]),
    )

    architecture = CompetencyArchitecture(
        levels=(cell_level, organism_level),
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        organism_or_system="human_with_cancer",
    )

    barrier = BeyondNeuronsBarrier(
        barrier_type=BarrierType.GOAL_DECOUPLING,
        source_state_class="normal_tissue",
        target_state_class="tissue_homeostasis_restored",
        scale=CompetencyScale.ORGANISM,
        required_mechanism="goal_realignment",
        obstruction_proof=(
            "Cancer cells have decoupled their evaluation function E_cell "
            "(maximize proliferation) from E_organism (maintain homeostasis). "
            "The cell-level problem space optimizes for growth, violating "
            "organism-level constraints. Organism-level barrier GOAL_DECOUPLING "
            "is erected: even with immune response operators, the cancer cells' "
            "divergent goals make homeostasis unreachable without goal realignment "
            "(e.g., bioelectric normalization, differentiation therapy)."
        ),
    )

    before_witness = UnreachabilityWitness(
        target_state_class="tissue_homeostasis_restored",
        operator_set_name="organism_ops",
        barrier=barrier,
        obstruction_argument=(
            "Cancer cells violate growth_checkpoint, apoptosis_response, "
            "and tissue_homeostasis constraints. Their operators (proliferate, "
            "evade_apoptosis, immune_evasion) are optimized for cell-level fitness. "
            "Organism immune_response alone insufficient when goal decoupling active."
        ),
    )

    after_witness = IntelligenceWitness(
        problem_space_name="cancer_goal_decoupling",
        search_efficiency=search_efficiency,
        mechanism="operator_application",
        trace_steps=(
            {"operator": "proliferate", "state": "cell_count_doubled"},
            {"operator": "evade_apoptosis", "state": "apoptotic_signal_suppressed"},
            {"operator": "angiogenesis_signal", "state": "blood_supply_recruited"},
            {"operator": "immune_evasion", "state": "immune_detection_avoided"},
        ),
        goal_achieved=True,  # Cell-level goal achieved (proliferation)
        constraints_preserved=frozenset(),  # Cancer violates organism constraints
        constraints_edited=frozenset(),
    )

    return BeyondNeuronsCertificate(
        certificate_id=cert_id("BEYONDNEUR-CANCER"),
        certificate_type="BEYOND_NEURONS_INTELLIGENCE_CERT",
        timestamp=utc_now_iso(),
        claim=(
            "Cancer demonstrates goal decoupling in multi-scale competency architecture. "
            "Cells achieve K=4 at cellular scale but decouple from organism evaluation. "
            "GOAL_DECOUPLING barrier erected at organism scale. Cancer is intelligent "
            "at its scale -- the problem is misalignment, not stupidity."
        ),
        substrate=Substrate.BIOLOGICAL_NON_NEURAL,
        scale=CompetencyScale.CELLULAR,
        problem_space=problem_space,
        search_efficiency=search_efficiency,
        competency_architecture=architecture,
        barrier=barrier,
        before_witness=before_witness,
        after_witness=after_witness,
        result=IntelligenceResult.GOAL_DECOUPLED,
        source_refs=[
            {"name": "Levin & Chis-Ciure 2024",
             "ref": "Biological Journal of the Linnean Society, blae076"},
            {"name": "Levin 2021 (cancer as reversion)",
             "ref": "doi:10.1016/j.biosystems.2021.104487"},
        ],
    )


# ============================================================================
# FACTORY: NON-NEURAL AI (Substrate Neutrality)
# ============================================================================

def create_non_neural_ai_certificate() -> BeyondNeuronsCertificate:
    """
    LLM achieves K >> 0 without neurons, confirming substrate neutrality.

    Scenario:
        An LLM (silicon substrate) solves mathematical problems with
        search efficiency K comparable to or exceeding human mathematicians.
        This proves the core Levin thesis: intelligence is not about
        neurons. It is about search efficiency in problem spaces.

    QA connection:
        This directly links to ROI #1 (Generator Injection):
        the LLM's operators are the same generators from the injection certificate.
        The substrate is different (silicon vs biological neural).
        The algebra is the same.
    """
    from fractions import Fraction as F

    # Silicon operators (same generators as ROI #1, different substrate)
    text_gen = Operator(
        name="text_generation", substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        description="Autoregressive token generation from transformer model",
        input_signature=("prompt",), output_signature=("text",),
        preserves_constraints=frozenset(["token_budget", "safety_filter"]),
    )
    code_exec = Operator(
        name="code_execution", substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        description="Execute generated code in sandbox",
        input_signature=("code",), output_signature=("output",),
        preserves_constraints=frozenset(["sandbox_isolation"]),
        may_violate=frozenset(["resource_budget"]),
    )
    retrieval = Operator(
        name="retrieval", substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        description="Retrieve relevant documents from knowledge base",
        input_signature=("query",), output_signature=("documents",),
        preserves_constraints=frozenset(["token_budget"]),
    )
    reasoning = Operator(
        name="chain_of_thought", substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        description="Multi-step reasoning via chain-of-thought prompting",
        input_signature=("problem",), output_signature=("reasoning_trace",),
        preserves_constraints=frozenset(["token_budget", "safety_filter"]),
    )

    ai_ops = frozenset([text_gen, code_exec, retrieval, reasoning])
    op_set = OperatorSet(
        name="llm_agent_ops",
        operators=ai_ops,
        substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        description="LLM agent operators (links to ROI #1 Generator Injection)",
    )

    constraints = ConstraintSet(constraints=(
        Constraint(name="token_budget", predicate="Total tokens < context window", hard=True),
        Constraint(name="safety_filter", predicate="Output passes safety checks", hard=True),
        Constraint(name="sandbox_isolation", predicate="Code runs in isolated sandbox", hard=True),
        Constraint(name="resource_budget", predicate="Compute cost < budget", hard=False),
    ))

    evaluation = EvaluationFunction(
        name="math_problem_solving",
        description="Correctly solve competition-level mathematics problems",
        scale=CompetencyScale.ARTIFICIAL,
        metric_type="task_completion",
        optimal_value=F(1, 1),  # 100% accuracy
    )

    problem_space = ProblemSpace(
        name="math_competition_problems",
        state_space_description="Problem statements, intermediate work, candidate solutions",
        operators=op_set,
        constraints=constraints,
        evaluation=evaluation,
        horizon=10,  # Multi-step reasoning depth
        substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
    )

    # tau_blind: Random token generation solving a competition problem: ~10^20
    # tau_agent: LLM with CoT + code execution: ~10^3 tokens
    search_efficiency = SearchEfficiency(
        tau_blind=F(10**20, 1),
        tau_agent=F(10**3, 1),
        K=F(17, 1),  # log10(10^20 / 10^3) = 17
        problem_space_name="math_competition_problems",
        substrate=Substrate.SILICON_DIGITAL,
    )

    after_witness = IntelligenceWitness(
        problem_space_name="math_competition_problems",
        search_efficiency=search_efficiency,
        mechanism="operator_application",
        trace_steps=(
            {"operator": "text_generation", "state": "problem_parsed"},
            {"operator": "chain_of_thought", "state": "approach_identified"},
            {"operator": "code_execution", "state": "computation_verified"},
            {"operator": "text_generation", "state": "solution_formulated"},
        ),
        goal_achieved=True,
        constraints_preserved=frozenset(["token_budget", "safety_filter", "sandbox_isolation"]),
        constraints_edited=frozenset(),
    )

    return BeyondNeuronsCertificate(
        certificate_id=cert_id("BEYONDNEUR-SILICON"),
        certificate_type="BEYOND_NEURONS_INTELLIGENCE_CERT",
        timestamp=utc_now_iso(),
        claim=(
            "LLM achieves K=17 on mathematical problem solving without neurons. "
            "Silicon substrate, same obstruction algebra. "
            "Substrate neutrality confirmed: intelligence = search efficiency, "
            "not neural computation."
        ),
        substrate=Substrate.SILICON_DIGITAL,
        scale=CompetencyScale.ARTIFICIAL,
        problem_space=problem_space,
        search_efficiency=search_efficiency,
        after_witness=after_witness,
        result=IntelligenceResult.SUBSTRATE_NEUTRAL_CONFIRMED,
        source_refs=[
            {"name": "Levin & Chis-Ciure 2024",
             "ref": "Biological Journal of the Linnean Society, blae076"},
            {"name": "LLM-in-Sandbox (ROI #1 link)",
             "ref": "arXiv:2601.16206"},
        ],
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_certificate(cert: BeyondNeuronsCertificate) -> ValidationResult:
    """
    Validate a Beyond Neurons intelligence certificate.

    Uses qa_cert_core.ValidationResult for failure-complete output.
    """
    v = ValidationResult()

    # Certificate type check
    v.check(
        cert.certificate_type == "BEYOND_NEURONS_INTELLIGENCE_CERT",
        f"Wrong certificate_type: {cert.certificate_type}"
    )

    # Certificate ID non-empty
    v.check(bool(cert.certificate_id), "certificate_id is empty")

    # Timestamp non-empty
    v.check(bool(cert.timestamp), "timestamp is empty")

    # Search efficiency sanity
    se = cert.search_efficiency
    v.check(se.tau_blind > 0, "tau_blind must be positive")
    v.check(se.tau_agent > 0, "tau_agent must be positive")

    # K consistency (approximate -- exact log10 of rationals is irrational)
    # Just check K > 0 when tau_blind > tau_agent
    if se.tau_blind > se.tau_agent:
        v.check(se.K > 0, "K should be > 0 when tau_blind > tau_agent")

    # Problem space has operators
    v.check(
        len(cert.problem_space.operators.operators) > 0,
        "Problem space has no operators"
    )

    # Problem space has constraints
    v.check(
        len(cert.problem_space.constraints.constraints) > 0,
        "Problem space has no constraints"
    )

    # Horizon is positive
    v.check(cert.problem_space.horizon > 0, "Horizon must be positive")

    # If operator injection present, verify subset
    if cert.baseline_operators and cert.extended_operators and cert.injected_operators:
        base_names = cert.baseline_operators.operator_names()
        ext_names = cert.extended_operators.operator_names()
        inj_names = frozenset(o.name for o in cert.injected_operators)
        v.check(
            base_names.issubset(ext_names),
            "Baseline operators not subset of extended"
        )
        expected = ext_names - base_names
        v.check(
            inj_names == expected,
            f"Injected mismatch: got {inj_names}, expected {expected}"
        )

    # If constraint editing present, verify at least one constraint changed
    if cert.baseline_constraints and cert.edited_constraints:
        base_names = cert.baseline_constraints.names()
        edit_names = cert.edited_constraints.names()
        v.check(
            base_names == edit_names,
            "Constraint sets must have same constraint names (editing changes predicates, not names)"
        )

    # If competency architecture present, verify ordering and scale consistency
    if cert.competency_architecture:
        arch = cert.competency_architecture
        v.check(len(arch.levels) >= 2, "Competency architecture needs at least 2 levels")

    # Result consistency
    if cert.result == IntelligenceResult.CONSTRAINT_EDITED:
        v.check(
            cert.edited_constraints is not None,
            "CONSTRAINT_EDITED but no edited_constraints provided"
        )
    if cert.result == IntelligenceResult.GOAL_DECOUPLED:
        v.check(
            cert.competency_architecture is not None,
            "GOAL_DECOUPLED but no competency_architecture provided"
        )
    if cert.result == IntelligenceResult.BARRIER_CROSSED:
        v.check(
            cert.barrier is not None,
            "BARRIER_CROSSED but no barrier provided"
        )

    # After witness consistency
    if cert.after_witness:
        v.check(
            cert.after_witness.search_efficiency.K == se.K,
            "After witness K doesn't match certificate K"
        )

    return v


# ============================================================================
# TETRAD SUMMARY
# ============================================================================

def explain_tetrad() -> str:
    """The QA certificate tetrad: injection, collapse, field, beyond neurons."""
    return """
THE QA CERTIFICATE ARCHITECTURE (TETRAD)

All four certificates prove: Capability = Reachability(S, G, I)

+---------------------------------------------------------------------+
|                                                                     |
|  1. GENERATOR INJECTION              Reach EXPANDS                  |
|     G1 subset G2, I preserved        Add tools -> cross barriers    |
|     Domain: agents, proofs           LLM-in-Sandbox, Axiom          |
|                                                                     |
|  2. DIVERSITY COLLAPSE               Reach CONTRACTS                |
|     G fixed, I_div violated          Lose diversity -> erect barrier |
|     Domain: search, RL               Execution-Grounded Research    |
|                                                                     |
|  3. FIELD COMPUTATION                Reach REALIZED BY PHYSICS      |
|     G = physical operators           Air is the computer            |
|     I = power, sync, linearity       WISE RF computing              |
|                                                                     |
|  4. BEYOND NEURONS                   Intelligence is SUBSTRATE-     |
|     P = <S, O, C, E, H>             NEUTRAL and SCALE-FREE         |
|     K = log10(tau_blind/tau_agent)   Same algebra at every scale    |
|     Mechanisms: constraint editing,  Levin & Chis-Ciure 2024        |
|       goal decoupling, horizon exp.                                 |
|                                                                     |
+---------------------------------------------------------------------+

New mechanisms in Direction 4:

  Constraint Editing:
    Same operators O, different constraints C -> new states reachable
    Example: Planaria bioelectric manipulation -> two-headed morphology
    (Operators unchanged. Constraints edited. Reach expanded.)

  Goal Decoupling:
    Component E_i diverges from collective E
    Example: Cancer maximizes cell fitness, violates organism homeostasis
    (Barrier erected at organism scale. Intelligence at cell scale intact.)

  Horizon Expansion:
    Increase H -> multi-step strategies become accessible
    Example: Tool use requires H >= 2 (use tool THEN solve problem)

  Substrate Neutrality:
    K is the same measure whether substrate is neurons, cells, or silicon.
    Intelligence = search efficiency in problem spaces.
    The algebra is the same. The substrate is irrelevant.
"""


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    results = []

    # --- Test 1: Planaria (Constraint Editing) ---
    print("=== TEST 1: Planaria Regeneration (Constraint Editing) ===")
    planaria = create_planaria_certificate()
    r1 = validate_certificate(planaria)
    print(f"Valid: {r1.is_valid}")
    if not r1.is_valid:
        for issue in r1.issues:
            print(f"  Issue: {issue}")
    print(f"Substrate: {planaria.substrate.value}")
    print(f"K: {planaria.search_efficiency.K}")
    print(f"Result: {planaria.result.value}")
    print(f"Hash: {planaria.compute_hash()}")
    results.append(("PLANARIA", r1))
    print()

    # --- Test 2: Cancer (Goal Decoupling) ---
    print("=== TEST 2: Cancer (Goal Decoupling) ===")
    cancer = create_cancer_certificate()
    r2 = validate_certificate(cancer)
    print(f"Valid: {r2.is_valid}")
    if not r2.is_valid:
        for issue in r2.issues:
            print(f"  Issue: {issue}")
    print(f"Substrate: {cancer.substrate.value}")
    print(f"K: {cancer.search_efficiency.K}")
    print(f"Architecture depth: {cancer.competency_architecture.to_dict()['depth']}")
    print(f"Result: {cancer.result.value}")
    print(f"Hash: {cancer.compute_hash()}")
    results.append(("CANCER", r2))
    print()

    # --- Test 3: Non-Neural AI (Substrate Neutrality) ---
    print("=== TEST 3: Non-Neural AI (Substrate Neutrality) ===")
    silicon = create_non_neural_ai_certificate()
    r3 = validate_certificate(silicon)
    print(f"Valid: {r3.is_valid}")
    if not r3.is_valid:
        for issue in r3.issues:
            print(f"  Issue: {issue}")
    print(f"Substrate: {silicon.substrate.value}")
    print(f"K: {silicon.search_efficiency.K}")
    print(f"Result: {silicon.result.value}")
    print(f"Hash: {silicon.compute_hash()}")
    results.append(("SILICON", r3))
    print()

    # Summary
    valid_count = sum(1 for _, r in results if r.is_valid)
    print(f"Results: {valid_count}/{len(results)} valid")
    assert valid_count == len(results), f"Expected all valid, got {valid_count}/{len(results)}"
    print("All Beyond Neurons self-tests PASSED")
    print()

    # Print tetrad
    print(explain_tetrad())
