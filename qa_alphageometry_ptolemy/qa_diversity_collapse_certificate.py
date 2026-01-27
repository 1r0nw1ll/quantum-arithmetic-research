"""
qa_diversity_collapse_certificate.py

QA Certificate Schema for Diversity Collapse Obstruction

Based on:
- Execution-Grounded Automated AI Research (arXiv:2601.14525):
  RL improves mean but collapses diversity, losing upper-tail reachability

Core QA Principle (DUAL of Generator Injection):
    Generator Injection:   G₁ ⊂ G₂ → Reach EXPANDS
    Diversity Collapse:    I_diversity violated → Reach CONTRACTS

    Same framework, opposite direction.
    Injection crosses barriers. Collapse erects them.

This certificate proves:
    Given search process P with diversity invariant I_div:
    IF I_div is violated (diversity drops below threshold),
    THEN the upper envelope of reachable states contracts,
    EVEN IF the mean improves.

    Mean improves ∧ Best stagnates ∧ Diversity collapses
    → MODE_COLLAPSE obstruction

Hard constraints:
- Exact scalars only (int/Fraction) — no floats
- Deterministic serialization
- Failure-completeness: every validation yields success OR obstruction proof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple, FrozenSet
from enum import Enum
from fractions import Fraction
import hashlib
import json
from datetime import datetime


# ============================================================================
# FOUNDATIONAL TYPES (shared with generator_injection_certificate)
# ============================================================================

Scalar = Union[int, Fraction]


def to_scalar(x: Any) -> Scalar:
    """Convert to exact scalar, rejecting floats unless explicitly converted."""
    if isinstance(x, bool):
        raise TypeError("Cannot convert bool to exact scalar")
    if isinstance(x, (int, Fraction)):
        return x
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**9)
    if isinstance(x, str):
        s = x.strip()
        if "/" in s or "." in s:
            return Fraction(s)
        return int(s)
    raise TypeError(f"Cannot convert {type(x)} to exact scalar (got {x})")


# ============================================================================
# DIVERSITY METRIC TYPES
# ============================================================================

class DiversityMetric(Enum):
    """How diversity is measured in the population."""
    TOKEN_ENTROPY = "token_entropy"               # Shannon entropy of token distributions
    EMBEDDING_DISPERSION = "embedding_dispersion"  # Mean pairwise distance in embedding space
    EDIT_DISTANCE = "edit_distance"                # Mean pairwise edit distance of proposals
    OPERATOR_HISTOGRAM = "operator_histogram"       # Entropy of operator/technique usage
    PHENOTYPE_COVERAGE = "phenotype_coverage"       # Fraction of distinct behavioral clusters


class SearchStrategy(Enum):
    """The search strategy being analyzed."""
    RL_POLICY_GRADIENT = "rl_policy_gradient"    # REINFORCE, PPO, GRPO
    EVOLUTIONARY = "evolutionary"                 # Mutation + selection + archive
    RANDOM_SEARCH = "random_search"              # Baseline: i.i.d. sampling
    BEAM_SEARCH = "beam_search"                  # Deterministic pruning


class CollapseVerdict(Enum):
    """Result of collapse analysis."""
    COLLAPSE_DETECTED = "collapse_detected"       # Mean up, best flat, diversity down
    NO_COLLAPSE = "no_collapse"                   # Diversity maintained
    INCONCLUSIVE = "inconclusive"                 # Mixed signals
    INVARIANT_PRESERVED = "invariant_preserved"   # Diversity above threshold throughout


# ============================================================================
# POPULATION SNAPSHOT
# ============================================================================

@dataclass(frozen=True)
class PopulationSnapshot:
    """
    State of the idea population at a given search step.

    Captures both performance and diversity at one point in time.
    """
    step: int

    # Performance metrics (exact)
    mean_reward: Scalar
    max_reward: Scalar
    min_reward: Scalar
    reward_std: Scalar

    # Diversity metric
    diversity_value: Scalar
    diversity_metric: DiversityMetric

    # Population size
    population_size: int

    # Optional: number of distinct phenotypes
    distinct_phenotypes: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "mean_reward", to_scalar(self.mean_reward))
        object.__setattr__(self, "max_reward", to_scalar(self.max_reward))
        object.__setattr__(self, "min_reward", to_scalar(self.min_reward))
        object.__setattr__(self, "reward_std", to_scalar(self.reward_std))
        object.__setattr__(self, "diversity_value", to_scalar(self.diversity_value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "mean_reward": str(self.mean_reward),
            "max_reward": str(self.max_reward),
            "min_reward": str(self.min_reward),
            "reward_std": str(self.reward_std),
            "diversity_value": str(self.diversity_value),
            "diversity_metric": self.diversity_metric.value,
            "population_size": self.population_size,
            "distinct_phenotypes": self.distinct_phenotypes,
        }


# ============================================================================
# SEARCH TRACE
# ============================================================================

@dataclass(frozen=True)
class SearchTrace:
    """
    Complete trace of a search process over time.

    This is the empirical data the certificate reasons over.
    """
    strategy: SearchStrategy
    snapshots: Tuple[PopulationSnapshot, ...]

    # Configuration
    total_steps: int
    budget_gpu_hours: Optional[Scalar] = None

    def __post_init__(self):
        if self.budget_gpu_hours is not None:
            object.__setattr__(self, "budget_gpu_hours", to_scalar(self.budget_gpu_hours))
        # Verify snapshots are ordered by step
        for i in range(1, len(self.snapshots)):
            if self.snapshots[i].step <= self.snapshots[i - 1].step:
                raise ValueError(f"Snapshots not ordered: step {self.snapshots[i].step} <= {self.snapshots[i - 1].step}")

    @property
    def initial(self) -> PopulationSnapshot:
        return self.snapshots[0]

    @property
    def final(self) -> PopulationSnapshot:
        return self.snapshots[-1]

    def mean_rewards(self) -> Tuple[Scalar, ...]:
        return tuple(s.mean_reward for s in self.snapshots)

    def max_rewards(self) -> Tuple[Scalar, ...]:
        return tuple(s.max_reward for s in self.snapshots)

    def diversity_values(self) -> Tuple[Scalar, ...]:
        return tuple(s.diversity_value for s in self.snapshots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "total_steps": self.total_steps,
            "budget_gpu_hours": str(self.budget_gpu_hours) if self.budget_gpu_hours else None,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


# ============================================================================
# DIVERSITY INVARIANT
# ============================================================================

@dataclass(frozen=True)
class DiversityInvariant:
    """
    The diversity invariant that collapse violates.

    I_diversity: diversity_value >= threshold for all steps

    When this invariant holds, the reachable set remains broad.
    When it breaks, the population converges to a narrow basin.
    """
    metric: DiversityMetric
    threshold: Scalar

    # How many consecutive steps below threshold = collapse
    collapse_window: int

    # How much max_reward must improve to count as "not plateaued"
    plateau_epsilon: Scalar

    # How many steps of plateau = stagnation
    plateau_window: int

    def __post_init__(self):
        object.__setattr__(self, "threshold", to_scalar(self.threshold))
        object.__setattr__(self, "plateau_epsilon", to_scalar(self.plateau_epsilon))

    def check_holds(self, trace: SearchTrace) -> Tuple[bool, Optional[int]]:
        """
        Check if the diversity invariant holds throughout the trace.

        Returns (holds, first_violation_step).
        If holds=True, first_violation_step is None.
        """
        consecutive_below = 0
        for snap in trace.snapshots:
            if snap.diversity_value < self.threshold:
                consecutive_below += 1
                if consecutive_below >= self.collapse_window:
                    # Collapse detected
                    violation_step = snap.step - self.collapse_window + 1
                    return False, violation_step
            else:
                consecutive_below = 0
        return True, None

    def check_plateau(self, trace: SearchTrace) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Check if the upper envelope (max_reward) has plateaued.

        Returns (plateaued, plateau_start_step, plateau_end_step).
        """
        max_rewards = trace.max_rewards()
        consecutive_flat = 0
        best_so_far = max_rewards[0]
        plateau_start = None

        for i, r in enumerate(max_rewards):
            if r > best_so_far + self.plateau_epsilon:
                best_so_far = r
                consecutive_flat = 0
                plateau_start = None
            else:
                if consecutive_flat == 0:
                    plateau_start = trace.snapshots[i].step
                consecutive_flat += 1
                if consecutive_flat >= self.plateau_window:
                    return True, plateau_start, trace.snapshots[i].step

        return False, None, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "threshold": str(self.threshold),
            "collapse_window": self.collapse_window,
            "plateau_epsilon": str(self.plateau_epsilon),
            "plateau_window": self.plateau_window,
        }


# ============================================================================
# COLLAPSE WITNESS
# ============================================================================

@dataclass(frozen=True)
class CollapseWitness:
    """
    Witness proving that diversity collapse occurred.

    This is the constructive proof:
    - Shows diversity dropped below threshold
    - Shows upper envelope plateaued
    - Shows mean still improved (ruling out general failure)
    """
    collapse_step: int           # First step where invariant violated
    plateau_start: int           # When max_reward stopped improving
    plateau_end: int             # End of observed plateau

    # The key triple that diagnoses collapse
    mean_improved: bool          # mean_reward[T] > mean_reward[0]
    best_improved: bool          # max_reward[T] > max_reward[0] + epsilon
    diversity_collapsed: bool    # diversity dropped below threshold

    # Deltas (exact)
    delta_mean: Scalar           # mean_reward[T] - mean_reward[0]
    delta_best: Scalar           # max_reward[T] - max_reward[0]
    delta_diversity: Scalar      # diversity[T] - diversity[0]

    def __post_init__(self):
        object.__setattr__(self, "delta_mean", to_scalar(self.delta_mean))
        object.__setattr__(self, "delta_best", to_scalar(self.delta_best))
        object.__setattr__(self, "delta_diversity", to_scalar(self.delta_diversity))

    @property
    def is_mode_collapse(self) -> bool:
        """
        The canonical collapse signature:
        Mean improves AND best stagnates AND diversity drops.

        This is the paper's key finding expressed as a predicate.
        """
        return self.mean_improved and not self.best_improved and self.diversity_collapsed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collapse_step": self.collapse_step,
            "plateau_start": self.plateau_start,
            "plateau_end": self.plateau_end,
            "mean_improved": self.mean_improved,
            "best_improved": self.best_improved,
            "diversity_collapsed": self.diversity_collapsed,
            "is_mode_collapse": self.is_mode_collapse,
            "delta_mean": str(self.delta_mean),
            "delta_best": str(self.delta_best),
            "delta_diversity": str(self.delta_diversity),
        }


@dataclass(frozen=True)
class PreservationWitness:
    """
    Witness proving that diversity was PRESERVED (no collapse).

    Used for the evolutionary search comparison.
    """
    min_diversity_observed: Scalar
    min_diversity_step: int
    diversity_above_threshold: bool

    # Upper envelope still improving
    best_improved: bool
    delta_best: Scalar

    def __post_init__(self):
        object.__setattr__(self, "min_diversity_observed", to_scalar(self.min_diversity_observed))
        object.__setattr__(self, "delta_best", to_scalar(self.delta_best))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_diversity_observed": str(self.min_diversity_observed),
            "min_diversity_step": self.min_diversity_step,
            "diversity_above_threshold": self.diversity_above_threshold,
            "best_improved": self.best_improved,
            "delta_best": str(self.delta_best),
        }


# ============================================================================
# DIVERSITY COLLAPSE OBSTRUCTION CERTIFICATE
# ============================================================================

@dataclass(frozen=True)
class DiversityCollapseObstruction:
    """
    Certificate proving diversity collapse as a reachability obstruction.

    DUAL OF GeneratorInjectionCertificate:

        Generator Injection:  G₁ ⊂ G₂ → Reach EXPANDS → barrier crossed
        Diversity Collapse:   I_div violated → Reach CONTRACTS → barrier erected

    The theorem:
        Under RL update without diversity-preserving generator,
        the reachable set collapses to a strict subset:
            Reach_t+k ⊂ Reach_t   (strictly smaller)
        even as mean performance improves.

    This is NOT a failure of the search algorithm.
    It is a STRUCTURAL PROPERTY of reward-maximizing updates
    without diversity pressure.
    """
    certificate_id: str
    timestamp: str

    # The invariant that was violated
    diversity_invariant: DiversityInvariant

    # The search process that collapsed
    collapsed_trace: SearchTrace

    # Evidence of collapse
    collapse_witness: CollapseWitness

    # Comparison: a search process that preserved diversity (optional)
    preserved_trace: Optional[SearchTrace] = None
    preservation_witness: Optional[PreservationWitness] = None

    # What generator would prevent collapse
    missing_generator: Optional[str] = None  # e.g., "entropy_bonus", "archive_elitism"

    # Result
    verdict: CollapseVerdict = CollapseVerdict.COLLAPSE_DETECTED

    # Metadata
    source_paper: Optional[str] = None
    domain: Optional[str] = None

    def __post_init__(self):
        # Verify collapse witness matches trace data
        if self.collapse_witness.is_mode_collapse and self.verdict != CollapseVerdict.COLLAPSE_DETECTED:
            raise ValueError("Witness shows mode collapse but verdict disagrees")

        # Verify collapsed trace uses a strategy susceptible to collapse
        if self.collapsed_trace.strategy not in (SearchStrategy.RL_POLICY_GRADIENT, SearchStrategy.BEAM_SEARCH):
            raise ValueError(
                f"Collapsed trace uses {self.collapsed_trace.strategy.value}, "
                "expected RL or beam search (strategies susceptible to collapse)"
            )

        # If comparison trace provided, verify it uses diversity-preserving strategy
        if self.preserved_trace is not None:
            if self.preserved_trace.strategy not in (SearchStrategy.EVOLUTIONARY, SearchStrategy.RANDOM_SEARCH):
                raise ValueError(
                    f"Preserved trace uses {self.preserved_trace.strategy.value}, "
                    "expected evolutionary or random (diversity-preserving strategies)"
                )

    def compute_hash(self) -> str:
        """Compute deterministic hash of certificate content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "certificate_id": self.certificate_id,
            "certificate_type": "DIVERSITY_COLLAPSE_OBSTRUCTION",
            "timestamp": self.timestamp,
            "diversity_invariant": self.diversity_invariant.to_dict(),
            "collapsed_trace": self.collapsed_trace.to_dict(),
            "collapse_witness": self.collapse_witness.to_dict(),
            "missing_generator": self.missing_generator,
            "verdict": self.verdict.value,
            "source_paper": self.source_paper,
            "domain": self.domain,
        }
        if self.preserved_trace is not None:
            result["preserved_trace"] = self.preserved_trace.to_dict()
        if self.preservation_witness is not None:
            result["preservation_witness"] = self.preservation_witness.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

def analyze_trace(
    trace: SearchTrace,
    invariant: DiversityInvariant,
) -> Tuple[CollapseVerdict, Optional[CollapseWitness], Optional[PreservationWitness]]:
    """
    Analyze a search trace for diversity collapse.

    Returns (verdict, collapse_witness_or_None, preservation_witness_or_None).
    Failure-complete: always returns a structured result.
    """
    # Check diversity invariant
    div_holds, violation_step = invariant.check_holds(trace)

    # Check plateau
    plateaued, plateau_start, plateau_end = invariant.check_plateau(trace)

    # Compute deltas
    delta_mean = trace.final.mean_reward - trace.initial.mean_reward
    delta_best = trace.final.max_reward - trace.initial.max_reward
    delta_div = trace.final.diversity_value - trace.initial.diversity_value

    mean_improved = delta_mean > 0
    best_improved = delta_best > invariant.plateau_epsilon
    diversity_collapsed = not div_holds

    if diversity_collapsed and plateaued:
        # Full collapse: diversity down + upper envelope flat
        witness = CollapseWitness(
            collapse_step=violation_step,
            plateau_start=plateau_start,
            plateau_end=plateau_end,
            mean_improved=mean_improved,
            best_improved=best_improved,
            diversity_collapsed=True,
            delta_mean=delta_mean,
            delta_best=delta_best,
            delta_diversity=delta_div,
        )
        verdict = CollapseVerdict.COLLAPSE_DETECTED
        return verdict, witness, None

    elif diversity_collapsed and not plateaued:
        # Diversity dropped but best still improving — inconclusive
        witness = CollapseWitness(
            collapse_step=violation_step,
            plateau_start=plateau_start or 0,
            plateau_end=plateau_end or trace.final.step,
            mean_improved=mean_improved,
            best_improved=best_improved,
            diversity_collapsed=True,
            delta_mean=delta_mean,
            delta_best=delta_best,
            delta_diversity=delta_div,
        )
        verdict = CollapseVerdict.INCONCLUSIVE
        return verdict, witness, None

    else:
        # Diversity preserved
        min_div = min(trace.diversity_values())
        min_step = next(s.step for s in trace.snapshots if s.diversity_value == min_div)
        pw = PreservationWitness(
            min_diversity_observed=min_div,
            min_diversity_step=min_step,
            diversity_above_threshold=True,
            best_improved=best_improved,
            delta_best=delta_best,
        )
        verdict = CollapseVerdict.INVARIANT_PRESERVED
        return verdict, None, pw


def create_collapse_certificate(
    rl_trace: SearchTrace,
    evo_trace: Optional[SearchTrace],
    invariant: DiversityInvariant,
    missing_generator: str = "entropy_bonus",
    source_paper: str = "arXiv:2601.14525",
    domain: str = "automated_research",
) -> DiversityCollapseObstruction:
    """
    Create a collapse certificate by analyzing RL trace (and optionally comparing to evolutionary).
    """
    # Analyze RL trace
    rl_verdict, rl_witness, _ = analyze_trace(rl_trace, invariant)

    if rl_witness is None:
        raise ValueError("RL trace did not exhibit collapse — cannot create collapse certificate")

    # Analyze evolutionary trace if provided
    evo_witness = None
    if evo_trace is not None:
        _, _, evo_witness = analyze_trace(evo_trace, invariant)

    return DiversityCollapseObstruction(
        certificate_id=f"DIVCOL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        diversity_invariant=invariant,
        collapsed_trace=rl_trace,
        collapse_witness=rl_witness,
        preserved_trace=evo_trace,
        preservation_witness=evo_witness,
        missing_generator=missing_generator,
        verdict=rl_verdict,
        source_paper=source_paper,
        domain=domain,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_certificate(cert: DiversityCollapseObstruction) -> Tuple[bool, List[str]]:
    """
    Validate a diversity collapse certificate.

    Returns (is_valid, list_of_issues).
    """
    issues = []

    # Check collapsed trace uses collapse-susceptible strategy
    if cert.collapsed_trace.strategy not in (SearchStrategy.RL_POLICY_GRADIENT, SearchStrategy.BEAM_SEARCH):
        issues.append(f"Collapsed trace uses {cert.collapsed_trace.strategy.value}, not RL/beam")

    # Check witness consistency
    w = cert.collapse_witness
    if cert.verdict == CollapseVerdict.COLLAPSE_DETECTED:
        if not w.is_mode_collapse:
            issues.append("COLLAPSE_DETECTED but witness does not show mode collapse signature")
        if not w.diversity_collapsed:
            issues.append("COLLAPSE_DETECTED but diversity_collapsed is False")

    # Verify collapse_step is within trace range
    steps = [s.step for s in cert.collapsed_trace.snapshots]
    if w.collapse_step < min(steps) or w.collapse_step > max(steps):
        issues.append(f"collapse_step {w.collapse_step} outside trace range [{min(steps)}, {max(steps)}]")

    # Verify deltas match trace data
    trace = cert.collapsed_trace
    expected_delta_mean = trace.final.mean_reward - trace.initial.mean_reward
    if w.delta_mean != expected_delta_mean:
        issues.append(f"delta_mean mismatch: witness={w.delta_mean}, computed={expected_delta_mean}")

    expected_delta_best = trace.final.max_reward - trace.initial.max_reward
    if w.delta_best != expected_delta_best:
        issues.append(f"delta_best mismatch: witness={w.delta_best}, computed={expected_delta_best}")

    # If preservation trace provided, check it
    if cert.preserved_trace is not None:
        if cert.preserved_trace.strategy not in (SearchStrategy.EVOLUTIONARY, SearchStrategy.RANDOM_SEARCH):
            issues.append(f"Preserved trace uses {cert.preserved_trace.strategy.value}, expected evolutionary/random")

    if cert.preservation_witness is not None:
        if not cert.preservation_witness.diversity_above_threshold:
            issues.append("Preservation witness claims diversity above threshold but it's False")

    return len(issues) == 0, issues


# ============================================================================
# DUALITY BRIDGE
# ============================================================================

def explain_duality() -> str:
    """
    Explain the structural duality between Generator Injection and Diversity Collapse.

    This is the core theoretical insight.
    """
    return """
GENERATOR INJECTION vs DIVERSITY COLLAPSE: Structural Duality

┌─────────────────────┬──────────────────────────────────────┐
│ Generator Injection  │ Diversity Collapse                   │
├─────────────────────┼──────────────────────────────────────┤
│ G₁ ⊂ G₂            │ I_div violated                       │
│ Reach EXPANDS       │ Reach CONTRACTS                      │
│ Barrier CROSSED     │ Barrier ERECTED                      │
│ New states reachable│ Former states become unreachable     │
│ Invariants preserved│ Diversity invariant violated         │
│ Emergence explained │ Convergence explained                │
├─────────────────────┼──────────────────────────────────────┤
│ Certificate type:    │ Certificate type:                    │
│ GENERATOR_INJECTION  │ DIVERSITY_COLLAPSE_OBSTRUCTION       │
├─────────────────────┼──────────────────────────────────────┤
│ Evidence structure:  │ Evidence structure:                   │
│ - UnreachabilityW   │ - SearchTrace (RL)                   │
│ - ReachabilityW     │ - CollapseWitness                    │
│ - Barrier object    │ - PreservationWitness (evolutionary) │
├─────────────────────┼──────────────────────────────────────┤
│ Mechanism:           │ Mechanism:                            │
│ Add generator →     │ Remove diversity pressure →           │
│ cross barrier       │ collapse to narrow basin              │
├─────────────────────┼──────────────────────────────────────┤
│ Fix:                 │ Fix:                                  │
│ (already fixed by   │ Add diversity-preserving generator    │
│  injection)         │ (entropy bonus, archive, elitism)     │
└─────────────────────┴──────────────────────────────────────┘

General Theorem:
    Reachability is controlled by the GENERATOR-INVARIANT pair (G, I).
    - Expanding G while preserving I → more reachable states
    - Violating I without expanding G → fewer reachable states
    - The GENERATOR INJECTION and DIVERSITY COLLAPSE certificates
      are the constructive witnesses for these two directions.
"""


# ============================================================================
# EXAMPLE / TEST
# ============================================================================

if __name__ == "__main__":
    from fractions import Fraction as F

    # ---- Build RL trace that collapses ----
    # Paper finding: mean improves, best STAGNATES, diversity collapses
    rl_snapshots = []
    for step in range(11):
        # Mean steadily improves (RL is good at this)
        mean = F(50 + step * 3, 1)
        # Best improves only in first 2 steps, then flatlines
        # Total delta_best = 2 which is <= plateau_epsilon of 2
        best = F(70 + min(step, 2), 1)
        # Diversity drops steadily (mode collapse)
        div = F(max(100 - step * 12, 5), 1)
        rl_snapshots.append(PopulationSnapshot(
            step=step,
            mean_reward=mean,
            max_reward=best,
            min_reward=F(30 + step, 1),
            reward_std=F(15 - step, 1),
            diversity_value=div,
            diversity_metric=DiversityMetric.EMBEDDING_DISPERSION,
            population_size=64,
        ))

    rl_trace = SearchTrace(
        strategy=SearchStrategy.RL_POLICY_GRADIENT,
        snapshots=tuple(rl_snapshots),
        total_steps=10,
        budget_gpu_hours=F(100, 1),
    )

    # ---- Build evolutionary trace that preserves diversity ----
    evo_snapshots = []
    for step in range(11):
        # Mean improves slower
        mean = F(50 + step * 2, 1)
        # Best keeps improving throughout
        best = F(70 + step * 4, 1)
        # Diversity stays high (fluctuates but never collapses)
        div = F(100 - step * 3 + (step % 3) * 5, 1)
        evo_snapshots.append(PopulationSnapshot(
            step=step,
            mean_reward=mean,
            max_reward=best,
            min_reward=F(30, 1),
            reward_std=F(20, 1),
            diversity_value=div,
            diversity_metric=DiversityMetric.EMBEDDING_DISPERSION,
            population_size=64,
        ))

    evo_trace = SearchTrace(
        strategy=SearchStrategy.EVOLUTIONARY,
        snapshots=tuple(evo_snapshots),
        total_steps=10,
        budget_gpu_hours=F(100, 1),
    )

    # ---- Define diversity invariant ----
    invariant = DiversityInvariant(
        metric=DiversityMetric.EMBEDDING_DISPERSION,
        threshold=F(30, 1),       # Diversity must stay above 30
        collapse_window=3,         # 3 consecutive steps below = collapse
        plateau_epsilon=F(2, 1),  # Must improve by >2 to count
        plateau_window=4,          # 4 steps flat = plateau
    )

    # ---- Create certificate ----
    cert = create_collapse_certificate(
        rl_trace=rl_trace,
        evo_trace=evo_trace,
        invariant=invariant,
        missing_generator="entropy_bonus_regularizer",
    )

    is_valid, issues = validate_certificate(cert)
    print(f"Diversity Collapse Certificate Valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    print()

    # Print the key diagnosis
    w = cert.collapse_witness
    print("=== COLLAPSE DIAGNOSIS ===")
    print(f"Mean improved:      {w.mean_improved} (delta: {w.delta_mean})")
    print(f"Best improved:      {w.best_improved} (delta: {w.delta_best})")
    print(f"Diversity collapsed: {w.diversity_collapsed}")
    print(f"Mode collapse:      {w.is_mode_collapse}")
    print(f"Collapse step:      {w.collapse_step}")
    print(f"Plateau:            [{w.plateau_start}, {w.plateau_end}]")
    print(f"Missing generator:  {cert.missing_generator}")
    print()

    # Print duality explanation
    print(explain_duality())

    # Print full JSON
    print("=== FULL CERTIFICATE ===")
    print(cert.to_json())
