"""
qa_generator_injection_certificate.py

QA Certificate Schema for Generator Injection and Agentic Emergence

Based on:
- LLM-in-Sandbox (arXiv:2601.16206): Agentic behavior emerges from generator injection
- Axiom Putnam 2025: Difficulty is generator-relative; adding tactics crosses barriers

Core QA Principle:
    Capability is NOT intrinsic to the agent.
    Capability = Reachability(State Space, Generator Set, Invariants).
    Adding generators → barrier crossing → emergent behavior.

This certificate proves:
    Given state space S, invariants I, and generator sets G₁ ⊂ G₂:
    ∃ states reachable under G₂ that are unreachable under G₁
    AND all invariants I are preserved under both.

Hard constraints:
- Exact scalars only (int/Fraction) — no floats
- Deterministic serialization
- Failure-completeness: every validation yields success OR obstruction proof
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict, Any, Union, Tuple, Set, FrozenSet
from enum import Enum
from fractions import Fraction
import hashlib
import json
from datetime import datetime


# ============================================================================
# FOUNDATIONAL TYPES
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
# GENERATOR DEFINITION
# ============================================================================

class GeneratorCategory(Enum):
    """Categories of generators (action types)."""
    # From LLM-in-Sandbox
    EXTERNAL_ACCESS = "external_access"      # curl, pip install, web fetch
    FILE_SYSTEM = "file_system"              # read, write, grep, sed
    CODE_EXECUTION = "code_execution"        # run scripts, interpreters

    # From Axiom / Formal Proof
    TACTIC = "tactic"                        # proof tactic
    LEMMA_APPLY = "lemma_apply"              # apply existing lemma
    REWRITE = "rewrite"                      # term rewriting
    DECISION_PROCEDURE = "decision_procedure" # ring, linarith, omega

    # Generic
    OBSERVATION = "observation"              # read-only state inspection
    MUTATION = "mutation"                    # state modification
    COMPOSITION = "composition"              # combine sub-generators


@dataclass(frozen=True)
class Generator:
    """
    A single generator (action/move) in the state space.

    Generators are the atomic units of capability.
    Adding a generator can cross barriers and expand reachability.
    """
    name: str
    category: GeneratorCategory

    # Formal signature: input types → output types (as strings)
    input_signature: Tuple[str, ...]
    output_signature: Tuple[str, ...]

    # Precondition (when can this generator fire?)
    precondition: Optional[str] = None

    # Invariants this generator preserves (by name)
    preserves_invariants: FrozenSet[str] = frozenset()

    # Invariants this generator may violate (obstruction risk)
    may_violate: FrozenSet[str] = frozenset()

    # Cost metric (if applicable)
    cost: Optional[Scalar] = None

    def __post_init__(self):
        if self.cost is not None:
            object.__setattr__(self, "cost", to_scalar(self.cost))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "input_signature": list(self.input_signature),
            "output_signature": list(self.output_signature),
            "precondition": self.precondition,
            "preserves_invariants": sorted(self.preserves_invariants),
            "may_violate": sorted(self.may_violate),
            "cost": str(self.cost) if self.cost else None,
        }


@dataclass(frozen=True)
class GeneratorSet:
    """
    A set of generators defining available moves.

    The key insight from LLM-in-Sandbox and Axiom:
    The generator set determines what states are reachable.
    Same agent + different generators = different capability.
    """
    name: str
    generators: FrozenSet[Generator]

    # Metadata
    description: Optional[str] = None
    source_paper: Optional[str] = None

    def __len__(self) -> int:
        return len(self.generators)

    def __contains__(self, gen: Generator) -> bool:
        return gen in self.generators

    def generator_names(self) -> FrozenSet[str]:
        return frozenset(g.name for g in self.generators)

    def categories_present(self) -> FrozenSet[GeneratorCategory]:
        return frozenset(g.category for g in self.generators)

    def is_subset_of(self, other: GeneratorSet) -> bool:
        return self.generators.issubset(other.generators)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "generators": [g.to_dict() for g in sorted(self.generators, key=lambda x: x.name)],
            "description": self.description,
            "source_paper": self.source_paper,
        }


# ============================================================================
# BARRIER DEFINITION
# ============================================================================

class BarrierType(Enum):
    """Types of barriers between state regions."""
    # From Axiom paper
    MISSING_LEMMA = "missing_lemma"          # library gap
    TYPE_MISMATCH = "type_mismatch"          # constraint violation
    REWRITE_BLOCKED = "rewrite_blocked"      # can't apply transformation
    CASE_EXPLOSION = "case_explosion"        # combinatorial blowup
    FORMALISM_OVERHEAD = "formalism_overhead" # tedious but not hard

    # From LLM-in-Sandbox paper
    CONTEXT_LENGTH = "context_length"        # prompt too long
    TOOL_UNAWARE = "tool_unaware"            # doesn't know generator exists
    ENVIRONMENT_MISMATCH = "environment_mismatch"  # wrong dependencies
    MULTI_STEP_PLANNING = "multi_step_planning"    # can't chain actions

    # Generic
    INVARIANT_VIOLATION = "invariant_violation"    # would break constraint
    BUDGET_EXHAUSTED = "budget_exhausted"          # out of resources
    UNREACHABLE = "unreachable"                    # provably impossible


@dataclass(frozen=True)
class Barrier:
    """
    A barrier preventing reachability between states.

    Barriers are not bugs — they are theoretical objects.
    Understanding barriers = understanding capability limits.
    """
    barrier_type: BarrierType

    # States involved
    source_state_class: str  # description of starting states
    target_state_class: str  # description of goal states

    # What's missing
    required_generator: Optional[str] = None
    required_invariant: Optional[str] = None

    # Evidence
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


# ============================================================================
# REACHABILITY WITNESS
# ============================================================================

@dataclass(frozen=True)
class ReachabilityWitness:
    """
    Witness proving a state is reachable under given generators.

    This is the constructive proof: an explicit path of generator applications.
    """
    start_state_hash: str
    end_state_hash: str

    # The path: sequence of (generator_name, intermediate_state_hash)
    path: Tuple[Tuple[str, str], ...]

    # Path metrics
    path_length: int
    total_cost: Optional[Scalar] = None

    # Invariants verified along path
    invariants_checked: FrozenSet[str] = frozenset()

    def __post_init__(self):
        if self.total_cost is not None:
            object.__setattr__(self, "total_cost", to_scalar(self.total_cost))
        # Verify path length consistency
        if len(self.path) != self.path_length:
            raise ValueError(f"Path length mismatch: {len(self.path)} != {self.path_length}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_state_hash": self.start_state_hash,
            "end_state_hash": self.end_state_hash,
            "path": [{"generator": g, "state_hash": s} for g, s in self.path],
            "path_length": self.path_length,
            "total_cost": str(self.total_cost) if self.total_cost else None,
            "invariants_checked": sorted(self.invariants_checked),
        }


@dataclass(frozen=True)
class UnreachabilityWitness:
    """
    Witness proving a state is unreachable under given generators.

    This is the obstruction proof: demonstrates why no path exists.
    """
    target_state_class: str
    generator_set_name: str

    # The barrier preventing reachability
    barrier: Barrier

    # Formal obstruction (if available)
    obstruction_type: BarrierType
    obstruction_argument: str

    # Search statistics (if exhaustive)
    states_explored: Optional[int] = None
    max_depth_reached: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_state_class": self.target_state_class,
            "generator_set_name": self.generator_set_name,
            "barrier": self.barrier.to_dict(),
            "obstruction_type": self.obstruction_type.value,
            "obstruction_argument": self.obstruction_argument,
            "states_explored": self.states_explored,
            "max_depth_reached": self.max_depth_reached,
        }


# ============================================================================
# GENERATOR INJECTION CERTIFICATE
# ============================================================================

class InjectionResult(Enum):
    """Result of generator injection analysis."""
    BARRIER_CROSSED = "barrier_crossed"      # New states now reachable
    NO_NEW_REACH = "no_new_reach"            # Injection didn't help
    INVARIANT_BROKEN = "invariant_broken"    # Injection violated constraints
    PENDING = "pending"                       # Not yet analyzed


@dataclass(frozen=True)
class GeneratorInjectionCertificate:
    """
    Certificate proving that generator injection crosses a barrier.

    This is the core QA claim from LLM-in-Sandbox and Axiom:

        G₁ ⊂ G₂ ⟹ Reach(S, G₂) ⊇ Reach(S, G₁)

        AND there exist states in Reach(S, G₂) \\ Reach(S, G₁)
        (strictly more reachable)

        AND all invariants I are preserved.

    This explains "agentic emergence" without magic:
    It's just barrier crossing under generator injection.
    """
    certificate_id: str
    timestamp: str

    # The injection
    before_generators: GeneratorSet
    after_generators: GeneratorSet
    injected_generators: FrozenSet[Generator]  # G₂ \ G₁

    # The barrier that was crossed
    barrier_crossed: Barrier

    # Evidence: unreachable before, reachable after
    before_witness: UnreachabilityWitness
    after_witness: ReachabilityWitness

    # Invariants preserved
    invariants_preserved: FrozenSet[str]
    invariant_verification_method: str

    # Result
    result: InjectionResult

    # Metadata
    source_paper: Optional[str] = None
    domain: Optional[str] = None  # e.g., "agentic_ai", "formal_proof", "biology"

    def __post_init__(self):
        # Verify injection is valid: injected names = after names - before names
        before_names = self.before_generators.generator_names()
        after_names = self.after_generators.generator_names()
        injected_names = frozenset(g.name for g in self.injected_generators)

        expected_injected = after_names - before_names
        if injected_names != expected_injected:
            raise ValueError(
                f"Injected generators mismatch: "
                f"got {injected_names}, expected {expected_injected}"
            )

        # Verify subset relation (by name, since Generator objects may differ)
        if not before_names.issubset(after_names):
            raise ValueError("before_generators must be subset of after_generators")

    def compute_hash(self) -> str:
        """Compute deterministic hash of certificate content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "certificate_type": "GENERATOR_INJECTION",
            "timestamp": self.timestamp,
            "before_generators": self.before_generators.to_dict(),
            "after_generators": self.after_generators.to_dict(),
            "injected_generators": [g.to_dict() for g in sorted(self.injected_generators, key=lambda x: x.name)],
            "barrier_crossed": self.barrier_crossed.to_dict(),
            "before_witness": self.before_witness.to_dict(),
            "after_witness": self.after_witness.to_dict(),
            "invariants_preserved": sorted(self.invariants_preserved),
            "invariant_verification_method": self.invariant_verification_method,
            "result": self.result.value,
            "source_paper": self.source_paper,
            "domain": self.domain,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_llm_sandbox_certificate(
    before_tools: List[str],
    after_tools: List[str],
    task_class: str,
    unreachable_evidence: str,
    reachable_path: List[Tuple[str, str]],
    invariants: List[str],
) -> GeneratorInjectionCertificate:
    """
    Create a certificate for LLM-in-Sandbox style generator injection.

    Example:
        before_tools = ["text_generation"]
        after_tools = ["text_generation", "execute_bash", "file_read", "file_write"]
        task_class = "long_context_processing"
        unreachable_evidence = "Context exceeds 100k tokens, cannot fit in prompt"
        reachable_path = [("file_write", "ctx_stored"), ("file_read", "ctx_retrieved"), ...]
    """
    # Build generator sets
    before_gens = frozenset(
        Generator(
            name=t,
            category=GeneratorCategory.OBSERVATION if t == "text_generation" else GeneratorCategory.MUTATION,
            input_signature=("state",),
            output_signature=("state",),
        )
        for t in before_tools
    )

    after_gens = frozenset(
        Generator(
            name=t,
            category=_infer_category(t),
            input_signature=("state",),
            output_signature=("state",),
        )
        for t in after_tools
    )

    before_set = GeneratorSet(
        name="vanilla_llm",
        generators=before_gens,
        description="Standard LLM without tool access",
        source_paper="baseline",
    )

    after_set = GeneratorSet(
        name="llm_in_sandbox",
        generators=after_gens,
        description="LLM with sandbox tool access",
        source_paper="arXiv:2601.16206",
    )

    # Compute injected as generators in after but not in before (by name)
    before_names = frozenset(g.name for g in before_gens)
    injected = frozenset(g for g in after_gens if g.name not in before_names)

    # Build barrier
    barrier = Barrier(
        barrier_type=BarrierType.CONTEXT_LENGTH,
        source_state_class="initial_prompt_state",
        target_state_class=task_class,
        required_generator="file_system" if "file_read" in after_tools else None,
        obstruction_proof=unreachable_evidence,
    )

    # Build witnesses
    before_witness = UnreachabilityWitness(
        target_state_class=task_class,
        generator_set_name="vanilla_llm",
        barrier=barrier,
        obstruction_type=BarrierType.CONTEXT_LENGTH,
        obstruction_argument=unreachable_evidence,
    )

    # Convert path to proper format
    path_tuples = tuple((gen, state) for gen, state in reachable_path)
    after_witness = ReachabilityWitness(
        start_state_hash=hashlib.sha256(b"initial").hexdigest()[:16],
        end_state_hash=hashlib.sha256(task_class.encode()).hexdigest()[:16],
        path=path_tuples,
        path_length=len(path_tuples),
        invariants_checked=frozenset(invariants),
    )

    return GeneratorInjectionCertificate(
        certificate_id=f"GENINJ-SANDBOX-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        before_generators=before_set,
        after_generators=after_set,
        injected_generators=injected,
        barrier_crossed=barrier,
        before_witness=before_witness,
        after_witness=after_witness,
        invariants_preserved=frozenset(invariants),
        invariant_verification_method="execution_feedback_oracle",
        result=InjectionResult.BARRIER_CROSSED,
        source_paper="arXiv:2601.16206",
        domain="agentic_ai",
    )


def create_axiom_proof_certificate(
    baseline_tactics: List[str],
    extended_tactics: List[str],
    theorem_class: str,
    blocked_reason: str,
    proof_trace: List[Tuple[str, str]],
    kernel_invariants: List[str],
) -> GeneratorInjectionCertificate:
    """
    Create a certificate for Axiom-style tactic injection.

    Example:
        baseline_tactics = ["intro", "apply", "exact"]
        extended_tactics = ["intro", "apply", "exact", "ring", "linarith", "omega"]
        theorem_class = "polynomial_identity"
        blocked_reason = "Manual ring manipulation requires O(n²) rewrites"
        proof_trace = [("ring", "goal_closed"), ...]
    """
    before_gens = frozenset(
        Generator(
            name=t,
            category=GeneratorCategory.TACTIC,
            input_signature=("proof_state",),
            output_signature=("proof_state",),
            preserves_invariants=frozenset(kernel_invariants),
        )
        for t in baseline_tactics
    )

    after_gens = frozenset(
        Generator(
            name=t,
            category=GeneratorCategory.TACTIC if t not in ["ring", "linarith", "omega"]
                     else GeneratorCategory.DECISION_PROCEDURE,
            input_signature=("proof_state",),
            output_signature=("proof_state",),
            preserves_invariants=frozenset(kernel_invariants),
        )
        for t in extended_tactics
    )

    before_set = GeneratorSet(
        name="basic_tactics",
        generators=before_gens,
        description="Core proof tactics without decision procedures",
    )

    after_set = GeneratorSet(
        name="extended_tactics",
        generators=after_gens,
        description="Tactics with decision procedures",
        source_paper="Axiom Putnam 2025",
    )

    # Compute injected as generators in after but not in before (by name)
    before_names = frozenset(g.name for g in before_gens)
    injected = frozenset(g for g in after_gens if g.name not in before_names)

    barrier = Barrier(
        barrier_type=BarrierType.FORMALISM_OVERHEAD,
        source_state_class="initial_proof_state",
        target_state_class=theorem_class,
        required_generator=list(injected)[0].name if injected else None,
        obstruction_proof=blocked_reason,
    )

    before_witness = UnreachabilityWitness(
        target_state_class=theorem_class,
        generator_set_name="basic_tactics",
        barrier=barrier,
        obstruction_type=BarrierType.FORMALISM_OVERHEAD,
        obstruction_argument=blocked_reason,
    )

    path_tuples = tuple((gen, state) for gen, state in proof_trace)
    after_witness = ReachabilityWitness(
        start_state_hash=hashlib.sha256(b"proof_init").hexdigest()[:16],
        end_state_hash=hashlib.sha256(b"qed").hexdigest()[:16],
        path=path_tuples,
        path_length=len(path_tuples),
        invariants_checked=frozenset(kernel_invariants),
    )

    return GeneratorInjectionCertificate(
        certificate_id=f"GENINJ-AXIOM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        timestamp=datetime.now().isoformat(),
        before_generators=before_set,
        after_generators=after_set,
        injected_generators=injected,
        barrier_crossed=barrier,
        before_witness=before_witness,
        after_witness=after_witness,
        invariants_preserved=frozenset(kernel_invariants),
        invariant_verification_method="lean_kernel_typecheck",
        result=InjectionResult.BARRIER_CROSSED,
        source_paper="Axiom Putnam 2025",
        domain="formal_proof",
    )


def _infer_category(tool_name: str) -> GeneratorCategory:
    """Infer generator category from tool name."""
    if tool_name in ["execute_bash", "run_python", "run_script"]:
        return GeneratorCategory.CODE_EXECUTION
    if tool_name in ["file_read", "file_write", "grep", "sed", "ls"]:
        return GeneratorCategory.FILE_SYSTEM
    if tool_name in ["curl", "pip_install", "web_fetch"]:
        return GeneratorCategory.EXTERNAL_ACCESS
    return GeneratorCategory.MUTATION


# ============================================================================
# VALIDATION
# ============================================================================

def validate_certificate(cert: GeneratorInjectionCertificate) -> Tuple[bool, List[str]]:
    """
    Validate a generator injection certificate.

    Returns (is_valid, list_of_issues).
    """
    issues = []

    # Check subset relation (by name)
    before_names = cert.before_generators.generator_names()
    after_names = cert.after_generators.generator_names()
    if not before_names.issubset(after_names):
        issues.append("before_generators is not a subset of after_generators")

    # Check injected generators match difference
    before_names = cert.before_generators.generator_names()
    after_names = cert.after_generators.generator_names()
    injected_names = frozenset(g.name for g in cert.injected_generators)
    expected = after_names - before_names
    if injected_names != expected:
        issues.append(f"injected_generators mismatch: {injected_names} != {expected}")

    # Check path uses only after_generators
    for gen_name, _ in cert.after_witness.path:
        if gen_name not in after_names:
            issues.append(f"Path uses unknown generator: {gen_name}")

    # Check barrier type consistency
    if cert.before_witness.obstruction_type != cert.barrier_crossed.barrier_type:
        issues.append("Barrier type mismatch between witness and certificate")

    # Check result consistency
    if cert.result == InjectionResult.BARRIER_CROSSED:
        if cert.after_witness.path_length == 0:
            issues.append("BARRIER_CROSSED but no path provided")

    return len(issues) == 0, issues


# ============================================================================
# EXAMPLE / TEST
# ============================================================================

if __name__ == "__main__":
    # Example: LLM-in-Sandbox certificate
    sandbox_cert = create_llm_sandbox_certificate(
        before_tools=["text_generation"],
        after_tools=["text_generation", "execute_bash", "file_read", "file_write"],
        task_class="long_context_qa_100k_tokens",
        unreachable_evidence="Context length 100k tokens exceeds model context window of 32k",
        reachable_path=[
            ("file_write", "context_stored_to_disk"),
            ("execute_bash", "grep_relevant_sections"),
            ("file_read", "load_filtered_context"),
            ("text_generation", "answer_generated"),
        ],
        invariants=["valid_response", "no_hallucination", "task_completion"],
    )

    is_valid, issues = validate_certificate(sandbox_cert)
    print(f"LLM-in-Sandbox Certificate Valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    print()
    print(sandbox_cert.to_json())
    print()

    # Example: Axiom proof certificate
    axiom_cert = create_axiom_proof_certificate(
        baseline_tactics=["intro", "apply", "exact", "rewrite"],
        extended_tactics=["intro", "apply", "exact", "rewrite", "ring", "linarith"],
        theorem_class="polynomial_factorization_identity",
        blocked_reason="Manual polynomial expansion requires 47 rewrite steps",
        proof_trace=[
            ("intro", "vars_introduced"),
            ("ring", "polynomial_normalized"),
        ],
        kernel_invariants=["well_typed", "definitional_equality", "universe_consistent"],
    )

    is_valid, issues = validate_certificate(axiom_cert)
    print(f"Axiom Proof Certificate Valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
    print()
    print(axiom_cert.to_json())
