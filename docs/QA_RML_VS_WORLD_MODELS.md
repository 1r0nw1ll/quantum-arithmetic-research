# QA-RML vs World Models: A Direct Mapping

**Reference**: Gupta & Pruthi, "Beyond World Models: Rethinking Understanding in AI Models" (arXiv:2511.12239v1)

---

## Executive Summary

The paper argues that **world models are insufficient for understanding**. QA-RML resolves this by design:

| Paradigm | What it learns | What it certifies | Understanding? |
|----------|---------------|-------------------|----------------|
| **World Models** | State → Next-state mapping | Prediction accuracy | ❌ No |
| **QAWM** | State → Reachability structure | Return-in-k (0.836 AUROC) | Partial |
| **QA-RML** | Generator → Obstruction certificates | Why paths don't exist | ✅ Yes |

**The key insight**: World models predict *what happens*. RML certifies *what cannot happen and why*.

---

## 1. Paradigm Architecture Comparison

### 1.1 Standard World Models

```
Input: s_t (state at time t)
Model: f_θ(s_t, a) → s_{t+1}
Training: minimize ||f_θ(s_t, a) - s_{t+1}||²
Output: Predicted next state
```

**What it captures**:
- Forward dynamics
- Trajectory simulation
- One-step predictions

**What it misses** (per the paper):
- Why certain states are unreachable
- Invariant structure governing outcomes
- Compression/explanation of trajectories

### 1.2 QAWM (QA World Model) — Intermediate Layer

```
Input: s = QAState (21-element invariant packet)
       g = Generator (σ, μ, λ₂, ν)
Model: QAWM(s, g) → {legal?, fail_type, return_in_k?}
Training: Multi-task (legality + fail classification + reachability)
Output: Structural predicates, not next-state
```

**Architecture** (from `qawm.py`):
```python
# 3-bucket feature strategy
Bucket A: (b/N, e/N, d/2N, a/3N, φ₉, φ₂₄)      # Normalized primitives
Bucket B: log-scaled invariants (B,E,D,A,X,C,F,G,H,I,J,K,W,Y,Z,h²,N_compat,Sum)
Bucket C: log1p(L_num), log1p(L_denom)         # Rational features

# Multi-task heads
Head 1: P(legal | s, g)           # Binary
Head 2: P(fail_type | s, g)       # 5-class
Head 3: P(return_in_k | s, g)     # Binary
```

**What QAWM adds**:
- Reachability structure (not just next-state)
- Failure classification
- Structural queries

**What QAWM still misses**:
- Full obstruction certificates
- Explanation of *why* unreachable

### 1.3 QA-RML — Full Understanding Layer

```
Input: (s, target_class, generator_set, depth_bound)
Model: RML policy + certificate generator
Training: Bandit REINFORCE on structural queries (not reward)
Output: {reachable?, path_witness | obstruction_certificate}
```

**Architecture** (from `rml_policy.py`):
```python
# Policy hierarchy
Random-Legal:  Baseline (5 oracle calls/step)
Oracle-Greedy: Upper bound using true return_in_k (8-12 calls/step)
QAWM-Greedy:   Learned structure scoring (1 call/step, 4.20 normalized success)
RMLPolicy:     QAWM hints + learned network (bandit REINFORCE)
```

**What RML adds**:
- **Obstruction certificates** (why paths don't exist)
- **Minimal witnesses** (shortest success path)
- **Structural learning** (topology, not dynamics)

---

## 2. The Paper's Three Case Studies → QA-RML Mapping

### 2.1 Case Study 1: Hofstadter's Domino Computer

**Paper's Claim**: Tracking domino states doesn't explain primality.

#### World Model Approach (Insufficient)
```
State: s ∈ {0,1}^N (domino up/down)
Dynamics: Simulate local propagation
Output: Final configuration
```
→ Can predict *what happens*, not *why* (Prime(n) governs outcome)

#### QA-RML Approach (Understanding)
```
Microstate: QAState with N = input encoding
Macro-invariant: Δ_prime(s) → {PRIME, COMPOSITE}
Certificate structure:

IF Prime(N):
  ObstructionCert {
    fail_type: GENERATOR_INSUFFICIENT,
    target: DIVISOR_STRETCH,
    evidence: "No divisor d|N exists for d ∈ {2,...,√N}",
    blocked_generator_set: {σ_divisor_test(d) for d in range}
  }
THEN: Only PRIME_STRETCH reachable

IF Composite(N):
  PathWitness {
    path: [σ_init → σ_divisor_test(d) → σ_trigger],
    witness: d where d|N,
    length: O(√N)
  }
THEN: DIVISOR_STRETCH reachable
```

**Key difference**: RML produces the invariant-to-outcome certificate, not just the trajectory.

### 2.2 Case Study 2: Proof Understanding

**Paper's Claim**: Verifying proof steps ≠ understanding the proof.

#### World Model Approach (Insufficient)
```
State: Proof AST node
Dynamics: Apply inference rule
Output: Valid/Invalid
```
→ Can verify *legality*, not *strategy* or *key steps*

#### QA-RML Approach (Understanding)

Using `qa_certificate.py` structures:

```python
# Verification (world model equivalent)
verification_trace = [
    MoveWitness(gen=InferenceRule("modus_ponens"), src=s1, dst=s2, legal=True),
    MoveWitness(gen=InferenceRule("universal_inst"), src=s2, dst=s3, legal=True),
    ...
]

# Understanding (RML addition)
understanding_cert = UnderstandingCertificate(
    schema="qa_proof_understanding/v1",

    # Strategy identification
    strategy=Strategy(
        type="involution_fixed_point_parity",  # Zagier style
        key_insight="Involution on pairs; fixed points counted mod 2",
    ),

    # Key step identification
    key_steps=[
        KeyStep(
            index=7,
            description="Define involution τ on (x,y,z) triples",
            necessity_witness=ObstructionEvidence(
                fail_type=FailType.GENERATOR_INSUFFICIENT,
                evidence="Without τ, no counting argument possible"
            )
        )
    ],

    # Compression metric
    compression_ratio=len(full_trace) / len(outline),  # e.g., 47 / 5 = 9.4

    # Counterfactual
    counterfactual=CounterfactualWitness(
        removed_step=7,
        obstruction="Proof incomplete: cannot establish bijection"
    )
)
```

**Mapping to existing QA structures**:
- `MoveWitness` = individual inference steps
- `ObstructionEvidence` = why alternative approaches fail
- `KeyStep` = non-routine moves with necessity certificates

### 2.3 Case Study 3: Bohr Theory (Problem Situation)

**Paper's Claim**: Understanding Bohr requires knowing the problem (discrete spectral lines), not just simulating electrons.

#### World Model Approach (Insufficient)
```
State: Electron orbit configuration
Dynamics: Classical or quantized transitions
Output: Predicted spectrum
```
→ Can simulate *what electrons do*, not *why quantization is necessary*

#### QA-RML Approach (Understanding)

Using `ProjectionContract` from `qa_certificate.py`:

```python
# Problem situation as reachability failure
classical_obstruction = ObstructionEvidence(
    fail_type=FailType.LAW_VIOLATION,
    law_name="Classical_Electrodynamics",
    measured_observables={
        "spectral_lines": "DISCRETE",
        "expected": "CONTINUOUS"
    },
    law_violation_delta=Fraction("∞"),  # Categorical mismatch
    evidence="Classical orbits → continuous radiation; observed: discrete"
)

# Theory upgrade as generator extension
bohr_upgrade = GeneratorExtension(
    original_generators=["classical_orbit_dynamics"],
    added_generators=["quantized_energy_levels", "discrete_transitions"],

    # Now reachable
    newly_reachable=PathWitness(
        target="DISCRETE_SPECTRAL_LINES",
        path=[
            MoveWitness(gen="quantized_E_n", src="electron_state", dst="level_n"),
            MoveWitness(gen="transition_ΔE", src="level_n", dst="level_m"),
            MoveWitness(gen="photon_emission", src="ΔE", dst="spectral_line_νₙₘ")
        ]
    ),

    # Understanding certificate
    understanding=ProblemSituationCert(
        gap="Classical mechanics cannot produce discrete spectra",
        resolution="Quantization postulates close the gap",
        necessity="ΔE = hν is forced by observation"
    )
)
```

---

## 3. The Falsifiability Resolution

### The Paper's Paradox

> "If you enrich world models with abstract states (primality, motivation, problem-situation), they become unfalsifiable and circular."

### QA-RML Resolution

RML **never allows arbitrary state enrichment**. Abstract objects must be:

1. **Derived via explicit operators** (not injected freely)
2. **Witnessed by verifiable evidence**
3. **Constrained by failure taxonomy**

```python
# ILLEGAL: Ad-hoc state injection (what the paper warns about)
state.add_property("is_prime", True)  # ← No derivation, unfalsifiable

# LEGAL: Derived invariant with witness (QA-RML approach)
primality_cert = InvariantDerivation(
    invariant="Prime(n)",
    derivation_operator=Δ_primality_test,
    witness=MillerRabinWitness(n, witnesses=[2,3,5,7]),
    falsifiable=True  # If witness fails, invariant is rejected
)
```

**Failure algebra enforces this** (from `qa_certificate.py`):

```python
class FailType(Enum):
    # This is the paper's concern, now formalized
    ADHOC_STATE_INJECTION = "abstract_object_without_derivation"

    # Legitimate failures have derivation
    OUT_OF_BOUNDS = "coordinate_exceeds_N"
    INVARIANT_VIOLATION = "derived_property_violated"
    SCC_UNREACHABLE = "topological_disconnection"
    GENERATOR_INSUFFICIENT = "no_path_under_generators"
```

---

## 4. Formal Comparison Table

| Dimension | World Models | QAWM | QA-RML |
|-----------|-------------|------|--------|
| **Primary query** | "What next?" | "Reachable in k?" | "Why (un)reachable?" |
| **Output type** | Next state | Boolean predicate | Certificate object |
| **Training signal** | Prediction loss | Multi-task (legal, fail, reach) | Structural queries (bandit) |
| **Handles impossibility** | ❌ No | Partial (fail_type) | ✅ Full obstruction certs |
| **Handles invariants** | ❌ Implicit only | Partial (21-element packet) | ✅ Explicit derivation |
| **Handles strategy** | ❌ No | ❌ No | ✅ Key steps + compression |
| **Handles problem-situation** | ❌ No | ❌ No | ✅ Gap + resolution certs |
| **Falsifiability** | Prediction error | AUROC metrics | Derivation constraints |
| **Oracle efficiency** | N/A | N/A | 4.20 vs 2.97 (1.41× gain) |

---

## 5. The Three-Layer Understanding Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: UNDERSTANDING (QA-RML)                                │
│  ─────────────────────────────────────────────────────────────  │
│  • Obstruction certificates (why impossible)                    │
│  • Minimal path witnesses (why this path)                       │
│  • Strategy/key-step identification                             │
│  • Problem-situation → resolution mapping                       │
│  • Compression metrics (outline vs full trace)                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: STRUCTURE (QAWM)                                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Reachability predicates (return_in_k)                        │
│  • Failure classification (5 types)                             │
│  • Legality prediction                                          │
│  • 21-element invariant packet                                  │
│  • AUROC 0.836 on structural queries                            │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: DYNAMICS (World Models)                               │
│  ─────────────────────────────────────────────────────────────  │
│  • State → Next-state prediction                                │
│  • Trajectory simulation                                        │
│  • Forward dynamics only                                        │
│  • No impossibility/obstruction handling                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Alignment

### 6.1 Existing Components Map

| Paper Concept | QA Implementation | File |
|--------------|-------------------|------|
| World model | `QAState` + generators | `qa_oracle.py` |
| Structural learning | `QAWM` multi-task heads | `qawm.py` |
| Reachability oracle | `return_in_k()` | `qa_oracle.py:return_in_k` |
| Obstruction evidence | `ObstructionEvidence` | `qa_certificate.py` |
| Path witness | `MoveWitness` chain | `qa_certificate.py` |
| Failure taxonomy | `FailType` enum | `qa_certificate.py` |
| Policy learning | `RMLPolicy` | `rml_policy.py` |
| Efficiency metric | Normalized success/call | `evaluate_paper3.py` |

### 6.2 New Components Needed (from paper)

```python
# Add to qa_certificate.py

class UnderstandingCertificate:
    """Full understanding certificate (paper's desiderata)"""
    schema: str = "qa_understanding_cert/v1"

    # From Layer 1 (optional)
    micro_trace: Optional[List[MoveWitness]]

    # From Layer 2 (required)
    reachability: bool
    fail_type: Optional[FailType]

    # Layer 3: Understanding (new)
    derived_invariants: Dict[str, Scalar]
    key_steps: List[KeyStep]
    strategy: Optional[Strategy]
    problem_gap: Optional[str]
    explanation_path: List[str]
    compression_ratio: float

    # Falsifiability
    derivation_witnesses: List[DerivationWitness]

class KeyStep:
    """Non-routine step with necessity certificate"""
    index: int
    description: str
    necessity_witness: ObstructionEvidence  # What fails without this step
    compression_contribution: float

class Strategy:
    """High-level proof/solution strategy"""
    type: str  # e.g., "involution_parity", "invariant_counting"
    key_insight: str
    prerequisite_knowledge: List[str]

class ProblemSituationCert:
    """Popper-style problem-situation understanding"""
    gap: str  # What the prior theory couldn't explain
    target_phenomenon: str
    resolution: str  # How new theory closes the gap
    necessity: str  # Why this resolution is forced
```

---

## 7. The QA Understanding Thesis (Formal Statement)

Let:
- `G_m` = microstate transition graph (world model)
- `Δ` = derivation operators producing invariants from states/traces
- `G_e` = explanation graph over derived objects
- `k` = depth bound

**Definition (QA-Understanding)**:

```
Understanding(system, query) holds iff ∃ certificate C such that:

1. REACHABILITY: C contains path_witness in G_m OR obstruction_evidence
2. INVARIANT: C.derived_invariants are computed via explicit Δ operators
3. COMPRESSION: |C.explanation_path| << |C.micro_trace| (if micro_trace exists)
4. NECESSITY: Each key_step has a counterfactual obstruction witness
5. FALSIFIABLE: All abstract claims in C have derivation_witnesses
```

**Theorem (RML Completeness)**:

> If `return_in_k(s, T, k, G) = True`, then RML can produce a path witness.
> If `return_in_k(s, T, k, G) = False`, then RML can produce an obstruction certificate.
> World models can only do the first.

---

## 8. Comparison to JEPA

From `qa_lab/qa_jepa_encoder.py`, JEPA variants map to Layer 1–2:

| JEPA Variant | QA Mapping | Layer |
|-------------|-----------|-------|
| I-JEPA | Image → QA tuple prediction | 1 |
| V-JEPA | Video frame → orbit prediction | 1 |
| LeJEPA | Multi-modal world model | 1 |
| TS-JEPA | Time series → QA dynamics | 1 |
| TD-JEPA | Temporal difference (partial reachability) | 1–2 |

**Gap**: No JEPA variant produces obstruction certificates or understanding in the paper's sense.

**RML + JEPA hybrid**: Use JEPA for efficient Layer 1 encoding, RML for Layer 3 understanding.

---

## 9. Benchmark Suite (Paper → QA)

### Benchmark 1: Domino Primality
```python
def domino_understanding_benchmark(n: int) -> UnderstandingCertificate:
    """
    Input: Integer n (encoded as domino input length)
    Expected output:
    - If Prime(n): ObstructionCert showing no divisor path
    - If Composite(n): PathWitness + divisor identification
    - Both: Explicit Prime(n) invariant derivation
    """
```

### Benchmark 2: Proof Understanding
```python
def proof_understanding_benchmark(proof_trace: List[InferenceStep]) -> UnderstandingCertificate:
    """
    Input: Valid proof trace
    Expected output:
    - Strategy identification
    - Key step extraction (with necessity witnesses)
    - Compression ratio > 3x
    - Outline generation
    """
```

### Benchmark 3: Theory Understanding
```python
def theory_understanding_benchmark(
    old_theory: GeneratorSet,
    new_theory: GeneratorSet,
    phenomenon: TargetClass
) -> UnderstandingCertificate:
    """
    Input: Theory upgrade (e.g., Classical → Bohr)
    Expected output:
    - Obstruction under old_theory
    - Path witness under new_theory
    - Problem-situation certificate (gap + resolution)
    """
```

---

## 10. One-Sentence Summary

> **World models predict what happens.**
> **QAWM predicts what can happen.**
> **RML certifies why some things cannot happen—and that's understanding.**

---

## References

1. Gupta & Pruthi (2025). "Beyond World Models: Rethinking Understanding in AI Models." arXiv:2511.12239v1.
2. QA Transition System (Paper 1). `qa_oracle.py`
3. QAWM World Model (Paper 2). `qawm.py`
4. RML Meta-Policy (Paper 3). `rml_policy.py`
5. QA Certificate Schema. `qa_certificate.py`
6. JEPA Integration. `qa_lab/qa_jepa_encoder.py`
