# Axiom Stratification Theorem (QA Form)

**Theorem ID:** QA_AXIOM_STRATIFICATION_v1
**Status:** Formalized
**Date:** 2026-01-24

---

## Theorem Statement

Let R be a reasoning task (e.g., a Putnam problem, an ML research question).

There exist three distinct but composable strata:

### Stratum 1: Intuition Stratum (IS)

- **Definition:** A compressed, observer-projected path sketch over R
- **Properties:**
  - Not required to be replayable
  - May skip "obvious" steps
  - Human-generated or LLM-generated
- **QA Interpretation:** Observer projection (non-executable)

### Stratum 2: Formalization Stratum (FS)

- **Definition:** An expanded, generator-restricted search over a discrete state space S
- **Properties:**
  - Each step must be locally admissible
  - Generator set G defines allowed transitions
  - Budget constraints (time, tokens, steps) may apply
- **QA Interpretation:** Discrete reachability under generators

### Stratum 3: Kernel Stratum (KS)

- **Definition:** A deterministic invariant oracle that accepts or rejects complete traces
- **Properties:**
  - Binary decision: ACCEPT or REJECT
  - No partial credit
  - Final arbiter of validity
- **QA Interpretation:** Invariant oracle / validator

---

## Main Result

**Theorem (Stratification):**

A solution to R is **valid** if and only if there exists a finite generator trace

```
tau = (g_1, g_2, ..., g_n)
```

such that:

1. `tau` is reachable from the initial state in FS, AND
2. The resulting state is accepted by KS.

---

## Asymmetry Properties

The three strata have **strict asymmetric implications**:

| Implication | Status |
|-------------|--------|
| IS plausibility => FS reachability | **FALSE** |
| FS reachability => KS acceptance | **FALSE** |
| KS acceptance => Valid solution | **TRUE** |
| IS plausibility => Valid solution | **FALSE** |

### Corollary 1: Intuition is Not Proof

A plausible-sounding argument (IS) does not guarantee:
- That it can be formalized (FS)
- That the formalization is correct (KS)

### Corollary 2: Formalization is Not Verification

A formalized proof attempt (FS) may still be:
- Rejected by the kernel (type error, missing lemma)
- Blocked by invariant violations

### Corollary 3: Kernel Dominance

KS acceptance is **sufficient** for validity, regardless of:
- Whether the human found the argument "intuitive"
- Whether the proof is "elegant" or "brute force"
- How long the formalization took

---

## Generalization Beyond Theorem Proving

This stratification applies to multiple domains:

| Domain | IS (Intuition) | FS (Formalization) | KS (Kernel) |
|--------|----------------|-------------------|-------------|
| **Math (Axiom)** | "Obvious" lemmas | Lean tactics | Lean kernel |
| **ML Training** | Loss intuition | Training code | Hardware execution |
| **AI Research (Stanford)** | Research ideas | Experiment code | Empirical results |
| **Physics** | Physical intuition | Equations | Experimental data |
| **QA System** | Semantic reasoning | Generator moves | Invariant checks |

---

## Failure Mode Analysis

Each stratum has characteristic failures:

### IS Failures (Intuition)
- "Obvious" step is actually false
- Handwavy argument hides complexity
- Conceptual gap unnoticed

### FS Failures (Formalization)
- `FORMALIZATION_GAP`: No generator for "obvious" step
- `CASE_EXPLOSION`: Combinatorial blowup
- `REWRITE_BLOCKED`: Equality not in usable form
- `BUDGET_EXHAUSTION`: Timeout before completion

### KS Failures (Kernel)
- `KERNEL_VIOLATION`: Type error, invalid proof term
- `INVARIANT_MISMATCH`: Constraint not satisfied
- `SOUNDNESS_VIOLATION`: Axiom misuse

---

## QA Certificate Schema

This theorem justifies a certificate type:

```json
{
  "certificate_type": "STRATIFICATION_WITNESS",
  "task": "description of R",
  "intuition_sketch": "optional IS-level description",
  "formal_trace": ["g_1", "g_2", "...", "g_n"],
  "generator_set": ["list of allowed generators"],
  "kernel_verdict": "ACCEPT | REJECT",
  "stratum_reached": "IS | FS | KS",
  "failure_mode": "null | failure type if rejected"
}
```

---

## Citation

```
Axiom Stratification Theorem (QA Form)
QA_AXIOM_STRATIFICATION_v1
Signal Experiments Research Group, 2026

Based on:
- Axiom AI "From Seeing Why to Checking Everything"
- Stanford "Execution-Grounded Automated AI Research" (arXiv:2601.14525)
```
