# QA_AXIOM_LEDGER_v1.0

This appendix formalizes **Axiom AI** and **Execution-Grounded AI Research** as QA-subsumed reasoning systems.

The ledger treats these not as single papers, but as **control-theoretic programs** operating over formal proof/research state spaces.

---

## Module Graph (Top-Level)

```
QA_AXIOM_LEDGER
|
+-- QA_AXIOM_SYSTEM
|    State Space: Lean proof states
|    Generators: Tactics, lemma application, rewriting
|    Invariants: Lean kernel acceptance
|
+-- QA_AXIOM_PUTNAM2025
|    Benchmark: 12 Putnam 2025 problems
|    Constraints: Exam-time budget
|    Outputs: Proof traces and failure witnesses
|
+-- QA_AXIOM_STRATIFICATION
|    Intuition layer
|    Formalization layer
|    Kernel validation layer
|
+-- QA_AXIOM_GENERATOR_INJECTION
|    Barrier-crossing via generator extension
|
+-- QA_AXIOM_FAILURE_ALGEBRA
|    Formalized obstruction classes
|
+-- QA_EXECUTION_GROUNDED_RESEARCH (Stanford 2026)
     State Space: Research hypothesis space
     Generators: Experiment proposals, code implementations
     Invariants: Empirical validation (execution results)
     Failure Algebra: Ideas that don't compile/converge/work
```

---

## Source Materials

| Source | Description | QA Module |
|--------|-------------|-----------|
| Axiom AI - Territory Essays | "From Seeing Why to Checking Everything" | QA_AXIOM_STRATIFICATION |
| Axiom AI - Putnam 2025 | 12 competition math problems | QA_AXIOM_PUTNAM2025 |
| Stanford 2026 (arXiv:2601.14525) | Execution-Grounded Automated AI Research | QA_EXECUTION_GROUNDED |

---

## Interpretation

### Core Thesis

Both Axiom and the Stanford Execution-Grounded paper demonstrate the same QA principle:

> **Valid reasoning requires replayable traces checked by a deterministic invariant oracle.**

- **Axiom**: Lean kernel is the invariant oracle
- **Stanford**: Empirical execution results are the invariant oracle
- **QA**: Generalizes both as instances of certificate-controlled reachability

### Why This Matters

1. **Axiom proves** that "checking everything" is achievable for formal math
2. **Stanford proves** that "executing everything" is achievable for AI research
3. **QA explains** why both work, when they fail, and how to generalize them

---

## Certificate Types Derived from This Mapping

### 1. PROOF_GRIND_WITNESS

Use when math insight is small but formalization overhead is large.

**Certifies:**
- Proof is valid
- Trace length / case splits exceed threshold
- Failures dominated by "formality" obstructions

### 2. GENERATOR_INJECTION_WITNESS

Use when adding a generator (lemma, tactic, experiment type) crosses a barrier.

**Certifies:**
- Unreachable under G0
- Reachable under G1 ⊃ G0
- Barrier is generator-relative, not absolute

### 3. EXECUTION_GROUNDED_CERTIFICATE

Use for AI research proposals.

**Certifies:**
- Idea compiles (syntactic validity)
- Idea converges (semantic validity)
- Idea improves baseline (empirical validity)

---

## Relation to QA Gold Standard Mappings

| QA Mapping | Domain | Invariant Oracle | Gauge Freedom |
|------------|--------|------------------|---------------|
| Generalization Bounds | Statistics | Operator norms | Overparametrization |
| NeuralGCM | Physics | Conservation laws | Neural params |
| Sparse Attention | Efficiency | Entropy/rank bounds | Redundant heads |
| **Axiom AI** | **Formal Reasoning** | **Lean kernel** | **Tactic choice** |
| **Execution-Grounded** | **AI Research** | **Empirical results** | **Method choice** |

---

## Key QA-Axiom Unification Claims

1. **Difficulty is generator-relative**: Adding lemmas/tactics changes reachability
2. **Intuition is projection**: Human "seeing why" is observer-projected compression
3. **Verification is kernel acceptance**: Lean/experiments are the final arbiter
4. **Failure is first-class**: Formalization gaps, budget exhaustion, kernel rejection

---

## Files in This Ledger

| File | Description |
|------|-------------|
| `QA_AXIOM_LEDGER.md` | This document |
| `QA_AXIOM_STRATIFICATION_THEOREM.md` | Formal theorem statement |
| `QA_COMPARISON_TABLE.md` | QA vs Axiom vs AlphaGeometry |
| `CHECKING_IS_NOT_ENOUGH.md` | Philosophical manifesto |
| `../schemas/QA_FAILURE_ALGEBRA.json` | Failure type schema |
| `../QA_MAP__AXIOM_AI.yaml` | Full YAML module spec |

---

## Status

**Created:** 2026-01-24
**Version:** 1.0.0
**Status:** COMPLETE - Ready for integration with QA certificate spine
