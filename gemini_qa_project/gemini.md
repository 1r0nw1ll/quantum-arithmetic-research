# Gemini Guidelines for Quantum Arithmetic (QA)

This document provides guidelines for working with the Quantum Arithmetic (QA) system, based on the canonical reference and other project documentation.

## Core Principles

1.  **Canonical Reference is King:** All work must strictly adhere to the definitions, formulas, and constraints outlined in `Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md`. This document is the single source of truth.
2.  **No Approximations:** Do not simplify, redefine, or approximate any symbols, formulas, or constraints from the canonical reference.
3.  **Explicit is Better than Implicit:** All constraints and definitions are explicitly stated in the canonical reference. Do not infer or assume any missing constraints.
4.  **Failures are Deterministic:** Failures are not random noise; they are deterministic, reproducible, and an integral part of the QA system.

## Session Start

At the beginning of any session working with QA, the following header must be used:

```
You must follow qa_canonical.md (QA Canonical Reference v1.0).
Do not redefine symbols, simplify formulas, or infer missing constraints.
If a needed definition is absent from qa_canonical.md, stop and ask.
All results must be consistent with the canonical checksums in section 12.
```

## Key Concepts

### State Space

*   **Primitive Coordinates:** `(b, e) ∈ ℤ₊²` (strictly positive integers)
*   **Derived Coordinates:** `d = b + e`, `a = b + 2e`. These are *always* derived and never independent.

### Invariant Packet

The 21-element invariant packet is derived from `(b, e)`. Key invariants include:

*   **Squares:** `B = b²`, `E = e²`, `D = d²`, `A = a²`
*   **Products:** `X = e*d`, `C = 2*e*d`, `F = b*a`
*   **Combined:** `G = D + E`, `L = (C*F)/12` (exact rational), `H = C + F`, `I = |C - F|`

### Generators

Generators are partial functions that act on the state space. The primary generators are:

*   `σ(b, e) = (b, e+1)` (Growth)
*   `μ(b, e) = (e, b)` (Swap)
*   `λ₂(b, e) = (2b, 2e)` (Scaling)
*   `ν(b, e) = (b/2, e/2)` (Contraction, only for even b, e)

### Control Theorems

*   **SCC Monotonicity:** Adding generators to a set can only decrease or maintain the number of Strongly Connected Components (SCCs).
*   **μ-Pairing:** The `μ` generator creates 2-cycles for off-diagonal states, significantly reducing the number of SCCs.
*   **Edge and Failure Counts:** The number of legal transitions and failures for each generator can be calculated with closed-form expressions.

## Workflow

1.  **Initialization:** Always start by loading and acknowledging the canonical reference.
2.  **State Definition:** Define states using primitive coordinates `(b, e)`.
3.  **Invariant Calculation:** Calculate the full 21-element invariant packet for each state.
4.  **Generator Application:** Apply generators to transition between states, respecting their legality conditions.
5.  **Analysis:** Analyze the resulting state space, including SCCs, reachability, and failure modes.
6.  **Verification:** Verify results against the canonical checksums and known theorems.

By following these guidelines, we can ensure that all work with the QA system is consistent, reproducible, and builds upon a solid, shared foundation.
