# QA Canonical Reference
**Version:** 1.0  
**Date:** December 29, 2024  
**Status:** IMMUTABLE — All QA-RML work must reference this document

---

## Purpose

This document is the **single source of truth** for Quantum Arithmetic (QA) definitions, formulas, and constraints. It is treated as an **axiom file** that must be imported, not contextual knowledge to be approximated.

**Critical Rules:**
- ❌ **No redefinition** of any symbol or formula
- ❌ **No simplification** or "close enough" variants
- ❌ **No implicit constraints** — everything is explicit
- ✅ **All QA work must cite this document** for definitions

---

## 1. State Space Definition

### 1.1 Primitive Coordinates

A QA state is defined by **primitive coordinates**:
```
(b, e) ∈ ℤ₊² 
```
where `b, e > 0` (strictly positive integers).

### 1.2 Derived Coordinates

From primitives `(b, e)`, we derive:
```
d = b + e
a = e + d = b + 2e
```

**Critical Constraint:** `d` and `a` are **always derived**. They are **never** independent degrees of freedom.

### 1.3 The 21-Element Invariant Packet

From `(b, e)`, the complete QA invariant structure is:

#### Base Squares
```
B = b²
E = e²
D = d² = (b+e)²
A = a² = (b+2e)²
```

#### Products and Cross-Terms
```
X = e·d
C = 2·e·d
F = b·a
```

#### Combined Invariants
```
G = D + E = d² + e²
L = (C·F)/12        [exact rational]
H = C + F
I = |C - F|         [positive difference]
J = d·b
K = d·a
W = X + K = d(e+a)
Y = A - D
Z = E + K = e² + (a·d)
h² = d²·a·b         [semi-minor diameter squared]
```

**Note on L:** This is an exact rational `Fraction(C*F, 12)`, not a floored integer.

**Note on I:** This is `abs(C - F)`, enforcing positivity.

**Note on h:** Store as `h²` (exact integer). Derive `h = sqrt(h²)` only for display.

---

## 2. Generator Algebra

### 2.1 Generator Definitions

Generators are **partial functions** on state space:

#### σ (sigma): Growth
```
σ(b, e) = (b, e+1)
```
**Legality:** `e+1 ≤ N` (where N is the Caps bound)  
**Failure:** OUT_OF_BOUNDS if `e+1 > N`

#### μ (mu): Swap
```
μ(b, e) = (e, b)
```
**Legality:** Always legal on square Caps(N,N)  
**Failure:** OUT_OF_BOUNDS only on non-square caps

#### λ₂ (lambda-2): Scaling
```
λ₂(b, e) = (2b, 2e)
```
**Legality:** `2b ≤ N` and `2e ≤ N`  
**Failure:** OUT_OF_BOUNDS if either exceeds N

**Note:** λ is a family of moves `λₖ(b,e) = (kb, ke)`. For canonical experiments, we use `k=2`.

#### ν (nu): Contraction
```
ν(b, e) = (b/2, e/2)   if b,e both even
         = FAIL        otherwise
```
**Legality:** Both `b` and `e` must be even  
**Failure:** PARITY if either is odd

### 2.2 Generator Set Notation

```
Σ_full = {σ, μ, λ₂, ν}
```

For specific experiments, we use subsets:
- `Σ₁ = {σ, λ₂}`
- `Σ₂ = {σ, μ, λ₂}`
- `Σ₃ = {σ, μ, λ₂, ν}`

---

## 3. Phase System

### 3.1 Phase Annotations

From derived coordinate `a`, we compute:

#### φ₉ (phi-9): Digital Root
```
φ₉(a) = digital_root(a)
      = ((a-1) mod 9) + 1    for a > 0
```

#### φ₂₄ (phi-24): Modular Residue
```
φ₂₄(a) = a mod 24
```

### 3.2 Fixed-q Definitions

A **phase constraint** `q` restricts the state space:

- `q = "none"`: No phase constraint (unconstrained)
- `q = "phi_9"`: Fix `φ₉(a)` constant
- `q = "phi_24"`: Fix `φ₂₄(a)` constant
- `q = "both"`: Fix both `φ₉` and `φ₂₄`
- `q = "family"`: Fix invariant packet (typically N, C, F)

**In Papers 1-2:** We use `q = "none"` (unconstrained) for main results.

---

## 4. Caps Lattice

### 4.1 Definition
```
Caps(N,N) = {(b,e) ∈ ℤ₊² : 1 ≤ b ≤ N, 1 ≤ e ≤ N}
```

**Cardinality:** `|Caps(N,N)| = N²`

### 4.2 Canonical Experiments

- **Caps(30,30):** 900 states
- **Caps(50,50):** 2500 states

---

## 5. Failure Taxonomy

### 5.1 Failure Types

When a generator `g` is illegal on state `s`, it returns a **typed failure**:

```
Fail(s, g) = (move, fail_type, ΔI)
```

where `fail_type ∈ {OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION}`

### 5.2 Failure Semantics

| Type | Cause | Example |
|------|-------|---------|
| OUT_OF_BOUNDS | Exceeds Caps lattice boundary | σ when e=N, λ₂ when 2b>N |
| PARITY | Violates even/odd constraint | ν when b or e is odd |
| PHASE_VIOLATION | Breaks fixed-q constraint | Any move changing φ₂₄ under q="phi_24" |
| INVARIANT | Violates derived constraint | (Rare in Caps; occurs in other QA universes) |
| REDUCTION | Violates non-reduction axiom | (Not active in Caps lattice) |

**Critical Property:** Failures are **deterministic and reproducible**, not stochastic.

---

## 6. Key Theorems (Paper 1)

### 6.1 SCC Count under μ-Pairing

**Theorem:**
```
For Caps(N,N) with generators {σ, μ, λ₂}:

#SCC = (N² + N)/2
max|SCC| = 2
```

**Proof sketch:** μ creates 2-cycles on off-diagonal states `(b,e), b≠e`. Diagonal states `(b,b)` are fixed points. Count: N diagonal + (N²-N)/2 pairs.

### 6.2 Edge Count Formulas

**Theorem:**
```
For Caps(N,N):

|σ-edges| = N(N-1)
|μ-edges| = N²
|λ₂-edges| = ⌊N/2⌋²
|ν-edges| = ⌊N/2⌋²
```

### 6.3 SCC Monotonicity

**Lemma:**
```
If Σ₁ ⊆ Σ₂, then #SCC(G_Σ₂) ≤ #SCC(G_Σ₁)
```

Adding generators can only merge components, never split them.

### 6.4 Connectivity Transition

**Observed Result (Caps(30,30)):**
```
{σ, λ₂}:         #SCC = 900  (all singletons)
{σ, μ, λ₂}:      #SCC = 465  (μ-pairs + diagonal)
{σ, μ, λ₂, ν}:   #SCC = 1    (full connectivity)
```

This is a **topological phase transition** induced by generator expansion.

---

## 7. Reachability Definitions

### 7.1 Return-in-k

For state `s`, target class `R* ⊆ S`, and horizon `k`:

```
return_in_k(s → R*, k, Σ) := 
    ∃ (g₁, ..., g_T) ∈ Σᵀ, T ≤ k : 
        g_T ∘ ⋯ ∘ g₁(s) ∈ R*
```

**Property:** This is **decidable** via bounded BFS.

### 7.2 Component Membership

States `s, t` are in the same SCC iff:
```
∃ forward path from s to t using Σ
AND
∃ reverse path from t to s using Σ
```

---

## 8. Naming Conventions (Frozen)

### 8.1 Symbols

| Symbol | Meaning | Never Redefine |
|--------|---------|----------------|
| b, e | Primitive coordinates | ✓ |
| d, a | Derived coordinates | ✓ |
| N | Caps lattice bound | ✓ |
| C, F | Cross-products (2ed, ba) | ✓ |
| σ, μ, λ, ν | Generators | ✓ |
| φ₉, φ₂₄ | Phase functions | ✓ |
| Σ | Generator set | ✓ |
| q | Phase constraint | ✓ |

### 8.2 Reserved Terms

- **"Primitive"**: (b, e) only
- **"Derived"**: d, a, and all invariants
- **"Female tuple"**: (Not used in current formalism; historical)
- **"Canonical"**: Refers to this document
- **"Oracle"**: Ground truth transition system (qa_oracle.py)

---

## 9. Non-Negotiable Constraints

### 9.1 Immutable Rules

1. **d = b+e, a = b+2e** — Never independent
2. **C ≠ F** — This is a theorem, not an assumption
3. **I = |C-F|** — Must be positive
4. **L = (C·F)/12** — Exact rational, not floored
5. **Failures are theorems** — Not noise, not probabilistic
6. **State space is integer manifold** — No continuous relaxation

### 9.2 Forbidden Operations

- ❌ Treating (b,e,d,a) as independent 4-tuple
- ❌ Simplifying L to an integer without explicit rounding
- ❌ Assuming C=F or C≈F
- ❌ Treating failures as stochastic
- ❌ Redefining generators without explicit versioning

---

## 10. Usage Protocol

### 10.1 Loading This File

At the start of any QA session, state:
```
"Load qa_canonical.md. All definitions are per this file.
Do not redefine, simplify, or approximate."
```

### 10.2 Citing in Papers

```latex
All QA definitions follow \cite{qa_canonical}.
State space and generators are defined in §1-2 of the canonical reference.
```

### 10.3 Code Implementation

```python
# qa_canonical.py
"""
Implements canonical QA definitions from qa_canonical.md
Version 1.0
"""

def construct_qa_state(b: int, e: int) -> QAState:
    """Builds state with exact 21-element packet per canonical spec"""
    d = b + e
    a = e + d  # = b + 2e
    
    # Squares
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    
    # Products
    X = e * d
    C = 2 * e * d
    F = b * a
    
    # Combined
    G = D + E
    L = Fraction(C * F, 12)  # Exact rational
    H = C + F
    I = abs(C - F)  # Positive difference
    J = d * b
    K = d * a
    W = X + K
    Y = A - D
    Z = E + K
    h2 = D * a * b
    
    # Phases
    phi_9 = digital_root(a)
    phi_24 = a % 24
    
    return QAState(
        b=b, e=e, d=d, a=a,
        B=B, E=E, D=D, A=A,
        X=X, C=C, F=F, G=G,
        L=L, H=H, I=I, J=J,
        K=K, W=W, Y=Y, Z=Z, h2=h2,
        phi_9=phi_9, phi_24=phi_24
    )
```

---

## 11. Version Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 29, 2024 | Initial canonical reference |

**Change Protocol:** Any modification to this file requires:
1. Version bump
2. Explicit changelog entry
3. Validation that all dependent code still passes tests
4. Re-verification of Paper 1 closed-form predictions

---

## 12. Validation Checksums

**Expected values for Caps(30,30) under Σ₃ = {σ,μ,λ₂,ν}:**
```
#States:  900
#Edges:   2220
#Fail:    1380
#SCC:     1
Max-SCC:  900
```

If these values don't match, the implementation is incorrect.

---

## Appendix: Why This Document Exists

QA is **not symbolic math** — it is:
- A geometric control theory
- A reachability algebra  
- A topology of failure modes

That means:
- **Definition drift = changing the manifold**
- **"Close enough" = false theorem**
- **Implicit assumptions = non-reproducible results**

This canonical reference ensures:
✅ Deterministic definitions  
✅ Reproducible experiments  
✅ Auditable claims  
✅ Reviewer-proof rigor

---

**END OF CANONICAL REFERENCE**

All QA-RML work must be consistent with this document.
Any conflict indicates an error in the derivative work, not this specification.
