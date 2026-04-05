# QA Axioms (Canonical v1.0)

These are the foundational axioms of the QA system. They are **non-negotiable**: any AI session, any experiment, any engineering artifact must be consistent with these definitions. Do not redefine symbols, simplify formulas, or infer missing constraints. If something isn't covered here, stop and check the source.

Source: `QA_AXIOMS_BLOCK.md` + cert family [107] `QA_CORE_SPEC.v1`

---

## A1 — State Space

**Primitive coordinates**: `(b, e) ∈ Z_{>0}²`

**Derived coordinates**:
- `d = b + e`
- `a = b + 2e`

**Critical rule**: `d` and `a` are derived — never independent. The 4-tuple `(b, e, d, a)` has only two degrees of freedom: `b` and `e`.

The working domain is **Caps(N, N)** = `{(b, e) ∈ Z_{>0}² | 1 ≤ b ≤ N, 1 ≤ e ≤ N}`.

For applied work: N = 24 (mod-24). For theoretical/Pythagorean work: N = 9 (mod-9).

---

## A2 — Invariant Packet (21 elements)

Every state `(b, e)` carries a full invariant packet. These 21 values must remain self-consistent across all transitions:

| Symbol | Formula | Notes |
|--------|---------|-------|
| B | b² | |
| E | e² | |
| D | d² = (b+e)² | |
| A | a² = (b+2e)² | |
| X | e·d | |
| C | 2·e·d | |
| F | b·a | |
| G | D + E | |
| L | (C·F)/12 | **exact rational** |
| H | C + F | |
| I | \|C - F\| | **always positive**; C ≠ F is a theorem |
| J | d·b | |
| K | d·a | |
| W | X + K | |
| Y | A - D | |
| Z | E + K | |
| h2 | d²·a·b | |

**Non-negotiable**: L is an exact rational (not a float approximation). I = |C - F| is strictly positive (C = F is impossible — this is proved, not assumed).

---

## A3 — Generator Algebra

Four generators acting on Caps(N, N). Each is a **partial function** — it may fail, and every failure has a named type.

| Generator | Action | Precondition | Failure modes |
|-----------|--------|-------------|---------------|
| **σ (sigma)** | `(b, e) → (b, e+1)` | e + 1 ≤ N | `OUT_OF_BOUNDS` |
| **μ (mu)** | `(b, e) → (e, b)` (swap) | always legal on square Caps | — |
| **λ (lambda)** | `(b, e) → (kb, ke)` for scalar k ≠ 0 | kb ≤ N and ke ≤ N | `ZERO_DENOMINATOR`, `OUT_OF_BOUNDS` |
| **ν (nu)** | `(b, e) → (b/2, e/2)` | b and e both even | `PARITY` |

When a generator fails, it produces a **classified failure event** — not an exception, not a silent skip. Every failure type is deterministic and reproducible.

---

## A4 — Phase Functions

Two phase functions map states to their modular representative:

- `φ₉(a) = digital_root(a) = ((a - 1) mod 9) + 1` for a > 0
- `φ₂₄(a) = a mod 24`

These are the "reduction" functions that project states onto the working modulus.

---

## A5 — Failure Taxonomy

Every failure in the QA system belongs to one of five named types:

| Fail type | Generator | Cause |
|-----------|-----------|-------|
| `OUT_OF_BOUNDS` | σ, λ | Result coordinate exceeds N |
| `PARITY` | ν | Input coordinates not both even |
| `PHASE_VIOLATION` | any | Phase function output inconsistent with state |
| `INVARIANT` | any | Invariant packet self-consistency broken |
| `REDUCTION` | ν | Division produces non-integer result |

There is no "unknown error" in QA. Every failure is classified.

---

## A6 — Reachability

Reachability is defined by **bounded BFS** over the orbit graph:
- `max_depth = 24` (sufficient to traverse any cosmos orbit)
- Two states are reachable from each other iff they are in the same invariant equivalence class and connected by a legal generator path within the bound
- BFS produces a **plan witness**: an ordered sequence of generators + intermediate states

---

## A7 — Non-Negotiables (must be preserved in every engineering artifact)

1. `d` and `a` are derived from `(b, e)` — they are never free variables
2. `L = (C·F)/12` is exact rational — do not approximate
3. `I = |C - F| > 0` always — this is a theorem, not an assumption
4. Failures are deterministic — no stochastic or continuous relaxation
5. The working domain excludes zero: `{1, 2, ..., N}`, not `{0, 1, ..., N-1}`
6. Use `d*d` (multiplication) not `d**2` (power) in all substrate computations — CPython's `pow()` calls `libm` and may differ by 1 ULP

---

## Canonical Session Header

Paste this at the start of any AI session to enforce axiom compliance:

```
You must follow QA Canonical Reference v1.0.
Do not redefine symbols, simplify formulas, or infer missing constraints.
If a needed definition is absent from this reference, stop and ask.
All results must be consistent with the canonical axioms:
- State: (b,e) primitive; d=b+e, a=b+2e derived; never independent
- Generators: σ (e+1), μ (swap), λ (scale), ν (halve if even)
- Failures: OUT_OF_BOUNDS, PARITY, PHASE_VIOLATION, INVARIANT, REDUCTION
- L=(C*F)/12 exact rational; I=|C-F|>0 always; zero excluded from domain
```

---

## Source References

- Full axiom source: `QA_AXIOMS_BLOCK.md`
- Machine-checkable kernel spec: `qa_alphageometry_ptolemy/qa_core_spec/`
- Canonical kernel cert: `qa_alphageometry_ptolemy/qa_core_spec/fixtures/qa_core_spec_minimal_pass.json`
- Validator: `python qa_alphageometry_ptolemy/qa_core_spec/qa_core_spec_validate.py --self-test`
