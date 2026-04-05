# QA State Space: Orbits, Families, and Structure

## The Three Orbit Families

Every state `(b, e)` in QA belongs to exactly one of three orbit families. This classification is determined by the Q(√5) norm `f(b, e) = b² + be - e²` and its 3-adic valuation.

### Singularity
- **Size**: 1 state (the fixed point)
- **Mod-9**: state (9, 9) — maps to (0, 0) under mod reduction
- **Character**: `F(0, 0) = (0, 0)` — the generator F fixes it
- **Dynamics**: no generator moves you out; this is pure quiescence
- **SVP analogue**: absolute stillness, no vibration

### Satellite
- **Size**: 8 states
- **Character**: 3D symmetric structure; μ-pairing gives 2-cycles for off-diagonal states
- **Q(√5) norm**: `v₃(f) ≥ 2` (divisible by 9)
- **Dynamics**: partial dynamics; some generators act, some fail
- **SVP analogue**: transitional resonance; a system in flux between modes

### Cosmos
- **Size**: 72 states (three orbits of 24 each, mod-9; 504 states under mod-24)
- **Character**: 1D linear structure; full generator dynamics; norm pair classes {1,8}, {4,5}, {2,7}
- **Q(√5) norm**: `v₃(f) = 0` (not divisible by 3)
- **Sub-families** (mod-9): Fibonacci {1,8}, Lucas {4,5}, Phibonacci {2,7}
- **Dynamics**: full orbit traversal; all generators produce lawful transitions or classified failures
- **SVP analogue**: full harmonic resonance; stable, structured, reproducible pattern

---

## The Q(√5) Structure

The invariant `f(b, e) = b² + be - e²` is the norm in the number field **Q(√5)**, where φ = (1+√5)/2 is the golden ratio.

Key facts:
- `f(b, e) = N(b + eφ)` — it is literally the algebraic norm of `b + eφ` in Z[φ]
- `N(φ²) = 1` → `f` is invariant under the generator `T = F²` (the QA Fibonacci matrix squared)
- `det([[b, d], [e, a]]) = f(b, e)` — orbit family is encoded in the 2×2 determinant
- The five Pythagorean families are exactly the φ-orbits in `GF(9)² = (Z[φ]/3Z[φ])²`

This is why the five Pythagorean triple families (Fibonacci, Lucas, Phibonacci, Tribonacci, Ninbonacci) emerge from QA naturally — they are the orbit structure of the golden ratio acting modulo 9.

---

## The Orbit Table (mod-24)

| Orbit type | Orbit length | Count | Notes |
|-----------|-------------|-------|-------|
| Cosmos (length 12) | 12 | 42 orbits | 504 states total |
| Cosmos (length 6) | 6 | 8 orbits | 48 states total |
| Cosmos (length 4) | 4 | 2 orbits | 8 states |
| Cosmos (length 3) | 3 | 5 orbits | 15 states |
| Satellite | 8 | 1 orbit | 8 states |
| Singularity | 1 | 1 | 1 state |

Total: 576 = 24² states accounted for.

---

## Failure Modes by Generator

Every generator has a defined failure behavior. These are not errors — they are **first-class information** about why a transition is impossible.

### σ (sigma): e → e+1
- **Success**: e < N
- **Failure**: `OUT_OF_BOUNDS` when e = N (top of lattice)

### μ (mu): swap (b,e) → (e,b)
- **Success**: always (on square Caps, swap is always in-bounds)
- **Failure**: none under standard Caps(N,N)

### λ (lambda): scale (b,e) → (kb, ke)
- **Success**: kb ≤ N and ke ≤ N
- **Failure**: `OUT_OF_BOUNDS` when scaled coordinates exceed N
- **Failure**: `ZERO_DENOMINATOR` when k = 0 is supplied

### ν (nu): halve (b,e) → (b/2, e/2)
- **Success**: b and e both even
- **Failure**: `PARITY` when either coordinate is odd

---

## The Gate Policy

Every QA certificate must pass through six validation gates in sequence:

| Gate | Function | What it catches |
|------|----------|----------------|
| Gate 0 | Mapping protocol intake | Missing or malformed protocol declaration |
| Gate 1 | Schema validation | Structural cert errors |
| Gate 2 | Generator uniqueness + invariant resolution | Duplicate names, unresolved invariant refs |
| Gate 3 | Failure algebra completeness | Empty failure taxonomy |
| Gate 4 | Invariant diff check | Silent invariant violations without a fail_type |
| Gate 5 | Canonical hash / Merkle integrity | Tamper-evidence; closes the replay chain |

**Never truncate to [0,1,2,3].** Gates 4 and 5 are what make QA certs externally auditable. A cert without them cannot be trusted.

---

## Logging Requirements

Every QA operation must log three fields at minimum:
- `move`: which generator was applied
- `fail_type`: the failure type emitted (null if success)
- `invariant_diff`: the change in invariant packet values

These three fields together make every state transition **fully reproducible** from the log alone.

---

## Reading the Orbit Graph

A QA orbit graph is a directed graph where:
- **Nodes** = states in Caps(N, N)
- **Edges** = legal generator applications
- **Strongly Connected Components (SCCs)** = sets of mutually reachable states

Key theorem (proved):
> Under generators {σ, μ, λ₂, ν}, the SCCs are exactly the μ-orbits: N diagonal singletons and (N²-N)/2 off-diagonal pairs. Thus #SCC = (N² + N)/2.

This tells you that **μ alone** creates the basic connectivity structure. σ, λ, and ν extend it.

---

## Source References

- Orbit structure: `CLAUDE.md` § Core Mathematical Framework
- Q(√5) structure: `memory/MEMORY.md` § Q(√5) Algebraic Structure
- Pythagorean families: `papers/in-progress/pythagorean-families/paper.tex`
- Kernel cert: `qa_alphageometry_ptolemy/qa_core_spec/`
- Mod-24 orbit table: `memory/MEMORY.md` § QA Synthetic Benchmark
