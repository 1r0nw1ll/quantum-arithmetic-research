# [135] QA Pythagorean Tree Cert

**Schema**: `QA_PYTHAGOREAN_TREE_CERT.v1`
**Directory**: `qa_alphageometry_ptolemy/qa_pythagorean_tree_cert_v1/`
**Validator**: `qa_pythagorean_tree_cert_validate.py`

## What It Certifies

The three **Barning-Hall/Berggren generator moves** in QA direction space, and the theorem linking each move to the Egyptian fraction first step k:

| Move | Formula | k of child | Algebraic reason |
|------|---------|------------|------------------|
| **M_A** | (d,e) → (2d−e, d) | k = 2 | (2d−e)/d = 2 − e/d ∈ (1,2) |
| **M_B** | (d,e) → (2d+e, d) | k = 3 | (2d+e)/d = 2 + e/d ∈ (2,3) |
| **M_C** | (d,e) → (d+2e, e) | k ≥ 4 | (d+2e)/e = d/e + 2 > 3 |

Each move preserves three properties:
1. **gcd = 1** — proof by Euclidean step: gcd(2d±e, d) = gcd(±e, d) = 1; gcd(d+2e, e) = gcd(d, e) = 1
2. **Opposite parity** (d−e odd) — 2d is even, so 2d±e and d±2e inherit parity from e or d
3. **Pythagorean triple**: F²+C²=G² for (F,C,G)=(d²−e², 2de, d²+e²)

## The k-Identification Theorem

> **For any PPT direction (d',e'), the first step k = ⌈d'/e'⌉ of the greedy Egyptian fraction expansion uniquely identifies which of the three moves generated it:**
>
> k=2 ↔ M_A child;  k=3 ↔ M_B child;  k≥4 ↔ M_C child

This is the bridge between cert [134] (Egyptian fraction expansion = *descent*) and cert [135] (tree generator moves = *ascent*). They are exact inverses.

**Parent recovery from child (d',e'):**
- k=2 → parent = (e', 2e'−d') [inverse M_A]
- k=3 → parent = (e', d'−2e') [inverse M_B]
- k≥4 → parent = (d'−2e', e') [inverse M_C]

## Root Uniqueness

(d,e) = (2,1) is the **unique root**: all three inverses yield e=0 or d=0 — no valid parent.

```
inv_A(2,1) = (1, 0)  — e=0, invalid
inv_B(2,1) = (1, 0)  — e=0, invalid
inv_C(2,1) = (0, 1)  — d=0, invalid
```

Every other PPT direction has exactly one valid parent (exactly one of the three inverses produces a valid PPT direction with smaller d).

## Fundamental Example

From root (2,1) — the 3-4-5 direction:

| Move | Child (d',e') | Triple | k |
|------|--------------|--------|---|
| M_A | (3,2) | 5-12-13 | 2 |
| M_B | (5,2) | 21-20-29 | 3 |
| M_C | (4,1) | 15-8-17 | 4 |

## Checks

| ID | Check |
|----|-------|
| PT_1 | `schema_version == 'QA_PYTHAGOREAN_TREE_CERT.v1'` |
| PT_2 | `d > e > 0` |
| PT_3 | `gcd(d,e) = 1` |
| PT_4 | `d−e` odd (PPT condition) |
| PT_A | M_A child (2d−e,d): gcd=1, parity ok, k=2, parent recovers |
| PT_B | M_B child (2d+e,d): gcd=1, parity ok, k=3, parent recovers |
| PT_C | M_C child (d+2e,e): gcd=1, parity ok, k≥4, parent recovers |
| PT_ROOT | (2,1) has no valid parent (all inverses invalid) |
| PT_W | ≥3 witnesses (witness fixture) |
| PT_F | Fundamental (d=2,e=1) present |

## Fixtures

| Fixture | Type | Expected |
|---------|------|----------|
| `pt_pass_fundamental.json` | Root (2,1) + its 3 children + root check | PASS |
| `pt_pass_witnesses.json` | 5 witnesses at depths 2–3, all three move types | PASS |

## Connection to Prior Art Convergence Stack

This cert is the formal intersection of four traditions:

1. **Barning (1963) / Hall (1970)**: The three 3×3 matrices on Pythagorean triples — this cert establishes their action in *direction space* (d,e) with the simple formulas above
2. **H. Lee Price (2008)**: His M1/M2/M3 Fibonacci-box column operations are exactly these three direction moves. Price's "Pythagorean tree" = Koenig tree in HAT/direction coordinates. Cert [132] established HAT=e/d; this cert establishes the three navigation moves.
3. **Ben Iverson (Pyth-1)**: Koenig series generates all prime triangles via these same three moves. The Koenig tree and the Barning-Hall/Price tree are the same object.
4. **Egyptian fractions [134]**: The greedy expansion descent (k=2,3,≥4) is the inverse of these three ascent moves. The whole tree is traversed by these two complementary operations.

**The convergence**: Babylon → Egypt → Euclid → Ben Iverson → Price → Wildberger → this cert. All recovered the same static geometry. Ben's unique contribution remains the T-operator and dynamics.

## Sources

- Barning, F.J.M. (1963). "On Pythagorean and quasi-Pythagorean triangles"
- Hall, A. (1970). "Genealogy of Pythagorean triads"
- H. Lee Price (2008). "The Pythagorean Tree: A New Species" — M1/M2/M3 via HAT Fibonacci boxes
- Ben Iverson, *Pythagoras and the Quantum World Vol. 1* — Koenig series
- Cert [132] QA_HAT (establishes HAT₁=e/d)
- Cert [134] QA_EGYPTIAN_FRACTION (inverse descent operation)
