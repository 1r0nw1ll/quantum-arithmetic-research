# [134] QA Egyptian Fraction Cert

**Schema**: `QA_EGYPTIAN_FRACTION_CERT.v1`
**Directory**: `qa_alphageometry_ptolemy/qa_egyptian_fraction_cert_v1/`
**Validator**: `qa_egyptian_fraction_cert_validate.py`

## What It Certifies

The **greedy Egyptian fraction expansion** of the HAT direction ratio `HAT₁ = e/d` for any primitive QA direction `(d,e)` with `gcd(d,e)=1` and `d > e > 0`:

> e/d = 1/k₁ + 1/k₂ + ... + 1/kₙ

where each `kᵢ = ⌈dᵢ/eᵢ⌉` (greedy), and the sequence of intermediate pairs `(d₀,e₀)=(d,e), (d₁,e₁), ..., (kₙ,1)` is the **Koenig descent path** from `(d,e)` to the root unit-fraction direction.

## Algorithmic Step

Given current direction `(dᵢ, eᵢ)`:

```
k = ⌈dᵢ/eᵢ⌉ = (dᵢ + eᵢ - 1) // eᵢ
next_num = k·eᵢ − dᵢ
next_den = k·dᵢ
g = gcd(next_num, next_den)
(dᵢ₊₁, eᵢ₊₁) = (next_den / g, next_num / g)
```

Terminal when `eᵢ = 1` — the direction is already a unit fraction `1/dᵢ`.

## Four Universal Properties

| Property | Statement |
|----------|-----------|
| **Sum** | Σ 1/kᵢ = e/d (exact, Fraction arithmetic) |
| **Greedy** | kᵢ = ⌈dᵢ/eᵢ⌉ at every step |
| **Increasing** | k₁ < k₂ < ... < kₙ (strictly) |
| **Primitive** | gcd(dᵢ,eᵢ) = 1 at every intermediate step |
| **Terminal** | Last step always has eₙ = 1 |

All four hold for every primitive `(d,e)` — verified exhaustively for `d,e ≤ 30`.

## Witnesses

| Direction | HAT₁ | Expansion | Length | Koenig path |
|-----------|------|-----------|--------|-------------|
| (2,1) | 1/2 | [1/2] | 1 | [(2,1)] |
| (3,2) | 2/3 | [1/2, 1/6] | 2 | [(3,2),(6,1)] |
| (5,2) | 2/5 | [1/3, 1/15] | 2 | [(5,2),(15,1)] |
| (5,3) | 3/5 | [1/2, 1/10] | 2 | [(5,3),(10,1)] |
| **(7,3)** | 3/7 | [1/3, 1/11, 1/231] | **3** | [(7,3),(21,2),(231,1)] |
| (4,3) | 3/4 | [1/2, 1/4] | 2 | [(4,3),(4,1)] |
| (8,3) | 3/8 | [1/3, 1/24] | 2 | [(8,3),(24,1)] |

The (7,3) witness is the only depth-3 expansion in the table — it also shows a non-trivial intermediate direction (21,2) which is itself a valid QA direction. Corrected 2026-07-06: its triple is (F,C,G)=(437,84,445) (F²+C²=445²=198025 ✓), and this is its own **primitive** triple (gcd(437,84,445)=1) — NOT "the 3-4-5 triple scaled" as previously stated; no integer k scales (3,4,5) to (437,84,445), and the triple doesn't reduce to a smaller one.

## Checks

| ID | Check |
|----|-------|
| EF_1 | `schema_version == 'QA_EGYPTIAN_FRACTION_CERT.v1'` |
| EF_2 | `d > e > 0` |
| EF_3 | `gcd(d,e) = 1` |
| EF_4 | Expansion sums exactly to e/d |
| EF_5 | Denominators strictly increasing |
| EF_6 | Each kᵢ = ⌈dᵢ/eᵢ⌉ (greedy, validated against recomputed expansion) |
| EF_7 | All intermediate (dᵢ,eᵢ) coprime |
| EF_8 | Terminal: last step has e = 1 |
| EF_W | ≥3 witnesses (witness fixture) |
| EF_F | Fundamental (d=2,e=1) present with denominators=[2] |

## Historical Chain

This is where QA meets **3600 years of prior art**:

1. **~1600 BCE — Rhind Papyrus (Egypt)**: The scribe Ahmes expanded rational slopes as sums of distinct unit fractions using exactly the greedy algorithm. Problem 2 of the Rhind papyrus: 2/3 = 1/2 + 1/6. That is our witness (3,2).

2. **Ben Iverson — Pyth-1**: Explicitly connects the Koenig series (I→H→I chain of prime triangles) to Egyptian fraction decompositions. The Koenig descent step IS the Egyptian fraction greedy step.

3. **H. Lee Price (2008)**: HAT₁ = e/d; each Egyptian fraction step = one Fibonacci-box column operation = one navigation step in the Barning-Hall/Berggren Pythagorean tree.

4. **QA cert stack**: This cert ([134]) is the formal bridge. The expansion algorithm is a pure arithmetic identity requiring only gcd and ceiling — no geometry, no calculus, no continuous approximation.

## Fixtures

| Fixture | Type | Expected |
|---------|------|----------|
| `ef_pass_fundamental.json` | Anchor — (d,e)=(2,1), expansion=[2], length=1 | PASS |
| `ef_pass_witnesses.json` | 6 witnesses covering lengths 1,2,3 + general theorem | PASS |
| `ef_fail_bad_expansion.json` | Falsifier: wrong denominator expansion not summing to e/d (added 2026-07-06) | FAIL |

## Connection to Prior Art Convergence Stack

See `docs/QA_PRIOR_ART_CONVERGENCE.md`.

- Predecessor: **[132]** QA_HAT (establishes HAT₁=e/d)
- Co-cert: **[133]** QA_EISENSTEIN (other use of the same direction elements W,Z,Y,F)
- Successor cert gap: **QA_PYTHAGOREAN_TREE_CERT.v1** — certifies that the full Barning-Hall tree = Koenig tree in HAT/Egyptian-fraction coordinates

## Sources

- Rhind Mathematical Papyrus (~1600 BCE), Problem 2 and table of 2/n decompositions
- Ben Iverson, *Pythagoras and the Quantum World Vol. 1* — Koenig series + Egyptian fractions
- H. Lee Price (2008), "The Pythagorean Tree: A New Species" — HAT=e/d, Fibonacci boxes
- QA vault 2025-03: `Quantum Arithmetic Pythagorean Triples (1).md` — Egyptian fractions in Koenig context

## Verification Note (2026-07-06)

Independently recomputed the greedy Egyptian-fraction expansion (exact
`Fraction` arithmetic) for all 7 witnesses — every denominator sequence
and Koenig descent path matches exactly, including the depth-3 (7,3)
case (expansion [3,11,231], path (7,3)→(21,2)→(231,1)). The validator
(`qa_egyptian_fraction_cert_validate.py`) already genuinely recomputes
the expansion, sum, greediness, coprimality, and termination from
`(d,e)` live using exact `Fraction` arithmetic — no fixture-trusting
gap, no bugs in any certified value.

**Found and fixed a real factual error in the (7,3) witness's
footnote**: it claimed the intermediate direction (21,2) "gives the
3-4-5 triple scaled." Independently computed its actual triple:
(F,C,G)=(437,84,445) — genuinely satisfies F²+C²=G² (198025=198025),
but `gcd(437,84,445)=1`, meaning it's already a **primitive** triple,
not a scaled copy of anything. No integer k satisfies
`(3k,4k,5k)=(437,84,445)`. This footnote wasn't checked by any
validator (pure prose), so it had no certification impact — fixed the
doc to state the correct, independently-verified triple.

**Follow-up (2026-07-06)**: found this family had zero FAIL fixtures
(part of a systemic gap found across 8 sibling families in the
125-139 cluster) and the same latent print-corruption bug first
discovered in cert [132]: a stray `print()` inside the
`result=="FAIL"` short-circuit that corrupts `--self-test`'s stdout
once a FAIL fixture exists to trigger it. Removed the print and added
`fixtures/ef_fail_bad_expansion.json` to close the gap and exercise
the fix.
