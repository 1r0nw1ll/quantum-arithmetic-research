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

The (7,3) witness is the only depth-3 expansion in the table — it also shows a non-trivial intermediate direction (21,2) which is itself a valid QA direction giving the 3-4-5 triple scaled: (441−4, 84, 445).

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
