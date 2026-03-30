# Family [104] — QA Feuerbach Parent Scale

**Cert root:** `qa_feuerbach_parent_scale_v1/`
**Validator:** `qa_feuerbach_parent_scale_v1/validator.py --self-test`
**Schema:** `QA_FEUERBACH_PARENT_SCALE_CERT.v1.schema.json`

## What it certifies

The **Feuerbach parent-scale law** for primitive Pythagorean triples: a purely geometric
construction (nine-point center + incenter) recovers the Barning-Berggren parent of any
primitive triple with exact scale factor 4.

### Exact construction

Place a primitive triple (C, F, G) as a right triangle at the origin, legs along axes.

```
r         = (C + F - G) / 2          # inradius (always an integer)
Incenter  = (r, r)
Nine-point center = (C/4, F/4)
Parent leg₁ = |C + 2F - 2G|
Parent leg₂ = |2C + F - 2G|
```

All arithmetic is **exact and integer** (no floating point).

### Claims

**Interior law** (all primitive triples except root):
Both computed legs are positive integers forming a primitive Pythagorean triple — the
Barning-Berggren parent. The implicit scale factor relative to the nine-point displacement
distance is always **4**.

**Root exception** (3, 4, 5):
```
|3 + 8 - 10| = 1,   |6 + 4 - 10| = 0   →  degenerate (one leg = 0)
```
(3, 4, 5) is the **unique** primitive triple with this degeneracy.

### QA interpretation

| Observable | QA meaning |
|---|---|
| Scale = 4 (interior) | Inverse transport law — generator-driven reachability |
| Degenerate at (3,4,5) | Boundary obstruction — inverse move undefined |
| QA closure scale = 2G = 10 | Root renormalization constant of the seed state |

This connects to the **Intertwining Theorem** (Pythagorean families paper, Theorem 6):
the τ map intertwines BEDA child operators with Barning 3×3 matrices. The Feuerbach
construction is the **geometric shadow** of the inverse Barning generator action.

## Gates

| Gate | Check |
|---|---|
| 1 | Schema anchor: required fields, schema_id const |
| 2 | Root exception: triple=[3,4,5], raw_legs=[0,1], qa_scale=10 |
| 3 | Sample recompute: parent_legs + sex match formula for every declared sample |
| 4 | Batch recompute: primitive_count, confirmed_count, uniqueness, sex_invariant |

## Secondary invariant (sex classification)

Every primitive triple has exactly one even leg, always divisible by 4:
- **male**: C ≡ 0 (mod 4)
- **female**: F ≡ 0 (mod 4)

Gate 4 verifies this for every triple in the batch.

## Fixtures

| File | Expected outcome |
|---|---|
| `fixtures/pass/l50_full.json` | PASS — 7 triples (G ≤ 50), 6 confirmed + root |
| `fixtures/fail/bad_scale_value.json` | FAIL Gate 4 — WRONG_SCALE_VALUE (5 ≠ 4) |
| `fixtures/fail/bad_root_qa_scale.json` | FAIL Gate 2 — WRONG_QA_SCALE (4 ≠ 10) |

## Connections to other families

- Pythagorean families paper (arXiv-ready 2026-03-10): five families = orbits of F=[[0,1],[1,1]] on (Z/9Z)²
- Barning-Berggren tree: semigroup words in {A, B, C} applied to primitive triples
- QA modular orbit structure: interior = cosmos reachability; root = boundary fixed point

## Source

Observed in Mathologer video on Feuerbach's theorem and the nine-point circle (~22:00 mark).
The QA interpretation and algebraic formula were developed from the geometric observation.
The formula is elementary and provable by direct substitution into the parametric form
(C, F, G) = (m²-n², 2mn, m²+n²).
