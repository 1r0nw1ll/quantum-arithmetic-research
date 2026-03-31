# QA_FEUERBACH_PARENT_SCALE_CERT.v1

Machine-checkable cert family for the Feuerbach parent-scale law on primitive Pythagorean triples.

## Primary claim

For every primitive Pythagorean triple (C, F, G) with C < F, place the right triangle at the origin with legs along the axes. Define:

```
r        = (C + F - G) / 2 = be          # inradius (integer); r=be in QA roots
Incenter = (r, r)
Nine-point center = (C/4, F/4)
Parent legs = ( |C + 2F - 2G|,  |2C + F - 2G| )
           = ( |4r - C|,        |4r - F|       )  # fully QA-native via r=be
```

In QA variables (C=2de, F=ab, X=ed=C/2, L=XF/6=abde/6=Area/6):
```
6L = CF/2 = FX = abde  # full triangle area
r  = be = 6L/s          # inradius, s=(C+F+G)/2
```

**Interior law** (all non-root triples): both legs are positive, form a primitive Pythagorean triple, and the implicit scale factor relative to the nine-point distance is exactly **4**.

**Root exception** (3, 4, 5):
- Computed legs = (0, 1) — degenerate (one leg = 0).
- QA interpretation: boundary obstruction; closure scale = **2G = 10**.
- (3, 4, 5) is the **unique** primitive triple with this degeneracy.

## QA interpretation

| Observable | QA meaning |
|---|---|
| Scale = 4 (interior) | Inverse transport law — generator-driven reachability |
| Degenerate at (3,4,5) | Boundary obstruction — inverse move undefined |
| Closure scale = 2G = 10 | Root renormalization constant of the seed state |

## Secondary invariant (sex classification)

Every primitive triple has exactly one even leg, always divisible by 4:
- **male**: C ≡ 0 (mod 4)  (even leg is C)
- **female**: F ≡ 0 (mod 4)  (even leg is F)

## Files

| File | Purpose |
|---|---|
| `schema.json` | JSON Schema for `QA_FEUERBACH_PARENT_SCALE_CERT.v1` |
| `validator.py` | 4-gate validator with `--self-test` |
| `mapping_protocol_ref.json` | Gate 0 intake protocol reference |
| `fixtures/pass/l50_full.json` | Valid cert for G ≤ 50 (7 triples) |
| `fixtures/fail/bad_scale_value.json` | Invalid: scale_value=5 (WRONG_SCALE_VALUE) |
| `fixtures/fail/bad_root_qa_scale.json` | Invalid: qa_scale=4 at root (WRONG_QA_SCALE) |

## Gates

| Gate | Check |
|---|---|
| 1 | Schema anchor: required fields, schema_id const |
| 2 | Root exception: triple=[3,4,5], raw_legs=[0,1], qa_scale=10 |
| 3 | Sample recompute: parent_legs + sex match formula |
| 4 | Batch recompute: primitive_count, confirmed_count, uniqueness, sex_invariant |

## Run

```bash
python qa_feuerbach_parent_scale_v1/validator.py --self-test
python qa_feuerbach_parent_scale_v1/validator.py qa_feuerbach_parent_scale_v1/fixtures/pass/l50_full.json
```

## Mathematical source

Feuerbach's theorem + Barning-Berggren tree. Observed in Mathologer video on the nine-point circle (https://www.youtube.com/watch?v=94mV7Fmbx88). The formula is exact and integer-only; no floating-point involved.

## Connection to other families

- Pythagorean families paper (arXiv-ready, 2026-03-10): five families = orbits of F=[[0,1],[1,1]] on (Z/9Z)²
- Intertwining Theorem (Theorem 6): τ(M_X·u) = R_X·τ(u) connects BEDA operators to Barning matrices
- Berggren tree structure = semigroup of words in {A, B, C}
- This cert demonstrates the **geometric shadow** of that semigroup action
