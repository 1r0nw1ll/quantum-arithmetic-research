# [132] QA Half-Angle Tangent Cert

**Schema**: `QA_HAT_CERT.v1`
**Directory**: `qa_alphageometry_ptolemy/qa_hat_cert_v1/`
**Validator**: `qa_hat_cert_validate.py`

## What It Certifies

The bridge between H. Lee Price's **Half-Angle Tangents (HATs)** and QA direction structure.

For any QA direction (d,e) generating triple (F,C,G) = (d²−e², 2de, d²+e²):

| Identity | Formula | Meaning |
|----------|---------|---------|
| **HAT₁** | `e/d = C/(G+F)` | Primary: `e/d`; Price's `tan(α)` |
| **HAT₂** | `(d-e)/(d+e) = F/(G+C)` | Secondary: Price's `tan(β)`, α+β=45° |
| **Spread** | `s = E/G = HAT₁²/(1+HAT₁²)` | Wildberger rational trig: `sin²(α)` |
| **HAT₁²** | `E/D = e²/d²` | Uppercase QA notation |

**Proportionality rule**: HAT₁ must be read as `C/(G+F)` from the triple — not as an independently reduced fraction. The numerator `C` (green quadrance) and denominator `G+F = 2d²` carry QA geometric meaning. Price encodes both HATs as columns of the **Fibonacci box** `[[e, d-e],[d, d+e]]`.

## Why It Matters

H. Lee Price's 2008 paper "The Pythagorean Tree: A New Species" independently arrived at the same tree structure Ben Iverson describes in Pyth-1. Price's construction:
- HATs → Fibonacci boxes → Pythagorean tree
is exactly QA's:
- direction ratios e/d → generation matrices M1/M2/M3 → Koenig tree

Ben had the tree. Price rediscovered it from a different angle. Mathologer then showed the Fibonacci connection visually. All three are one thing.

The HAT spread relationship `s = E/G` connects directly to Wildberger's rational trig (family [44]) where `s = sin²(α)` is the spread. So HATs sit exactly at the intersection of Price, Wildberger, and QA.

## Checks

| ID | Check |
|----|-------|
| HAT_1 | `schema_version == 'QA_HAT_CERT.v1'` |
| HAT_2 | `hat1 == e/d` (reduced fraction) |
| HAT_3 | `hat1 == C/(G+F)` (Price formula) |
| HAT_4 | `hat2 == (d-e)/(d+e)` |
| HAT_5 | `hat2 == F/(G+C)` (Price formula) |
| HAT_6 | `hat1² == E/D` |
| HAT_7 | `spread s == E/G` |
| HAT_8 | `spread s == HAT1²/(1+HAT1²)` (equivalent via D+E=G) |
| HAT_W | ≥3 witnesses (witness fixture) |
| HAT_F | Fundamental witness (d=2,e=1) has HAT1=1/2 |

## Fixtures

| Fixture | Type | Expected |
|---------|------|----------|
| `hat_pass_fundamental.json` | Anchor — (d,e)=(2,1), 3-4-5, Fibonacci box det=1 | PASS |
| `hat_pass_witnesses.json` | 5 witnesses d=2..5, general theorem | PASS |
| `hat_fail_bad_hat1.json` | Falsifier: wrong HAT1 fraction + broken Fibonacci-box determinant (added 2026-07-06) | FAIL |

## Verification Note (2026-07-06)

Independently recomputed F, C, G, HAT1, HAT2, spread, and the Fibonacci
box determinant by hand for every witness in both PASS fixtures (e.g.
(5,2): F=21,C=20,G=29, HAT1=2/5, HAT2=3/7, spread=4/29, box=[[2,3],[5,7]]
det=2*7-3*5=-1) — all correct. Confirmed `check_direction`/
`check_witness` already genuinely recompute from (d,e) rather than
trusting fixture data (verified via a planted-error regression test).
No bugs in the certified math.

**Found and fixed a real, confirmed doc/docstring bug**: both this
family doc's title and the validator's own docstring/argparse
description said "family 131" / "[131]", but the actual
`qa_meta_validator.py` registration is family **132** (matches the file
name `132_qa_hat.md` and the doc slug `"132_qa_hat"`). Fixed both.

**Found and closed one real gap**: this family had zero FAIL fixtures,
the same pattern found in sibling certs [142]/[143]. Added
`fixtures/hat_fail_bad_hat1.json` with two independent, genuinely-
detectable planted violations (wrong HAT1 fraction, wrong Fibonacci-box
determinant) — confirmed both caught by `check_witness` directly.
Updated `_self_test()` accordingly.

**Adding the FAIL fixture exposed a second, real, previously-latent
bug**: `validate()` had a stray `print("  SKIP detailed checks — cert
declares FAIL")` inside the `result == "FAIL"` short-circuit branch.
This is harmless when `validate()` is called interactively via `main()`,
but `_self_test()` calls `validate()` in-process and expects `main()`'s
`--self-test` output to be pure JSON on stdout — the meta-validator's
subprocess wrapper (`_validate_hat_cert_family` in `qa_meta_validator.py`)
captures that stdout and does `json.loads()` on it. With no FAIL fixture
in the family, `validate()` was never called with `result=="FAIL"`
during self-test, so this print statement never fired and the bug
stayed dormant. Adding `hat_fail_bad_hat1.json` triggered it immediately
— the full meta-validator sweep failed with `qa_hat_cert self-test
returned non-JSON: Expecting value: line 1 column 1 (char 0)`. Removed
the print statement; re-ran the full 464-family sweep to confirm it now
passes cleanly. This is a genuine example of a coverage fix (adding a
missing FAIL fixture) surfacing a real, independent bug that pure code
review would not have caught.

## Connection to Prior Art Convergence Stack

See `docs/QA_PRIOR_ART_CONVERGENCE.md` for the full stack.
This cert is the formal link between node 6 (H. Lee Price) and QA.

- Predecessor: **[44]** QA_RATIONAL_TRIG (Wildberger 5 laws — spread `s=E/G`)
- Predecessor: **[125]** QA_CHROMOGEOMETRY (C=green quadrance, G+F=2d²)
- Successor cert gap: **QA_PYTHAGOREAN_TREE_CERT.v1** (Price tree = Koenig tree)
- Successor cert gap: **QA_EGYPTIAN_FRACTION_CERT.v1** (e/d as unit fractions)

## Sources

- H. Lee Price (2008), "The Pythagorean Tree: A New Species"
- QA vault 2025-07: `Lee price half-angle tangent.md` (HAT→QA formalization with Python)
- QA vault 2025-03: `Quantum Arithmetic Pythagorean Triples (1).md`
- Ben Iverson Pyth-1: Koenig series + Egyptian fractions (original source)
