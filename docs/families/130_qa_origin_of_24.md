# [130] QA Origin of 24 Cert

**Schema**: `QA_ORIGIN_OF_24_CERT.v1`
**Directory**: `qa_alphageometry_ptolemy/qa_origin_of_24_cert_v1/`
**Validator**: `qa_origin_of_24_cert_validate.py`

## What It Certifies

The **dual derivation of mod-24** as the natural modulus of QA arithmetic.

Two independent routes both produce exactly 24 at the fundamental Pythagorean direction (d,e)=(2,1), which encodes the 3-4-5 triangle:

- **Route 1 (Pyth-1)**: H²-G² = 49-25 = 24, where H=C+F=7 and G=d²+e²=5
- **Route 2 (Crystal)**: G²-I² = 25-1 = 24, where G=5 and I=C-F=1

The key identity: **H²-G² = G²-I² = 2CF** for any direction (d,e), where C=2de (green quadrance) and F=d²-e² (red quadrance). Since C²+F²=G² (the Pythagorean condition), (C+F)²-G² = 2CF.

For all primitive Pythagorean directions (gcd(d,e)=1, d+e odd): **24 | 2CF always**. The minimum value 24 is achieved at the fundamental direction (d,e)=(2,1).

## Why 24 Is Not Arbitrary

Ben Iverson identified the mod-24 structure of QA as *Plato's independent 2nd dimension* — not an engineering choice but a mathematical inevitability forced by Pythagorean geometry. The same 24 appears as:

1. The QA cosmos orbit period π(9)=24 (Pisano period of Fibonacci mod 9)
2. The minimum gap H²-G²=24 at the fundamental Pythagorean direction
3. The crystal identity 7²-5²=24 in QA-4 Iota geometry

These three coincidences are one: they all arise from the Fibonacci/golden-ratio structure of Z[φ] and its reduction mod 9.

## Proof Sketch

For any direction (d,e) with C=2de, F=d²-e², G=d²+e², H=C+F, I=C-F:

```
H² - G² = (C+F)² - G² = C² + 2CF + F² - G²
        = 2CF              [since C² + F² = G²]

G² - I² = G² - (C-F)² = G² - C² + 2CF - F²
        = 2CF              [same identity]
```

Divisibility by 24: 2CF = 4de(d-e)(d+e). For primitive triples:
- **÷8**: d+e is odd → exactly one of d,e is even → de is even → 4·de = 8·(integer)
- **÷3**: if neither d nor e is ≡0(mod 3), then d≡1,e≡2 or vice versa → d+e≡0(mod 3)

## Checks

| ID | Check |
|----|-------|
| O24_1 | `schema_version == 'QA_ORIGIN_OF_24_CERT.v1'` |
| O24_2 | `C = 2*d*e` |
| O24_3 | `F = d²-e²` |
| O24_4 | `G = d²+e²` |
| O24_5 | `H = C+F` |
| O24_6 | `I = C-F` |
| O24_7 | `H²-G² = declared value` |
| O24_8 | `G²-I² = declared value` |
| O24_9 | `H²-G² = G²-I²` (dual consistency) |
| O24_G | `general_theorem.statement` present |
| O24_W | ≥3 witnesses with correct H²-G² values |
| O24_F | Fundamental witness (d=2,e=1) has H²-G²=24 |
| O24_D | All witness H²-G² divisible by 24 |

## Fixtures

| Fixture | Type | Expected |
|---------|------|----------|
| `origin24_pass_3_4_5.json` | Anchor — (d,e)=(2,1), 3-4-5 triangle | PASS |
| `origin24_pass_general.json` | General theorem — 6 witnesses d≤5 | PASS |

## Running

```bash
# Validate all fixtures
python qa_alphageometry_ptolemy/qa_origin_of_24_cert_v1/qa_origin_of_24_cert_validate.py

# Self-test (for meta-validator)
python qa_alphageometry_ptolemy/qa_origin_of_24_cert_v1/qa_origin_of_24_cert_validate.py --self-test

# Single fixture
python qa_alphageometry_ptolemy/qa_origin_of_24_cert_v1/qa_origin_of_24_cert_validate.py \
    qa_alphageometry_ptolemy/qa_origin_of_24_cert_v1/fixtures/origin24_pass_3_4_5.json
```

## Connection to QA Math

- **Family [125]** (Chromogeometry): C=2de is the green quadrance. The identity 2CF is the product of the two non-hypotenuse chromogeometric quadrances.
- **Family [128]** (Spread Period): The orbit period π(9)=24 is the same 24 derived here from geometry.
- **Family [127]** (UHG Null): The triple (F,C,G) forms a null point in UHG; H²-G²=2CF is an identity of null quadrangles.
- **Source**: Ben Iverson Pyth-1 (primary); QA-4 Iota crystal geometry.
