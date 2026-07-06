# Family [161] QA_ECEF_RATIONAL_CERT.v1

## One-line summary

Geodetic-to-ECEF coordinate conversion expressed entirely using spreads (s=sin²) and crosses (c=1-s) — no transcendental functions needed.

## Mathematical content

### Classical vs Rational ECEF

| | Classical | Rational (QA) |
|---|---|---|
| N | a / sqrt(1 - e²·sin²φ) | N² = a² / (1 - e²·s_φ) |
| X | (N+h)·cos(φ)·cos(λ) | X² = (N+h)²·c_φ·c_λ |
| Y | (N+h)·cos(φ)·sin(λ) | Y² = (N+h)²·c_φ·s_λ |
| Z | (N(1-e²)+h)·sin(φ) | Z² = (N(1-e²)+h)²·s_φ |

where s_φ = sin²(φ) = spread of latitude, c_φ = 1 - s_φ = cross of latitude.

### Key identity

X² + Y² = (N+h)²·c_φ·(c_λ + s_λ) = (N+h)²·c_φ

This follows from c + s = 1 (the fundamental spread-cross identity in rational trigonometry).

### Witness points

| City | Lat | Lon | s_φ | s_λ |
|------|-----|-----|-----|-----|
| London | 51.48°N | 0.00°W | 0.6121 | ~0 |
| Tokyo | 35.68°N | 139.65°E | 0.3401 | 0.4192 |
| Sydney | 33.87°S | 151.21°E | 0.3106 | 0.2320 |
| Sao Paulo | 23.55°S | 46.63°W | 0.1596 | 0.5285 |
| North Pole | 90°N | 0° | 1.0 | 0.0 |
| Equator/PM | 0° | 0° | 0.0 | 0.0 |

All six verify X², Y², Z² match classical ECEF to < 0.1 m² (< 0.3mm positional accuracy).

## Tier classification

**Tier 1 — Exact reformulation.** Squaring the classical ECEF formulas replaces sin/cos with spreads/crosses. The resulting formulas are algebraically identical. Float precision limits the match to ~0.1 m²; with exact rational arithmetic the match would be perfect.

## Practical advantage

For integer/fixed-point systems (FPGA, embedded LiDAR, drone processors):
- No trig lookup tables needed
- No Taylor series truncation
- No accumulated float error from chained sin/cos calls
- All intermediate quadrances exact at mm² precision

## Sources

- WGS84: NIMA Technical Report 8350.2, 3rd ed. (2000)
- N.J. Wildberger, *Divine Proportions* (2005), Ch. 1-2

## Validator

`qa_alphageometry_ptolemy/qa_ecef_rational_cert_v1/qa_ecef_rational_cert_validate.py --self-test`

## Verification Note (2026-07-06)

Independently recomputed s_φ/s_λ for all six witness cities from their
real lat/lon coordinates — all match the doc's table (two entries round
differently in the 4th decimal: Tokyo 0.3402 vs 0.3401, São Paulo
0.5284 vs 0.5285 — negligible rounding, not a bug). Independently
computed both classical ECEF (X,Y,Z via sin/cos) and rational ECEF
(X²,Y²,Z² via spreads/crosses) from scratch using the real WGS84
constants (a=6378137.0, e²=2f−f², confirmed in the [156] audit) for all
six cities: all differences are float-precision noise on the order of
10⁻³ to 10⁻¹⁹ m², far under the claimed <0.1 m² tolerance — exactly what
an algebraic identity should produce. Confirmed the fundamental
X²+Y²=(N+h)²c_φ identity follows trivially from c_λ+s_λ=1.

Validator confirmed genuinely computing both classical and rational
forms live using the real WGS84 constants and comparing them at
runtime, not fixture-trusting. `--self-test` passes on both fixtures.
No bugs found.
