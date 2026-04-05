# Family [189] QA_DALE_CIRCLE_CERT.v1

## One-line summary

Dale Pond's integer circle construction: P=2W (diameter), Q=P (circumference), R=W² (area) — pi disappears in QA circular units.

## Mathematical content

### Dale Pond's circle elements (svpwiki.com, 1998)

"When Ben said there was no way to define a circle with Quantum Arithmetic, I developed three different ways to do this." — Dale Pond

| Element | Definition | Meaning |
|---------|-----------|---------|
| P | 2W | Circle diameter (QA units) |
| Q | P = 2W | Circle circumference (QA units) |
| R | W² | Circle area (QA units) |

### Key insight: QA circular units (pi=1 convention)

Dale defines QA circular units where pi=1: P=Q means diameter=circumference by **convention**, not by structural theorem. This is analogous to natural units in physics (c=1, h-bar=1). The algebraic identities P=2W, Q=P, R=W² are exact by definition. Dale's contribution is identifying W as the natural radius-like quantity for QA circles. Extends element count to 25 (with S from [190]).

### Verification (Unity Block)

(b,e,d,a) = (1,1,2,3): W=8, P=16, Q=16, R=64.

Verified for 7 Pythagorean directions with consistent results.

## Checks

| ID | Description |
|----|-------------|
| DC_1 | schema_version == 'QA_DALE_CIRCLE_CERT.v1' |
| DC_P | P = 2*W for all witnesses |
| DC_Q | Q = P (diameter = circumference in QA units) |
| DC_R | R = W*W for all witnesses |
| DC_W | W correctly derived from (b,e) |
| DC_SRC | source attribution to Dale Pond |
| DC_WITNESS | >= 3 witnesses |
| DC_F | fail_ledger well-formed |

## Source grounding

- **svpwiki.com/Quantum+Arithmetic+Elements**: Dale Pond, "Unpublished Notes", December 1998
- **Ben Iverson**: original 21 elements (b–Z); circles declared impossible
- **Dale Pond**: three circle construction methods; P, Q, R elements

## Connection to other families

- **[152] Equilateral Triangle Cert**: W = equilateral side; P, Q, R derived from W
- **[133] Eisenstein Norm Cert**: F²-FW+W²=Z²; W is the Eisenstein equilateral side
- **[190] Equilateral Height Cert**: S = equilateral height; S, W, P form equilateral-circle bridge

## Fixture files

- `fixtures/dc_pass_circle.json` — 7 directions with P, Q, R verified
- `fixtures/dc_fail_bad_p.json` — FAIL fixture for testing
