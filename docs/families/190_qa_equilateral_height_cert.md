# Family [190] QA_EQUILATERAL_HEIGHT_CERT.v1

## One-line summary

Element S = d²e = d·X = D·e: Dale Pond's 25th QA element. Dale labeled it "Height of equilateral triangle" but geometrically S = d·(C/2) is a rectangle area (semi-major × half-base), NOT a height.

## Mathematical content

### Element S (svpwiki.com item #25)

| Definition | Formula | Equivalence |
|-----------|---------|-------------|
| Primary | S = d²·e | d-squared times e |
| Via half-base | S = d·X | d times X (where X=de=C/2) |
| Via semi-major | S = D·e | D times e (where D=d²) |

**Geometric correction**: Dale's label "Height of equilateral triangle" is incorrect. For Unity Block (1,1,2,3): W=8, so equilateral height = 8√3/2 ≈ 6.93, but S=4. S is actually the area of the rectangle d × (C/2) = semi-major × half-base.

### Verification (7 directions)

| (b,e) | d | S | d·X | D·e |
|-------|---|---|-----|-----|
| (1,1) | 2 | 4 | 2×2=4 | 4×1=4 |
| (2,1) | 3 | 9 | 3×3=9 | 9×1=9 |
| (3,2) | 5 | 50 | 5×10=50 | 25×2=50 |
| (5,2) | 7 | 98 | 7×14=98 | 49×2=98 |
| (5,3) | 8 | 192 | 8×24=192 | 64×3=192 |

All three definitions give identical results (algebraic identity).

## Checks

| ID | Description |
|----|-------------|
| EH_1 | schema_version == 'QA_EQUILATERAL_HEIGHT_CERT.v1' |
| EH_S | S = d*d*e for all witnesses |
| EH_DX | S = d*X verified |
| EH_DE | S = D*e verified |
| EH_W | >= 3 witnesses |
| EH_F | fail_ledger well-formed |

## Source grounding

- **svpwiki.com/Quantum+Arithmetic+Elements**: item 25, "S = d*X = Height of equilateral triangle"
- **Dale Pond**: extended Ben Iverson's 21 elements with S (and P, Q, R)

## Connection to other families

- **[152] Equilateral Triangle Cert**: W = equilateral side; S = height of same triangle
- **[189] Dale Circle Cert**: P, Q, R derived from W; S + circle elements complete the equilateral-circle bridge
- **[148] Sixteen Identities Cert**: S extends the named element set beyond the original 16

## Fixture files

- `fixtures/eh_pass_height.json` — 7 directions with S verified three ways
- `fixtures/eh_fail_wrong_s.json` — FAIL fixture for testing
