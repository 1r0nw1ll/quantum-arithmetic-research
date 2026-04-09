# Family [204] QA_SKINNER_HEBREW_METROLOGY_CERT.v1

## One-line summary

J. Ralston Skinner's 1875 "Source of Measures" contains a metrological system built on powers of 9 (kernel 6561=9⁴), deriving the number 24 via digital roots (Garden-Eden), producing 72² as the solar day characteristic, and treating pi exclusively as an observer projection output — all T2-compliant. Seven claims verified, three honestly qualified.

## Background

**J. Ralston Skinner** (1830s–1890s) published *Key to the Hebrew-Egyptian Mystery in the Source of Measures* in 1875. The book argues that Hebrew letter-values encode a metrological system connecting the Great Pyramid, the Temple of Solomon, and British imperial measures through a unified numerical framework built on Parker's Quadrature.

**Parker's Quadrature**: A circle-squaring system using the ratio 6561:20612, where 6561 = 9⁴ = 3⁸ = 81². The ratio approximates π to ~1.23×10⁻⁶ error. Critically, Skinner derives this ratio from integer arithmetic, never using continuous π as an input — pi appears only as an output comparison.

## Mathematical content (7 verified claims)

### PARK: Parker kernel 6561 = 9⁴

The generative base of the entire system is 6561 = 9⁴ = 3⁸. Factorization is purely {2,3}-smooth. dr(6561) = 9.

### EDEN: Garden-Eden = 24 via digital roots

The Hebrew word for Garden-Eden (gn-odn) has standard gematria values [3, 50, 70, 4, 50] = 177. Skinner reads "characteristic values" (digital roots): [3, 5, 7, 4, 5] = **24**. The characteristic value operation IS the digital root: dr(50)=5, dr(70)=7. The sum 24 = QA applied modulus = π(9).

### SOLAR: 5184 = 72² (Cosmos orbit)

The solar day characteristic 5184 = 72 × 72 = 144 × 36. dr(5184) = 9, mod24(5184) = 0. The number 72 is exactly the QA Cosmos orbit pair count (72 pairs in the 24-cycle). 144 = Adam = 2 × 72. Factorization: 2⁶ × 3⁴ (pure {2,3}-smooth).

### ADAM: All roads lead to 9

Adam = 144 → dr = 9 (1+4+4). Woman = 135 → dr = 9 (1+3+5). Serpent = Tet = 9. "Adam, and the woman, and the serpent, are one in the center of this garden, as the number 9." Additionally: 144 − 135 = **9**. The difference IS the digital root.

### BRIDGE: Factor 6 connects mod-9 to mod-24

6 = 2 × 3 bridges the theoretical (mod-9) and applied (mod-24) moduli:
- 6 × 4 = **24** (applied modulus)
- 6 × 60 = **360** (circle degrees)
- 6⁴ × 4 = **5184** (solar day)
- 6² = **36** (area unit)
- 6³ = **216** (Plato's number)

### MET: Metius dr-closure

355/113 ≈ π (error 2.67×10⁻⁷). dr(355) = 4, dr(113) = 5. Sum = **9** (QA modulus). The Metius approximation's numerator and denominator are dr-complementary to 9.

### T2: Pi as observer projection

All Skinner derivations use integer arithmetic. Pi appears only as output comparison:
- 5153 = 5184 − 31 (integer subtraction, not π×r²)
- 20612 = 5153 × 4 (integer multiplication, not 2πr)
- 6561:20612 COMPARED to π, not DERIVED from π

This is Theorem NT practiced in 1875.

## Qualified claims (honest failures)

### Q_EL: El=31 is NOT a full QA generator

31 mod 9 = 4. Order of 4 in (Z/9Z)* is 3, generating only {1, 4, 7}. True generators of (Z/9Z)* are 2 and 5 (order 6 = φ(9)). Skinner's claim of El as "universal generator" fails in the QA sense, though it works as a fixed subtractive constant.

### Q_PALINDROME: dr-preservation under reversal is trivial

Digit reversal (144↔441, 135↔531) preserves digit sum in any positional base. This is a base-10 arithmetic identity, not a structural property.

### Q_PARKER_PI: Parker pi is mediocre

20612/6561 ≈ 3.14159388 (error ~1.23×10⁻⁶). Metius 355/113 ≈ 3.14159292 (error ~2.67×10⁻⁷, about 5× better). The value of Parker's ratio is not its accuracy but that 6561 = 9⁴.

## Checks

| ID | Description |
|----|-------------|
| SKM_1 | schema_version == 'QA_SKINNER_HEBREW_METROLOGY_CERT.v1' |
| SKM_PARK | 6561 = 9⁴; {2,3}-smooth |
| SKM_EDEN | characteristic sum = 24; digital roots verified |
| SKM_SOLAR | 5184 = 72²; dr=9; mod24=0 |
| SKM_ADAM | dr(144)=dr(135)=9; 144−135=9 |
| SKM_BRIDGE | factor 6 derivations verified |
| SKM_MET | dr(113)+dr(355)=9 |
| SKM_NUM | numerical checks pass |
| SKM_W | >= 5 witnesses |
| SKM_F | falsifier: El=31 generates only {1,4,7} |

## Source grounding

- **Skinner** (1875): *Key to the Hebrew-Egyptian Mystery in the Source of Measures*. Archive.org: https://archive.org/details/keytohebrewegypt00skin
- **Skinner** (1885): *Hebrew Metrology* (8pp pamphlet)
- **Parker**: The Quadrature of the Circle (reproduced in Skinner 1875)

## Connection to other families

- **[202] Hebrew Mod-9 Identity**: Aiq Bekar = digital root = Skinner's "characteristic value"
- **[203] Sefer Yetzirah Combinatorics**: 4!=24 as structural constant
- **[192] Dual Extremality**: π(9)=24 grounds the mod-24 connection
- **[130] Origin of 24**: independent derivation of 24
- **[153] Keely Triune**: Skinner's triune (3-fold) structure

## Fixture files

- `fixtures/skm_pass_verified.json` — 7 verified claims + 3 qualified claims with full witnesses
- `fixtures/skm_pass_numerical.json` — powers of 9, digital root tests, subtraction chain, factor 6 chain, El generator test
- `fixtures/skm_fail_el_generator.json` — El=31 as full generator (fails: order 3, not 6)
