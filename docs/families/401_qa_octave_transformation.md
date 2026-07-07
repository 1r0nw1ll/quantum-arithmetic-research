<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Sierpinski (1962) ISBN 978-0-486-43293-4, Dale (2026) Five Families paper Theorem 4 -->
# [401] QA Octave Transformation

**Cert family**: `qa_octave_transformation_cert_v1`
**Claim**: For a primitive male BEDA tuple (b,e,d,a), the octave transform (b,e)→(2e,b) produces the female triple (C',F',G') = (2F, 2C, 2G) with gcd exactly 2 — computational proof of Theorem 4 in the Five Families paper.

## Statement

**Transform**: (b,e,d,a) → (b',e',d',a') = (2e, b, a, 2d)

**Identities verified**:
- d' = b'+e' = 2e+b = a ✓ (A2 preserved)
- a' = b'+2e' = 2e+2b = 2d ✓ (A2 preserved)
- C' = 2d'e' = 2·a·b = 2F = 2·ab
- F' = a'b' = 2d·2e = 4de = 2·2de = 2C
- G' = (e')²+(d')² = b²+a² = 2(e²+d²) = 2G
- gcd(C',F',G') = gcd(2F,2C,2G) = 2·gcd(F,C,G) = 2·1 = 2

**Key identity for G'=2G**: b²+a² = b²+(b+2e)² = 2b²+4be+4e² = 2(b²+2be+2e²) = 2(e²+(b+e)²) = 2(e²+d²) = 2G

## Male-Female Pairs (Table 4, §6)

| Family | Male (b,e) | Male (C,F,G) | Female (b',e') | Female (C',F',G') |
|---|---|---|---|---|
| Fibonacci | (1,1) | (4,3,5) | (2,1) | (6,8,10) |
| Phibonacci | (3,1) | (8,15,17) | (2,3) | (30,16,34) |
| Lucas | (1,2) | (12,5,13) | (4,1) | (10,24,26) |
| Lucas | (1,3) | (24,7,25) | (6,1) | (14,48,50) |

Note: in each case (C',F',G') = (2F, 2C, 2G) — legs interchange and all scale by 2.

## Exhaustive Results (C5)

Over b,e ∈ {1..20}² (400 pairs):
- **400/400**: G' = 2G identity b²+a² = 2(e²+d²) — zero failures
- **169 primitive males found**: all 169 produce female triple with gcd = 2

## QA Orbit Structure

The octave transformation maps Cosmos/Satellite generators (primitives) to their first octave echoes:
- Male tuples (primitive, gcd≠2) = fundamental orbit positions
- Female tuples (gcd=2) = octave echoes of males under scale-2 rescaling

## Checks

- **C1**: BEDA A2 preserved under transform: d'=b'+e' and a'=b'+2e' for 6 pairs — PASS
- **C2**: Female = (2F, 2C, 2G) for 7 pairs — PASS
- **C3**: gcd(female) = 2 for 10 primitive males — PASS
- **C4**: All 4 paper Table 4 rows match — PASS
- **C5**: Exhaustive {1..20}²: 400/400 G'=2G, 169/169 primitives gcd=2 — PASS

## Chain

- Extends [398] (Five Families Complete Partition — Layer 0)
- Extends [400] (BEDA Classical Subfamilies — Layer 2)
- Provides Layer 1 gender structure to the three-layer taxonomy
- (1,1) male → (2,1) female: Fibonacci seed → Lucas seed; octave twin relationship

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the full octave
transform (d'=a, a'=2d, C'=2F, F'=2C, G'=2G) in a fresh script for all
400 pairs b,e∈{1..20} — 400/400 pass, zero failures — and confirmed
169 primitive pairs produce a female triple with gcd exactly 2. Genuine
falsifiable algebra, no fixture-trusting gap.
