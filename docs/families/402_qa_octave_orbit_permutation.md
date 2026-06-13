<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Wall (1960) doi:10.1080/00029890.1960.11989541 -->
# [402] QA Octave Orbit Permutation

**Cert family**: `qa_octave_orbit_permutation_cert_v1`
**Claim**: The digital-root octave map σ: (b,e)→(dr(2e),b) is a permutation of {1..9}² with cycle type **(1, 4², 12⁶)** and order 12; it maps each QA orbit class to itself.

## Statement

σ is the digital-root shadow of the Octave Transformation from cert [401]:
- In integers: (b,e,d,a) → (2e, b, a, 2d)
- Restricted to digital roots: (b,e) → (dr(2e), b)

σ is the linear map M = [[0,2],[1,0]] acting on (ℤ/9ℤ)² (1-indexed).

## Cycle Type and Orbit Preservation

| Orbit Class | #Pairs | σ-cycles | Cycle Length |
|---|---|---|---|
| Singularity | 1 | 1 fixed point | 1 |
| Satellite | 8 | 2 four-cycles | 4 |
| Cosmos | 72 | 6 twelve-cycles | 12 |

**Total**: 1×1 + 2×4 + 6×12 = 1 + 8 + 72 = 81 ✓

**Order(σ) = 12**: σ^k ≠ identity for k ∈ {1,2,3,4,6}, and σ^12 = identity on all 81 pairs.

## Satellite 4-Cycles (C5)

The two Satellite 4-cycles are:
- **Cycle A**: (3,3) → (6,3) → (6,6) → (3,6) → (3,3)
- **Cycle B**: (6,9) → (9,6) → (3,9) → (9,3) → (6,9)

These correspond to the two geometric halves of the 8-element Satellite orbit.

## Orbit Preservation — Algebraic Reason

Membership in {Satellite, Singularity} is determined by divisibility by 3:
- Satellite: 3|b and 3|e, but not 9|b and 9|e simultaneously
- Singularity: 9|b and 9|e (i.e., b≡0≡e mod 9, which in {1..9} is b=e=9)

Under σ: dr(2e) ≡ 2e (mod 9). Since gcd(2,9)=1 (2 is a unit mod 9), 3|dr(2e) iff 3|e. So:
- σ maps Satellite pairs (3|b, 3|e) to Satellite pairs (3|dr(2e), 3|b) ✓
- σ maps Singularity (b=e=9) to (dr(18),9) = (9,9) ✓

## Mirroring of Cycle Type and Orbit Sizes

The cycle type (1, 4², 12⁶) **mirrors** the orbit partition (1, 8, 72):
- 1 = 1 × 1 (Singularity fixed point = Singularity orbit size)
- 8 = 2 × 4 (two 4-cycles = Satellite orbit size)
- 72 = 6 × 12 (six 12-cycles = Cosmos orbit size)

This is not a coincidence: the Pisano period of Cosmos is 24 = 2×12, and the Cosmos is covered by 6 σ-cycles each of length 12 = period(Cosmos)/2.

## Checks

- **C1**: σ is a bijection: 81 domain → 81 distinct images — PASS
- **C2**: Cosmos→Cosmos (72 pairs), Satellite→Satellite (8 pairs), Singularity fixed — PASS
- **C3**: Cycle type = {1:1, 4:2, 12:6} — PASS
- **C4**: σ^12 = identity; σ^k ≠ identity for k ∈ {1,2,3,4,6} — order = 12 — PASS
- **C5**: Satellite 4-cycles = {(3,3),(6,3),(6,6),(3,6)} and {(6,9),(9,6),(3,9),(9,3)} — PASS

## Chain

- Extends [401] (Octave Transformation: integer-level transform)
- Extends [398] (Five Families Complete Partition: Table 1 = the 9×9 grid σ permutes)
- Connected to [281] (Pisano periods: Cosmos period = 24 = 2×|12-cycle|)
