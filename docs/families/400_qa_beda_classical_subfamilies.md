<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Sierpinski (1962) ISBN 978-0-486-43293-4, Dale (2026) Five Families paper Theorem 3 -->
# [400] QA BEDA Classical Subfamilies

**Cert family**: `qa_beda_classical_subfamilies_cert_v1`
**Claim**: The three classical subfamilies (Fermat, Pythagoras, Plato) of primitive Pythagorean triples have elegant BEDA characterizations — computational proof of Theorem 3 in the Five Families paper.

## Statement

Three classical subfamilies defined since antiquity admit single-equation BEDA characterizations:

| Subfamily | Classical Definition | BEDA Characterization | Proof |
|---|---|---|---|
| Fermat | \|C − F\| = 1 | \|b² − 2e²\| = 1 (Pell boundary) | C − F = 2de − ab = 2e² − b² |
| Pythagoras | (d − e)² = 1 | b = 1 | d − e = b, so (d−e)² = b² = 1 |
| Plato | \|G − F\| = 2 | e = 1 (b odd for primitive) | G − F = 2e², so \|G−F\| = 2 ↔ e = 1 |

## Key Algebraic Identities

**Fermat**: C − F = 2de − ab = 2(b+e)e − b(b+2e) = 2e² − b²

So |C − F| = 1 is exactly the Pell equation |b² − 2e²| = 1.
Pell solutions in {1..30}²: (1,1),(3,2),(7,5),(17,12) — 4 pairs.

**Pythagoras**: d − e = (b+e) − e = b, so (d−e)² = b². Equals 1 iff b = 1.

**Plato**: G − F = (e²+d²) − (d²−e²) = 2e². Equals 2 iff e = 1.

## Paper Examples (Table, §5)

| Subfamily | (b, e) | (C, F, G) | Property |
|---|---|---|---|
| Fermat | (1,1) | (4,3,5) | \|4−3\|=1 |
| Fermat | (3,2) | (20,21,29) | \|20−21\|=1 |
| Fermat | (7,5) | (120,119,169) | \|120−119\|=1 |
| Pythagoras | (1,1) | (4,3,5) | (2−1)²=1 |
| Pythagoras | (1,2) | (12,5,13) | (3−2)²=1 |
| Pythagoras | (1,3) | (24,7,25) | (4−3)²=1 |
| Plato | (1,1) | (4,3,5) | 5−3=2 |
| Plato | (3,1) | (8,15,17) | 17−15=2 |
| Plato | (7,1) | (16,63,65) | 65−63=2 |

## Checks

- **C1**: C − F = 2e² − b² identity, 5 examples + 7 Pell solutions — PASS
- **C2**: d − e = b, so (d−e)²=1 iff b=1, 7 pairs — PASS
- **C3**: G − F = 2e², so |G−F|=2 iff e=1, 8 pairs — PASS
- **C4**: All 9 paper table examples match — PASS
- **C5**: Exhaustive b,e ∈ {1..30}: 4 Fermat pairs, 30 Pythagoras (b=1), 15 primitive Plato (e=1, b odd), 0 mismatches — PASS

## Chain

- Extends [398] (Five Families Complete Partition — Layer 0)
- Layer 2 of the three-layer taxonomy in the Five Families paper
- Fermat boundary = Pell solutions |b²−2e²|=1; connects to classical Pell equation theory
