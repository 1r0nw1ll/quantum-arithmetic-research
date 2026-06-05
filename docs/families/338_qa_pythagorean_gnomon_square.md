# [338] QA Pythagorean Gnomon and Square Identities

**Family**: `qa_pythagorean_gnomon_square_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* pp.37-39, 43-46

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | F=d²-e²=(d+e)(d-e)=ab since b=d-e and a=d+e; 11 pairs | PASS |
| C2 | C=2de=2be+2e²=2e(b+e): three-form equivalence | PASS |
| C3 | C²=4E²+4EF where E=e²; 10 pairs | PASS |
| C4 | A=a² and B=b² are 5-par (≡1 mod 4); squares of odd integers | PASS |
| C5 | D=d² and E=e² always have opposite par-types (4-par vs 5-par); 12 pairs | PASS |

## Core Structural Result

### F as Difference of Squares (C1)

Since $b = d - e$ (cert [337]) and $a = d + e$:

$$F = ab = (d+e)(d-e) = d^2 - e^2$$

This gives F a second representation beyond F=ab: it is always the **difference of two consecutive Pythagorean squares**.

### Three Forms of C (C2)

$$C = 2de = 2be + 2e^2 = 2e(b+e) = 2e \cdot d$$

All three forms are identical algebraically. The expanded form C=2be+2e² appears in Iverson's gnomon analysis.

### Gnomon Identity C²=4E²+4EF (C3)

$$C^2 = 4E^2 + 4EF$$

where $E = e^2$. Proof: $4E^2 + 4EF = 4e^4 + 4e^2 F = 4e^2(e^2 + F)$. Since $F = d^2 - e^2$:

$$4e^2(e^2 + d^2 - e^2) = 4e^2 \cdot d^2 = (2de)^2 = C^2 \checkmark$$

### Par-Types of Squares (C4, C5)

| Identity | Formula | Parity | Par-type |
|----------|---------|--------|----------|
| A = a² | square of odd | odd × odd ≡ 1 (mod 4) | **5-par** |
| B = b² | square of odd | odd × odd ≡ 1 (mod 4) | **5-par** |
| D = d² | square of d | one of {4-par, 5-par} | opposite to E |
| E = e² | square of e | one of {4-par, 5-par} | opposite to D |

Since d and e have opposite parities (cert [333]: gcd(b,e)=1 with b odd → e and d=b+e have opposite parities):
- e even → E=e²≡0 (mod 4) → **4-par**; d odd → D≡1 (mod 4) → **5-par**
- e odd → E≡1 (mod 4) → **5-par**; d even → D≡0 (mod 4) → **4-par**

### Worked Example: (b,e)=(3,2) → triangle (20-21-29)

| Parameter | Value | Par-type |
|-----------|-------|----------|
| a=7, A=49 | 49≡1 (mod 4) | 5-par ✓ |
| b=3, B=9 | 9≡1 (mod 4) | 5-par ✓ |
| d=5, D=25 | 25≡1 (mod 4) | 5-par |
| e=2, E=4 | 4≡0 (mod 4) | 4-par |
| F=21 | 5²-2²=25-4=21 ✓ | — |
| C²=400 | 4(16)+4(4)(21)=64+336=400 ✓ | — |

## Observer Projection Note (Theorem NT)

"Gnomon", "rectangle", "ellipse" are observer-layer labels. The causal structure: integer identities C²=4E²+4EF, F=d²-e², and par-type arithmetic (mod 4) on squared bead numbers. No continuous geometry enters.

**Depends on**: [336] Pythagorean 16 Identities; [337] Ellipse J,K; [333] Female QN Parity; [151] QA Par Numbers
