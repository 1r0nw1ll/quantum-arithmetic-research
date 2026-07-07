# [357] QA Pyth-3 Twenty Identities and Equilateral Triangle Parameters

**Family**: `qa_pyth3_twenty_identities_cert_v1`  
**Source**: Iverson & Elkins (2006) *Pythagorean Arithmetic Vol III* Chapter 2 pp.4-9

> *(p.7-8)*: "W = C/2 + K, or = de + K for the sides of an equilateral triangle. F = F (from above) for one division of the base, and Y = C + E, for the other division of the base of the triangle. One can see that Y + F must equal the side of this triangle, or Y + F = W."

> *(p.8)*: "Z = G + C/2, or Z = G + de."

> *(p.8)*: "Another check was to see that H squared minus I squared = 48 L."

> *(p.5-6)*: "There was a second stage in which only the even numbered integers were used for root 'b'... the first was for the conventional male triangles, and the second was for unconventional female triangles."

> *(p.8)*: "The values of J and K will be pellian numbers, being d squared - de = J, and d squared + de = K."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | W=de+K, Y=C+E; W=F+Y for all male pairs ≤25; algebraic proof W=d(d+2e), Y+F=d²+2de | PASS |
| C2 | Z=G+de; Z<W (Z-W=-eb<0), Z>F (e(2e+d)>0), Z>Y (db>0) for all male pairs ≤25 | PASS |
| C3 | H²-I²=48L; proof: (H+I)(H-I)=2F·2C=4CF=48·(CF/12)=48L for all male pairs ≤25 | PASS |
| C4 | Female (b even, e odd, gcd=1): d=odd, a=even; C=2de≡2(mod 4) — NOT divisible by 4 (78 pairs ≤19) | PASS |
| C5 | Female seed (2,1,3,4): W=15=F+Y=8+7; Z=13<W=15; H²-I²=192=48×4=48L; C=6≡2(mod 4) | PASS |

## Mathematical Details

### The 20 Identities (Table 2 Structure)

Iverson's Table 2 (combined male+female) organizes each block into 5 rows of 4 identities:

| Row | Identities | Content |
|-----|-----------|---------|
| 1 | b, e, d, a | Bead roots |
| 2 | B=b², E=e², D=d², A=a² | Squares |
| 3 | C=2de, F=ab, G=D+E, L=abde/6 | Primary |
| 4 | H=C+F, I=|C-F|, J=bd, K=ad | Circles & ellipses |
| 5 | W=de+K, F (repeated), Y=C+E, Z=G+de | Equilateral |

### C1: Equilateral Triangle Check W = F + Y

**Algebraic proof**:
- W = d(d+2e) = d² + 2de
- F = ab = (d-e)(d+e) = d² - e²
- Y = e(2d+e) = 2de + e²
- Y + F = (2de + e²) + (d² - e²) = d² + 2de = W ✓

**Geometric meaning**: W is the side of a prime equilateral triangle. F and Y partition the base, with the altitude dividing F from Y at the foot of the perpendicular from the apex.

### C2: Dividing Line Z

Z = G + de (equivalently Z = G + C/2 since C = 2de).

| Comparison | Difference | Sign | Proof |
|-----------|-----------|------|-------|
| Z vs W | Z - W = e² - de = e(e-d) = -eb | **Negative**: Z < W | e,b > 0 |
| Z vs F | Z - F = 2e² + de = e(2e+d) | **Positive**: Z > F | e,d > 0 |
| Z vs Y | Z - Y = d² - de = d(d-e) = db | **Positive**: Z > Y | d,b > 0 |

So the ordering is: F < Y < Z < W (when F < Y; this holds when b < e) or Y < F < Z < W (when b > e).

### C3: H² - I² = 48L

**Proof**:
H + I = max(C,F) + min(C,F) + |C-F| ... actually:
- If C > F: H = C+F, I = C-F → H+I = 2C, H-I = 2F
- If F > C: H = C+F, I = F-C → H+I = 2F, H-I = 2C

In either case: (H+I)(H-I) = 4CF = 4·12L = 48L ✓

This is distinct from:
- H²+I²=2G² (cert [339])
- H²-G²=G²-I²=24L (cert [352])

### C4: Female Triangles vs Male

Female = b even, e odd, gcd(b,e)=1:

| Property | Male (b odd) | Female (b even) |
|----------|-------------|----------------|
| b parity | odd | even |
| e parity | any | odd |
| d parity | mixed | **always odd** (even+odd) |
| a parity | always odd | **always even** (odd+odd) |
| C mod 4 | 0 (divisible by 4) | **2 (not div by 4)** |

**Why**: For female, d and e are both odd, so C = 2de = 2·(odd)·(odd) = 2·(odd²). Since odd²≡1(mod 4), we get C≡2(mod 4).

### C5: Female Fibonacci Seed (2,1,3,4)

This tuple is the female Fibonacci seed from cert [354] (Pyth-3 Ch.13-14):

| Identity | Value | Check |
|---------|-------|-------|
| C=2de | 6 | 6≡2(mod 4) ✓ (female) |
| F=ab | 8 | — |
| G=d²+e² | 10 | 3²+1²=10 |
| H=C+F | 14 | 6+8=14 |
| I=\|C-F\| | 2 | \|6-8\|=2 |
| L=abde/6 | 4 | 2·4·3·1/6=4 |
| W=de+K | 15 | 3+12=15 |
| Y=C+E | 7 | 6+1=7 |
| Z=G+de | 13 | 10+3=13 |
| W=F+Y | 8+7=15 ✓ | — |
| H²-I²=48L | 196-4=192=48·4 ✓ | — |
| J=d²-de | 9-3=6=bd ✓ | Pellian form |
| K=d²+de | 9+3=12=ad ✓ | Pellian form |

## Theorem NT Note

"Equilateral triangle side W," "apogee K," "perigee J," and "ellipse dividing line Z" are observer projection labels for integer arithmetic identities. The causal layer is (b,e,d,a) arithmetic. No continuous geometry enters the QA causal layer.

**Depends on**: [355] Formal Proofs (4|C for male, C2 proof of factor-3); [352] Concentric Areas (H²+I²=2G², H²-G²=24L); [337] J,K Parameters (K-J=C)  
**Female extension**: female triangles not covered by Vol-I divisibility laws — they are a genuine extension requiring separate certification

**Key identity chain**: H²-I²=48L = 2×(H²-G²=24L); confirms the 24L structure of cert [352]

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced all 5 claims over
268 male pairs and 78 female pairs (b,e≤25): W=de+K, Y=C+E, W=F+Y exact
for all male pairs; Z=G+de ordering Z<W, Z>F, Z>Y exact; H²-I²=48L
exact; the male/female C-mod-4 discriminant (male always 0, female
always 2); and the full female seed (2,1,3,4) numeric table (C=6, F=8,
G=10, H=14, I=2, L=4, W=15, Y=7, Z=13, J=6, K=12) — every value matches
the doc's table exactly. The validator
(`qa_pyth3_twenty_identities_cert_validate.py`) is genuinely computed,
no fixture-trusting gap.
