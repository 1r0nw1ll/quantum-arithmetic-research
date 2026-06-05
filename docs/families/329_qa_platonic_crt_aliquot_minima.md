# [329] QA Book 4 Synthesis: Platonic Edges, CRT Triple Uniqueness, Aliquot Minima

**Family**: `qa_platonic_crt_aliquot_minima_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.53-65 "ALIQUOT PARTS", "PAR TYPES", "PLATONIC SOLIDS", "PARAMETERS"  
**Depends on**: [327] Harmonic Cycle, [328] Aliquot Parts

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Platonic solid edge count: E = F × epf / 2 verified for all 5 solids | PASS |
| C2 | Euler's formula V-E+F=2 holds; face types: triangle (3 solids), square (1), pentagon (1) | PASS |
| C3 | Triple (n mod 3, n mod 4, n mod 5) uniquely identifies n ∈ {1..60}; lcm(3,4,5)=60 | PASS |
| C4 | Three smallest aliquot parts: 30=lcm(2,3,5), 42=lcm(2,3,7), 210=lcm(2,3,5,7); all divisible by 6 | PASS |
| C5 | Musical intervals: major third=5:4, major fifth=3:2, major seventh=15:8=5/4×3/2; all 7-smooth | PASS |

## Platonic Solid Edge Count (C1, C2)

Iverson (p.61): "The number of edges on any Platonic solid is equal to the number of edges per face multiplied by half the number of faces."

| Solid | V | E | F | epf | F×epf/2 | V-E+F |
|-------|---|---|---|-----|---------|-------|
| Tetrahedron | 4 | 6 | 4 | 3 | 6 | 2 |
| Cube | 8 | 12 | 6 | 4 | 12 | 2 |
| Octahedron | 6 | 12 | 8 | 3 | 12 | 2 |
| Dodecahedron | 20 | 30 | 12 | 5 | 30 | 2 |
| Icosahedron | 12 | 30 | 20 | 3 | 30 | 2 |

The face polygon types (triangle, square, pentagon) are exactly the three shapes that generate synchronous points in the 60-unit harmonic cycle (cert [327]):
- Triangle ← pair (4,5) → 3 sync points
- Square ← pair (3,5) → 4 sync points  
- Pentagon ← pair (3,4) → 5 sync points

This is not a coincidence: both structures derive from the coprimeness of {3,4,5} and lcm(3,4,5)=60.

## CRT Triple Uniqueness (C3)

Iverson introduces a **triple par-classification** (p.56): every integer has three simultaneous class labels:
- **Tri-class** (mod 3): 2-tri (≡2), 3-tri (≡0), 4-tri (≡1)
- **Par-class** (mod 4): 2-par, 3-par, 4-par, 5-par (cert [326])
- **Pent-class** (mod 5): 2-pent through 7-pent

The key claim: "No two integers below 60 will have the same combined classification."

This follows immediately from the Chinese Remainder Theorem: since gcd(3,4)=gcd(4,5)=gcd(3,5)=1, the map n ↦ (n mod 3, n mod 4, n mod 5) is a bijection from {0,...,59} to Z/3 × Z/4 × Z/5. The period is lcm(3,4,5) = 60 — the same harmonic cycle period.

**Iverson's examples** (p.56):
- 37: 37 mod 3=1 → 4-tri; 37 mod 4=1 → 5-par; 37 mod 5=2 → 7-pent
- 32: 32 mod 3=2 → 2-tri; 32 mod 4=0 → 4-par; 32 mod 5=2 → 7-pent

Both are 7-pent but differ in tri and par class → distinct triples (1,1,2) vs (2,0,2).

**Connection to harmonic cycle**: The uniqueness period = 60 = lcm(3,4,5) is also the period of Iverson's fundamental harmonic cycle. The CRT structure ensures that within one harmonic cycle, every integer has a unique "address" in the {tri,par,pent} triple space.

## Three Smallest Aliquot Parts (C4)

Iverson (p.53): "There are three different, smallest aliquot parts. These are 2×3×5=30; 2×3×7=42; and 2×3×5×7=210 in any scale of units."

| Aliquot | Factorization | Divisible by 6? | LCM identity |
|---------|--------------|-----------------|--------------|
| 30 | 2×3×5 | Yes (30/6=5) | lcm(2,3,5) |
| 42 | 2×3×7 | Yes (42/6=7) | lcm(2,3,7) |
| 210 | 2×3×5×7 | Yes (210/6=35) | lcm(2,3,5,7) |

The relationship: 210/30 = 7 and 210/42 = 5, so each smaller aliquot part divides 210. Equivalently, 30 and 42 are aliquot parts of 210 (with unique primes 7 and 5 respectively).

## Musical Intervals (C5)

Iverson (p.62) states the three fundamental musical ratios:

| Interval | Ratio | Factored |
|----------|-------|----------|
| Major third | 5:4 | 5 : 2² |
| Major fifth | 3:2 | 3 : 2 |
| Major seventh | 15:8 | (3×5) : 2³ |

All ratios are 7-smooth (prime factors ≤ 7). The major seventh is the product of the major third and major fifth: 5/4 × 3/2 = 15/8. This multiplicative structure means the seventh is the aliquot combination of the third and fifth intervals.

## Observer Projection Note (Theorem NT)

The geometric labels (Platonic solid names, interval names) and physical metaphors (wave packets, musical harmony) are observer-layer descriptions of underlying integer divisibility and modular structure. The causal layer: gcd/lcm arithmetic, CRT, prime factorizations. The geometric/acoustic interpretations are projections.
