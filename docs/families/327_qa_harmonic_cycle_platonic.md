# [327] QA Harmonic Cycle: Platonic Polygon Inscription

**Family**: `qa_harmonic_cycle_platonic_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.10-12 "MULTIPLE WAVES", "THE HARMONIC CYCLE", "OTHER CYCLES", "MOST BASIC MULTIPLE CYCLE"  
**Depends on**: [326] Euclid Four Par Types

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 60-unit harmonic cycle = lcm(3,4,5) = 3×4×5; waves (3,4,5) are pairwise coprime | PASS |
| C2 | Pair (3,4): lcm=12; synchronous points in 60-cycle = {12,24,36,48,60}; 5 equally spaced → regular pentagon | PASS |
| C3 | Pair (3,5): lcm=15; synchronous points = {15,30,45,60}; 4 equally spaced → square | PASS |
| C4 | Pair (4,5): lcm=20; synchronous points = {20,40,60}; 3 equally spaced → equilateral triangle | PASS |
| C5 | Four fundamental harmonic cycles: 30=lcm(2,3,5), 42=lcm(2,3,7), 60=lcm(3,4,5), 105=lcm(3,5,7) | PASS |

## The Harmonic Cycle

Iverson's key construction (p.10): take a circle of 60 units in circumference, and plot three sinusoidal waves of period 3, 4, and 5 units simultaneously around this closed baseline. All three close at the same point: 3×4×5 = 60.

Within one 60-unit cycle, each pair of waves reaches a synchronous point (both simultaneously at their full value) at every multiple of their LCM:

| Pair | LCM | Synchronous points | Count | Shape |
|------|-----|--------------------|-------|-------|
| (3,4) | 12 | {12, 24, 36, 48, 60} | 5 | Pentagon |
| (3,5) | 15 | {15, 30, 45, 60} | 4 | Square |
| (4,5) | 20 | {20, 40, 60} | 3 | Equilateral triangle |

The synchronous points of each pair are equally spaced around the 60-unit circle, so connecting them traces the three fundamental regular polygons.

Iverson (p.10): "Connecting these in-phase, (Synchronous), points for each pair will inscribe a pentagon, a square, and an equilateral triangle respectively. These happen to be the three plane shapes which compose the five different Platonic solids."

## Platonic Solid Connection

The five Platonic solids have exactly three face types:
- **Triangle** (equilateral): tetrahedron, octahedron, icosahedron
- **Square**: cube (hexahedron)
- **Pentagon**: dodecahedron

The harmonic cycle (3,4,5) = 60 simultaneously generates all three face polygons as its pairwise synchronous-point sets. This is not a coincidence but a structural consequence of the unique property of 3, 4, 5 as pairwise-coprime generators of 60 = 3×4×5.

## Fundamental Harmonic Cycles (C5)

Beyond the (3,4,5) cycle, Iverson identifies other fundamental cycles formed from three prime-wave periods:

| Cycle | Composition | LCM check |
|-------|-------------|-----------|
| 30 | (2, 3, 5) | lcm(2,3,5) = 30 = 2×3×5 |
| 42 | (2, 3, 7) | lcm(2,3,7) = 42 = 2×3×7 |
| 60 | (3, 4, 5) | lcm(3,4,5) = 60 = 3×4×5 |
| 105 | (3, 5, 7) | lcm(3,5,7) = 105 = 3×5×7 |

Each cycle period is the LCM (= product when pairwise coprime) of its three constituent wave periods. Iverson notes that any physically realizable harmonic cycle must contain factors of 2, 3, 5, and/or 7 — the first four primes are the building blocks.

## Observer Projection Note (Theorem NT)

The polygon shapes (pentagon, square, triangle) and cycle periods (30, 42, 60, 105) are **observer-layer labels** on the underlying integer LCM structure. The causal layer is pure integer arithmetic: gcd, lcm, divisibility. The geometric interpretation (Platonic solids, wave diagrams) is the observer projection — valid but not causally prior to the arithmetic.
