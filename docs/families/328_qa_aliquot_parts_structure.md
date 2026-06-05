# [328] QA Aliquot Parts Structure

**Family**: `qa_aliquot_parts_structure_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.13-14 "ALIQUOT PARTS", "EXAMPLE", "QUESTIONS"  
**Depends on**: [327] Harmonic Cycle Platonic Inscription

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Integers coprime-to-30 in (0,30) = {1,7,11,13,17,19,23,29}; 8 values; complement pairs sum to 30; 5²+5=3³+3=2⁵-2=30 | PASS |
| C2 | Aliquot candidates from {32,3,5,7,11}: 3 valid {3360,5280,7392}, 2 invalid {1155,12320}; matches Iverson p.14 | PASS |
| C3 | Wave gear sync: shared aliquot A=18330, unique primes 53 and 7 → sync at 6800430; X completes 7 cycles, Y completes 53 | PASS |
| C4 | Three special forms of 30: 5²+5=30, 3³+3=30, 2⁵-2=30; factored as 5×6, 3×10, 2×15 | PASS |
| C5 | Aliquot count = n for n prime factors; valid requires {power-of-2, 3, and 5-or-7}; from {2,3,5,7}: 2 valid, 2 invalid | PASS |

## Aliquot Parts Definition

An **aliquot part** of a quantum wave is formed by taking the full set of prime factors in the quantum number and leaving out exactly one. Iverson (p.13):

> "An aliquot part of such a wave will be the product of all prime numbers, excepting one of them, in its quantum number."

**Validity constraint** (p.14): every valid aliquot part must contain:
- A power of 2 (the "2" family — provides even-cycle structure)
- The prime 3 (required by Pythagoras/Fibonacci base-triangle constraint)
- At least one of 5 or 7 (the first primes beyond the {2,3} base)

This mirrors the harmonic minimum: 30=2×3×5 or 42=2×3×7 are the smallest valid harmonic cycles.

## Worked Example from Iverson p.14 (C2)

Quantum wave with prime factors {32, 3, 5, 7, 11} (where 32=2⁵ supersedes bare 2):

| Excluded | Product | Has 2? | Has 3? | Has 5 or 7? | Valid? |
|----------|---------|--------|--------|-------------|--------|
| 32 | 3×5×7×11 = 1155 | No | Yes | Yes | **INVALID** |
| 3 | 32×5×7×11 = 12320 | Yes | No | Yes | **INVALID** |
| 5 | 32×3×7×11 = **7392** | Yes | Yes | Yes (7) | VALID |
| 7 | 32×3×5×11 = **5280** | Yes | Yes | Yes (5) | VALID |
| 11 | 32×3×5×7 = **3360** | Yes | Yes | Yes (5,7) | VALID |

## Wave Gear Synchronization (C3)

Two waves X and Y share aliquot part A, with unique primes p_x and p_y respectively:
- Wave X period = A × p_x
- Wave Y period = A × p_y
- Synchronization point = lcm(A×p_x, A×p_y) = A×p_x×p_y (when gcd(p_x,p_y)=1)
- At sync: wave X completes p_y cycles; wave Y completes p_x cycles

**Iverson's example** (p.13): A = 2×3×5×13×47 = 18330, p_x = 53, p_y = 7.

"After wave X goes through its wave 7 times it will be equal to wave Y going through its cycle 53 times." ✓ — the aliquot part A acts as the gear-teeth: each "tooth" is one aliquot cycle, and the unique primes determine how many teeth each wheel has.

## The 30-Unit Prime Symmetry (C1)

Iverson directs the reader to compute 30=2×3×5 and list primes to 30 excluding {2,3,5}:

Forward: 1, 7, 11, 13, 17, 19, 23, 29  
Reversed: 29, 23, 19, 17, 13, 11, 7, 1

Every pair sums to 30. This is equivalent to the statement that if p is coprime to 30 and 0<p<30, then 30-p is also coprime to 30 — a consequence of the multiplicativity of the Euler totient.

Iverson notes three special forms (p.14):
- 5² + 5 = 30 → 5×(5+1) = 5×6
- 3³ + 3 = 30 → 3×(9+1) = 3×10
- 2⁵ − 2 = 30 → 2×(16−1) = 2×15

Each factored form is one of the basic harmonic cycles (30, and its divisors 15, 10, 6), multiplied by 2, 3, or 5.

## Observer Projection Note (Theorem NT)

The "bonding" language (two waves mesh like gears) and "carrier wave" language are observer-layer metaphors for the integer LCM structure. The causal layer: gcd(p_x,p_y)=1 → lcm(A×p_x, A×p_y) = A×p_x×p_y. The bond strength, harmonic enrichment, and physical resonance are projections from this integer fact.
