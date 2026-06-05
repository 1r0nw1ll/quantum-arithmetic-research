# [331] QA Phasing In: Odd-Wave Half-Integer Half-Cycle Structure

**Family**: `qa_phasing_in_odd_wave_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.8, 11-12  
"PHASING IN", "OTHER CYCLES"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Odd periods {3,5,7,9,11,13,15}: 2×⌊p/2⌋ ≠ p — half-cycle p/2 is non-integer | PASS |
| C2 | Even periods {2,4,6,8,10,12}: 2×(p//2) = p — half-cycle is an exact integer | PASS |
| C3 | Coprime odd pairs: lcm(p,q)=p×q; lcm is also odd — no half-cycle even at full sync | PASS |
| C4 | Par-types: 2-par (≡2 mod 4) and 4-par (≡0) have integer half-cycles; 3-par (≡3) and 5-par (≡1) do not | PASS |
| C5 | For odd p: no integer k < p satisfies 2k=p; half-cycle sync impossible before full period | PASS |

## Phasing In — The Core Claim

Iverson introduces "phasing in" to describe when two waves share a synchronous (half-cycle) alignment point before their first full synchronous point. For a wave of period p, the half-cycle occurs at p/2. The key observation:

| Period type | p/2 | Integer? | Phasing-in possible? |
|------------|-----|----------|---------------------|
| Odd p | non-integer (half-integer) | No | No |
| Even p | integer | Yes | Yes |

For odd p, there is no integer k with 2k = p, so no discrete half-cycle sync point exists within one period.

## Par-Type Connection (C4)

This connects directly to cert [326]'s par-type classification:

| Par type | n mod 4 | Parity | Half-cycle? |
|---------|---------|--------|-------------|
| 4-par | 0 | even | Yes — n//2 is integer |
| 2-par | 2 | even | Yes — n//2 is integer |
| 5-par | 1 | odd | No |
| 3-par | 3 | odd | No |

The par-type structure completely determines whether phasing-in is possible: even par-types (2-par, 4-par) allow half-cycle sync; odd par-types (3-par, 5-par) do not.

## Coprime Odd Wave Pairs (C3)

For two coprime odd periods p, q: since gcd(p,q)=1, lcm(p,q)=p×q. Since both p and q are odd, p×q is also odd. This means the first full synchronization point is odd, and therefore also has no integer half-cycle. Two odd-period waves are "phase-locked" exclusively to their full synchronous points.

| Pair (p,q) | lcm | lcm odd? |
|-----------|-----|---------|
| (3,5) | 15 | Yes |
| (3,7) | 21 | Yes |
| (5,7) | 35 | Yes |
| (7,9) | 63 | Yes |

## Observer Projection Note (Theorem NT)

The labels "phasing in", "half-cycle", "wave synchronization" are observer-layer descriptions. The causal structure is pure integer divisibility: whether 2 | p. The integer/non-integer distinction at p/2 is a discrete property of p's parity. No continuous wave functions enter the QA layer.

**Depends on**: [326] Euclid Four Par Types, [325] To-Be-Prime
