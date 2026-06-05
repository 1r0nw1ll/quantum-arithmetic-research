# [326] QA Euclid Four Par Types

**Family**: `qa_euclid_four_par_types_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, p.8 "EUCLID'S 4 NUMBER TYPES"; pp.5-7 "TIME SYNCHRONIZATION", "EVEN NUMBERS", "GRAPHICS"  
**Depends on**: [325] To-Be-Prime / Euclid Prop 28

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Four par types partition {1,…,24} into 4 equal groups of 6: 4-par={4n}, 5-par={4n+1}, 2-par={4n+2}, 3-par={4n+3} | PASS |
| C2 | Synchronous period: coprime p,q → lcm(p,q)=p×q; non-coprime → lcm<p×q; Ben's (4,6)→gcd=2, lcm=12<24 | PASS |
| C3 | Quarter-harmonic at ⌊p×q/4⌋ and 3×⌊p×q/4⌋ for coprime odd p,q; (3,7)→(5,15), (5,9)→(11,33) match Ben's p.7 examples | PASS |
| C4 | Odd par-type multiplication (mod 4): 3×3=5-par, 5×5=5-par, 3×5=3-par; verified exhaustively for all odd integers {1..23}² | PASS |
| C5 | 2-par has v₂=1, 4-par has v₂≥2 → gcd(2-par, 4-par)≥2 always; zero coprime pairs in all {1..24}² 2-par/4-par combinations | PASS |

## The Four Par Types

Iverson identifies four classes of integers (p.8), corresponding to residues mod 4:

| Par type | Residue mod 4 | Iverson's label | Description |
|----------|--------------|-----------------|-------------|
| 4-par | 0 | even-even | Divisible by 4; v₂ ≥ 2 |
| 5-par | 1 | odd-odd | 4n+1 form |
| 2-par | 2 | even-odd | Exactly one factor of 2; v₂ = 1 |
| 3-par | 3 | odd-even | 4n-1 = 4n+3 form |

Iverson (p.8): "Contemporary mathematics acknowledged the 4n+1 and 4n-1 integers, but completely missed the full gravity of them. It also completely omitted the differentiation between 2-par and 4-par integers."

The 2-par vs 4-par distinction is crucial: a 2-par number (like 6) and a 4-par number (like 4) have lcm=12, not 24, because they share a factor of 2. Ben's "EVEN NUMBERS" section (p.7) demonstrates this with the (4,6) table: they synchronize at 12 (half their product), not 24.

## Time Synchronization (C2)

Ben's synchronous progression for coprime p,q (pp.5-6): both waves start at 1 and advance simultaneously. After p×q steps they return to (0,0) simultaneously for the first time. For non-coprime p,q, the period is lcm < p×q — the shared factor "collapses" part of the cycle.

The table for (4,6): they synchronize at 12 = lcm(4,6) instead of 24 = 4×6, because gcd(4,6)=2. Iverson: "They act as though the two numbers were '3' and '4'." — the coprime core (3,4) determines the period (lcm=12), multiplied by the common factor (2) doesn't contribute to extending the cycle.

## Par-Type Multiplication Table (C4)

The odd par types form a closed multiplicative system (mod 4):

| × | 3-par (≡3) | 5-par (≡1) |
|---|-----------|-----------|
| **3-par (≡3)** | 5-par (9≡1) | 3-par (3≡3) |
| **5-par (≡1)** | 3-par (3≡3) | 5-par (1≡1) |

This mirrors the multiplicative group (Z/4Z)* = {1,3}, which is isomorphic to Z/2Z. The 5-par class acts as the identity, and the 3-par class acts as the non-identity element.

**Physical consequence** (Iverson p.8): Two waves of the SAME odd par type (both 3-par or both 5-par) reinforce each other at quarter-points of their product. Two waves of OPPOSITE odd par types (one 3-par, one 5-par) form a "null wave packet" — cancellation. This wave physics is the observer-layer label on the integer multiplication table.

## Quarter-Harmonic Points (C3)

Ben's graphic examples (p.7):
- Waves (3,7): both 3-par. Product=21. Harmonic surge at 5¼ and 15¾ → integer floor: points 5 and 15. ✓
- Waves (5,9): both 5-par (gcd(5,9)=1 since 9=3² but gcd(5,9)=1 ✓). Product=45. Quarter at 11¼ → 11 and 33. ✓

The quarter-point rule means the maximum composite amplitude occurs at 1/4 and 3/4 of the full period — not at the midpoint. This is analogous to the quarter-wavelength resonance condition in physical wave theory.
