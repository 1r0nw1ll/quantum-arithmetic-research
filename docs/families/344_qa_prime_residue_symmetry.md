# [344] QA Prime Residue Symmetry mod 30/60 and Cycle Coincidence

**Family**: `qa_prime_residue_symmetry_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapters XII-XIII pp.26-50

> "1+29=30, 7+23=30, 11+19=30, and 13+17=30. This set leaves out the prime numbers 2,  
>  3, and 5, in that 2×3×5=30, and the remaining prime numbers are set symmetrically  
>  within the number system."
> "The cycles of prime numbers, when combined will make a cycle which is equal to  
>  their product."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | φ(30)=8: exactly 8 residues mod 30 coprime to 30: {1,7,11,13,17,19,23,29} | PASS |
| C2 | 4 symmetric pairs about 15 each summing to 30: 1+29, 7+23, 11+19, 13+17 | PASS |
| C3 | lcm(p,q)=p·q for coprime primes; 6 named pairs + 36 prime pairs verified | PASS |
| C4 | lcm(2,3,5)=30; lcm(3,5,7)=105; overall cycle = product for coprime primes | PASS |
| C5 | φ(60)=16: 16 coprime residues mod 60 form 8 pairs each summing to 60 | PASS |

## Structure of Coprime Residues

### Mod 30: φ(30)=8 (C1, C2)

The 30 integers 2×3×5 are the first three primes. All primes > 5 fall in the 8 coprime residue classes mod 30. These residues pair symmetrically about the midpoint 15:

$$\{1, 7, 11, 13, 17, 19, 23, 29\}$$

| Pair | Sum | Prime in pair? |
|------|-----|----------------|
| 1 + 29 | 30 | 29 is prime |
| 7 + 23 | 30 | both prime |
| 11 + 19 | 30 | both prime |
| 13 + 17 | 30 | both prime |

**Proof**: r is coprime to 30 iff r is coprime to 2, 3, and 5. Then (30-r) is coprime to 2 (since 30 is even, 30-r is even iff r is even; r coprime to 2 → r odd → 30-r odd), coprime to 3 (30≡0 mod 3, so 30-r≡-r mod 3, coprime iff r coprime), and coprime to 5 (same). So r coprime to 30 ↔ (30-r) coprime to 30.

### Mod 60: φ(60)=16 (C5)

For 60=3×4×5=2²×3×5: φ(60)=φ(4)·φ(3)·φ(5)=2×2×4=16. The 16 coprime residues form 8 pairs:

1+59=60, 7+53=60, 11+49=60, 13+47=60, 17+43=60, 19+41=60, 23+37=60, 29+31=60

(Iverson's specific list from Ch.XII p.30.)

### Cycle Coincidence = LCM (C3, C4)

For coprime prime periods (wave lengths) p and q: lcm(p,q)=p×q since gcd=1.

Iverson's specific coincidences:
- lcm(2,3)=6: 2 and 3 coincide every 6 units
- lcm(2,5)=10: 2 and 5 coincide every 10 units  
- lcm(3,5)=15: 3 and 5 coincide every 15 units
- lcm(3,7)=21: 3 and 7 coincide every 21 units
- lcm(5,7)=35: 5 and 7 coincide every 35 units

And for triples of mutually coprime primes:
- lcm(2,3,5) = 2×3×5 = **30** (first full coincidence cycle)
- lcm(3,5,7) = 3×5×7 = **105** (first full coincidence cycle for odd primes 3,5,7)

The half-cycle symmetry point: at lcm/2, all cycles are simultaneously in their "symmetric" position.

## Observer Projection Note (Theorem NT)

"Wave," "cycle," "coincidence point," "symmetry about half-cycle" are observer-layer labels on integer modular arithmetic. The causal structure: φ(30)=8 residues (Euler totient), r+(30-r)=30 (arithmetic), lcm(p,q)=p×q for coprime primes (gcd identity). No continuous geometry or time enters.

**Depends on**: [342] Pythagorean Divisibility Laws; [343] Fibonacci Bead Quadruple
