# [361] QA Pyth-1 Primeness, Parity and Shape

**Family**: `qa_pyth1_primeness_parity_shape_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter IV pp.42-49

> *(p.44)*: "C must satisfy the formula: C = 2be + 2e². And C² must satisfy the formula C² = 4E² + 4EF."

> *(p.41)*: "The value of G must always be a 5-par number... It is also often a 5-pent number and can have no divisor smaller than 5."

> *(p.41-42)*: "H and I are considered to be always prime, but when the value is not prime it can have no divisor less than 7."

> *(p.44)*: "F will be either a 3-par or a 5-par number... These two conditions occur when a and b are of the same, or the opposite, quaternary parity."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | C=2be+2e² (in terms of b,e only); C²=4E²+4EF (gnomon identity) | PASS |
| C2 | F²=G²-C² (complement of Pythagorean theorem); F=d²-e²=(d-e)(d+e)=ab | PASS |
| C3 | G has no prime factor <5; proof: G≡1(mod 4) → 2∤G; gcd(d,e)=1 → 3∤G | PASS |
| C4 | H and I have no prime factor <7; proof: H,I always odd; 2∉QR(mod 3,5) → 3,5∤H,I | PASS |
| C5 | F is 5-par iff a≡b(mod 4); F is 3-par iff a≢b(mod 4) | PASS |

## Mathematical Details

### C1: C as a Quadratic in e

Substituting d = b+e into C = 2de:

**C = 2de = 2(b+e)e = 2be + 2e²**

This expresses C purely in terms of b and e (without d), useful for computing C from the two root inputs.

**Gnomon identity C² = 4E² + 4EF**:

Proof: C² = (2de)² = 4d²e² = 4e²(e² + ab) = 4E² + 4·E·F

where the step e² + ab = e² + (d²-e²) = d² uses F = ab = d²-e². ✓

Geometrically (Figure 2 in Iverson): C² can be drawn as an L-shaped gnomon with:
- Two legs of width 2E, internal length F, external length G
- The gnomon area is C² and the subtracted inner square has area F²

### C2: F as Difference of Squares

F = ab = (d+e)(d-e) = d² - e²

This means F is the difference of two consecutive squares (D - E = d²-e²). Rearranging the Pythagorean theorem:

C² + F² = G² → **F² = G² - C²**

Geometrically: F² is an L-shaped gnomon with outer square G×G and inner square C×C removed. The gnomon has legs of width B = b² and internal length C, external length G.

### C3: Minimum Prime Factor of G

**Theorem**: For all prime Pythagorean pairs, every prime factor of G is ≥ 5.

| Prime | Why excluded from G |
|-------|---------------------|
| 2 | G = d²+e² ≡ 1 (mod 4) → G is always odd |
| 3 | 3\|G = d²+e² requires d²≡0 and e²≡0 (mod 3) → 3\|d and 3\|e, but gcd(d,e)=1: impossible |

The smallest prime factor found across 512 pairs (b,e≤35) is 5, confirming the bound is tight.

Example: b=7, e=2, d=9, a=11 → G=85=5×17. The factor 5 is attainable.

**Not always prime**: Iverson's claim G is "always prime under the definition given" is a liberal definition: G may be a power of a prime ≥5, and even products of primes ≥5. The key structural claim is "no divisor smaller than 5."

### C4: Minimum Prime Factor of H and I

**Theorem**: For all prime Pythagorean pairs, every prime factor of H=C+F and I=|C-F| is ≥ 7.

| Prime | Why excluded from H, I |
|-------|------------------------|
| 2 | C is 4-par (even), F is odd → H=C+F is odd; I=\|C-F\| is odd |
| 3 | p=3: 2 is not a QR mod 3 (QRs={0,1}) → (t±1)²≡2(mod 3) has no solution, where t=d/e; hence p∤(C±F) |
| 5 | p=5: 2 is not a QR mod 5 (QRs={0,1,4}) → (t±1)²≡2(mod 5) has no solution; hence p∤(C±F) |

**Unified proof for p∈{3,5}**: Write C/F = 2de/(d²-e²) = 2t/(t²-1) where t=d/e. For p|(C+F): C/F≡-1(mod p) → 2t≡-(t²-1)(mod p) → t²+2t-1≡0(mod p) → (t+1)²≡2(mod p). Since 2 is not a QR mod 3 or mod 5, no solution exists.

**Why 7 is the minimum**: The 3-4-5 triple has H=7 (prime, factor=7 ✓), confirming that 7 CAN appear as a factor of H.

The forbidden exceptions (11, 19, 43 from Iverson's text) are not proven here but are consistent with the verified numerical data.

### C5: Par-Class of F

F = ab where a and b are both odd (from cert [360] C3 and definition).

| a mod 4 | b mod 4 | F=ab mod 4 | F par-class |
|---------|---------|-----------|-------------|
| 1 (5-par) | 1 (5-par) | 1 | 5-par |
| 3 (3-par) | 3 (3-par) | 9≡1 | 5-par |
| 1 (5-par) | 3 (3-par) | 3 | 3-par |
| 3 (3-par) | 1 (5-par) | 3 | 3-par |

**F is 5-par iff a≡b (mod 4)** (same quaternary parity)  
**F is 3-par iff a≢b (mod 4)** (opposite quaternary parity)

Distribution across 268 prime pairs with b,e≤25: 131 have F 5-par, 137 have F 3-par. Both classes are achieved — F has no fixed par-class.

## Theorem NT Note

"Gnomon," "rectangle," "triangle," "prime," "shape" are observer projection labels for integer arithmetic structure. The prime factor bounds on G, H, I are consequences of modular arithmetic on bead values, not geometric properties.

**Depends on**: [360] Prime Triangle Structure (G is 5-par; a always odd); [355] Formal Proofs (C is 4-par; d,e opposite parity); [338] Gnomon Square (F=d²-e²); [359] Nightside Energy (3-par/5-par arithmetic)  
**Key invariant**: the prime exclusions 2,3,5 from G and 2,3,5 from H,I are both driven by modular arithmetic — 2 is not a QR mod 3 or mod 5, which is a fundamental property of the quadratic residue structure of these primes

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced all 5 claims: C=2be+2e²
and C²=4E²+4EF exact; F²=G²-C² and F=d²-e² exact; G's minimum prime
factor is exactly 5 across 512 pairs (b,e≤35); H and I's minimum prime
factor is exactly 7 across the same range; F's par-class split (131
5-par / 137 3-par across 268 pairs) matches the doc precisely.
Independently confirmed the underlying quadratic-residue fact (2 is not
a QR mod 3 — QRs are {0,1} — nor mod 5 — QRs are {0,1,4}). The validator
(`qa_pyth1_primeness_parity_shape_cert_validate.py`) is genuinely
computed, no fixture-trusting gap.
