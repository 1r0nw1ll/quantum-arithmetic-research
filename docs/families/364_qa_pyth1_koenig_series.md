# [364] QA Pyth-1 Koenig Series and Tree of Life

**Family**: `qa_pyth1_koenig_series_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter VII pp.73-86

> *(p.67)*: "(d+e)²+(d-e)²=2G. But since d+e=a and d-e=b the formula reduces to a²+b²=2G, or A+B=2G. This means that the value of G is the median value between A and B."

> *(p.76-77)*: "Two main branches: 1,(5),7,(13),17,(25),31,... and 1,(5),7,(17),23,(37),47,... The first main branch maintains the value of b at a constant, b=1, and the second branch maintains e at a constant, e=1."

> *(p.72)*: "49 is both functionally prime and prime by the definition used in Quantum Arithmetic since it is a power of a prime number, and therefore has only one prime factor."

> *(p.72)*: "the notable exception of 2, 3, 11, 19, and 43" [never appear as H or I values]

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | A+B=2G for all prime Pythagorean pairs; G is the arithmetic mean of A and B | PASS |
| C2 | I(1,e)=2e²−1 for all e≥1 (b=1 Koenig branch); sequence 1,7,17,31,49,71,97,... | PASS |
| C3 | I(b,1)=b²−2 for all odd b≥1 (e=1 Koenig branch); sequence 1,7,23,47,79,119,... | PASS |
| C4 | Primes p≡3 or 5 (mod 8) never appear as H or I values; only p≡±1(mod 8) can | PASS |
| C5 | 49=7² is QA-functionally-prime; I(1,5)=49 is the first composite in the b=1 branch | PASS |

## Mathematical Details

### C1: A+B=2G — G as Arithmetic Mean

**Proof**:

A + B = a² + b² = (d+e)² + (d-e)²  
= (d²+2de+e²) + (d²−2de+e²)  
= 2d² + 2e² = 2G ✓

This implies G = (A+B)/2: G is exactly the arithmetic mean of A and B. Iverson calls G the "median value between A and B."

Verified for all 512 prime pairs (b,e)≤35.

### C2: The b=1 Koenig Branch: I(1,e)=2e²−1

When b=1, all (1,e) are primitive pairs since gcd(1,e)=1. Then:

d = 1+e, a = 1+2e  
C = 2de = 2(1+e)e  
F = ab = (1+2e)·1 = 1+2e  
I = |C−F| = |2e+2e²−1−2e| = |2e²−1| = 2e²−1

(since 2e²≥2>1 for all e≥1)

**Sequence**: e=1→1, e=2→7, e=3→17, e=4→31, e=5→49, e=6→71, e=7→97, e=8→127, e=9→161, e=10→199

Verified for e=1..49.

### C3: The e=1 Koenig Branch: I(b,1)=b²−2

When e=1, all (b,1) are primitive pairs since gcd(b,1)=1. Then:

d = b+1, a = b+2  
C = 2de = 2(b+1)·1 = 2b+2  
F = ab = (b+2)·b = b²+2b  
I = |C−F| = |2b+2−b²−2b| = |2−b²| = b²−2 (for b≥3)

For b=1: I = |2−1| = 1 ✓

**Sequence**: b=1→1, b=3→7, b=5→23, b=7→47, b=9→79, b=11→119, b=13→167, b=15→223, ...

Note: 119=7×17 is the first composite in this branch (at b=11). Verified for odd b=1..99.

### C4: Prime Exclusion by Quadratic Residue

H and I are instances of the quadratic form x²−2y²:

- H = C+F = 2de+(d−e)(d+e) = d²+2de−e² = (b+2e)²−2e² = a²−2e²
- I = |C−F| = |(d−e)²−2e²| = |b²−2e²|

A prime p is represented by x²−2y² if and only if 2 is a quadratic residue mod p, which occurs if and only if p≡±1(mod 8) (i.e., p≡1 or p≡7 mod 8).

| p mod 8 | 2∈QR(p)? | Appears as H,I? |
|---------|-----------|-----------------|
| 1 | Yes | Yes |
| 7 | Yes | Yes |
| 3 | No | Never |
| 5 | No | Never |

**First excluded primes** (≡3 mod 8): 3,11,19,43,59,67,...  
**First excluded primes** (≡5 mod 8): 5,13,29,37,53,61,...  
(Note: 2 is excluded since H,I are always odd per cert [361] C4.)

Iverson's "notable exceptions 2,3,11,19,43" are the first five primes excluded from H,I values.

Verified numerically: no prime p≡3 or 5(mod 8) appears in any H or I value for (b,e)≤50, all primes p<200. Primes ≡1,7(mod 8) verified present: 7,17,23,31,41,47,71,73.

### C5: 49=7² as QA-Functionally-Prime

Iverson defines "functionally prime" in QA as: a number is functionally prime if it has only one prime factor (i.e., is a prime power p^k for k≥1). This extends the standard prime concept to include prime powers.

**I(1,5)=49**:
- b=1, e=5; d=6, a=11
- C = 2×6×5 = 60; F = 11×1 = 11
- I = |60−11| = 49 = 7² ✓

49 is QA-functionally-prime because it has only one prime factor (7), even though 49 is composite in the standard sense.

The b=1 branch I-values for e=1..5: 1, 7, 17, 31, **49**

- e=1: I=1 (unit)
- e=2: I=7 (prime)  
- e=3: I=17 (prime)  
- e=4: I=31 (prime)  
- e=5: I=49=7² (first composite; but QA-functionally-prime)

This illustrates the QA refinement of primeness: the Koenig series encounters prime powers before encountering genuinely multi-factor composites.

## Theorem NT Note

"Concentric circles," "electron orbit," "Tree of Life," and geometric visualizations in Iverson's text are observer projections. The Koenig series and its properties are purely about integer arithmetic — differences, squares, and congruences — not physical or geometric objects.

The "Tree of Life" metaphor (branching from the (1,1) root into b=1 and e=1 branches, then subdividing) describes structure in the table of prime pairs. The algebraic claims C1-C5 are the discrete, falsifiable content; the geometric imagery is interpretive.

**Depends on**: [360] Prime Triangle Structure; [361] Primeness Parity Shape (H,I odd); [362] Internal Relationships (A+B=2G chain); [363] External Relationships (2-par exclusion)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced all 5 claims: A+B=2G
exact; the b=1 Koenig branch I(1,e)=2e²-1 exact (sequence
1,7,17,31,49,...); the e=1 branch I(b,1)=b²-2 exact (sequence
1,7,23,47,79,119,...); the quadratic-residue prime-exclusion rule (no
prime ≡3 or 5 mod 8 ever appears as an H/I value, verified for all
primes <200 across pairs (b,e)≤50) — independently re-derived the
underlying number theory fact (2 is a QR mod p iff p≡±1 mod 8, a
standard consequence of the second supplement to quadratic
reciprocity); and 49=7² as the first QA-functionally-prime composite in
the b=1 branch. The validator (`qa_pyth1_koenig_series_cert_validate.py`)
is genuinely computed, no fixture-trusting gap.
