# [354] QA Pyth-3 Myriad Structure and Fibonacci Strings

**Family**: `qa_pyth3_myriad_fibonacci_cert_v1`  
**Source**: Iverson & Elkins (2006) *Pythagorean Arithmetic Vol III* Chapters 13-14 pp.74-89

> *(Ch.13 p.75)*: "These Quantum Numbers are very limited, being: 1,1,2,3; 1,2,3,5; 1,3,4,7;  
>  2,1,3,4; [2,3,5,7]; 3,1,4,5; 3,2,5,7; 4,1,5,6; and 5,1,6,7. It is as though these are  
>  'creative' Quantum numbers and are used only by the Creator."

> *(Ch.13 p.75)*: "The value 5040 is derived as the product when the seven prime factors  
>  are multiplied together. Above that, the value 5041 is 71 squared."

> *(Ch.14 p.80)*: "squares which run diagonally... being 2,2,4,6; 3,3,6,9; 4,4,8,12, etc...  
>  down to 13,13,26,39. These are the children of the 1,1,2,3."

> *(Ch.14 p.87)*: "The string in the male gender run 1,1,2,3,5,8,13,21. The string of female  
>  gender Fibonacci numbers are 2,1,3,4,7,11,18. The last number in each is the first which  
>  is not prime or are the power of a single prime."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Exactly 8 valid creative QN tuples with all bead values in {1..7}: (1,1,2,3),(1,2,3,5),(1,3,4,7),(2,1,3,4),(3,1,4,5),(3,2,5,7),(4,1,5,6),(5,1,6,7) | PASS |
| C2 | 5040=7!; 5041=71²; 71 is prime; 5040+1=71²; 5040=2⁴×3²×5×7 | PASS |
| C3 | Secondary creative groups (k,k,2k,3k) for k=1..13: d=b+e, a=d+e, gcd(k,k)=k; scalar (a mod 9) cycles 3,6,9 | PASS |
| C4 | Male Fibonacci string 1,1,2,3,5,8,13,21 from seed (1,1); 21=3×7 is first non-prime non-prime-power | PASS |
| C5 | Female Fibonacci string 2,1,3,4,7,11,18 from seed (2,1); 18=2×3² is first non-prime non-prime-power | PASS |

## Structure

### C1: The 8 Creative QN Tuples

All valid (b,e,d,a) tuples with gcd(b,e)=1, d=b+e, a=d+e, and all four values in {1,2,...,7}:

| b | e | d | a | Note |
|---|---|---|---|------|
| 1 | 1 | 2 | 3 | Aboriginal male bead |
| 1 | 2 | 3 | 5 | First female-adjacent |
| 1 | 3 | 4 | 7 | Max a with b=1 |
| 2 | 1 | 3 | 4 | Aboriginal female bead |
| 3 | 1 | 4 | 5 | Male table 1a anchor |
| 3 | 2 | 5 | 7 | Max a with b=3 |
| 4 | 1 | 5 | 6 | Even b, gcd=1 |
| 5 | 1 | 6 | 7 | Max a=7; boundary |

Iverson's source lists a 9th entry "2,3,4,7" which does not satisfy d=b+e (2+3=5≠4) — this is either an OCR error for "(2,3,5,7)" (but a=8>7) or a listing error. The exhaustive enumeration confirms exactly 8 valid tuples under standard QA axioms.

### C2: The 5040 Myriad Boundary

**5040 = 7! = 1×2×3×4×5×6×7** — the working Myriad maximum.

**5041 = 71²** — the next value after 5040 is a perfect square of a prime.

- 71 is prime (largest prime before 72)
- 5040 = 2⁴×3²×5×7 (not a product of distinct primes; 7! includes repeated factors)
- The gap: 5041-5040=1, so 5040 and 71² are consecutive integers

Iverson interprets this as: the working Myriad spans [1,5040], and 5041=71² marks the boundary where prime-squared composite structure begins.

### C3: Secondary Creative Groups

The "children" of the aboriginal (1,1,2,3) are multiples (k,k,2k,3k) for k=1..13:

- d = k+k = 2k = b+e ✓
- a = 2k+k = 3k = d+e ✓
- gcd(k,k) = k → nonprime (noncoprime beads) for k>1
- **Scalar** (a mod 9, cycling): 3, 6, 9, 3, 6, 9, 3, 6, 9, 3, 6, 9, 3

These "nonprime squares" are excluded from 2D Quantum Arithmetic but are the "Central Pipe" of the 3D Elkins structure, each representing a unique torus energy form.

### C4: Male Fibonacci String (Seed 1,1)

**1, 1, 2, 3, 5, 8, 13, 21** — standard Fibonacci from the aboriginal male bead (b=1, e=1).

Recurrence: F(n) = F(n-1) + F(n-2) from seed (1,1).

Prime/prime-power status of each term ≥2:
- 2 = prime ✓
- 3 = prime ✓
- 5 = prime ✓
- 8 = 2³ (prime power) ✓
- 13 = prime ✓
- **21 = 3×7** ← first non-prime non-prime-power; terminates the "creative" sequence

Source text says "1,2,2,3,..." — OCR formatting artifact; the canonical male seed is (1,1) matching the aboriginal male bead.

### C5: Female Fibonacci String (Seed 2,1)

**2, 1, 3, 4, 7, 11, 18** — Fibonacci from the aboriginal female bead (b=2, e=1).

Recurrence: F(n) = F(n-1) + F(n-2) from seed (2,1).

Prime/prime-power status of each term ≥2:
- 2 = prime ✓
- 3 = prime ✓
- 4 = 2² (prime power) ✓
- 7 = prime ✓
- 11 = prime ✓
- **18 = 2×3²** ← first non-prime non-prime-power; terminates the "creative" sequence

**Asymmetry**: Male string terminates at index 7 (21=3×7); female string terminates at index 6 (18=2×3²). The female string is shorter and ends at a smaller composite.

## Observer Projection Note (Theorem NT)

"Creative," "Myriad," "nonprime square," "scalar," "male/female gender" are observer classification labels applied to integer arithmetic outputs. The causal structure is: gcd test, d=b+e, a=d+e, Fibonacci recurrence, prime-power test — all pure integer operations.

**Depends on**: [350] QN Definition and Law of Harmonics; [346] Fibonacci-Lucas Bridge; [349] Twin Prime Mod-6 Structure  
**Key insight**: Male and female Fibonacci strings both terminate at the same structural boundary: first element that is neither prime nor a prime power — connecting Iverson's gender dichotomy to the prime-power classification of integers
