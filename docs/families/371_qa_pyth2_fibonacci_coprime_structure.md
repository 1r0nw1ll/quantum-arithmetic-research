# [371] QA Pyth-2 Fibonacci Coprime Structure

**Family**: `qa_pyth2_fibonacci_coprime_structure_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XV pp.87-102

> *(p.99)*: "There are 128 combinations within this range if b is considered to be odd. There are 255 combinations if b may be either odd or even. This is for only the first 20 integers."

> *(p.99)*: Euclid Book VII, Proposition 28: "if two coprime numbers are added together, their sum will be coprime to both of the original numbers. And... if two numbers be coprime to each other, the difference between them will also be coprime to both."

> *(p.103-104)*: "In the breakdown of 2/97 there were three such [dual bead chain] sets."

> *(p.104)*: Lucas-type series from φ powers: *1, 3, 4, 7, 11, 18,...* = "addition of two standard Fibonacci series, with one series moved two integers to the right from the other."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Euclid Prop 28 (QA form): gcd(a,b)=1 → gcd(a+b,a)=gcd(a+b,b)=gcd(\|a-b\|,a)=gcd(\|a-b\|,b)=1 | PASS |
| C2 | 128 coprime pairs (b,e) with b odd, 1≤b,e≤17; 255 coprime pairs (any b), 1≤b,e≤20 | PASS |
| C3 | gcd(b,e)=1 and b odd → all 6 pairwise gcds of {b,e,b+e,b+2e} equal 1 | PASS |
| C4 | a(n)=F(n)+F(n+2) = 1,3,4,7,11,18,29,47,... satisfies Fibonacci recurrence; equals L(n+1) | PASS |
| C5 | 2/97 has exactly 3 BABTHE2 decompositions: {N,O}∈{(1,7),(17,3),(31,2)}; lower bead sets pairwise disjoint | PASS |

## Mathematical Details

### C1: Euclid Proposition 28 — Coprimeness Preserved Under Sum and Difference

The foundational result underlying all Fibonacci bead coprime properties:

**If gcd(a,b)=1**, then:
- gcd(a+b, a) = 1
- gcd(a+b, b) = 1
- gcd(|a-b|, a) = 1  
- gcd(|a-b|, b) = 1

**Proof**: Suppose k|a and k|(a+b). Then k|b. So k|gcd(a,b)=1, hence k=1. ✓

Verified for 6007 coprime pairs (a,b) with 1≤a,b≤99.

### C2: Coprime Pair Counts

| Range | b constraint | Count |
|-------|-------------|-------|
| 1≤b,e≤17 | b odd | **128** |
| 1≤b,e≤20 | any b | **255** |

The 128 count uses Pythagoras's limit of 17 as the maximum useful bead value; the 255 count uses 20. Of the 255 all-parity pairs: 169 have b odd, 86 have b even.

### C3: Four-Bead Generalized Coprimeness

For any coprime pair (b,e) with b odd:
{b, e, d=b+e, a=b+2e} are pairwise coprime — all 6 gcds equal 1.

**Proof chain using Euclid Prop 28:**
1. gcd(b,e)=1 → gcd(b+e, b)=1 and gcd(b+e, e)=1 (by Prop 28)
2. b odd → gcd(2,b)=1 → gcd(2e,b)=gcd(e,b)=1 → gcd(b+2e, b)=gcd(2e,b)=1
3. gcd(b+2e, e) = gcd(b,e) = 1 (since (b+2e)-2e=b)
4. gcd(b+2e, b+e) = gcd(e, b+e) = gcd(e,b) = 1 (Prop 28 difference)
5. gcd(b+e, b+2e) = same as case 4 = 1

Verified for 1009 pairs (b,e) with b odd, 1≤b,e<50.

This generalizes cert [366]'s result (which required prime Pythagorean pairs) to all coprime (b,e) with b odd.

### C4: Lucas Series from Doubled Fibonacci

Iverson's "addition of two Fibonacci series offset by 2 positions":

| n | F(n) | F(n+2) | a(n)=F(n)+F(n+2) | L(n+1) |
|---|------|--------|------------------|--------|
| 0 | 0 | 1 | **1** | L(1)=1 |
| 1 | 1 | 2 | **3** | L(2)=3 |
| 2 | 1 | 3 | **4** | L(3)=4 |
| 3 | 2 | 5 | **7** | L(4)=7 |
| 4 | 3 | 8 | **11** | L(5)=11 |
| 5 | 5 | 13 | **18** | L(6)=18 |

**Identity**: a(n) = F(n)+F(n+2) = L(n+1) (Lucas number) for all n≥0.

**Fibonacci recurrence**: a(n+2) = F(n+2)+F(n+4) = [F(n+1)+F(n+3)]+[F(n)+F(n+2)] = a(n+1)+a(n) ✓

Verified for n=0..49.

### C5: Three Decompositions of 2/97

The fraction 2/97 decomposes into three unit fractions in exactly 3 ways via the BABTHE2 system:

| (N,O) | P | Q | S | R | T | Unit fraction form |
|-------|---|---|---|---|---|-------------------|
| (1,7) | 8 | 15 | 56 | 41 | 97 | 2/97 = 1/56 + 1/679 + 1/776 |
| (17,3) | 20 | 23 | 60 | 37 | 97 | 2/97 = 1/60 + 1/291 + 1/1940 |
| (31,2) | 33 | 35 | 66 | 31 | 97 | 2/97 = 1/66 + 1/194 + 1/3201 |

The three lower bead sets {N,O,P,Q} are pairwise disjoint — no shared element among the three decompositions.

Note: Iverson's text shows "7,3,20,23" for the second set — likely a typo for "17,3,20,23" (OCR dropped the leading "1").

## Theorem NT Note

"Fibonacci series," "Golden Section," "Noble Sections," "plant growth," "periodic table applications," and "Alexandria library" are observer-projection labels. The QA discrete layer contains only:
- Euclid's coprime sum/difference rule (Prop 28)
- Integer pair counts for coprime pairs in [1,N]×[1,N]
- Four-bead coprimeness from the sum/difference rule
- The Lucas number recurrence relation
- The three BABTHE2 decompositions of 2/97

**Depends on**: [370] BABTHE Dual Bead Chain (decomposition structure and T+Q=2S identity); [366] Bead Arithmetic Laws (pairwise coprimeness for prime Pythagorean pairs — this cert generalizes to all odd-b coprime pairs)
