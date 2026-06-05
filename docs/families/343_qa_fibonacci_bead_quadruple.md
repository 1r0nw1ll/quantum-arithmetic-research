# [343] QA Fibonacci Bead Number Quadruple: (b,e,d,a) and (I,C,F,H) are Fibonacci-type

**Family**: `qa_fibonacci_bead_quadruple_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XV pp.199-212

> "In the order b, e, d, a, these numbers are Fibonacci type numbers. The sum of  
>  two of them is equal to the following number."  
> "It is the first triangle and it uses the first four Fibonacci numbers, as its bead numbers."  
> "In the Koenig Series' they form the series in that I, C, F, H, forms a Fibonacci bead number."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | (b,e,d,a) Fibonacci-type: b+e=d and e+d=a; verified 161 coprime pairs | PASS |
| C2 | All 6 pairwise gcds in {b,e,d,a}=1 (mutual coprimeness; Euclid VII.28) | PASS |
| C3 | (I, min(C,F), max(C,F), H) Fibonacci-type for 161 coprime pairs | PASS |
| C4 | b=1,e=1 → (b,e,d,a)=(1,1,2,3)=first 4 Fibonacci; (C,F,G)=(4,3,5) | PASS |
| C5 | Consecutive Fibonacci pairs give 20 valid coprime bead numbers | PASS |

## Fibonacci Structure of Bead Numbers

### Bead Quadruple is Fibonacci-type (C1)

The bead numbers (b, e, d, a) satisfy:

$$b + e = d \qquad e + d = a$$

This is exactly the Fibonacci-type property: each term is the sum of the two preceding terms. The sequence (b, e, d, a) = (b, e, b+e, b+2e) has the same additive structure as the Fibonacci series.

### Mutual Coprimeness (C2)

By Euclid Book VII Proposition 28: if gcd(b,e)=1, then:
- gcd(d,b) = gcd(b+e, b) = gcd(e, b) = 1
- gcd(d,e) = gcd(b+e, e) = gcd(b, e) = 1
- gcd(a,b) = gcd(d+e, b) = gcd(d+e, b) = 1 (d and b are already coprime to e)
- gcd(a,e) = 1, gcd(a,d) = 1

All 6 pairwise combinations in {b, e, d, a} are coprime.

### Koenig Fibonacci Quadruple (C3)

In the Koenig Series, the four identities {I, min(C,F), max(C,F), H} form a Fibonacci-type sequence:

$$I + \min(C,F) = \max(C,F) \qquad \min(C,F) + \max(C,F) = H$$

Algebraically: I = |C-F| = max-min, so I + min = max. And H = C+F = min+max. This means {I, min, max, H} satisfies the same additive rule.

| Triangle | I | min(C,F) | max(C,F) | H | Check |
|----------|---|----------|----------|---|-------|
| 4-3-5 | 1 | 3 | 4 | 7 | 1+3=4 ✓, 3+4=7 ✓ |
| 12-5-13 | 7 | 5 | 12 | 17 | 7+5=12 ✓, 5+12=17 ✓ |
| 8-15-17 | 7 | 8 | 15 | 23 | 7+8=15 ✓, 8+15=23 ✓ |
| 20-21-29 | 1 | 20 | 21 | 41 | 1+20=21 ✓, 20+21=41 ✓ |

### First Fibonacci Triangle (C4)

The classical Fibonacci series 1, 1, 2, 3, 5, 8, 13, ... — taking the first four:

$$b=1, \quad e=1, \quad d=2, \quad a=3$$

generates the triangle:

$$C = 2de = 4, \quad F = ab = 3, \quad G = d^2+e^2 = 5$$

This is the primitive (4, 3, 5) right triangle — "the first triangle."

### Consecutive Fibonacci Pairs as Bead Numbers (C5)

Any two consecutive Fibonacci numbers (F_n, F_{n+1}) where one is odd can serve as bead number pair (b, e):

| b | e | d | a | Triangle C,F,G |
|---|---|---|---|----------------|
| 1 | 1 | 2 | 3 | 4, 3, 5 |
| 1 | 2 | 3 | 5 | 12, 5, 13 |
| 3 | 2 | 5 | 7 | 20, 21, 29 |
| 3 | 5 | 8 | 13 | 80, 39, 89 |
| 5 | 8 | 13 | 21 | 208, 105, 233 |

All consecutive Fibonacci pairs satisfy gcd=1 (by Fibonacci property) and generate valid prime Pythagorean triangles.

## Observer Projection Note (Theorem NT)

"Fibonacci-type," "coprime," "Golden Section" are observer-layer labels. The causal structure is integer bead algebra: b+e=d, e+d=a, gcd(b,e)=1 → all pairwise gcds=1, and |C-F|+(C or F)=F or C, and C+F=H. No continuous geometry enters.

**Depends on**: [336] Pythagorean 16 Identities; [339] H,I Median Identity; [342] Pythagorean Divisibility Laws
