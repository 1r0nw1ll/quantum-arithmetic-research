# [342] QA Pythagorean Divisibility Laws (Ch. IX Proofs 1-9, 11)

**Family**: `qa_pythagorean_divisibility_laws_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter IX pp.95-99

> "Statement (8): 'Within every triangle, the base, C, will contain the factor 4.'  
>  PROOF: Since either d or e must be even (Statement 1), and since C=2de, there will  
>  be two even factors of C and it will be a 4-par number and divisible by 4."  
> "Statement (11): 'The value of G is always a 5-par (4n+1) number.'  
>  PROOF: The square of an even number is 4-par, and the square of an odd number is 5-par.  
>  G=d²+e² is the sum of a 4-par and a 5-par number, and therefore must be a 5-par number."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Exactly one of d,e is even; a is always odd — 161 coprime pairs b,e<20 | PASS |
| C2 | Factor 3 in {b,e,d,a}: at least one bead divisible by 3 — 361 pairs | PASS |
| C3 | C always 4-par (divisible by 4): C=2de, one of d,e even → factor 4 — 361 pairs | PASS |
| C4 | G always 5-par (G≡1 mod 4): G=d²+e²=4-par+5-par=5-par — 361 pairs | PASS |
| C5 | Area CF/2 divisible by 6: C·F divisible by 12 — 361 pairs | PASS |

## Divisibility Laws

### C1: Parity Structure of {b, e, d, a}

Since b is always odd:
- If e is odd → d=b+e=odd+odd=**even**; e is odd
- If e is even → d=b+e=odd+even=**odd**; e is even

In either case exactly one of {d,e} is even. Since a=d+e is always an odd+even or even+odd sum, **a is always odd**.

Parity signature of bead quadruple:
- e even: (b,e,d,a) = (odd, **even**, odd, odd)
- e odd: (b,e,d,a) = (odd, odd, **even**, odd)

### C2: Factor 3 in Every Bead Set

Iverson's tri-classification proof (Statement 4): every integer is 2-tri (≡−1 mod 3), 3-tri (≡0 mod 3), or 4-tri (≡+1 mod 3). Exhausting all combinations of {b mod 3, e mod 3}:

| b mod 3 | e mod 3 | divisible by 3 |
|---------|---------|----------------|
| 0 | any | b |
| any | 0 | e |
| 1 | 1 | a=b+2e≡3≡0 |
| −1 | −1 | a=b+2e≡−3≡0 |
| 1 | −1 | d=b+e≡0 |
| −1 | 1 | d=b+e≡0 |

All 9 cases covered: **one of {b,e,d,a} divisible by 3** in every case.

### C3: C is Always 4-par

C = 2de. From C1: one of {d,e} is even, say d=2k. Then C = 2·(2k)·e = 4ke. So **4 | C**.

Verified for 361 coprime pairs b,e<30.

### C4: G is Always 5-par (≡1 mod 4)

G = d²+e². Square rules mod 4:
- Even²: (2k)²=4k²≡0 (mod 4) — **4-par**
- Odd²: (2k+1)²=4k²+4k+1≡1 (mod 4) — **5-par**

From C1: one of d,e is even, the other odd. So G = (4-par)+(5-par) ≡ 0+1=1 (mod 4). Therefore **G is always 5-par**.

Verified for 361 coprime pairs b,e<30.

### C5: Area Divisible by 6

Area = CF/2. Two ingredients:
1. **Divisible by 2**: C=4k → CF/2=2kF (even)
2. **Divisible by 3**: From C2, 3 divides one of {b,e,d,a}. Since F=ab and C=2de, 3 divides F or C. Hence 3 divides CF.

Combined: **CF divisible by 12**, so area=CF/2 divisible by 6.

The unit area is CF/2=6 for the basic (3,4,5) triangle (b=1,e=2: C=12, F=3, area=6). Every prime Pythagorean triangle area is an integer multiple of 6. This is the meaning of L=CF/12=abde/6: L counts how many "unit triangles" of area 6 fit in the given triangle.

Verified for 361 coprime pairs b,e<30.

## Observer Projection Note (Theorem NT)

"Factor 3," "4-par," "5-par," "area divisible by 6" are observer-layer divisibility labels. The causal structure is integer bead algebra: parity of b+e, mod-3 exhaustion of 9 cases, and the product identities C=2de, F=ab. No continuous geometry enters.

**Depends on**: [336] Pythagorean 16 Identities; [338] Pythagorean Gnomon Square; [337] Ellipse J,K 2D Triple
