# [325] QA To-Be-Prime / Euclid VII Proposition 28 Coprimeness Chain

**Family**: `qa_to_be_prime_euclid_prop28_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, p.2-3, "TO BE PRIME"  
**Depends on**: [320] Quantize Algorithm (BEDA coordinates)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Neighbor coprimeness: gcd(n, n+1) = 1 for all n ∈ {1,…,100} | PASS |
| C2 | Odd skip-2 coprimeness: gcd(odd n, n+2) = 1 for odd n ∈ {1,…,99}; gcd(2,4)=2 confirms Iverson's "provided n is not divisible by 2" qualifier | PASS |
| C3 | Euclid VII Prop 28 SUM: for all a,b ∈ {1,…,50} with gcd(a,b)=1, gcd(a+b,a)=1 and gcd(a+b,b)=1; ~1500 coprime pairs verified | PASS |
| C4 | BEDA coprimeness chain: for all mod-9 Cosmos pairs (b,e) with gcd(b,e)=1, Prop 28 applied twice gives gcd(d,b)=gcd(d,e)=1 (d=b+e) then gcd(a,d)=gcd(a,e)=1 (a=b+2e=d+e) | PASS |
| C5 | Euclid VII Prop 28 DIFF: for all a>b ∈ {1,…,50} with gcd(a,b)=1, gcd(a−b,a)=1 and gcd(a−b,b)=1 | PASS |

## Euclid VII Proposition 28

The foundational theorem behind QA coprimeness. Iverson's statement (p.2):

> "Book VII, Proposition 28 states that the sum and the difference between two coprime integers will be prime to both of them. This creates the four integer sequence which has been called the 'quantum number'."

In modern notation: if gcd(a,b)=1, then:
- gcd(a+b, a)=1 and gcd(a+b, b)=1 (SUM direction — C3)
- gcd(a−b, a)=1 and gcd(a−b, b)=1 (DIFF direction — C5)

**Proof**: gcd(a+b, a) = gcd(b, a) = 1 (by subtraction property of gcd). Similarly for the rest. The cert verifies this exhaustively rather than symbolically.

## BEDA Chain (C4)

The QA quadruple derives its coprimeness structure entirely from Prop 28:

```
Step 1:  gcd(b, e) = 1  [given: coprime seed]
         → gcd(b+e, b) = gcd(d, b) = 1   [Prop 28 sum]
         → gcd(b+e, e) = gcd(d, e) = 1   [Prop 28 sum]

Step 2:  gcd(d, e) = 1  [from Step 1]
         → gcd(d+e, d) = gcd(a, d) = 1   [Prop 28 sum, since a=b+2e=d+e]
         → gcd(d+e, e) = gcd(a, e) = 1   [Prop 28 sum]
```

This establishes that for coprime Cosmos seeds (gcd(b,e)=1), the BEDA quadruple has the pairwise coprimeness structure that underlies the harmonic theory in Ch.4 of QA-4.

## Iverson's Coprimeness Ladder

Iverson builds three rungs (p.2):

1. **±1 neighbor**: gcd(n, n±1) = 1 always (C1)
2. **±2 skip for odds**: gcd(odd n, n±2) = 1 (C2) — fails for even n since gcd(2k, 2k+2)=2
3. **±prime**: gcd(n, n±p) = 1 when gcd(n,p)=1 — a corollary of Prop 28

Together these show that coprimeness is "dense" in the integers — most nearby integers are coprime — which is why the QA integer pair (b,e) can nearly always be chosen to be coprime.
