# [349] QA Twin Prime Mod-6 Structure

**Family**: `qa_twin_prime_mod6_cert_v1`  
**Source**: Iverson & Elkins (2006) *Pythagorean Arithmetic Vol III* Chapter 5 pp.24-25

> "all so-called prime numbers are one of a pair of twin primes. So let us start by  
>  going through and listing them up to 100. They are: 5-7, 11-13, 17-19, 23-25, 29-31,  
>  35-37, 41-43, 47-49, 53-55, 59-61, 65-67, 71-73, 77-79, 83-85, 89-91, 95-97, 101-103.  
>  Note the total absence of the prime number 3. This is what creates the twin primes.  
>  The intervening integer between the twin primes is always divisible by 3."

> *(Euclid VII.28, p.25)*: "The sum of two numbers which are prime to each other, are  
>  also prime to them. The difference of two numbers which are prime to each other is  
>  also prime to them."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Iverson twin pairs (6k-1, 6k+1) have midpoints 6k divisible by 6; verified k=1..17 (up to 101-103) | PASS |
| C2 | All primes p > 3 satisfy p ≡ ±1 (mod 6); no prime >3 is ≡ 0, 2, 3, or 4 (mod 6) | PASS |
| C3 | Squares of primes ≥5 are ≡ 1 (mod 6): 25≡1, 49≡1, 121≡1 (Iverson's exceptions) | PASS |
| C4 | Products of two distinct primes ≥5 are ≡ ±1 (mod 6): {35,55,65,77,85,91,95} verified | PASS |
| C5 | Euclid VII.28: gcd(b,e)=1, b odd → (b,e,d,a) all mutually coprime (QA bead structure) | PASS |

## Structure

### Twin Prime Mod-6 Pattern (C1-C4)

Iverson's extended definition includes all pairs (6k-1, 6k+1) where:
- **True prime pairs**: both endpoints are conventional primes (5-7, 11-13, 17-19, ...)
- **Type 1 exceptions**: one endpoint is a prime power (23-**25**, 47-**49**, 119-**121**)
- **Type 2 exceptions**: one endpoint is a product of two primes ≥5 (**35**-37, 53-**55**, 65-67, 71-73, 77-79, 83-**85**, 89-**91**, **95**-97)

All three categories have midpoints divisible by 6 and endpoints ≡ ±1 (mod 6) by construction.

**Why all primes >3 are ≡ ±1 (mod 6)**: Every integer is ≡ 0,1,2,3,4,5 (mod 6). The classes 0 (divisible by 6), 2 (divisible by 2), 4 (divisible by 2), and 3 (divisible by 3) cannot be prime. Only ≡1 and ≡5≡−1 remain. The absence of the prime factor 3 forces all such primes into the ±1 classes.

**Inheritance**: If p≡±1 (mod 6) then p²≡1 (mod 6). If p≡±1 and q≡±1 then pq≡±1 (mod 6). Both exception types inherit the ±1 residue class from their prime factors.

### Euclid VII.28 and QA Bead Structure (C5)

Iverson's Fibonacci configuration (b, e, d, a) with d=b+e, a=d+e=b+2e:

| Pair | Condition | Result |
|------|-----------|--------|
| (b, e) | gcd(b,e)=1 | given |
| (d, b) | Euclid VII.28 on sum | gcd(d,b)=1 |
| (d, e) | Euclid VII.28 on sum | gcd(d,e)=1 |
| (a, d) | VII.28 on (d,e): a=d+e | gcd(a,d)=1 |
| (a, e) | VII.28 on (d,e): a=d+e | gcd(a,e)=1 |
| (a, b) | b odd, p\|gcd(a,b)→p\|2e, p odd→p\|e, contradiction | gcd(a,b)=1 |

When b is odd (Iverson's prime Pythagorean requirement), all 6 pairwise gcd values equal 1. The four bead numbers (b, e, d, a) are **automatically mutually coprime** — no verification needed beyond the initial gcd(b,e)=1 and b odd.

Verified for 161 coprime pairs (b odd, e any) with b, e ∈ [1, 19].

## Observer Projection Note (Theorem NT)

Mod-6 residue classes are observer labels on integer arithmetic. The causal structure: integer prime factorization, modular arithmetic mod 6, gcd computation via Euclidean algorithm. No continuous functions enter the QA causal layer.

**Depends on**: [344] Prime Residue Symmetry; [348] Fibonacci Divisibility Periods  
**Enables**: [350+] QA Quantum Number definition (integer with 4-7 prime factors)
