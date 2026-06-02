# [302] QA Pisano mod-8 CRT

**Family**: `qa_pisano_mod8_crt_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [298] Orbit Grade Decomposition, [299] Cayley-Hamilton Fibonacci-Lucas, [301] 3-Adic Filtration

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | π(8)=12: Fibonacci mod 8 has period exactly 12; M^12≡I mod 8; no proper divisor of 12 gives I | PASS |
| C2 | M^6≡5·I mod 8: F(5)=5, F(6)=0 mod 8; the 2-adic half-period scalar; 5²≡1 mod 8 | PASS |
| C3 | CRT: π(24)=lcm(π(8),π(3))=lcm(12,8)=24; M^24≡I mod 24; confirmed via mod-3 and mod-8 components | PASS |
| C4 | M^12 mod 24 = 17·I: F(11)=89≡17, F(12)=144≡0, F(13)=233≡17 mod 24; 17²=289≡1 mod 24 | PASS |
| C5 | Exact order 24 in GL(2,Z/24Z): M^24≡I; M^k≢I for k∈{1,2,3,4,6,8,12} | PASS |

## Key result

The Cosmos period 24 arises from the **Chinese Remainder Theorem**:

```
Z/24Z  ≅  Z/8Z  ×  Z/3Z
π(24) = lcm( π(8),  π(3) ) = lcm( 12,  8 ) = 24
```

The three **grade scalars** — the scalar value of M^(half-period) at each modulus — form a coherent CRT triple:

| Modulus | Half-period k | M^k mod m | Scalar | Property |
|---------|--------------|-----------|--------|----------|
| 9 | 12 | 8·I | 8≡−1 | grade inversion ([298]) |
| 8 | 6 | 5·I | 5 | 5²≡1 mod 8 |
| 24 | 12 | 17·I | 17 | 17²≡1 mod 24 |

CRT consistency: 17≡8 mod 9 (=−1 mod 9) and 17≡1 mod 8. So the mod-24 scalar 17 is exactly what the CRT predicts from combining −1 mod 9 (from [298]) and +1 mod 8.

This closes the CRT picture: the Cosmos period 24 is not just a number — it is the **lcm of two independent Pisano periods** (for the prime 3 and the prime power 8=2³), and the grade structure at each modulus is consistent across the CRT decomposition.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.X (Fibonacci), Ch.V (CRT)
- Wall, D.D. (1960) Fibonacci Primitive Roots, *Amer. Math. Monthly* 67(6):525–532, DOI 10.1080/00029890.1960.11989541 (Pisano periods for prime powers)
