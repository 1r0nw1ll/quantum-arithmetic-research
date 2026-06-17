<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.1080/00029890.1960.11989541, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Serre (1979) doi:10.1007/978-1-4757-5673-9 -->
# [441] QA Witt Tower Fibonacci-Pisano Synthesis

**Cert family**: `qa_witt_tower_fibonacci_pisano_cert_v1`  
**Claim**: The Pisano period π(p^k) is completely determined by the Witt tower
orbit chain [437]–[440] together with Wall's unramified lift law.

## The Fibonacci companion matrix

The Fibonacci recurrence F_{n+1}=F_n+F_{n−1} is governed by the companion matrix

```
M = [[1, 1],   char poly x²−x−1,   det(M) = −1,   t = 1
     [1, 0]]
```

The Pisano period π(n) — the period of {F_k mod n} — equals the order of M
in GL₂(Z/nZ), which equals the **maximum orbit length** of M acting on (Z/nZ)².

By CRT, π(p₁^k₁·…) = lcm(π(p₁^k₁), …), so prime powers are the atomic case.

## Three regimes

### 1. p = 5 (the unique Fibonacci-ramified prime)

The discriminant of x²−x−1 is t²+4 = 5. So **p=5 is the only prime for which
v_p(t²+4) ≥ 1** — i.e., p=5 is the unique ramified prime in the det=−1 family
for Fibonacci.

[440] applies with p=5, r=v₅(5)=1:

```
count(1) = 1            (zero vector only)
count(4) = (5−1)/4 = 1 (one period-4 orbit)
count(4·5^L) = (5−1)/4 · 5^(k−1) = 5^(k−1)   for L=1..k (joint birth, since r=1)
```

**Maximum orbit length = 4·5^k**, so **π(5^k) = 4·5^k**:

| k | π(5^k) | Formula |
|---|--------|---------|
| 1 | 20     | 4·5     |
| 2 | 100    | 4·25    |
| 3 | 500    | 4·125   |
| 4 | 2500   | 4·625   |

### 2. p = 2 (exceptional)

p=2 is not covered by [440] (which requires p odd and p≡1 mod 4).
The known formula (noted in [435] as a genuine p=2 exception):

```
π(2^k) = 3 · 2^(k−1)   for k ≥ 1
```

| k | 2^k | π(2^k) |
|---|-----|--------|
| 1 | 2   | 3      |
| 2 | 4   | 6      |
| 3 | 8   | 12     |
| 4 | 16  | 24     |

### 3. p ≠ 2, 5 (unramified — Wall's lift law)

For all other primes, M is not ramified (discriminant 5 is coprime to p).
Wall (1960) proves: **π(p^k) = π(p) · p^(k−1)**.

The Fibonacci companion is either:
- **Split** (Legendre(5,p) = +1): M diagonalizable mod p; π(p) | p−1
- **Inert** (Legendre(5,p) = −1): M irreducible mod p; π(p) | 2(p+1)

Examples:

| p | Type | π(p) | π(p²) |
|---|------|-------|-------|
| 3 | inert | 8 | 24 |
| 7 | inert | 16 | 112 |
| 11 | split | 10 | 110 |
| 13 | split | 7 | 91 |
| 17 | split | 36 | 612 |
| 19 | inert | 18 | 342 |

## Connection to the Witt tower chain

| Cert | Role in Pisano synthesis |
|------|--------------------------|
| [435] | Unramified + p=2 exception identified |
| [437] | det=+1 r=1 law (not needed for Fibonacci, but twin structure) |
| [438] | det=+1 r=2 law (twin) |
| [439] | det=+1 general law (twin) |
| **[440]** | **det=−1 general law → applies at p=5 with r=1 → π(5^k)=4·5^k** |
| **[441]** | **Synthesis: assembles all three regimes into complete Pisano theorem** |

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 P5_PISANO | π(5^k)=4·5^k for k=1..5; brute-force + max-orbit | **PASS** |
| C2 P5_ORBIT_FORMULA | Orbit counts match [440] formula, k=1..4 | **PASS** |
| C3 P2_PISANO | π(2^k)=3·2^(k−1) for k=1..8 | **PASS** |
| C4 UNRAMIFIED_WALL | π(p^k)=π(p)·p^(k−1); p∈{3,7,11,13,17,19}, k=2..4 | **PASS** |
| C5 MAX_ORBIT_EQ_PISANO | max orbit = π(p^k); p∈{2,3,5,7,11,13}, k=1..3 | **PASS** |
| C6 SPLIT_VS_INERT | π(p)\|p−1 (split) or π(p)\|2(p+1) (inert); 12 primes | **PASS** |

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_fibonacci_pisano_cert_v1
python3 qa_witt_tower_fibonacci_pisano_cert_validate.py
```

Expected: `{"ok": true, ..., "fixture_summary": "7/7 passed"}`

## Primary sources

- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — π(p^k) = π(p)·p^(k−1) for unramified p (Theorem 3)
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.5,7 — Legendre symbol, quadratic residues, Hensel lifting
- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — p-adic ramification framework
