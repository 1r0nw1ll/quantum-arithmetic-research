# [391] QA σ-Norm Negation and Eigenline Zero Set

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_sigma_norm_negation_cert_v1/`

## Claim

The QA shift σ(a,b) = (a+b, a) IS multiplication by φ on ℤ[φ] under the identification (a,b) ↔ a·φ+b. Since the algebraic norm N(φ) = −1, σ is a norm-negating endomorphism, and the norm-zero locus mod p is exactly the union of σ-eigenlines.

| Check | Result |
|-------|--------|
| PHI_MULT_IDENTITY: σ(a,b)=(a+b,a) = φ·(a·φ+b) symbolically and on 8 pairs | PASS |
| NORM_OF_PHI: N(φ)=N(1,0)=0²+0·1−1²=−1; N(ψ)=N(−1,1)=−1 for Galois conjugate ψ=1−φ | PASS |
| SIGMA_NEGATES_NORM: N(σ(a,b))=−N(a,b) for 12 integer pairs | PASS |
| SIGMA_SQ_PRESERVES: N(σ²(a,b))=N(a,b) for 8 pairs | PASS |
| EIGENLINE_NORM_ZERO: at split/ram {5,11,19,29,41,59,61,71}: norm-zero locus = σ-eigenline union; at inert {2,3,7,13,17,23,43,47}: anisotropic | PASS |
| DISCRIMINANT_5: disc(−a²+ab+b²)=1−4(−1)(1)=5=disc(ℚ(√5)/ℚ); Legendre symbols consistent with [386] | PASS |
| FIBONACCI_ORBIT: σ^k(1,0)=(F_{k+1},F_k) for k=0..14 | PASS |

8 fixtures: 7 PASS, 1 designed FAIL (σ does NOT preserve the norm, it negates it).

## The algebraic identity

For z = a·φ + b ∈ ℤ[φ]:

```
φ·z = φ·(a·φ + b)
    = a·φ² + b·φ
    = a·(φ+1) + b·φ       [since φ² = φ+1]
    = a + (a+b)·φ
```

So the new (φ-coefficient, constant) = **(a+b, a) = σ(a,b)**. This is the exact QA shift.

Therefore: **σ = multiplication by φ on ℤ[φ]**, and every iterate σ^k = multiplication by φ^k.

## Norm negation

The norm of φ:

```
N(φ) = N(1·φ + 0) = 0² + 0·1 − 1² = −1
```

Since the norm is multiplicative (it's the field norm from ℚ(√5)/ℚ):

```
N(σ(z)) = N(φ·z) = N(φ)·N(z) = (−1)·N(z) = −N(z)
```

So **σ negates the norm, and σ² preserves it**. This makes σ an "anti-isometry" of the quadratic form N(a,b) = b²+ab−a².

## Norm-zero locus = eigenline union

At split prime p (p ≡ ±1 mod 5), the norm factors mod p:

```
N(a,b) = b² + ab − a² = (b·r₁ − a)(b·r₂ − a) / (−1)  [in 𝔽_p]
```

where r₁, r₂ are the two roots of x²−x−1 mod p. Therefore:

```
N(a,b) ≡ 0 mod p  ⟺  a ≡ r₁·b mod p  or  a ≡ r₂·b mod p
```

Each condition defines one σ-eigenline: {(r_i·k, k) : k ∈ 𝔽_p*}. The eigenline is precisely the eigenvector line of σ for eigenvalue r_i (cert [388]).

At inert p, x²−x−1 is irreducible over 𝔽_p, so N(a,b) is anisotropic — no non-trivial zeros. This is why the Cosmos has no "eigenline" structure for inert primes.

## Fibonacci orbit

Since σ = ×φ and φ^k = F_k·φ + F_{k-1} (standard Fibonacci identity):

```
σ^k(1, 0) = φ^k · φ  →  (F_{k+1}, F_k)
```

The orbit of φ = (1,0) under σ traces the consecutive Fibonacci pairs. This is not coincidental: it IS the Fibonacci recurrence, embodied as the φ-multiplication orbit.

## Discriminant

The binary quadratic form N as −X²+XY+Y² has discriminant:

```
Δ = B² − 4AC = 1² − 4·(−1)·1 = 5
```

This equals the discriminant of ℚ(√5)/ℚ. The splitting criterion (p splits iff disc is QR mod p, cert [386]) is the Legendre symbol (5|p), consistent with discriminant 5.

## What this certifies foundationally

Every cert in the [385]–[390] chain is a consequence of this identity:

| Cert | What it showed | Why (via [391]) |
|------|---------------|-----------------|
| [385] | QA orbits = RM structure by ℤ[φ] | σ = ×φ IS the RM action |
| [386] | Primes split by x²−x−1 root count | Roots = σ-eigenvalues; splitting = eigenspace decomposition |
| [387] | Witt carry distinguishes 3 Cosmos sub-orbits | J = coset of norm class under σ in W₂(𝔽₉) |
| [388] | Period-ord_min orbits = σ-eigenspaces | Eigenspaces = norm-zero directions (this cert) |
| [389] | Period law at p²: {1} ∪ Periods(p) ∪ p·Periods(p) | Norm of φ^k mod p² lifts via Witt vectors |
| [390] | Galois τ permutes both HMF pair and Fibonacci roots | τ acts on ℤ[φ] as the norm-preserving involution, swapping eigenvalues of σ |

## Primary sources

- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*, 2nd ed. ISBN 978-0-387-97329-6. Ch. 13: norm forms, quadratic characters, discriminant 5.
- Neukirch, J. (1999). *Algebraic Number Theory*. ISBN 978-3-540-65399-8. Ch. I §7: norm, trace, different, discriminant for ℤ[φ].
- Wall, D.D. (1960). Fibonacci series modulo m. *American Mathematical Monthly* 67(6):525–532. doi:10.1080/00029890.1960.11989541. Fibonacci matrix and Pisano period.

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified in a fresh, separate
script: σ(a,b)=φ·(a·φ+b) holds exactly for 30 random pairs (using ring
multiplication (a₁,b₁)(a₂,b₂)=(a₁a₂+a₁b₂+b₁a₂, a₁a₂+b₁b₂) derived from
φ²=φ+1); N(φ)=N(ψ)=−1; σ negates the norm and σ² preserves it for
random pairs; the Fibonacci orbit σᵏ(1,0)=(F_{k+1},F_k) holds for
k=0..14; discriminant 1−4(−1)(1)=5. Genuine falsifiable algebra — my
first attempt at independent verification actually used an incorrect
multiplication formula and produced a false mismatch, corrected and
re-confirmed. No fixture-trusting gap in the validator itself.
