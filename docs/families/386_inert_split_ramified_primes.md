# [386] QA Inert/Split/Ramified Prime Classification

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_inert_split_ramified_primes_cert_v1/`

## Claim

For **ℤ[φ] = 𝒪_{ℚ(√5)}** (ring of integers, φ²=φ+1), every prime p classifies as inert/split/ramified exactly by the root structure of x²−x−1 mod p:

> (C1) **POLY_CLASSIFY**: 0 roots → inert; 1 root → ramified (only p=5); 2 roots → split. Verified exhaustively for all primes ≤ 50.
>
> (C2) **MOD5_CRITERION**: inert ↔ p mod 5 ∈ {2,3}; split ↔ p mod 5 ∈ {1,4}; p=2 is inert. Verified for all primes ≤ 200.
>
> (C3) **INERT_PRIMITIVE_ELEMENT**: φ generates GF(4)× (ord=3) and GF(9)× (ord=8) for inert primes p=2,3. The 8-cycle QA Satellite is literally φ cycling through all of GF(9)×.
>
> (C4) **SPLIT_IDEMPOTENTS**: For p=11, Z[φ]/(11) ≅ F₁₁ × F₁₁ via CRT idempotents e₁=(2,8), e₂=(10,3) satisfying e₁²=e₁, e₂²=e₂, e₁e₂=0, e₁+e₂=1. Unit group |((ℤ[φ]/11)×)| = 100 = (p−1)².
>
> (C5) **RAMIFIED_NILPOTENT**: For p=5, x²−x−1 has double root at 3; (φ−3)=(2,1) satisfies (2,1)²=(0,0) mod 5. Unit group |((ℤ[φ]/5)×)| = 20 = p²−p.

| Check | Result |
|-------|--------|
| POLY_CLASSIFY: root count classifies all primes ≤ 50 | PASS |
| MOD5_CRITERION: p mod 5 determines class for all primes ≤ 200 | PASS |
| INERT_PRIMITIVE_ELEMENT: φ primitive in GF(4) and GF(9) | PASS |
| SPLIT_IDEMPOTENTS: CRT decomposition for p=11 | PASS |
| RAMIFIED_NILPOTENT: (2,1)²=0 mod 5 | PASS |

## Algebraic structure by class

| Class | Condition | ℤ[φ]/(p) structure | Unit group order | QA orbit structure |
|-------|-----------|---------------------|------------------|-------------------|
| Inert | p mod 5 ∈ {2,3} or p=2 | GF(p²) — field | p²−1 | 3-orbit (Cosmos/Satellite/Sing) |
| Split | p mod 5 ∈ {1,4} | F_p × F_p — not local | (p−1)² | No clean 3-orbit structure |
| Ramified | p=5 | Local, nilpotent | p²−p = 20 | Nilpotent stratum |

## Connection to QA orbits (cert [385])

Cert [385] established that the QA orbit partition for mod-9 (inert p=3) equals the prime ideal filtration. This cert explains **why** that structure exists and **when** it fails:

- **Inert primes** (p=2,3,7,...): ℤ[φ]/(p) is a field GF(p²); ℤ[φ]/(p²) is local. The 3-tier orbit structure (Cosmos=units, Satellite=(p)\{0}, Singularity={0}) is a theorem about local rings.
- **Split primes** (p=11,19,...): ℤ[φ]/(p) = F_p × F_p breaks into two independent copies. No single maximal ideal → no 3-orbit filtration.
- **Ramified prime** (p=5): Nilpotent element (φ−3) creates a distinct geometry.

## GF(9) and the Satellite orbit

The key connection to QA dynamics: φ=(0,1) in ℤ[φ]/(3) is a primitive element of GF(9)× (order 8). The QA Satellite orbit under b↦((b+e−1) mod 9)+1 IS the multiplicative orbit of φ in GF(9)×. This gives an exact algebraic identification:

```
QA Satellite 8-cycle  ≅  ⟨φ⟩ = GF(9)×
```

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_inert_split_ramified_primes_cert_v1
python qa_inert_split_ramified_primes_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {"POLY_CLASSIFY": true, "MOD5_CRITERION": true, "INERT_PRIMITIVE_ELEMENT": true, "SPLIT_IDEMPOTENTS": true, "RAMIFIED_NILPOTENT": true}, ...}`

## Lineage

- Extends **[385]** (QA Orbit Prime Ideal Filtration — inert p=3 case)
- Lays groundwork for **[387]** (Witt vector sub-orbit invariant — Teichmüller lifts)
- Lays groundwork for **[388]** (split prime orbit geometry — ℤ[φ]/(p) for p≡1,4 mod 5)

## Primary sources

- Neukirch, J. (1999). *Algebraic Number Theory*. Springer. ISBN 978-3-540-65399-8. §I.8 (splitting of primes in quadratic fields, Legendre symbol criterion)
- Wall, D.D. (1960). Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime. *American Mathematical Monthly* 67(6):525–532. [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541)
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. Springer. ISBN 978-0-387-97329-6. Ch.5 (Legendre symbol, quadratic reciprocity)

## What this cert does NOT claim

- Does not claim φ is primitive in GF(p²) for ALL inert primes (e.g. ord(φ)=16 in GF(49), not 48)
- Does not claim the 3-orbit structure exists for split or ramified primes
- Does not certify the geometry of ℤ[φ]/(p²) for p>3 (only the mod-p structure is certified)
