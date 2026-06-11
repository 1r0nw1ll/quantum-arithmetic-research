# [385] QA Orbit Prime Ideal Filtration

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_orbit_prime_ideal_filtration_cert_v1/`

## Claim

Under the map `f(b,e) = (b mod 9) + (e mod 9)·φ` into **ℤ[φ]/(9)** (where φ²=φ+1, the ring of integers of ℚ(√5)):

> The QA orbit partition {Cosmos, Satellite, Singularity} coincides exactly with the **prime ideal filtration** {(ℤ[φ]/9)×, (3)/(9)\{0}, {0}} induced by the unique inert prime (3) over 3 in ℤ[φ] = 𝒪_{ℚ(√5)}.

Five sub-claims certified:

| Check | Tag | Result |
|-------|-----|--------|
| x²−x−1 irreducible over F₃ → 3 is inert → ℤ[φ]/(9) is local | IRREDUCIBLE_MOD3 | PASS |
| Every element outside (3) is a unit; every element of (3) is a non-unit; \|units\|=72 | LOCAL_RING | PASS |
| The 72 Cosmos pairs biject exactly with (ℤ[φ]/9)× | COSMOS_UNITS | PASS |
| The 8 Satellite pairs map to (3)\{0}; (9,9) maps to {0} | IDEAL_STRATA | PASS |
| Orbit periods 24/8/1 = Pisano π(9)/π(3)/1 → Nikolaev Lemma 3.3 real multiplication | NIKOLAEV_RM | PASS |

## Algebraic facts

- φ² = φ+1 (characteristic polynomial x²−x−1 = 0; irreducible mod 3)
- Norm form: N(a+bφ) = a²+ab−b² (indefinite over ℝ)
- ℤ[φ]/(3) = GF(9)  [3 inert ⟹ quotient is a field]
- ℤ[φ]/(9) is local with unique maximal ideal m = {a+bφ : 3∣a ∧ 3∣b}, |m|=9
- |(ℤ[φ]/9)×| = 81−9 = 72, exponent = 24 = Pisano π(9)

## Connection to Nikolaev 2024

The QA step automorphism σ: (b,e) ↦ (((b+e−1) mod 9)+1, b) acts on ℤ[φ]/(9) and generates an eventually periodic Bratteli diagram (periods 24/8/1). By Nikolaev (2024) Lemma 3.3, this implies the associated C\*-algebra has **real multiplication by ℤ[φ] = 𝒪_{ℚ(√5)}** — the maximal order of the real quadratic field ℚ(√5). The real multiplication triple is (Λ,[I],K) = (ℤ[φ],[ℤ[φ]],ℚ(√5)).

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_orbit_prime_ideal_filtration_cert_v1
python qa_orbit_prime_ideal_filtration_cert_validate.py --self-test
```

Expected output: `{"ok": true, "checks": {"IRREDUCIBLE_MOD3": true, "LOCAL_RING": true, "COSMOS_UNITS": true, "IDEAL_STRATA": true, "NIKOLAEV_RM": true}, ...}`

## Lineage

- Extends **[291]** (Fibonacci matrix orbit periods — Pisano period proof)
- Extends **[306]** (mod-24 CRT — lcm(π₃, π₈) = 24)
- Algebraic mechanism for **[281]** (QA Pisano-Orbit Correspondence)

## Primary sources

- Nikolaev, I. (2024). "Quantum arithmetic." [doi.org/10.48550/arXiv.2412.09148](https://doi.org/10.48550/arXiv.2412.09148) — Lemma 3.3 (eventually periodic Bratteli diagram ↔ real multiplication on Serre C\*-algebra)
- Wall, D.D. (1960). Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime. *American Mathematical Monthly* 67(6):525–532. [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano periods π(3)=8, π(9)=24

## What this cert does NOT claim

- Does not claim the orbit-ring identification extends to mod-24 (mod-24 has split prime 2, giving ℤ[φ]/(24) = ℤ[φ]/(8) × ℤ[φ]/(3), which is not local)
- Does not claim QA generates the full Serre C\*-algebra (only that the orbit periods satisfy Nikolaev's periodicity condition)
- Does not certify any continuous-parameter model; all arithmetic is exact integer
