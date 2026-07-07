# [389] QA Witt Tower Orbit Refinement

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_orbit_refinement_cert_v1/`

## Claim

For σ(a,b) = (a+b mod m, a) acting on **ℤ[φ]/(p²)**:

> The periods at level p² are **exactly** {1} ∪ Periods_nt(p) ∪ p·Periods_nt(p), where Periods_nt(p) = non-trivial periods at level p. The Witt multiplier is always p. Period-1 (the zero element) does NOT lift to period-p.

| Check | Result |
|-------|--------|
| WITT_REFINEMENT_LAW: Periods(p²) = {1} ∪ Periods_nt(p) ∪ p·Periods_nt(p) for p ∈ {7,11,13,17,19,29,31,41} | PASS |
| TIER_COUNTS: inert/split-equal → 3 tiers; split-unequal → 5 tiers at level p² | PASS |
| WITT_MULTIPLIER_IS_P: every new period is exactly p× a level-p period | PASS |
| ZERO_DOES_NOT_LIFT: period-p does not appear at level p² | PASS |
| ORBIT_COUNTS_P4: total elements = p⁴ at level p² | PASS |

## Level p → level p² tier structure

| p | Class | Periods at p | Periods at p² | Tiers(p²) |
|---|-------|-------------|---------------|-----------|
| 7  | inert          | {1, 16}      | {1, 16, 112}              | 3 |
| 11 | split-unequal  | {1, 5, 10}   | {1, 5, 10, 55, 110}       | 5 |
| 13 | inert          | {1, 28}      | {1, 28, 364}              | 3 |
| 17 | inert          | {1, 36}      | {1, 36, 612}              | 3 |
| 19 | split-unequal  | {1, 9, 18}   | {1, 9, 18, 171, 342}      | 5 |
| 29 | split-unequal  | {1, 7, 14}   | {1, 7, 14, 203, 406}      | 5 |
| 31 | split-unequal  | {1, 15, 30}  | {1, 15, 30, 465, 930}     | 5 |
| 41 | split-equal    | {1, 40}      | {1, 40, 1640}             | 3 |

## The Witt tower law

The refinement law says: to get Periods(p²) from Periods(p):
1. Keep all periods from level p (old-part, embedded via ℤ[φ]/(p) ↪ ℤ[φ]/(p²))
2. Add p times each non-trivial period (new-part, genuinely level-p² orbits)
3. Do NOT add period-p (the zero orbit does not lift)

This is the computational signature of the **Witt vector tower** W₁(GF(p²)) ↪ W₂(GF(p²)) for inert p, and W₁(F_p×F_p) ↪ W₂(F_p×F_p) for split p.

## Why the multiplier is exactly p

For a unit u ∈ (ℤ[φ]/p)× lifted to ũ ∈ (ℤ[φ]/p²)×, the Hensel lift satisfies σ^k(ũ) ≡ σ^k(u) mod p. So if σ^k(u) = u (period k at level p), then σ^k(ũ) ≡ ũ mod p but not necessarily = ũ. The true period of ũ at level p² divides p·k. In fact it equals p·k for "generic" units (those not in the kernel of the Teichmüller section), giving the p-fold multiplier. The Teichmüller section elements have period k exactly (not p·k) — these are the "old-part."

## Hecke / Atkin-Lehner interpretation

The orbit decomposition at level p² mirrors the **old-form/new-form decomposition**:
- **Old-part** (periods k ∈ Periods_nt(p)): orbits that factor through level p. These come from the two degeneracy maps ℤ[φ]/(p²) → ℤ[φ]/(p) (mod p) and the identity section ℤ[φ]/(p) ↪ ℤ[φ]/(p²).
- **New-part** (periods p·k): orbits with genuine conductor p². These correspond to "newforms at level p²" in the Atkin-Lehner sense.

The **tier-count discriminant**: both inert p and split-equal p give 3 tiers at level p², while split-unequal p gives 5 tiers. This is a FINER test than the level-p discrimination (where split-equal and inert both have 2 non-trivial tiers). The Hecke algebra at p² "sees" whether the eigenspace orbit at level p is present.

## Finer discriminant table

| Level p tiers | Level p² tiers | Prime class |
|---|---|---|
| 2 ({1, π(p)}) | 3 | Inert or split-equal |
| 3 ({1, ord_min, π(p)}) | 5 | Split-unequal |

The 5-tier signature at level p² identifies split-unequal primes even if one only observes the level-p² dynamics without prior knowledge of the level-p structure.

## What this cert does NOT claim

- Does not certify the tower structure at level p³ or beyond (the p-multiplier persists, but the induction has not been verified above p²)
- Does not identify WHICH period-k orbits at level p² are old-part vs new-part — only that both classes exist and have the stated periods
- Does not prove the Atkin-Lehner theorem algebraically — the Hecke interpretation is observational

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_orbit_refinement_cert_v1
python qa_witt_tower_orbit_refinement_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "6/6 passed"}`

## Lineage

- Extends **[387]** (Witt carry sub-orbit invariant at p=3 inert)
- Extends **[388]** (split prime orbit geometry at level p)
- Closes the local picture: **[385]** (orbit filtration = RM), **[386]** (prime classification), **[387]** (Witt J at level p inert), **[388]** (eigenspace at level p split), **[389]** (tower p → p² for all types)

## Primary sources

- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano period tower
- Serre, J.-P. (1979). *Local Fields*. [doi.org/10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) §II.4
- Neukirch, J. (1999). *Algebraic Number Theory*. ISBN 978-3-540-65399-8 §II.3
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-computed orbit periods at
both level p and level p² from scratch (fresh script, direct
enumeration of σ(a,b)=(a+b mod m, a) orbits on (ℤ/m)², not reusing the
validator's ℤ[φ] representation) for all 8 primes {7,11,13,17,19,29,
31,41} — every period set matches the doc's table exactly, including
the 3-tier vs 5-tier split (inert/split-equal vs split-unequal). Genuine
falsifiable arithmetic, not fixture-trusting.
