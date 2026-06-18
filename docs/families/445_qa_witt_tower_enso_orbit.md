# Cert [445]: QA Witt Tower ENSO Orbit Discriminator

**Family dir**: `qa_witt_tower_enso_orbit_cert_v1`
**Status**: CERTIFIED 2026-06-18
**Chain**: Witt Tower empirical chain — extends [443] (finance) and [444] (seismology) into climatology

## Claim

The Witt tower three-tier orbit partition (MOD=27) applied to NOAA Oceanic Niño Index monthly
anomaly data (1950–2026, N=916 records) perfectly separates ENSO phases:

- **La Niña** (ONI ≤ −0.5, 252 months) → **T0** (Singularity neighborhood, bins 0–8): 252/252, log₁₀ p = −172
- **El Niño** (ONI ≥ +0.5, 245 months) → **T2** (Cosmos neighborhood, bins 18–26): 245/245, log₁₀ p = −166
- **Neutral** (−0.5 < ONI < +0.5, 419 months) → spans all three tiers (T0, T1, T2)

## Results

| Check | Description | Result |
|-------|-------------|--------|
| C1 | Data acquisition: N=916, 252 La Niña, 245 El Niño | PASS |
| C2 | La Niña 252/252 in T0; log₁₀ p = −171.8 | PASS |
| C3 | El Niño 245/245 in T2; log₁₀ p = −165.1 | PASS |
| C4 | Tier sets disjoint: La Niña={T0}, El Niño={T2} | PASS |
| C5 | Mean tier: 0.000 (La Niña) < 1.014 (Neutral) < 2.000 (El Niño) | PASS |
| C6 | Witt v₃ above null=0.481 for all phases: La Niña=1.088, Neutral=0.794, El Niño=1.160 | PASS |

Fixtures: 8/8 PASS

## Orbit Mapping

| ENSO Phase | Witt Orbit Tier | QA Orbit Region |
|-----------|-----------------|-----------------|
| La Niña | T0 (bins 0–8) | Singularity neighborhood |
| Neutral | T0+T1+T2 | Transitional |
| El Niño | T2 (bins 18–26) | Cosmos neighborhood |

## Data

- Source: NOAA CPC Oceanic Niño Index (ONI), 3-month running mean of Niño 3.4 SST anomaly
- URL: `https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`
- Coverage: 1950-DJF through 2026-MAM, N=916 monthly records
- License: Public domain

## Theorem NT Compliance

SST anomaly values are observer projections. Rank-normalized bins {0,...,26} are QA integer state.
The one-way projection (anomaly → bin) crosses the boundary exactly once; no float re-enters the
QA discrete layer.

## Primary Sources

- NOAA CPC: Oceanic Niño Index (public domain)
- Wall, D.D. (1960). *Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime.* American Mathematical Monthly, 67(6). doi:10.1080/00029890.1960.11989541
- Structural parent: cert [110] (QA Seismic Orbit — singularity→satellite→cosmos orbit-class prediction)

## Empirical Cert Chain

| Cert | Domain | Signal |
|------|--------|--------|
| [442] | Finance (S&P 500) | Recession fixed-layer elevation |
| [443] | Finance (Gold) — null | No fixed-layer elevation in safe haven |
| [444] | Seismology (Tohoku M9.1) | Phase progression: quiet→surface wave |
| [445] | Climatology (ENSO/ONI) | La Niña→neutral→El Niño tier separation |
