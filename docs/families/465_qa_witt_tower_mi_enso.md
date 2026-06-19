# [465] QA Witt Tower Mutual Information — ENSO Physical Domain Certificate

## Claim

The Witt tower orbit-tier partition (T0/T1/T2, MOD=27) carries I = 1.07 bits of mutual information with the ENSO phase label (La Niña / Neutral / El Niño) — 70% of the label's Shannon entropy — far exceeding the permutation null (0/5000 shuffles exceed observed MI). This cert introduces **mutual information as the 5th QA feature type** for physical domain certs [442]-[452], complementing amplitude RMS, zero-crossing rate, Poisson count, and spectral entropy.

## MI Feature Type

The four prior feature types (RMS, ZCR, Poisson count, spectral entropy) each quantify a different signal property per time window and test whether orbit-class conditional means differ. MI is different: it measures the information-theoretic dependency between orbit class and event label at the corpus level, capturing any form of statistical association — including multi-class non-monotone effects — in a single scalar (bits).

MI allows cross-domain comparison: a higher MI value in domain A vs domain B directly means orbit class is more informative about the event in domain A.

## Results

**Dataset:** NOAA Oceanic Niño Index (ONI), monthly, 1950–2026, N=916 months.

**ENSO Phase Counts:**
| Phase | N | Threshold |
|---|---|---|
| La Niña | 252 | ONI ≤ −0.5°C |
| Neutral | 419 | −0.5 < ONI < 0.5 |
| El Niño | 245 | ONI ≥ 0.5°C |

**Orbit Tier Counts:**
| Tier | Bins | N |
|---|---|---|
| T0 (Singularity) | 0–8 | 306 |
| T1 (Satellite) | 9–17 | 305 |
| T2 (Cosmos) | 18–26 | 305 |

**3×3 Contingency Table (dominant cells in bold):**

|  | La Niña | Neutral | El Niño |
|---|---|---|---|
| **T0** | **252** | 53 | 0 |
| **T1** | 0 | **305** | 0 |
| **T2** | 0 | 60 | **245** |

**Information-Theoretic Summary:**

| Metric | Value |
|---|---|
| MI observed | **1.0745 bits** |
| H(ENSO phase) | 1.5373 bits |
| MI ratio (captured) | **69.9%** |
| Permutation null (N=5000) | 0/5000 exceed MI_obs |
| perm_p | **0.0000** |

**Pointwise MI (diagonal cells):**

| Cell | Count | PMI (bits) |
|---|---|---|
| (T0, La Niña) | 252 | **+1.582** |
| (T1, Neutral) | 305 | **+1.128** |
| (T2, El Niño) | 245 | **+1.587** |

All diagonal PMI values are positive (over-represented relative to independence). The off-diagonal neutral leakage (53 months in T0, 60 in T2) suppresses total MI below the theoretical maximum of 1.537 bits but is structurally expected: neutral months are transitional and span all tiers.

## QA Mapping

- **Observer projection**: ONI float anomaly (SST, observer measurement)
- **QA integer state**: `bin = rank * 27 // N ∈ {0,...,26}` (rank-normalized into Z/27Z)
- **Orbit tier**: `tier = bin // 9 ∈ {0, 1, 2}` (Witt tower T0/T1/T2 partition)
- **MI computation**: empirical joint/marginal frequencies from integer tier × phase table
- **Theorem NT**: ONI anomaly is observer projection; rank bins are QA integer state; phase labels are observer classification output; no float enters the QA layer

## Relationship to Cert [445]

Cert [445] established the same dataset via hypergeometric tests:
- La Niña: 252/252 in T0 (log10_p = −172)
- El Niño: 245/245 in T2 (log10_p = −166)

MI [465] reframes this in information-theoretic units: **70% of ENSO phase entropy is captured by the QA orbit-tier partition alone**. The hypergeometric finding (perfect concentration of extremes) is the special case MI reveals in aggregate.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: DATA | N=916 ≥ 900, all 3 ENSO phases detected | PASS |
| C2: TIER_BALANCE | T0=306, T1=305, T2=305 (within ±3% of N/3) | PASS |
| C3: MI_MAGNITUDE | MI=1.0745 bits > 0.8 threshold | PASS |
| C4: PERM_SIG | 0/5000 null shuffles ≥ MI_obs (perm_p=0.0000 < 0.001) | PASS |
| C5: MI_RATIO | ratio=0.699 > 0.60 threshold | PASS |
| C6: DIAGONAL_DOM | All diagonal PMI values positive (+1.582, +1.128, +1.587) | PASS |

## Primary Sources

- NOAA CPC Oceanic Niño Index (public domain) via psl.noaa.gov
- Wall, H. S. (1960). doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [445] QA Witt Tower ENSO Orbit Discriminator (hypergeometric validation, same dataset)
- [110] QA Witt Tower Framework (structural parent)
- [442] QA Witt Tower Cross-Domain Regime Discriminator (physical chain base)
- [448] QA Witt Tower Tohoku Aftershock Orbit Discriminator
- [449] QA Witt Tower GOES SEP Orbit Discriminator
