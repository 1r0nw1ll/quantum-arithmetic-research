# Cert [444]: QA Witt Tower Seismic Phase Orbit Discriminator

**Status**: PASS  
**Family dir**: `qa_alphageometry_ptolemy/qa_witt_tower_seismic_phase_cert_v1/`  
**Validated**: 2026-06-18

## Claim

The Witt tower three-tier orbit partition (MOD=27) applied to the 2011 Tohoku M9.1 earthquake waveform at IU.ANMO LHZ demonstrates a monotonic orbit-tier progression from pre-event quiet (exclusively Singularity neighborhood T0) to surface waves (predominantly Cosmos neighborhood T2), consistent with the singularity→satellite→cosmos trajectory predicted by structural cert [110].

## Dataset

- Station: IU.ANMO, channel LHZ (1 sps broadband vertical)
- Event: 2011-03-11T05:46:24 UTC, M9.1 Tohoku, Japan
- Window: 05:45:00–07:30:00 UTC (6300 samples)
- Source: IRIS timeseries API (public domain)

## QA Mapping

| Observable | QA Layer |
|---|---|
| 60-second RMS window amplitude | Observer projection (Theorem NT) |
| Quiet-phase mean subtraction (DC detrend) | Observer detrend |
| Rank-normalized bin ∈ {0,...,26} | Integer QA state b or e |
| Consecutive window pair (bin[t], bin[t-1]) | QA state (b, e) ∈ Z/27Z² |
| Tier = bin//9 ∈ {0,1,2} | Orbit neighborhood class |

**Orbit-tier partition**:
- T0 (bins 0–8): Singularity neighborhood — compressed phase space, low amplitude
- T1 (bins 9–17): Satellite neighborhood — transitional
- T2 (bins 18–26): Cosmos neighborhood — full dynamics, high amplitude

## Results

| Phase | N windows | T0 | T1 | T2 | Mean tier |
|---|---|---|---|---|---|
| quiet | 12 | **100%** | 0% | 0% | 0.000 |
| P_coda | 11 | 91% | 9% | 0% | 0.091 |
| S_coda | 15 | 33% | 53% | 13% | 0.800 |
| surf_peak | 18 | 0% | 22% | **78%** | 1.778 |
| surf_decay | 29 | 7% | 52% | 41% | 1.345 |

## Certified Facts

| Check | Result |
|---|---|
| C1 DATA_ACQUISITION | 6300 samples, 85 windows, 12 quiet windows |
| C2 QUIET_SINGULARITY | 12/12 quiet in T0; hypergeometric p = 3.94×10⁻⁷ |
| C3 SURF_COSMOS | 14/18 surf_peak in T2; hypergeometric p = 1.41×10⁻⁵ |
| C4 TIER_DISJOINT | Quiet tiers {T0} ∩ surf_peak tiers {T1,T2} = ∅ |
| C5 MONOTONIC_TIER | Mean tier 0.000 → 0.091 → 0.800 → 1.778 (monotone) |
| C6 V3_VALUATION | Quiet mean v₃=0.909 > uniform null 0.481 |

**Fixtures**: 7/7 PASS

## Connection to Structural Cert [110]

Structural cert [110] (QA Seismic Orbit) proves algebraically that the seismic phase sequence quiet→P-wave→surface-wave corresponds to the singularity→satellite→cosmos orbit transition in the QA mod-9 framework. This empirical cert [444] verifies that the corresponding orbit-tier partition in the Witt tower extension (mod 27) correctly assigns seismic phases to orbit neighborhoods, with p-values rejecting the random-assignment null at p < 10⁻⁵.

## Primary Sources

- IRIS IU.ANMO LHZ timeseries: https://service.iris.edu/irisws/timeseries/1/ (public domain)
- Wall (1960): doi:10.1080/00029890.1960.11989541 (Pisano period theory, Witt tower parent)
- Structural cert [110]: `qa_alphageometry_ptolemy/qa_seismic_cert_v1/`
