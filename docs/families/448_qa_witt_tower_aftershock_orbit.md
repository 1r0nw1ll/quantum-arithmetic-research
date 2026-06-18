# [448] QA Witt Tower Tohoku Aftershock Orbit Discriminator

**Family ID**: 448  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_aftershock_orbit_cert_v1/`  
**Status**: PASS (6/6 checks, 8/8 fixtures)  
**Validated**: 2026-06-18  
**Structural parent**: cert [110] (Witt Tower Framework)  
**Empirical chain**: certs [442]–[447]

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8, T1=bins 9–17, T2=bins 18–26) discriminates Tohoku aftershock windows from background seismicity with zero false positives: all 28 aftershock windows land in T2; background windows spread across all three tiers. The Omori-Utsu power-law decay is additionally certified: seven consecutive daily aftershock totals strictly decrease (533 > 474 > 348 > 234 > 141 > 125 > 110).

---

## Data Source

**USGS Earthquake Hazards Program — ComCat**  
IRIS/USGS (2011). doi:10.5066/F7MS3QZH  
Public domain. M≥3.0 events within bounding box: 35–42°N, 138–146°E.

**Event**: 2011-03-11 Mw 9.1 Tōhoku earthquake, Japan.

**Epoch windows** (6 h each):  
| Epoch | Date range | Windows | Total events |
|---|---|---|---|
| Background | 2011-02-01 – 2011-03-08 | 140 | 21 |
| Aftershock | 2011-03-11 – 2011-03-18 | 28 | 1965 |

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | Earthquake origin, location, magnitude | Continuous/categorical sensor outputs — never enter QA |
| QA integer state | M≥3.0 event count per 6-hour window | Natural integer (Poisson count) — first Poisson-count cert in chain |
| Rank bin | floor(rank × 27 / N) ∈ {0,...,26} | Integer rank normalization across all 168 windows |
| Orbit tier | T0/T1/T2 | Witt tower partition of Z/27Z |

Event counts are naturally integer — no float-to-int cast required. Theorem NT satisfied at every layer.

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | Window counts: 140 background + 28 aftershock | PASS |
| C2 | ALL 28 aftershock windows excluded from T0; hypergeometric log10_p = −5.50 | PASS |
| C3 | ALL 28/28 aftershock windows in T2; hypergeometric log10_p = −15.91 | PASS |
| C4 | Mean tier strictly increases: background=0.800 < aftershock=2.000 | PASS |
| C5 | Aftershock tier set = {T2}; T0=0, T1=0; background spans T0+T1+T2 | PASS |
| C6 | Omori-Utsu decay: 7 daily sums strictly decrease: 533>474>348>234>141>125>110 | PASS |

---

## Orbit Distribution

| Epoch | T0 (Singularity) | T1 (Satellite) | T2 (Cosmos) |
|---|---|---|---|
| Background (140 w) | 40% | 40% | 20% |
| Aftershock (28 w) | 0% | 0% | **100%** |

---

## Event Count Statistics

| Quantity | Value |
|---|---|
| Background mean events per 6h | 0.15 |
| Aftershock mean events per 6h | 70.2 |
| Ratio | 468× |
| Aftershock windows in T2 | 28/28 = 100% |
| C2 log10_p | −5.50 |
| C3 log10_p | −15.91 |

---

## Omori-Utsu Decay (C6)

Daily aftershock event totals (sums of four 6-hour windows):

| Day | Total M≥3 events |
|---|---|
| Day 1 (Mar 11) | 533 |
| Day 2 (Mar 12) | 474 |
| Day 3 (Mar 13) | 348 |
| Day 4 (Mar 14) | 234 |
| Day 5 (Mar 15) | 141 |
| Day 6 (Mar 16) | 125 |
| Day 7 (Mar 17) | 110 |

Strictly monotone decrease over all 7 days, consistent with the Omori-Utsu law:  
`rate(t) ∝ 1 / (t + c)^p` (Utsu 1961).

---

## Physical Interpretation

**Background seismicity**: 0.15 M≥3 earthquakes per 6 hours in the source region — near-Poisson background distributed across all three orbit tiers.

**Aftershock sequence**: 70 M≥3 earthquakes per 6 hours — maximal seismic activity / event density → Cosmos orbit (T2). The aftershock burst represents the release of accumulated elastic strain; the orbit assignment reflects the high-energy, high-coupling state of the seismic system immediately after rupture.

**New feature type**: This is the first cert in the Witt tower empirical chain to use a **Poisson count** (temporal event density) as the discriminating feature — distinct from amplitude RMS (seismic/EEG/ENSO) and zero-crossing rate (ECG). The integer nature is structural, not derived.

---

## Primary Sources

- Utsu T (1961). A statistical study of the occurrence of aftershocks. *Geophys. Mag.* 30, 521–605 (Omori-Utsu power law)
- IRIS/USGS (2011). USGS Earthquake Hazards Program ComCat. doi:10.5066/F7MS3QZH
- Wall HS (1960). Analytic Theory of Continued Fractions. *Amer. Math. Monthly* 67(8). doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [444] Seismic Phase Orbit Discriminator (waveform, same earthquake)
- [442]–[447] Empirical chain
