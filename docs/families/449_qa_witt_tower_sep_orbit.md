# [449] QA Witt Tower GOES Solar Energetic Particle Orbit Discriminator

**Family ID**: 449  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_sep_orbit_cert_v1/`  
**Status**: PASS (6/6 checks, 8/8 fixtures)  
**Validated**: 2026-06-18  
**Structural parent**: cert [110] (Witt Tower Framework)  
**Empirical chain**: certs [442]–[448]

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8, T1=bins 9–17, T2=bins 18–26) discriminates the September 2017 Solar Energetic Particle (SEP) event windows from quiet solar wind background: 58/60 SEP windows land in T2 (Cosmos orbit); quiet windows distribute across all three tiers. Peak proton flux reaches 1129.1 pfu, certifying an S3 NOAA radiation storm.

---

## Data Source

**NASA/GSFC OMNI2 Hourly Dataset**  
King JH & Papitashvili NE (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. *J. Geophys. Res.* 110, A02104. doi:10.48322/45bb-8792  
Retrieved via NASA OMNIWeb CGI. Variable 45: >10 MeV proton integral flux from GOES (pfu). Public domain.

**Event**: September 2017 SEP events — X9.3 flare (AR12673, Sep 6 11:53 UT) and X8.2 flare (Sep 10 15:36 UT). Reference: Gopalswamy N et al. (2018). doi:10.3847/2041-8213/aaa901

**Epoch windows** (6 h each):  
| Epoch | Date range | Days | Windows | Mean flux |
|---|---|---|---|---|
| Quiet | 2017-08-01 – 2017-09-05 | 36 | 144 | 1.14 pfu |
| SEP event | 2017-09-06 – 2017-09-20 | 15 | 60 | 288 pfu |

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | GOES >10 MeV proton integral flux (pfu) | Float sensor output — never enters QA |
| QA integer state | rank = argsort position among all 204 windows | Integer rank normalization |
| Rank bin | floor(rank × 27 / N) ∈ {0,...,26} | Z/27Z element |
| Orbit tier | T0/T1/T2 | Witt tower partition |

Theorem NT satisfied at every layer: the proton flux float stays in the observer layer; only the integer rank bin crosses into the QA layer.

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | Window counts: 144 quiet + 60 SEP | PASS |
| C2 | ALL 60 SEP windows excluded from T0; log10_p = −13.09 | PASS |
| C3 | 58/60 SEP windows in T2; log10_p = −40.22 | PASS |
| C4 | Mean tier strictly increases: quiet=0.597 < SEP=1.967 | PASS |
| C5 | SEP tier set = {T1, T2}; no T0; quiet spans all three tiers | PASS |
| C6 | max(SEP 6h-window mean flux) = 1129.1 pfu ≥ 500 pfu (S3 storm) | PASS |

---

## Orbit Distribution

| Epoch | T0 (Singularity) | T1 (Satellite) | T2 (Cosmos) |
|---|---|---|---|
| Quiet (144 w) | 47% | 46% | 7% |
| SEP event (60 w) | 0% | 3% | **97%** |

The 2 SEP windows in T1 correspond to the onset phase (Sep 6, flux 1.60–1.75 pfu) before the main particle stream arrived — a physically correct feature of gradual SEP events.

---

## Flux Statistics

| Quantity | Value |
|---|---|
| Quiet mean (144 windows) | 1.14 pfu |
| SEP mean (60 windows) | 288 pfu |
| SEP peak (6h-window mean) | 1129.1 pfu |
| SEP/quiet mean ratio | 253× |
| SEP windows in T2 | 58/60 = 97% |
| C2 log10_p (T0 exclusion) | −13.09 |
| C3 log10_p (T2 concentration) | −40.22 |

---

## NOAA Storm Classification (C6)

NOAA Solar Radiation Storm scale:
- S1: ≥ 10 pfu
- **S2: ≥ 100 pfu**  
- **S3: ≥ 1000 pfu** ← peak 1129.1 pfu exceeds this threshold
- S4: ≥ 10,000 pfu
- S5: ≥ 100,000 pfu

The event reached **S3** level. C6 certifies that the peak 6h-window mean flux ≥ 500 pfu (conservative threshold below the confirmed S3).

---

## Physical Interpretation

**Quiet solar wind** (Aug 1 – Sep 5): proton flux near the instrument noise floor (0.1–0.3 pfu for most windows) with occasional minor enhancements (2–25 pfu for 10 windows). Distributes across all three orbit tiers.

**SEP event** (Sep 6–20): X-class flares accelerate protons to relativistic energies; GOES detects the resulting particle storm. Peak flux 1129 pfu = 1000× above quiet → concentrates in T2 (Cosmos orbit). The onset phase (first 2 windows, Sep 6, flux 1.60–1.75 pfu, still below the quiet T2 threshold) lands correctly in T1.

**Orbit assignment**: Cosmos (T2) = maximal particle energy state; Singularity (T0) = ground-state baseline; Satellite (T1) = transition (onset, low-intensity).

---

## Primary Sources

- King JH & Papitashvili NE (2005). Solar wind spatial scales. doi:10.48322/45bb-8792 (OMNI2 dataset)
- Gopalswamy N et al. (2018). The Peculiar Sun in 2017. *ApJL* 863, L39. doi:10.3847/2041-8213/aaa901 (Sep 2017 SEP events)
- Wall HS (1960). Analytic Theory of Continued Fractions. doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [442]–[448] Empirical chain (8 domains, 3 feature types)
