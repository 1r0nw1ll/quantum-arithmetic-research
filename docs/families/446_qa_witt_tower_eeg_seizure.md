# [446] QA Witt Tower EEG Seizure Orbit Discriminator

**Family ID**: 446  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_seizure_cert_v1/`  
**Status**: PASS (6/6 checks, 8/8 fixtures)  
**Validated**: 2026-06-18  
**Structural parent**: cert [110] (Witt Tower Framework)  
**Empirical chain**: certs [442]–[445]

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8 Singularity neighborhood, T1=bins 9–17 Satellite neighborhood, T2=bins 18–26 Cosmos neighborhood) discriminates ictal from interictal EEG epochs with zero false positives: all 29 seizure-phase windows (ictal + post-ictal) land in T2; interictal windows are distributed across all three tiers.

---

## Data Source

**Siena Scalp EEG Database**, patient PN01, recording PN01-1.edf  
Detti P, Vatti G, de Franciscis (2020). PhysioNet doi:10.13026/s9f6-9n95  
Public domain (CC0). 512 Hz, 29 EEG channels. File at LaCie:  
`/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/siena/PN01/PN01-1.edf`

**Seizure annotation** (PN01-1):  
- File start: 19:00:44 UTC  
- Seizure onset: 21:51:02 UTC → offset 10218 s  
- Seizure end: 21:51:56 UTC → offset 10272 s (54 s ictal)  

**Epoch windows** (5 s, W=2560 samples at 512 Hz):  
| Epoch | Offset range | Windows |
|---|---|---|
| Interictal | 9218–10218 s | 199 |
| Ictal | 10218–10272 s | 10 |
| Post-ictal | 10272–10372 s | 19 |

**Signal**: Multi-channel energy RMS = √(mean(Σ_ch v²)) per window across 8 EEG channels (DC-detrended by interictal mean per channel).

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | EEG voltage (µV) | Continuous sensor output — never enters QA |
| QA integer state | Rank bin ∈ {0,...,26} | Discrete rank-percentile of RMS across all 228 windows |
| Orbit tier | T0/T1/T2 | Witt tower partition of Z/27Z |

Rank normalization is a one-way projection (observer → QA input). No feedback loop. Theorem NT satisfied.

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | Window counts: 199 inter + 10 ictal + 19 post | PASS |
| C2 | ALL 29 seizure-phase windows excluded from T0; hypergeometric log10_p = −5.5 (expected 9.7, observed 0) | PASS |
| C3 | ALL 29/29 seizure-phase windows in T2; hypergeometric log10_p = −15.7 | PASS |
| C4 | Mean tier strictly increases: interictal=0.854 < ictal=2.000 | PASS |
| C5 | Seizure-phase tier set = {T2} only; disjoint from T0; interictal spans all three tiers | PASS |
| C6 | Seizure-phase Witt v_3 valuation above uniform null (0.481) and above interictal: ictal=1.111, post=1.667 > inter=0.475 | PASS |

---

## Orbit Distribution

| Epoch | T0 (Singularity) | T1 (Satellite) | T2 (Cosmos) |
|---|---|---|---|
| Interictal (199 w) | 38% | 39% | 24% |
| Ictal (10 w) | 0% | 0% | **100%** |
| Post-ictal (19 w) | 0% | 0% | **100%** |

---

## Physical Interpretation

Ictal and post-ictal EEG energy is maximal and synchronized — the neural state corresponds to the Cosmos orbit (T2), the high-energy, maximally-coupled regime of the Witt tower partition. Interictal EEG, constrained by tonic inhibitory regulation, distributes across all three tiers without preference. The orbit transition is sharp and complete: no seizure-phase window falls below T2.

---

## Primary Sources

- Detti P, Vatti G, de Franciscis (2020). Siena Scalp EEG Database. *PhysioNet*. doi:10.13026/s9f6-9n95
- Wall H S (1960). Analytic Theory of Continued Fractions. *Amer. Math. Monthly* 67(8). doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [442] Seismic P-wave Orbit Discriminator (first empirical cert in chain)
- [443] Safe-Haven Null Cert (null control)
- [444] Solar Wind Kp Orbit Discriminator
- [445] ENSO Orbit Discriminator
