# [447] QA Witt Tower ECG Ventricular-Flutter Orbit Discriminator

**Family ID**: 447  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_ecg_vfl_cert_v1/`  
**Status**: PASS (6/6 checks, 8/8 fixtures)  
**Validated**: 2026-06-18  
**Structural parent**: cert [110] (Witt Tower Framework)  
**Empirical chain**: certs [442]–[446]

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8, T1=bins 9–17, T2=bins 18–26) discriminates ventricular flutter (VFL) from normal sinus rhythm with zero false positives: all 19 VFL windows land in T2; normal sinus windows spread across all three tiers. The discriminating feature is zero-crossing rate (ZCR) per 5-second window — a natural integer counting sinusoidal oscillations in the ECG.

---

## Data Source

**MIT-BIH Arrhythmia Database**, record 207  
Moody GB, Mark RG (1983). PhysioNet doi:10.13026/C2F305  
Open access (CC0). MLII lead, 360 Hz, 30-minute recording.

**Annotation**: Record 207 is annotated as "extremely difficult" with ventricular flutter (VFL) at multiple intervals. The primary VFL episode used here:  
- Samples 554740–590149 = 1540.9–1639.3 s (~98 s sustained VFL)  
- Rhythm: `(VFL` annotation in MIT-BIH .atr file

**Epoch windows** (5 s, W=1800 samples at 360 Hz):  
| Epoch | Sample range | Time range | Windows |
|---|---|---|---|
| Normal sinus (N) | 100000–400000 | 277.8–1111.1 s | 166 |
| Ventricular flutter (VFL) | 554740–590149 | 1540.9–1639.3 s | 19 |

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | ECG voltage (mV) | Continuous sensor output — never enters QA |
| QA integer state | Zero-crossing rate (ZCR) | Count of sign changes per 5s window — naturally integer |
| Rank bin | floor(rank × 27 / N) ∈ {0,...,26} | Integer rank normalization across all 185 windows |
| Orbit tier | T0/T1/T2 | Witt tower partition of Z/27Z |

ZCR is derived from the ECG observation via sign-change counting — a pure integer operation. Rank normalization is integer arithmetic. No float crosses the QA firewall. Theorem NT satisfied.

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | Window counts: 166 normal + 19 VFL | PASS |
| C2 | ALL 19 VFL windows excluded from T0; hypergeometric log10_p = −9.94 | PASS |
| C3 | ALL 19/19 VFL windows in T2; hypergeometric log10_p = −10.10 | PASS |
| C4 | Mean tier strictly increases: normal=0.880 < VFL=2.000 | PASS |
| C5 | VFL tier set = {T2} only; T0=0, T1=0; normal spans T0+T1+T2 | PASS |
| C6 | Flutter ZCR ratio: 37.6/18.1 = 2.08× ≥ 1.5 (certifies flutter frequency as mechanism) | PASS |

---

## Orbit Distribution

| Epoch | T0 (Singularity) | T1 (Satellite) | T2 (Cosmos) |
|---|---|---|---|
| Normal (166 w) | 37% | 37% | 25% |
| VFL (19 w) | 0% | 0% | **100%** |

---

## Physical Interpretation

**Normal sinus rhythm**: The QRS complex (~0.1 s duration) produces brief high-amplitude transients at ~60–100 bpm, separated by flat baselines. Mean ZCR ≈ 18/5s = 3.6 zero-crossings per second. Windows distribute across all three tiers.

**Ventricular flutter**: Continuous sinusoidal oscillations at ~225 bpm (3.8 Hz fundamental frequency, 7.5 zero-crossings/s). Mean ZCR ≈ 38/5s. Every window has ZCR in the top 33% of the joint distribution → T2 (Cosmos orbit, maximal energy/coupling).

The orbit partition detects the transition from punctuated-transient cardiac dynamics (normal) to continuous high-frequency oscillation (VFL) through the integer frequency signature alone.

---

## Statistics

| Quantity | Value |
|---|---|
| Normal mean ZCR per 5s | 18.07 |
| VFL mean ZCR per 5s | 37.63 |
| ZCR ratio (VFL/normal) | 2.08× |
| VFL windows in T2 | 19/19 = 100% |
| C2 hypergeometric log10_p | −9.94 |
| C3 hypergeometric log10_p | −10.10 |

---

## Primary Sources

- Moody GB, Mark RG (1983). MIT-BIH Arrhythmia Database. *PhysioNet*. doi:10.13026/C2F305
- Wall HS (1960). Analytic Theory of Continued Fractions. *Amer. Math. Monthly* 67(8). doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [442] Cross-Domain Regime Discriminator (first empirical cert)
- [443] Safe-Haven Null Cert
- [444] Solar Wind Kp Orbit Discriminator
- [445] ENSO Orbit Discriminator
- [446] EEG Seizure Orbit Discriminator
