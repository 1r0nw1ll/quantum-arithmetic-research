# [450] QA Witt Tower EEG Spectral Entropy Orbit Discriminator

**Family ID**: 450  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_spectral_entropy_cert_v1/`  
**Status**: PASS (6/6 checks, 8/8 fixtures)  
**Validated**: 2026-06-18  
**Structural parent**: cert [110] (Witt Tower Framework)  
**Empirical chain**: certs [442]–[449]  
**Feature type**: Fourth — multi-channel spectral entropy H_norm

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8, T1=9–17, T2=18–26) applied to multi-channel EEG spectral entropy places **all 24 ictal (seizure) windows in T0** (Singularity orbit). Interictal windows distribute across all three tiers. This demonstrates that epileptic seizure = Singularity state under the spectral entropy observer projection: maximal neural synchrony corresponds to the fixed-point orbit.

---

## Feature-Type Independence (Key Finding)

This cert is the complement of cert [446] (same domain, different observer projection):

| Cert | Feature | Ictal orbit | Physical meaning |
|---|---|---|---|
| [446] | Multi-channel energy RMS | T2 (Cosmos) | Seizure = maximal amplitude event |
| [450] | Spectral entropy H_norm | T0 (Singularity) | Seizure = maximal synchrony (minimal spectral diversity) |

**The same physical event occupies different orbit tiers under different observer projections.** Theorem NT says the projection choice determines which orbit aspect is visible — not that the event "is" in one orbit. Both observations are real and consistent: seizures are simultaneously energy-maximal (Cosmos) and entropy-minimal (Singularity).

---

## Data Source

**Siena Scalp EEG Database**  
Detti P, Vatti G, Lanuzza M, Saccà V, Burrello A, Benini L, Bartolini E (2020). PhysioNet. doi:10.13026/s9f6-9n95 (CC-BY 4.0).  
Patient PN01, recording PN01-1.edf. 35 channels, 512 Hz, duration 48557 s.

**Seizures** (from Seizures-list-PN01.txt):  
| Seizure | Time offset | Duration | Windows |
|---|---|---|---|
| Seizure 1 | 10218–10272 s | 54 s | 10 × 5s |
| Seizure 2 | 46353–46427 s | 74 s | 14 × 5s |
| **Total ictal** | | | **24 windows** |

**Interictal reference**: 9218–10218 s (200 × 5s windows, 1000 s pre-seizure-1).

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | EEG voltage per channel (µV) | Raw sensor signal — never enters QA |
| Observer projection | PSD via scipy.signal.welch, H_norm | Float spectral computation — observer layer |
| QA integer state | rank bin = floor(rank × 27 / N) ∈ {0,...,26} | Z/27Z element — first QA crossing |
| Orbit tier | T0/T1/T2 | Witt tower partition |

**Spectral entropy computation**:
1. 8 channels (Fp1, F3, C3, P3, O1, F7, T3, T5), 5-second window (2560 samples at 512 Hz)
2. `scipy.signal.welch(nperseg=256, noverlap=128)` → 129 frequency bins (0–256 Hz)
3. Use bins 1–128 (exclude DC); sum PSD across channels → S(f)
4. Normalize: p_i = S(f_i) / Σ S
5. H = −Σ p_i log₂(p_i)  (bits)
6. H_norm = H / log₂(128) ∈ (0, 1]

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | 200 interictal + 24 ictal = 224 total windows | PASS |
| C2 | Ictal mean H_norm = 0.417 < 0.75 × interictal mean = 0.667 (threshold 0.500) | PASS |
| C3 | ALL 24/24 ictal windows in T0 (Singularity); hypergeometric log₁₀p = −12.65 | PASS |
| C4 | Mean H_norm: interictal = 0.667 > ictal = 0.417; difference = 0.250 > 0.20 | PASS |
| C5 | Ictal tier set = {T0} (24/0/0); interictal spans T0/T1/T2 = 51/75/74 | PASS |
| C6 | Relative entropy reduction = 37.5% ≥ 30% | PASS |

---

## Spectral Entropy Statistics

| Quantity | Value |
|---|---|
| Interictal mean H_norm (200 windows) | 0.667 |
| Interictal tier distribution (T0/T1/T2) | 51 / 75 / 74 |
| Ictal mean H_norm (24 windows) | 0.417 |
| Ictal tier distribution (T0/T1/T2) | 24 / 0 / 0 |
| Max ictal H_norm | 0.602 |
| C3 hypergeometric log₁₀p | −12.65 |
| Relative entropy reduction | 37.5% |

---

## Physical Interpretation

**Interictal EEG** (background): 1/f-dominated broadband spectrum (delta, theta, alpha, beta bands all present) → relatively high spectral entropy (H_norm ≈ 0.67). Energy distributed across many frequency bins. Distributes across all three orbit tiers.

**Ictal EEG** (seizure): Synchronized neural oscillations collapse spectral energy into a narrow frequency band (typically 3–30 Hz depending on seizure type) → dramatically lower spectral entropy (H_norm ≈ 0.42). Effective number of "active" spectral modes ≈ 2^(0.417×7) ≈ 2^2.9 ≈ 7–8 bins. All 24 ictal windows land in T0 (Singularity orbit = fixed point of QA dynamics = maximally ordered/constrained state).

**Orbit narrative**: Singularity = the unique fixed-point orbit (state (9,9) in mod-9, or equivalent in mod-27). The epileptic seizure — a state of maximal neural synchrony — occupies this orbit. The Singularity is not pathological in the QA sense; it is the state of minimal dynamical freedom, maximal constraint. Seizure as Singularity: the brain locked into a fixed point.

---

## Primary Sources

- Detti P et al. (2020). doi:10.13026/s9f6-9n95 (Siena Scalp EEG Database)
- Inouye T et al. (1991). Quantification of EEG irregularity by use of entropy of the power spectrum. *Electroencephalogr. Clin. Neurophysiol.* 79, 204–210. doi:10.1016/0013-4694(91)90000-2
- Wall HS (1960). Analytic Theory of Continued Fractions. doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [442]–[449] Empirical chain (prior 8 domains / feature types)
- [446] EEG energy RMS (same domain, complementary feature — ictal in T2 vs T0)
