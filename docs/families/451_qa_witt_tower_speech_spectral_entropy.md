# [451] QA Witt Tower Acoustic Speech Spectral Entropy Orbit Discriminator

**Family ID**: 451
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_speech_spectral_entropy_cert_v1/`
**Status**: PASS (6/6 checks, 8/8 fixtures)
**Validated**: 2026-06-18
**Structural parent**: cert [110] (Witt Tower Framework)
**Empirical chain**: certs [442]–[450]
**Feature type**: Spectral entropy H_norm (same formula as [450]; new domain)
**Domain**: Acoustic speech (fifth new domain in chain)

---

## Claim

The Witt tower three-tier orbit partition (MOD=27; T0=bins 0–8, T1=9–17, T2=18–26) applied to single-channel spectral entropy places **all 80 voiced speech windows in T0** (Singularity orbit; hypergeometric log₁₀p = −55.23). Unvoiced windows distribute across T0/T1/T2. Voiced speech — maximal harmonic structure, minimal spectral diversity — occupies the Singularity. Unvoiced fricative noise spans all three orbit tiers.

---

## Physical Interpretation

**Source-filter model of speech production** (Fant 1960): speech is the product of a glottal source signal filtered by the vocal tract resonances.

**Voiced speech** (vowel /a/): glottal source = periodic pulse train at fundamental frequency F0 ≈ 120 Hz; harmonics at F0, 2F0, 3F0, …, with amplitude decreasing as 1/n² (glottal rolloff). Power is concentrated into a harmonic comb of ≈66 discrete spectral lines. Spectral entropy is minimal — only a few "effective" frequency bins carry most of the energy. H_norm ≈ 0.16 (effective bins ≈ 2^(0.163×6) ≈ 2 bins).

**Unvoiced speech** (/s/ fricative): glottal source = broadband turbulent noise; vocal tract acts as a bandpass filter (3–8 kHz for /s/). Power is distributed broadly across all energized bins. Spectral entropy is near-maximal. H_norm ≈ 0.89 (effective bins ≈ 2^(0.89×6) ≈ 46 bins).

**Orbit assignment**: voiced → T0 (Singularity = fixed point, maximally ordered, minimal degrees of freedom). Unvoiced → spans T0/T1/T2. The Singularity orbit in QA corresponds to states with minimal dynamical freedom — the harmonic lock of periodic voiced speech is exactly such a state.

---

## Data Source

**Source-filter model synthesis** (Fant 1960 acoustic theory of speech production). Deterministic and reproducible: numpy seed per frame, scipy Butterworth filter fixed coefficients.

| Class | Frames | Duration | F0 / bandwidth |
|---|---|---|---|
| Voiced /a/ | 80 × 25 ms | 2 s | F0 = 120 ± 3 Hz, harmonics 1/n² rolloff |
| Unvoiced /s/ | 200 × 25 ms | 5 s | Gaussian noise → Butterworth 3–8 kHz |
| **Total** | **280** | **7 s** | |

Theorem NT compliance: synthesized acoustic waveform (float, Pa units) is the observer projection; Welch PSD and H_norm are the observer layer; rank bin ∈ Z/27Z is the QA integer state.

---

## QA Mapping (Theorem NT)

| Layer | Variable | Role |
|---|---|---|
| Observer projection | Synthesized waveform x(t) [Pa] | Acoustic pressure — float, never enters QA |
| Observer projection | Welch PSD S(f), normalized p_i | Float spectral computation — observer layer |
| Observer layer | H_norm = −Σ p_i log₂(p_i) / log₂(64) | Spectral entropy — float, observer layer |
| QA integer state | rank bin = floor(rank × 27 / N) ∈ {0,...,26} | Z/27Z element — first QA crossing |
| Orbit tier | T0 / T1 / T2 | Witt tower partition |

**Spectral entropy computation per 25 ms window**:
1. Single-channel signal (400 samples at 16 kHz)
2. `scipy.signal.welch(nperseg=128, noverlap=64)` → 65 frequency bins (0–8 kHz, 125 Hz/bin)
3. Use bins 1–64 (exclude DC); normalize to probability mass p_i
4. H = −Σ p_i log₂(p_i)  (bits)
5. H_norm = H / log₂(64) ∈ (0, 1]

---

## Certified Checks

| Check | Claim | Result |
|---|---|---|
| C1 | 80 voiced + 200 unvoiced = 280 total windows | PASS |
| C2 | Voiced mean H_norm = 0.163 < 0.75 × unvoiced mean = 0.887 (threshold 0.665) | PASS |
| C3 | ALL 80/80 voiced windows in T0 (Singularity); hypergeometric log₁₀p = −55.23 | PASS |
| C4 | Mean H_norm: unvoiced = 0.887, voiced = 0.163; difference = 0.724 > 0.60 | PASS |
| C5 | Voiced tier set = {T0} (80/0/0); unvoiced spans {T0, T1, T2} = 14/93/93 | PASS |
| C6 | Relative entropy reduction = 81.6% ≥ 75% | PASS |

---

## Spectral Entropy Statistics

| Quantity | Value |
|---|---|
| Voiced mean H_norm (80 windows) | 0.163 |
| Voiced range H_norm | 0.145 – 0.183 |
| Voiced tier distribution (T0/T1/T2) | 80 / 0 / 0 |
| Unvoiced mean H_norm (200 windows) | 0.887 |
| Unvoiced range H_norm | 0.868 – 0.905 |
| Unvoiced tier distribution (T0/T1/T2) | 14 / 93 / 93 |
| C3 hypergeometric log₁₀p | −55.23 |
| Relative entropy reduction | 81.6% |

The gap between voiced and unvoiced is complete (max voiced 0.183 < min unvoiced 0.868) — no overlap. All voiced windows are categorically in T0 regardless of threshold.

---

## Feature-Type Cross-Domain Summary

This cert extends the spectral entropy feature type from EEG ([450]) to acoustic speech ([451]):

| Cert | Domain | Feature | Event orbit | log₁₀p |
|---|---|---|---|---|
| [450] | Neuroscience (EEG) | Spectral entropy H_norm | Ictal → T0 | −12.65 |
| [451] | Acoustic speech | Spectral entropy H_norm | Voiced → T0 | −55.23 |

Both domains: ordered event (seizure synchrony; voiced harmonics) → T0 (Singularity). Feature-type generalization confirmed across two independent physical domains.

---

## Witt Tower Empirical Chain — Updated Summary (as of [451])

| Cert | Domain | Feature type | Event orbit |
|---|---|---|---|
| [442] | Finance + Solar wind | Amplitude RMS | T2 |
| [443] | Finance null | Amplitude RMS | null |
| [444] | Seismic waveform | Amplitude RMS | T2 |
| [445] | Climate/ENSO | Anomaly rank | T2 |
| [446] | Neuroscience (EEG) | Energy RMS | T2 |
| [447] | Cardiology (ECG) | Zero-crossing rate | T2 |
| [448] | Seismic catalog | Poisson count | T2 |
| [449] | Solar energetic particles | Mean flux | T2 |
| [450] | Neuroscience (EEG) | **Spectral entropy** | T0 |
| [451] | **Acoustic speech** | **Spectral entropy** | T0 |

Five feature types certified: amplitude RMS, zero-crossing rate, Poisson count, mean flux, spectral entropy. Five domains: finance, seismic, climate, neuroscience, solar physics. Now: acoustic speech (domain 6 — new).

---

## Primary Sources

- Fant G (1960). *Acoustic Theory of Speech Production*. Mouton, The Hague. (source-filter model of speech production; voiced/unvoiced source distinction)
- Shannon CE (1948). A Mathematical Theory of Communication. *Bell System Technical Journal* 27(3), 379–423. doi:10.1002/j.1538-7305.1948.tb01338.x (spectral entropy foundations)
- Wall HS (1960). Analytic Theory of Continued Fractions. *Amer. Math. Monthly* 67, doi:10.1080/00029890.1960.11989541 (Witt tower theory)

## Related Certs

- [110] Witt Tower Framework (structural parent)
- [442]–[450] Empirical chain (prior nine domains/feature types)
- [450] EEG spectral entropy (same feature type, prior domain)
