# [478] QA Witt Tower Real Recorded Speech

**Replication of cert [451] on real human speech (not Fant synthesis).**

## Claim

Voiced phoneme frames in **real recorded speech** preferentially occupy QA Tier 0
(T0 = low spectral entropy = periodic/harmonic spectrum). This directly answers
the objection that cert [451]'s Fant 1960 synthesis result is an artifact of
the synthesis model.

## Data

- **Source**: Real recorded human speech (Anthony Robbins, *Personal Training System*)
- **Format**: MP3, 44100 Hz stereo, 43 minutes total
- **Analysis window**: 120s from t=10s (skipping intro music)
- **Total frames**: 11,998 (25ms Hanning, 10ms hop)
- **Voiced detection**: top-40% energy frames (RMS ≥ 60th percentile)

## Pipeline (identical to cert [451])

1. Load audio via soundfile → float amplitude (observer projection)
2. 25ms Hanning-windowed frames, 10ms hop
3. H_norm = spectral entropy from rfft power / log₂(N/2+1) [observer]
4. Rank-bin H_norm over all frames → bins ∈ {0..26} → tier = bin//9 [QA state]
5. Voiced detection: RMS energy ≥ 60th percentile threshold (observer)

## Results

| Metric | Value | Baseline |
|--------|-------|----------|
| P(T0 \| voiced) | **0.5005** | 0.333 (uniform) |
| P(T0 \| unvoiced) | 0.2220 | 0.333 |
| P(voiced \| T0) | **0.6005** | 0.400 (voice frac.) |
| P(T2 \| voiced) | **0.1207** | 0.333 (depleted) |
| Voiced T0 excess | **+16.7 pp** above uniform | — |
| Permutation p (one-sided) | **0.0000** | — |

Voiced frames: T0=50.0%, T1=37.9%, T2=12.1%

Voiced frames avoid T2 (noise-like, irregular spectrum) and concentrate in T0
(harmonic, periodic spectrum). Real physical human speech shows the **identical
structural prediction** as cert [451] for synthetic Fant speech.

## Theorem NT Compliance

- Observer: waveform → Hanning window → rfft → power → H_norm (float)
- Observer: H_norm → rank → bin ∈ Z/27Z (int cast)
- QA state: tier = bin // 9 ∈ {0, 1, 2} [integer, never float]
- Energy threshold → voiced mask (observer projection only)
- No float state enters QA logic

## Certified Checks

| Check | Description | Result |
|-------|-------------|--------|
| C1 | P(T0\|voiced) > 0.45 | PASS (0.5005) |
| C2 | P(T0\|voiced) > 1.4 × P(T0\|unvoiced) | PASS (2.25×) |
| C3 | Permutation p < 0.001 | PASS (0.0000) |
| C4 | P(voiced\|T0) > 0.50 | PASS (0.6005) |
| C5 | P(T2\|voiced) < 0.20 | PASS (0.1207) |
| C6 | Voiced T0 excess > +0.10 | PASS (+0.1672) |

## Primary Sources

- Fant G (1960). Acoustic Theory of Speech Production. ISBN:9789027916006
- Stevens KN (1998). Acoustic Phonetics. ISBN:9780262692502

## Related Certs

- [451] QA Witt Tower Speech Spectral Entropy (Fant synthesis; same pipeline)
- [467] QA Witt Tower Cross-Domain MI Survey (speech included as domain)
- [468] QA Witt Tower MI Ceiling Theory (theoretical framework)
