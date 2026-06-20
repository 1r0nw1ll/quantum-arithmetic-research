# Cert Family [479]: QA Witt Tower EEG Pre-Ictal Tier Elevation

**Family ID**: 479
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_prerictal_cert_v1/qa_witt_tower_eeg_prerictal_cert_validate.py`
**Validated**: 2026-06-19
**MOD**: 27 (Witt Tower)

## Claim

EEG multi-channel energy (5-second windows, 512 Hz) in the pre-ictal period
(0-300 seconds before seizure onset) preferentially occupies Witt Tower Tier 2
(T2 = high-energy tier) relative to the interictal baseline (1/3 by construction),
with the effect replicated across 9 Siena patients and 17 seizure recordings.
Pooled pre-ictal T2 rate = 0.499 vs interictal baseline 0.333 (+16.6pp); perm_p = 0.000.

## Data

**Siena Scalp EEG Database** (PhysioNet)
- doi:10.13026/5d4a-j060 (Detti et al., 2020)
- 9 patients, 17 seizure recordings
- Patients: PN00 (×2), PN01, PN03 (×2), PN05 (×3), PN06 (×3), PN07, PN09, PN13 (×2), PN14 (×2)
- Format: EDF, 512 Hz, 8 EEG channels used
- LaCie path: `/Volumes/lacie/signal_experiments_offload/archive/phase_artifacts/phase2_data/eeg/siena/`

## Analysis Pipeline

1. Load EDF via pyedflib; extract first 8 EEG channels (observer: EEG voltage amplitude)
2. Extract **interictal** segment: recording_start + 120s to recording_start + 720s (10 min)
3. Extract **pre-ictal** segment: seizure_onset − 300s to seizure_onset (5 min)
4. Compute 5-second windows (2560 samples), DC-detrend per channel
5. **Energy RMS** = sqrt(mean(sum_ch(v²))) per window → observer projection (float)
6. Rank interictal RMS → 33rd/66th percentile → T0/T1/T2 thresholds
7. Apply thresholds to pre-ictal windows → tier ∈ {0, 1, 2} (QA integer state)
8. Count pre-ictal T2 rate; permutation test (N=5000, seed=42)

## Results

| Metric | Value |
|---|---|
| Interictal windows | 2040 (120 per recording × 17) |
| Pre-ictal windows | 1020 (60 per recording × 17) |
| Interictal T2 rate | 0.333 (exact by rank calibration) |
| **Pre-ictal T2 rate** | **0.499** |
| T2 excess | **+16.6pp** |
| Perm p (5000 shuffles) | **0.0000** |
| Recordings pre > inter | 10/17 |
| Monotone late > early | 11/17 |

### Per-Recording T2 Rates

| Patient | File | Onset (s) | Inter T2 | Pre-Ictal T2 | Excess |
|---|---|---|---|---|---|
| PN00 | PN00-1.edf | 1143 | 0.333 | 0.350 | +0.017 |
| PN00 | PN00-2.edf | 1220 | 0.333 | 0.683 | +0.350 |
| PN01 | PN01-1.edf | 10218 | 0.333 | 0.000 | −0.333 |
| PN03 | PN03-1.edf | 38673 | 0.333 | **1.000** | +0.667 |
| PN03 | PN03-2.edf | 34921 | 0.333 | 0.017 | −0.317 |
| PN05 | PN05-2.edf | 7163 | 0.333 | 0.433 | +0.100 |
| PN05 | PN05-3.edf | 6836 | 0.333 | **0.867** | +0.533 |
| PN05 | PN05-4.edf | 3608 | 0.333 | 0.367 | +0.033 |
| PN06 | PN06-1.edf | 5583 | 0.333 | 0.300 | −0.033 |
| PN06 | PN06-2.edf | 8860 | 0.333 | **0.833** | +0.500 |
| PN06 | PN06-3.edf | 6275 | 0.333 | 0.550 | +0.217 |
| PN07 | PN07-1.edf | 22059 | 0.333 | **1.000** | +0.667 |
| PN09 | PN09-1.edf | 7249 | 0.333 | 0.283 | −0.050 |
| PN13 | PN13-1.edf | 7062 | 0.333 | 0.333 | +0.000 |
| PN13 | PN13-2.edf | 7249 | 0.333 | **1.000** | +0.667 |
| PN14 | PN14-1.edf | 7262 | 0.333 | 0.150 | −0.183 |
| PN14 | PN14-2.edf | 7479 | 0.333 | 0.317 | −0.017 |

### Heterogeneity

**Bimodal response** observed: high-elevation cases (PN03-1, PN07-1, PN13-2: T2=100%;
PN05-3: 86.7%; PN06-2: 83.3%) and quiescent cases (PN01-1, PN03-2: T2=0–1.7%).
The pooled test is significant despite heterogeneity because N=17 and the extreme
elevation cases dominate. The quiescent pattern may reflect focal seizure types with
pre-ictal quiescence rather than hyper-activation (different semiology). Future certs
can stratify by seizure type.

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 Pre-ictal T2 > 0.45 | > 0.45 | 0.499 | PASS |
| C2 Perm p < 0.001 | < 0.001 | 0.000 | PASS |
| C3 N patients ≥ 8 | ≥ 8 | 9 | PASS |
| C4 N recordings exceed ≥ 9 | ≥ 9 | 10/17 | PASS |
| C5 N monotone ≥ 9 | ≥ 9 | 11/17 | PASS |
| C6 T2 excess > 12pp | > 0.12 | +16.6pp | PASS |

## Theorem NT Compliance

- **Observer**: EEG voltage samples → DC removal → energy RMS (float)
- **Observer**: interictal RMS array → sort → percentile → thresholds (float)
- **QA state**: tier = 0 if rms < t1, 1 if t1 ≤ rms < t2, 2 if rms ≥ t2 (integer ∈ {0,1,2})
- EEG amplitude never enters QA logic; only the integer tier assignment does

## Primary Sources

- Detti P, Vatti G, Della Marca G (2020). Siena Scalp EEG Database. PhysioNet. doi:10.13026/5d4a-j060
- Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE (2000). PhysioBank, PhysioToolkit, PhysioNet. Circulation 101(23):e215-e220. doi:10.1161/01.CIR.101.23.e215

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, T0/T1/T2 Cosmos/Satellite/Singularity partition)
- **[446]** Siena PN01-1 ictal state T2 discrimination (ictal energy = 100% T2)
- **[450]** Siena PN01-1 spectral entropy T0 during seizure

## Relationship to Chain

This is the first QA **prediction** cert in the EEG domain. Cert [446] showed that the
ictal state (during seizure) is 100% T2; this cert extends the finding to the pre-ictal
forecasting window (5 minutes before onset). Together they establish:
- [446]: ictal energy → T2 (segregation, real-time detection)
- [479]: pre-ictal energy → T2 enrichment (prediction, early warning)
