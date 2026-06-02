# Pre-Registration: Siena Scalp EEG Replication of Combined-2 Protocol

**Status**: PRE-REGISTERED — committed before any Siena EDF signal data was read  
**Date**: 2026-06-02  
**Dataset**: Siena Scalp EEG Database v1.0.0, PhysioNet DOI 10.13026/5d4a-j060  
**Script**: `eeg_siena_combined2.py`

---

## Motivation

The CHB-MIT exploratory combined-2 result (12 patients, Fisher p=6.6×10⁻³) was
not pre-registered: the model (f0_sing + f2_sat, signed z-score) was developed
after observing the first few held-out patients. This replication tests the same
model on a fully independent dataset with the protocol fixed before data access.

---

## Dataset

- **Name**: Siena Scalp EEG Database
- **Source**: PhysioNet `https://physionet.org/files/siena-scalp-eeg/1.0.0/`
- **Patients**: PN00, PN01, PN03, PN05, PN06, PN07, PN09, PN10, PN11, PN12, PN13, PN14, PN16, PN17 (14 total)
- **Sampling rate**: 512 Hz
- **Channels**: 28–34 per patient; pipeline uses first 23 (standard 10-20 coverage)
- **Annotation format**: `Seizures-list-PNxx.txt` with wall-clock `HH.MM.SS` timestamps

---

## Protocol (fixed, no post-hoc changes)

### Feature extraction

**Feature F0 — Singularity topographic fraction (signed z-score)**  
For each 10-second window:
1. Divide into 1-second sub-windows; compute RMS topography (23-element vector).
2. Assign each sub-window to one of 4 KMeans microstates (fit on 70% training-interictal windows for that patient).
3. For each adjacent microstate pair, look up QA orbit family of `TRANSITION_TABLE[(s1, s2)]` at m=24.
4. `f0_sing_raw` = fraction of transitions that land in Singularity orbit.
5. `f0_sing_z` = (f0_sing_raw − μ_interictal_train) / σ_interictal_train  (signed, no abs).

**Feature F2 — Satellite multiband fraction (signed z-score)**  
For each 10-second window:
1. Bandpass-filter into 5 bands (delta 1–4 Hz, theta 4–8, alpha 8–13, beta 13–30, gamma 30–50).
2. Compute per-band RMS topography; fit 4 KMeans microstates (same training split).
3. Count Satellite-orbit transitions across adjacent (band × sub-window) microstate pairs.
4. `f2_sat_raw` = Satellite fraction; `f2_sat_z` = signed z-score from training-interictal baseline.

### Model

- **Baseline**: logistic regression on `delta_power_scalar` alone (1 feature, L2 penalty).
- **Augmented**: logistic regression on `[delta_power_scalar, f0_sing_z, f2_sat_z]` (3 features, L2 penalty).
- **Split**: 70% train / 30% test, stratified by label, `random_state=42`.
- **Metric**: McFadden pseudo-R² (ΔR² = R²_augmented − R²_baseline).
- **Significance**: likelihood-ratio test, df=2 (two added features).

### Inclusion / exclusion

- **Included**: patients with ≥ 12 labeled ictal windows after extraction.
- **Excluded**: patients with < 12 ictal windows (degenerate LR fit). Reason recorded in output.

### Aggregation

Fisher's method: χ² = −2 Σ ln(p_i), df = 2k (k = number of included patients).

### Primary endpoint

Fisher combined p < 0.05 across included patients.

### Secondary endpoints

1. Fraction of included patients with positive ΔR² (expected > 50% if mechanism holds).
2. Mean ΔR² and SE.
3. Polarity pattern: are there reversal patients (strong ictal z-score opposite to mean)?

---

## Orbit constants (identical to CHB-MIT pipeline)

| Microstate | QA state (b, e) |
|-----------|----------------|
| A_frontal  | (8, 3)  |
| B_occipital| (5, 16) |
| C_right    | (11, 19)|
| D_baseline | (24, 24)|

Orbit family classification: m=24 Fibonacci shift,
`orbit_fam(b, e)` → "Cosmos" / "Satellite" / "Singularity".

---

## What constitutes success / failure

| Outcome | Interpretation |
|---------|---------------|
| Fisher p < 0.05, majority positive ΔR² | Replication: mechanism generalises to Siena |
| Fisher p ≥ 0.05, majority positive ΔR² | Trend only: CHB-MIT result may not generalise cleanly |
| Fisher p ≥ 0.05, mixed/negative ΔR² | Non-replication: CHB-MIT result was dataset-specific |
| Fisher p < 0.05, majority negative ΔR² | Anomalous: orbit displacement predicts with reversed polarity |

All four outcomes are reported honestly. Non-replication is good science.

---

## Commit attestation

This document is committed at git SHA (to be recorded in results file) **before**
`eeg_siena_combined2.py` is run on any Siena EDF file. The git log provides
a tamper-evident timestamp.
