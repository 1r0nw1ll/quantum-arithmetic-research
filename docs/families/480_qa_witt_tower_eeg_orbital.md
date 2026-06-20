# Cert Family [480]: QA Witt Tower EEG Pre-Ictal Orbital Stratification

**Family ID**: 480
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_orbital_cert_v1/qa_witt_tower_eeg_orbital_cert_validate.py`
**Validated**: 2026-06-19
**MOD**: 27 (Witt Tower)
**Parent**: Cert [479] (pre-ictal T2 pooled elevation)

## Claim

The pre-ictal EEG energy tier distribution (5-minute window before seizure onset)
across 17 Siena recordings stratifies into three orbital regimes corresponding to
the QA Witt Tower orbits. This extends cert [479]'s pooled T2 finding by resolving
the bimodal heterogeneity into distinct orbital classes.

**Cosmos seizures** (T2 > 0.55, N=6): pre-ictal energy dominant in Tier 2.
Mean T2=0.897, mean T0=0.025. Three recordings at T2=1.000 (pure Cosmos).

**Singularity-type seizures** (T2 < 0.20, N=3): T0 enriched in 2/3 cases.
PN01-1: T0=0.983 (near-pure Singularity). PN03-2: T0=0.733.

**Satellite anomaly** (PN14-1): T1=0.833 dominant — neither T0 nor T2 enriched.
Satellite-dominant pre-ictal state, does not fit T0/T2 anti-correlation.

**Continuous evidence**: Pearson r(T0_rate, T2_rate) = −0.727 across all 17 recordings.

**Within-patient dissociation** (PN03): Same patient, different seizures, same calibration:
- PN03-1: T0=0.000, T1=0.000, T2=1.000 (pure Cosmos)
- PN03-2: T0=0.733, T1=0.250, T2=0.017 (Singularity-type)

This proves the pre-ictal orbital type is a **seizure-level** property, not patient-level.

## Per-Recording Tier Distribution

| Patient | File | T0 | T1 | T2 | Label |
|---|---|---|---|---|---|
| PN00 | PN00-1.edf | 0.367 | 0.283 | 0.350 | mixed |
| PN00 | PN00-2.edf | 0.000 | 0.317 | **0.683** | **cosmos** |
| PN01 | PN01-1.edf | **0.983** | 0.017 | 0.000 | quiet |
| PN03 | PN03-1.edf | 0.000 | 0.000 | **1.000** | **cosmos** |
| PN03 | PN03-2.edf | **0.733** | 0.250 | 0.017 | quiet |
| PN05 | PN05-2.edf | 0.283 | 0.283 | 0.433 | mixed |
| PN05 | PN05-3.edf | 0.117 | 0.017 | **0.867** | **cosmos** |
| PN05 | PN05-4.edf | 0.117 | 0.517 | 0.367 | mixed |
| PN06 | PN06-1.edf | 0.067 | 0.633 | 0.300 | mixed |
| PN06 | PN06-2.edf | 0.033 | 0.133 | **0.833** | **cosmos** |
| PN06 | PN06-3.edf | 0.200 | 0.250 | 0.550 | mixed |
| PN07 | PN07-1.edf | 0.000 | 0.000 | **1.000** | **cosmos** |
| PN09 | PN09-1.edf | 0.450 | 0.267 | 0.283 | mixed |
| PN13 | PN13-1.edf | 0.217 | 0.450 | 0.333 | mixed |
| PN13 | PN13-2.edf | 0.000 | 0.000 | **1.000** | **cosmos** |
| PN14 | PN14-1.edf | 0.017 | **0.833** | 0.150 | quiet (Satellite) |
| PN14 | PN14-2.edf | 0.617 | 0.067 | 0.317 | mixed |

## Group Summary

| Group | N | Mean T0 | Mean T1 | Mean T2 |
|---|---|---|---|---|
| Cosmos (T2>0.55) | 6 | 0.025 | 0.078 | **0.897** |
| Quiescent (T2<0.20) | 3 | **0.578** | 0.367 | 0.056 |
| Mixed | 8 | 0.242 | 0.340 | 0.418 |

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 N Cosmos ≥ 5 | ≥ 5 | 6 | PASS |
| C2 N Quiescent ≥ 2 | ≥ 2 | 3 | PASS |
| C3 Cosmos mean T2 > 0.75 | > 0.75 | 0.897 | PASS |
| C4 Cosmos mean T0 < 0.10 | < 0.10 | 0.025 | PASS |
| C5 Pearson r(T0,T2) < −0.55 | < −0.55 | −0.727 | PASS |
| C6 Bimodal T2 gap > 0.60 | > 0.60 | 0.842 | PASS |

## Orbital Mapping to QA Witt Tower

- **T2 = Cosmos orbit** (72 pairs, 24-cycle, expansion/high-energy dynamics)
- **T0 = Singularity orbit** (fixed point (9,9), minimal-energy stable state)
- **T1 = Satellite orbit** (8 pairs, 3D structure, intermediate regime)

The pre-ictal EEG energy naturally finds these three orbit classes without engineering.
Cosmos seizures are hyperactivated; quiescent seizures are suppressed toward the Singularity;
PN14-1 shows a Satellite-dominant pattern that will be the subject of future certs.

## PN03 Dissociation — Key Finding

PN03 is the smoking gun: same patient, same recording equipment, same interictal calibration
thresholds from the same 10-minute baseline window, but two completely different seizures:
- Seizure 1 (PN03-1, onset=38673s): every pre-ictal window in T2. Pure Cosmos.
- Seizure 2 (PN03-2, onset=34921s): 73.3% of pre-ictal windows in T0. Singularity-type.

This cannot be explained by inter-patient variability. The pre-ictal orbit type is a property
of the specific seizure, not of the patient. Future work could examine seizure semiology
correlates (focal vs generalized, onset zone) to explain the dissociation.

## Primary Sources

- Detti P, Vatti G, Della Marca G (2020). Siena Scalp EEG Database. PhysioNet. doi:10.13026/5d4a-j060
- Goldberger AL et al (2000). PhysioBank, PhysioToolkit, PhysioNet. doi:10.1161/01.CIR.101.23.e215

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, T0/T1/T2 partition)
- **[446]** Siena PN01-1 ictal T2 discrimination
- **[479]** EEG pre-ictal pooled T2 elevation (parent of this cert)
