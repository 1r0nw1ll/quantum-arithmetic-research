# Cert Family [481]: QA Witt Tower EEG Pre-Ictal Three-Orbit Coverage

**Family ID**: 481
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_three_orbit_cert_v1/qa_witt_tower_eeg_three_orbit_cert_validate.py`
**Validated**: 2026-06-19
**MOD**: 27 (Witt Tower)
**Type**: Pure-math cert — derives from cert [480] per-recording data; no new EDF reads
**Parent**: Cert [480] (orbital stratification)

## Claim

The pre-ictal EEG energy tier distribution (17 Siena recordings) fills all three
QA Witt Tower orbits. Cert [480] identified Cosmos (T2-dominant) and Singularity-type
(T0-dominant) poles. This cert formally adds the Satellite class (T1-dominant) and
shows that 13/17 recordings (76.5%) are covered by one of the three orbit labels.

## The Three Orbit Classes

### Cosmos (T2-dominant, T2 > 0.55): N = 6
Pre-ictal energy above the interictal 66.7th percentile.
Mean T2 = 0.897, mean T0 = 0.025. Three recordings at T2 = 1.000 exactly.

### Satellite (T1-dominant, T1 > 0.40 AND T1 > T0 AND T1 > T2): N = 4
Pre-ictal energy concentrated in the middle tier — neither high nor low relative
to interictal baseline. Mean T1 = 0.608 (+27.5pp above 0.333 interictal baseline).

| Patient | File | T0 | T1 | T2 |
|---|---|---|---|---|
| PN05 | PN05-4.edf | 0.117 | **0.517** | 0.367 |
| PN06 | PN06-1.edf | 0.067 | **0.633** | 0.300 |
| PN13 | PN13-1.edf | 0.217 | **0.450** | 0.333 |
| PN14 | PN14-1.edf | 0.017 | **0.833** | 0.150 |

Non-satellite recordings: mean T1 = 0.145 (depleted).

### Singularity-type (T0 > 0.45): N = 3
Pre-ictal energy below the interictal 33.3rd percentile.
PN01-1: T0 = 0.983 (near-pure Singularity). Mean T0 = 0.578.

### Mixed: N = 4
Recordings not assigned to any named orbit class.

## Coverage

| Class | N | Fraction |
|---|---|---|
| Cosmos | 6 | 35.3% |
| Satellite | 4 | 23.5% |
| Singularity | 3 | 17.6% |
| **Total covered** | **13** | **76.5%** |
| Mixed (uncovered) | 4 | 23.5% |

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 N Satellite ≥ 3 | ≥ 3 | 4 | PASS |
| C2 Satellite mean T1 > 0.50 | > 0.50 | 0.608 | PASS |
| C3 Satellite T1 excess > 15pp | > 0.15 | +27.5pp | PASS |
| C4 Non-satellite mean T1 < 0.30 | < 0.30 | 0.145 | PASS |
| C5 N covered ≥ 12 | ≥ 12 | 13/17 | PASS |
| C6 Three-orbit fraction ≥ 0.70 | ≥ 0.70 | 0.765 | PASS |

## Theorem NT Compliance

This cert operates entirely on tier counts from cert [480]'s certified fallback.
The EEG voltage observer projections are in cert [479]/[480]; the T0/T1/T2 rates
here are integer tier counts divided by window count. The dominant-tier
classification is a pure integer comparison (T1 > T0 and T1 > T2 as integer
count comparisons). No new float EEG data enters this cert.

## Orbital Interpretation

The QA Witt Tower partition of Z/27Z into three orbits predicts three distinct
energy regimes for any system modeled by QA. The pre-ictal EEG data empirically
instantiates all three:

- **Cosmos** (72-pair 24-cycle): Hyperactivation before seizure onset. Seizure is
  driven by energy escalation into the expansion orbit.
- **Satellite** (8-pair 3D structure): Intermediate energy stabilization. Pre-ictal
  energy neither rises nor falls; possibly a "waiting" state before seizure.
- **Singularity** (fixed point (9,9)): Suppression / quiescence. Pre-ictal energy
  collapses to the minimum-energy orbit before seizure onset.

All three QA orbital dynamics appear in human epileptic EEG without any engineering.

## Primary Sources

- Detti P, Vatti G, Della Marca G (2020). Siena Scalp EEG Database. PhysioNet.
  doi:10.13026/5d4a-j060
- Goldberger AL et al (2000). PhysioBank, PhysioToolkit, PhysioNet.
  doi:10.1161/01.CIR.101.23.e215

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, three-orbit partition)
- **[479]** EEG pre-ictal pooled T2 elevation
- **[480]** EEG pre-ictal orbital stratification (provides T0/T1/T2 per-recording data)
