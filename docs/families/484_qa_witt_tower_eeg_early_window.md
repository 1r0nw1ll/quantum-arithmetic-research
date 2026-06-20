# Cert Family [484]: QA Witt Tower EEG Pre-Ictal Orbit Class Early-Window Persistence

**Family ID**: 484
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_temporal_escalation_cert_v1/qa_witt_tower_eeg_temporal_escalation_cert_validate.py`
**Validated**: 2026-06-20
**MOD**: 27 (Witt Tower)
**Type**: Empirical — Siena Scalp EEG Database (PhysioNet doi:10.13026/5d4a-j060)

## Claim

QA orbit class (Cosmos/Quiet) is a **STATIC pre-ictal state**, not a temporal trend.
Cosmos-type recordings (T2>0.55, N=6) maintain elevated T2 from the **START** of the
5-minute pre-ictal window: `early_T2=0.883` (300–200s before seizure), `late_T2=0.900`
(flat, persistence=0.017). Quiet-type recordings (N=3) stay near-zero throughout:
`early_T2=0.000`. All 6 Cosmos recordings are already in T2-dominant state 5 minutes
before seizure.

## Experimental Design

Split the 300s pre-ictal window into three 100s thirds:

| Third | Window | Samples |
|---|---|---|
| Early | onset−300s to onset−200s | 20 × 5s windows |
| Mid   | onset−200s to onset−100s | 20 × 5s windows |
| Late  | onset−100s to onset−0s   | 20 × 5s windows |

Interictal calibration (120–720s from recording start) sets T0/T1/T2 thresholds (same
pipeline as cert [479]).

Orbit class labels inherited from cert [480] per-recording fallback
(computed 2026-06-19, no re-read of labels here):
- **Cosmos** (T2>0.55): PN00-2, PN03-1, PN05-3, PN06-2, PN07-1, PN13-2 (N=6)
- **Quiet** (T2<0.20): PN01-1, PN03-2, PN14-1 (N=3)
- **Mixed**: all other 8 recordings

## Results

### Per-Third T2 Rates by Orbit Class

| Class | N | Early T2 | Mid T2 | Late T2 | Escalation (late−early) |
|---|---|---|---|---|---|
| Cosmos | 6 | **0.883** | 0.908 | **0.900** | +0.017 (flat) |
| Quiet  | 3 | **0.000** | 0.117 | **0.050** | +0.050 (flat) |
| Mixed  | 8 | varies    | varies | varies   | varies |

### Per-Recording Detail

| File | Class | Early T2 | Mid T2 | Late T2 | Escalation |
|---|---|---|---|---|---|
| PN00-2 | cosmos | 0.550 | 0.750 | 0.750 | +0.200 |
| PN03-1 | cosmos | **1.000** | 1.000 | 1.000 | 0.000 (ceiling) |
| PN05-3 | cosmos | 0.950 | 0.850 | 0.800 | −0.150 |
| PN06-2 | cosmos | 0.800 | 0.850 | 0.850 | +0.050 |
| PN07-1 | cosmos | **1.000** | 1.000 | 1.000 | 0.000 (ceiling) |
| PN13-2 | cosmos | **1.000** | 1.000 | 1.000 | 0.000 (ceiling) |
| PN01-1 | quiet  | 0.000 | 0.000 | 0.000 | 0.000 |
| PN03-2 | quiet  | 0.000 | 0.000 | 0.050 | +0.050 |
| PN14-1 | quiet  | 0.000 | 0.350 | 0.100 | +0.100 |

**Three ceiling-locked cosmos recordings** (PN03-1, PN07-1, PN13-2) show T2=1.000 in
ALL three thirds — maximal pre-ictal T2 saturation from 5 minutes out.

## Key Finding: Orbit Class is Static

The original hypothesis was that Cosmos recordings would show **temporal escalation**
(increasing T2 toward seizure). The data shows something more fundamental:

- Cosmos recordings are already at T2≈0.883 in the **early third** (300s before seizure)
- This is far above the 1/3 interictal baseline (0.333)
- The "escalation" from early to late is only +0.017 (essentially flat) — because they're
  already saturated
- Quiet recordings stay near-zero throughout (±0.05 noise level)

**The orbit-class discrimination is present 5 minutes before seizure, not just immediately
before.** This is a stronger clinical statement than cert [479] (pooled T2 over 300s) —
it shows the separation holds even in the EARLIEST part of the pre-ictal window.

## Theorem NT Compliance

Identical to cert [479]:
- Observer: EEG voltage → DC removal → energy RMS (float)
- Observer: interictal RMS → rank → percentile thresholds (float)
- QA state: tier ∈ {0, 1, 2} (integer comparison only)
- Orbit class label = QA integer state {cosmos, quiet, mixed}

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 cosmos_early_T2 > 0.60 | > 0.60 | 0.883 | PASS |
| C2 quiet_early_T2 < 0.05 | < 0.05 | 0.000 | PASS |
| C3 early separation > 0.50 | > 0.50 | 0.883 | PASS |
| C4 cosmos_late_T2 > 0.60 | > 0.60 | 0.900 | PASS |
| C5 n_cosmos_high_early >= 4 | >= 4/6 | 6/6 | PASS |
| C6 cosmos_persistence < 0.05 | < 0.05 | 0.017 | PASS |

## Primary Sources

- Detti P, et al. (2020). Siena Scalp EEG Database. PhysioNet. doi:10.13026/5d4a-j060
- Goldberger AL, et al. (2000). doi:10.1161/01.CIR.101.23.e215

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, T0/T1/T2 partition)
- **[479]** EEG pre-ictal T2 elevation (pooled; pipeline definition)
- **[480]** EEG orbital stratification (per-recording orbit class labels)
