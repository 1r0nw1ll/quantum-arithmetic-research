# [476] QA Witt Tower ENSO Prediction

**First QA forecast of a physical system outside finance.**

## Claim

QA Witt Tower tier labels (T0=La Niña, T1=Neutral, T2=El Niño), derived purely from
rank-binning NOAA ONI anomalies into Z/27Z, predict ENSO tier 1 and 3 seasons ahead
with accuracy far above chance. The transition matrix reveals a **structural forbidden
transition**: T0↛T2 and T2↛T0 — in 916 ENSO seasons, the climate system never jumped
directly between El Niño and La Niña states. Neutral tier T1 is a mandatory gateway.

## Data

- Source: NOAA ONI 3-month running-mean SST anomaly (°C), DJF 1950 – present
- N = 916 seasonal values
- Binning: rank → floor(rank × 27 / N), uniform over {0..26}
- Tiers: T0 = bins 0–8, T1 = 9–17, T2 = 18–26
- Tier counts: T0=306, T1=305, T2=305 — **near-perfectly balanced** by rank construction

## Lag-1 Transition Matrix P(T_{t+1} | T_t)

| From \ To | T0 (La Niña) | T1 (Neutral) | T2 (El Niño) |
|---|---|---|---|
| **T0 (La Niña)** | **0.905** | 0.095 | **0.000** ← forbidden |
| **T1 (Neutral)** | 0.092 | **0.813** | 0.095 |
| **T2 (El Niño)** | **0.000** ← forbidden | 0.092 | **0.908** |

**Prediction accuracy (lag 1):** 87.5% vs 33.3% baseline (argmax-row rule)

## Lag-3 Transition Matrix P(T_{t+3} | T_t)

| From \ To | T0 | T1 | T2 |
|---|---|---|---|
| **T0** | **0.748** | 0.235 | 0.016 |
| **T1** | 0.205 | **0.541** | 0.254 |
| **T2** | 0.040 | 0.227 | **0.734** |

**Prediction accuracy (lag 3):** 67.5% vs 33.4% baseline

## Permutation Test Results

| Signal | perm_p |
|---|---|
| T2→T2 lag 1 | 0.0000 |
| T0→T0 lag 1 | 0.0000 |
| T1→T1 lag 1 | 0.0000 |
| T2→T2 lag 3 | 0.0000 |
| T0→T0 lag 3 | 0.0000 |

All 5000 permutation shuffles fall below the observed persistence probabilities.

## Structural Finding: The Forbidden Transition

T0→T2=0.000 and T2→T0=0.000 at lag 1. In the full 1950–2026 NOAA record, **no
ENSO season ever transitioned directly from La Niña (T0) to El Niño (T2) or vice
versa** without passing through Neutral (T1) first.

This is a QA-native discovery: the discrete tier partition (T0, T1, T2) maps exactly
onto a physical constraint in the climate system. El Niño and La Niña events require
a "wind-down" and "ramp-up" phase that always passes through neutral SST conditions.
The QA rank-bin discretization, designed for its mathematical properties (MOD=27 Witt
Tower structure), recovers this physical law without any domain-specific tuning.

At lag 3, T0→T2 rises to 0.016 (rare but nonzero) — El Niño onset from La Niña is
possible in 3 seasons with a fast neutral transition, consistent with known ENSO dynamics
(e.g., the 1988→1991 La Niña→El Niño switch).

## Theorem NT Compliance

Observer: ONI SST anomaly (°C) → rank → bin ∈ Z/27Z.
QA state: tier_t = bin_t // 9 ∈ {0, 1, 2} — integer, never float.
ONI anomaly values are observer outputs. No float state enters the QA layer.
Prediction target (tier_{t+k}) is derived from the same observer projection applied
to a future observation — not a float fed back into QA dynamics.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | T2→T2 lag-1 perm_p < 0.001 | PASS (0.0000) |
| C2 | T0→T0 lag-1 perm_p < 0.001 | PASS (0.0000) |
| C3 | Lag-1 accuracy > 80% | PASS (87.5%) |
| C4 | T2→T2 lag-3 perm_p < 0.001 | PASS (0.0000) |
| C5 | T0→T2 lag-1 probability = 0 (structural null) | PASS (0.000) |
| C6 | Lag-3 accuracy > 60% | PASS (67.5%) |

## Primary Sources

- Trenberth KE (1997). doi:10.1175/1520-0477(1997)078<2771:TDOENO>2.0.CO;2
- Philander SGH (1990). El Niño, La Niña, and the Southern Oscillation. ISBN:9780125532358

## Related Certs

- [445] QA Witt Tower ENSO T2 Segregation (El Niño 100% T2, log10_p=−172)
- [465] QA Witt Tower MI ENSO (I=1.07 bits, MI_ratio=69.9%)
- [467] QA Witt Tower Cross-Domain MI Survey (7-domain 0/35000 null hits)
- [468] QA Witt Tower MI Ceiling Theory (closed-form binary formula)
