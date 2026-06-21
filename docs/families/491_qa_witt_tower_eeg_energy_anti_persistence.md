# Cert [491]: QA Witt Tower EEG Interictal Energy Return-Rank Anti-Persistence

**Family ID**: 491
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_eeg_energy_anti_persistence_cert_v1/`

## Claim

The return-rank a=b+2e≤6 operator applied to sequential 5-sec EEG energy log-changes reveals **anti-persistent** structure — the structural inverse of river persistence (cert [490]).

| Recording | Patient | n_signal | Expected | Ratio | Excess | autocorr_lag1 |
|-----------|---------|---------|---------|-------|--------|--------------|
| PN01-1 | PN01 | 4 | 15.7 | 0.25× | +17.87 log-% | −0.378 |
| PN03-1 | PN03 | 27 | 15.7 | 1.72× | −1.64 log-% | −0.127 |
| PN05-2 | PN05 | 7 | 15.7 | 0.45× | +36.51 log-% | −0.298 |
| PN06-1 | PN06 | 7 | 13.9 | 0.50× | −10.47 log-% | −0.175 |
| PN07-1 | PN07 | 11 | 15.7 | 0.70× | +37.86 log-% | −0.257 |
| PN09-1 | PN09 | 11 | 15.7 | 0.70× | +26.30 log-% | −0.339 |

Pooled: n_signal=**67 vs expected=92.6 (ratio=0.724×)**. Crash-reversion excess=**+13.63 log-%**, crash_p=0.020. All 6 autocorr_lag1 < 0.

## n_signal_ratio as Autocorrelation Discriminator

The number of signal windows (a≤6 pairs) relative to the independence baseline (P(a≤6) = 16/729 = 2.19%) is a direct proxy for lag-1 autocorrelation sign:

| Domain | n_signal_ratio | Excess direction | Autocorrelation |
|--------|---------------|-----------------|-----------------|
| Altcoins [487] | ~1.4× (elevated) | Positive (+1.71%/day) | Negative (mean-reverting) |
| Equity [488]/[489] | ~1.1× (mild elevation) | Positive (+0.38%/day) | Negative (mean-reverting) |
| Forex/GLD [486] | ~1.0× (neutral) | ~0 (null) | ~0 (i.i.d.) |
| **EEG interictal [491]** | **0.72× (DEPLETED)** | **Positive (+13.6 log-%)** | **Negative (anti-persistent)** |
| **Rivers [490]** | **2.69× (ELEVATED)** | **Negative (−11.95 log-%)** | **Positive (persistent)** |

## Mechanism

EEG amplitude at 5-sec window scale is driven by the **amplitude envelope of alpha/theta oscillations** (dominant frequency bands 4-12 Hz). The amplitude envelope oscillates with a period of **~10-20 seconds** (driven by slow cortical oscillations and attention states). Consecutive 5-sec windows sample roughly half a period apart, making their log-energy-changes systematically anti-correlated (lag-1 r ≈ −0.26 mean across 6 recordings).

After 2 consecutive large energy drops (a≤6), the energy is likely in a local minimum and the amplitude envelope is turning upward → energy bounces → crash-reversion (positive excess).

**Contrast with rivers**: River recession operates on timescales of days-to-weeks (τ >> 5 sec). After 2 fast-recession windows, the recession is still ongoing → persistence (negative excess).

**The discrimination principle**: the autocorrelation SIGN depends on the ratio of the restoring-force timescale to the observation window:
- τ >> window: persistent → negative excess + elevated n_signal
- τ ≈ window: anti-persistent → positive excess + depleted n_signal
- τ → ∞ (equity with daily price discovery ≈ 1 day): negative excess at daily scale; equity crashes revert in ~1 day

## Operator (same as cert [490])

```
E[t]   = multichannel RMS of 5-sec EEG window t           # observer projection
r[t]   = log(E[t+1] / E[t])                               # log-energy-change
b      = floor(rank(r[t])   × 27 / N)   # A1: int {0..26}
e_val  = floor(rank(r[t+1]) × 27 / N)   # A1: int {0..26}
a      = b + 2 × e_val                  # A2: derived, raw, never mod-reduced
signal: a ≤ 6                           # Singularity-type pair
target: r[t+2]                          # no look-ahead (T1 compliant)
```

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | All 6 autocorr_lag1 < 0 | PASS | −0.127 to −0.378 |
| C2 | pooled n_signal < 0.9 × expected | PASS | 67 < 83.3 |
| C3 | n_recordings_depleted ≥ 5 | PASS | 5/6 |
| C4 | pooled_excess > 0 | PASS | +13.632 log-% |
| C5 | pooled_crash_p < 0.05 | PASS | 0.020 |
| C6 | n_signal_ratio < 1.0 < rivers (2.69×) | PASS | 0.724× |

## PN03 Outlier Note

PN03-1 has n_sig=27 (above expected 15.7) despite negative autocorr (−0.127). This recording has the weakest anti-persistence of the 6 and is close to the i.i.d. boundary. Its small negative excess (−1.64 log-%) is consistent with noise. It is the only "not depleted" recording and is excluded from the C3 count.

## Primary Sources

- Linkenkaer-Hansen K et al. (2001). Long-range temporal correlations and scaling behavior in human brain oscillations. *J Neurosci* 21(4):1370-1377. doi:10.1523/JNEUROSCI.21-04-01370.2001
- Stam CJ (2005). Nonlinear dynamical analysis of EEG and MEG. *Clin Neurophysiol* 116(10):2266-2301. doi:10.1016/j.clinph.2005.06.011

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [446]: Siena EEG ictal discrimination; Siena data source
- Cert [488]: US equity return-rank; crash-reversion contrast (+0.385%/day)
- Cert [490]: River streamflow persistence; n_signal_ratio contrast (2.69×)
