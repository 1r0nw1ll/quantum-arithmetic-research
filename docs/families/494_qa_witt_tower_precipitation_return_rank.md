# Cert [494]: QA Witt Tower Daily Precipitation Return-Rank Persistence

**Family ID**: 494
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_precipitation_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 operator applied to monthly-deseasonalised log1p(precipitation) anomalies reveals **positive persistence (wet/dry spells) sitting between rivers [490] and temperature [492]** in the discrimination ladder — 3.05× expected, despite Pearson autocorrelation of only 0.20–0.44.

| Station | n_days | n_signal | Ratio | autocorr_log1p | Excess (log-units) | pers_p |
|---------|--------|---------|-------|---------------|-------------------|--------|
| Chicago, IL | 9132 | 594 | 2.96× | 0.225 | −0.276 | 0.0 |
| Minneapolis, MN | 9132 | 571 | 2.85× | 0.200 | −0.246 | 0.0 |
| Seattle, WA | 9132 | 649 | 3.24× | 0.408 | −0.671 | 0.0 |
| Miami, FL | 9132 | 629 | 3.14× | 0.437 | −0.572 | 0.0 |

4/4 negative excess. Pooled: n_sig = **2443** (**3.05× expected** 801.6), excess = **−0.44 log-units**. All pers_p = 0.0.

## Key Surprise: Rank Clustering Exceeds Pearson Autocorr Prediction

Precipitation Pearson autocorr (0.20–0.44) is much lower than temperature (0.69–0.77), yet the n_signal_ratio (3.05×) is comparable:

| Domain | Pearson autocorr | n_signal_ratio |
|--------|-----------------|----------------|
| Temperature [492] | 0.69–0.77 | 3.40× |
| **Precipitation [494]** | **0.20–0.44** | **3.05×** |
| Rivers [490] | n/a (log-change) | 2.69× |

**Why**: Precipitation is heavy-tailed (many zeros). The log1p transformation reduces skewness, and the rank-bin operator captures **rank autocorrelation (Spearman)**, not Pearson. For heavy-tailed distributions, tail dependence (extreme event clustering) gives Spearman autocorrelation >> Pearson autocorrelation at the same persistence level. Consecutive dry days cluster strongly because weather systems (synoptic τ~3-7 days) determine dry spells.

## Discrimination Ladder Position

| Domain | Cert | n_sig_ratio | Mechanism |
|--------|------|------------|-----------|
| EEG interictal | [491] | 0.72× anti-persistent | Amplitude envelope modulation |
| GLD/forex | [486] | ~1.0× null | No dominant microstructure |
| Equity/crypto | [488]/[482] | 1.1–1.4× | Mean-reverting microstructure |
| Rivers | [490] | 2.69× | Maillet recession τ~days-weeks |
| **Precipitation** | **[494]** | **3.05×** | **Wet/dry spells τ~3-7 days** |
| Temperature | [492] | 3.40× | Synoptic blocking τ~3-7 days |
| Ocean SST | [493] | 4.43× | Thermal inertia τ~months |

Precipitation and temperature share the same synoptic forcing (τ~3-7 days) but precipitation's heavier tail makes its absolute rank clustering slightly lower than temperature's anomaly persistence.

## Operator

- **Variable**: `precipitation_sum` (mm/day) from Open-Meteo ERA5
- **Transform**: `log_val[t] = log(1 + precip[t])` (reduces skewness; log1p preserves zeros)
- **Anomaly**: `anom[t] = log_val[t] − monthly_mean_log[month(t)]`
- **Bins**: `b = floor(rank(anom[t]) × 27 / N)` (∈ {0..26})
- **Signal**: `a = b + 2×e_val ≤ 6` (A2 derived, raw; both days low-precipitation anomaly)
- **Target**: `anom[t+2]` (no look-ahead)
- **Period**: 2000-01-01 to 2024-12-31 (~9132 days × 4 stations)

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | All 4 stations autocorr_log1p > 0 | PASS | 0.200–0.437 |
| C2 | Pooled excess < 0 | PASS | −0.44 log-units |
| C3 | n_negative == 4/4 | PASS | 4 |
| C4 | All 4 pers_p < 0.001 | PASS | 0.0 all |
| C5 | Pooled ratio > 2.5 | PASS | 3.05 |
| C6 | Pooled ratio > certified river ratio [490] | PASS | 3.05 > 2.69 |

## Primary Sources

- Trenberth KE (1999). Conceptual framework for changes of extremes of the hydrological cycle with climate change. *Climatic Change* 42:327-339. doi:10.1023/A:1005488920935
- Zolina O et al. (2013). Changes in the duration of European wet and dry spells during the last 60 years. *Journal of Climate* 26(6):2022-2047. doi:10.1175/JCLI-D-12-00552.1

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [490]: River persistence (2.69× reference; precipitation exceeds by 0.36×)
- Cert [492]: Temperature persistence (3.40× reference; precipitation sits just below)
- Cert [493]: SST persistence (4.43× reference)
- Cert [482]: Return-rank operator definition
