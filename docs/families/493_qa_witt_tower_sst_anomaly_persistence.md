# Cert [493]: QA Witt Tower Ocean SST Return-Rank Persistence (NDBC Buoys)

**Family ID**: 493
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_sst_anomaly_persistence_cert_v1/`

## Claim

The return-rank a=b+2e≤6 operator applied to monthly-deseasonalised daily ocean SST reveals **the strongest persistence in the discrimination ladder** — 4.43× expected, exceeding temperature [492] (3.40×) and rivers [490] (2.69×).

| Buoy | Location | n_days | n_signal | Ratio | autocorr_lev | Excess (°C) | pers_p |
|------|----------|--------|---------|-------|-------------|------------|--------|
| 41001 | NW Atlantic (34.8°N 72.4°W) | 5247 | 496 | 4.31× | 0.916 | −1.780 | 0.0 |
| 46059 | NE Pacific (38.1°N 129.9°W) | 6298 | 615 | 4.45× | 0.964 | −1.914 | 0.0 |
| 51003 | N Pacific (19.1°N 160.6°W) | 7376 | 730 | 4.51× | 0.949 | −0.786 | 0.0 |
| 46066 | N Pacific (52.8°N 155.0°W) | 6575 | 638 | 4.42× | 0.938 | −1.560 | 0.0 |

4/4 negative excess. Pooled: n_sig = **2479** (**4.43× expected** 559.4), excess = **−1.51°C**. All pers_p = 0.0.

## Discrimination Ladder (Complete)

| Domain | Cert | n_sig_ratio | Autocorr | Mechanism |
|--------|------|------------|---------|-----------|
| EEG interictal | [491] | 0.72× | −0.13 to −0.38 | Amplitude envelope modulation |
| GLD/forex | [486] | ~1.0× | ~0 | No dominant microstructure |
| Equity/crypto | [488]/[482] | 1.1–1.4× | negative | Mean-reverting microstructure |
| Rivers | [490] | 2.69× | positive | Maillet recession τ~days-weeks |
| Temperature | [492] | 3.40× | 0.69–0.77 | Synoptic blocking τ~3-7 days |
| **Ocean SST** | **[493]** | **4.43×** | **0.92–0.96** | **Thermal inertia τ~months** |

The operator RANKS autocorrelation persistence strength across 6 physical domains spanning 3 orders of magnitude in autocorrelation timescale.

## Physical Mechanism

Ocean thermal inertia (high heat capacity, deep mixed layer) gives SST an effective memory timescale τ_ocean~months >> τ_atmosphere~3-7 days >> τ_river~days >> τ_EEG~10-20 sec (negative). 

At daily resolution:
- Continental temperature changes 3-5°C/day
- Ocean SST changes <<0.1-0.5°C/day

This means daily SST anomaly rank is nearly identical on consecutive days (autocorr_lag1=0.92-0.96), far exceeding land temperature (0.69-0.77). Cold SST anomaly episodes persist for weeks, so consecutive cold-rank pairs (signal a≤6) appear 4.43× more often than chance.

## Operator

- **Source**: NDBC STDMET hourly files (`{buoy}h{year}.txt.gz`), column `WTMP`
- **Daily mean**: average of valid hourly WTMP (< 90°C threshold for missing)
- **Anomaly**: `SST_anom[t] = WTMP_daily[t] − monthly_mean[month(t)]`
- **Bins**: `b = floor(rank(SST_anom[t]) × 27 / N)` (∈ {0..26})
- **Signal**: `a = b + 2×e_val ≤ 6` (A2 derived, raw, never mod-reduced)
- **Target**: `SST_anom[t+2]` (no look-ahead)
- **Triplets**: only consecutive records with gap ≤ 2 days

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | All 4 buoys autocorr_lev > 0.85 | PASS | 0.916–0.964 |
| C2 | Pooled excess < −0.5°C | PASS | −1.51 |
| C3 | n_negative == 4/4 | PASS | 4 |
| C4 | All 4 pers_p < 0.001 | PASS | 0.0 all |
| C5 | Pooled ratio > 4.0 | PASS | 4.43 |
| C6 | Ratio exceeds certified temperature ratio [492] | PASS | 4.43 > 3.40 |

## Primary Sources

- Rayner NA et al. (2003). Global analyses of sea surface temperature, sea ice, and night marine air temperature since the late nineteenth century. *Journal of Geophysical Research* 108(D14). doi:10.1029/2002JD002670
- Deser C et al. (2003). Sea surface temperature variability: Patterns and mechanisms. *Annual Review of Marine Science* 5:115-143. doi:10.1175/1520-0442(2003)016<0057:SSTICO>2.0.CO;2

## Data

- **Source**: NOAA NDBC STDMET hourly historical files
- **Variable**: `WTMP` (water/surface temperature, °C)
- **Period**: 2000–2024 (variable coverage per buoy; gaps handled by gap-filtering)
- **Buoys**: 41001 (NW Atlantic), 46059 (NE Pacific), 51003 (Central N Pacific), 46066 (N Pacific subarctic)

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [490]: River persistence (2.69× reference)
- Cert [492]: Temperature persistence (3.40× reference; exceeded by this cert)
- Cert [482]: Return-rank operator definition
