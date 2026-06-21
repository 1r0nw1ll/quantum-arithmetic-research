# Cert [492]: QA Witt Tower Daily Temperature Anomaly Return-Rank Persistence

**Family ID**: 492
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_temperature_persistence_cert_v1/`

## Claim

The return-rank a=b+2e≤6 operator applied to monthly-deseasonalised daily temperature **anomaly levels** reveals **strong persistence in atmospheric climate** — stronger than rivers [490] (3.40× vs 2.69× expected):

| Station | Climate Zone | n_signal | Expected | autocorr_lev | Excess (°C) | pers_p |
|---------|-------------|---------|---------|-------------|------------|--------|
| Chicago, IL | Continental | 674 | 208.4 | 0.704 | −6.498 | 0.0 |
| Minneapolis, MN | Cold Continental | 713 | 208.4 | 0.752 | −7.730 | 0.0 |
| Seattle, WA | Maritime | 757 | 208.4 | 0.766 | −4.138 | 0.0 |
| Miami, FL | Subtropical | 690 | 208.4 | 0.691 | −2.462 | 0.0 |

4/4 positive autocorrelation. Pooled: n_sig = **2834** (**3.40× expected** 833.6), excess = **−5.195°C**. All pers_p = 0.0 (0/5000 null shuffles more extreme). This exceeds the certified river ratio of 2.69× (cert [490]).

## Operator Design

**Key insight**: Temperature persistence lives in the anomaly LEVEL, not the log-change.
- **Log(T_K[t+1]/T_K[t])** has autocorr ≈ 0 — temperature changes are approximately i.i.d.
- **T_anom[t] = TMAX[t] − monthly_mean[month(t)]** has autocorr 0.69–0.77 — anomaly levels are strongly persistent

Monthly deseasonalisation removes the annual cycle, leaving the synoptic-scale persistence signal.

**Operator**:
- b = floor(rank(T_anom[t]) × 27 / N)  (bin ∈ {0..26})
- e_val = floor(rank(T_anom[t+1]) × 27 / N)
- a = b + 2×e_val  (A2: derived, raw, never mod-reduced)
- Signal: a ≤ 6  (both consecutive days cold anomaly — bottom-left of rank space)
- Target: T_anom[t+2]  (next-day anomaly, no look-ahead)

## n_signal_ratio Discrimination Ladder

The operator cleanly separates autocorrelation regimes across three physical domains:

| Domain | Cert | Mechanism | n_signal_ratio | Autocorr sign |
|--------|------|-----------|----------------|---------------|
| EEG interictal | [491] | Amplitude envelope modulation | 0.724× | NEGATIVE (anti-persistent) |
| Independent baseline | — | i.i.d. shuffles | 1.000× | ≈ 0 |
| Rivers | [490] | Maillet exponential recession | 2.69× | POSITIVE |
| **Atmosphere** | **[492]** | **Synoptic blocking/persistence** | **3.40×** | **POSITIVE (strongest)** |

The operator does not just detect persistence — it **ranks** the strength of persistence across physical systems.

## Physical Mechanism

Synoptic weather patterns (high/low pressure systems) persist on timescales τ ≈ 3–7 days >> 1 day. A cold anomaly day is highly likely to be followed by another cold anomaly day because the blocking pattern has not dissipated. This is the **Namias blocking** phenomenon (1952): once established, synoptic ridges/troughs persist for days to weeks.

The signal selects days where both t and t+1 are cold-anomaly days (both in the bottom-left rank bin). The next-day target is systematically below the mean (negative excess), confirming cold anomalies cluster.

**Why stronger than rivers**: Synoptic patterns (τ ~ 3–7 days) have longer decorrelation times than river recession (τ ~ days). More persistent autocorrelation → more extreme n_signal clustering → higher ratio.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | All 4 stations autocorr_lev > 0 | PASS | 0.691–0.766 |
| C2 | Pooled excess < −1.0°C | PASS | −5.195 |
| C3 | n_negative == 4/4 | PASS | 4 |
| C4 | All stations pers_p < 0.001 | PASS | 0.0 all |
| C5 | Pooled n_signal_ratio > 3.0 | PASS | 3.40 |
| C6 | Ratio exceeds certified river ratio [490] | PASS | 3.40 > 2.69 |

## Data

- **Source**: Open-Meteo ERA5 historical archive (no API key required)
- **Variable**: `temperature_2m_max` (daily maximum temperature)
- **Period**: 2000-01-01 to 2025-12-31 (~9497 days per station)
- **Climate zones**: Chicago (41.85, −87.65), Minneapolis (44.88, −93.22), Seattle (47.61, −122.33), Miami (25.77, −80.19)

## Primary Sources

- Namias J (1952). The annual course of month-to-month persistence in climatic anomalies. *Bulletin of the American Meteorological Society* 33(7):279–285. doi:10.1175/1520-0477-33.7.279
- Wallace JM & Gutzler DS (1981). Teleconnections in the geopotential height field during the Northern Hemisphere winter. *Monthly Weather Review* 109(4):784–812. doi:10.1175/1520-0493(1981)109<0784:TITGHF>2.0.CO;2

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [490]: River streamflow return-rank (persistence reference, 2.69×)
- Cert [491]: EEG interictal energy return-rank (anti-persistence contrast, 0.724×)
- Cert [482]: BTC/ETH return-rank (operator definition)
