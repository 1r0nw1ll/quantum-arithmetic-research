# [516] QA Witt Tower AR(1)-Baseline Reranking

**Cert slug**: `qa_witt_tower_ar1_baseline_reranking_cert_v1`
**Family ID**: 516
**Derived**: 2026-07-04
**Companion to**: [490], [491], [492], [493], [494], [495] (audits their published
discrimination-ladder ranking without retracting their underlying data)

## Claim

The "discrimination ladder" documented across certs [490]-[495] (rivers,
precipitation, temperature, ocean SST, EEG interictal, 1-min FX) ranks domains
by raw `n_signal_ratio`. That ranking conflates two different things:

1. How strongly the domain is **already** autocorrelated at lag 1 — a
   well-documented, pre-existing fact for each domain (ocean thermal inertia,
   synoptic blocking, etc.), cited in the certs' own primary sources.
2. How much the QA rank-bin operator reveals **beyond** what that known
   autocorrelation alone predicts.

Re-ranking by (2) instead of raw ratio **nearly inverts** the published
ladder. The two domains presented as "STRONGEST PERSISTENCE" — ocean SST and
temperature — show close to zero genuine excess beyond a plain
AR(1)/correlated-Gaussian null at the *same* reported lag-1 autocorrelation.
Their raw ratios are essentially a restatement of already-known
autocorrelation magnitudes, not a new QA-specific finding. The domains ranked
lower in the original ladder — EEG interictal, precipitation, rivers, FX —
show substantial genuine excess: these are the results *not* already
explained by textbook correlation.

This does **not** retract [490]-[495]'s underlying data or `n_signal_ratio`
computations, which are independently re-verified here as correct. It
corrects the interpretive framing: "highest raw ratio" was implicitly read as
"most novel finding," when the two are only the same thing for a domain with
near-zero baseline autocorrelation.

## Method

For each domain, simulate a plain AR(1) Gaussian process `x[t] = rho*x[t-1] +
sqrt(1-rho²)*eps[t]` at the domain's own reported lag-1 autocorrelation `rho`
(pooled across that domain's stations/patients/pairs, taken directly from the
certified fallback data in [490]-[495]). Apply the *identical* rank-bin
operator (`b,e ∈ {0..26}` via full-sample rank, `a=b+2e≤6`) the real certs
use, and measure the ratio of the AR(1) process's own signal count to its
theoretical independence baseline (`16/729` per triplet). `excess_ratio =
observed_ratio / AR1_predicted_ratio`.

## Empirical Record (2026-07-04, pure-Python Monte Carlo, seed=0, n=8000, 40 trials)

| Domain | ρ | AR(1)-predicted | Observed | Excess ratio |
|---|---|---|---|---|
| SST [493] | +0.942 | 4.23 | 4.432 | **1.05** |
| Temperature [492] | +0.728 | 3.36 | 3.400 | **1.01** |
| Rivers [490] | +0.310 | 1.88 | 2.689 | 1.43 |
| Precipitation [494] | +0.318 | 1.91 | 3.048 | 1.60 |
| EEG interictal [491] | -0.262 | 0.44 | 0.725 | 1.64 |
| FX 1-min [495] | -0.127 | 0.71 | 1.009 | 1.42 |

**Reranked by genuine excess** (descending): EEG (1.64) > Precipitation (1.60)
> Rivers (1.43) > FX (1.42) > SST (1.05) > Temperature (1.01) — nearly the
exact inverse of the original raw-ratio ladder (SST 4.43 > Temp 3.40 > Precip
3.05 > Rivers 2.69 > FX 1.009 > EEG 0.72).

Rivers' `rho` was live-fetched from USGS NWIS (log-return lag-1 autocorrelation,
4 gauges: Potomac 0.385, Hudson 0.203, Missouri 0.306, Eel 0.345, pooled mean
0.310) since cert [490]'s fallback record does not store an autocorr field —
reproducible via `reproduce_ar1_baseline.py --fetch-rivers`.

## Checks (ARB = AR1-Reranking Baseline)

| Check | Claim |
|---|---|
| ARB_SIM_NULL | AR(1) simulation at rho=0 gives ratio ≈ 1.0 (simulator sanity check) |
| ARB_SST_NEAR_NULL | SST excess_ratio in [0.9, 1.15] |
| ARB_TEMP_NEAR_NULL | Temperature excess_ratio in [0.9, 1.15] |
| ARB_EEG_EXCESS | EEG excess_ratio > 1.3 |
| ARB_PRECIP_EXCESS | Precipitation excess_ratio > 1.3 |
| ARB_RIVERS_EXCESS | Rivers excess_ratio > 1.3 |
| ARB_FX_EXCESS | FX excess_ratio > 1.3 |
| ARB_RERANK_INVERTS | reranking by excess_ratio differs from raw-ratio ranking |
| ARB_WITNESS | recorded rho values match the historical live-fetch/certified record |

The validator's gating checks are fast, deterministic lookups against the
recorded empirical values (not a re-run of the Monte Carlo on every
self-test). `reproduce_ar1_baseline.py` regenerates the simulation fresh —
verified to reproduce the cited numbers exactly before this cert was
committed.

## What This Means for the Original Ladder

The underlying data and computations in [490]-[495] are sound — this is not
an E8-style bug. It's a reminder that "signal beats an assumed-independent
null" and "signal beats what you'd already expect from known correlation" are
different claims, and a ladder built on the former can rank domains in the
opposite order of the latter. SST and temperature remain real, correctly
computed results — they're just not surprising once you already know oceans
have more thermal inertia than the atmosphere.

## Primary Sources

- Rayner, N.A. et al. (2003). "Global analyses of sea surface temperature."
  J. Geophys. Res. 108(D14). DOI 10.1029/2002JD002670.
- Namias, J. (1952). "Long range weather forecasting." AMS.
- This cert's own `reproduce_ar1_baseline.py` (pure-Python Monte Carlo).

## Parents

None — standalone audit cert. Companions: [490], [491], [492], [493], [494], [495].
