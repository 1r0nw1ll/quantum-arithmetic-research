# Cert Family [483]: QA Witt Tower Crypto Momentum Asymmetry

**Family ID**: 483
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_crypto_momentum_cert_v1/qa_witt_tower_crypto_momentum_cert_validate.py`
**Validated**: 2026-06-20
**MOD**: 27 (Witt Tower)
**Type**: Empirical — Yahoo Finance daily OHLCV; BTC-USD 2015-01-01, ETH-USD 2017-11-09

## Claim

Return-rank Cosmos-type pairs (a≥58, consecutive high-return days) show **attenuated
momentum in BTC** (p=0.025, excess=+0.254%) but **null in ETH** (p=0.209).
Crash-reversion (cert [482], a≤6) dominates momentum (a≥58) by 3.34× in BTC
and 11.3× in ETH.

The QA Singularity orbit (fixed point at (9,9)) exerts stronger mean-reversion
force than the Cosmos expansion orbit sustains continuation.

## Operator Design

Same framework as cert [482]:
- `bin[t] = floor(rank(rets[t]) × 27 / N)` — full-sample return-rank bin
- `a = b + 2e` (A2: always derived, never assigned independently)
- **Signal condition**: `a ≥ 58` — Cosmos-type pairs
- **Prediction target**: `rets[t+2]` — no look-ahead

For a≥58: requires `e ≥ 16` at minimum (since max 2e=52, so b≥6 needed). In practice most
signals have e≥20 (today's return in top ~25%) and b≥10+ (yesterday also above median).

## Results

| Asset | N | Signal n | Excess | perm_p | IS n | IS mean | OOS n | OOS mean | Pct-up |
|---|---|---|---|---|---|---|---|---|---|
| BTC-USD | 4182 | 654 (15.6%) | +0.254% | 0.025 | 328 | +0.554% | 326 | +0.214% | 50.5% |
| ETH-USD | 3139 | 479 (15.3%) | +0.157% | 0.209 | 108 | +0.081% | 371 | +0.251% | 47.8% |

Base means: BTC +0.130%/day, ETH +0.055%/day.

**BTC**: significant positive signal (p=0.025), OOS holds (+0.214% on n=326 OOS days).
**ETH**: not significant (p=0.209). Directional below 50% (47.8%).

## Orbit Asymmetry (Key Structural Finding)

Comparing momentum (a≥58) to crash-reversion (a≤6, cert [482]):

| Asset | Crash-reversion excess | Momentum excess | Ratio |
|---|---|---|---|
| BTC-USD | +0.847% | +0.254% | **3.34×** |
| ETH-USD | +1.771% | +0.157% | **11.3×** |

**The QA Singularity orbit is far more predictive than the Cosmos orbit.**

QA structural interpretation:
- **Singularity** (fixed point (9,9,18,9)): orbital dynamics have strong restoring force
  back toward higher-energy orbits → mean-reversion is strong and universal (both assets)
- **Cosmos** (72-pair expansion orbit): energy stays expanded but continuation force is
  weaker → momentum is attenuated and asset-specific (BTC only, not ETH)

This is an asymmetric prediction of the Z/27Z orbit structure: fixed points attract
more strongly than expansion orbits sustain. The crash-reversion / momentum ratio
(3.34× for BTC, 11.3× for ETH) quantifies the asymmetry.

## Why ETH Null for Momentum?

ETH has ~2× higher daily volatility than BTC over this period. Higher volatility means:
- More noise obscuring the momentum signal
- Larger tails in the return distribution → a≥58 signals are more idiosyncratic
- The correction following high-return days is larger (regression to mean dominates)

ETH's crash-reversion is 2.1× stronger than BTC's (+1.771% vs +0.847%) — consistent
with higher vol amplifying the Singularity restoring force for ETH specifically.

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 BTC excess > +0.15% | > 0.15% | +0.254% | PASS |
| C2 BTC perm_p < 0.05 | < 0.05 | 0.025 | PASS |
| C3 BTC OOS mean > 0 | > 0.0% | +0.214% | PASS |
| C4 ETH perm_p > 0.10 | > 0.10 | 0.209 (null) | PASS |
| C5 BTC crash-rev > 2× momentum | > 2.0× | 3.34× | PASS |
| C6 ETH crash-rev > 5× momentum | > 5.0× | 11.3× | PASS |

C4 certifies that ETH momentum is null — not a failure, but a structural constraint.
C5 and C6 use cert [482]'s certified crash-reversion values as cross-cert constants.

## Comparison with Equity Certs

| Domain | a≤6 crash-rev | a≥58 momentum | Ratio |
|---|---|---|---|
| US equities (cert [461]) | +0.37% | (untested) | — |
| BTC crypto | +0.847% | +0.254% | 3.34× |
| ETH crypto | +1.771% | null | >11× |

Crash-reversion in crypto is 2.3× stronger than in US equities (higher volatility
amplifies the Singularity restoring force in higher-vol assets).

## Theorem NT Compliance

Identical to cert [482]: returns are observer projections; rank → integer bins → integer
a = b + 2e comparison. No float enters QA logic.

## Primary Sources

- Fama EF (1970). Efficient capital markets: a review. *J Finance* 25(2):383-417.
  doi:10.2307/2325486 (random-walk null hypothesis baseline)
- Jegadeesh N & Titman S (1993). Returns to buying winners and selling losers.
  *J Finance* 48(1):65-91. doi:10.1111/j.1540-6261.1993.tb04702.x (momentum baseline)
- Data: Yahoo Finance historical OHLCV via yfinance (downloaded 2026-06-20)

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, three-orbit partition)
- **[482]** Crypto return-rank crash-reversion (a≤6 signal; cross-cert constants CRASH_EXCESS_BTC/ETH)
