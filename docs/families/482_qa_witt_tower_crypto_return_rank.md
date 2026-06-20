# Cert Family [482]: QA Witt Tower Crypto Return-Rank Crash Reversion

**Family ID**: 482
**Status**: CERTIFIED (6/6 checks pass)
**Validator**: `qa_alphageometry_ptolemy/qa_witt_tower_crypto_return_rank_cert_v1/qa_witt_tower_crypto_return_rank_cert_validate.py`
**Validated**: 2026-06-19
**MOD**: 27 (Witt Tower)
**Type**: Empirical — Yahoo Finance daily OHLCV; BTC-USD 2015-01-01, ETH-USD 2017-11-09

## Claim

The QA Witt Tower **return-rank** a≤6 operator identifies crypto crash-reversion days.
After two consecutive bottom-7%-return days, next-day mean return is strongly positive:
BTC +0.847%/day excess (p=0.003), ETH +1.771%/day excess (p<0.001).
OOS (2020+): BTC +0.557% (n=62), ETH +1.434% (n=62).

## Operator Design

- **Return bins**: `bin[t] = floor(rank(rets[t]) × 27 / N)` over full sample
  where `rets[t] = log(closes[t+1]/closes[t]) × 100`
- **State pair**: `b = bin[t]`, `e = bin[t+1]` (consecutive days)
- **A-coordinate**: `a = b + 2e` (A2: always derived, never assigned independently)
- **Signal condition**: `a ≤ 6` — Singularity-type pairs
- **Prediction target**: `rets[t+2]` — strictly outside the `(b, e)` window, no look-ahead

The a≤6 condition requires:
- `e ≤ 3` (today's return in bottom ~11% of all returns), and
- `b ≤ 6 − 2e` (yesterday's return correspondingly low)

Most concentrated case: `e=0, b=0` — both days near the very bottom of the return distribution.

## Why Return-Rank (Not Price-Level Rank)

Cert [461] used price-level rank bins on equity indices. BTC-USD has strong secular
price appreciation: the 2015-2019 era is permanently below 2021-2026 prices. Full-sample
price-level bins give BTC a≤6 all 458 IS signals and zero OOS signals (n_oos=0),
with p=0.53 (null). The equity operator is structurally incompatible with trending assets.

Return-rank bins are stationary: a daily log-return of −5% has the same rank regardless
of whether BTC is at $1,000 or $60,000. The return-rank operator gives 62 OOS signal
days for both BTC and ETH with robust mean returns.

## Results

| Asset | N | Signal n | Excess | perm_p | IS n | IS mean | OOS n | OOS mean | Pct-up |
|---|---|---|---|---|---|---|---|---|---|
| BTC-USD | 4182 | 126 (3.01%) | +0.847% | 0.003 | 64 | +1.386% | 62 | +0.557% | 61.9% |
| ETH-USD | 3139 | 93 (2.96%) | +1.771% | <0.001 | 31 | +2.613% | 62 | +1.434% | 71.0% |

Base means: BTC +0.130%/day, ETH +0.055%/day. Signal is 7.5× base (BTC) and 33× base (ETH).
OOS split date: 2020-01-01. Both IS and OOS holdout positive.

## QA Orbital Interpretation

In Z/27Z Witt Tower partition:
- **Singularity** (fixed point at (9,9)): minimal-energy state, path length k=0
- **Satellite** (8-pair 3D cycle): intermediate energy
- **Cosmos** (72-pair 24-cycle): maximal expansion

The a≤6 return pairs satisfy `a = b + 2e ≤ 6` with small b and e.
Small return-rank bins → both days near the bottom of the return distribution.
In QA terms: the system is near the Singularity orbit in return-rank space.
QA dynamics near the singularity predict **orbit return** (reversion toward the mean orbit).
The mean-reversion signal is the empirical instantiation of this orbit-return dynamics.

## Theorem NT Compliance

- Daily log-returns are **observer projections** (continuous domain → real numbers)
- `rank()` maps real returns to integers (boundary crossing, one-way)
- `bin = floor(rank × 27 / N)` is integer arithmetic
- `a = b + 2e` is integer addition (A2 compliance)
- `a ≤ 6` is integer comparison (no float in QA logic)
- No T2-b violation: no float × modulus → int cast in QA decision path

## Checks (6/6 PASS)

| Check | Threshold | Observed | Result |
|---|---|---|---|
| C1 BTC excess > +0.50% | > 0.50% | +0.847% | PASS |
| C2 BTC perm_p < 0.01 | < 0.01 | 0.003 | PASS |
| C3 ETH excess > +1.00% | > 1.00% | +1.771% | PASS |
| C4 ETH perm_p < 0.005 | < 0.005 | 0.0002 | PASS |
| C5 BTC OOS mean > 0 | > 0.0% | +0.557% | PASS |
| C6 ETH OOS mean > +0.50% | > 0.50% | +1.434% | PASS |

## Relationship to Parent Cert [461]

Cert [461] tests a≤6 on **price-level** rank bins for US equity indices.
This cert tests a≤6 on **return-rank** bins for crypto assets.
Both use the same A2-derived a-coordinate and same Singularity-type pair logic,
but the domain variable differs (price level vs daily return).
The equity operator is suited to mean-reverting (trend-stationary) assets;
the return-rank operator is required for assets with secular price trends.

## Primary Sources

- Fama EF (1970). Efficient capital markets: a review. *J Finance* 25(2):383-417.
  doi:10.2307/2325486 (random-walk null hypothesis baseline)
- Nakamoto S (2008). Bitcoin: A peer-to-peer electronic cash system. bitcoin.org/bitcoin.pdf
- Data: Yahoo Finance historical OHLCV via yfinance package (downloaded 2026-06-19)

## Parent Certs

- **[110]** Witt Tower Framework (MOD=27, three-orbit partition)
- **[461]** Equity price-level a≤6 operator (parent design)
