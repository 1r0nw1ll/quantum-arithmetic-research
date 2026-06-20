# Cert [486]: QA Witt Tower Cross-Asset Return-Rank Crash Reversion Scope

**Family ID**: 486
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_cross_asset_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 crash-reversion operator (certified for BTC/ETH in cert [482]) is **CRYPTO-SPECIFIC**. Tested on four non-crypto assets spanning different asset classes and volatility levels:

| Asset | Class | Vol/day | Excess | perm_p | Verdict |
|-------|-------|---------|--------|--------|---------|
| GLD | Gold ETF | 1.15% | +0.075% | 0.216 | NULL |
| EURUSD | Forex | 0.69% | +0.057% | 0.148 | NULL |
| GBPUSD | Forex | 0.59% | -0.039% | 0.790 | NULL |
| USO | Oil ETF | 2.35% | -0.215% | 0.873 | NULL |

**Key falsification**: USO (oil ETF, highest non-crypto daily vol at 2.35%) shows NEGATIVE excess (−0.215%), directly falsifying the hypothesis that crash-reversion scales with volatility.

**Ratio**: BTC excess (+0.847%, cert [482]) is 11.3× the highest non-crypto excess (GLD +0.075%).

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | GLD perm_p > 0.05 (NULL) | PASS | 0.216 |
| C2 | EURUSD perm_p > 0.05 (NULL) | PASS | 0.148 |
| C3 | All 4 non-crypto assets NULL | PASS | 4/4 |
| C4 | max non-crypto excess < 0.15% | PASS | +0.075% |
| C5 | BTC excess ratio > 5× max non-crypto | PASS | 11.3× |
| C6 | USO excess < GLD excess (vol-scaling falsified) | PASS | −0.215% < +0.075% |

## QA Structure

- **Operator**: return-rank bins `b=floor(rank(rets[t])×27/N)`, `e=floor(rank(rets[t+1])×27/N)`
- **Signal**: `a = b + 2e ≤ 6` (A2 derived, raw — never mod-reduced)
- **Target**: `rets[t+2]` (no look-ahead)
- **Permutation test**: N=5000 shuffles, seed=42
- **Theorem NT**: daily log-returns are observer projections; rank→bin crossing is the QA boundary

**QA structural interpretation**: Crash-reversion under Singularity-type orbit (a≤6) requires mean-reverting microstructure (bid-ask bounce, panic-sell/buyer-of-last-resort dynamics at the daily scale). Crypto markets have this structure. Commodities (USO/oil) tend toward supply-shock momentum after crash days. Gold and forex have insufficient volatility for the a≤6 operator to select economically significant events.

## Primary Sources

- Fama EF (1970). Efficient capital markets: a review. *Journal of Finance* 25(2):383-417. doi:10.2307/2325486
- Nakamoto S (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
- Data: Yahoo Finance historical OHLCV via yfinance

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [482]: BTC/ETH return-rank crash-reversion (operator definition)
- Cert [474]: GLD crash-pair null (equity operator context)
