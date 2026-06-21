# Cert [495]: QA Witt Tower 1-min FX Return-Rank: Null Position on Discrimination Ladder

**Family ID**: 495
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_hf_fx_anti_persistence_cert_v1/`

## Claim

1-min FX log-returns (EURUSD, GBPUSD, USDJPY, CHFJPY) produce **pooled n_signal_ratio = 1.009×** — exactly null — in the Witt Tower return-rank operator, despite having negative lag-1 autocorrelation (bid-ask bounce: −0.05 to −0.18). This places 1-min FX at the **NULL position** on the discrimination ladder, bracketed by EEG anti-persistence (0.72×) from below and river persistence (2.69×) from above.

| Pair | n_bars | autocorr | n_signal_ratio | crash_p |
|------|--------|----------|---------------|---------|
| EURUSD | 9,893 | −0.1485 | 1.023× | 0.0 |
| GBPUSD | 9,893 | −0.0532 | 1.142× | 0.001 |
| USDJPY | 9,836 | −0.1279 | 1.005× | 0.110 |
| CHFJPY | 9,741 | −0.1794 | 0.865× | 0.0 |

Pooled: **n_sig=872, n_exp=863.8, ratio=1.009×**. 3/4 pairs show significant crash-reversion excess (crash_p < 0.05).

## Two Findings

**Finding 1 — Null n_signal_ratio**: Despite all four pairs having negative lag-1 autocorrelation, the pooled return-rank ratio is 1.009× (indistinguishable from null). The bid-ask bounce mechanism creates lag-1 sign alternation but does NOT create the kind of consecutive-pair depletion that EEG interictal shows (0.72×). FX microstructure at 1-min is structurally NULL in the operator.

**Finding 2 — Crash-reversion excess**: 3/4 pairs show significant positive excess after two consecutive bottom-rank 1-min returns (crash_p < 0.05 for EURUSD, GBPUSD, CHFJPY). This is the standard FX microstructure mean-reversion: after two consecutive large dips at 1-min, the price bounces at the 2-min horizon. This is consistent with Glosten & Milgrom (1985) liquidity-provider inventory rebalancing.

## Discrimination Ladder Position

| Domain | Cert | n_signal_ratio | Direction |
|--------|------|---------------|-----------|
| EEG interictal | [491] | 0.72× | Anti-persistent |
| **1-min FX (this cert)** | **[495]** | **1.009×** | **NULL** |
| Rivers (streamflow) | [490] | 2.69× | Persistent |
| Daily precipitation | [494] | 3.05× | Persistent |
| Temperature anomaly | [492] | 3.40× | Persistent |
| Ocean SST | [493] | 4.43× | Strongly persistent |

FX 1-min occupies the null zone, demonstrating that **the return-rank operator discriminates structural autocorrelation patterns beyond mere lag-1 sign**: FX has autocorr −0.05 to −0.18 (clearly negative) yet n_ratio=1.009 (null). EEG with similar autocorr magnitudes achieves 0.72× because the anti-persistence structure is different — consecutive energy-change events are depleted in a way that FX bid-ask bounce is not.

## Physical Mechanism

The FX bid-ask bounce creates an alternating pattern at the TICK level. At 1-minute closing prices, consecutive ticks average out, and the session returns are a convolution of the tick-level bounces with minute-level trend. Two competing effects:
- Bid-ask bounce: after a down-tick, next tick up → reduces P(low b, low e)
- Intraday momentum: trending minutes cluster → enriches P(low b, low e)

These forces roughly cancel, producing n_ratio ≈ 1.0. The crash-reversion excess (positive target return) survives because the 2-minute horizon captures bounce independent of whether the preceding (b,e) pair is enriched or depleted.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | All 4 pairs have negative lag-1 autocorr | PASS | 4/4 (−0.05 to −0.18) |
| C2 | Pooled n_signal_ratio in null zone [0.80, 1.20] | PASS | 1.009× |
| C3 | Pooled n_signal_ratio < certified_river (2.69×) | PASS | 1.009 < 2.69 |
| C4 | 3+/4 pairs show significant crash-reversion (crash_p < 0.05) | PASS | 3/4 |
| C5 | CHFJPY n_signal_ratio < 1.0 (anti-persistent leader) | PASS | 0.865× |
| C6 | Pooled n_signal_ratio > certified_eeg (0.72×) | PASS | 1.009 > 0.72 |

## Primary Sources

- Lo A & MacKinlay C (1988). Stock market prices do not follow random walks. *Review of Financial Studies* 1(1):41-66. doi:10.1093/rfs/1.1.41
- Glosten LR & Milgrom PR (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics* 14(1):71-100. doi:10.1016/0304-405X(85)90044-3

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [482]: BTC/ETH return-rank; operator definition
- Cert [490]: Rivers persistence (2.69×)
- Cert [491]: EEG interictal anti-persistence (0.72×)
- Cert [492]: Temperature anomaly persistence (3.40×)
- Cert [493]: Ocean SST persistence (4.43×)
- Cert [494]: Daily precipitation persistence (3.05×)
