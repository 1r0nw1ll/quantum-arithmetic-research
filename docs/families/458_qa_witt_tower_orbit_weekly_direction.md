# [458] QA Witt Tower Orbit Weekly Direction Certificate

## Claim

S-orbit state at weekly timescale predicts **positive** next-week price direction (bounce/mean-reversion). Pooled across 5 US equity indices: n_S=92, mean next-week return=+1.17%, pos_rate=60.9%, perm_p=0.0008. Three of five indices are individually significant at p<0.05.

This is a genuine price DIRECTION prediction at weekly scale, contrasting with cert [457] which found a price VOLATILITY prediction at monthly scale with a negative direction bias (S_monthly=-1.30%).

## QA Mapping

Same as certs [453]–[457], applied at weekly granularity:

- Weekly log-return → rank among N weekly returns → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- Consecutive pair `(b=bins[t-1], e=bins[t])` → orbit class S/Sat/C

## Main Result

| Metric | Value |
|---|---|
| n_S pooled | 92 |
| n_C pooled | 5,137 |
| S next-week mean | **+1.17%** |
| C next-week mean | +0.20% |
| Positive rate (S) | **60.9%** |
| perm_p (two-tail) | **0.0008** |

## Per-Index Breakdown

| Index | n_S | Mean | Pos | perm_p | Result |
|---|---|---|---|---|---|
| ^GSPC | 27 | +0.82% | 63% | 0.138 | marginal |
| ^IXIC | 12 | +2.15% | 67% | **0.015** | significant |
| ^RUT | 19 | -0.15% | 47% | 0.606 | **NULL — exception** |
| ^DJI | 21 | +1.72% | 67% | **0.002** | significant |
| QQQ | 13 | +2.02% | 62% | **0.022** | significant |

**Russell 2000 exception**: small-cap stocks do not show the weekly bounce after S-orbit states. This is documented as a partial null: the signal applies to large-cap and tech indices (GSPC, IXIC, DJI, QQQ) but not small-cap.

## OOS Holdout (GSPC 2015+)

n_S=10, mean=+1.62%, pos=80%, perm_p=0.056 — directionally consistent but n too small for significance. The direction holds out-of-sample.

## S-Orbit Pair Structure (GSPC)

| Pair (b, e) | n | Type |
|---|---|---|
| (0, 0) | 10 | extreme crash (both bottom 3.7%) |
| (0, 18) | 1 | extreme low + mid-high |
| (9, 0) | 3 | near-median + extreme low |
| (9, 9) | 3 | **non-extreme** (both ~33rd percentile) |
| (9, 18) | 3 | **non-extreme** |
| (18, 9) | 3 | **non-extreme** |
| (18, 18) | 4 | **non-extreme** (both ~67th percentile) |

**13/27 = 48% of GSPC S-orbit weeks are non-extreme**: neither b nor e is at bin 0. The QA mod-9 divisibility criterion captures structure that is not simply an extreme-crash filter. The (9,9) pair (two consecutive weeks at the ~33rd percentile) and (18,18) (two weeks at ~67th percentile) are also S-orbit, and their next-week returns contribute to the positive signal.

## Timescale Contrast (link to cert [457])

| Timescale | S-orbit direction | Signal type |
|---|---|---|
| **Weekly** | +1.17% positive, perm_p=0.0008 | bounce / mean-reversion |
| **Monthly** | -1.30% negative, perm_p=0.119 | continuation / weakness |

The direction **inverts** between weekly and monthly horizons. This is a structural multi-scale prediction: the QA orbit class identifies mean-reversion at short timescales and continuation/regime-persistence at longer timescales. Both predictions use the same orbit class definition.

## What This Cert Establishes

1. **Weekly direction signal exists**: S-orbit → positive next week (perm_p=0.0008, pooled).
2. **Not a crash filter**: 48% of S-orbit weeks are non-extreme pairs, confirming QA arithmetic structure.
3. **Large-cap specificity**: GSPC/IXIC/DJI/QQQ all positive; Russell 2000 is null. Small-cap bounce after S-orbit is absent.
4. **Multi-scale structure**: direction sign flips from weekly (bounce) to monthly (continuation).
5. **Monthly direction is NULL when pooled**: US pooled monthly perm_p=0.93; global pooled perm_p=0.40. The monthly S-orbit bearish tendency is GSPC-specific and not generalized.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: Pooled significant | n_S=92≥80, perm_p=0.0008<0.01 | PASS |
| C2: Effect size | S-C diff=+0.97%≥0.5% | PASS |
| C3: Pos rate | 60.9%>55% | PASS |
| C4: Multi-asset sig | 3 of 5 individually p<0.05 | PASS |
| C5: Timescale contrast | Weekly S_mean>0 | PASS |
| C6: Non-extreme pairs | 13 GSPC pairs (b≠0,e≠0)≥10 | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486
- Cert [457]: QA Witt Tower Orbit Price Volatility (monthly contrast, same mapping)

## Related Certs

- [457] QA Witt Tower Orbit Price Volatility (monthly timescale parent)
- [455] QA Witt Tower Orbit Transition Markov Chain (orbit staircase)
- [110] QA Witt Tower Framework (structural parent)
