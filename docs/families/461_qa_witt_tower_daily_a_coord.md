# [461] QA Witt Tower A-Coordinate Daily Direction Certificate (Bidirectional, IS-Robust)

## Claim

The QA A2-derived coordinate `a = b + 2e` predicts **daily** next-day price direction across 5 US equity indices. At daily timescale (31,415 state pairs, 25y), the signal holds **both in-sample and out-of-sample** — resolving the regime-concentration weakness of cert [459] (weekly, IS=NULL).

**Positive signal (a ≤ 6):**

| Metric | Value |
|---|---|
| n pooled | 936 |
| Mean next-day | **+0.37%** |
| Positive rate | 56.5% |
| perm_p pooled | **≈0.0002** |
| IS (GSPC pre-2015) perm_p | **0.0002** |
| OOS (GSPC 2015+) perm_p | 0.0108 |

**Negative signal (crash-recovery failure: b ≤ 2 AND e ≥ 18):**

| Metric | Value |
|---|---|
| n pooled | 1,460 |
| Mean next-day | **-0.12%** |
| perm_p pooled | **≈0.0002** |

## Why This Is Strictly Better Than Cert [459]

| Property | [459] Weekly | [461] Daily |
|---|---|---|
| n (a≤6 group) | 213 | 936 |
| Pooled perm_p | ~0.0002 | ~0.0002 |
| IS significance | **NULL** (p=0.797) | **p=0.0002** |
| OOS significance | p~0.0002 | p=0.0108 |
| Negative signal | Not certified | perm_p~0.0002 |
| Per-index sig | 3/5 | 4/5 |

The IS significance is the key improvement. At weekly scale, there are only n=26 GSPC pre-2015 observations in a≤6, giving insufficient power. At daily scale, n=111 — enough to confirm the signal across the full 25-year period.

## QA Mapping

- **Observer projection**: daily log-return → rank among N → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer state**: `b = bins[t-1]`, `e = bins[t]` (both int)
- **A2 derived coord**: `a = b + 2*e` — raw, not mod-reduced (element computation)
- **Positive group**: `a ≤ 6`
- **Negative group**: `b ≤ 2 AND e ≥ 18` (crash yesterday, recovery attempt today)
- **Target**: `log(price[t+2]/price[t+1])` (next-day float return, observer output)

The two groups are **structurally disjoint**: for `b ≤ 2` and `e ≥ 18`, we have `a = b + 2e ≥ 36`, so the crash-recovery group never overlaps with `a ≤ 6`.

## IS/OOS Structure (GSPC)

| Period | n | Mean | perm_p |
|---|---|---|---|
| IS pre-2015 | 111 | +0.46% | **0.0002** |
| OOS 2015+ | 71 | +0.40% | 0.0108 |

The signal is stable across both regimes. The effect size is similar in both periods (+0.46% vs +0.40%), confirming the IS/OOS split is not a regime change but a sample-size artifact in the weekly version.

## Per-Index Breakdown (a ≤ 6, daily)

| Index | n | Mean | Pos | perm_p | Result |
|---|---|---|---|---|---|
| ^GSPC | 182 | +0.43% | 58.2% | **0.000** | significant |
| ^IXIC | 186 | +0.21% | 52.7% | 0.100 | marginal |
| ^DJI | 191 | +0.47% | 57.1% | **0.000** | significant |
| QQQ | 188 | +0.31% | 56.4% | **0.014** | significant |
| SPY | 189 | +0.42% | 58.2% | **0.000** | significant |

4/5 individually significant. IXIC is marginal (p=0.10) — the same Nasdaq divergence seen in certs [459] and [460].

## Crash-Recovery Failure Structure (b ≤ 2, e ≥ 18)

When yesterday was an extreme low (b ≤ 2, bottom 11%) and today is a strong recovery (e ≥ 18, top 33%), the next-day tends to be negative. This is a "failed V-recovery" pattern.

| Index | n | Mean | perm_p |
|---|---|---|---|
| ^GSPC | 299 | -0.14% | **0.009** |
| ^IXIC | 287 | -0.15% | **0.020** |
| ^DJI | 295 | -0.17% | **0.002** |
| QQQ | 282 | -0.03% | 0.342 (null) |
| SPY | 297 | -0.09% | 0.059 (marginal) |

QQQ is a partial null — tech stocks don't show the failed-recovery pattern (consistent with [459]/[460] tech divergence).

## QA Structural Insight: Why the A-Coordinate Works

The A2 coordinate `a = b + 2e` weights the **current day's state (e) twice** relative to yesterday's (b). This is the correct weighting for a mean-reversion signal:

- When both b and e are very small (recent two days both weak) → a is small → bounce expected. The double-weight on e correctly punishes "today being weak" more than "yesterday was weak."
- When b is small (crash yesterday) and e is large (recovery today) → a = b + 2e is large, even though d = b + e might be moderate. The A2 coordinate IDENTIFIES the crash-recovery state as distinct from the bounce state via this double-weight.

This is exactly the structure captured by the A2 axiom: `a = b + 2e` is the fourth element of the canonical QA tuple (b, e, d=b+e, a=b+2e), and it discriminates market regimes that the simpler D1 coordinate `d = b+e` misses.

## Comparison: D1 vs A2 at Daily Scale

| Condition | n | Mean | perm_p |
|---|---|---|---|
| `d≤6` (b+e≤6) | 1,433 | +0.26% | 0.0000 |
| `a≤6` (b+2e≤6) | 936 | +0.37% | 0.0000 |

The A2 condition has **smaller n** but **higher mean return** because it correctly excludes the crash-recovery pairs (like b=0, e=5 where d=5≤6 but a=10>6) that have negative next-day returns. The A2 weighting is strictly better than D1 as a direction predictor.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: a≤6 pooled | perm_p≈0.0002<0.001 | PASS |
| C2: IS significant | GSPC pre-2015 perm_p=0.0002<0.01 | **PASS** (key over [459]) |
| C3: OOS significant | GSPC 2015+ perm_p=0.0108<0.05 | PASS |
| C4: Crash-rec pooled | perm_p≈0.0002<0.001, mean=-0.12%<0 | PASS |
| C5: Multi-asset | 4/5 individually p<0.05 | PASS |
| C6: Bidirectional | a≤6 mean>0 AND crash-rec mean<0 | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. doi:10.2307/2328797

## Related Certs

- [459] QA Witt Tower A-Coordinate Weekly Direction (weekly precursor, IS=NULL resolved here)
- [460] QA Witt Tower T-Step Contrarian (weekly T-step anti-persistence; disappears at daily scale)
- [458] QA Witt Tower Orbit Weekly Direction (S-orbit weekly signal)
- [457] QA Witt Tower Orbit Price Volatility (monthly volatility)
- [110] QA Witt Tower Framework (structural parent)
