# [472] QA Witt Tower Crash Pair Exit Strategy

## Claim

For (0,0) crash-pair signals (certs [463], [470]), **Strategy A (exit at day+1 close)
is Sharpe-dominant** over Strategy B (hold to day+3 close) in both US and INTL markets.

The raw return difference between A and B is NOT statistically significant (US p=0.51,
INTL p=0.97). The extra +0.42% (US) from holding comes with ~50% more return variance.

**Operative verdict: exit at day+1 close.**

## Strategy Definitions

| Strategy | Entry | Exit | Per-trade return |
|---|---|---|---|
| A (optimal) | Open day+1 | Close day+1 | log_ret[t+1] |
| B (hold) | Open day+1 | Close day+3 | log_ret[t+1]+[t+2]+[t+3] |
| C (re-entry) | Open day+3 | Close day+3 | log_ret[t+3] |

All enter at the open following a (0,0) crash pair (bins[t-1]=0 AND bins[t]=0).

## Results (2026-06-19)

### US (5 indices × 25y, n=131 pooled signals)

| Strategy | Mean return | Std (per trade) | Sharpe | perm_p vs ctrl |
|---|---|---|---|---|
| A: exit day+1 | +1.465% | 3.988% | **0.367** | 0.0000 |
| B: hold day+3 | +1.889% | 5.975% | 0.316 | 0.0000 |
| C: re-enter day+2 | +0.836% | 3.803% | 0.220 | 0.0000 |

A vs B raw difference: p = 0.506 (NOT significant)

### INTL (6 ETFs × 25y, n=161 pooled signals)

| Strategy | Mean return | Std (per trade) | Sharpe | perm_p vs ctrl |
|---|---|---|---|---|
| A: exit day+1 | +1.903% | 4.242% | **0.449** | 0.0000 |
| B: hold day+3 | +1.882% | 6.833% | 0.275 | 0.0000 |
| C: re-enter day+2 | +0.248% | 4.046% | 0.061 | 0.0438 |

A vs B raw difference: p = 0.974 (NOT significant)

## Interpretation

Strategy B captures nominally more return (+0.42% US, −0.02% INTL), but:
- The difference is statistically indistinguishable from noise
- Holding through day+2 adds +50% return variance (US: 5.98% vs 3.99% std)
- The Sharpe degradation from A to B: −0.051 (US) and −0.173 (INTL)

Strategy C (re-entering at day+2 close) is strictly dominated: less mean return
AND lower Sharpe than both A and B.

The day+2 mean-reversion identified in cert [470] (−0.41%) is real but symmetric —
it adds noise without net value. Exit before day+2 is the rational choice.

## Theorem NT Compliance

Observer layer: daily log-return → rank → bin in Z/27Z.
QA state: b=bins[t-1]=0, e=bins[t]=0 (integer pair).
Per-trade returns are observer outputs. Sharpe ratio is computed from integer-bin
flagged returns — no QA state is modified by the strategy comparison.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | Strategy A US perm_p < 0.001 | PASS (0.0000) |
| C2 | A Sharpe > B Sharpe, US pooled | PASS (0.367 > 0.316) |
| C3 | A Sharpe > B Sharpe, INTL pooled | PASS (0.449 > 0.275) |
| C4 | A vs B raw difference NOT sig (US p > 0.05) | PASS (p=0.506) |
| C5 | Strategy C dominated (Sharpe C < Sharpe A, both groups) | PASS |
| C6 | A Sharpe > 0.30 in both US and INTL | PASS (0.367, 0.449) |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Cert [463] (crash pair bounce, US+INTL)
- Cert [470] (3-day bounce-giveback-recovery profile)

## Related Certs

- [110] QA Witt Tower Framework
- [463] QA Witt Tower Crash Pair Bounce
- [470] QA Witt Tower Crash Pair Bounce Persistence
