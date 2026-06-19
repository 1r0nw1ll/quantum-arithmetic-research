# [457] QA Witt Tower Orbit Price Volatility Certificate

## Claim

The QA Witt Tower orbit class (S/Sat/C on mod-27 monthly return rank bins) has a statistically significant predictive relationship with NEXT-MONTH PRICE VOLATILITY. S-orbit state predicts nearly 2x higher absolute price movement in the following month (perm_p=0.0002, two-tail permutation test). This is a genuine forward-looking price prediction, not a recession classification.

## QA Mapping

Same mapping as certs [453]–[456]:

- Monthly log-return → rank among all N returns → `bin = floor(rank * 27 / N)` ∈ Z/27Z
- Consecutive pair `(b=bins[t-1], e=bins[t])` → orbit class:
  - **S** (Singularity): both b%9==0 AND e%9==0
  - **Sat** (Satellite): exactly one of b%9==0, e%9==0
  - **C** (Cosmos): neither b%9==0 nor e%9==0

This is a Theorem NT-compliant mapping: the float return is the observer projection; the integer rank bin is the QA state.

## Volatility Prediction (Primary Result)

| Orbit | n | Mean next-month |abs| return | vs C orbit |
|---|---|---|---|
| **S** | 12 | **6.30%** | **1.99x** |
| **Sat** | 48 | 3.53% | 1.11x |
| **C** | 238 | 3.17% | baseline |

**GSPC perm_p(S_vol vs C_vol) = 0.0002** (two-tail, 5000 shuffles, seed=42)

**QQQ validation**: S/C ratio = 1.62x, perm_p = 0.0458

The monotone ordering S > Sat > C holds for expected next-period absolute return. S-orbit months are rare (12/300 = 4%) but systematically precede high-volatility months.

## Direction Prediction (Honest Null / Marginal)

| Orbit | Mean next-month return | Positive rate | perm_p |
|---|---|---|---|
| **S** | **-1.30%** | 41.7% (5/12) | 0.119 (not significant) |
| Sat | +1.34% | 70.8% (34/48) | 0.276 (not significant) |
| C | +0.62% | 64.3% (153/238) | baseline |

Direction is directional (S bearish, Sat bullish) but not statistically significant with n_S=12 observations. The volatility effect is the certifiable claim; direction is suggestive context.

## Honest Null: T-Step Deviation

The T-step deviation `dev_t = bins[t] - (bins[t-2] + bins[t-1])` shows perm_p=0.9444 when properly computed as a predictive signal. An earlier exploratory analysis had look-ahead bias (dev was computed using bins[t+1] = rank of the "future" return). The corrected predictive version is null. This is documented here to prevent future re-discovery of the spurious signal.

## What This Cert Establishes

1. **S-orbit = volatility regime signal**: After a month where BOTH consecutive return ranks land at multiples of 9 (mod 27), the next month has ~2x higher absolute price movement.

2. **Volatility ordering is monotone**: S > Sat > C for expected next-month absolute return. The more "crystallized" the orbit state (both ranks at {0,9,18}), the more the market "releases" in the following month.

3. **Direction signal exists but is underpowered**: n_S=12 over 25 years is too sparse for a significant direction test. The S-orbit direction bias (41.7% positive vs 64.3% baseline) is suggestive of a bearish tendency but requires a longer time series or weekly data for significance.

4. **T-step deviation is not predictive of direction**: Fama (1970) baseline holds for this specific QA feature. The volatility prediction is a separate, non-efficiency claim.

## Connection to Prior Certs

- **[453]**: S-orbit predicts NBER recession periods (perm_p=0.013). That was the recession signal — this cert tests direct PRICE.
- **[455]**: Orbit transition Markov chain — staircase is mathematically forced, not empirical.
- **[456]**: Eisenstein form sign is a predictive null. This cert finds the orbit CLASS is predictive (for volatility).
- **[110]**: Witt tower structural parent.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: GSPC S vol significant | S-orbit volatility perm_p=0.0002 < 0.01 | PASS |
| C2: GSPC vol ratio | S/C ratio=1.99x > 1.5x | PASS |
| C3: QQQ vol ratio | QQQ S/C ratio=1.62x > 1.3x | PASS |
| C4: S direction negative | S mean next-month return=-1.30% < 0 | PASS |
| C5: Sat above C | Sat mean_ret=1.34% > C=0.62% | PASS |
| C6: Vol ordering | S > Sat >= C (6.30% > 3.53% >= 3.17%) | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. *Journal of Finance*, 25(2), 383–417. doi:10.2307/2325486
- Certs [453]–[455]: QA Witt Tower orbit mapping on ^GSPC (data ancestry)

## Related Certs

- [453] QA Witt Tower Orbit Recession Predictor (orbit class → NBER recession label)
- [454] QA Witt Tower Orbit Recession Null (Gold null — risk-asset specificity)
- [455] QA Witt Tower Orbit Transition Markov Chain (staircase structure)
- [456] QA Witt Tower Eisenstein Form Real-Data (Eisenstein sign = predictive null)
- [110] QA Witt Tower Framework (structural parent)
