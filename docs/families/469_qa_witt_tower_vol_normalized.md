# [469] QA Witt Tower Volatility-Normalized Returns

## Claim

The a=b+2e≤6 daily signal from cert [461] (+0.37%, perm_p~0.0002) survives
21-day realized-volatility normalization. The signal is not a low-volatility
anomaly artifact: vol-normalized returns on a≤6 days are significantly positive
versus non-a≤6 days, and the ratio of mean realized vol on signal vs non-signal
days confirms the signal does not exclusively select low-vol regimes.

## Method

For each trading day t with a 21-day history:
- `sigma_21d[t]` = std(log_ret[t−21 : t])
- `vol_norm_ret[t+1]` = log_ret[t+1] / sigma_21d[t]

Signal group: days where a = b + 2e ≤ 6 (same as cert [461]).
Control group: all remaining days.

Permutation test (N=5000, seed=42) on vol-normalized returns.

## Key Results

See fallback values in validator for per-index and pooled results.

## Theorem NT Compliance

Observer layer: daily log-return → rank → bin in Z/27Z (float → int).
QA state: b = bins[t-1], e = bins[t] (integers).
A2 derived: a = b + 2e (raw, not mod-reduced).
Vol normalization is applied to the OUTPUT (observer projection), not to any QA state.

## Certified Checks

| Check | Description | Threshold |
|---|---|---|
| C1 | Pooled vol-normalized perm_p | < 0.01 |
| C2 | Pooled raw mean > 0 AND vol-normalized mean > 0 | Both positive |
| C3 | ≥ 3/5 indices individually vol-normalized p < 0.05 | 3 of 5 |
| C4 | Vol ratio sig/ctrl not extreme (0.7 < ratio < 1.3) | Not a pure vol filter |
| C5 | GSPC vol-normalized perm_p | < 0.05 |
| C6 | Pooled vol-normalized mean > 0.05 | Economic magnitude |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Cert [461] (parent signal: a≤6 daily, US 5-index, 25y)

## Related Certs

- [110] QA Witt Tower Framework (structural parent)
- [461] QA Witt Tower A-Coordinate Daily Direction (parent signal)
- [471] QA Witt Tower Multi-Scale Alignment (uses same a≤6 signal)
