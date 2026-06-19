# [470] QA Witt Tower Crash Pair Bounce Persistence

## Claim

The (0,0) rank-bin crash-pair bounce from cert [463] (day+1 US +1.46%, INTL +1.90%)
persists over 3 trading days. The 3-day cumulative return is positive and significant,
establishing the actionable hold duration for the crash-pair signal.

## Method

For each day t where bins[t-1] = 0 AND bins[t] = 0 (both consecutive bottom-bin days):
- day+1 return: log_ret[t+1]
- day+2 return: log_ret[t+2]
- day+3 return: log_ret[t+3]
- 3-day cumulative: sum of day+1 through day+3

Control group: all non-(0,0) days.
3-day cumulative control: random samples of 3 non-(0,0) days.

Permutation test (N=5000, seed=42) at each horizon.

## Theorem NT Compliance

Observer layer: daily log-return → rank → bin in Z/27Z.
QA state: b = bins[t-1] = 0, e = bins[t] = 0.
The (0,0) condition is a pure integer-bin test. Returns at t+k are observer outputs.

## Certified Checks

| Check | Description | Threshold |
|---|---|---|
| C1 | Day+1 US pooled perm_p | < 0.001 (replication of [463]) |
| C2 | Day+1 INTL pooled perm_p | < 0.001 (replication of [463]) |
| C3 | 3-day cumulative US mean | > 0 |
| C4 | 3-day cumulative US perm_p | < 0.05 |
| C5 | Day+2 US pooled mean | > −0.2% (no strong reversal) |
| C6 | 3-day cum mean > 1.0% | Persistence of economically meaningful magnitude |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Cert [463] (parent signal: (0,0) crash pair, US+INTL day+1 bounce)

## Related Certs

- [110] QA Witt Tower Framework (structural parent)
- [463] QA Witt Tower Crash Pair Bounce (parent signal)
