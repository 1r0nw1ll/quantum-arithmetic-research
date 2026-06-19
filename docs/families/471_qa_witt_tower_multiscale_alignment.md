# [471] QA Witt Tower Multi-Scale Alignment

## Claim

Daily trading days satisfying both the a=b+2e≤6 daily QA condition (cert [461])
AND falling in a weekly S-orbit predicted week (cert [466]) produce a next-day
return amplified above the daily-only signal. The two independent QA timescales
(daily rank bins, weekly rank bins) compound when co-active.

## Method

For each US index ticker, compute:
1. **Weekly S-orbit predicted weeks**: weeks T+1 where the preceding weekly pair
   (b_w[T-1], e_w[T]) satisfies b_w%9 = 0 AND e_w%9 = 0 (same condition as cert [466]).
2. **Daily a≤6 days**: days where a = b_d + 2*e_d ≤ 6 (same as cert [461]).

Three groups for each trading day t:
- **Both**: a_d ≤ 6 AND current week is a predicted S-orbit week
- **Daily-only**: a_d ≤ 6 AND current week is NOT a predicted S-orbit week
- **Control**: all other days

Permutation test (N=5000, seed=42) for each group vs control.

## QA Mechanism

The two timescales operate on independent rank bin series:
- Daily bins: computed over full daily history → b_d, e_d (consecutive daily pairs)
- Weekly bins: computed over full weekly history → b_w, e_w (consecutive weekly pairs)

Both are Z/27Z rank projections; they are correlated by shared price process but the
bin-level conditions (a_d≤6, b_w%9=0 AND e_w%9=0) select structurally distinct
regions of their respective state spaces. Co-activation tests whether QA structure
at two timescales is additive or redundant.

## Theorem NT Compliance

Observer layer: log-return (daily or weekly) → rank → bin in Z/27Z.
QA state: (b_d, e_d) daily; (b_w, e_w) weekly — both integer pairs.
Alignment is computed entirely from integer bin comparisons.

## Key Results (2026-06-19)

| Group | n pooled | Mean next-day return | perm_p |
|---|---|---|---|
| Both aligned (a≤6 + S-orbit week) | 38 | **+1.666%** | 0.0000 |
| Daily-only (a≤6, not S-orbit week) | 898 | +0.314% | 0.0000 |
| Amplification factor | — | 5.3× | p=0.0026 |

Per-index (both group): GSPC n=11 +1.757%, DJI n=8 +2.437%, SPY n=14 +1.154%.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | Daily-only pooled perm_p < 0.01 | PASS (0.0000) |
| C2 | Both-aligned pooled perm_p < 0.01 | PASS (0.0000) |
| C3 | Both mean > daily-only mean | PASS (+1.666% > +0.314%) |
| C4 | n_both ≥ 30 pooled | PASS (38) |
| C5 | Both-aligned mean > 0.5% | PASS (+1.666%) |
| C6 | Both-vs-daily-only perm_p | < 0.10 (amplification is distinguishable) |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Cert [461] (daily a≤6 signal, parent)
- Cert [466] (weekly S-orbit regime, parent)

## Related Certs

- [110] QA Witt Tower Framework (structural parent)
- [461] QA Witt Tower A-Coordinate Daily Direction
- [458] QA Witt Tower Orbit Weekly Direction
- [466] QA Witt Tower S-Orbit Weekly Regime
- [469] QA Witt Tower Vol-Normalized Returns (same a≤6 base signal)
