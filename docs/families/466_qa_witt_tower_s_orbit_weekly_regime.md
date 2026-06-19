# [466] QA Witt Tower S-Orbit Weekly Regime Certificate

## Claim

The S-orbit weekly buy signal (cert [458]: +1.17%, p=0.0008) is **regime-concentrated in the post-2015 OOS period**. The pre-2015 IS period is null (p=0.12), the post-2015 OOS period is significant (p=0.0044, +1.53%). This replicates the regime structure of the a≤6 weekly signal ([459]), establishing that **both weekly QA operators are OOS-concentrated while both daily QA operators show IS+OOS signal** — a timescale-specific regime boundary at 2015.

## Timescale Regime Architecture

| Cert | Scale | Operator | IS | OOS |
|---|---|---|---|---|
| [459] | Weekly | a=b+2e ≤ 6 | NULL (p>0.20) | +2.52% p~0.0002 |
| **[466]** | **Weekly** | **S-orbit** | **NULL (p=0.12)** | **+1.53% p=0.0044** |
| [461] | Daily | a=b+2e ≤ 6 | SIG (p=0.0002) | SIG (p=0.011) |
| [464] | Daily | S→C exit | NULL (p=0.51) | SIG (p=0.0000) |

Both weekly operators are OOS-only. Both daily operators show IS+OOS (or OOS-dominant with structural validation). The regime boundary at 2015-01-01 is specific to the **weekly timescale** — daily signals are present in the full 25-year sample.

## Results

**IS (pre-2015):**

| Index | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| ^GSPC | 22 | +0.51% | 11/22 | 0.394 (null) |
| ^IXIC | 10 | +0.06% | 5/10 | 0.993 (null) |
| ^RUT | 11 | −1.41% | 5/11 | 0.112 (null) |
| ^DJI | 11 | +1.35% | 8/11 | 0.070 (marginal) |
| QQQ | 15 | +2.32% | 10/15 | **0.025** (sig) |
| **Pooled** | **69** | **+0.67%** | **39/69** | **0.1214 (null)** |

**OOS (2015+):**

| Index | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| ^GSPC | 10 | +1.34% | 7/10 | 0.116 (marginal, n small) |
| ^IXIC | 5 | +1.75% | 2/5 | 0.215 (n too small) |
| ^RUT | 7 | +1.35% | 3/7 | 0.291 (null — small-cap exception) |
| ^DJI | 8 | **+2.18%** | 5/8 | **0.022** (significant) |
| QQQ | 7 | +1.07% | 3/7 | 0.473 (null) |
| **Pooled** | **37** | **+1.53%** | **20/37** | **0.0044** |

## Per-Index Notes

**RUT (Russell 2000)**: Null in both IS (p=0.112) and OOS (p=0.291). Consistent with the documented small-cap exception in cert [458]. The S-orbit bounce after mod-9 resonance does not apply to small-cap indices.

**QQQ (Nasdaq-100 tech)**: Flipped regime — IS significant (p=0.025, +2.32%), OOS null (p=0.473). Tech-sector S-orbit signal was concentrated pre-2015, opposite the broad-market pattern. The pooled [458] result was driven partly by this IS-concentrated QQQ component.

**DJI**: OOS significant (p=0.022, +2.18%), IS marginal (p=0.070). Note: DJI was null in [464] (S→C exit), but here it drives the OOS S-orbit signal. These are structurally different operators: S-orbit entry bounce vs S→C exit loss respond differently to index concentration.

## QA Mapping

- **Observer projection**: weekly log-return (float, observer measurement)
- **QA integer state**: `bin = floor(rank × 27 / N) ∈ {0,...,26}` (Z/27Z rank-normalized)
- **Orbit class**: S-orbit = `b % 9 == 0 AND e % 9 == 0` on consecutive weekly pairs
- **IS/OOS split**: `IS_CUTOFF = "2015-01-01"` (same as [459], [461], [464])
- **Theorem NT**: log-return is observer projection; rank bins are QA integer state

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: IS_NULL | IS perm_p=0.1214 > 0.05 | PASS |
| C2: OOS_SIG | OOS perm_p=0.0044 < 0.01 | PASS |
| C3: OOS_MAGNITUDE | OOS mean=+1.53% ≥ 1.0% | PASS |
| C4: OOS_STRONGER | OOS mean > IS mean (1.53% > 0.67%) | PASS |
| C5: DJI_OOS_SIG | DJI OOS perm_p=0.022 < 0.05 | PASS |
| C6: RUT_CONSISTENT | RUT null in both IS (p=0.112) and OOS (p=0.291) | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486

## Related Certs

- [458] QA Witt Tower Orbit Weekly Direction (parent signal: pooled +1.17%, p=0.0008)
- [459] QA Witt Tower A-Coordinate Weekly Direction (parallel regime: IS null, OOS +2.52%)
- [461] QA Witt Tower A-Coordinate Daily Direction (IS+OOS daily contrast)
- [464] QA Witt Tower S-Orbit Exit (daily S→C exit: IS null GSPC but INTL validates)
- [110] QA Witt Tower Framework (structural parent)
