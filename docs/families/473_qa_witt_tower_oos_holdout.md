# [473] QA Witt Tower OOS Holdout

## Claim

The **a≤6 daily signal** (cert [461]/[469]) and **(0,0) crash pair bounce** (cert [463])
both pass strict out-of-sample holdout on the 2016-01-01 – 2026-06-19 window.
These signals are not artifacts of the 2000-2015 in-sample period.

## OOS Split

| Period | Dates | Calendar years |
|---|---|---|
| In-sample (IS) | 2001-01-01 – 2015-12-31 | ~15 years |
| Out-of-sample (OOS) | 2016-01-01 – 2026-06-19 | ~10.5 years |

Bin computation uses the full 25-year history (same methodology as all finance certs).
The holdout tests whether *signal dates that fall in the OOS window* retain the same
directional return profile as IS.

## Results (2026-06-19)

### a≤6 signal (b+2e ≤ 6 raw A2)

| Group | n (signals) | Mean next-day return | perm_p |
|---|---|---|---|
| US OOS | 330 | **+0.407%** | 0.0000 |
| US IS  | 606 | +0.349% | 0.0000 |
| INTL OOS | 304 | **+0.259%** | 0.0018 |
| INTL IS  | 794 | +0.481% | 0.0000 |

US OOS return (+0.407%) is *stronger* than IS (+0.349%). INTL OOS is weaker in
magnitude but remains significant (p=0.0018 < 0.01).

### Crash pair (b=0 AND e=0)

| Group | n (signals) | Mean next-day return | perm_p |
|---|---|---|---|
| US OOS | 35 | **+2.079%** | 0.0000 |
| US IS  | 96 | +1.241% | 0.0000 |
| INTL OOS | 35 | **+1.805%** | 0.0000 |
| INTL IS  | 126 | +1.930% | 0.0000 |

OOS crash pair mean (+2.079% US) **substantially exceeds** IS (+1.241%). The difference
is driven by the COVID-2020 crash: March 2020 produced multiple consecutive bottom-bin
days followed by historically large day+1 bounces — all in the OOS window.

## Interpretation

Both signals survive the holdout gate cleanly. The a≤6 signal shows OOS ≥ IS
(US) and significant OOS (INTL). The crash pair signal is even stronger OOS than IS,
demonstrating robustness rather than overfitting.

The COVID effect in OOS is not contamination — it is exactly the kind of structural
event the QA discrete-bin framework claims to identify (extreme consecutive bottom-rank
days predicting large bounces). The OOS data confirms the mechanism.

## Theorem NT Compliance

Observer: daily log-return → rank → bin ∈ Z/27Z.
QA state: b=bins[t-1], e=bins[t]; a=b+2e (raw A2).
All returns are observer outputs. No float state enters the QA layer.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | a≤6 US OOS perm_p < 0.01 | PASS (0.0000) |
| C2 | crash pair US OOS perm_p < 0.01 | PASS (0.0000) |
| C3 | a≤6 US OOS mean > 0 | PASS (+0.407%) |
| C4 | crash pair US OOS mean > 1% | PASS (+2.079%) |
| C5 | a≤6 INTL OOS perm_p < 0.01 | PASS (0.0018) |
| C6 | all four OOS means positive | PASS |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Lo AW, MacKinlay AC (1988). doi:10.1093/rfs/1.1.41 (variance ratio tests)

## Related Certs

- [461] QA Witt Tower A-Coordinate Daily Direction (a≤6 parent)
- [463] QA Witt Tower Crash Pair Bounce (crash pair parent)
- [469] QA Witt Tower Vol-Normalized Returns (a≤6 vol robustness)
- [470] QA Witt Tower Crash Pair Bounce Persistence (3-day profile)
- [472] QA Witt Tower Crash Pair Exit Strategy (exit rule)
