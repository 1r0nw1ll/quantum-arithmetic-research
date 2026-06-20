# [475] QA Witt Tower Vol-Sizing — Equal-Weight Optimal

## Claim

Vol-targeting position sizing (weight inversely proportional to trailing 21-day
realized volatility) **degrades Sharpe ratio** for QA crash-bounce signals.
**Equal-weight sizing is Sharpe-optimal.**

The mechanism is structural: a≤6 and (0,0) crash pair signals fire specifically into
high-volatility regimes (vol_ratio=1.69, cert [469]). Vol-targeting reduces position
on high-vol days — precisely the days generating alpha. The high vol is the signal,
not noise to be suppressed.

## Sizing Definitions

| Strategy | Per-trade weight | Per-trade return |
|---|---|---|
| Equal-weight (EW) | 1 | `rets[t+1]` |
| Vol-targeted (VT) | `1 / std(rets[t-20:t])` | `rets[t+1] / σ_t` |

σ_target cancels in the Sharpe ratio; VT weight is `1/σ_t` without loss of generality.

## Results (2026-06-19)

### Crash pair (0,0) signal

| Group | EW Sharpe | VT Sharpe | EW/VT ratio | Degradation |
|---|---|---|---|---|
| US pooled (n=129) | **0.3663** | 0.1905 | **1.92×** | −48% |
| INTL pooled (n=159) | **0.4482** | 0.3242 | 1.38× | −28% |

### a≤6 signal

| Group | EW Sharpe | VT Sharpe | EW/VT ratio |
|---|---|---|---|
| US pooled (n=931) | **0.1429** | 0.1244 | 1.15× |
| INTL pooled (n=1090) | **0.1465** | 0.0820 | 1.79× |

Vol-targeted returns remain positive in all cases (crash pair US VT Sharpe=0.191 > 0),
confirming the underlying signal survives. But equal-weight strictly dominates.

## Interpretation

Vol-targeting is motivated by the assumption that return-per-unit-risk is constant
across vol regimes — in that world, vol-targeting equalizes risk exposure and improves
Sharpe. For standard momentum or value signals this often holds.

For QA crash-bounce signals it fails because the assumption is violated:

1. **Crash pair specifically selects high-vol days** (bins[t-1]=0 AND bins[t]=0 means
   two consecutive bottom-rank return days — this is definitionally a high-vol regime).
2. The bounce alpha on those days (mean +1.47%) is **caused by** the extreme regime,
   not independent of it.
3. Vol-targeting sets `w = 1/σ_t` → smaller position on crash pair days, larger on
   ordinary days. This inverts the information.

The correct risk model for QA crash-bounce signals: **size equally per signal trigger**.
The signal is already a vol-regime selector; further vol-adjustment double-adjusts.

## Connection to cert [469]

Cert [469] showed that a≤6 signal days have vol_ratio=1.69 vs control. This cert
quantifies the Sharpe cost of vol-adjusting into that finding: EW/VT ratio=1.92 for
crash pair. The two certs together bound the answer: "high vol IS the signal" (469)
and "removing it costs 48% Sharpe" (475).

## Theorem NT Compliance

Observer: daily log-return → rank → bin ∈ Z/27Z.
QA state: b=bins[t-1], e=bins[t]; a=b+2e (raw A2).
Both EW and VT returns are observer outputs. The vol window `std(rets[t-20:t])` is
an observer computation on prior returns; it never feeds back as QA state.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | Crash pair EW Sharpe > VT Sharpe (US) | PASS (0.366 > 0.191) |
| C2 | a≤6 EW Sharpe > VT Sharpe (US) | PASS (0.143 > 0.124) |
| C3 | Crash pair EW Sharpe > VT Sharpe (INTL) | PASS (0.448 > 0.324) |
| C4 | Crash pair EW US Sharpe > 0.30 | PASS (0.366) |
| C5 | Crash pair VT US Sharpe > 0 (signal not destroyed) | PASS (0.191) |
| C6 | Crash pair EW/VT Sharpe ratio > 1.5 (substantial) | PASS (1.92×) |

## Primary Sources

- Moreira A, Muir T (2017). doi:10.1111/jofi.12513 (managed portfolios)
- Barroso P, Santa-Clara P (2015). doi:10.1016/j.jfineco.2014.11.003 (momentum vol-mgmt)

## Related Certs

- [461] QA Witt Tower A-Coordinate Daily Direction (a≤6 signal)
- [463] QA Witt Tower Crash Pair Bounce (crash pair signal)
- [469] QA Witt Tower Vol-Normalized Returns (vol_ratio=1.69 established)
- [472] QA Witt Tower Crash Pair Exit Strategy (equal-weight EW Sharpe=0.367 baseline)
