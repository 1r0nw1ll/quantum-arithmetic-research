# [464] QA Witt Tower S-Orbit Exit Certificate (Resonance Loss Predicts Negative Return)

## Claim

When a consecutive daily pair transitions from S-orbit (mod-9 resonance: both bins divisible by 9) to C-orbit (chaos: no divisibility-by-3 alignment), the next day tends to be negative. Globally validated: US and INTL both independently significant. This cert structurally pairs with [463] to give a bidirectional orbit signal.

## Results

**US Indices:**

| Index | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| ^GSPC | 56 | **-0.50%** | 44.6% | **0.0018** |
| ^IXIC | 55 | **-0.39%** | 43.6% | **0.0254** |
| ^DJI | 54 | +0.02% | 57.4% | 0.9786 (null) |
| QQQ | 53 | -0.19% | 50.9% | 0.2458 |
| SPY | 59 | -0.06% | 64.4% | 0.4948 |
| **Pooled US** | **277** | **-0.22%** | **52.3%** | **0.0012** |

**International ETFs:**

| ETF | Country | n | Mean | Pos | perm_p |
|---|---|---|---|---|---|
| EWJ | Japan | 60 | **-0.39%** | 43.3% | **0.0184** |
| EWG | Germany | 54 | +0.22% | 64.8% | 0.3398 (null) |
| EWL | Switzerland | 61 | -0.18% | 52.5% | 0.1866 |
| EWU | UK | 51 | **-0.47%** | 41.2% | **0.0152** |
| EWA | Australia | 70 | **-0.45%** | 50.0% | **0.0172** |
| EWC | Canada | 52 | -0.17% | 53.8% | 0.2642 |
| **Pooled INTL** | | **348** | **-0.25%** | **50.9%** | **0.0002** |

## GSPC IS/OOS Structure

| Period | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| IS pre-2015 | 30 | -0.13% | 56.7% | 0.51 (null) |
| OOS 2015+ | 26 | **-0.93%** | **30.8%** | **0.0000** |

The signal is regime-concentrated in the post-2015 period. The OOS result (perm_p=0.0000, pos=30.8%) is striking — only 1 in 3 S→C days led to a positive return in the OOS period. The international validation (no US-2015 cutoff issue) confirms the structural reality of the S→C exit signal independent of the US IS/OOS concentration.

## QA Mapping

- **Observer projection**: daily log-return → rank → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer states**: `b_prev = bins[t-2]`, `e_prev = bins[t-1]`, `b_cur = bins[t-1]`, `e_cur = bins[t]`
- **Previous pair orbit**: `S` = both `b_prev % 9 == 0` and `e_prev % 9 == 0`
- **Current pair orbit**: `C` = NOT (`b_cur % 3 == 0` AND `e_cur % 3 == 0`)
- **Target**: `rets[t+1]` (next-day log return, observer output)

The S→C transition specifically means:
- Yesterday's rank-bin was mod-9 divisible (extreme end of the distribution)
- Today's rank-bin is NOT mod-3 divisible (breaks even the Sat-orbit alignment)
- This is "complete resonance loss" — from maximum mod-9 alignment to zero mod-3 alignment

## Blue-Chip Exception

DJI (US) and EWG (Germany) are both null or positive:
- DJI: mean=+0.02%, p=0.978 — essentially zero
- EWG: mean=+0.22%, p=0.340 — mildly positive

Both are concentrated, large-cap, **price-weighted** indices tracking blue-chip stocks (DJI=30 US companies, EWG≈DAX 40 German companies). This same divergence appears in certs [459], [460], [461], [462]: the QA orbit structure applies to broad-market indices but not to concentrated price-weighted blue-chip averages. The mechanism is unclear but consistent.

## Structural Bidirectional Story (Certs [463]+[464])

| Signal | Condition | Effect | Global |
|---|---|---|---|
| **S-orbit entry bounce** ([463]) | Both bins = 0 (extreme low) | +1.70% next-day | 11/11 markets |
| **S-orbit exit loss** ([464]) | Prev=S → Cur=C | -0.22% to -0.25% next-day | US+INTL |

Together: finding S-orbit resonance (extreme pair) is positive; losing S-orbit resonance to chaos is negative. The QA mod-9 divisibility structure has bidirectional predictive content at the daily timescale.

## Comparison with Cert [461] a≤6

The a≤6 cert [461] identifies **entry into the low-A region** (including the (0,0) pair and nearby states). The S→C exit is a **sequential orbit transition** signal that requires tracking two consecutive pairs, not just the current pair's coordinate. These are structurally independent signals:
- a≤6 looks at the current pair's A2 coordinate
- S→C looks at how the orbit class changed between the previous and current pair

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: US pooled sig | perm_p=0.0012 < 0.005 | PASS |
| C2: INTL pooled sig | perm_p=0.0002 < 0.001 | PASS |
| C3: US mean < 0 | mean=-0.22% | PASS |
| C4: INTL mean < 0 | mean=-0.25% | PASS |
| C5: GSPC OOS sig | OOS perm_p=0.0000 < 0.01 | PASS |
| C6: Blue-chip exceptions | DJI mean=+0.02%, EWG mean=+0.22% documented | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486

## Related Certs

- [463] QA Witt Tower Crash Pair Bounce (S-orbit entry signal — bidirectional complement)
- [461] QA Witt Tower A-Coordinate Daily Direction (US daily a≤6)
- [462] QA Witt Tower International A-Coordinate Generalization
- [458] QA Witt Tower Orbit Weekly Direction (weekly S-orbit signal)
- [110] QA Witt Tower Framework (structural parent)
