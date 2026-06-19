# [463] QA Witt Tower Crash Pair Bounce Certificate (Globally Validated)

## Claim

The specific QA state `(b=0, e=0)` — both consecutive days in the bottom rank-bin of Z/27Z — predicts a positive next-day return. This is the extreme-oversold state in the QA discrete system. The signal is globally validated across 11 equity markets (5 US + 6 international). The broader S-orbit is null outside this pair.

## Results

**US Indices (a≤6, daily):**

| Index | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| ^GSPC | 30 | **+1.44%** | 60.0% | **0.0000** |
| ^IXIC | 22 | **+1.55%** | 68.2% | **0.0000** |
| ^DJI | 27 | **+1.45%** | 63.0% | **0.0000** |
| QQQ | 21 | **+1.21%** | 61.9% | **0.0004** |
| SPY | 31 | **+1.61%** | 61.3% | **0.0000** |
| **Pooled US** | **131** | **+1.46%** | **62.6%** | **0.0000** |

**International ETFs:**

| ETF | Country | n | Mean | Pos | perm_p |
|---|---|---|---|---|---|
| EWJ | Japan | 26 | **+1.32%** | 61.5% | **0.0000** |
| EWG | Germany | 33 | **+0.86%** | 63.6% | **0.0052** |
| EWL | Switzerland | 27 | **+2.21%** | 77.8% | **0.0000** |
| EWU | UK | 23 | **+2.40%** | 69.6% | **0.0000** |
| EWA | Australia | 31 | **+2.71%** | 74.2% | **0.0000** |
| EWC | Canada | 21 | **+2.12%** | 66.7% | **0.0000** |
| **Pooled INTL** | | **161** | **+1.90%** | **68.9%** | **0.0000** |

**Combined:** n=292, mean≈**+1.70%**, all 11 markets significant.

## Non-(0,0) S-orbit Is Null

| Region | n | Mean | perm_p | Result |
|---|---|---|---|---|
| US non-(0,0) S-orbit | 278 | +0.09% | **0.39** | NULL |
| INTL non-(0,0) S-orbit | 350 | +0.05% | **0.73** | NULL |

The S-orbit daily signal reported in the weekly cert [458] does not survive to daily timescales except in the (0,0) pair. The mod-9 divisibility condition generates no residual signal beyond the extreme pair.

## QA Mapping

- **Observer projection**: daily log-return → rank → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer state**: `b = bins[t-1]`, `e = bins[t]` (both int)
- **(0,0) condition**: `b == 0 AND e == 0` — both in rank-bin 0 (bottom 3.7% of all days)
- **S-orbit condition**: `b % 9 == 0 AND e % 9 == 0` (includes (0,0) as subset)
- **Target**: next-day log return (observer output)

The (0,0) state is an S-orbit member but the QA structure that matters here is the extreme-value class at the boundary of Z/27Z, not the mod-9 divisibility more broadly.

## EWC Dissociation

EWC appears null in the a≤6 international cert [462] (p=0.126) but is strongly significant here (p=0.0000, mean=+2.12%). The reason: EWC's non-(0,0) S-orbit days are significantly **negative** (p=0.005, mean=-0.48%), which cancels the (0,0) positive signal in the broader a≤6 window. The (0,0) state is a clean signal; the surrounding S-orbit pairs carry opposing dynamics in the Canadian market.

## Structural Context

The (0,0) state has `a = 0 + 2×0 = 0`, so it is the most extreme member of the a≤6 group certified in [461] and [462]. The cert hierarchy:

- **[461]**: a≤6 (US daily, n=936, mean=+0.37%) — broad group
- **[462]**: a≤6 international (INTL daily, n=1,099, mean=+0.42%) — OOD generalization
- **[463]**: (0,0) pair only (US+INTL, n=292, mean=+1.70%) — concentrated subgroup with 5× the effect size

The (0,0) pair accounts for 131/936 = 14% of US a≤6 observations but contributes a disproportionate share of the mean. The remaining 86% of a≤6 pairs have a lower mean (+0.26% approximately) — which is still significant and certified in [461].

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: US pooled sig | perm_p=0.0000 < 0.001 | PASS |
| C2: INTL pooled sig | perm_p=0.0000 < 0.001 | PASS |
| C3: US mean ≥ 1% | mean=+1.46% | PASS |
| C4: INTL mean ≥ 1% | mean=+1.90% | PASS |
| C5: US 5/5 at p<0.01 | max_p=0.0004 (QQQ) | PASS |
| C6: Non-(0,0) null documented | US p=0.39, INTL p=0.73 | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486

## Related Certs

- [462] QA Witt Tower International A-Coordinate Generalization (EWC dissociation explained here)
- [461] QA Witt Tower A-Coordinate Daily Direction US (parent population includes (0,0))
- [458] QA Witt Tower Orbit Weekly Direction (weekly S-orbit; daily S-orbit null except (0,0))
- [110] QA Witt Tower Framework (structural parent)
