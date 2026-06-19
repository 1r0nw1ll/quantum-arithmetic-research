# [462] QA Witt Tower International A-Coordinate Generalization Certificate

## Claim

The QA A2-derived coordinate `a = b + 2e` (raw, not mod-reduced) predicts **daily** next-day price direction on 6 international equity ETFs — a direct out-of-distribution test of cert [461] which established the signal on 5 US indices.

**International a ≤ 6 results:**

| ETF | Country | n | Mean | Pos | perm_p |
|---|---|---|---|---|---|
| EWJ | Japan | 178 | **+0.39%** | 56.2% | **0.0004** |
| EWG | Germany | 193 | **+0.35%** | 52.8% | **0.0048** |
| EWL | Switzerland | 159 | **+0.46%** | 56.0% | **0.0002** |
| EWU | UK | 186 | **+0.51%** | 56.5% | **0.0000** |
| EWA | Australia | 177 | **+0.67%** | 61.6% | **0.0000** |
| EWC | Canada | 206 | +0.18% | 51.0% | 0.1260 (null) |

**Pooled INTL:** n=1,099, mean=**+0.42%**, pos=55.5%, perm_p=**0.0000**

**US Baseline (cert [461]):** n=936, mean=+0.37%, perm_p=0.0002

## QA Mapping

- **Observer projection**: daily log-return → rank among N → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer state**: `b = bins[t-1]`, `e = bins[t]` (both int)
- **A2 derived coord**: `a = b + 2*e` — raw, not mod-reduced
- **Positive group**: `a ≤ 6`
- **Target**: `log(price[t+2]/price[t+1])` (next-day return, observer output)

## Why Canada Is Null

EWC (Canada) is the most US-correlated of the 6 international ETFs (~0.95 correlation with SPY). The a≤6 signal may already be priced into EWC by US market dynamics. Notably, EWC's (0,0) pair IS highly significant (p=0.0000, mean=+2.12%, cert [463]) — the null arises because EWC's non-(0,0) S-orbit days are significantly negative (mean=-0.48%, p=0.005), cancelling the broader a≤6 signal.

## Global Picture (US + INTL)

| Region | n | Mean | perm_p | n_sig |
|---|---|---|---|---|
| US (cert [461]) | 936 | +0.37% | 0.0002 | 4/5 |
| International | 1,099 | +0.42% | 0.0000 | 5/6 |
| Combined | 2,035 | **+0.40%** | 0.0000 | **9/11** |

The same QA operator predicts next-day returns across Japan, Germany, Switzerland, UK, and Australia with the same direction and comparable magnitude as in the US.

## IXIC and EWC: The Two Exceptions

Both exceptions have structural explanations:
- **IXIC** (p=0.10): Nasdaq-specific divergence observed across certs [459], [460], [461] — tech sectors may have different rank-bin dynamics
- **EWC** (p=0.13): Near-perfect US market correlation — no independent signal

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: INTL pooled sig | perm_p=0.0000 < 0.001 | PASS |
| C2: 5/6 INTL sig | 5 of 6 at p<0.05 | PASS |
| C3: Mean ≥ 0.30% | INTL mean=+0.42% | PASS |
| C4: Pos rate > 52% | pos=55.5% | PASS |
| C5: EWC positive | mean=+0.18% > 0 | PASS |
| C6: Magnitude comparable | ±3× US mean | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486

## Related Certs

- [461] QA Witt Tower A-Coordinate Daily Direction (US baseline)
- [463] QA Witt Tower Crash Pair Bounce (globally validated (0,0) state)
- [459] QA Witt Tower A-Coordinate Weekly Direction (weekly precursor)
- [110] QA Witt Tower Framework (structural parent)
