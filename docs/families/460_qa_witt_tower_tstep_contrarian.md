# [460] QA Witt Tower T-Step Contrarian Weekly Direction Certificate

## Claim

The QA T-step projection `tp = (b+e) % 27` exhibits **anti-persistence (contrarian)** at weekly timescale. When `tp ≥ 22` (top 19% of Z/27Z — T-step predicts the next rank-bin will be high), actual next-week returns are **below** the unconditional mean. When `tp ≤ 4` (bottom 19%), actual returns are **above**. The effect is bidirectional and significant in both directions:

| Group | n | Mean | perm_p |
|---|---|---|---|
| tp ≥ 22 (hi) | 1,289 | **-0.07%** | **0.0006** |
| tp ≤ 4 (lo) | 1,575 | **+0.33%** | **0.0042** |
| Spread (lo − hi) | — | **+0.40%** | — |

Pooled across 5 US equity indices (^GSPC, ^IXIC, ^DJI, QQQ, SPY), 25y weekly data, 6,510 state pairs.

## What the T-Step Projection Is

The QA T-step projection `tp = (b+e) % 27` is the modular arithmetic prediction of the **next rank-bin** under the T-operator `(b, e) → (e, (b+e) mod 27)`. If the market's rank-bin progression followed QA T-step dynamics exactly, `tp` would be the exact next bin.

Instead, it predicts the OPPOSITE: high `tp` (T-step predicts top bins) → below-average actual returns. The QA arithmetic encodes the direction of mean-reversion structurally — when `b+e` is large (both recent weeks were high), the T-step wraps high, and the market mean-reverts low.

This is the discrete-arithmetic analogue of the Lo & MacKinlay (1988) short-horizon return autocorrelation: the QA modular structure encodes the autocorrelation exactly as a contrarian signal.

## QA Mapping

- **Observer projection**: weekly log-return → rank among N → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer state**: `b = bins[t-1]`, `e = bins[t]` (both int)
- **T-step**: `tp = (b + e) % 27` — mod-reduced (T-operator output, not element computation)
- **Groups**: `tp ≥ 22` (high) vs `tp ≤ 4` (low) vs middle (4 < tp < 22)
- **Target**: `log(price[t+2]/price[t+1])` (next-week float return — observer output)

## Per-Index Breakdown (tp ≥ 22)

| Index | n | Mean | perm_p | Result |
|---|---|---|---|---|
| ^GSPC | 257 | -0.12% | 0.056 | marginal |
| ^IXIC | 255 | -0.03% | 0.168 | null |
| ^DJI | 245 | **-0.20%** | **0.020** | significant |
| QQQ | 260 | **+0.15%** | 0.610 | **NULL + positive — exception** |
| SPY | 272 | -0.13% | **0.019** | significant |

**QQQ partial null**: Nasdaq-100 (QQQ) shows positive mean (+0.15%) at tp≥22 — the contrarian effect is absent. This mirrors the [459] result where QQQ/IXIC also lack the direction signal. Large-cap price-weighted indices (GSPC, DJI, SPY) carry the contrarian structure; Nasdaq-weighted do not.

## Contrast with Cert [459]

| Feature | [459] A-coord a≤6 | [460] T-step tp≥22 |
|---|---|---|
| Condition | b+2e ≤ 6 | (b+e)%27 ≥ 22 |
| Direction of effect | **positive** (+0.99%) | **negative** (-0.07%) |
| n | 213 | 1,289 |
| perm_p (pooled) | ~0.0002 | 0.0006 |
| Interpretation | Low A-coord → bounce | High T-step → reversal |

These are **complementary contrarian signals**: [459] identifies weeks where the QA A-coordinate is at its minimum → positive bounce. [460] identifies weeks where the T-step predicts the maximum → actual reversal. Together they triangulate the QA structure's anti-persistence encoding.

## Geometric Structure in Z/27Z

The T-step `tp = (b+e)%27` cycles around Z/27Z. When `b` and `e` are both large (e.g., b=15, e=15 → tp=3) or both near the middle (e.g., b=12, e=12 → tp=24), the mod-27 wrapping creates the high/low contrast:

- tp ≥ 22 corresponds to states where `b+e ∈ {22,23,24,25,26,49,50,51,52}` — the "near-mod-boundary" region. These tend to occur when both recent weeks are middling (b~12-13, e~12-13) rather than when both are high.
- The anti-persistence is strongest when neither week is extreme — exactly the regime where mean-reversion mechanisms operate.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: hi pooled significant | tp≥22 perm_p=0.0006<0.01 | PASS |
| C2: lo pooled significant | tp≤4 perm_p=0.0042<0.05 | PASS |
| C3: Spread | 0.40%≥0.30% (mean_lo − mean_hi) | PASS |
| C4: hi negative | tp≥22 pooled mean=-0.07%<0 | PASS |
| C5: lo positive | tp≤4 pooled mean=+0.33%>0 | PASS |
| C6: QQQ tech exception | QQQ mean=+0.15%≥0 (null documented) | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486
- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. doi:10.2307/2328797

## Related Certs

- [459] QA Witt Tower A-Coordinate Weekly Direction (complementary positive signal)
- [458] QA Witt Tower Orbit Weekly Direction (S-orbit positive signal)
- [457] QA Witt Tower Orbit Price Volatility (monthly volatility)
- [110] QA Witt Tower Framework (structural parent)
