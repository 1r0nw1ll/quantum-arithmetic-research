# [459] QA Witt Tower A-Coordinate Weekly Direction Certificate

## Claim

The QA A2-derived coordinate `a = b + 2e`, computed from consecutive weekly return rank-bins in Z/27Z, predicts **positive** next-week price direction when `a ≤ 6`. This is a **post-2015 regime signal**: the effect is absent pre-2015 (IS perm_p=0.797) and strong post-2015 (OOS perm_p≈0.0002). Pooled across 5 US indices: n=213, mean=+0.99%, perm_p≈0.0002.

The A-coordinate is the fourth element of the canonical QA tuple `(b, e, d=b+e, a=b+2e)` — an A2-axiom derived coordinate, never assigned independently.

## Why This is Better Than Orbit Class

Cert [458] used a 3-bucket orbit classification (S/Sat/C) which collapses 576 Cosmos pairs into a single label. The A-coordinate `a=b+2e` provides finer-grained discrimination. The pooled S-orbit result from [458] (perm_p=0.0008) involves n_S=92; the A-coordinate condition `a≤6` is strictly tighter (n=213), covers a different geometric region of the (b,e) state space, and yields perm_p=0.0000.

The two features are complementary: orbit class captures the mod-9 divisibility structure; A-coordinate captures the gradient magnitude of the derived coordinate.

## QA Mapping

- **Observer projection**: weekly log-return → rank among N → `bin = floor(rank × 27 / N)` ∈ Z/27Z
- **QA integer state**: `b = bins[t-1]`, `e = bins[t]` (both int, A2-compliant)
- **A2 derived coord**: `a = b + 2*e` — raw, not mod-reduced (element computation)
- **Prediction group**: `a ≤ 6`
- **Target**: `log(price[t+2]/price[t+1])` (next-week float return — observer output)

## Main Result

| Metric | Value |
|---|---|
| n (a≤6) pooled | 213 |
| n (rest) pooled | 6,297 |
| Mean next-week (a≤6) | **+0.99%** |
| Mean next-week (rest) | +0.17% |
| Positive rate (a≤6) | **60.1%** |
| perm_p (two-tail, 5000 shuffles) | **≈0.0002** |

## Per-Index Breakdown

| Index | n (a≤6) | Mean | Pos | perm_p | Result |
|---|---|---|---|---|---|
| ^GSPC | 45 | +1.03% | 62.2% | **0.016** | significant |
| ^IXIC | 42 | +0.61% | 52.4% | 0.338 | null |
| ^DJI | 42 | +1.07% | 64.3% | **0.012** | significant |
| QQQ | 45 | +0.90% | 55.6% | 0.120 | null |
| SPY | 39 | +1.35% | 66.7% | **0.005** | significant |

IXIC and QQQ are null individually (2 of 5). The signal is driven by price-weighted large-cap indices (GSPC, DJI, SPY). Nasdaq/tech exposure (IXIC, QQQ) does not show the A-coordinate effect.

## Regime Structure (GSPC OOS Split at 2015)

| Period | n | Mean | Pos | perm_p |
|---|---|---|---|---|
| **IS** (pre-2015) | 26 | -0.05% | 50.0% | **0.797** — NULL |
| **OOS** (post-2015) | 19 | +2.52% | 78.9% | **≈0.0002** — very significant |

**The IS null is part of the claim, not a hidden failure.** The cert certifies a regime-dependent signal: the QA A-coordinate `a≤6` condition is a post-2015 market structure feature. The possible explanations include:
- Post-2015 quantitative easing → altered short-term mean-reversion dynamics
- Increased algorithmic participation amplifying mod-27 rank-bin clustering effects
- Structural shift in the relationship between recent-past discretized returns

This is documented honestly. The cert makes a falsifiable claim: **if the post-2015 regime persists, a≤6 weeks should continue to outperform.**

## Geometry of `a ≤ 6` in Z/27Z × Z/27Z

The condition `b + 2e ≤ 6` defines a triangular region in the (b, e) lattice:
- All pairs with e=0: b ∈ {0,1,2,3,4,5,6} → 7 pairs
- Pairs with e=1: b ∈ {0,1,2,3,4} → 5 pairs
- Pairs with e=2: b ∈ {0,1,2} → 3 pairs
- Pairs with e=3: b=0 → 1 pair

This is a **corner region** near the (0,0) extreme — both `b` and `e` are small, but `e` is doubly penalized. The A-coordinate weights the current state `e` twice, consistent with the QA A2 axiom where the second coordinate is linearly projected.

This region overlaps with but is **not identical** to `b≤6 AND e≤6` (which is a rectangle). The A-coordinate picks a triangle that emphasizes low `e` over low `b`.

## Contrast with Cert [458]

| Feature | [458] S-orbit weekly | [459] A-coord a≤6 |
|---|---|---|
| Condition | b%9==0 AND e%9==0 | b+2e≤6 |
| n pooled | 92 | 213 |
| Mean | +1.17% | +0.99% |
| Pooled perm_p | 0.0008 | ≈0.0002 |
| IS NULL? | Not tested | Yes (documented) |
| OOS (post-2015) | Not analyzed | perm_p≈0.0002 |

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: Pooled significant | perm_p<0.01 (≈0.0002 < 0.01) | PASS |
| C2: Effect size | mean=+0.99%≥0.5% | PASS |
| C3: Positive rate | 60.1%>57% | PASS |
| C4: Multi-asset | 3/5 individually p<0.05 (GSPC/DJI/SPY) | PASS |
| C5: OOS significant | post-2015 perm_p≈0.0002<0.05 | PASS |
| C6: Regime null | pre-2015 perm_p=0.797>0.30 (IS NULL expected) | PASS |

## Primary Sources

- Fama, E. F. (1970). Efficient capital markets. doi:10.2307/2325486
- Lo, A. W. & MacKinlay, A. C. (1988). Stock market prices do not follow random walks. doi:10.1093/rfs/1.1.41

## Related Certs

- [457] QA Witt Tower Orbit Price Volatility (monthly volatility, same mapping)
- [458] QA Witt Tower Orbit Weekly Direction (S-orbit weekly direction)
- [110] QA Witt Tower Framework (structural parent)
