# [443] QA Witt Tower Safe-Haven Null

**Status**: CERTIFIED 2026-06-18  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_safe_haven_null_cert_v1/`  
**Parent cert**: [439] QA Witt Tower General v_p Period Law  
**Positive control**: [442] QA Witt Tower Cross-Domain Regime Discriminator

## Claim

The Witt tower filter bank (cert [439]) produces a certified null for safe-haven assets
(Gold): recession fixed-layer elevation is absent because safe-haven crisis returns do not
cluster at the fixed-point states of the companion matrix.

## Companion

`M = [[5,-1],[1,0]]`, `p = 3`, `r = 1` (det=+1, cert [439]), tower level `k = 3` (mod 27).

## Certified Facts

| # | Claim | Result |
|---|---|---|
| C1 | Gold birth fraction ∈ [55%, 75%] | 68.6% ✓ |
| C2 | Gold recession fixed_frac ≤ 2% | 0.0% ✓ |
| C3 | Permutation null p ≥ 0.15 (null NOT rejected) | p=1.000 ✓ |
| C4 | Fixed-point locus = {(0,0),(9,9),(18,18)} | exact match ✓ |
| C5 | GFC mean rank bin > 12 (above mid of Z/27Z) | 14.2 ✓ |

## Geometric Ground Truth (C4)

The companion `M = [[5,-1],[1,0]]` mod 27 has exactly three fixed-point states:

```
{(0,0), (9,9), (18,18)}
```

**Proof**: `M·(b,e) = (5b−e, b)`. Fixed-point condition:
- `b ≡ e mod 27`
- `5b − e ≡ b mod 27` → `4b ≡ 0 mod 27` → `b ≡ 0 mod 9`

Only three solutions in Z/27Z: b∈{0,9,18}, e=b. Verified exhaustively over all 729 states.

## Why the Null is Geometrically Necessary

Fixed-layer elevation requires consecutive return pairs at rank bins ≡ 0 mod 9:
bins {0, 9, 18} in Z/27Z.

| Asset type | Crisis behavior | Rank bins | Fixed-layer? |
|---|---|---|---|
| GSPC (cert [442]) | Consecutive large losses | 0–3 (near bin 0) | **Yes** (9.7%) |
| Gold (cert [443]) | Rally or oscillation | 13–26 (upper half) | **No** (0.0%) |

Gold's GFC states have mean rank bin 14.2 — above the midpoint of Z/27Z — confirming
gold state pairs are systematically in the upper range, away from the three fixed-point bins.

## Permutation Test (C3): p = 1.000

The observed recession vs expansion delta is 0.4pp (recession 0.0%, expansion 0.4%).
In the entire pool of 255 states only ~1 reaches period\_val = 0.
Any permutation split of 20 recession / 235 expansion states produces the same or higher
delta by chance: p = 1.000. This is the maximum possible null — the filter bank has zero
signal on Gold across 25 years and all three NBER recessions.

## Domain — Gold Monthly Returns (~2000–2026)

Encoding: rank-normalized log-returns → Z/27Z; state `(rank[t], rank[t-1])`

| Regime | Fixed-layer | n months |
|---|---|---|
| Expansion (NBER) | 0.4% | 235 |
| Recession (NBER) | 0.0% | 20 |
| GFC only (2007-12/2009-06) | 0.0% | 16 |

NBER recessions: 2001-03/11 (dot-com), 2007-12/2009-06 (GFC), 2020-02/04 (COVID).

## Theorem NT Compliance

Log-returns → rank bins is a one-way observer projection. The filter bank classifies;
output never re-enters the QA layer as a causal input.

## Contrast with Cert [442]

| Instrument | Recession fixed-layer | Δ vs expansion | Perm p | Verdict |
|---|---|---|---|---|
| GSPC (cert [442]) | 9.7% | +8.8pp | 0.015 | Elevated |
| Gold (cert [443]) | 0.0% | −0.4pp | 1.000 | **Null** |

The same filter bank on the same recession labels produces opposite outcomes —
confirming the discriminator is sensitive to orbit location, not just the recession label.

## Related Certs

- [439] QA Witt Tower General v_p Period Law (parent — defines filter bank)
- [442] QA Witt Tower Cross-Domain Regime Discriminator (positive control)
- Scripts: `62_qa_witt_multi_instrument.py`, `63_qa_witt_rolling_window.py`
