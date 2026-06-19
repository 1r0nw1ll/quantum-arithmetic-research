# [454] QA Witt Tower Orbit Recession Null (Gold)

## Claim

Certified null for cert [453]: Gold futures (GC=F) S-orbit months do NOT concentrate in NBER recession months (0/2 = 0%, perm p=1.000). The recession concentration signal from cert [453] is risk-asset specific. Safe havens (Gold, TLT) register null; risk assets (GSPC, QQQ) register signal. The orbit staircase (S→C=0, C→S=0) is universal across all four assets.

## Cross-Asset Pattern

| Asset | Type | n_S | k_S_rec | % | perm p | Verdict |
|---|---|---|---|---|---|---|
| ^GSPC | Risk (large cap) | 12 | 4 | 33% | 0.013 | SIGNAL |
| QQQ | Risk (tech) | 6 | 4 | 67% | 0.0006 | SIGNAL |
| GC=F | Safe haven (gold) | 2 | 0 | 0% | 1.000 | NULL |
| TLT | Safe haven (bonds) | 1 | 0 | 0% | 1.000 | NULL |

Base recession rate (NBER, 2001–2024): ~8.4%.

## Gold S-State Months

Both Gold S-state months fall **after** the last NBER recession trough (April 2020):

| Month | (b, e) | Recession? | Next return |
|---|---|---|---|
| **2020-06** | (18, 18) | no (post-recession) | +9.1% |
| **2021-04** | (9, 18) | no (post-recession) | +7.4% |

Gold S-states signal post-stress recovery rallies, not crash onset — the opposite directional interpretation from GSPC S-states.

## Return Bias Inversion

| | Gold | S&P 500 ([453]) |
|---|---|---|
| mean_ret_after_S | **+8.2%** | **−1.3%** |
| neg_rate_S | 0.0% | 58.3% |
| mean_ret_after_C | +0.9% | +0.6% |

## QQ Positive Control

QQQ (Nasdaq-100) shows the strongest S-orbit recession signal of any asset tested:
- 4/6 S-months in recession (67%)
- log₁₀p = −3.29
- perm p = 0.0006

This confirms the mechanism is not an GSPC artifact — it strengthens across tech-dominated risk assets.

## Orbit Staircase (Universal Property)

The S→C=0, C→S=0 constraint holds for **all four assets** independently. Gold S→C=0 and Gold C→S=0. This confirms the orbit staircase is a structural property of how monthly returns traverse orbit space, independent of asset class. The recession concentration is what varies.

## Physical Interpretation

During market stress, risk assets (equities) collapse in price → log-returns rank at the very bottom → both b and e land in {0,9,18} (the lowest bins) → Singularity orbit. The ranking reflects the stress compression.

Gold behaves oppositely: during stress, gold rallies → log-returns rank high → bins land in the upper third → Cosmos orbit. Gold's Singularity states only appear in the post-recession period when the gold rally peak is normalised out by the full-sample ranking, moving those months to lower relative bins.

This is an observer-projection effect: the same QA orbit class (S = lowest bin pair) means "extreme compression" in equities and "plateau after a rally" in gold, because the physical meaning of the rank is asset-specific.

## Data Source

- **Gold**: GC=F monthly adjusted close, Yahoo Finance 25-year window (N=255 pairs)
- **NBER recessions**: 2001-03/11, 2007-12/2009-06, 2020-02/04
- **QQQ positive control**: Yahoo Finance 25-year window (N=299 pairs)

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: Gold sample | N=255 (≥200), n_rec=20 (≥15) | PASS |
| C2: Orbit staircase | Gold S→C=0, C→S=0 | PASS |
| C3: Gold null | k_S_rec=0/2, perm p=1.000 (≥0.20) | PASS |
| C4: Risk vs safe-haven | Gold p=1.000 (null) vs GSPC p=0.013 (signal) | PASS |
| C5: Direction inversion | Gold mean_ret_after_S=+8.2% > 0 (vs GSPC −1.3%) | PASS |
| C6: QQQ positive control | 4/6 (67%), log₁₀p=−3.29, perm p=0.0006 | PASS |

## Primary Sources

- Wall, H. S. (1960). doi:10.1080/00029890.1960.11989541
- NBER Business Cycle Dating Committee. www.nber.org/cycles

## Related Certs

- [443] QA Witt Tower Safe-Haven Null (fixed-layer version, same Gold asset)
- [453] QA Witt Tower Orbit Recession Predictor (GSPC signal cert)
- [110] QA Witt Tower structural parent
- [442] Cross-domain regime discriminator
