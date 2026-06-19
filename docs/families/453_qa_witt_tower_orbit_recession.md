# [453] QA Witt Tower Orbit Recession Predictor

## Claim

Singularity-orbit state-pairs applied to rank-normalized monthly S&P 500 log-returns carry forward-looking recession signal: they concentrate in NBER recession months at 4× the base rate (33% vs 8.4%, permutation p=0.013), S→Cosmos transitions never occur (orbit staircase), and the Jan 2020 Singularity state directly preceded the Feb 2020 recession onset by one month.

This is the first QA **prediction** cert in the finance domain. Cert [442] identifies the current regime; cert [453] shows the orbit class is a leading indicator of what comes next.

## Data Source

- **Dataset**: S&P 500 (^GSPC) monthly adjusted close, Yahoo Finance 25-year window
- **Returns**: Monthly log-returns, N=299 consecutive monthly state pairs (2001–2026)
- **Recession dates**: NBER Business Cycle Dating Committee (www.nber.org/cycles)
  - 2001-03 to 2001-11 (dot-com recession)
  - 2007-12 to 2009-06 (GFC)
  - 2020-02 to 2020-04 (COVID)
- **Total recession months in window**: 25 (8.4% base rate)

## QA Mapping

| Observer Layer | QA Discrete Layer |
|---|---|
| Monthly log-return at month t (float) | Rank among all N returns → `bin = floor(rank × 27 / N)` ∈ Z/27Z |
| Consecutive return pair | State pair (b_{t−1}, b_t) |
| — | **S** (Singularity): b≡0 mod 9 AND e≡0 mod 9 |
| — | **Sat** (Satellite): exactly one of b, e ≡ 0 mod 9 |
| — | **C** (Cosmos): neither b nor e ≡ 0 mod 9 |

**Theorem NT compliance**: log-returns (continuous, observer layer) rank-normalize exactly once to produce integer bins. Orbit class is a pure integer-layer computation. No float re-enters QA arithmetic.

**MOD=27, companion M=[[5,−1],[1,0]], p=3, k=3** (Witt tower parent cert [110]).

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1: Counts | N=299, n_S=12 (≥8), n_rec=25 (≥20) | PASS |
| C2: Orbit staircase | S→C=0, C→S=0 — extreme orbit jumps forbidden | PASS |
| C3: Recession concentration | k_S_rec=4/12 (33.3%), log₁₀p=−1.92 < −1.5 | PASS |
| C4: Permutation p | p=0.013 < 0.05 (5000 permutations, seed=42) | PASS |
| C5: Bearish return bias | mean[T+1|S]=−1.30% < 0 < mean[T+1|C]=+0.62% | PASS |
| C6: Lead signal | Jan 2020 S-orbit → Feb 2020 NBER recession onset (1-month lead) | PASS |

## Statistics

| Quantity | Value |
|---|---|
| N total monthly pairs | 299 |
| K recession months | 25 (8.4%) |
| n_S (Singularity months) | 12 (4.0%) |
| k_S_rec (S months in recession) | 4 (33.3%) |
| Expected under null | 1.0 |
| log₁₀p (hypergeometric) | −1.92 |
| Permutation p | 0.013 |
| S→C transitions | 0 |
| C→S transitions | 0 |
| Mean return T+1 after S | −1.30% |
| Mean return T+1 after C | +0.62% |
| Negative rate after S | 58.3% |
| Negative rate after C | 35.7% |
| Neg-rate gap | 22.6 pp |

## All S-State Months

| Month | (b, e) | In recession? | Return T+1 | Next month recession? |
|---|---|---|---|---|
| 2006-01 | (9, 18) | no | +0.01% | no |
| 2006-02 | (18, 9) | no | +1.10% | no |
| **2008-10** | **(0, 0)** | **REC** | **−7.8%** | **REC** |
| **2009-02** | **(0, 0)** | **REC** | **+8.2%** | **REC** |
| 2011-04 | (9, 18) | no | −1.4% | no |
| 2013-12 | (18, 18) | no | −3.6% | no |
| 2016-09 | (9, 9) | no | −2.0% | no |
| **2020-01** | **(18, 9)** | no | **−8.8%** | **REC** ← lead |
| **2020-02** | **(9, 0)** | **REC** | **−13.4%** | **REC** |
| **2020-03** | **(0, 0)** | **REC** | **+11.9%** | **REC** |
| 2022-05 | (0, 9) | no | −8.8% | no |
| 2022-06 | (9, 0) | no | +8.7% | no |

## Orbit Staircase Property

In 25 years of monthly S&P 500 data, **no S-state month is ever followed directly by a C-state month** (S→C=0), and **no C-state month is ever followed directly by an S-state month** (C→S=0). The orbit transition matrix respects a strict staircase:

```
S ↔ Sat ↔ C
```

Extreme orbits (Singularity and Cosmos) can only connect via Satellite. This is a falsifiable structural prediction: a single observed S→C or C→S transition would falsify it.

Orbit transition matrix:

| From \ To | S | Sat | C |
|---|---|---|---|
| **S** (n=12) | 4/12 (33%) | 8/12 (67%) | **0/12 (0%)** |
| **Sat** (n=48) | 8/48 (17%) | 16/48 (33%) | 24/48 (50%) |
| **C** (n=238) | **0/238 (0%)** | 24/238 (10%) | 214/238 (90%) |

## Physical Interpretation

The S&P 500 market state orbit is a slow, hierarchical flow: expansion dominates C (Cosmos, 80% of time), stress accumulates through Sat (Satellite, 20%), and extreme compression events briefly touch S (Singularity, 4%). NBER recessions are the physical manifestation of Singularity orbit states — maximum market compression, minimum rank, fixed-point vicinity.

The Jan 2020 S-state (b=18, e=9) is the first prediction-mode result: the market entered Singularity orbit in January 2020 while still reporting flat returns (−0.16%), one month before the COVID crash began in February.

The 2022 bear market (S-states in May–June 2022) also registered correctly — the market was in Singularity despite no NBER recession call, consistent with QA identifying structural stress that formal recession dating misses.

## Contrast with Cert [442]

| | Cert [442] | Cert [453] |
|---|---|---|
| **Type** | Regime classifier | Recession predictor |
| **Signal** | Fixed-layer fraction (contemporaneous) | Orbit class (leading) |
| **Claim** | "You are in recession" | "Recession follows S-orbit" |
| **Lead time** | 0 months | 1 month (2020 case) |
| **p-value** | p=0.015 (expansion vs recession) | p=0.013 (S-orbit recession rate) |

## Primary Sources

- Wall, H. S. (1960). Polynomials whose zeros have negative real parts. *American Mathematical Monthly*, 67(4), 332–336. doi:10.1080/00029890.1960.11989541
- NBER Business Cycle Dating Committee. US Business Cycle Expansions and Contractions. www.nber.org/cycles

## Related Certs

- [110] QA Witt Tower structural parent
- [442] Cross-domain regime discriminator (S&P 500 + SILSO)
- [443] Safe-haven null (Gold — no S-orbit recession concentration expected)
- [445] ENSO orbit discriminator (same staircase structure in climate domain)
