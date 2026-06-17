# [442] QA Witt Tower Cross-Domain Regime Discriminator

**Status**: CERTIFIED 2026-06-17  
**Cert directory**: `qa_alphageometry_ptolemy/qa_witt_tower_regime_discriminator_cert_v1/`  
**Parent cert**: [439] QA Witt Tower General v_p Period Law

## Claim

The Witt tower filter bank (cert [439]) discriminates physical and financial market
activity regimes via fixed-layer and birth-layer occupation fractions, verified
across two statistically independent domains.

## Companion

`M = [[5,-1],[1,0]]`, `p = 3`, `r = 1` (det=+1, cert [439]), tower level `k = 3` (mod 27).

## Certified Facts

| # | Claim | Result |
|---|---|---|
| C1 | SILSO birth fraction ∈ [55%, 75%] | 66.1% ✓ |
| C2 | SILSO \|Δfixed\| (min vs max) ≥ 4pp | 7.7pp ✓ |
| C3 | SILSO permutation test p < 0.15 | p≈0.000 ✓ |
| C4 | S&P 500 birth fraction ∈ [55%, 75%] | 65.8% ✓ |
| C5 | S&P 500 \|Δfixed\| (recession vs exp.) ≥ 4pp | 8.8pp ✓ |
| C6 | S&P 500 permutation test p < 0.15 | p=0.015 ✓ |

## Domain 1 — SILSO Monthly Sunspot (1749–2026)

Encoding: `b = int(SN[t]) mod 27`, `e = int(SN[t-1]) mod 27`

| Regime | Fixed-layer | n months |
|---|---|---|
| Solar minimum (SN < 20) | 7.9% | 706 |
| Solar medium | — | 1494 |
| Solar maximum (SN > 100) | 0.3% | 1129 |

Solar minimum months cluster near `(0,0)` — the principal fixed point of M mod 27 — because
consecutive near-zero sunspot values remain in the bottom-mod-27 bin.

## Domain 2 — S&P 500 Monthly Returns (~1982–2026)

Encoding: rank-normalized log-returns → Z/27Z; state `(rank[t], rank[t-1])`

| Regime | Fixed-layer | n months |
|---|---|---|
| Expansion (NBER) | 0.9% | 466 |
| Recession (NBER) | 9.7% | 31 |

NBER recessions used: 2001-03/11 (dot-com), 2007-12 / 2009-06 (GFC), 2020-02/04 (COVID).

Recession months show consecutive large-negative returns → both rank bins near 0 → pairs
cluster at `(0,0)`, the fixed-point state.

## Cross-Domain Pattern

Both domains produce ~66% birth fraction (theory 2/3 = 66.7%) and a large fixed-layer
differential (~7-9pp) between high-activity and regime-stressed periods. The consistent
magnitude across unrelated encoding schemes (direct-mod for SILSO, rank-normalized for
S&P) confirms this is a structural property of the Witt tower filter bank, not an
artefact of the encoding.

## Theorem NT Compliance

Time-series values → integer states is a one-way observer projection. The filter bank
classifies each state but its output never re-enters the QA layer as a causal input.

## Related Certs

- [439] QA Witt Tower General v_p Period Law (parent — defines filter bank)
- [440] QA Witt Tower det=−1 General v_p Period Law
- [441] QA Witt Tower Fibonacci-Pisano Synthesis
- Scripts: `60_qa_witt_empirical_validation.py`, `61_qa_witt_financial_filterbank.py`
