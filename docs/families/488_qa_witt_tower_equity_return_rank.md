# Cert [488]: QA Witt Tower Equity Return-Rank Crash Reversion

**Family ID**: 488
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_equity_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 crash-reversion operator **also works for US equity indices**, revising the scope picture from cert [486]. The boundary is not equity/crypto but magnitude.

| Asset | n_signal | Excess/day | perm_p |
|-------|---------|-----------|--------|
| SPY | 203 | +0.382% | 0.0000 |
| QQQ | 210 | +0.452% | 0.0002 |
| GSPC | 196 | +0.468% | 0.0000 |
| IXIC | 214 | +0.238% | 0.0088 |

4/4 individually significant. Pooled excess: **+0.385%/day**.

## Key Structural Finding

The scope constraint on return-rank crash-reversion is **MAGNITUDE**, not asset class:

| Asset class | Pooled excess | Mechanism |
|-------------|--------------|-----------|
| Altcoins (cert [487]) | +1.709%/day | Thin order books; strong microstructure bounce |
| BTC/ETH (cert [482]) | +0.847–1.771%/day | Deep but volatile; institutional bounce |
| Equity (this cert) | +0.385%/day | Mean-reverting; institutional arbitrage dampens |
| Non-crypto (cert [486]) | NULL or negative | No mean-reverting microstructure at daily scale |

**Operator equivalence**: Return-rank equity excess (+0.385%) ≈ price-level equity excess (+0.370%, cert [461]). For equities both operators are equivalent because equity returns are already stationary (no secular price trend). The return-rank operator only adds value over price-level for trending assets (crypto), where price-level rank is non-stationary.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | SPY perm_p < 0.01 | PASS | 0.0000 |
| C2 | GSPC perm_p < 0.01 | PASS | 0.0000 |
| C3 | n_significant ≥ 3 | PASS | 4/4 |
| C4 | pooled excess > 0.20%/day | PASS | +0.385% |
| C5 | equity weaker than BTC (0.847%) | PASS | 0.385% < 0.847% |
| C6 | return-rank parity with price-level (<50% diff) | PASS | 4% difference |

## QA Structure

- **Operator**: return-rank bins, same as cert [482]
- **Signal**: `a = b + 2e ≤ 6` (A2 derived, raw)
- **Target**: `rets[t+2]` (no look-ahead)
- **Note on ^DJI**: yfinance data unavailable for this ticker; SPY/QQQ/GSPC/IXIC cover large-cap/tech/broad-market sufficiently

## Primary Sources

- Fama EF (1970). Efficient capital markets. *Journal of Finance* 25(2):383-417. doi:10.2307/2325486
- Lo A & MacKinlay C (1988). Stock market prices do not follow random walks. *Review of Financial Studies* 1(1):41-66. doi:10.1093/rfs/1.1.41

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [461]: Equity price-level a≤6 baseline (+0.37%/day)
- Cert [482]: BTC/ETH return-rank crash-reversion (operator)
- Cert [486]: Non-crypto scope (GLD/FX/oil NULL)
- Cert [487]: Altcoin scope (crypto class confirmed)
