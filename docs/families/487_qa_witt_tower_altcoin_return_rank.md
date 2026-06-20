# Cert [487]: QA Witt Tower Altcoin Return-Rank Crash Reversion Scope

**Family ID**: 487
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_altcoin_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 crash-reversion operator extends to the full crypto asset class. All four altcoins tested are individually significant:

| Asset | Type | Vol/day | n_signal | Excess | perm_p |
|-------|------|---------|---------|--------|--------|
| SOL | L1 blockchain | 6.21% | 60 | +2.094% | 0.004 |
| BNB | Exchange token | 4.85% | 84 | +1.966% | 0.0002 |
| ADA | L1 blockchain | 5.89% | 89 | +1.344% | 0.023 |
| DOGE | Meme token | 6.67% | 96 | +1.430% | 0.027 |

4/4 positive, 4/4 significant (perm_p < 0.05). Pooled excess: **+1.709%/day**, pooled perm_p = 0.0 (0/5000 null shuffles exceeded).

**ALL altcoin excesses exceed BTC certified (+0.847%)** — consistent with BTC having the deepest institutional order books, which dampen the bounce through faster arbitrage. Less liquid assets show stronger crash-reversion.

**DOGE significant**: The social/meme token also shows crash-reversion (p=0.027), confirming the gate is not "institutional crypto" but daily-scale mean-reverting market microstructure.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | n_positive ≥ 3 | PASS | 4/4 |
| C2 | pooled excess > 0.10%/day | PASS | +1.709% |
| C3 | n_significant ≥ 3 (p < 0.05) | PASS | 4/4 |
| C4 | max excess > 0.50%/day | PASS | SOL +2.094% |
| C5 | min excess > max non-crypto (0.075%) | PASS | ADA +1.344% |
| C6 | pooled perm_p < 0.01 | PASS | 0.0 |

## Magnitude Hierarchy (across cert chain)

| Asset | Excess/day | Ratio vs equity |
|-------|-----------|----------------|
| SOL | +2.094% | 5.4× |
| BNB | +1.966% | 5.1× |
| ETH (cert [482]) | +1.771% | 4.6× |
| DOGE | +1.430% | 3.7× |
| ADA | +1.344% | 3.5× |
| BTC (cert [482]) | +0.847% | 2.2× |
| Equity (cert [488]) | +0.385% | 1.0× |
| GLD (cert [486]) | +0.075% | 0.2× |
| USO (cert [486]) | −0.215% | negative |

## QA Structure

- **Operator**: return-rank bins `b=floor(rank(rets[t])×27/N)`, `e=floor(rank(rets[t+1])×27/N)`
- **Signal**: `a = b + 2e ≤ 6` (A2 derived, raw)
- **Target**: `rets[t+2]` (no look-ahead)
- **Permutation test**: N=5000 shuffles, seed=42

## Primary Sources

- Fama EF (1970). Efficient capital markets. *Journal of Finance* 25(2):383-417. doi:10.2307/2325486
- Nakamoto S (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [482]: BTC/ETH return-rank crash-reversion (operator definition)
- Cert [486]: Cross-asset scope (non-crypto 4/4 NULL)
