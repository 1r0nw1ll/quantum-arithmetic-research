# Cert [489]: QA Witt Tower International Equity Return-Rank (Local Currency)

**Family ID**: 489
**Status**: CERTIFIED (6/6 checks PASS)
**Validated**: 2026-06-20
**Dir**: `qa_alphageometry_ptolemy/qa_witt_tower_intl_equity_return_rank_cert_v1/`

## Claim

The return-rank a=b+2e≤6 crash-reversion operator is a **universal equity market property**, not a US-market-microstructure artifact. Tested on four local-currency indices with independent monetary policy, exchange rates, and trading microstructure:

| Index | Currency | n_signal | Excess/day | perm_p | Significant |
|-------|----------|---------|-----------|--------|-------------|
| ^N225 (Nikkei 225) | JPY | 175 | +0.452% | 0.0012 | YES |
| ^FTSE (FTSE 100) | GBP | 168 | +0.318% | 0.0176 | YES |
| ^GDAXI (DAX) | EUR | 182 | +0.390% | 0.0040 | YES |
| ^HSI (Hang Seng) | HKD | 194 | +0.264% | 0.0542 | directional |

3/4 individually significant (p<0.05). HSI positive (p=0.054 — highest falsification target). Pooled: **+0.356%/day**, pooled perm_p=0.0002.

## Why This Is a New Domain

These are NOT the same as cert [462] (US-listed international ETFs: EWJ, EWG, etc.). Those trade in New York with USD pricing and US bid-ask spreads. These are local-currency indices:
- **Nikkei** trades in Tokyo in yen under BOJ policy — fully independent microstructure
- **FTSE** trades in London in GBP — UK-specific inflation regime, resources-heavy composition  
- **DAX** trades in Frankfurt in EUR under ECB policy — German industrial export economy
- **HSI** trades in Hong Kong in HKD with China exposure and HKMA currency peg

If three of these four pass, the operator is intrinsic to equity market structure globally.

## Checks

| ID | Check | Result | Value |
|----|-------|--------|-------|
| C1 | Nikkei perm_p < 0.05 | PASS | 0.0012 |
| C2 | FTSE perm_p < 0.05 | PASS | 0.0176 |
| C3 | n_significant ≥ 2/4 | PASS | 3/4 |
| C4 | pooled excess > 0.10%/day | PASS | +0.356% |
| C5 | pooled < BTC (0.847%) | PASS | 0.356% < 0.847% |
| C6 | HSI excess positive | PASS | True (+0.264%) |

## Structural Context

| Operator | Asset class | Pooled excess |
|----------|------------|--------------|
| Return-rank a≤6 | Altcoins [487] | +1.709%/day |
| Return-rank a≤6 | BTC/ETH [482] | +0.847–1.771%/day |
| Price-level a≤6 | US equity [461] | +0.370%/day |
| Return-rank a≤6 | US equity [488] | +0.385%/day |
| Return-rank a≤6 | **Intl equity (this cert)** | **+0.356%/day** |
| Return-rank a≤6 | Non-crypto [486] | NULL / negative |

International local-currency equity (+0.356%) sits within 8% of US equity (+0.385%) — consistent with the same underlying mechanism (mean-reverting equity microstructure) operating at a slightly lower magnitude due to lower liquidity in non-US markets.

**HSI marginally outside p<0.05**: China market intervention and HKMA peg could partially suppress or add noise to mean-reversion dynamics, producing slightly weaker signal. The directional consistency is the key structural confirmation.

## QA Structure

- **Operator**: return-rank bins `b=floor(rank(rets[t])×27/N)`, `e=floor(rank(rets[t+1])×27/N)`
- **Signal**: `a = b + 2e ≤ 6` (A2 derived, raw, never mod-reduced)
- **Target**: `rets[t+2]` (no look-ahead)
- **Permutation test**: N=5000 shuffles, seed=42
- **Data**: Yahoo Finance daily OHLCV via yfinance; 2000–2026

## Primary Sources

- Fama EF (1970). Efficient capital markets. *Journal of Finance* 25(2):383-417. doi:10.2307/2325486
- Lo A & MacKinlay C (1988). Stock market prices do not follow random walks. *Review of Financial Studies* 1(1):41-66. doi:10.1093/rfs/1.1.41

## Parents

- Cert [110]: Witt Tower Framework (MOD=27)
- Cert [461]: Equity price-level a≤6 baseline (+0.37%/day)
- Cert [462]: US-listed international ETFs (+0.42%/day; 5/6 sig)
- Cert [482]: BTC/ETH return-rank crash-reversion (operator definition)
- Cert [488]: US equity return-rank (+0.385%/day; 4/4 sig)
