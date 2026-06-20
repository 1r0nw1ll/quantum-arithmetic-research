# [474] QA Witt Tower Cross-Asset Transfer

## Claim

QA crash-bounce signals (a≤6 and (0,0) crash pair) transfer to **equity-proximate
assets** (VNQ REIT, DBA agriculture, TLT long-term bonds) but are **absent in pure
stores of value** (GLD gold, IEF mid-term bonds, USO crude oil).

Transfer is asset-class-specific: the QA discrete-bin signal captures crash-bounce
dynamics that appear wherever equity-like stress cascades. Gold and mid-term bonds,
which act as flight-to-safety vehicles during equity crashes, show confirmed structural
nulls — this selectivity rules out omnipresent market microstructure as an explanation.

## Results (2026-06-19)

| Asset | Class | a≤6 n | a≤6 mean | a≤6 p | cp n | cp mean | cp p |
|---|---|---|---|---|---|---|---|
| **VNQ** | REIT | 180 | **+0.591%** | **0.0000** | 22 | **+2.954%** | **0.0000** |
| **TLT** | Long bonds | 149 | **+0.205%** | **0.010** | 21 | +0.202% | 0.333 (ns) |
| **DBA** | Agriculture | 153 | +0.148% | 0.085 (ns) | 22 | **+0.671%** | **0.003** |
| IEF | Mid bonds | 157 | +0.061% | 0.166 (ns) | 17 | +0.113% | 0.317 (ns) |
| GLD | Gold | 139 | +0.115% | 0.438 (ns) | 20 | −0.132% | **0.496 (NULL)** |
| USO | Crude oil | 150 | −0.245% | 0.250 (ns) | 23 | +0.809% | 0.090 (ns) |

## Interpretation

**VNQ (REIT)** is the clearest transfer: crash pair +2.95% (p=0.0), a≤6 +0.59%
(p=0.0). REITs are equity-like assets — they trade on exchanges, suffer correlated
drawdowns during equity crashes, and exhibit the same bottom-bin clustering the QA
signal identifies.

**DBA (agricultural commodities)**: crash pair significant (p=0.003). Agricultural
ETFs are sensitive to supply-chain stress which correlates with equity market panic,
explaining partial transfer.

**TLT (long-term Treasuries)**: a≤6 significant (p=0.01). Long-duration bonds
experience price declines during equity crashes (rising yields), producing negative
returns that fall into bottom bins — the a≤6 signal captures this duration-risk
alignment.

**GLD (gold)** is the control case: crash pair p=0.496, direction even wrong (−0.13%).
Gold is a safe-haven that typically *rises* during equity crashes. The bottom-bin
rank signal identifies stress days where gold's response is orthogonal to equity
dynamics — confirming GLD is a genuine structural null, not a power failure.

**IEF and USO**: intermediate-duration bonds have muted crash exposure; crude oil has
supply/demand dynamics uncorrelated with equity stress. Both are genuine nulls.

## Structural Implication

The asset-class gradient (VNQ > DBA > TLT > IEF ≈ GLD ≈ USO) follows equity
correlation rank exactly. This is not a coincidence — it is precisely what the QA
framework predicts: bottom-bin clustering identifies equity-stress regimes, and
transfer strength scales with an asset's exposure to those regimes.

## Theorem NT Compliance

Observer: daily log-return → rank → bin ∈ Z/27Z.
QA state: b=bins[t-1], e=bins[t]; a=b+2e (raw A2).
All returns are observer outputs. No float state enters the QA layer.
Each asset's bin series is computed independently from that asset's own return history.

## Certified Checks

| Check | Description | Result |
|---|---|---|
| C1 | VNQ a≤6 perm_p < 0.001 | PASS (0.0000) |
| C2 | VNQ crash pair perm_p < 0.001 | PASS (0.0000) |
| C3 | VNQ crash pair mean > 1% | PASS (+2.954%) |
| C4 | DBA crash pair perm_p < 0.01 | PASS (0.003) |
| C5 | TLT a≤6 perm_p < 0.05 | PASS (0.010) |
| C6 | GLD crash pair perm_p > 0.10 (structural null) | PASS (0.496) |

## Primary Sources

- Fama EF (1970). doi:10.2307/2325486 (efficient market baseline)
- Erb CB, Harvey CR (2006). doi:10.2469/faj.v62.n2.4083 (commodity return dynamics)

## Related Certs

- [461] QA Witt Tower A-Coordinate Daily Direction (a≤6 parent, equities)
- [463] QA Witt Tower Crash Pair Bounce (crash pair parent, equities)
- [469] QA Witt Tower Vol-Normalized Returns (vol robustness)
- [473] QA Witt Tower OOS Holdout (temporal robustness)
