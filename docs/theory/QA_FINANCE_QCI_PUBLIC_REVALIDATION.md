# QCI Public-Corpus Re-Validation

**Session:** `lab-finance-qci-revalidation`
**Date:** 2026-04-16
**Scope:** `/home/player2/signal_experiments/qci_v2_real_finance.py` run as-is under the author's original pipeline (k-means k=6 on 6D standardized features, window=63, mod=24), with the four-surrogate null stack. Re-validation of the private QCI finding (r=-0.22, p<10^-8) referenced by the framework audit (Claude 2026-04-16a).
**Premise tested:** Does the framework-audit's claim "every time the framework changes, the finance signal comes back" hold on publicly-runnable code?

## 1. Script inspection summary

`qci_v2_real_finance.py` (248 lines, single `main()`, no imports from other repo modules) implements end-to-end the following pipeline, presented in its docstring as "the same pipeline as the validated script 35":

- **Assets (N=6):** `SPY, QQQ, IWM, TLT, GLD, BTC-USD`. All fetched via `yfinance.download(ticker, period="10y")` (Aroussi 2019).
- **Feature vector (D=6 per day):** 5-day log return `log(P_t / P_{t-5})` per asset, then rolling z-score with window=63. The 6 features are the 6 assets; there is **no** explicit engineering of regime/vol/skew features beyond the standardized 5-day log-return. This is narrower than a generic "6 features" description.
- **Clustering:** `KMeans(n_clusters=6, n_init=10, random_state=42)` (Pedregosa et al. 2011), fitted on the **first half** of the standardized matrix, predicted on the full range. This is the OOS split — the second half is held out from cluster fitting.
- **CLUSTER_MAP:** `{0:8, 1:16, 2:24, 3:5, 4:3, 5:11}`. Comment annotates "domain-tuned (as in script 35)."
- **QCI rule:** for each t, `b_t = CMAP[label_t]`, `e_t = CMAP[label_{t+1}]`, predicted next-state `pred = qa_mod(b+e, 24)` (Theorem NT–compliant mod with no-zero convention — line 40: `((int(x)-1) % m) + 1`), actual = `CMAP[label_{t+2}]`. Match rule is binary equality.
- **Rolling QCI:** `t_match.rolling(63, min_periods=32).mean()` — the 63-day rolling hit rate of the T-operator predictions.
- **Continuous target (NOT binary, despite docstring):** `rv_future = daily_ret.shift(-21).rolling(21).std() * sqrt(252)` — 21-day forward realized vol of SPY, annualized. The docstring says "future vol > 75th pctile"; the code uses continuous `rv_future` as the pearsonr target. This is a **docstring/code mismatch** but the continuous form is stricter.
- **Primary statistic:** Pearson r (Pearson 1896) between `real_qci_oos` and `rv_future_oos` on the intersection index (OOS, second half).
- **Partial r:** residualize both QCI and future vol on current realized vol (21-day trailing std), then pearsonr the residuals — the "beyond current vol" partial correlation.
- **Four nulls (N=200 each):**
  - `phase_randomized` — rFFT, keep amplitudes, add uniform random phases; same phases across assets to preserve cross-correlation (Theiler et al. 1992).
  - `ar1` — fit φ per asset, regenerate AR(1) stream with matched innovation variance (Box, Jenkins & Reinsel 1994).
  - `block_shuffled` — permute 50-day blocks (Künsch 1989).
  - `row_permuted` — full time permutation.
- **Null statistics:** for each surrogate, run the same pipeline, compute r and partial r vs the same `rv_future`. Report null mean, std, z-score of real value, and rank p = fraction of nulls with `|r_null| >= |r_real|` (two-sided).

Completeness check: all imports satisfy (`yfinance 0.2.66`, `sklearn 1.8.0`, `scipy 1.15.3`, `numpy 2.2.4`, `pandas 2.3.2`). No helper modules referenced. No cached data path. Runnable standalone. `rand_phases[0]=0` and `rand_phases[-1]=0` on even T correctly zero the DC and Nyquist bins for a real-valued surrogate — no DC/Nyquist phase-randomization bug of the kind noted in Claude (2026-04-16b).

## 2. Run metadata

| Field | Value |
|---|---|
| Data source | yfinance, real pulls, all 6 assets reachable |
| Date range | 2016-04-18 to 2026-04-15 |
| Raw trading days after `dropna` | 2513 |
| OOS alignment window (after 5-day returns, 63-day z-score, half split, 2-step lookahead, forward 21-day vol) | **n=1171** |
| Run wall time | ~4 min (yfinance fetch dominated) |
| Warnings | 6x `FutureWarning` from yfinance about `auto_adjust=True` default change — no effect on result |
| Errors | None |
| Synthetic fallback invoked? | **No** — real data used throughout |

The 2513-day count reflects the intersection of all 6 assets' histories (BTC-USD has the shortest reliable adjusted history among the 6 and sets the floor).

## 3. Primary result

```
REAL QCI vs future vol (OOS, n=1171):
    r        = +0.4355, p = 5.4e-56
    Partial r (beyond lagged RV) = +0.2556, p = 4.1e-19
```

**The correlation is STRONGLY POSITIVE.**

The private result quoted in the framework audit (Claude 2026-04-16a, §Pattern 8; originating OB 2026-03-25T04:12:53) is `r = -0.22, p < 10^-8` (partial r after lagged RV: `-0.2154`). The public-corpus result under the same pipeline claim is `r = +0.4355` — **opposite sign, roughly double magnitude**.

Sign reading: positive r means **high QCI match rate predicts high future realized volatility**. Interpretation inverts: the private script's high-QCI was framed as *low-stress* (predictive stability); the public script's high-QCI is *high-stress* (predictive turbulence). If the pipelines were truly identical the numbers would be within statistical error. They are not — they are ~6σ apart in sign.

## 4. Null comparison

200 surrogates per method. `rank_p` is the two-sided fraction of null |r| that meet-or-exceed real |r|. Real r = +0.4355; real partial r = +0.2556.

| Null method | Valid N | E[r_null] | σ[r_null] | z (real) | rank p (real r) | Beats? | Null partial mean | rank p (partial) | Partial beats? |
|---|---|---|---|---|---|---|---|---|---|
| phase_randomized | 200 | -0.0047 | 0.1671 | +2.63 | 0.0050 | YES | -0.0018 | 0.0350 | YES |
| ar1              | 200 | +0.0017 | 0.1553 | +2.79 | 0.0050 | YES | -0.0012 | 0.0450 | YES |
| block_shuffled   | 200 | -0.0077 | 0.1689 | +2.62 | 0.0100 | YES | -0.0075 | 0.0550 | NO  |
| row_permuted     | 200 | +0.0222 | 0.1838 | +2.25 | 0.0100 | YES | +0.0171 | 0.0800 | NO  |

Raw-r verdict: **real r beats all four nulls at the two-sided 5% level.** Partial r (beyond lagged RV) beats phase_randomized and ar1 but not block_shuffled or row_permuted at 5%. The block_shuffled/row_permuted nulls preserve enough structure for 50-day blocks (resp. create high-variance nulls by destroying all temporal structure) that a 21-day-forward vol target correlates with the rolling-63 QCI through residual autoregressive structure that current-vol residualization does not fully absorb.

Important qualifier: phase-randomized null `std = 0.167` is large. Real r=+0.44 is only 2.63σ beyond a near-zero null mean, not the 5-10σ one might naively expect from p<10^-56 on n=1171. The reason is that the parametric Pearson p-value (`stats.pearsonr`) assumes iid samples; the rolling-63 QCI and rolling-21 forward-vol both induce strong serial correlation that the surrogate null *does* account for. **The 2.5-2.8σ rank-p<0.01 against phase-random / AR(1) is the trustworthy significance, not the parametric p<10^-56.** The signal is real but its effective information content is an order of magnitude smaller than the parametric p implies.

## 5. Verdict

**Outcome 3 (variant): Sign-flipped reproduction.**

The public replica produces a **strongly significant correlation in the opposite sign** from the private result, with ~2x the magnitude. The surrogate comparisons confirm the public signal is not an artifact of linear spectrum, per-asset autocorrelation, or block structure — it beats all four nulls at 5% on raw r and two of four on partial r. The sign inversion means one of the following is true:

1. **The public replica diverges structurally from private script 35** at one or more of: CLUSTER_MAP, asset list, window choice, target definition, or the sign convention of how QCI "predictive stability" vs "predictive turbulence" is encoded. The docstring claims "same pipeline"; the numbers say otherwise.
2. **The private result was unstable** in some hyperparameter the public replica cannot inherit — e.g. a specific asset subset, a different CLUSTER_MAP tuned on different training data, or a forward-vol target with different horizon. QCI is highly sensitive to CLUSTER_MAP (arbitrary label→state assignment), and `{0:8, 1:16, 2:24, 3:5, 4:3, 5:11}` is one of 6!=720 permutations.
3. **Signal is real, sign is map-dependent.** Under k-means label permutation (labels 0-5 are arbitrary), CLUSTER_MAP is a nuisance parameter. Two runs with different maps can give r of opposite sign but similar |r| because what QCI measures is the *rolling rate of a consistent symbolic rule firing*, and which direction that rule covaries with vol depends on the map. If this is the case, the right statistic is `|r|` across a distribution of cluster-map permutations, or a correctly map-invariant reformulation.

(3) is the benign reading. It also says "QCI's sign depends on an arbitrary relabeling, which is a framework problem." The positive r and its surrogate-beating survive regardless of which of (1)/(2)/(3) is true — meaning QCI does find QA-native structure above the linear-spectrum null. But the audit's specific claim was `r = -0.22, p < 10^-8` and that number does not reproduce.

The audit asks whether "every time the framework changes, finance signal comes back." **It does come back, in magnitude and in significance against nulls.** What does not come back is the sign. Any downstream use of QCI that depends on sign is **not supported** by the public replica.

## 6. Observer 3 implication

Per the audit's §3 (Claude 2026-04-16a), Observer 3 is a cross-asset synchrony-on-decile design conditional on QCI surviving re-validation. The re-validation is **ambiguous**: the framework produces signal, but sign-inverted relative to the private precedent. Implications:

1. **Observer 3 can proceed on the basis that a QA-native observer family beats the Phase P null stack on the public corpus** — that is the architectural claim the audit asked to be validated, and it holds (4/4 on raw r, 2/4 on partial).
2. **Observer 3 should NOT inherit the sign convention from QCI.** Any pre-registration that predicts a negative r between some observer statistic and future vol, motivated by "QCI was -0.22," is unsafe. Predict magnitude and null-beats, not sign.
3. **The CLUSTER_MAP / label-invariance question should be answered first** — ideally by running `qci_v2_real_finance.py` across a distribution of random CLUSTER_MAPs and reporting the distribution of r. If the sign flips roughly at random across permutations and |r| is stable, QCI is a map-dependent framework and its published sign numbers are not reproducible without the exact map. If |r| depends strongly on the map, QCI has a tuning problem. Either way, Observer 3's synchrony statistic should be designed to be invariant to arbitrary cluster-relabeling (e.g. use orbit *partition* information directly, not CLUSTER_MAP-mediated arithmetic). This is a 1-day follow-up, not a blocker for Observer 3 launch.
4. **The docstring/code mismatch** ("future vol > 75th pctile" in docstring vs continuous `rv_future` in pearsonr) does not change this run's verdict but should be reconciled if the script is cited as the canonical QCI replica.

Recommendation: proceed with Observer 3 per the corrected spec, but amend the pre-reg to (a) predict |r| and null-beats only, not sign; (b) design the observer statistic to be invariant to the arbitrary-labeling degree of freedom that QCI's CLUSTER_MAP exposes. The framework-audit's launch-readiness gate — "QCI re-validates on public" — is met in the magnitude-and-null-beats sense and unmet in the exact-number sense.

## Artifacts

- Run log: `/tmp/qci_public_revalidation.log`
- Script (unmodified): `/home/player2/signal_experiments/qci_v2_real_finance.py`
- This document: `/home/player2/signal_experiments/docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md`

## References

- Aroussi, R. (2019). *yfinance: Yahoo! Finance market data downloader*. https://github.com/ranaroussi/yfinance
- Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (1994). *Time Series Analysis: Forecasting and Control* (3rd ed.). Prentice Hall. ISBN 978-0130607744.
- Claude (2026-04-16a). *QA Finance Framework Audit*. `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md`. Internal primary source; OB capture 2026-04-17T00:28:18Z.
- Claude (2026-04-16b). *HAR-Orbit Collinearity Diagnostic*. `docs/theory/QA_HAR_ORBIT_COLLINEARITY_DIAGNOSTIC.md`. Internal primary source.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217-1241. https://doi.org/10.1214/aos/1176347265
- Pearson, K. (1896). Mathematical contributions to the theory of evolution. III. Regression, heredity, and panmixia. *Philosophical Transactions of the Royal Society A*, 187, 253-318. https://doi.org/10.1098/rsta.1896.0007
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825-2830. arxiv.org/abs/1201.0490
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., & Farmer, J. D. (1992). Testing for nonlinearity in time series: the method of surrogate data. *Physica D*, 58(1-4), 77-94. https://doi.org/10.1016/0167-2789(92)90102-S
