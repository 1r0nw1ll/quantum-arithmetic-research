# Phase P Observer 1 — Pre-Registration

**Protocol authority:** `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` §2 Observer 1 (single-timescale per-asset, baseline control).

**Session:** `lab-finance-phase-p-observer1`
**Date:** 2026-04-16
**Script (uncommitted scratch):** `qa_lab/phase_p_observer1.py`

This document is committed **before** any test-fold null evaluation is run. Per protocol §1.3, the (statistic, threshold, null) tuple is fixed upon commit.

---

## 1. Data

| Field | Value |
|---|---|
| Asset | SPY (single asset; Observer 1 is single-asset by design) |
| Source | `yfinance` (`yf.Ticker("SPY").history(period="max", auto_adjust=True)`) |
| Cached CSV | `/tmp/spy_daily.csv` |
| Fields used | `Date`, `Close` (auto-adjusted) |
| RV definition | **`RV_proxy_daily_close_log_return_squared`** — `RV_t = (log(Close_t / Close_{t-1}))^2`. This is a proxy; intraday 5-min bars are unavailable beyond ~60 days via `yfinance`. |
| Working series | `log_RV_t = log(max(RV_t, 1e-12))` (floor applied to ~0 returns to keep log finite) |
| `n_days` (after dropping first log-return NaN) | **8359** trading days |
| Date range | **1993-02-01** → **2026-04-16** |

Observer 1 is a pipeline control; the daily-return-squared proxy is acceptable per protocol § "Data" bullet 3. The label `RV_proxy_daily_close_log_return_squared` is used throughout code and output.

## 2. Train/test split (LOCKED)

- Sort ascending by `Date` (already sorted).
- `train_fraction = 0.60`.
- `train_end_idx = floor(0.60 * 8359) = 5015` (zero-based; `log_rv[:5015]` is training).
- **Train fold dates:** `1993-02-01` → approx. `2013-01-18` (index 0..5014).
- **Test fold dates:** approx. `2013-01-22` → `2026-04-16` (index 5015..8358).

Test-fold statistics will not be inspected beyond the single run emitted by the pipeline script after this document is committed.

## 3. Quantization (training-fold frozen)

```
edges = np.percentile(log_RV_train, [100*k/9 for k in range(1, 9)])   # 8 edges → 9 deciles
rank  = bisect.bisect_left(edges, log_RV_t) + 1   # integer in {1..9}
b_t   = qa_mod(rank, 9)                            # A1-compliant; identity for rank in {1..9}
```

Edges are computed on `log_RV[:5015]` only. The same frozen `edges` array is used for the held-out fold and for all null replicates.

## 4. Generator inference (cert [209])

```
e_t = ((b_{t+1} - b_t - 1) mod 9) + 1,   e_t ∈ {1..9}
```

**Causal conditioning (leakage fix, 2026-04-16).** `e_t` depends on `b_{t+1}`. Using `orbit(b_t, e_t)` to forecast `log_RV[t+1]` would leak the forecast target into the conditioning variable (via `b_{t+1}`). The first scratch run produced DM = +2.42 which triggered the protocol's leakage audit clause. Fix: at forecast time t, use `orbit_{t-1} = orbit_family(b_{t-1}, e_{t-1})` (the most recently completed transition at the time the forecast is made). Forecast targets `log_RV[t+1]`; conditioning label is `orbit_{t-1}` (known by time t). This is the locked design; all subsequent training and test-fold computation uses it. Pre-fix DM is not reported as a result — it is a diagnostic artifact retained in commit history.

## 5. Orbit classification

```
orbit_t = orbit_family(b_t, e_t, m=9)   # via qa_orbit_rules.py
```

Expected training-fold distribution under a uniform tuple baseline: cosmos 72/81 ≈ 88.9%, satellite 8/81 ≈ 9.9%, singularity 1/81 ≈ 1.2%. Observed counts will be logged in `phase_p_observer1_results.json`; the run does not condition on them.

Pooling rule: any orbit class with fewer than **30 training observations** is pooled into `cosmos` for the orbit-conditioned HAR fit. The `pool_map` used is logged in the results JSON.

## 6. HAR-RV models

Both fit on **log-RV** (standard HAR-log specification).

**Classical HAR (pooled baseline, Corsi 2009):**
```
log_RV_{t+1} = β_0 + β_d · log_RV_t + β_w · mean(log_RV_{t-4..t}) + β_m · mean(log_RV_{t-21..t}) + ε
```

**Orbit-conditioned HAR (QA alternative):**
Same four-parameter linear model, fit separately per orbit class on the training fold. The coefficient set used at test time `t` is selected by `orbit_{t-1}` (the causally-available conditioning label). If an orbit class has fewer than `MIN_OBS_PER_ORBIT = 30` training observations, it is pooled into cosmos per §5.

Both models require `t ≥ 22` for features to be defined. Training indices: `t ∈ [22, train_end-2]` with `y_train[t] = log_RV[t+1]`. Test indices: `t ∈ [train_end, T-2]`.

## 7. Primary statistic (Diebold-Mariano on QLIKE)

Per-observation forecast converted from log-RV predictions back to RV space via `RV_pred = exp(log_RV_pred)` before computing QLIKE:

```
L_t = RV_{t+1} / predicted_RV_{t+1} - log(RV_{t+1} / predicted_RV_{t+1}) - 1      # Patton 2011
DM  = mean(L_A - L_B) / SE_{HAC}(L_A - L_B)                                       # Newey-West, lag = 5
```

A = classical pooled HAR, B = orbit-conditioned HAR. **Positive DM** ⇒ orbit-conditioned has strictly lower mean QLIKE (QA augmentation beneficial). HAC standard error via Newey-West with fixed lag = 5.

## 8. Three nulls (all applied on test fold via pipeline surrogates)

**N_boot = 500** per null. Random seeds: block-bootstrap uses seed `42`; AR(1) uses seed `43`; phase-randomized uses seed `44`.

### 8.1 Stationary block bootstrap (Politis & Romano 1994)

Block length: `mean_block = ceil(5 × lag_*)`, where `lag_*` is the smallest lag at which sample ACF of training-fold `log_RV` falls below `0.1` (max search lag 100). Bootstrap is applied to the **full** `log_RV` series (length 8359). For each of 500 draws: resample via geometric-runs stationary bootstrap with parameter `p = 1/mean_block`, then re-run the full pipeline **using the frozen training-fold edges** (no re-computation of edges on bootstrap samples) and compute test-fold DM.

### 8.2 AR(1)-matched surrogate

Fit `log_RV_t = α + ρ·log_RV_{t-1} + σ·ε_t` on `log_RV[:5015]`. For each of 500 draws: simulate an AR(1) series of length 8359 starting at `log_RV[0]` with Gaussian innovations, then re-run the pipeline with frozen edges.

### 8.3 Phase-randomized surrogate (Theiler et al. 1992)

For each of 500 draws: take the real FFT of `log_RV`, randomize non-DC (and non-Nyquist for even length) phases uniformly on `[-π, π]` while preserving magnitudes and Hermitian symmetry, inverse-FFT to a real series of length 8359, then re-run the pipeline with frozen edges.

## 9. Pre-registered threshold

Two-tailed p-value per null: `p_null = mean(|DM_null| ≥ |DM_real|)`. Direction is **two-tailed** (LOCKED) because Observer 1 is a pipeline control with expected outcome NULL; one-tailed would bias against detecting unexpected anti-signal.

Bonferroni across three nulls:
```
threshold_bonferroni = 0.05 / 3 = 0.01666...
```

**Rejection criterion:** all three of `p_bb`, `p_ar`, `p_ph` < `0.0167` simultaneously.

## 10. Direction of alternative

**Two-tailed** |DM| test. Locked.

## 11. Expected outcome

**NULL** across all three nulls. Observer 1 is a pipeline control; rejection would indicate data leakage. The first pre-fix scratch run did reject (DM = +2.42) and was diagnosed as look-ahead leakage via `e_t`'s dependence on `b_{t+1}`. The fix (§4) uses `orbit_{t-1}` for conditioning. Post-fix sanity run produced DM = -0.52. If the final run with full nulls still rejects, STOP and re-audit (do not claim the result).

## 12. Environment

| Package | Version |
|---|---|
| numpy | 2.2.4 |
| pandas | 2.3.2 |
| scipy | 1.15.3 |
| statsmodels | 0.14.6 |
| yfinance | 0.2.66 |
| arch (present but not imported) | 7.x |
| Python | system `python3` |

## 13. Git discipline

- This file is committed **before** `phase_p_observer1_results.json` is generated with final null runs.
- The script `qa_lab/phase_p_observer1.py` is NOT committed (scratch per protocol).
- Commit SHA of this pre-registration will be recorded in the results JSON after commit.
- No force-push. No `--no-verify`.

## 14. Artifacts on completion

1. `phase_p_observer1_results.json` — full DM / null distributions / p-values / coefficients / orbit counts.
2. One Open Brain thought tagged `phase-p-observer1`.
3. `collab_broadcast` event `phase_gate_null` (expected) or `phase_gate_passed` (unexpected; triggers audit).
