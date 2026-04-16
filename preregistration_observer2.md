# Phase P Observer 2 — Pre-Registration

**Protocol authority:** `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` §2 Observer 2 (three-timescale stacked per-asset cascade).

**Session:** `lab-finance-phase-p-observer2`
**Date:** 2026-04-16
**Script (uncommitted scratch):** `qa_lab/phase_p_observer2.py`
**Predecessor:** Observer 1 — NULL across all three nulls (committed `phase_p_observer1_results.json`, pre-reg SHA `91b601a`).

This document is committed **before** any test-fold null evaluation is run. Per protocol §1.3, the (statistic, threshold, null) tuple is fixed upon commit. Script imports the patched `phase_randomize` (Theiler 1992 — preserves DC + Nyquist phases) and other shared primitives from `qa_lab/phase_p_observer1.py`.

---

## 1. Data

| Field | Value |
|---|---|
| Asset | SPY (single asset; cascade across timescales of the same asset) |
| Source | Cached CSV `/tmp/spy_daily.csv` (same bytes as Observer 1, NOT re-fetched) |
| Fields used | `Date`, `Close` (auto-adjusted) |
| RV definition | **`RV_proxy_daily_close_log_return_squared`** — `RV_t = (log(Close_t / Close_{t-1}))^2`. Identical to Observer 1. |
| Floor | `max(RV_t, 1e-12)` so `log_RV_t` finite. |
| `n_days` (after dropping first log-return NaN) | **8359** trading days |
| Date range | **1993-02-01** → **2026-04-16** |

Same dataset as Observer 1 to keep O1 / O2 directly comparable.

## 2. Train/test split (LOCKED — identical to Observer 1)

- Sort ascending by `Date` (already sorted).
- `train_fraction = 0.60`.
- `train_end_idx = floor(0.60 * 8359) = 5015` (zero-based; `log_rv[:5015]` is training).
- **Train fold dates:** `1993-02-01` → `2012-12-27` (index 0..5014).
- **Test fold dates:** `2012-12-28` → `2026-04-16` (index 5015..8358).

## 3. Featurization — three timescales

For each trading day `t`:

```
RV_t            = (log(Close_t / Close_{t-1}))^2          # same as Observer 1
RV_t^{(5)}      = mean(RV_{t-4..t})                        # 5-day window
RV_t^{(22)}     = mean(RV_{t-21..t})                       # 22-day window

log_RV_t^{(k)}  = log(max(RV_t^{(k)}, 1e-12))   for k ∈ {1, 5, 22}
```

The k=1 stream IS the Observer 1 series (identity reuse). Streams k=5 and k=22 are new.

Rows with insufficient history (`t < 21` for the monthly stream) are dropped before any fitting / quantization. The training fold for quantization edge computation is therefore `t ∈ [21, 5014]` for each stream (`5015 - 21 = 4994` training rows). HAR feature alignment further requires `t ≥ 22` (one extra row to allow the lagged HAR features to exist), which is the standard offset.

## 4. Quantization (training-fold-frozen, per timescale; A1-compliant)

For each timescale `k ∈ {1, 5, 22}`, **independently**:

```
edges_k = np.percentile(log_RV_train^{(k)}, [100*j/9 for j in range(1, 9)])    # 8 edges → 9 deciles
rank    = bisect.bisect_left(edges_k, log_RV_t^{(k)}) + 1                      # int in {1..9}
b_{t,k} = qa_mod(rank, 9)                                                      # A1: identity for rank in {1..9}
```

Edges are computed on each timescale's training-fold log-RV only (no cross-fold leakage). The same frozen `edges_k` are used for the test fold and for ALL surrogate replicates per null.

Verification: for every (t, k), assert `b_{t,k} ∈ {1..9}`. No zeros.

## 5. Generator inference per timescale (cert [209])

For each timescale `k` independently:

```
e_{t,k}     = ((b_{t+1,k} - b_{t,k} - 1) mod 9) + 1   # e_{t,k} ∈ {1..9}
orbit_{t,k} = orbit_family(b_{t,k}, e_{t,k}, 9)
```

**Causal conditioning (carried forward from Observer 1's leakage fix):** `e_{t,k}` depends on `b_{t+1,k}`. To avoid look-ahead leakage when forecasting `log_RV_{t+1}^{(1)}`, the conditioning labels at forecast time `t` use `orbit_{t-1,k}` (the most recently completed transition at time `t`). Pre-fix Observer 1 produced DM=+2.42; post-fix Observer 1 produced DM=-0.52 — pre-reg authority `91b601a`. Same fix applied here per-timescale.

## 6. Orbit-distribution logging (training fold, per timescale)

The training-fold orbit distribution is logged separately for each k. Comparison points:

- Uniform-tuple baseline on mod-9: cosmos 72/81 ≈ 88.9%, satellite 8/81 ≈ 9.9%, singularity 1/81 ≈ 1.2%.
- Observer 1 daily (k=1): cosmos 87.9%, satellite 9.6%, singularity 2.5% (binomial z = 8.3 vs uniform).

Scientific question: does singularity enrichment strengthen at short timescales (daily; clustering signal) or long timescales (monthly; regime persistence)? Both directions are logged.

The singularity-enrichment statistic does NOT enter the gate decision; it is descriptive.

## 7. HAR-RV models (both fit on log-RV)

**Classical HAR (pooled baseline, Corsi 2009 — IDENTICAL to Observer 1's pooled HAR):**
```
log_RV_{t+1} = β_0 + β_d · log_RV_t^{(1)} + β_w · log_RV_t^{(5)} + β_m · log_RV_t^{(22)} + ε
```
4 parameters.

**Cascade-augmented HAR (QA alternative):** add 6 main-effect dummies (satellite/singularity per timescale, cosmos as reference) plus 2 interaction dummies:
```
log_RV_{t+1} = β_0 + β_d · log_RV_t^{(1)} + β_w · log_RV_t^{(5)} + β_m · log_RV_t^{(22)}
             + γ_1 · I(orbit_{t-1,1}  = satellite)
             + γ_2 · I(orbit_{t-1,1}  = singularity)
             + γ_3 · I(orbit_{t-1,5}  = satellite)
             + γ_4 · I(orbit_{t-1,5}  = singularity)
             + γ_5 · I(orbit_{t-1,22} = satellite)
             + γ_6 · I(orbit_{t-1,22} = singularity)
             + δ_1 · I(orbit_{t-1,1}  = satellite   AND orbit_{t-1,5} = satellite)
             + δ_2 · I(orbit_{t-1,1}  = singularity AND orbit_{t-1,5} = singularity)
             + ε
```
12 parameters total (4 HAR + 6 main + 2 interaction). Cosmos is the reference class (its dummy is dropped to avoid perfect multicollinearity).

**Pooling rule for sparse classes:** if any class `(orbit_{t,k} = satellite)` or `(orbit_{t,k} = singularity)` has fewer than **30 training observations**, that dummy is dropped from the design matrix (i.e., merged with cosmos) and the event is recorded in `pooling_events` in the results JSON. Same threshold as Observer 1's `MIN_OBS_PER_ORBIT`.

Both models share the same training-row alignment: rows `t ∈ [22, train_end-2]` with `y_train[t] = log_RV[t+1]`. Test rows: `t ∈ [train_end, T-2]`.

## 8. Primary statistic (Diebold-Mariano on QLIKE)

Per-observation predictions converted from log-RV to RV via `RV_pred = exp(log_RV_pred)`, then Patton (2011) QLIKE:

```
L_t = RV_{t+1} / RV_pred_{t+1} - log(RV_{t+1} / RV_pred_{t+1}) - 1
DM  = mean(L_classical - L_augmented) / SE_HAC(L_classical - L_augmented)
```

- **A = classical pooled HAR**, **B = cascade-augmented HAR**.
- DM convention: `DM = mean(L_A - L_B)`. **Positive DM ⇒ augmented has lower mean QLIKE (augmented better)**.
- HAC standard error: Newey-West with **lag = 5** (matches Observer 1's lag for direct comparability).

**Auxiliary RMSE statistic on log-RV predictions** (used by the effect-size floor):
```
RMSE_classical  = sqrt(mean( (log_RV_test - pred_log_RV_classical)^2 ))
RMSE_augmented  = sqrt(mean( (log_RV_test - pred_log_RV_augmented)^2 ))
RMSE_reduction  = (RMSE_classical - RMSE_augmented) / RMSE_classical    # signed; positive = augmented better
```

## 9. Three nulls (all applied via pipeline surrogates with frozen training-fold edges)

**N_boot = 500 per null** (matching Observer 1). Random seeds: block-bootstrap uses `42`, AR(1) uses `43`, phase-randomized uses `44`. All three nulls operate on the **daily** `log_RV` series; the cascade (5-day, 22-day means) is recomputed downstream from the surrogate daily series, since cascade values are deterministic functions of the daily series. This preserves Observer 1's null architecture exactly.

**For each null replicate:** generate a daily-frequency surrogate `log_RV_surrogate` of length 8359; then internally:
1. Recompute `RV_surrogate = exp(log_RV_surrogate)` to get back to RV.
2. Recompute the 5-day and 22-day rolling means → `RV^{(5)}_surrogate`, `RV^{(22)}_surrogate`.
3. Take logs again to get `log_RV^{(k)}_surrogate` for k ∈ {1, 5, 22}.
4. Recompute decile edges per scale on the surrogate's training portion (per protocol §"Critical" note: edges must be re-computed on the surrogate's training portion, not copied from real data; this is what tests whether the training-fold edge structure itself is signal).
5. Quantize, infer e per scale, classify orbits per scale.
6. Fit both models on training; compute test-fold DM.

### 9.1 Stationary block bootstrap (Politis & Romano 1994)

Block length: `mean_block = ceil(5 × lag_*)`, where `lag_*` is the smallest lag at which sample ACF of training-fold daily `log_RV` falls below `0.1` (max search lag 100). Geometric-runs stationary bootstrap with parameter `p = 1/mean_block`. Imported verbatim from `phase_p_observer1.py`.

### 9.2 AR(1)-matched surrogate

Fit `log_RV_t = α + ρ · log_RV_{t-1} + σ · ε_t` on training-fold daily `log_RV[:5015]`. Simulate AR(1) of length 8359 starting at `log_RV[0]` with Gaussian innovations. Imported verbatim from `phase_p_observer1.py`.

### 9.3 Phase-randomized surrogate (Theiler et al. 1992)

Patched `phase_randomize` from `phase_p_observer1.py` (preserves DC + even-n Nyquist phases). The Observer 1 finalization caught a bug where zeroed DC phase flipped log-RV sign; the fix is in `qa_lab/phase_p_observer1.py:422-443` and is imported here.

## 10. Pre-registered threshold (DUAL — both must hold)

Two-tailed p-value per null: `p_null = mean(|DM_null| ≥ |DM_real|)`. Direction is **two-tailed** (LOCKED). Per protocol §2 Observer 2: there is no strong directional prior; the augmentation could also overfit (DM could go negative).

Bonferroni across three nulls: `threshold = 0.05 / 3 = 0.0167`.

**Gate (BOTH must hold for "Observer 2 PASSES"):**

1. `p_block_bootstrap < 0.0167` AND `p_ar1 < 0.0167` AND `p_phase_random < 0.0167` (triple-null rejection at Bonferroni-adjusted threshold), AND
2. `RMSE_reduction ≥ 0.02` (test-fold log-RV RMSE reduction ≥ 2 percentage points; cascade-augmented strictly better).

Either condition failing alone ⇒ Observer 2 NULL (per protocol §2 calibration of HAR-vs-AR(1) Corsi headroom).

## 11. Direction of alternative

**Two-tailed** |DM| test. Locked.

## 12. Modulus

Primary: `m = 9` (theoretical QA modulus). `m = 24` is **NOT** run unless `m = 9` rejects (per protocol §2 confirmatory rule). If `m = 9` passes the gate, a single confirmatory `m = 24` run is logged in the results JSON as a robustness check (does not constitute a new iteration; same observer under modulus variation).

## 13. Pooling rule for sparse orbit classes

If any single orbit indicator has fewer than `MIN_OBS_PER_ORBIT = 30` training observations, the corresponding dummy column is dropped from the augmented-HAR design matrix and the event is recorded under `pooling_events` in the JSON. The interaction dummies are dropped if either of their constituent main-effect dummies is dropped.

## 14. Expected outcome

**Uncertain.** This is the first real test of QA augmentation in HAR-RV. Observer 1 was designed as the pipeline control with NULL expected (and confirmed). Observer 2's outcome is genuinely unknown: the singularity 2.5% > 1.2% enrichment Observer 1 surfaced suggests there IS some QA-native structure in daily log-RV; whether that structure is strong enough to clear three matched nulls and a 2% RMSE-reduction floor — and whether it survives the 8-parameter overhead — is the empirical question.

If the volatility-clustering-persistence hypothesis holds, augmented should win (DM > 0). If the augmentation overfits, DM could be negative. Either is admissible per pre-reg.

## 15. Environment

| Package | Version |
|---|---|
| Python | system `python3` (3.13.x) |
| numpy | 2.2.4 |
| pandas | 2.3.2 |
| scipy | 1.15.3 |
| statsmodels | 0.14.6 |
| yfinance | 0.2.66 (not invoked; cached CSV reused) |

Seed: `42` (matches Observer 1 RNG seed).

## 16. Git discipline

- This file is committed **before** `phase_p_observer2_results.json` is generated.
- The script `qa_lab/phase_p_observer2.py` is NOT committed (scratch per protocol).
- Commit SHA of this pre-registration is recorded in the results JSON after commit.
- No force-push. No `--no-verify`.
- No cert reservation. Observer 2 does not gate a cert by itself; Observer 3 or the full ladder does.

## 17. Artifacts on completion

1. `phase_p_observer2_results.json` — full DM, three null distributions, p-values, both model coefficients, decile edges per scale, per-scale orbit distributions (train + test), pooling events.
2. One Open Brain thought tagged `phase-p-observer2`.
3. `collab_broadcast` event `phase_gate_passed` (if the dual gate is met) or `phase_gate_null` (otherwise).

## 18. Stop conditions

- If the script encounters any axiom violation mid-run (A1, A2, S1, S2, T1, T2-b), STOP and fix; do not bypass with `noqa` / suppress.
- If a single null phase exceeds ~30 minutes of wall time, drop `N_boot` for that null to **200** and document in the results JSON `metadata.n_boot_actual` (per protocol step 9 fallback).
- If the gate is met unexpectedly (all three p < 0.0167 AND RMSE_reduction ≥ 2%), do NOT claim a cert. Only Observer 3 (or the full ladder) gates a cert. Broadcast `phase_gate_passed` and stop.
