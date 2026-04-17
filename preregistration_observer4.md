# Phase P Observer 4 — Pre-Registration

**Protocol authority:** `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` §1 (post-X1 correction, committed 795a7f4) + `docs/theory/QA_FINANCE_O4_AND_FRAMEWORK_REVIEW.md` §6 (critic sign-off with five amendments, all incorporated below).

**Session:** `lab-finance-phase-p-observer4`
**Date:** 2026-04-17
**Script (uncommitted scratch):** `qa_lab/phase_p_observer4.py` (not yet written; will be implemented per this pre-reg, linter-clean)
**Predecessor:** Observer 1+2 = decile-rank-log-RV observer with three variants, **one** iteration slot consumed of 5 (post-X1 re-accounting). Observer 4 is the first family-crossing observer; name retained as "4" for continuity with the pre-correction iteration count and audit trail.

This document is committed **before** any test-fold evaluation is run. Per protocol §1.3, the (statistic, threshold, null, training-gate, prerequisite) tuple is fixed upon commit. Any subsequent change counts as a new iteration.

---

## 1. Data — 5-asset panel

| Ticker | Source | `n_rows` (yfinance `period=max, auto_adjust=True`) | Date range | SHA-256 (`/tmp/<ticker>_daily.csv`) |
|---|---|---:|---|---|
| GLD | yfinance | 5385 | 2004-11-18 → 2026-04-16 | `e9b653e219f4995819f3d8a1ddceea1d7ee38dafe6e1336d9797c487d85fff0e` |
| SPY | yfinance | 8360 | 1993-01-29 → 2026-04-16 | `09ec8dfae5669511659b2ef34c5cc87f45ed8dbc89308e2c60b37c83c6afe3b0` |
| TLT | yfinance | 5967 | 2002-07-30 → 2026-04-16 | `2b558b0f4ea28033625e561a76d7e990df4f4b83b20698dbba543b21dd93620a` |
| USO | yfinance | 5036 | 2006-04-10 → 2026-04-16 | `cc1e55120045f3f6374127683e278f7cd694655e4c2aab75836abdd75ea7d97b` |
| UUP | yfinance | 4813 | 2007-03-01 → 2026-04-16 | `6e4290242ddd7b4996d33e9ffb51e55b469b9824347cb83bf8214f1b61bc1c88` |

**Asset order (LOCKED, alphabetical per critic amendment 1):** `GLD, SPY, TLT, USO, UUP`.

**Panel alignment:** inner-join on `Date` across the 5 CSVs. Expected panel length ≈ 4813 (gated by UUP's start date 2007-03-01). Script must report exact `n_panel` after alignment.

**Daily return:** `r_{t,i} = log(Close_{t,i} / Close_{t-1,i})` per asset `i` on the aligned panel.

## 2. Train/test split (LOCKED)

- Sort ascending by `Date`.
- `train_fraction = 0.60`.
- `train_end_idx = floor(0.60 * n_panel)` (zero-based; `panel[:train_end_idx]` is training).
- Test fold is `panel[train_end_idx:]`.
- Expected dates: training `2007-03-01` → ≈ `2018-08`; test `≈ 2018-09` → `2026-04-16`. Script must record exact boundary dates.

## 3. Observer — α sign-vector-index (family-crossing from decile-rank)

**Critic-selected over β and γ per review §2** (`docs/theory/QA_FINANCE_O4_AND_FRAMEWORK_REVIEW.md:109-120`). Fresh quantization family: sign-pattern-index.

Per trading day `t`:

1. **Sign tie-break (LOCKED per critic amendment 2):** `sign(r) = +1 if r >= 0, else -1`. Zero boundary resolved deterministically.

2. **Sign-vector-index construction:** for assets `i ∈ {GLD, SPY, TLT, USO, UUP}` in alphabetical order (positions 0..4):
   ```
   s_{t,i} = (r_{t,i} >= 0)  # boolean
   idx_t   = Σᵢ s_{t,i} × 2^i    # integer in {0..31}
   ```

3. **QA firewall cast (A1-compliant):**
   ```
   b_t = qa_mod(idx_t + 1, 9)   # canonical qa_mod from qa_lab/qa_observer/core.py
   ```
   Modulus `m = 9` primary; `m = 24` confirmatory (X6 sensitivity only — NOT a new iteration per §1.2).

4. **Generator inference via [209]:**
   ```
   e_t = ((b_{t+1} - b_t - 1) % 9) + 1
   ```
   T2 crossing count = 2 (input at step 3, output at step 6). Steps 1-2 are observer-layer; 3-5 are QA-discrete-layer; 6 is observer-layer projection.

5. **Orbit classification:**
   ```
   orbit_t = orbit_family(b_t, e_t, 9)   # ∈ {singularity, satellite, cosmos}
   ```
   from `qa_orbit_rules.py:72`.

6. **Categorical observable:** the sequence `(orbit_1, orbit_2, ..., orbit_{n-1})` is the Observer 4 output.

## 4. Primary statistic — marginal orbit-transition MI on 3×3

Mutual information `I(orbit_{t-1}; orbit_t)` over the 3×3 empirical contingency of consecutive orbit pairs. Marginal (NOT conditional on lagged RV or any observer-side variable) per prior O3-critic concern #2 and this review's Part 3.

**Estimator (LOCKED):** plug-in MI with **Miller-Madow bias correction** (Miller 1955): `MI_MM = MI_plugin + (k − 1)² / (2n)` where `k = 3` orbit classes and `n` is contingency-table total count. At `n ≈ 4813 × 0.4 ≈ 1925` test fold, bias correction ≈ `(3−1)² / (2·1925) ≈ 0.001 nats`.

Rationale for marginal over conditional: conditional MI requires binning the conditioning variable, which re-imports decile-rank as a nuisance side-channel (prior O3-critic concern #2). Sign-vector-index is structurally non-rank; marginal MI cleanly measures "does orbit transition structure exist beyond marginal orbit distribution."

## 5. Two nulls (LOCKED, QA-native-first per post-X1 §1.2)

**Null A — Permuted-generator (QA-native).** On the real `b`-sequence, randomly permute the `e`-sequence: `e^shuf = rng.permutation(e_real)`. Recompute `orbit^shuf_t = orbit_family(b_t, e^shuf_t, 9)`. Compute MI on permuted orbit sequence. Destroys generator temporal order while preserving `b`-marginal and `b`-autocorrelation. Tests "does the QA generator add structure beyond `b` itself."

**Null B — Marginal-orbit-iid (QA-native).** Compute empirical orbit marginal on training fold. Draw `orbit^shuf_t` iid from that marginal. Compute MI. Destroys all transition structure while preserving marginal. Tests "does transition structure exist at all."

**NO observer-side nulls** (no block-bootstrap, no AR(1), no phase-randomized, no HAR-frozen) — dropped per critic concerns #3-4 and post-X1 §1.2 QA-native requirement.

**Null parameters (LOCKED per critic amendment 4):**
- `n_null = 1000` permutations per null.
- `seed_null_A = 42`, `seed_null_B = 43`.

**Two-tailed rank-p:** fraction of null MI ≥ real MI.

## 6. Pre-registered thresholds

**Primary gate (LOCKED):** rank-p < 0.05 on BOTH nulls (AND-conjunction, uncorrected). No Bonferroni because the two nulls destroy complementary structure (generator-order vs transition-existence); effective α with weak correlation between them ≈ 0.025. Review §3 concern 3.6.

**Effect-size floor (soft gate per review concern 3.7):** test-fold `MI_MM ≥ 0.5 × training-fold MI_MM`. Failure triggers warning in the results JSON; does not by itself NULL the verdict.

**Predict |effect|, not signed** (MI is nonnegative by construction; this is automatic).

## 7. Training-fold kill-gates (evaluated before test fold is touched)

All four must pass on training fold, else declare training-fold NULL and do not compute test-fold statistic:

1. **Miller-Madow MI ≥ 0.005 nats** on 3×3 training contingency.
2. **χ² independence test p < 0.01** on training 3×3.
3. **Expected-cell count ≥ 5** in every cell of the 3×3 (`χ² validity requirement`).
4. **Orbit-marginal minimum ≥ 5%** on training (per critic amendment 3; prevents degenerate marginals from making MI meaningless).

Training-fold failure → declare NULL with diagnostic, budget slot consumed, do NOT touch test fold.

## 8. Prerequisite — synthetic panel calibration (LOCKED per critic amendment 5)

**Before** Observer 4 is applied to real panel's training fold, calibrate on synthetic independent panels to verify the null's FPR is not inflated.

**Synthetic construction (LOCKED per critic concern 4.2):** each seed draws an iid empirical-bootstrap of each asset's own training-fold daily returns (preserves per-asset marginal distribution including tail shape; destroys cross-sectional co-movement). This is Horowitz 2001 bootstrap for cross-sectional independence.

**Run:** apply full α pipeline (§3-5) to `N = 500` synthetic seeds. For each seed, compute real-MI and two null rank-p's. Measure **empirical FPR**: fraction of seeds where both rank-p < 0.05 simultaneously (the primary gate condition).

**Pass condition (LOCKED):** empirical FPR ≤ 0.05 with 95% Clopper-Pearson upper CI ≤ 0.06 at `N = 500`. At `p = 0.05, n = 500`, 95% upper CI ≈ 0.073 — so actual pass requires observed FPR around 0.03 or below.

**Fail consequence (LOCKED):** do NOT run Observer 4 on real panel's training fold. Move to fallback β-with-T2-b-fix (`b_t = qa_mod(decile_rank(φ_t, edges_from_training), 9)` where `φ_t = λ₁ / Σ λ_k` of rolling-22-day correlation matrix). Re-run X2 + X3 for β. Budget slot still counts as consumed ("O4 attempt").

**If β also fails prerequisite:** exit the finance observer-iteration track at NULL. Do NOT attempt γ (volume-return joint) — rejected by critic as still rank-family + volume-confound.

## 9. X6 sensitivity analyses (only if X5 test-fold PASS)

1. **Asset-order sensitivity.** Run α under all `5! = 120` permutations of asset ordering. Report MI distribution across orderings. Cert-grade witness requires ≥ 100/120 orderings achieve rank-p ≤ 0.05 (critic amendment 1).
2. **Modulus confirmatory.** Same observer at `m = 24`. Confirmatory, not a new iteration.
3. **Held-out sector-ETF panel.** Re-run on `XLF, XLE, XLI, XLK, XLV` (alphabetical: XLE, XLF, XLI, XLK, XLV). Independent 5-asset panel. Replication here is required for cert-grade witness.

## 10. Budget accounting (post-X1)

| Slot | Consumer | Status |
|---|---|---|
| 1 | Decile-rank log-RV observer (O1/O2/proposed-O3 variants) | CONSUMED, NULL revoked → framework-limited |
| 2 | Sign-vector-index (this pre-reg, α) | PENDING — this pilot |
| 3 | Reserved | Likely eigenspectrum β if α NULLs, or next family if α PASSES and a follow-up is warranted |
| 4 | Reserved | |
| 5 | Reserved | |

## 11. Stopping criteria

- **Training gate fail:** declare NULL for α, move to β per §8 fallback. Budget slot 2 consumed.
- **Prerequisite FPR fail:** α is mis-calibrated, move to β. Slot 2 consumed.
- **Test-fold primary gate fail (rank-p ≥ 0.05 on either null):** declare NULL. Slot 2 consumed. Do NOT launch β immediately; return to Phase R framework and re-derive a genuinely different observer candidate. Avoid the "just try β" reflex.
- **Test-fold primary gate PASS + effect-size soft gate PASS:** proceed to X6 sensitivity. If ≥ 100/120 orderings replicate AND sector-ETF panel replicates, produce cert candidate. If sensitivity fails, declare MARGINAL with documented scope.

## 12. Environment

- `python` 3.13.7
- `numpy` 2.2.4
- `scipy` 1.15.3 (Miller-Madow via scipy digamma if used; otherwise manual bias formula)
- `statsmodels` (if Miller-Madow package used)
- `yfinance` 0.2.x (for panel refresh; data already fetched, hashes above)

## 13. Commit discipline

- This pre-reg is committed to git BEFORE any script is run on real panel training fold.
- Commit SHA of this pre-reg is recorded in `phase_p_observer4_results.json` metadata.
- Data SHA-256 hashes in §1 must match at test-fold run time; script validates before proceeding.
- Synthetic prerequisite results are reported in a separate `phase_p_observer4_synthetic_prereq.json` with its own SHA-256 of the prereq script.

## 14. Protocol provenance

Critic sign-off chain:
- Framework audit: `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md` (revoked O1/O2 NULL)
- O3 critic: `docs/theory/QA_FINANCE_O3_PLAN_CRITIC_REVIEW.md` (DO-NOT-LAUNCH on corrected O3 → family-crossing required)
- O4 + framework review: `docs/theory/QA_FINANCE_O4_AND_FRAMEWORK_REVIEW.md` (SIGNED-OFF-WITH-AMENDMENTS; α selected; five amendments incorporated in §3, §5, §7, §8, §9)
- Framework-doc X1 correction: commit `795a7f4`

This pre-reg is the operational contract for Observer 4 execution.
