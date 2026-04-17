# QA Finance Framework Audit — Phase P Observer 1 & 2

**Role:** Adversarial critic.
**Date:** 2026-04-16.
**Scope:** `qa_lab/phase_p_observer1.py`, `qa_lab/phase_p_observer2.py`, their pre-regs, their results JSONs, and the HAR×orbit-dummy collinearity diagnostic. Predecessors (prior Phase 2.5, QCI, [209] finance synchrony) considered where they constrain the audit.
**Premise I am testing:** Will's prior — "the signal always seems to disappear but only on finance how curious" — against the concrete design choices Observers 1/2 smuggled in.

Primary sources grounding this audit: Corsi (2009) HAR-RV DOI 10.1093/jjfinec/nbp001; Diebold & Mariano (1995) DOI 10.1080/07350015.1995.10524599; Politis & Romano (1994) stationary block bootstrap DOI 10.1080/01621459.1994.10476870; Theiler, Eubank, Longtin, Galdrikian & Farmer (1992) DOI 10.1016/0167-2789(92)90102-S; Patton (2011) DOI 10.1016/j.jeconom.2010.03.034; Newey & West (1987) DOI 10.2307/1913610. See `## References`.

Companion artifacts: `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/` (cert [209]); `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md`; `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`; `docs/theory/QA_HAR_ORBIT_COLLINEARITY_DIAGNOSTIC.md`; `preregistration_observer1.md`; `preregistration_observer2.md`.

---

## Section 1 — Audit of the five suspected smuggling patterns

Evidence format: `path:line` plus inline quote where load-bearing.

### Pattern 1 — "All three nulls are continuous-layer, all preserve marginal + linear autocorrelation."

**CONFIRMED, and worse than stated.**

- Block bootstrap (`phase_p_observer1.py:340-356`): Politis–Romano on the daily `log_rv` array. Preserves exact marginal distribution and approximately preserves the autocorrelation up to block length. Block length chosen from the ACF-crossing-0.1 lag (`phase_p_observer1.py:327-338`) — the block length is **tuned to the linear autocorrelation the null is trying to preserve**. This is the strongest possible preservation of linear memory consistent with a bootstrap.
- AR(1)-matched (`phase_p_observer1.py:381-399`): simulates a single-lag linear Gaussian process. Preserves lag-1 linear autocorrelation by construction; destroys higher-order and non-linear structure.
- Phase-randomized (`phase_p_observer1.py:422-443`): Theiler et al. 1992. Preserves the **entire power spectrum**, therefore preserves **every linear autocorrelation coefficient** exactly; destroys only phase (i.e. time-reversal and non-linear couplings).

All three nulls preserve linear autocorrelation structure. Combined with the Bonferroni-AND gate (p < 0.0167 on **all three** — `phase_p_observer1.py:512`, `phase_p_observer2.py:544-547`), rejection requires the statistic to be **non-linear, non-stationary, and phase-sensitive**. That is a narrow window by design.

Not found in the scripts: a **QA-native null** (e.g. i.i.d. sample from the marginal orbit distribution, or a random QA-legal generator walk from the real initial state). The three chosen nulls are inherited from general time-series convention without a QA-specific justification. The pre-regs cite Politis–Romano, Theiler, and generic AR(1) — none references Theorem NT or the QA discrete layer. The only QA touches on the null side are that edges are frozen and the pipeline is re-run; nothing about the null itself is QA-aware.

**The Bonferroni-AND across three near-colinear nulls has an opposite-direction problem** too: it inflates type II error. If a QA signal is partly captured by linear autocorrelation (very plausible for volatility clustering), it will show up as elevated DM under the real data *and* elevated DM under the nulls, so |DM|_real does not sit in the tail of the null distribution. This is exactly what the results show: real |DM| ≈ 0.52 sits near the center of null distributions whose std ≈ 1.0–1.4 (`phase_p_observer1_results.json:86-107`). The design makes it essentially impossible for any statistic that shares spectral support with log-RV to reject. The protocol (`QA_FINANCE_OBSERVER_ITERATION.md §1.2`) demands "strictly higher rejection rate against **autocorrelation-preserving** nulls" — that language explicitly commits to nulls that preserve the thing QA finance most likely looks at, which in this framing is volatility persistence. Any observer whose signal lives in vol persistence is pre-eliminated.

### Pattern 2 — "DM on QLIKE is a continuous log-ratio loss on continuous target."

**CONFIRMED.**

- QLIKE (`phase_p_observer1.py:194-197`): `L = rv_true/rv_pred − log(rv_true/rv_pred) − 1` (Patton 2011). Continuous in both inputs. Applied to `rv_true = exp(y_test)` and `rv_pred = exp(log_rv_pred)` (`phase_p_observer1.py:296-305`, `phase_p_observer2.py:368-372`).
- DM (`phase_p_observer1.py:201-219`): Newey–West HAC on the continuous loss-differential series (Newey & West 1987).

This is standard econometric practice, but note the chain of compositions it bolts onto the QA layer:

1. Discrete orbit label at t−1 →
2. Linear regression coefficient selection →
3. Linear prediction of continuous `log_RV_{t+1}` →
4. `exp(·)` →
5. QLIKE against continuous truth →
6. Time-series mean with HAC variance.

Steps 2–6 are all observer-side; that is spec-legal per `QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1 §"Projection / Validation Layer"`. What is not spec-legal is the **design inference**: QA orbit class is used as a *switching variable for OLS coefficients*. The orbit label is one of three values; it is collapsed inside an OLS β-matrix, which is linear and smooth in the target. In Observer 2 the same label is collapsed into an additive dummy on the continuous target, which is even more linear. The orbit categorical variable is being tested against the SAME loss function that rewards capturing the conditional mean of a continuous process — a loss the whole HAR machine was designed around.

So QLIKE-on-log-RV-prediction is observer-legal, but it is also the statistic most aggressively absorbed by HAR's three AR-like features. **Using QLIKE as the final discriminator while keeping HAR as the absorbing baseline is what makes this a narrow channel for QA structure to show through.**

### Pattern 3 — "Decile-rank observer makes b_{t+1}−b_t a linear-autocorrelation derivative; HAR's β_m absorbs it."

**CONFIRMED STRUCTURALLY; the collinearity diagnostic refutes the verbal story but confirms the mechanical effect.**

The observer is `b_t = qa_mod(bisect_left(edges, log_RV_t) + 1, 9)` where edges are the 9-quantiles of training-fold log-RV (`phase_p_observer1.py:86-105`, `phase_p_observer2.py:147-180`). This is strictly monotone in `log_RV_t`. Then `e_t = ((b_{t+1} − b_t − 1) mod 9) + 1` via [209] (`phase_p_observer1.py:109-114`). So `e_t` is a non-linear **but piecewise-monotone** function of the **difference of decile ranks of two consecutive log-RVs**.

The orbit label at scale k=1 is a non-linear function of `(b_t, e_t) = (rank_t, rank_{t+1} − rank_t mod 9)`. On the dummy-level this is nearly orthogonal to HAR (max VIF 4.43, max individual-dummy R² 0.42, `QA_HAR_ORBIT_COLLINEARITY_DIAGNOSTIC.md §1`), so the crude "HAR absorbs it" narrative is wrong. But the collinearity diagnostic *confirms the mechanical outcome*: partial R² of the 8 orbit dummies beyond HAR is **0.00141**, F(8, 4980) = 0.88, p = 0.534 (same file, Test C). The dummies are orthogonal to HAR **and** carry no marginal forecast content for `log_RV_{t+1}`.

The deeper design problem is this: when the observer is a rank of the same scalar whose next value is being predicted, "is this observation in decile 3 vs decile 7" is a projection of the same variable that already appears on both sides of the regression. HAR doesn't need to absorb the dummy through β_m; HAR already has `log_RV_t`, and the orbit label is a coarsened 9-level rounding of `log_RV_t`-joined-with-its-next-increment. Anything the rank signals about the conditional mean of `log_RV_{t+1}` is either (i) already in the `log_RV_t` column, or (ii) in the increment `log_RV_{t+1} − log_RV_t`, which is precisely the **forecast error** HAR is fitting. The scalar-rank observer cannot add information to a regression whose regressors are already a smooth transform of that scalar.

This is the pattern that matters. It doesn't look like p-hacking — the design is textbook HAR econometrics — but it guarantees that the QA layer lives entirely inside the null-space of the HAR design matrix for this particular target. **The observer and the loss are colinear.**

### Pattern 4 — "Target is continuous log_RV_{t+1}; QA-native targets unexplored."

**CONFIRMED.**

`y = log_rv[train_idx + 1]` in both scripts (`phase_p_observer1.py:270`, `phase_p_observer2.py:329`). No QA-native target is defined anywhere in either script. Candidates that **could** have been tested but were not:

- next-orbit-class conditional distribution (3-class categorical)
- orbit transition matrix off-diagonal mass vs null
- orbit-run-length distribution
- next-period **b** (9-class categorical)
- f-drift magnitude (`| f(b_{t+1}, e_{t+1}) − f(b_t, e_t) | mod m`), which exists in OB (2026-03-25) as "QA analog of GARCH σ²"

The decision to forecast the same continuous variable HAR forecasts means the yardstick for "better" is exactly the yardstick HAR was engineered to optimize, on the dataset HAR was engineered around. QA would be scored against Corsi's home field using Corsi's rules.

### Pattern 5 — "Observer 1 → 2 → proposed 3 all share the same observer family (decile-rank of log-RV)."

**CONFIRMED.**

- Observer 1: `b = decile_rank(log_RV_t^{(1)})`
- Observer 2: three streams `b_k = decile_rank(log_RV_t^{(k)})`, k ∈ {1, 5, 22}
- Proposed Observer 3 (`QA_FINANCE_OBSERVER_ITERATION.md §2 Observer 3`): same featurization extended to N=5 assets, with cross-asset synchrony added. `b_{t,k,i} = decile_rank(log_RV_{t,k,i})` per asset.

All three are the same observer family: **decile rank of log realized variance**, varied over `(asset, timescale)`. The EEG 1→4 progression was (threshold) → (argmax band) → (topographic k-means) → (correlation-matrix eigenspectrum). Each step is a *structurally different projection of the same raw signal* (amplitude → frequency → spatial → inter-channel mode). The finance 1→2→3 sequence is a parameter sweep over `(window, asset, count of streams stacked)` of the **same scalar quantile observer**. The protocol's own definition (`QA_FINANCE_OBSERVER_ITERATION.md §1.2`: "Parameter tweaks to the same observer (different window size, different quantile count) are **not** a new observer") is violated in spirit if not in letter: O1 and O2 are parameter tweaks of each other, and proposed O3 is a cross-sectional extension of O2's streams. Three nominally separate observers, one observer family — the EEG ladder iterated across observer *families*, not within one.

### Additional smuggling patterns not on the list

**Pattern 6 — The orbit distribution is near-trivially determined by the decile choice.**

On the training fold, Observer 1 reports cosmos 87.9% / satellite 9.6% / singularity 2.5% (`phase_p_observer1_results.json:39-43`). The uniform-tuple baseline on mod-9 is 88.9% / 9.9% / 1.2%. The observed distribution is **within 1–2% of uniform** on two of three classes. Singularity is the only class showing meaningful enrichment (2.5% vs 1.2% = z ≈ 8), which is a 126-observation class out of 4992. Observer 2 inflates this to 530 singularities at k=22 out of 4992 by coarsening the signal (`phase_p_observer2_results.json`; cf. `QA_HAR_ORBIT_COLLINEARITY_DIAGNOSTIC.md` orbit_counts_train). But the reason the singularity count inflates is mechanical: a long rolling mean stuck in one decile (`b_{t+1} = b_t`) makes `e_t = 9`, which forces `orbit_family(9,9)` = singularity (cf. `qa_orbit_rules.py:91-92`). The observer doesn't discover singularity; it constructs it by the interaction of "rolling mean" and "decile rank." That is not QA structure, that is decile-rank algebra of averaged processes.

**Pattern 7 — Causal orbit fix applied to the only feature that survived.**

The "leakage fix" (`phase_p_observer1.py:251-276`) changes conditioning from `orbit_{t}` to `orbit_{t-1}` because `e_t = ((b_{t+1} − b_t − 1) mod 9) + 1` depends on `b_{t+1}`. Mechanically correct. But the fix moves from "forecast uses label that contains `b_{t+1}`" (DM = +2.42, rejected as leakage) to "forecast uses label that contains `b_t`, which is already in the regressor." After the fix, the label is one step stale relative to any information HAR doesn't already have. The post-fix DM of −0.52 is not weak evidence of no signal; it is weak evidence that a one-step-lagged coarse rank of what HAR already sees adds nothing — which is not a test of QA. **The fix is correct, but it pushes the label into redundancy with the HAR regressors.** The alternative (use `orbit_{t-1}` for one-step-ahead forecast of `b_{t+1}` *class*, where causality is maintained without the label collapsing into HAR's existing feature set) was not considered.

**Pattern 8 — Private QCI finding was not re-projected into Phase P.**

OB 2026-03-31T23:58 reports QCI robustness: 67/80 configurations significant, partial correlation `QCI` vs future vol | lagged RV = **−0.2154, p < 10⁻⁸** on SPY. That is a finance QA finding where the *partial correlation beyond lagged RV* is substantial and highly significant — the opposite pattern of Observer 1/2. QCI uses k-means clusters with k=6 on 6-D standardized features + window=63 (same structure OB 2026-04-05 cites in the Bearden-finance-framework analysis) rather than a scalar decile rank. QCI is the closest thing in the corpus to a validated-positive QA finance observer, and Phase P was built as if it did not exist. The Phase R audit (OB 2026-04-16T19:37) catalogues QCI as "architecturally superseded by [209]" — but [209] is the *generator-inference cert*; [209]'s contribution to finance is the cross-asset synchrony r=+0.037 that the existing-art audit also flags as "not robust stand-alone." Observer 3's proposed cross-asset synchrony is a strict return to the [209] result with cascade added. The lineage is (QCI, validated-positive on partial) → (replaced by [209] synchrony, weaker) → (Observer 3 planned as [209] extension). The validated-positive predecessor was dropped rather than ported.

### Theorem-NT legality of the compositions

QLIKE, DM, exp, block bootstrap — all observer-side compositions acting on observer-side predictions. Spec-legal per `QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1`. The T2 firewall is crossed once on input (`log_RV → qa_mod(rank)`) and not violated. A1, A2, S1, S2, T1 are respected in the code — I looked for `**2`, float `(b,e)`, direct `d` assignment, and the patterns linter rule T2-D-N warns about; none are present. **Axiom compliance is clean. The smuggling is not in the axioms — it is in the *choice of observer family and target* relative to the baseline, which the axioms do not constrain.** That is the category error the prior Claude made: checking the linter, seeing green, and concluding "QA was given a fair shot." Axioms constrain the QA layer; they do not constrain whether the observer projection puts real QA structure into contact with the target.

---

## Section 2 — Observer 1 and 2 NULL verdicts: TRUSTWORTHY or REVOKE

**REVOKE.**

The NULL verdicts are *consistent with the design* — I have no reason to doubt the numbers themselves (the code is linter-clean and the collinearity diagnostic mechanically explains them). What they do not support is any claim about QA finance. The design has:

1. Nulls that all preserve the linear autocorrelation of log-RV (Pattern 1).
2. A loss and a target engineered for HAR's home field (Patterns 2, 4).
3. An observer that is a rank of the same scalar being predicted, guaranteed to be a nonlinear-but-coarse transform of the regressors (Pattern 3).
4. A one-step-lagged conditioning label that pushes the QA information into redundancy with HAR's daily term (Pattern 7).
5. An "iteration" that is a parameter sweep within one observer family (Pattern 5), in explicit tension with the protocol's own definition.
6. An orbit distribution that is constructed by the interaction of rolling-mean + decile-rank arithmetic (Pattern 6), not an emergent QA property.

Each of these individually is defensible as a conservative design choice. Compounded, they make the experiment a test of the proposition "can coarsening `log_RV_t` to 9 deciles beat `log_RV_t` itself inside a linear regression on `log_RV_{t+1}`, tested against nulls that preserve everything linear." The answer to that proposition is known a priori: no. The NULL is overdetermined — it reflects the experimental frame, not the framework.

The same observer family (QCI: k-means k=6 on 6-D features + window=63, partialling against lagged RV) produced **r=−0.22, p<10⁻⁸ finance** in the private corpus (OB 2026-03-31T23:58). The same architecture *for text* was explicitly diagnosed as "framework-limited not domain-limited" (MEMORY reference, 2026-04-05). Every time the framework changes, the signal comes back. Every time the framework is "decile rank scalar → HAR → QLIKE," the signal disappears. That is not QA failing in finance; that is one framework failing, repeatedly, on finance.

**Minimal fix to recover trustworthiness within the decile-rank family:** impossible without giving up the HAR baseline or the DM-on-QLIKE statistic. Any decile-rank of a scalar input to a regression whose regressors are already a smooth transform of that scalar will hit the collinearity diagnostic's 0.00141 partial R² floor. The observer family itself is the smuggling vector. Do not iterate within it; exit to a structurally different projection.

---

## Section 3 — Corrected Observer 3 spec

The load-bearing fixes: use a projection that is **not** a monotone transform of the target; test a **categorical** alternative hypothesis on a **QA-native target**; build a null that destroys QA orbit structure while **preserving** HAR's linear autocorrelation so that any rejection is attributable to the QA layer, not to AC structure the nulls already let through.

### A. Observer — Signal Dynamics (canonical b, cross-asset-synchrony e)

**Choice: signed-return magnitude observer combined with cross-asset-synchrony generator.**

- `b_{t,i} = qa_mod(signed_decile(r_{t,i}) + 5, 9)` per asset i — signed-decile of **daily return** `r_t = log(close_t / close_{t-1})`, mapped into `{1..9}` so that decile 0 (strongest down) → 1, decile 4 (flat) → 5, decile 8 (strongest up) → 9. Edges computed from training-fold returns on the asset itself. (Not rank of log-RV — rank of **signed return**, which is independent of volatility magnitude.)
- `e_t` inferred from **cross-asset joint state**: `e_t = ((B_{t+1} − B_t − 1) mod 9) + 1` where `B_t = qa_mod(Σ_i b_{t,i}, 9)` is the modular sum across N=5 panel assets at time t. This is the [209] inference applied to a **panel-derived state** rather than a per-asset scalar.
- Orbit via `orbit_family(B_t, e_t, 9)`.

**Why this clears §1:**
- `b` is signed return, not log-RV. Return and log-RV are near-orthogonal at daily horizon (the squared-return link means `r` carries sign information that `|r|²` lost).
- `e` is cross-asset; it cannot be reconstructed from any single-asset time-series regression.
- `B_t` is a 9-state collapse of a 5-asset joint configuration; its conditional distribution given past `B` captures panel-level orbit structure that HAR-per-asset cannot see.
- Orbit class carries information about direction (sign), cross-sectional synchrony, and their interaction — three axes HAR does not span.

Panel: SPY, TLT, GLD, USO, UUP — the same panel [209] already has a witness on. QCI used 6 assets; 5-panel is conservative vs that precedent.

**Alternatives evaluated and rejected:**
- *Cross-sectional rank* (asset rank at time t among peers): tempting but collapses to "which asset is most volatile today" — known structural property, not QA. Rejected.
- *Realized skew/kurt*: higher moments of returns would be a structural alternative to log-RV, but they are still scalar per-asset. Worse, they require intraday bars which yfinance doesn't provide past ~60 days. Rejected on data.
- *Volume–return joint*: volume conditions matter, but adding volume introduces a second exogenous data stream and confounds the "QA adds structure" claim with "volume adds structure." Rejected.
- *Mean-crossing count*: interesting but collapses to a 1-per-window variance measure, re-introduces the Observer-1 problem under a different name. Rejected.

### B. Test statistic — Categorical orbit-transition structure

**Choice: mutual information `I(orbit_{t-1}; orbit_t)` on the panel-derived orbit sequence, compared against the QA-native null via permutation.**

Specifically:
1. Compute `{orbit_t}_{t=1..T}` on test fold (test fold defined exactly as O1/O2, `train_end = 5015`).
2. Estimate `Î_real = MI(orbit_{t-1}, orbit_t)` via plug-in estimator on the 3×3 empirical transition count.
3. Under each null (below), compute `Î_null` on the surrogate orbit sequence.
4. p = fraction of `|Î_null| ≥ |Î_real|` (two-tailed, since MI is non-negative but deviation-from-independence is signed via the χ² decomposition if we track the sign of (observed − expected) in the dominant cell).

**Why categorical statistic on a categorical structure:**
- Orbit label is 3-class. MI on the transition matrix is the natural category-measure of "does the past orbit predict the next orbit beyond the marginal." It is loss-function-free, HAR-free, and DM-free.
- It is not a regression on a continuous target, so it cannot be absorbed by any linear baseline.
- Sample size at N_test ≈ 3343 with 3 orbit classes is comfortable for a 3×3 contingency test (expected count per cell ≥ 30 under any non-degenerate marginal).

**Secondary statistic (reported but not gating):** orbit-path persistence length distribution, Kolmogorov–Smirnov vs null. Added as a descriptive diagnostic because the panel's monthly-scale behavior may manifest as longer orbit runs.

### C. Null — Marginal-orbit i.i.d. draw combined with HAR-frozen surrogate

**Three nulls, two of them QA-native:**

1. **Marginal-orbit i.i.d.** — draw each surrogate `orbit_t^{null}` i.i.d. from the empirical test-fold orbit marginal. Destroys transition structure, preserves marginals exactly. This is the QA-categorical analog of phase randomization.
2. **Permuted-generator** — keep `{b_t}` fixed; shuffle `{e_t}` uniformly at random across t. Orbit label is re-computed from `(b_t, e_t^{shuffled})`. Preserves the b-marginal and the e-marginal exactly; destroys the generator↔state coupling. QA-native. This is the cleanest test of whether the [209] generator inference is producing coupled structure or noise.
3. **HAR-frozen residual surrogate** — fit classical HAR (Corsi 2009) on real returns (not RV), extract residuals, surrogate returns = HAR predictions + phase-randomized residuals. Re-quantize, re-build B, re-infer e, re-classify orbits, compute MI. This null preserves exactly the HAR-captured structure and **destroys everything else**. If the real MI rejects this null, the rejected quantity is strictly non-HAR structure.

Null #3 is the inverse of the three Phase P nulls: instead of destroying QA and preserving linear AC, it preserves linear AC via HAR and destroys QA. Any rejection of the HAR-frozen null is evidence of QA-native information beyond HAR.

Bonferroni across three nulls: p < 0.0167 each (same as O1/O2). Effect-size floor: `Î_real ≥ Î_null_mean + 0.01 nats` (bits would bias toward easier rejection; nats with floor aligned to QCI's effect-size magnitude).

### D. Target variable — Categorical

**Target:** `orbit_t` given `{(b_{t-1}, e_{t-1}), ..., (b_{t-k}, e_{t-k})}` for small k. We are not predicting `log_RV_{t+1}` — we are testing whether the orbit sequence carries transition information the nulls do not.

Framed as a hypothesis on a categorical target avoids (i) QLIKE's continuous-loss absorption, (ii) HAR's home-field advantage, and (iii) the observer-absorbs-into-regressors collapse. The categorical target is also test-set-consistent: no train/test regression coefficient fit is needed.

### E. Axiom-compliance per step

| Step | Layer | Axioms exercised |
|---|---|---|
| 1. Load returns for 5 assets | Observer-layer input | none (continuous float) |
| 2. Compute decile edges per asset on training fold | Observer-layer | none (continuous float) |
| 3. `b_{t,i} = qa_mod(signed_decile(r_{t,i}) + 5, 9)` | **T2 firewall crossing (once)** | A1 (b ∈ {1..9}), S2 (int after quantization) |
| 4. `B_t = qa_mod(Σ_i b_{t,i}, 9)` | QA discrete layer | A1, S2 (int arithmetic, qa_mod enforces no-zero) |
| 5. `e_t = ((B_{t+1} − B_t − 1) mod 9) + 1` | QA discrete layer | A1 on e, A2 not invoked (no d/a derivation yet) |
| 6. `orbit_t = orbit_family(B_t, e_t, 9)` | QA discrete layer | Full axiom set inside `orbit_family` |
| 7. `MI(orbit_{t-1}, orbit_t)` on test fold | Projection layer | T2-d (statistics in projection) |
| 8. Null surrogates (three kinds) | Projection layer (nulls 1,2 are QA-native but act outside the discrete layer; null 3 is observer-level) | none violated |
| 9. Two-tailed p via empirical null | Projection layer | T2-d |

No `**2` anywhere (S1). Python int throughout the QA layer (S2). No continuous time (T1). No feedback from projection to QA (T2-e). `B` and `e` are always in {1..9} by construction (A1). `d` and `a` not invoked (they are not used in MI-on-orbit-transition; if one later wants f-drift as a secondary stat, A2 is invoked there). Compliant.

---

## Section 4 — Launch readiness

**Ready to pre-register, with three non-negotiable components added.**

### Minimum pre-reg content

1. **Statistic, threshold, null tuple locked** — MI on 3×3 orbit transition matrix; two-tailed p; Bonferroni p < 0.0167 across three nulls.
2. **Panel and data hashes frozen** — SHA-256 of the five CSVs as at commit time. No re-fetch between pre-reg commit and test-fold evaluation.
3. **Train/test split identical to O1/O2** — same `train_fraction = 0.60`, same `train_end_idx = 5015` for SPY, same range per asset; no re-optimization.
4. **Observer family declared structurally new** — not a parameter sweep; signed-return b + panel-sum B + cross-asset generator e is a different projection from decile-rank log-RV. Commit this in the pre-reg body so the iteration budget accounting is honest.
5. **Two pre-registered kill gates on training-fold diagnostic** — run **before** the test-fold statistic:
   - `I_train(orbit_{t-1}, orbit_t) − I_train_marginal_null_mean ≥ 0.01 nats`
   - chi-square of 3×3 transition matrix vs independence has p < 0.01 on training fold
   Failing either gate = declare NULL on training fold, do **not** compute test-fold statistic. This is the mechanism the collinearity diagnostic retrospectively proposed for O3 (partial-R² floor + F-test gate) ported to a categorical test.
6. **Null-compatibility check** — the HAR-frozen-residual surrogate must be verified to preserve the training-fold autocorrelation of returns to within 5% at lags 1–22. If it does not, the null is not actually preserving HAR structure and should be re-fit before test.
7. **Separate blinded scoring script** `run_blinded_orbit_transition.py` that accepts the panel hash + pre-reg hash + null seeds as input and emits the three p-values + MI estimate. Same pattern as the protocol demands for HAR-RV.
8. **No O4 fallback pre-authorized.** Per `QA_FINANCE_OBSERVER_ITERATION.md §1.3`, a fail on O3 under this spec consumes iteration 3 of 5. Commit to the stopping rule.

### Remaining gaps the pre-reg cannot close by itself

- **Panel-sum observer is one of many legitimate choices.** Picking it over (for example) `(b_{t,i}, b_{t,j})` pair-matrix is a design call. Justification in this doc is: panel-sum is the minimal extension of [209]'s existing witness architecture, and its one-number output makes the transition MI tractable. If pre-reg reviewers want a second observer as a robustness check, add the cross-asset-synchrony statistic from `QA_FINANCE_OBSERVER_ITERATION.md §2 Observer 3` as a secondary — but keep it secondary and Bonferroni-penalize.
- **The QCI (r=−0.22, p<10⁻⁸ private) has not been re-run on the public corpus under the current protocol.** Before O3 consumes a budget slot, re-running QCI under the Phase P null architecture is a cheaper test of the framework. That is not a reason to block O3, but the design doc should name it as "pre-work: validate QCI survives the Phase P null stack before spending an iteration on a new observer family."
- **If the HAR-frozen null is too aggressive** (i.e. passes everything because HAR leaves too little residual structure to test), fall back to an AR(p)-frozen null with p selected on training by BIC, same surrogate logic. Pre-register the fallback so it is not a post-hoc choice.

### Honest pushback on my own spec

A skeptic would say: you have replaced one QA-native-rank observer with another QA-native-rank observer (signed-return decile), swapped target from continuous to categorical, and built a null you believe destroys the right thing. You will reject on MI if (i) the panel-sum observer has real transition structure, which is *likely* given QCI's r=−0.22 precedent, or (ii) something in the data's daily cross-sectional pattern makes 9-way-decile-sum land non-uniformly. If (ii), this spec is a different smuggling pattern. The kill-gate at training-fold MI ≥ 0.01 nats is the honest guard against that — it tests whether the structure is there before spending the test fold.

The harder adversarial question: if this spec also returns NULL, does it update toward "QA doesn't add in finance"? **Partially, yes** — if panel-sum + cross-asset-e + categorical MI on 3×3 + HAR-frozen null all return NULL, the interpretation is "the most promising QA-native observer family consistent with [209]'s existing witness does not survive a properly-constructed QA null." That is a substantial update. It is not evidence against Theorem NT; it is evidence that the specific observer family and target pair I have chosen do not contain exploitable QA-native signal. The MEMORY rule `feedback_qa_never_dead_implementation_first.md` applies: suspect implementation first, re-audit observer family second, declare "observer gap across this class" third. Per `feedback_finance_not_closed.md` the framing is still observer gap, not framework verdict — but the iteration budget does get consumed.

If Will's prior (signal disappears only in finance, suggesting smuggling) is right, this spec should produce a positive. If Will's prior is wrong, this spec produces another NULL and the corpus of finance NULLs grows — but for a different reason than Patterns 1–7 above, which is itself progress. Either outcome is informative, which O1 and O2's designs did not achieve.

**Launch readiness: go, pending pre-reg commit and QCI re-validation on the public corpus (1–2 hour pre-work).**

---

## References

- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics* 7(2):174–196. DOI: 10.1093/jjfinec/nbp001.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3):253–263. DOI: 10.1080/07350015.1995.10524599.
- Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica* 55(3):703–708. DOI: 10.2307/1913610.
- Patton, A.J. (2011). "Volatility Forecast Comparison Using Imperfect Volatility Proxies." *Journal of Econometrics* 160(1):246–256. DOI: 10.1016/j.jeconom.2010.03.034.
- Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association* 89(428):1303–1313. DOI: 10.1080/01621459.1994.10476870.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J.D. (1992). "Testing for nonlinearity in time series: the method of surrogate data." *Physica D* 58(1-4):77–94. DOI: 10.1016/0167-2789(92)90102-S.

Companion artifacts cited: `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/` (cert [209] generator inference); `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md`; `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`; `docs/theory/QA_HAR_ORBIT_COLLINEARITY_DIAGNOSTIC.md`; `preregistration_observer1.md`; `preregistration_observer2.md`; `qa_orbit_rules.py`; `qa_lab/qa_observer/core.py`; `qa_lab/phase_p_observer1.py`; `qa_lab/phase_p_observer2.py`; `qa_lab/har_orbit_collinearity_check.py`; MEMORY `feedback_qa_never_dead_implementation_first.md`; MEMORY `feedback_finance_not_closed.md`; MEMORY `feedback_theorem_nt_observer_projection.md`; MEMORY `feedback_map_best_to_qa.md`.

OB thoughts referenced: 2026-03-25T04:12:53 (QCI robustness, partial r=−0.2154 p<10⁻⁸); 2026-03-28T10:15–10:46 (EEG Observer 1→3 progression); 2026-04-09T01:56 (RNS eigenspectrum, ΔR²=+0.138 p=0.001); 2026-04-09T04:16 (Signal Dynamics canonical observer +0.085 beyond Obs3); 2026-04-16T22:03 (Phase P collinearity diagnostic); 2026-04-16T19:37 (QA-finance existing-art audit).

---

*Word count ≈ 3,800.*
