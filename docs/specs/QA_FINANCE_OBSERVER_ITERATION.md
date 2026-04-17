# QA Finance Observer-Iteration Protocol

*Specification session: 2026-04-16. Scope: formalize how to iterate observer designs in QA finance without p-hacking, modeled on the EEG Observer 1 → 2 → 3 progression. Sibling inputs: `docs/theory/QA_FINANCE_SOTA_SURVEY.md` (Phase R candidate ranking), `docs/theory/QA_FINANCE_EXISTING_ART_AUDIT.md` (Phase R existing-art inventory), user memory `feedback_finance_not_closed.md` (framing rule).*

Primary sources grounding this protocol: Corsi (2009) HAR-RV DOI 10.1093/jjfinec/nbp001; Lo & MacKinlay (1988) variance-ratio DOI 10.1093/rfs/1.1.41; Diebold & Mariano (1995) DOI 10.1080/07350015.1995.10524599; Politis & Romano (1994) stationary block bootstrap DOI 10.1080/01621459.1994.10476870; Ansari & Peter (2025) Chronos-2 arXiv:2510.15821; Benjamini & Hochberg (1995) FDR DOI 10.1111/j.2517-6161.1995.tb02031.x. See `## References` at the end.

---

## 1. General protocol

### 1.1 What an "observer" is in QA

An **observer** is a function `O: X_cont → {1, ..., m}^T` that maps continuous-domain data `X_cont` (returns, realized volatility, prices) into a discrete QA state sequence over the integer alphabet `{1, ..., m}`. Under Theorem NT (Observer Projection Firewall), the observer crosses the T2 firewall **exactly once on the input side**: continuous data enters, a discrete sequence leaves, and from that point onward QA arithmetic (`qa_step`, `orbit_family`, f-values, CRT cross-modulus checks) operates on integers only. Float × modulus → int casts inside the QA layer are T2-b violations (see `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`).

Concretely, an observer in this protocol is specified by three parts:

1. **Featurization** `φ: X_cont → R^d` — a continuous pre-projection (e.g., log realized-variance at timescale k, cross-asset correlation eigenspectrum, turn-angle θ). May be vector-valued.
2. **Quantization** `q: R^d → Z_m` — a discrete map onto `{1, ..., m}` via `qa_mod` (canonical definition in `qa_lab/qa_observer/core.py:27`, `((x - 1) % m) + 1`). Quantile-rank, k-means cluster index, or decile bucket are all admissible; raw percentile-bin with analyst-chosen edges is admissible but has already failed twice (see §1.4).
3. **Generator inference** — under cert [209] canonical algebra, `b_t = q(φ(x_t))` and `e_t = ((b_{t+1} − b_t − 1) mod m) + 1`. This is the A1-compliant unique inverse (`qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/qa_signal_generator_inference_cert_validate.py:49-74`). No hand-chosen CMAP, no hardcoded QUINTILE_TO_STATE lookup. An observer that names its own `e` via a regime table instead of [209] inference is an **analyst-chosen observer**, not a generator-inferred observer, and must be flagged as such.

An observer does **not** specify the downstream statistic. The statistic (e.g., orbit-family χ² on stress-vs-calm; Diebold-Mariano on forecast losses) is a separate decision and is pre-registered before the observer is tested (§1.3).

### 1.2 What "better observer" means operationally

Observer `O_{k+1}` is **better than** `O_k` iff it produces a **strictly higher rejection rate against autocorrelation-preserving nulls on held-out data**, measured by a pre-registered statistic at a pre-registered significance threshold.

The nulls used in this protocol must include **at least one QA-native null** — one whose construction operates entirely in the QA discrete layer. Examples: permuted-generator (shuffle `e_t` keeping `b_t`, recompute orbit); marginal-orbit-iid (draw `orbit_t` iid from empirical orbit marginal); random-legal-walk (random `(b, e)` walk on {1..m} × {1..m} started at the same initial state). Observer-side nulls (block-bootstrap on continuous observables per Politis & Romano 1994, AR(1)-matched surrogates, phase-randomized per Theiler et al. 1992, HAR-frozen residual) may supplement but cannot substitute. The rationale is Theorem NT: QA dynamics is discrete, so a null for "does QA structure exist" must destroy QA structure but preserve everything else, and that operation is most cleanly expressed at the discrete layer. Observer-side nulls preserve linear autocorrelation of the observer's continuous input; a QA signal that shares spectral support with its input will not produce a tail-event statistic against such nulls (framework-audit Pattern 1, grounded in `phase_p_observer1.py:340-443`).

"Rejection rate" means: on held-out data, the observed statistic exceeds the null's critical value for the pre-registered threshold. "Strictly higher" means `O_{k+1}` rejects all three nulls when `O_k` rejects only two, or `O_{k+1}` rejects at the pre-registered threshold when `O_k` only rejects at a weaker threshold. Parameter tweaks to the same observer (different window size, different quantile count) are **not** a new observer — they are the same observer under a design variation; each such variation is its own iteration and counts against budget (§1.5).

Effect-size improvement without null rejection on held-out data does **not** count. This rule exists to avoid the Phase 2.5 mistake, where a candidate that "numerically improves" under a specific architecture never cleared matched nulls (`docs/theory/QA_FINANCE_EXISTING_ART_AUDIT.md` §2).

**Observer-family discipline.** An "observer family" is identified by its quantization class: {rank-of-scalar, k-means-of-feature-vector, sign-pattern-index, correlation-eigenspectrum, joint-bucket-of-multiple-features} are distinct families. Iteration `O_{k+1}` is admissible as a *new observer* only if it crosses family from `O_k`. Variation that keeps the family (different scalar, different asset, different timescale, cross-section of the same rank) is a *within-family variant* and is reported as a single observer with its variants rather than as a new iteration. The EEG progression (§1.5) is a canonical cross-family example: threshold-fallback, dominant-band argmax, topographic k-means, RNS correlation-matrix eigenspectrum are four distinct quantization classes. Iteration in finance is held to the same discipline.

### 1.3 Iteration rules that prevent p-hacking

**Held-out discipline.** Data is split *before* observer design begins into a training fold, a held-out test fold, and (where available) a second OOS fold after any foundation-model pretraining cutoff. Observer design is tuned on the training fold only. The test fold is locked: it is not looked at until the designer commits to "this is the next observer" and runs the blinded scoring script.

**Pre-registration.** Each observer iteration `k` commits a tuple `(statistic_k, threshold_k, null_k)` to `preregistration.md` under the cert directory **before** test-fold evaluation. The commit is to git; after pre-registration is committed, the three nulls and the statistic are fixed. Any change to `statistic_k`, `threshold_k`, or the null design after pre-registration constitutes a new iteration (i.e., counts against budget) and must include a diff-log entry explaining why the prior design was abandoned.

**Blinding.** Test-fold statistics are computed by a separate script (`run_blinded_eval.py`) that takes the observer spec + data path as input and emits a result JSON. The observer designer does not see test-fold results until they have frozen their candidate observer. This is the one rule most easily violated by habit; enforcement is procedural, not technical, and should be treated with the same discipline as "don't look at the test set" in ML practice.

**Stopping rule.** Iteration halts when either condition holds:
- Three consecutive observers fail to improve the pre-registered statistic on held-out (even if one or two reject training-fold nulls).
- A budget cap of **5 observers** is reached — the EEG precedent (§1.5) converged in 3; a 5-observer cap is generous headroom.

**Family-wise error control.** Over the iteration sequence, Bonferroni correction is applied to the per-iteration threshold: if pre-registered threshold is p < 0.05 and five observers are evaluated, the effective threshold is p < 0.01 for any individual observer's claim. Hierarchical FDR (Benjamini & Hochberg 1995) may substitute when iterations are planned in advance as a ranked family; Bonferroni is the default because iteration order in observer design is typically path-dependent.

**Freeze.** The first observer that rejects all three nulls on held-out at the pre-registered Bonferroni-adjusted threshold **freezes** the iteration. No "just one more tweak" is allowed. The freeze produces the cert witness; further observer-design work becomes a new cert family (new pre-registration, new held-out split).

### 1.4 Memory rules this protocol inherits

- `feedback_qa_never_dead_implementation_first.md`: If an observer fails on the *training* fold where structure is presumed to exist (e.g., the single-asset control that should largely reproduce a prior NULL), assume implementation error first, debug the observer before iterating.
- `feedback_adversarial_testing.md`: Real data first, synthetic second. Never re-implement validated code — bolt the null test onto the existing observer pipeline. Check the null design for circularity (the block-bootstrap block size must be independent of the quantization window).
- `feedback_finance_not_closed.md`: When an observer fails, the finding is "observer design gap," not a framework verdict against QA. This is the framing rule that makes iteration legitimate.

### 1.5 EEG precedent

The EEG observer progression is **documented in Open Brain and in the runtime scripts, but is not cert-formalized**. No `qa_alphageometry_ptolemy/*eeg*/` cert family exists; glob-search returns empty. The progression lives in: OB 2026-03-28T10:15 (Observer 1 baseline, synthetic-tuned thresholds), OB 2026-03-28T10:29 (Observer 1 calibrated), OB 2026-03-28T10:46 (three-observer comparison), OB 2026-04-08T01:39 (RNS baseline), OB 2026-04-08T01:56 (RNS eigenspectrum iteration). Runtime artifacts: `eeg_orbit_classifier.py`, `eeg_rns_observer.py`, `eeg_rns_real_test.py`.

The hard progression on CHB-MIT chb01 (n=80, nested-model ΔR² test; Shoeb 2009 CHB-MIT corpus):

| Observer | Featurization | ΔR² | p | Verdict |
|---|---|---:|---:|---|
| 1. Threshold-fallback | Relative bandpower thresholds (hand-picked for synthetic) | +0.0023 | 0.88 | NULL |
| 2. Dominant-band | Argmax over bandpower | +0.0098 | 0.58 | NULL — 1/f dominance collapses states |
| 3. Topographic k-means | k-means on normalized per-channel RMS | +0.0447 | 0.087 | Trending; confirmed positive at n=640 across 10 patients (ΔR²=+0.210) |
| 4. RNS eigenspectrum | Broadband correlation-matrix eigenvalues, mirror-paired | +0.138 | 0.001 (\*\*) | Freeze — rejects all three nulls on chb01 held-out |

Each step fixed an observer-layer deficiency identified by the previous iteration: (1→2) synthetic-tuned thresholds unusable on real EEG; (2→3) dominant-band loses spatial co-activation structure; (3→4) topographic k-means misses inter-channel mode structure that the eigenspectrum captures. The iteration converged in 3 steps to a clear positive, and a 4th step improved it further; budget was not exhausted.

The two rules that were actually followed (and that I am inheriting here): each observer was a *structural redesign* of the previous (not a hyperparameter tweak), and each iteration's failure mode was diagnosed mechanistically before moving to the next. The rules that were *not* formalized and that this protocol adds: held-out discipline, pre-registration, blinded scoring, Bonferroni across iterations, freeze.

**The cross-family property is normative, not incidental.** The EEG progression worked because each iteration was mathematically a different projection of the same underlying signal — amplitude cross a frequency-band cutoff (Obs 1), frequency-band argmax (Obs 2), spatial k-means over per-channel RMS (Obs 3), inter-channel correlation-matrix eigenspectrum (Obs 4). Each successive observer captured information the previous one discarded by construction. The finance iteration inherits this property: an iteration that keeps the rank-of-scalar family (e.g., decile rank of log-RV at a different timescale, or extended cross-asset) is *not* an iteration under this protocol — it is a variant of the previous observer. The iteration counter advances only on family crossings.

---

## 2. Decile-rank log-RV observer — three design variants

*The three designs below constitute a single observer (quantization family: rank-of-scalar-log-RV) with variants (Observer 1 = single-asset single-timescale; Observer 2 = stacked-timescale per-asset; Observer 3 = cross-asset extension). Per §1.2 family discipline, this is one iteration in the 5-iteration budget, not three. Phase P executed Observers 1 and 2 on SPY daily log-RV and reached NULL on both; the framework-audit revoked those NULL verdicts as framework-limited (decile-rank + DM-on-QLIKE + AC-preserving nulls); this section is retained for historical/traceability purposes and as the baseline variant the next family-crossing observer must beat.*

The Phase R top candidate was HAR-RV + orbit-conditioning (Corsi 2009, score 27 in the SOTA survey). The variant ladder below maps onto the HAR three-timescale cascade at daily / weekly (5-day) / monthly (22-day) windows.

**Axiom-compliance note (load-bearing).** The tuple-shape match `(daily, weekly, monthly) ↔ (b, e, d, a)` is **only** superficial. The QA algebra `a = b + 2e` does not hold for monthly RV derived from daily and weekly components — monthly RV is an average over 22 days, not 1 + 2·5. Claiming the identity would violate A2. The correct form is **three stacked [209] applications at timescales k ∈ {1, 5, 22} trading days**, with `b_k` from `qa_mod` of log-RV at timescale k, `e_k` inferred via [209] from `b_k`'s forward step, and `d_k = b_k + e_k`, `a_k = b_k + 2·e_k` derived per-timescale per the canonical [209] algebra. The three timescales are orthogonal observers stacked in parallel, not components of one tuple.

### Observer 1 — single-timescale per-asset (baseline control)

- **Featurization.** `RV_t^{(1)} = sum of squared 5-min log-returns within trading day t` for one asset (SPY). Take `log(RV_t^{(1)})` to stabilize scale. Barndorff-Nielsen & Shephard (2002) DOI 10.1111/1467-9868.00336 establishes the realized-variance estimator.
- **Quantization.** `b_t = qa_mod(decile_rank(log_RV_t^{(1)}), m=9)`. The decile rank is computed on the training fold and frozen; held-out uses the training-fold decile edges (no data leakage). Modulus choice `m=9` is theoretical (primary QA modulus); `m=24` applied is a parameter variation and does not count as a new observer.
- **Generator.** `e_t = ((b_{t+1} − b_t − 1) mod 9) + 1` via [209] inference. Orbit class = `orbit_family(b_t, e_t, 9)` via `qa_orbit_rules.py:72`.
- **No cross-asset, no cascade.** This is the control.
- **Primary statistic.** Orbit-conditioned HAR-RV vs pooled HAR-RV, Diebold-Mariano (1995) test on 1-day-ahead forecast losses (QLIKE). Orbit-conditioned HAR-RV fits three separate `(β_0, β_d)` regressions, one per orbit class, and picks coefficients by the current observation's orbit. Pooled HAR is the classical `RV_{t+1} = β_0 + β_d·RV_t + β_w·RV_t^{(w)} + β_m·RV_t^{(m)} + ε` (Corsi 2009).
- **Nulls.** (a) Stationary block-bootstrap (Politis & Romano 1994) on log-RV with block length = ceil(5 × lag at which ACF crosses 0.1); (b) AR(1)-matched with lag-1 AC fit on training; (c) phase-randomized log-RV (Theiler et al. 1992).
- **Pre-registered threshold.** DM p < 0.05 per null, Bonferroni-adjusted to p < 0.05/3 = 0.0167 across the three nulls. Expected outcome: **NULL**. This observer is deliberately close to the Phase 2.5 setup and should largely reproduce its non-rejection (`qa_lab/phase2_5_quantization_compare.py`, 3/15 rejects, not at full-gate). The point is to verify the pipeline end-to-end on the weakest observer before adding structure. If Observer 1 *unexpectedly* rejects all three nulls, stop and audit for data leakage.

### Observer 2 — three-timescale stacked per-asset (cascade)

- **Featurization.** Three parallel log-RV streams at k ∈ {1, 5, 22}: `log_RV_t^{(k)}`.
- **Quantization.** `b_{t,k} = qa_mod(decile_rank(log_RV_t^{(k)}), m=9)` per timescale. Three independent b-streams. Decile edges frozen per timescale from training fold.
- **Generator.** `e_{t,k} = ((b_{t+1,k} − b_{t,k} − 1) mod 9) + 1` via [209] per timescale. Orbit class per (t, k) via `orbit_family(b_{t,k}, e_{t,k}, 9)`.
- **Primary statistic.** DM test of the cascade-augmented HAR-RV `RV_{t+1} = f(RV cascade, orbit_{t,1}, orbit_{t,5}, orbit_{t,22}, orbit_{t,1} × orbit_{t,5})` vs classical HAR. The orbit × scale interaction is the QA-specific signal. Corsi 2009's classical HAR over AR(1) baseline improves RMSE by ~30% on SPX; a plausible sub-fraction for the QA augmentation is **RMSE reduction ≥ 2% absolute, DM p < 0.0167** (Bonferroni across three nulls). This threshold is conservative — it accepts a ~7% sub-share of Corsi's headline effect, large enough to be economically interesting, small enough that finding it inside HAR's shadow is not trivial.
- **Nulls.** Same three as Observer 1, applied to the cascade-augmented residuals.
- **Modulus.** Primary `m=9`. If `m=9` rejects, a single confirmatory `m=24` run is logged (not a new iteration; same observer under applied-modulus variation). `m=24` is the applied modulus per the QA architecture; its role here is robustness, not a separate hypothesis.

### Observer 3 — three-timescale cross-asset cascade

- **Featurization.** Three parallel log-RV streams at k ∈ {1, 5, 22} per asset, for the N=5 asset panel {SPY, TLT, GLD, USO, UUP} (same panel as [209]'s validated-positive witness). At each (t, k), featurize as the panel vector `(log_RV_{t,k,1}, ..., log_RV_{t,k,N})`.
- **Quantization.** Per-asset decile rank → `b_{t,k,i}` as in Observer 2. Cross-asset generator synchrony at (t, k) is the fraction of assets whose `e_{t,k,i}` falls in the same mod-9 residue class. This is the [209] cross-asset synchrony statistic, which already returned `r = +0.037, p = 0.025` on this exact panel (`sgi_pass_default.json:71`).
- **Generator.** `e_{t,k,i}` per asset via [209]; synchrony `S_{t,k} = max_r (1/N) · |{i : e_{t,k,i} ≡ r mod 9}|`.
- **Primary statistic.** DM test of cascade + synchrony-augmented HAR vs classical HAR. Synchrony enters as three interaction terms `S_{t,k}`. Primary alternative hypothesis: synchrony at monthly scale (k=22) gates cascade amplification. Threshold: DM p < 0.0167 across all three nulls, and **raw correlation of S_{t,22} with next-period RV ≥ +0.05** (half a standard deviation above the [209] witness of +0.037, to avoid accepting the [209] seed as evidence for its own extension).
- **Nulls.** Block-bootstrap, AR(1)-matched, phase-randomized — all applied per-asset before synchrony construction, so that cross-asset synchrony is computed on surrogates with destroyed cross-asset phase relationships. The block bootstrap uses a single synchronized block-draw index across assets to preserve (but not create) cross-asset structure present in each bootstrap replicate.
- **Modulus.** `m=9` primary, `m=24` confirmatory. CRT cross-modulus consistency (per the RNS-EEG pattern, OB 2026-04-08) is a secondary gate: synchrony that holds in both moduli passes; synchrony only at one is a weaker claim.

### Ordering and stop-check

Observer 1 → 2 → 3 is the mandatory order. Each observer commits its pre-registration before the next runs. If Observer 2 fails on held-out, Observer 3 may still proceed only if the failure is diagnosed as a *structural* gap that Observer 3 addresses (here: per-asset cascade may lack cross-asset information). If the failure is diagnosed as "implementation error" (per memory rule), fix Observer 2 and re-run; this counts as a continuation, not a new iteration. Budget cap across all three observers + any variations: 5.

---

## 3. Alternate-candidate sketches

**Lo-MacKinlay VR + turn-angle.** The Phase R existing-art audit ranked this the top fresh-mapping candidate (Lo & MacKinlay 1988). `φ(x)` is the QA path-state trajectory on `(b, e)`; turn-angle `θ_t` is the angle at successive triples. Under cert [209] inference, `b_t = qa_mod(decile_rank(log_r_t), m=9)`, `e_t = ((b_{t+1} − b_t − 1) mod 9) + 1`, and θ_t is computed on the 2D embedding of the (b, e) path. Primary statistic: local VR estimator conditioned on θ_t quartile vs global VR; DM-style test with block-bootstrap null. The observer ladder would be Observer 1 = single-asset turn-angle baseline (likely NULL, matches the Track-D (SPY, TLT) orbit-classifier NULL), Observer 2 = cross-asset turn-angle synchrony, Observer 3 = orbit-family × θ-quartile joint. Same protocol shape; no re-derivation needed.

**Chronos-2 + QA orbit prior on tokens.** Chronos-2's ~4096-symbol scaled-quantization vocabulary (Ansari & Peter 2025, arXiv:2510.15821) is already an observer projection at m=4096, and the pretraining cutoff (Q3 2025 per Amazon Science blog) defines the held-out boundary automatically. Observer ladder: Observer 1 = zero-shot Chronos next-token distribution (baseline), Observer 2 = Chronos + QA orbit-class bias on token logits (add `log P(orbit_class(next_token) | orbit_class(current_token))` to the logit layer using `qa_mod` at m=4096 or a sub-alphabet), Observer 3 = Chronos + orbit-class bias + cross-asset orbit-synchrony prior (the same [209] cross-asset mechanism transposed to token space). Primary statistic: directional-accuracy on held-out (post-pretraining-cutoff) equity returns, with DM vs vanilla zero-shot Chronos. The MDPI 2024 TSFM-overlap warning (47–184% inflation) is why pretraining cutoff discipline is load-bearing here, not a nice-to-have. Deferred to when ML compute is available; structurally the observer protocol above applies unchanged.

---

## 4. Stopping criteria and failure modes

**Green — proceed to cert.** At least one observer rejects all three nulls (block-bootstrap, AR(1)-matched, phase-randomized) on the training fold AND on held-out at the Bonferroni-adjusted threshold. Freeze the observer, write the cert (candidate: `QA_HAR_RV_ORBIT_CONDITIONING_CERT.v1`, reserve family number at time of freeze), bolt the blinded evaluation script onto the cert fixture.

**Terminal NULL — observer gap real and exhausted.** Observer budget (5) is exhausted AND no observer rejects all three nulls on held-out. Declare NULL, file `phase4_null.md` with one-paragraph diagnosis per observer explaining the specific structural deficiency. This is a publishable result: "under the HAR-RV cascade mapping at these five data panels, orbit-conditioning does not improve volatility forecasts at the DM-p < 0.05 (Bonferroni 0.0167) level against autocorrelation-preserving nulls." Per `feedback_finance_not_closed.md`, the framing is observer gap, not framework verdict.

**Overfitting signature — likely implementation issue.** All three nulls rejected on training fold AND held-out fails for ≥3 consecutive observers. The training-fold rejection means the observer is finding something real on training; the held-out failure means it's not generalizing. Diagnosis options: (i) decile edges computed on full series instead of training-only (data leakage); (ii) look-ahead in block-bootstrap block construction (block overlaps training/test boundary); (iii) the orbit-family labels are capturing something that is tight within the training window (e.g., a single market regime) and doesn't persist. Declare NULL with **"overfitting diagnosis"** tag, cite the specific failure class, do not iterate further on this mapping.

**Observer 1 fails on training — implementation error, not QA failure.** Per `feedback_qa_never_dead_implementation_first.md`: Observer 1 is the control and should reproduce (or weakly outperform) the Phase 2.5 NULL pattern on training. If it fails to even match the prior NULL, assume `qa_mod`, decile quantization, or [209] generator inference is wrong *in the pipeline*, not in QA. Debug the observer itself, run `self_test()` on `qa_orbit_rules.py`, re-run Observer 1. This does not count against the 5-observer budget.

**Partial success — reject training, pass held-out one null only.** Do not claim cert-grade result. Log as "promising, not converged," proceed to the next observer. Counts against budget. This is the analog of EEG Observer 3's chb01 ΔR²=+0.0447 p=0.087 (trending, not sig), which was only confirmed by scaling to 10 patients — the finance analog is scaling to more assets or to the full VME octet.

**Path-dependence red flag.** If the first observer to freeze is highly dependent on a specific hyperparameter (e.g., window size, quantile count), re-run freeze with ±20% perturbation on each hyperparameter. If the result does not survive perturbation, do not freeze; iteration continues.

---

## References

- Ansari, A.F. & Peter, O. et al. (2025). "Chronos-2: From Univariate to Universal Forecasting." arXiv:2510.15821.
- Barndorff-Nielsen, O.E. & Shephard, N. (2002). "Econometric Analysis of Realized Volatility and its Use in Estimating Stochastic Volatility Models." *Journal of the Royal Statistical Society B* 64(2):253–280. DOI: 10.1111/1467-9868.00336.
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society B* 57(1):289–300. DOI: 10.1111/j.2517-6161.1995.tb02031.x.
- Bonferroni, C.E. (1936). "Teoria statistica delle classi e calcolo delle probabilità." *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze* 8:3–62.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics* 7(2):174–196. DOI: 10.1093/jjfinec/nbp001.
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics* 13(3):253–263. DOI: 10.1080/07350015.1995.10524599.
- Lo, A.W. & MacKinlay, A.C. (1988). "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test." *Review of Financial Studies* 1(1):41–66. DOI: 10.1093/rfs/1.1.41.
- Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association* 89(428):1303–1313. DOI: 10.1080/01621459.1994.10476870.
- Shoeb, A.H. (2009). *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment.* PhD thesis, MIT. CHB-MIT corpus: https://physionet.org/content/chbmit/1.0.0/.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J.D. (1992). "Testing for nonlinearity in time series: the method of surrogate data." *Physica D* 58(1-4):77–94. DOI: 10.1016/0167-2789(92)90102-S.

Companion repo artifacts cited:

- `docs/theory/QA_FINANCE_SOTA_SURVEY.md` (Phase R candidate ranking; this protocol's source for HAR-RV as top candidate).
- `docs/theory/QA_FINANCE_EXISTING_ART_AUDIT.md` (Phase R existing-art inventory; this protocol's source for Observer 1 expected-NULL calibration).
- `qa_lab/qa_observer/core.py` (canonical `qa_mod` definition, line 27).
- `qa_orbit_rules.py` (canonical `orbit_family`, line 72; `qa_step`, line 59).
- `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/qa_signal_generator_inference_cert_validate.py` (cert [209] `e_t` generator inference, lines 49–74).
- `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/fixtures/sgi_pass_default.json:71` (cross-asset synchrony witness, `r = +0.037, p = 0.025`).
- `qa_lab/phase2_5_quantization_compare.py` (Phase 2.5 NULL reference for Observer 1 calibration).
- `~/.claude/projects/-home-player2-signal-experiments/memory/feedback_finance_not_closed.md` (framing rule: observer gap, not framework verdict).
- `~/.claude/projects/-home-player2-signal-experiments/memory/feedback_qa_never_dead_implementation_first.md` (implementation-error-first rule for Observer 1 failure).
- `~/.claude/projects/-home-player2-signal-experiments/memory/feedback_adversarial_testing.md` (null-design circularity check).

OB thought timestamps grounding §1.5: 2026-03-28T10:15:29Z (Observer 1 baseline), 2026-03-28T10:29:35Z (Observer 1 calibrated), 2026-03-28T10:46:58Z (three-observer comparison with hard ΔR² numbers), 2026-04-08T01:39:28Z (RNS Observer baseline), 2026-04-08T01:56:29Z (RNS eigenspectrum iteration).

---

*Word count: ~3,080. Scope: protocol specification only. No code, no experiments.*
