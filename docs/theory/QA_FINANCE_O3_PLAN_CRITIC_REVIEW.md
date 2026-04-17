# Observer 3 Launch Plan — Adversarial Critic Review

**Session:** `lab-finance-o3-critic` (fresh critic process; not the prior framework-audit agent).
**Date:** 2026-04-16.
**Scope:** The Observer 3 launch plan proposed on the main thread (panel `B_t = qa_mod(Σᵢ signed-return-decile b_{t,i}, 9)` across 5 assets; [209] e_t inference on `B_t`; 3×3 orbit-transition partial MI; three nulls; training-fold kill-gates; pre-reg thresholds calibrated to the QCI public re-val).
**Priors incorporated:** Claude (2026-04-16a) `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md` (prior audit's §3 corrected spec); Claude (2026-04-16c) `docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md` (sign-flipped r=+0.44, partial r=+0.26, 0/4 Bonferroni on partial); Corsi (2009) HAR-RV; Miller (1955) MI finite-sample bias; Politis & Romano (1994); Theiler et al. (1992); `memory/feedback_theorem_nt_observer_projection.md`; `memory/feedback_qa_never_dead_implementation_first.md`; `memory/feedback_map_best_to_qa.md`; `memory/feedback_finance_not_closed.md`.

Adversarial mode. Pandering is useless. Will's standing prior is that standard math gets smuggled into QA finance analyses and the signal disappears only on finance. I check this plan against that concern concretely.

---

## 1. Numbered concerns

### Concern 1 — The partial-MI conditioning step is under-specified and almost certainly re-imports rank structure.

The plan says "3×3 orbit-transition mutual information, PARTIAL (conditional on lagged RV / HAR predictions)." MI is a joint-distribution functional; "partial" is not a canonical MI operation the way it is a canonical regression operation. To compute conditional MI `I(orbit_{t-1}; orbit_t | Z_t)` you must *discretise* `Z_t` (lagged RV or HAR prediction), because both are continuous (Kraskov, Stögbauer & Grassberger 2004 KSG gives an alternative, but it is not distributional-assumption-free in finite samples). The moment you bin lagged RV into k buckets, the conditioning event `Z ∈ bin_j` is itself a rank statistic of the same scalar the [209] observer family already ranks. That re-introduces the Pattern 3 collinearity the prior audit identified (Claude 2026-04-16a §Pattern 3: "the observer is a rank of the same scalar being predicted"). With `B_t = Σ b_{t,i} mod 9`, the problem is slightly diluted across five assets, but only slightly — when all five assets move together (common in equity panels during vol events) `B_t` inherits the decile-rank structure of the panel-aggregate return, which is not orthogonal to decile-rank of realized vol once conditioning on lagged RV forces bucketing.

**Evidence:** panel correlation of SPY/TLT/GLD/USO/UUP daily returns is positive during stress (flight-to-correlation) and small-to-moderate during calm regimes. The plan neither specifies the conditioning estimator (plug-in vs kNN vs KSG) nor the discretisation rule for `Z`. That is a p-hacking degree of freedom, not a pre-registered statistic.

### Concern 2 — Effect-size anchor (2.6σ) comes from a result the plan itself says is 0/4 under Bonferroni on the correct statistic.

The public QCI re-val beat 4/4 nulls on **raw** r, but **0/4 under Bonferroni at α = 0.0125 on partial r** (Claude 2026-04-16c §4). The plan anchors its pre-reg thresholds on "~2.6σ honest effect size." But the quantity being ported — partial correlation beyond lagged RV — did not clear Bonferroni in the re-val. Using that same partial-beyond-lagged-RV frame as the O3 target statistic, while calibrating α to an effect size that in the precedent did not survive the same multiple-comparison correction, is anchoring on the one half of the precedent that's weakest.

Concretely: the partial-r rank-p's in the re-val were 0.035, 0.045, 0.055, 0.080. The plan's Bonferroni-adjusted threshold for O3 is 0.0167 per null. If O3 reproduces the QCI partial strength *exactly*, it fails at the proposed threshold on all four nulls. The plan is setting a threshold the nearest piece of validated QA-finance evidence fails.

### Concern 3 — The HAR-frozen residual surrogate is a defensible null but the plan gives it priority it doesn't deserve.

The prior audit (Claude 2026-04-16a §3.C) introduced the HAR-frozen null specifically to invert the three Phase-P nulls: instead of preserving linear AC and destroying QA, it preserves HAR (Corsi 2009) and destroys everything else. That's sound *as a null*. But it is not QA-native; it smuggles HAR's linear functional form into null construction. If HAR is mis-specified (non-linear volatility dynamics, Corsi 2009's well-known heavy-tail residuals), the "HAR-frozen residuals" null leaks exactly the non-linear structure QA might detect, because those dynamics live in the residuals HAR didn't capture.

The HAR residual has known non-Gaussian / clustered properties. Phase-randomising HAR residuals (Theiler et al. 1992 method) in particular preserves the residual's spectrum but destroys its non-Gaussian higher-order couplings. Any real QA structure that lives in those couplings *and* in returns-beyond-vol-magnitude will be credited to "structure beyond HAR" — which is the effect the plan wants — but also to "the specific non-Gaussianity of HAR residuals that the phase-randomisation destroys," which the plan doesn't want to claim. The plan cannot distinguish these two.

### Concern 4 — The permuted-generator null is the only genuinely QA-native null in the set, and it is the weakest of the three.

Null #2 (shuffle e_t keeping b_t) is the only null whose construction is stated entirely in QA primitives. It is also the easiest for the observer to beat, because if the generator inference from [209] is producing *any* cross-time-step structure, shuffling e_t destroys it and the real MI beats shuffled MI trivially. This is not in itself bad — but it means the Bonferroni-AND across three nulls (marginal-iid, permuted-generator, HAR-frozen) is carried by the two *non-QA-native* nulls, not by the QA-native one. The advertised "all three nulls reject" verdict would in practice be "marginal-iid rejects, HAR-frozen rejects, permuted-generator rejects easily." Two of three are doing the real work and both are observer-layer constructions.

### Concern 5 — "Signed-return decile panel-sum mod 9" is rank-with-sign, which under panel aggregation is nearly rank-autocorrelation of aggregate return.

The plan's claim of non-rankness rests on "signed return is near-orthogonal to log-RV at daily horizon." That is roughly true *per asset*. But `B_t = qa_mod(Σᵢ b_{t,i}, 9)` aggregates five signed-decile ranks and takes mod 9. For N=5 assets with b ∈ {1..9}, `Σ b ∈ {5..45}` and `B_t ∈ {1..9}`. Under panel-return co-movement, `Σ b_{t,i}` is tightly related to the decile rank of the *cross-sectional-average signed return* — a scalar per-day quantity. The mod-9 wrap adds periodic noise but does not break the underlying rank-statistic correlation with "how coordinated was the move today." That is a linear-autocorrelation-adjacent quantity. The permuted-generator null destroys e-structure but preserves `b` and hence preserves `B_t`; the marginal-iid null destroys orbit structure but the orbit depends on `(B_t, e_t)` where `B_t` retains the aggregate-rank structure; only the HAR-frozen null directly attacks return-level rank structure, and it does so via HAR's particular linear frame.

So the observer is not as non-rank as claimed. It is "signed rank of cross-sectional average return, wrapped mod 9." At the panel level, under realistic correlation regimes, that is close enough to a rank statistic of a scalar that Pattern 3 partially survives.

### Concern 6 — Observer 3 is still in the same observer family as O1 and O2.

The prior audit (Claude 2026-04-16a §Pattern 5) found O1/O2/proposed-O3 are all the same observer family: "decile rank of log realized variance, varied over (asset, timescale)." The main-thread plan swaps the ranked quantity from log-RV to signed return and adds panel-sum. That *is* a structural change — but it is a structural change *within the decile-rank observer family*. The EEG 1→4 progression was threshold → argmax band → topographic k-means → correlation-matrix eigenspectrum: four *structurally distinct projections*, not four variants of the same rank. If O3 is "still decile-rank, just of a different scalar with cross-asset sum," the iteration budget accounting is being gamed. The prior audit flagged this explicitly and recommended "exit to a structurally different projection." The plan does not.

A genuinely structurally-different observer to compare against: `qa_mod(sign-of-return-vector-pattern-index(t), 9)` where the pattern is the 2^N sign vector treated as a categorical label, no ranking. That would be non-rank. Another: correlation-matrix eigenspectrum of the 5-asset rolling correlation matrix, discretised to QA states by eigengap magnitude — non-rank, captures structure-of-coupling not rank-of-level. The plan chose neither.

### Concern 7 — Kill-gates at MI ≥ 0.01 nats + χ² p < 0.01 are thresholds without a pre-specified estimator.

Plug-in MI on a 3×3 transition table is biased upward at small sample; MI(X,Y) = 0 is not the expected value of the plug-in estimator under independence for finite samples (Miller 1955; Paninski 2003). At n ~ 1000 per training fold the finite-sample bias in plug-in MI on 3×3 is on the order of (k-1)²/(2n) ≈ 0.002 nats — 20% of the 0.01-nat kill-gate floor. That is tunable: a discretisation or smoothing choice can move the plug-in estimate by several multiples of 0.002 nats. The gate does not specify whether Miller–Madow bias correction is applied, whether the estimator is plug-in or KSG, or what counts as the "orbit_t" conditional — and the prior audit's Pattern 6 (orbit distribution is near-uniform and singularity count is a decile-rank artifact) applies here unchanged: a cell in the 3×3 table with 30 observations out of 1000 has enough leverage to move plug-in MI by 0.005+ nats under a small re-binning. **The gate is not robust against the single most likely p-hacking pattern here: tuning discretisation of the conditioning variable to nudge training MI above the floor.**

### Concern 8 — There is a cheaper prerequisite the plan skips.

Before spending Observer 3 of 5: run the proposed pipeline on a synthetic null panel — 5 independent random-walk log-price series, same length, same training/test split. Compute partial MI with the same nulls. If the synthetic panel's MI beats *any* of the three nulls with non-negligible frequency under repeat seeds (say > 10%), the MI statistic is noise-dominated on this observer family and Observer 3 is a waste of budget. If the synthetic panel's MI cleanly rejects under zero of the three nulls across 100+ seeds, the test has calibrated false-positive rate and Observer 3 can proceed. This is a half-day of compute and it disambiguates the "is this observer test sensitive" question from the "does finance have QA-native orbit-transition structure" question. The plan does not include it. It should.

---

## 2. Pressure-test question verdicts

**Q1 — Effect-size anchor (2.6σ) right given QCI public 0/4 Bonferroni on partial?**
**Amend.** Either drop the Bonferroni correction and pre-register primary-statistic as raw MI (matching the public result that survived at 4/4 raw), or drop the 2.6σ anchor and pre-register a weaker floor (1.5σ or rank-p ≤ 0.05 on ≥ 2 of 3 nulls). Holding Bonferroni at 0.0167 per null while anchoring effect size to a QCI number that itself fails Bonferroni is internally inconsistent.

**Q2 — Partial MI specification properly formed?**
**No-go as written.** Conditional MI requires a stated conditioning estimator. The plan's "partial MI conditional on lagged RV / HAR predictions" collapses to (i) plug-in conditional MI with binned Z, which re-imports rank, or (ii) KSG conditional MI (Kraskov et al. 2004), which is distributional-assumption-free but finite-sample-biased in a direction that depends on nuisance dimensionality. Either commits to a specific choice pre-registration or the statistic is not pre-registered.

**Q3 — HAR-frozen null legitimately QA-native?**
**Amend.** It is legitimate as an observer-side null but is **not QA-native**. The plan should label it as such and not count it toward the "two of three QA-native" claim. Combined with Q4, that means the null set has one QA-native null (permuted-generator) and two observer-side nulls. Bonferroni-AND across a set dominated by observer-side nulls carries the known Pattern 1 problem in weakened form.

**Q4 — Training-fold gates adequate?**
**No-go as written.** 0.01-nat floor on plug-in MI is tunable via discretisation of the conditioning variable. The gate must specify the exact estimator (plug-in with Miller–Madow or KSG), the exact binning rule for any continuous conditioning variable, and an effective-sample-size floor (e.g., minimum expected count ≥ 5 per cell in the 3×3 conditional-on-Z stratum). Without these the gate is a discretion surface, not a gate.

**Q5 — Smuggling patterns in this plan?**
**Three found, listed above:** (Concern 1) partial-MI re-imports rank via conditioning variable binning; (Concern 5) panel-sum mod 9 is still rank-with-sign at aggregate level; (Concern 6) same observer family as O1/O2 under a different scalar. None of these are fatal individually; compounded they make the test narrower than advertised.

**Q6 — Observer genuinely non-rank?**
**No.** Signed-return decile → panel sum → mod 9 is "rank-with-sign of cross-sectional average return, wrapped." Under realistic panel correlation it retains substantial rank structure. Claim of non-rankness is overstated.

---

## 3. Launch verdict

**LAUNCH-WITH-AMENDMENTS, AFTER RUNNING PREREQUISITE.**

Specifically: do NOT launch Observer 3 in the current form. Before launch:

1. **Prerequisite (mandatory):** run the exact O3 pipeline (observer + statistic + three nulls) on 100 seeds of a synthetic independent random-walk 5-asset panel, same length, same train/test split. Confirm false-positive rate against each null is ≤ 0.05 per null and ≤ 0.01 Bonferroni-corrected. If FPR is inflated on any null, the test is not calibrated on this observer family and O3 should not consume a budget slot. Half-day of compute.

2. **Then** amend the pre-reg per §4 and launch.

If the prerequisite shows the test is not calibrated, the plan moves to DO-NOT-LAUNCH, and the iteration-budget accounting becomes: spend the half-day on the prerequisite, gain evidence the MI statistic on this observer class is noise-dominated, drop the observer family, design a structurally different observer (eigenspectrum, sign-pattern-index, or a volume-joint) for the next slot.

---

## 4. Replacement text for pre-reg fields (if amendments accepted)

**Primary statistic (replacement):**
> Plug-in mutual information `I(orbit_{t-1}; orbit_t)` computed on the 3×3 empirical transition table of the panel-derived orbit sequence on the held-out test fold, with Miller–Madow bias correction applied (Miller 1955). No conditioning on continuous nuisance variables in the primary statistic. The HAR-adjustment claim is tested via the HAR-frozen-residual null only (see below), not via conditional-MI.

**Secondary statistic (descriptive, not gating):**
> Difference in plug-in MI between panel-orbit transitions and marginal orbit-count-preserving iid surrogates, reported with bootstrap 95% CI (Politis & Romano 1994), for descriptive effect-size reporting only.

**Null set (replacement — mark QA-nativeness explicitly):**
> (a) Marginal-orbit iid [QA-native]: draw surrogate orbit_t^{null} iid from empirical test-fold orbit marginal.
> (b) Permuted-generator [QA-native]: shuffle e_t uniformly at random, keep b_t fixed, recompute orbit.
> (c) HAR-frozen residual [OBSERVER-SIDE, NOT QA-native]: fit HAR on real returns (Corsi 2009), phase-randomise residuals (Theiler et al. 1992), re-quantize, re-build B, re-infer e, re-classify orbits. Reported separately; null (c)'s rejection is labelled "beyond-HAR structure in observer layer," not "QA-native signal beyond HAR."

**Threshold (replacement):**
> Primary: real MI beats nulls (a) AND (b) at rank-p ≤ 0.05 uncorrected. Bonferroni correction dropped on primary because (a) and (b) test complementary null hypotheses (marginal structure vs generator structure) and are not replicate tests of the same hypothesis. Null (c) reported as sensitivity analysis only.

**Training-fold kill-gates (replacement):**
> (i) plug-in MI with Miller–Madow correction ≥ 0.005 nats on training fold (floor set to 1.5× finite-sample bias magnitude at n_train, not 0.01 nats);
> (ii) χ² test of 3×3 transition independence has p < 0.01 on training fold with Yates continuity correction AND minimum expected count ≥ 5 per cell;
> (iii) effective sample size for each 3×3 cell under the test statistic ≥ 30.
> Failing any of (i), (ii), (iii) = declare training NULL, do not compute test-fold statistic. No re-binning post-gate.

**Prerequisite (new, mandatory):**
> Run the full pipeline on 100 seeds of synthetic iid-log-return 5-asset panels (each asset: log-return ~ N(0, σ²), σ matched to asset's full-sample std). For each null (a), (b), (c), record rejection rate across seeds. Prerequisite passes if rejection rate ≤ 0.05 for each null at uncorrected α = 0.05. If any null's rejection rate exceeds 0.05, Observer 3 is declared not-calibrated and launch is cancelled.

**Effect-size floor (replacement):**
> Test-fold MI ≥ 0.5 × training-fold MI, AND training-fold MI passes all three kill-gates above. No anchor to the QCI public-corpus 2.6σ number; the QCI partial-r result fails Bonferroni on partial and is not an appropriate calibration anchor.

**Observer family declaration (new, mandatory):**
> Observer 3 is declared a structural variant *within the decile-rank observer family* established by Observer 1 and Observer 2, with cross-asset aggregation as the novel element. A negative result on O3 updates toward "the decile-rank observer family does not contain exploitable QA-native signal on this panel" and consumes iteration 3 of 5. A positive result must be replicated on a held-out asset panel (e.g., sector ETFs) before being reported as a confirmed QA-finance finding.

---

## 5. Honest meta

If Will's prior is right (signal disappears only in finance because of smuggling), this plan in its current form will produce another NULL for the same reason O1 and O2 did, only packaged as a categorical test instead of a regression test. The fixes above shift the test from "narrow and discretion-laden" to "narrow and pre-committed" — they do not make the observer family structurally different. The deeper move, consistent with the framework audit's §3 hard pushback on itself, is: skip O3-as-rank-variant, design O4 as a structurally-different projection (sign-vector-index, eigenspectrum, or volume-joint), and spend this budget slot on the prerequisite synthetic-panel calibration instead.

If the plan holds as proposed (ignoring the above amendments), my verdict is **DO-NOT-LAUNCH**. If the amendments are accepted and the prerequisite passes, it is **LAUNCH-WITH-AMENDMENTS**. If the prerequisite fails, it is **DO-NOT-LAUNCH** and the slot is refunded to design a non-rank observer.

---

## References

- Claude (2026-04-16a). *QA Finance Framework Audit — Phase P Observer 1 & 2.* `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md`. Internal primary source.
- Claude (2026-04-16c). *QCI Public-Corpus Re-Validation.* `docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md`. Internal primary source.
- Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics* 7(2):174–196. DOI: 10.1093/jjfinec/nbp001.
- Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). "Estimating mutual information." *Physical Review E* 69:066138. DOI: 10.1103/PhysRevE.69.066138.
- Miller, G. A. (1955). "Note on the bias of information estimates." In *Information Theory in Psychology*, ed. H. Quastler, Free Press, pp. 95–100.
- Paninski, L. (2003). "Estimation of Entropy and Mutual Information." *Neural Computation* 15(6):1191–1253. DOI: 10.1162/089976603321780272.
- Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association* 89(428):1303–1313. DOI: 10.1080/01621459.1994.10476870.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J. D. (1992). "Testing for nonlinearity in time series: the method of surrogate data." *Physica D* 58(1-4):77–94. DOI: 10.1016/0167-2789(92)90102-S.

Memory authorities cited: `memory/feedback_theorem_nt_observer_projection.md`; `memory/feedback_qa_never_dead_implementation_first.md`; `memory/feedback_map_best_to_qa.md`; `memory/feedback_finance_not_closed.md`.

---

*Word count ≈ 2,000.*
