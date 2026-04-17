# QA Finance O4 + Framework-Doc Review

**Session:** `lab-finance-o4-and-framework-critic` (fresh adversarial critic; not the framework-audit agent, not the O3 critic agent).
**Date:** 2026-04-17.
**Scope:** (1) End-to-end review of `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` as a protocol doc. (2) Review of the main-thread Observer 4 redesign (candidates α/β/γ, marginal MI statistic, two-null set, synthetic-panel prerequisite, Bonferroni-dropped α=0.05 AND-conjunction).
**Prior critics incorporated:** `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md` (revoked O1/O2 NULL verdicts; proposed "corrected O3"); `docs/theory/QA_FINANCE_O3_PLAN_CRITIC_REVIEW.md` (DO-NOT-LAUNCH on that corrected O3; found conditional MI re-imports rank, effect-size anchor internally inconsistent, observer still rank-family).
**Primary-source grounding:** every claim about a file or line is cited. Where something is *absent* in the repo, I say "absent."

---

## Part 1 — Phase R framework-doc audit (`docs/specs/QA_FINANCE_OBSERVER_ITERATION.md`)

### (a) §1 general protocol — does it distinguish "parameter sweep within family" from "family crossing"?

**Partially — in letter, not in practice.** §1.2 (lines 30-33) explicitly says: *"Parameter tweaks to the same observer (different window size, different quantile count) are **not** a new observer — they are the same observer under a design variation; each such variation is its own iteration and counts against budget."* That language rules out pure hyperparameter sweeps. But §1.2 never defines "observer family" as opposed to "observer." It gives three admissible quantizers as examples ("Quantile-rank, k-means cluster index, or decile bucket," line 18) without stating that transitioning between those three is the structural move that makes an iteration qualitatively different. The EEG precedent in §1.5 (lines 58-72) is framed as "each observer was a *structural redesign* of the previous" (line 72) — but "structural redesign" is never operationalized into a checklist the iterator is required to apply before declaring a new observer.

**The consequence, as the prior O3 critic and the framework audit both found, is that §2's Observer 1 → Observer 2 → Observer 3 ladder is three variants of decile-rank log-RV** (`phase_p_observer1.py:101` `qa_mod(int(rank), 9)` on log-RV; §2 Observer 2 stacks the same rank at k ∈ {1, 5, 22}; §2 Observer 3 extends the same rank cross-asset). The protocol lets this stand because the rule on line 31 reads "parameter tweaks = same observer" without operationalizing the inverse: "cross-family move = genuinely new observer." Observer 2 vs Observer 1 is *defended in the doc* as a structural change (three stacked timescales, Axiom-compliance note at lines 80), but under the EEG-family rubric it is still rank-of-the-same-scalar.

**Verdict (a): NEEDS CORRECTION.** §1.2 must add a "family crossing requirement" clause and an enumerated family taxonomy.

**Replacement text proposed for §1.2 end (insert after line 33):**
> **Observer-family discipline.** An "observer family" is identified by its quantization class: {rank-of-scalar, k-means-of-feature-vector, sign-pattern-index, correlation-eigenspectrum, joint-bucket-of-multiple-features} are distinct families. Iteration `O_{k+1}` is admissible as a *new observer* only if it crosses family from `O_k`. Variation that keeps the family (different scalar, different asset, different timescale, cross-section of the same rank) is a *within-family variant* and is reported as a single observer with its variants rather than as a new iteration. The EEG progression (§1.5) is a canonical cross-family example: threshold-fallback, dominant-band argmax, topographic k-means, RNS correlation-matrix eigenspectrum are four distinct quantization classes. Iteration in finance is held to the same discipline.

### (b) §1.5 EEG precedent — does it make cross-family normative?

**No.** §1.5 is descriptive, not normative. Lines 63-68 enumerate the four EEG observers with their featurizations (bandpower threshold / argmax / k-means-on-RMS / eigenspectrum). Line 72 names "structural redesign" as "the two rules that were actually followed," but the word "family" is absent from §1.5 entirely (I checked — no occurrences of "observer family" or "family crossing" in the file). §1.5 treats the EEG progression as a *successful example* the finance protocol inherits in spirit, not as a constraint it inherits in law.

This is the single most consequential gap: the prior critic's finding *"my O1/O2/proposed-O3 don't cross"* is a direct consequence of §1.5 describing EEG observers without demanding that finance observers inherit the cross-family property.

**Verdict (b): NEEDS CORRECTION.** §1.5 must add an explicit normative paragraph.

**Replacement text proposed for §1.5 (append after line 72):**
> **The cross-family property is normative, not incidental.** The EEG progression worked because each iteration was mathematically a different projection of the same underlying signal — amplitude cross a frequency-band cutoff (Obs 1), frequency-band argmax (Obs 2), spatial k-means over per-channel RMS (Obs 3), inter-channel correlation-matrix eigenspectrum (Obs 4). Each successive observer captured information the previous one discarded by construction. The finance iteration inherits this property: an iteration that keeps the rank-of-scalar family (e.g., decile rank of log-RV at a different timescale, or extended cross-asset) is *not* an iteration under this protocol — it is a variant of the previous observer. The iteration counter advances only on family crossings.

### (c) §2 Observer 1/2/3 ladder — qualitatively different projections, or parameter sweep?

**Parameter sweep.** Confirmed against `phase_p_observer1.py:86-114` and `phase_p_observer2.py` (the Phase P runtime). Observer 1 `b_t = qa_mod(bisect_left(edges, log_RV_t)+1, 9)` (phase_p_observer1.py:100-101). Observer 2 replicates this per-timescale with edges per-timescale on training (doc §2 lines 95-96). Observer 3 (sketched in doc §2 lines 103-108) replicates per-asset per-timescale with cross-asset synchrony added as a separate sidecar statistic. All three are decile-rank of log-RV under `bisect_left`, varied over (timescale, asset, synchrony-aggregate). The framework-audit's Pattern 5 finding (lines 80-86 of that doc) is correct and is grounded in the runtime. The prior O3 critic's concern 6 (same observer family) reiterates it.

**Verdict (c): NEEDS CORRECTION.** §2 must re-label the Observer 1/2/3 ladder as a single decile-rank-log-RV observer with three design variants (single-asset, stacked-timescale, cross-asset), consuming **one** iteration slot of the five. The budget accounting in §1.3 (line 46, "A budget cap of 5 observers") must be reread with this reclassification in mind.

**Replacement text proposed for §2 heading (at line 76):**
> ## 2. Decile-rank log-RV observer — three design variants
> *The three designs below constitute a single observer (quantization family: rank-of-scalar-log-RV) with variants (Observer 1 = single-asset single-timescale; Observer 2 = stacked-timescale per-asset; Observer 3 = cross-asset extension). Per §1.2 family discipline, this is one iteration in the 5-iteration budget, not three.*

### (d) §1 three-null set — justified as QA-native, or inherited from time-series convention?

**Inherited.** §1.2 (lines 26-29) specifies block-bootstrap (Politis-Romano 1994), AR(1)-matched, phase-randomized (Theiler 1992). The justification is two sentences referencing Politis-Romano and Theiler — both continuous-time-series methods. The framework audit's Pattern 1 finding (lines 22-30 of that doc, grounded in `phase_p_observer1.py:340-356, 381-399, 422-443`) correctly notes that all three nulls preserve linear autocorrelation of log-RV and that the Bonferroni-AND across three near-colinear nulls inflates type II error. The prior O3 critic's concerns 3 and 4 reiterate the asymmetry: only permuted-generator is truly QA-native; the HAR-frozen null and marginal-orbit-iid are observer-side alternatives.

§1.2 does not discuss QA-native null construction at all. There is no paragraph labeled "QA-native nulls" in the doc; grep for "QA-native" returns no matches in the file.

**Verdict (d): NEEDS CORRECTION.** §1.2 null specification must distinguish QA-native from observer-side nulls and require at least one QA-native null in any null set.

**Replacement text proposed for §1.2 nulls (replace lines 26-29):**
> The nulls used in this protocol must include at least one **QA-native null** — one whose construction operates entirely in the QA discrete layer. Examples: permuted-generator (shuffle `e_t` keeping `b_t`, recompute orbit), marginal-orbit-iid (draw `orbit_t` iid from empirical orbit marginal), random-legal-walk (random `(b, e)` walk on {1..m} × {1..m} started at the same initial state). Observer-side nulls (block-bootstrap on continuous observables, AR(1)-matched, phase-randomized, HAR-frozen residual) may supplement but cannot substitute. The rationale is Theorem NT: QA dynamics is discrete, so a null for "does QA structure exist" must be destroy-QA-structure-but-preserve-everything-else, and that operation is most cleanly expressed at the discrete layer. Observer-side nulls preserve linear autocorrelation of the observer's continuous input; a QA signal that shares spectral support with its input will not produce a tail-event statistic against such nulls.

### (e) Observer 3 sketch — family-crossing, or rank-family variant?

**Rank-family variant.** Doc §2 lines 103-108: `b_{t,k,i} = qa_mod(decile_rank(log_RV_{t,k,i}), m=9)` per asset, per timescale. Synchrony `S_{t,k}` is defined in terms of `e_{t,k,i}` residue classes — which are themselves derived from the same decile rank via [209]. Cross-asset extension does not change the quantization family. This matches the framework-audit Pattern 5 conclusion and the prior O3 critic's concern 6.

**Verdict (e): NEEDS CORRECTION.** The current Observer 3 sketch must be demoted from "Observer 3" to "Observer 1 cross-asset variant" and the protocol must leave Observer 2+ slots open for genuine family-crossings (sign-vector-index, eigenspectrum, joint-bucket-multi-feature, etc.).

---

### Overall framework-doc verdict

**FRAMEWORK-DOC-NEEDS-CORRECTION.** The doc is not unsalvageable — §1.3 pre-registration discipline, §1.4 memory-inheritance rules, §4 stopping criteria are all sound and load-bearing. The correctable errors are concentrated in §1.2 (family discipline + QA-native null requirement) and §1.5 (normative cross-family claim) and §2 (re-labeling the three "observers" as one observer's variants). Four replacement-text blocks supplied above. Total word change is ~350 words added, ~100 words replaced; the doc stays under its current 3,080-word scope.

**Budget-accounting consequence:** once §2 is re-labeled, the iteration budget used to date is **1 observer (decile-rank log-RV with three variants)**, not 3 as currently accounted (§1.3 line 46 budget of 5). This frees four slots — but per the prior O3 critic's finding that the proposed O3 is still rank-family, the "corrected O3" would have consumed the *second* slot, not the fourth. The O4 redesign below gets that second slot, with three unused slots in reserve.

---

## Part 2 — Observer 4 candidate selection

Three candidates proposed: α (sign-vector-index), β (eigenspectrum), γ (volume-return joint). I apply the four selection criteria from the task spec.

### α — Sign-vector-index

- **Crosses family?** Yes. Sign-pattern-as-categorical-index is a distinct quantization family from rank-of-scalar. The 5-asset sign pattern encoded as integer ∈ {0..31} has no monotone-rank structure; it is a topological label on the cross-section.
- **Axiom-compliant (b, e, d, a)?** Yes. `idx_t ∈ {0..31}`, `b_t = qa_mod(idx_t + 1, m)` for m ∈ {9, 24}, `e_t` via [209]. A1 respected (qa_mod returns {1..m}); S2 respected (int throughout); T1 respected (path time k is the integer index t).
- **Immune to linear-autocorrelation absorption?** Largely yes. Sign flips are correlated across assets during stress, but the integer encoding `Σᵢ (s_{t,i}==+1) × 2^i` is highly non-monotone in the cross-sectional average sign — a single asset flipping produces jumps like 0 → 1 → 3 → 2 etc. rather than smooth transitions. The transition count on 32 states can be collapsed to 3 orbit classes and is a categorical statistic. Marginal MI on the 3×3 orbit transition matrix is not reachable by any linear combination of returns on the same panel.
- **Survives synthetic prerequisite?** Needs verification. The 32-state index under independent random-walk seeds has a near-uniform marginal (each sign-pattern occurs ~1/32 by symmetry); the orbit marginal on mod-9 is also near-uniform. Transition MI under iid surrogate should be at the finite-sample bias floor. Plausibly calibrated.

**Concerns found in α:**
- **α-smuggling-1:** the encoding `Σᵢ (s==+1) × 2^i` is a specific ordering of the 5 assets. Under asset relabeling (SPY ↔ TLT swap), `idx` changes non-trivially but the orbit class may or may not change. This is a design choice that introduces an implicit "canonical asset order." The prior O3 critic's finding that arbitrary cluster-relabeling in QCI created sign-lability (OB 2026-04-17T00:39 `docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md:55-57`) applies in a weaker form here. Mitigation: pre-register the asset order as alphabetical (GLD, SPY, TLT, USO, UUP) and add a sensitivity analysis reporting MI under all 5! = 120 asset orderings.
- **α-smuggling-2:** sign of daily return uses the zero boundary, which is an observer-layer choice. A "zero return" day is rare on daily data for 5 assets (roughly zero probability) but ties at the decile-edge of zero must be resolved deterministically. Pre-register the tie-break rule (e.g., `sign(r) = +1 if r >= 0`).

### β — Eigenspectrum index

- **Crosses family?** Yes. Correlation-matrix eigenvalue is the EEG Observer 4 family verbatim (doc §1.5 line 68: "RNS eigenspectrum"). This is the cleanest cross-family analog available in the finance setting.
- **Axiom-compliant?** Partially. The issue is the quantization step: `b_t = qa_mod(round(φ_t × m), m)`. This is `int(continuous × m)` — a textbook **T2-b float-modulus-cast violation** (CLAUDE.md "T2 (Firewall): Float × modulus → int cast is a QA violation"). It's recoverable: quantize via decile edges of `φ_t` computed on training fold, then `qa_mod(rank_of_phi, m)`. But as written, the candidate β specification smuggles a T2-b violation that the framework audit's Section 1 §"Theorem-NT legality" (line 104 of that doc) would flag.
- **Immune to linear-AC absorption?** Partially. `φ_t = λ₁ / Σ λ_k` is a function of the rolling-22-day correlation matrix. Under HAR's cascade regressors, the monthly RV component carries similar information to the monthly dominant-eigenvalue fraction (both are averages over 22 days of jointly-moving price moves). The statistic *differs* from HAR's monthly term but is not orthogonal to it.
- **Survives synthetic prerequisite?** Uncertain. Under iid independent-asset random walks, `φ_t → 1/N` (uniform), and rolling estimate will fluctuate around 0.2 with bias. Under the null, MI on transitions of quantized `φ_t` may not be at the bias floor (because `φ_t` has its own autocorrelation from the rolling window), which inflates FPR. This is a calibration risk β introduces that α does not.

**Concerns found in β:**
- **β-smuggling-1:** the eigenvalue computation operates on a rolling correlation matrix whose autocorrelation structure is dictated by the window length 22. Under iid surrogate, a 22-day rolling eigenvalue has AR(≈22)-like structure by construction, leading to non-bias-floor MI under permuted-generator null (which preserves the b-marginal and the b-autocorrelation but shuffles e). The prerequisite may fail on β even when the null is correct. α has the same issue but much milder because sign-pattern-index has memory only through the underlying panel, not through a rolling-window aggregation step.
- **β-smuggling-2:** float-mod cast T2-b as noted.

### γ — Volume-return joint

- **Crosses family?** Weakly. Joint-decile-bucket of (return, volume) is *technically* a new quantization family (joint-bucket-of-multiple-features), but in practice `ret_decile × 10 + vol_decile` is still built from two scalar rank operations. The prior O3 critic's concern 6 still largely applies: "same observer family under a different scalar" becomes "same observer family under a Cartesian product of scalars."
- **Axiom-compliant?** Yes, if deciles are rank-based (not float-mod-cast).
- **Immune to linear-AC absorption?** No. Volume and return are individually captured in extended HAR specifications (HAR-V, HAR-RV-V variants in the literature: Bollerslev, Patton & Quaedvlieg 2016 DOI 10.1016/j.jeconom.2016.02.010). If an HAR-V-augmented baseline absorbs volume's marginal contribution, the joint-decile collapses to rank(return) again.
- **Prerequisite?** Volume data through yfinance — confirmed reachable (used in `qci_v2_real_finance.py:7-30` per OB 2026-04-17T00:28). Prerequisite itself is doable, but the observer is weaker than α on the crosses-family criterion.

**Concerns found in γ:**
- **γ-smuggling-1:** volume at daily resolution from yfinance is reported-volume (subject to after-hours adjustments, exchange-venue aggregation artifacts). The framework audit's Pattern 2/3 (continuous-target-plus-linear-baseline) partially reapplies: volume-decile is a rank of the same volume time-series HAR-V would use as a regressor.
- **γ-smuggling-2:** introducing volume as an additional observable confounds "QA adds structure" with "volume adds structure" (this is explicit in Claude 2026-04-16a §3.A rejection of volume-joint, line 152). The prior critic-of-own-audit already flagged this.

### Selection

**Pick α (sign-vector-index).** Reasoning:

1. It most cleanly crosses family (topological label vs rank-of-scalar; no shared structure with the Phase P O1/O2 observer).
2. Axiom-compliant without the T2-b cast β introduces.
3. Has no shared AR-structure-by-construction (β's rolling correlation matrix is an AC-inflation risk for the prerequisite).
4. Does not confound "QA adds structure" with "another observable adds structure" (γ's volume confound).
5. α-smuggling-1 (asset-order sensitivity) and α-smuggling-2 (sign tie-break) are pre-registerable, not structural.

β is the correct second choice if α fails the synthetic prerequisite; γ is rejected.

---

## Part 3 — Test statistic, nulls, thresholds pressure-test

### Marginal MI `I(orbit_{t-1}; orbit_t)` on 3×3

**Concerns:**
- **Concern 3.1 (finite-sample bias, inherited from prior O3 critic concern 7):** plug-in MI on 3×3 has upward bias ≈ (k-1)²/(2n) = 4/(2·1171) ≈ 0.0017 nats on the O3 OOS scale, similar on O4 training. The 0.5× training-to-test effect-size floor (task spec) is a reasonable overfit guard but is itself vulnerable to this bias. Amendment: specify **Miller-Madow bias correction** (Miller 1955) for both training and test. The task spec says "Miller-Madow bias-corrected, floor ≥ 0.005 nats on training" — good. Confirmed the task spec already addressed this.
- **Concern 3.2 (3×3 cell-count risk):** CLAUDE.md orbit families on mod-9 are singularity/satellite/cosmos. Uniform baseline is 1.2%/9.9%/88.9% (framework audit §Pattern 6, lines 91-93). 32-state sign-pattern → 3-orbit collapse via [209] needs verification that the marginal isn't near-degenerate. Amendment: **add a training-fold marginal-check**: min orbit marginal ≥ 5%, else observer is declared degenerate and variant retried.

### Two nulls: permuted-generator + marginal-orbit iid

**Concerns:**
- **Concern 3.3 (null power balance):** marginal-orbit-iid destroys all transition structure → easy to beat if there is any. Permuted-generator destroys the e-path structure but preserves the b-path structure (the sign-vector-index itself). Under α, the b-series comes from the sign pattern; shuffling e (which is inferred from b's own next-value via [209]) breaks the self-generated transition logic but preserves the original sign-sequence's rank. These are complementary: one tests "is there transition info at all" and the other tests "is the QA generator adding anything beyond the b-marginal." The task spec's "no Bonferroni, AND-conjunction at uncorrected α=0.05" is defensible because the two nulls destroy complementary structure (the prior O3 critic Q1 made a similar argument against Bonferroni on complementary nulls).
- **Concern 3.4 (the dropped HAR-frozen null):** the task spec drops HAR-frozen. Given the framework audit explicitly introduced HAR-frozen as the "inverse of three Phase P nulls" (audit §3.C, lines 177-180), dropping it means O4 loses the direct test of "beyond HAR structure." That is defensible IF the observer is sufficiently non-rank that HAR absorption is not a concern — and α's sign-pattern-index genuinely is non-rank. For α, dropping HAR-frozen is appropriate. For β it would not be.
- **Concern 3.5 (null seed specification absent in task spec):** the task spec doesn't specify n_null (bootstrap replicates) or seed. Amendment: **pre-register n_null = 1000 per null, seeds ∈ {42, 43}** (one per null). This is a pre-reg completeness issue, not a structural concern.

### Thresholds: primary MI rank-p < 0.05 AND-conjunction

**Concerns:**
- **Concern 3.6:** AND-conjunction at α=0.05 on two nulls gives an effective α somewhere between 0.0025 (if nulls were independent) and 0.05 (if nulls were perfectly correlated). In practice the two nulls test different aspects of structure (marginal vs generator), so they are neither independent nor identical. The effective α is approximately 0.025. This is stricter than the O1/O2 Bonferroni-0.0167 each (which requires all three, still α≈0.017 AND), and about equal to the QCI public re-val's partial-r rank-p's of 0.035-0.080 (`QA_FINANCE_QCI_PUBLIC_REVALIDATION.md:65-68`). Confirmed: the effect-size floor is reachable if the signal is real. **No amendment needed.**
- **Concern 3.7 (effect-size floor at 0.5× training):** the 0.5× multiplier is a heuristic. For plug-in MI on 3×3, train/test MI ratios in ML practice range from 0.3 (severe overfit) to 0.9 (well-generalized). 0.5× is a middle-of-road choice. Acceptable as-is but document as a **soft gate** (fails triggers warning, not NULL).

### Training-fold kill-gates

Task spec: MI ≥ 0.005 nats, χ² p < 0.01, expected cells ≥ 5.

**Concerns:**
- **Concern 3.8:** all three gates are specified — the prior O3 critic's concern 7 ("gates not robust against tuning") is met by the Miller-Madow + expected-cell-count requirement. Remaining hole: **the gate doesn't specify what happens if gates pass on training but the prerequisite (synthetic panel) fails** — i.e., the observer is calibrated in the real-panel kill-gate sense but the MI statistic has inflated FPR under iid. Amendment: **the synthetic prerequisite passes BEFORE the real-panel training-fold is touched**.

---

## Part 4 — Prerequisite design

Task spec: 100 synthetic iid-log-return 5-asset panel seeds, FPR ≤ 0.05 per null at pre-registered threshold.

**Concerns:**
- **Concern 4.1 (N=100 seeds):** at 100 seeds, the standard error on an estimated FPR is sqrt(p·(1-p)/100) ≈ sqrt(0.05·0.95/100) ≈ 0.022. So an observed FPR of 0.05 has 95% CI [0.016, 0.112]. This means a true FPR of 0.08 has ~60% probability of being measured as ≤ 0.05, passing the prerequisite when it shouldn't. Amendment: **increase to 500 seeds**, or tighten the pass threshold to empirical FPR ≤ 0.03 at N=100 (implies true FPR ≤ ~0.06 with 95% conf).
- **Concern 4.2 (distribution choice):** "synthetic independent random-walk 5-asset panel" underspecifies: are the random walks Gaussian, Student-t, return-sd matched to real? For α to be well-calibrated on the real panel, the synthetic must match real panel's *per-asset* marginal distribution (heavy-tail for equity returns) while breaking cross-asset correlation. Amendment: **pre-register: synthetic = iid bootstrap of each asset's own training-fold daily returns** (preserves marginal, destroys cross-sectional structure). This is the standard "empirical bootstrap null for cross-sectional independence" in finance (Horowitz 2001).
- **Concern 4.3 (prerequisite failure consequence):** task spec says "otherwise the statistic is mis-calibrated and the launch is cancelled." Good. Amendment strengthening: **if α fails the prerequisite, move to β with β-smuggling-2 (T2-b cast) repaired** (quantize by rank-of-eigenfraction, not float-mod-cast). If β fails, do not attempt γ; exit the finance observer-iteration track at NULL per budget accounting.

**Prerequisite-design verdict:** SOUND with 3 amendments (N=500 or tighter pass threshold; empirical-bootstrap synthetic; fallback to β-with-T2-b-fix specified).

---

## Part 5 — Execution order (risk-minimizing)

Per user standing instruction: "produce a recommended execution order that minimizes risk and avoids building agent integration on top of unresolved epistemic/schema problems."

**Phase X1 — Framework-doc correction (1-2 hours, sequential, blocks all else).**
- Apply the four replacement-text blocks from Part 1 to `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md`.
- Re-count the iteration budget: Phase P decile-rank-log-RV = 1 observer (with 3 variants), not 3.
- Exit: doc committed; budget re-accounted; O1/O2/proposed-O3 reported as variants in a single row.

**Phase X2 — Observer 4 pre-registration (half-day).**
- Commit `preregistration_observer4.md` specifying α sign-vector-index, marginal MI on 3×3, permuted-generator + marginal-orbit-iid nulls, Miller-Madow bias correction, AND-conjunction at uncorrected α=0.05, training kill-gates (MI ≥ 0.005 nats, χ² p < 0.01, expected cells ≥ 5, orbit-marginal min ≥ 5%), test-fold 0.5× floor as soft gate, asset-order = alphabetical, sign tie-break = `r ≥ 0 → +1`.
- SHA-256 of 5 asset CSVs frozen at pre-reg commit time.
- Exit: pre-reg committed to git; data hashes frozen.

**Phase X3 — Synthetic prerequisite (half-day compute, blocks X4).**
- Run α on 500 seeds of empirical-bootstrap 5-asset panels (each seed: iid sample from each asset's own training-fold daily returns, preserving marginal; cross-asset synchrony destroyed).
- Measure empirical FPR per null at uncorrected α=0.05.
- Exit-PASS: each null's empirical FPR ≤ 0.03 at N=500 (or ≤ 0.05 at N=500 with 95% upper-CI ≤ 0.06). Proceed to X4.
- Exit-FAIL: move to β (with T2-b-fix: rank-of-eigenfraction quantization). Reset X2-X3 for β. Budget slot still counts as "O4 attempt".

**Phase X4 — Observer 4 training-fold diagnostic (1 day).**
- Apply α pipeline to real panel's training fold. Check all four kill-gates.
- Exit-PASS: all gates met. Proceed to X5.
- Exit-FAIL: declare training NULL, log diagnostic. Do NOT compute test-fold statistic. Exit budget slot consumed.

**Phase X5 — Observer 4 test-fold evaluation (half-day).**
- Run `run_blinded_orbit_transition_v2.py` (separate script; designer does not see test results until script emits JSON). Compute MI, two nulls, p-values.
- Exit-PASS: MI rank-p ≤ 0.05 on both nulls, test MI ≥ 0.5× training MI. Report POSITIVE, write cert candidate. Budget consumed: 2/5 observers (decile-rank + sign-vector).
- Exit-FAIL: declare NULL. Budget consumed: 2/5. Three slots remain. Do NOT immediately launch β or γ; return to Phase R existing-art review and map a genuinely different observer family (e.g., eigenspectrum-with-T2-b-fix, or a QCI-adjacent k-means-with-partition-info-invariant re-design per OB 2026-04-17T00:39:39 follow-up item).

**Phase X6 — Sensitivity analyses (only if X5 PASS).**
- Asset-order sensitivity: run α under all 120 asset orderings, report MI distribution.
- m ∈ {9, 24} confirmatory: same observer, different modulus.
- Held-out panel: sector ETFs (XLF, XLE, XLI, XLK, XLV) as independent-panel replication.
- Exit: sensitivity report committed. Cert-grade witness finalized only if original result survives ≥ 100/120 orderings at rank-p ≤ 0.05 and replicates on the sector-ETF panel.

**Critical ordering note.** X1 precedes everything else because running O4 under a framework doc that doesn't distinguish family-crossing from parameter-sweep is building on unresolved schema — exactly what the user's standing instruction prohibits. X3 precedes X4 because launching on real data under a statistic with uncalibrated FPR is the prior critic's #1 concern about the O3 plan.

---

## Part 6 — Sign-off verdict

### Framework doc (Part 1)
**SIGNED-OFF-WITH-AMENDMENTS.** Four replacement-text blocks in Part 1 must be applied. See §1.2 family-discipline insert, §1.5 normative cross-family append, §2 heading re-label, §1.2 null-specification replacement.

### Observer 4 plan (Parts 2–4)
**SIGNED-OFF-WITH-AMENDMENTS.** Selected candidate α over β and γ. Five amendments required:
1. Pre-register asset order as alphabetical (GLD, SPY, TLT, USO, UUP) with all-120-ordering sensitivity in X6 (addresses α-smuggling-1).
2. Pre-register sign tie-break `r ≥ 0 → +1` (addresses α-smuggling-2).
3. Add training-fold marginal-check (min orbit marginal ≥ 5%, else variant retry) (addresses Concern 3.2).
4. Pre-register n_null = 1000 per null, seeds ∈ {42, 43} (addresses Concern 3.5).
5. Prerequisite tightened: N=500 seeds, empirical-bootstrap synthetic, fallback to β-with-T2-b-fix (addresses Concerns 4.1, 4.2, 4.3).

### Smuggling patterns found in this O4 plan (new ones, beyond prior critics)
- **α-smuggling-1 (asset-order sensitivity):** `Σᵢ (s==+1) × 2^i` implicitly canonicalizes asset order. New; not found by prior critics.
- **α-smuggling-2 (sign tie-break undefined):** zero-return edge case. New.
- **β-smuggling-1 (rolling-window AR-structure by construction):** inflates synthetic prerequisite FPR. New.
- **β-smuggling-2 (float-mod-cast T2-b violation):** CLAUDE.md explicit T2-b violation in β specification `qa_mod(round(φ × m), m)`. New.

The plan's two nulls + AND-conjunction + Miller-Madow + training kill-gates + synthetic prerequisite already address the 11 prior smuggling patterns. The four new patterns above are all pre-registerable / structural fixes, not observer-family issues.

### If amendments not applied
Reverts to **NOT-SIGNED-OFF** on O4 (α-smuggling-1 specifically makes the effect size dependent on asset-ordering convention, which is a p-hacking degree of freedom the pre-reg must close).

---

## References

- Bollerslev, T., Patton, A.J. & Quaedvlieg, R. (2016). "Exploiting the errors: A simple approach for improved volatility forecasting." *Journal of Econometrics* 192(1):1-18. DOI: 10.1016/j.jeconom.2016.02.010.
- Horowitz, J.L. (2001). "The Bootstrap." In *Handbook of Econometrics* 5:3159-3228. Elsevier. DOI: 10.1016/S1573-4412(01)05005-X.
- Miller, G.A. (1955). "Note on the bias of information estimates." In *Information Theory in Psychology*, Free Press, pp. 95-100.
- Paninski, L. (2003). "Estimation of Entropy and Mutual Information." *Neural Computation* 15(6):1191-1253. DOI: 10.1162/089976603321780272.
- Pearson, K. (1896). "Mathematical contributions to the theory of evolution." *Philosophical Transactions of the Royal Society A* 187:253-318.
- Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap." *Journal of the American Statistical Association* 89(428):1303-1313. DOI: 10.1080/01621459.1994.10476870.
- Theiler, J., Eubank, S., Longtin, A., Galdrikian, B. & Farmer, J.D. (1992). "Testing for nonlinearity in time series: the method of surrogate data." *Physica D* 58(1-4):77-94. DOI: 10.1016/0167-2789(92)90102-S.

**Companion artifacts cited, with file/line grounding:**
- `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` lines 18, 26-29, 30-33, 46, 58-72, 80, 95-96, 103-108 (framework doc sections audited).
- `docs/theory/QA_FINANCE_FRAMEWORK_AUDIT.md` lines 22-30 (Pattern 1), 80-86 (Pattern 5), 91-93 (Pattern 6), 104, 152, 177-180 (prior audit).
- `docs/theory/QA_FINANCE_O3_PLAN_CRITIC_REVIEW.md` lines 15-18 (concern 1), 19-25 (concern 2), 43-47 (concern 6), 48-50 (concern 7), 51-54 (concern 8) (prior O3 critic).
- `docs/theory/QA_FINANCE_QCI_PUBLIC_REVALIDATION.md` lines 47-57, 65-68 (QCI public re-val: raw r=+0.4355, partial r=+0.2556, 4/4 raw / 2/4 partial Bonferroni).
- `qa_lab/phase_p_observer1.py` lines 86-105 (decile quantization), 100-101 (qa_mod call), 109-114 (e inference), 194-197 (QLIKE), 340-356 (block bootstrap), 381-399 (AR(1)), 422-443 (phase randomization).
- `qa_lab/qa_observer/core.py` line 27 (`qa_mod` canonical).
- `qa_orbit_rules.py` lines 59, 72, 85-95 (`qa_step`, `orbit_family`, axiom asserts).
- `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/qa_signal_generator_inference_cert_validate.py` lines 48-74 ([209] A1-closure + uniqueness verification).
- `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/fixtures/sgi_pass_default.json` line 71 (finance witness: raw r=+0.037 p=0.025 (*); partial ns).

**Absent in current repo (explicit):**
- No `qa_alphageometry_ptolemy/*eeg*/` cert family (confirmed in framework-doc §1.5 line 59 — EEG progression is in runtime + OB, not cert-formalized).
- No "observer family" or "QA-native null" terminology in `docs/specs/QA_FINANCE_OBSERVER_ITERATION.md` (grep on the file; zero occurrences).
- No `preregistration_observer4.md` yet (only `preregistration_observer1.md`, `preregistration_observer2.md` via glob).
- No `run_blinded_orbit_transition_v2.py` yet.

**Unverifiable in current repo state:**
- The prior O3 critic doc references OB entries that exist in the OB recent-thoughts dump at `/home/player2/.claude/projects/-home-player2-signal-experiments/29fb4e72-0233-454c-810f-46096fdef9b5/tool-results/mcp-open-brain-recent_thoughts-1776392944068.txt` but I have not verified every OB citation in the prior critic docs against that dump. My own citations of OB go through that dump and are verified (lines 23, 45-76, 79, 96, etc.).

---

*Word count: ~2,950. Section counts: six numbered parts + references. No code, no experiments.*
