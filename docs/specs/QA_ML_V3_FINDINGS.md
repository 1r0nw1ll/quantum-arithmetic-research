# QA-ML v3 Pilot — Findings (2026-05-15)

> Status: **pilot results only**. Per the v3 plan
> (`docs/specs/QA_ML_V3_STRUCTURE_DISCOVERY_PLAN.md`), the pilot was a
> feasibility check for the ML-as-discoverer thesis. This note reports
> what the pilot showed and what to do next; it does **not** propose any
> new cert family.

> Plan reference: `docs/specs/QA_ML_V3_STRUCTURE_DISCOVERY_PLAN.md` (commit `b48cafd`).
> Experiment: `experiments/qa_ml/04_orbit_structure_discovery.py`.
> Protocol: `experiments/qa_ml/benchmark_protocol_v3.json` (9/9 gates PASS).
> Results: `experiments/qa_ml/results_v3_structure_discovery.json`.
> Distilled tree text: `experiments/qa_ml/results_v3_decision_tree.txt`.

## Scope

- **Task**: T1 only (3-class shortcut failure-mode prediction: correct /
  undercount / overclaim).
- **Models**: `qa_full_logreg` (sklearn LogisticRegression on the 23-int
  v3 feature packet) and `decision_tree` (`DecisionTreeClassifier`
  max_depth=12, class_weight=balanced).
- **Training moduli**: `M_train = {9, 10, 12, 15, 18, 20, 21, 24, 25, 30}`
  (10 moduli, 3,816 states).
- **Test moduli**: `M_test = {7, 8, 11, 30, 45, 75}` (6 moduli, 8,784
  states). Note `m=30` appears in both as an in-distribution check; the
  out-of-distribution moduli are 7, 8, 11, 45, 75.

## Headline result

The decision tree **structurally rediscovered cert [278]** at depth 3 of
the distilled tree:

```text
phi_b <= 0.5  AND  phi_e <= 0.5  AND  fac_3 <= 0.5  →  class 2 (overclaim)
phi_b <= 0.5  AND  phi_e <= 0.5  AND  fac_3 >  0.5  →  class 0 (correct)
```

Translating: when the shortcut says satellite (both coordinates
divisible by `m // 3`, i.e. `phi_b = phi_e = 0`) **and** 3 does not
divide m (`fac_3 = 0`), the tree predicts overclaim. **This is exactly
the [278] rule.**

Empirical rediscovery scores on held-out test moduli:

| Model | macro F1 | bal. acc. | rediscover_277 | rediscover_278 | m=8 overclaim | m=75 undercount |
|---|---|---|---|---|---|---|
| `qa_full_logreg` | 0.367 | 0.484 | 0.677 | 0.389 | 0.000 | 0.031 |
| `decision_tree` | **0.729** | **0.797** | 0.417 | **1.000** | **1.000** | 0.000 |

The tree:
- ✅ **Perfectly rediscovers [278] overclaim** (1.000 across the test
  3 ∤ m moduli, plus 1.000 recall on the m=8 boundary exception where
  the actual overclaim count is 15 not 9 — the cert's structural rule
  generalizes even though the count rule doesn't).
- ⚠ **Partially rediscovers [277] undercount** (0.417). The tree's
  in-distribution behavior on m=30 (k=2, seen at training time)
  matters; out-of-distribution k=3 (m=45) and k=5 (m=75) are missed.
- ❌ **Cannot extrapolate to m=75** (0.0). The training set has k=1
  and k=2; the tree learned the gcd-signature for those specific k's
  rather than the (k, 3k), (k, k), (3k, k) parametric pattern.

The LogReg baseline is worse than the tree across the board, confirming
the failure-mode boundaries are non-linear in the v3 feature space.

## Feature importance from the tree

Top 8 features by Gini importance:

```text
phi_b      0.460   (b mod m//3)
fac_3      0.164   (3-adic valuation of m)
m_mod_5    0.135   (m mod 5)
phi_e      0.134   (e mod m//3)
psi_e      0.038   (e mod 5)
gcd_e_m    0.026
gcd_b_m    0.024
psi_b      0.013   (b mod 5)
```

Reading this: the model's primary discriminators are exactly the
features the certified rules use. `phi_b, phi_e` carry the divisor
shortcut; `fac_3` distinguishes the over-claim regime ([278]); `m_mod_5,
psi_b, psi_e, gcd_*` carry the [277] cluster.

## Interpretation

**The tree learned the divisor shortcut, then corrected it.** Branches
that test `phi_b <= 0.5 AND phi_e <= 0.5` recover the shortcut's
predictions; branches that test `fac_3` reject the predictions when
3 ∤ m. The structure of the corrections is exactly the [278] rule.

**The undercount rule was not fully learned.** The [277] gcd-signature
decomposition `{(k, 3k): 8, (k, k): 16, (3k, k): 8}` is parametric in
k = m / 15. The tree had m=15 (k=1) and m=30 (k=2) in training. It
learned specific gcd values, not the parametric form, so test m=45/75
(k=3, k=5) miss the rule. To rediscover [277] cleanly, the model needs
either:

1. More training k values, OR
2. A feature that encodes k-relative position (e.g.
   `gcd(b, m) / (m // 15)`), OR
3. A non-tree model (GCN, symbolic regression) that can express
   "parameter-relative" rules.

## Decision criteria reading

Per the v3 plan, this pilot lands as:

> **Partial success.** One certified rule rediscovered cleanly ([278]);
> the other near-rediscovered (0.417 < 0.70 threshold) with a clear
> structural explanation (parametric in k, training only covered k ∈
> {1, 2}).

The pilot validates that **ML-as-discoverer is tractable on this
problem**: a single distilled decision tree extracted a certified rule
from data and the m=8 boundary exception was handled correctly. The
remaining gap is the k-extrapolation problem on [277], which has three
clear remediation paths.

## What this pilot proved

1. The 23-feature v3 packet is sufficient for the decision tree to
   express [278]. No GCN was needed.
2. Decision-tree distillation is a viable T3 extraction method when the
   target rule lives at modest depth (≤ 4) and uses a small number of
   features. The [278] rule meets both conditions.
3. The CART's first three splits expose the divisor shortcut's failure
   surface in human-readable form. No additional symbolic regression or
   rule-mining was needed to recover this rule.
4. **m=8 generalization works** even though the model never saw m=8 in
   training. The tree learned the structural condition (3 ∤ m AND
   shortcut-says-satellite) rather than a memorized table.

## What this pilot does NOT prove

- That [277] is rediscoverable. The pilot's training set was too narrow
  in k. A v3.1 run with `M_train ⊇ {15k : k ∈ {1, 2, 3, 4, 5}}` could
  confirm or refute this.
- That the GCN baseline helps. The pilot omitted qa_gcn; whether the
  graph topology adds signal beyond the v3 feature packet is still
  open.
- That T2 (period prediction) is tractable. Only T1 was tested.
- That m=75 anomaly can be flagged by ML. The model treated m=75 like
  any other 15k modulus; the period-doubling anomaly requires a richer
  representation than per-state features.

## Proposed v3.1 next moves (decision-gated)

Three options, each one focused experiment:

**Option A — k-extrapolation test for [277].**
Extend `M_train` to `{9, 12, 15, 18, 20, 21, 24, 25, 27, 30, 36, 45,
60, 75, 90}` (15 moduli covering k=1..6 plus controls). Re-run pilot.
If the tree's rediscover_277 jumps to ≥ 0.95, **[277] is also
rediscoverable** — strong v3 success.

**Option B — qa_gcn baseline.**
Add the v2 GCN trained per-modulus over the v3 feature packet, evaluate
T1 with the same M_train / M_test split. Compares feature-only vs.
graph + feature on the structure-discovery task. If GCN beats tree,
the graph topology carries information that distillation needs to
exploit (e.g. graph-distillation methods like GraphLIME).

**Option C — T2 pilot.**
Repeat the T1 pilot setup on T2b (5-class period prediction) instead.
Tests whether the feature packet is sufficient for period-class
prediction, or whether ψ_b, ψ_e (mod-5 phase) alone is enough.

**Recommendation**: **Option A first**. The pilot's only soft spot is
k-extrapolation on [277], and the fix is mechanical (more training
moduli, same models). If A succeeds, we have a clean two-cert rediscovery
result — the strongest possible v3.0 deliverable — and B / C become
optional follow-ups.

## Out-of-scope (do not pursue without an explicit ask)

- A separate cert family for the rediscovery result. The v3 thesis
  rules this out by design.
- Symbolic regression (PySR) or RIPPER mining. The tree gave us [278]
  cleanly; trying more methods on the same target is redundant.
- A wider orbit-period sweep. The [279] draft remains parked.

## Update 2026-05-15: Option A null result + sharper diagnosis

**Option A** (extended `M_train` to k ∈ {1, 2, 3, 4, 6} in the m=15k
family, held out k=5 (m=75), k=7 (m=105), k=8 (m=120)) did **not**
improve [277] rediscovery. `rediscover_277` went from 0.417 to 0.292
across the new test moduli; `m=75` undercount remained 0; [278] stayed
at 1.000.

Inspecting the Option A tree text
(`results_v3_option_a_decision_tree.txt`) revealed the structural cause:

```text
|--- fac_3 <= 0.50          (3 ∤ m — the [278] regime)
|   |--- phi_e <= 0.50
|   |   |--- phi_b <= 0.50
|   |   |   |--- class: 2   ← [278] rule, perfect
|--- fac_3 >  0.50          (3 | m — the [277] regime)
|   |--- psi_b <= 0.50
|   |   |--- class: 0
|   |--- psi_b >  0.50
|   |   |--- fac_5 <= 0.50
|   |   |   |--- class: 0
|   |   |--- fac_5 >  0.50
|   |   |   |--- m <= 52.50  ← splits per modulus, not parametrically
|   |   |   |   |--- ...complex per-feature thresholds...
```

The tree's split **`m <= 52.50`** is the smoking gun. At max_depth=12,
CART literally branches on absolute modulus value to handle the [277]
regime — it memorizes per-modulus thresholds rather than learning the
parametric `(k, 3k), (k, k), (3k, k)` signature shape. Adding more `k`
values to training gave the tree more reference points to memorize,
not a parametric rule to abstract.

The model failed at m=75 specifically because m=75 has `fac_5=2`
(the only 5²·3 modulus in the dataset), and the tree learned that
`fac_5=1` cases work one way while `fac_5=2` was out of distribution.
The Option A run validated this: the tree partially extrapolated to
m=105 and m=120 (`fac_5=1`) but completely failed at m=75 (`fac_5=2`).

### Refined v3 thesis

The original v3 thesis ("ML can rediscover certified rules from data")
needs qualification:

> **Refined thesis (2026-05-15):** A simple feature-distillation
> approach (sklearn CART with the v3 packet) can rediscover
> **non-parametric structural rules** that decompose into a small
> Boolean combination of feature-threshold tests — exactly the form of
> [278]. The same approach cannot rediscover **parametric rules** that
> are quantified over a continuous index (k = m/15 in [277]) because
> CART cannot express parametric quantification over numerical features.

[278] is `∀ m. 3 ∤ m ∧ m ≥ 7 ∧ m ≠ 8 → (the 9 false satellites are
exactly {a·m//3, b·m//3 : a, b ∈ {1, 2, 3}})`. The tree captures the
universal quantifier as a feature-threshold conjunction.

[277] is `∀ k. 0 < k ≤ K_max → (32 missed satellites at m=15k partition
by gcd-signature as (k, 3k):8, (k, k):16, (3k, k):8)`. The tree cannot
express "the 32 missed states sit at gcd values that are integer
multiples of k" because k is a parameter relative to m, not a feature
of (b, e).

## Pepe (2025) connection — context worth capturing

The user pointed to *Machine Learning with Geometric Algebra* (Pepe,
PhD thesis, Cambridge, August 2025, `/Users/player3/Downloads/2025-pepe.pdf`,
228 pages). The thesis's premise directly addresses what QA-ML v3.0
just discovered:

> "if, as it is often said, ML is a clever rebranding of linear algebra,
> then geometric problems in ML deserve to be tackled with GA, an
> extension of linear algebra designed to represent geometric objects
> and perform transformations on them naturally and compactly."

By analogy: **QA problems in ML deserve to be tackled with QA-native
features** that embed the algebraic structure (orbit period, generator
graph, k-relative gcd signatures), not just generic integer features.
Cert [276] showed this for graph topology (GCN beats MLP). v3.0 just
showed it for [278] (the right feature combination distills the rule).
v3.0's [277] failure points at the next gap: **embed `k = m // 15` as a
first-class feature, not as a learned threshold**.

Pepe's chapters 2, 4, 5 (rotations as rotors, pose estimation, PDE
solving with STA) all share this pattern: embed the right algebra into
the model architecture, get lower errors and interpretable
intermediates. The QA analog is what cert [276] already did
(generator-graph adjacency as a fixed structural input). For [277]
rediscovery, the QA analog would be a **k-quotient feature transform**
that exposes the parametric ratio directly.

## v3.1 recommendation (sharpened)

Given the Option A null result and the Pepe-style structural lesson,
the right next step is **NOT** to add another model class or more
training data. It's to **add k-quotient features** to the v3 packet
and re-test:

```text
gcd_b_m_over_k = gcd(b, m) // (m // 15)     where m // 15 == k for m=15k
gcd_e_m_over_k = gcd(e, m) // (m // 15)
gcd_ratio_signature = (gcd_b_m_over_k, gcd_e_m_over_k)
```

For m = 15k missed-satellite pairs, this signature equals exactly
{(1, 3), (1, 1), (3, 1)} regardless of k. The decision tree could
then express the [277] rule as a single feature-threshold combination
on these k-quotient features.

If `rediscover_277 ≥ 0.95` after adding these features, the v3 thesis
is confirmed in its refined form: **non-parametric rules are
distillable directly; parametric rules become distillable after the
right invariant features are exposed**. This makes the rule-extraction
workflow a two-stage discovery: (i) find the parametric invariant
from theory or graph structure, (ii) distill the rule from data.

## Lineage

```text
e7b2af0  fix(qa_orbit_rules): canonical orbit_family via orbit_period
[277]    Pisano 5-factor under-count boundary
[278]    No-3-divisor over-claim boundary
418b73c  docs(orbit): park [279], write synthesis closing the cert chain
b48cafd  docs(qa_ml): v3 structure-discovery plan
aa6adb1  feat(qa_ml): v3 pilot — rediscovers [278] from data
<this>   v3 Option A run + Pepe synthesis + v3.1 k-quotient proposal
```

## Update 2026-05-15 (round 2): v3.1 k-quotient run

Added two features to the v3 packet per the round-1 v3.1 proposal:

```python
gcd_b_m_over_k = gcd(b, m) // max(1, m // 15)
gcd_e_m_over_k = gcd(e, m) // max(1, m // 15)
```

Smoke test (`tools/qa_ml/qa_features_v3.py`) verifies the k-quotient
invariance: for missed-satellite gcd signatures at m = 15k (k ∈
{1, 2, 3, 5, 8}), the features collapse to fixed (1, 3), (1, 1), (3, 1)
regardless of k. The packet grew from 23 to 25 features.

### Run A: pilot training + k-quotient (no factor-structure fix)

Re-ran original pilot config (M_train = 10 small moduli) with new
features. `rediscover_277` = 0.333 (vs original pilot 0.417 — slightly
worse). The features were available but the training set didn't
expose the parametric invariant cleanly across k.

### Run B: Option A training + k-quotient (Option A retest)

Same Option A M_train (k=1, 2, 3, 4, 6 in m=15k space) with k-quotient
in packet. `rediscover_277` went from 0.292 to 0.219; m=75 stayed at 0.
Deeper trees (max_depth ∈ {18, 25, None}) made it worse (0.062, 0.042,
0.021) — overfit per-modulus when given more capacity.

**Diagnosis**: the `fac_3 > 0` branch of training had **zero**
fac_5=2 examples. Train fac_5 distribution in the 3|m branch:
`{1: 14850, 0: 3591}`. Test fac_5 distribution in 3|m: `{1: 25425,
2: 5625, 0: 1089}` — the 5625 fac_5=2 test states all sit at m=75
and are completely OOD on fac_5.

### Run C: v3.1 = Option A training + m=150 + k-quotient features

Added m=150 = 2·3·5² to training. m=150 has the **same prime factor
structure as m=75** (both are 3¹·5²). Hypothesis: if training has at
least one modulus with the test's factor pattern, the k-quotient
features should enable parametric extrapolation across k.

Result (verified 2026-05-15):

```text
                model  macro_f1  rediscover_277  rediscover_278  m75_under
       qa_full_logreg     0.392           0.333           1.000      0.375
        decision_tree     0.744           0.458           1.000      0.500
```

**m=75 undercount jumped from 0.000 → 0.500.** Overall
`rediscover_277: 0.292 → 0.458`. The k-quotient features now sit at
positions #2 (gcd_b_m_over_k: 0.282) and #4 (gcd_e_m_over_k: 0.117) in
the tree's feature importance ranking — behind only m_mod_3 (0.492).

LogReg also improved: `m=75 undercount: 0.000 → 0.375`. The
representational fix helped both model classes.

### v3.2 refined thesis

```text
ML rediscovers cert rules from data IF:
  1. The parametric invariant is exposed as a feature (k-quotient).
  2. Training covers the test set's prime factor structures.
  3. The model class has enough depth to express the rule — but not so
     much that it overfits the long tail (CART at "natural" depth).
```

Conditions 1+2 pushed m=75 rediscovery from 0 to 0.5. Conditions 1+2
are sufficient for [278] (no parametric extrapolation needed). [277]
requires all three plus diverse training across factor patterns.

### Why rediscover_277 isn't 0.95 yet

The remaining gap from 0.458 to 0.95 is OOD factor structure on m=105
(only fac_7≥1 test point) and m=120 (only fac_2=3 test point with 3|m).
Same pattern as m=75 before adding m=150: the tree can extrapolate
WITHIN a factor structure it has seen, not BETWEEN factor structures.

To reach `rediscover_277 ≥ 0.95`, training would need diverse fac_7,
fac_2 examples. But that approach starts to feel like "make the
training set look like the test set," which defeats the rule-
rediscovery goal.

### Honest v3.1 conclusion

> Decision-tree distillation, even with the right k-quotient features,
> cannot extrapolate across prime factor structures. It can only apply
> within-structure rules learned from training. To go further, the
> model class itself needs to express "ignore the prime structure of
> m and apply the same parametric rule" — i.e. an **equivariant model
> in the GA-style sense of Pepe (2025)**.

This is a clean stopping point for the distillation thread. The v3
deliverable is: (i) [278] rediscovered cleanly by distillation, (ii)
[277] partially rediscovered with feature engineering, (iii) full
[277] rediscovery requires an equivariant architecture, not more
features or training data.

### v3.2 proposal (NOT auto-promoted)

If the user wants to chase `rediscover_277 ≥ 0.95`, the right v3.2 is
a **modulus-factor-equivariant model**: predictions should be invariant
under `(b, e, m) → (c·b, c·e, c·m)` for prime scalings c. Build it as
an equivariant GCN over the QA generator graph with shared parameters
across modulus orbits. This is the Pepe-thesis pattern (chapters
2/3/4/5: rotors, GA-equivariant projector, sandwich product layers,
STAResNet) applied to QA's modular structure.

That's a v3.2 research project, not a 10-minute experiment. Flagged
here; NOT auto-promoted without explicit user direction.

## Update 2026-05-15 (round 3): v3.2 — equivariance architecture, STRONG SUCCESS

Per v3.1's honest conclusion ("decision-tree distillation cannot
extrapolate across prime factor structures; needs an equivariant model
in the GA-style sense of Pepe (2025)"), Will Dale greenlit a full v3.2
build to test the equivariance hypothesis structurally.

### Phase 0 — equivariance precondition verified empirically

Critical pre-check before designing the architecture: is the
hypothesized invariance actually exact?

| Test | Scope | Match rate |
|---|---|---|
| Orbit period equivariance | m ∈ {7..25}, c ∈ {2..7}, all (b, e) | **900/900 = 100%** |
| Failure-mode equivariance, cert [277] scope | m = 15k, k ∈ {1..5}, missed pairs × c ∈ {2..7} | **768/768 = 100%** |
| Failure-mode equivariance, cert [278] scope | 3 ∤ m overclaim pairs × c ∈ {2..7} | 36/243 = 14.81% |

Result: **`orbit_period` is fully equivariant** under
`(b, e, m) → (cb, ce, cm)`. The canonical orbit_family inherits this.
The divisor shortcut's `(m // 3)` predicate is NOT scale-equivariant,
so [278]'s failure mode breaks under scaling.

Implication: an architecturally equivariant model is exact on the
[277] regime by construction — but the regime asymmetry must be
handled, not assumed.

### Phase 1 — v3.2.0: canonical features added to v3.1 packet

Added 5 canonical features to qa_features_v3:
```python
g = gcd(gcd(b, e), m)
canonical_b = b // g
canonical_e = e // g
canonical_m = m // g
canonical_k = canonical_m // 15 if canonical_m % 15 == 0 else 0
```

Smoke test:
```text
(1,3,15)  → canonical=(1,3,15) g=1
(2,6,30)  → canonical=(1,3,15) g=2
(3,9,45)  → canonical=(1,3,15) g=3
(5,15,75) → canonical=(1,3,15) g=5
(8,24,120)→ canonical=(1,3,15) g=8
```

All c-scalings of (1, 3, 15) collapse to (1, 3, 15). Packet grew from
25 to 30 features.

Re-ran with `QA_ML_V3_OPTION=v3_2_0` on the same M_train / M_test as
v3.1.

```text
Results summary:
                model  macro_f1  rediscover_277  rediscover_278  m8_over  m75_under
        decision_tree     0.793           0.979           1.000    1.000      1.000
       qa_full_logreg     0.472           0.854           1.000    1.000      0.562

Top tree feature importances:
  m_mod_3       0.486   (regime split)
  canonical_m   0.476   (within-regime invariant)
  phi_e         0.029
  phi_b         0.008
```

**Both v3.2 success criteria met at the FIRST phase.** Tree gets:
- `rediscover_277 = 0.979` (jumped from v3.1's 0.458, +0.521)
- `rediscover_278 = 1.000` (unchanged)
- **m=75 undercount = 1.000** (jumped from 0.500, full extrapolation)

The tree's top split is `m_mod_3 ≤ 0.5` (the 3|m vs 3∤m regime split).
Inside the 3|m branch, `canonical_m` becomes the second-tier
discriminator: `canonical_m > 13.5` (i.e. canonical_m = 15) captures
the [277] regime exactly, and deeper splits identify the missed
satellites within it. The tree learned to use canonical_m as a
regime marker for [277].

### Phase 2 — v3.2.1: canonical-only features (architectural test)

To validate the structural claim, ran the same script with feature
mask restricted to the 5 canonical features only (b', e', m', g, k').

```text
                model  rediscover_277  rediscover_278  m8_over  m75_under
        decision_tree           1.000           0.000    0.133      1.000
       qa_full_logreg           1.000           1.000    0.800      1.000
```

**Equivariance hypothesis architecturally confirmed:**
- `rediscover_277 = 1.000` perfect (canonical-only model is exact on
  the equivariant regime by construction)
- `rediscover_278 = 0.000` collapsed (tree) — equivariance breaks for
  [278], so a canonical-only tree throws away the discriminator
- LogReg surprisingly handles [278] (0.800 on m=8) because logistic
  features can pick out specific canonical_m values that correlate
  with overclaim regime

The asymmetric result is exactly what the equivariance precondition
predicted: [277] is exact under canonicalization; [278] is not.

### v3.2 verdict — STRONG SUCCESS

Per the plan's tier thresholds:

> **Strong:** rediscover_277 ≥ 0.95 AND rediscover_278 ≥ 0.95 AND
> m=75 = 1.000 AND m=8 = 1.000

v3.2.0 hits all four:
- rediscover_277 = 0.979 ≥ 0.95 ✓
- rediscover_278 = 1.000 ≥ 0.95 ✓
- m=75 undercount = 1.000 ✓
- m=8 overclaim = 1.000 ✓

v3.2.2 (hybrid model) is not needed — v3.2.0's tree naturally learned
to use canonical features for the equivariant regime and non-canonical
features for the non-equivariant regime, all within a single
DecisionTreeClassifier.

### Final v3 thesis (after v3.2)

```text
ML rediscovers cert rules from data IF:
  1. The right algebraic invariant is exposed in the feature space.
       — k-quotient features (v3.1) help WITHIN a factor structure
       — canonical features (v3.2) enforce equivariance across
         factor structures
  2. Training is non-degenerate (any cert-scope example suffices when
     the model can leverage equivariance).
  3. The model class can express the rule at modest depth (CART at
     max_depth ≈ 12).
```

v3.2 needed only condition 1 (canonical features as algebraic
invariant). Training did NOT need to cover every prime factor
structure — the canonical map collapses all factor structures to the
same representative.

### Pepe (2025) connection — confirmed across QA-ML chain

The pattern across the v3 chain matches Pepe's thesis claim that
"geometric problems in ML deserve to be tackled with [the right
algebra]":

| Layer | Algebra-aware feature/structure | Effect |
|---|---|---|
| [276] | QA generator graph as adjacency | GCN > MLP by +0.14..+0.27 macro F1 |
| v3 pilot | qa_full_v3 features (phi_b, fac_3, ...) | Tree distills [278] cleanly (1.000) |
| v3.1 | k-quotient features (gcd_*_m_over_k) | [277] within-structure rediscovery improves (0.46 from 0.29) |
| v3.2 | canonical features (b/g, e/g, m/g, k) | Both certs rediscovered (0.98 / 1.00). Equivariance closes the gap. |

Each layer of the QA-ML chain corresponds to a deeper algebraic
structure being exposed. The cleanest result (v3.2.0) is also the
simplest architecturally: 5 features added to a standard sklearn
tree.

### What v3.2 did NOT prove

- That a structural equivariant model (Option γ, shared-parameter
  GCN) would beat v3.2.0. v3.2.0 already hits the ceiling; v3.3 is
  unnecessary.
- That T2 (orbit period class) is similarly easy. T2 was deferred in
  the v3 plan and remains untested.
- That the pattern extends to other cert families (Whittaker, etc.).
  v3.2 is bounded to the divisor-shortcut failure surface.

### Lineage (round 3)

```text
9c75f55..5b6c5cc  (prior chain)
7792e14           v3.1 — k-quotient features, partial rediscovery
b48cafd           v3.0 design plan
<this>            v3.2 — equivariance precondition + canonical features;
                  STRONG SUCCESS at v3.2.0; v3.2.1 confirms architectural
                  mechanism
```

### Clean stop for the v3 chain

The v3 thesis is fully validated. Both certified rules rediscovered
from data using a single sklearn decision tree on 30 integer features.
The Pepe-style algebra-aware pattern is empirically confirmed across
the QA-ML chain. Further structural extensions (v3.3 GCN, T2, other
cert families) are deferred — they would not change the v3
conclusion.

## External references added

- **Pepe, A. (2025).** *Machine Learning with Geometric Algebra:
  Multivectors for Modelling, Understanding and Computing.* PhD
  thesis, University of Cambridge, Department of Engineering, Darwin
  College. 228 pp. (companion file
  `/Users/player3/Downloads/2025-pepe.pdf`). Establishes the broader
  pattern: embedding the right algebra into ML architecture gives
  lower errors, equivariance, and interpretable intermediates. The
  QA-ML chain's [276] (graph topology), [277]/[278] (algebraic rule
  rediscovery), and v3.1 (k-quotient feature transform) are all
  instances of this pattern applied to QA's modular structure.

## References

- `docs/specs/QA_ML_V3_STRUCTURE_DISCOVERY_PLAN.md` — full v3 plan.
- `docs/specs/QA_ML_ORBIT_DISCOVERY_SYNTHESIS.md` — chain-closing
  synthesis that motivated v3.
- `qa_alphageometry_ptolemy/qa_orbit_no_3_divisor_overclaim_cert_v1/` — [278].
- `qa_alphageometry_ptolemy/qa_orbit_pisano_5_factor_boundary_cert_v1/` — [277].
- `qa_orbit_rules.py` — canonical orbit_family + divisor shortcut helper.
- Wall, D. D. (1960). DOI: 10.1080/00029890.1960.11989541.
