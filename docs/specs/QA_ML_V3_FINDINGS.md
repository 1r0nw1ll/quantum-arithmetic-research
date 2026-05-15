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
