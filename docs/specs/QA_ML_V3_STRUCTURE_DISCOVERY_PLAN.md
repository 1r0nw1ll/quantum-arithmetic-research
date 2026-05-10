# QA-ML v3 — Structure Discovery Plan

> Status: **DESIGN DOC ONLY**. No code, no certs, no sweep extensions
> until this plan is reviewed. Per `docs/specs/QA_ML_ORBIT_DISCOVERY_SYNTHESIS.md`,
> the v3 thesis is **ML-as-discoverer**: train models that re-derive the
> certified rules ([277], [278]) from data, treat the certs as the
> evaluation oracle, and surface candidate new theorems only when they
> pass an independent check. Cert promotion is **not** a downstream
> deliverable of this plan.

> Primary source for the GCN architecture: Kipf & Welling 2017,
> Semi-Supervised Classification with Graph Convolutional Networks,
> ICLR. arxiv:1609.02907.

> Primary source for the orbit-period framing: Wall, D. D. (1960),
> Fibonacci series modulo m, American Mathematical Monthly 67(6),
> 525-532. DOI: 10.1080/00029890.1960.11989541.

## Thesis

The previous QA-ML certs ([276] graph topology, [277] under-count, [278]
over-claim) showed the QA generator graph carries information about the
divisor shortcut's failure surface. v3 turns that result into a
**theorem-discovery workflow**: train a model that maps `(b, e, m)` →
shortcut failure mode and orbit period, then **read the model**
(symbolic extraction, decision-rule mining, gradient-attribution
analysis) and check whether the extracted rules match the certified
ones.

The success metric is **rule rediscovery**, not predictive accuracy
alone. A model that predicts perfectly via opaque embeddings is a
weaker result than a model that predicts somewhat worse but emits a
human-readable rule equivalent to the [277] gcd-signature theorem.

## Goals

- **G1.** Train an end-to-end model that predicts `(failure_mode,
  period_class)` for `(b, e, m)` triples.
- **G2.** Extract symbolic rules from the trained model. Score by
  structural overlap with [277] / [278] / canonical orbit_family.
- **G3.** If extraction produces a rule that does **not** match a
  certified rule, treat it as a **theorem candidate** and run an
  independent verification (sweep test, structural derivation, or
  hostile auditor pass) before reporting. Do **not** auto-promote to
  cert family.

## Non-goals

- **Not** a new cert family. v3 produces a synthesis report and
  optionally a `findings.md`, not a `qa_*_cert_v1/` directory.
- **Not** a wider orbit-period sweep. The training set comes from the
  existing certified moduli plus a controlled extension; no
  exploratory enumeration up to `m = 1000`.
- **Not** another GCN scaling study. v2 / [276] already showed the
  graph helps; v3 uses the GCN as a fixed component, not a free
  variable.
- **Not** a paper draft. Synthesis is internal; whether to publish is
  a separate decision.

## Tasks

### T1 — Shortcut failure-mode prediction

**Three-class classification**: for each `(b, e, m)` triple, predict
which failure mode (if any) the divisor shortcut exhibits at that
state:

```text
class 0  correct                    canonical(b,e,m) == shortcut(b,e,m)
class 1  undercount_missed_sat      canonical = "satellite", shortcut ≠ "satellite"
class 2  overclaim_false_sat        shortcut = "satellite", canonical ≠ "satellite"
```

Ground truth from `qa_orbit_rules.orbit_family` and
`qa_orbit_rules.orbit_family_divisor_shortcut`. Highly imbalanced (most
states are class 0). Use balanced accuracy + macro F1 + per-class recall.

The cert claims dictate the expected structure of the decision boundary:

| Region | Expected class | Source |
|---|---|---|
| `m = 15k`, missed pairs partition by gcd-signature | class 1 | [277] |
| `3 ∤ m, m ≥ 7, m ≠ 8`, 3×3 grid | class 2 | [278] |
| Singularity `(m, m)` | class 0 | trivial |
| Otherwise | class 0 | by construction |

A model that learns clean class boundaries here is approximating the
[277] / [278] rules. Extracting those boundaries is T3.

### T2 — Orbit-period prediction

**Multi-class classification with structured output.** Two formulations
to compare:

**T2a — exact period.** Predict `orbit_period(b, e, m) ∈ {1, 2, 3, ...}`
as integer regression or large-cardinality classification. Hard
(unbounded), but the most informative target.

**T2b — bucketed period class.** Predict
`period_class ∈ {1, 4, 8, π(m), other}`. Five buckets capture the
qualitative orbit structure surfaced by the spectrum sweep:

- `1` — singularity
- `4` — appears iff `5 | m` (parked observation O3)
- `8` — satellite class ([277] / [278] domain)
- `π(m)` — Pisano-period maximum (or `2 · π(m)` for m=75)
- `other` — intermediate periods (3, 6, 12, 16, 20, 24, 40, 60, 100,
  120, 200 in the existing sweep)

T2b is more interpretable. T2a tests the hard case. Both should be run.

### T3 — Symbolic rule extraction

**Take the trained T1/T2 models and read them.** Methods to compare:

1. **Decision-tree distillation.** Train a shallow CART/decision tree
   to mimic the model's predictions. Extract the path conditions.
   Score: do path conditions look like `(m // 3) | b ∧ (m // 3) | e`
   or `m mod 5 == 0 ∧ ...` ?

2. **Rule mining (RIPPER, BRL, RuleFit).** Mine if-then rules from
   the model's predictions over a held-out test set. Score: does the
   rule set contain the certified rules as approximate matches?

3. **Symbolic regression (PySR / similar).** For T2a, fit a closed-form
   `period(b, e, m)` over the integer features
   `(b, e, d, a, C, F, G, b mod m//3, e mod m//3, m, factorization
   features)`. The space of expressions is large; restrict to GP-style
   integer arithmetic plus mod and gcd.

4. **Saliency / attribution.** For the GCN, compute integrated-gradient
   or GNNExplainer-style edge-attribution. Identify which generator
   edges (`σ`, `μ`, `λ_2`, `ν`) the model attends to. Hypothesis: edges
   labeled `λ_2` and `ν` carry most of the failure-mode signal at
   `5 | m` (since they implement the multiplicative structure that
   produces the missed satellites).

**Rediscovery score** (per extracted rule R):

```text
rediscover(R, cert) = | states where R(state) == cert(state) | / | states |
                     for state in held-out test set
```

A rule with rediscover ≥ 0.95 against [277] or [278] is a "match". A
rule with rediscover ∈ [0.7, 0.95] is a "near-match" worth investigating
for a refined claim. Below 0.7 is not a rediscovery.

## Baselines

| Baseline | Features | Notes |
|---|---|---|
| **raw_mlp** | `(b, e, m)` | Sanity-check lower bound; v1 showed this collapses to majority class |
| **qa_full_logreg** | `(b, e, d, a, C, F, G, b mod m//3, e mod m//3)` | v1 best feature-only baseline |
| **qa_gcn** | qa_full + symmetric-normalized adjacency from σ/μ/λ_2/ν | v2 / [276] result; main candidate for T3 extraction |
| **rule_miner** | qa_full + factorization features | RIPPER/BRL applied directly without a deep model — establishes whether the rules can be mined without ML at all |

The rule_miner baseline is the most interesting: if a non-ML rule
miner already extracts [277] / [278] from labeled data, the deep model
is doing redundant work. If only the deep model recovers them, the
graph topology / non-linear features are doing necessary work.

## Dataset

### Training moduli

```text
M_train = {6, 9, 10, 11, 12, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27,
           28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 44, 45, 48, 50,
           54, 55, 60, 63, 65, 70}
```

35 moduli covering:
- `3 | m` (with and without `5 | m`)
- `3 ∤ m` (with and without `5 | m`)
- prime moduli, prime-power moduli, coprime composites
- `m ≤ 70` to keep training-set size manageable (`Σ m^2 ≈ 60,000`)

### Held-out test moduli

```text
M_test = {7, 8, 13, 14, 17, 19, 23, 29, 31, 37, 41, 42, 75, 90, 100, 120}
```

16 moduli, including:
- `m = 8`: the [278] boundary exception (4×4 grid → 15 overclaims)
- `m = 75`: the [279] parked anomaly (max-period doubling)
- larger composites (90, 100, 120) to stress-test extrapolation
- prime moduli (7, 13, 17, 19, 23, 29, 31, 37, 41) to test no-3-divisor
  behavior

The `m=8` and `m=75` test points are the **honest tests of
rediscovery**: a model that learned [278] should classify `m=8` as a
boundary exception (overclaim ≠ 9); a model that learned the spectrum
should flag `m=75` as anomalous.

### Per-state features

```text
integer:
  b, e
  d = b + e
  a = b + 2e
  C = 2 d e
  F = a b
  G = d² + e²
  φ_b = b mod (m // 3)        (m≥3)
  φ_e = e mod (m // 3)
  ψ_b = b mod 5
  ψ_e = e mod 5
  gcd(b, m)
  gcd(e, m)
  is_singularity              (b == m AND e == m)

modulus:
  m
  prime factorization vector  (counts of 2, 3, 5, 7, 11)
  m_div_3 = m // 3
  m_mod_3 = m % 3
  m_mod_5 = m % 5
  pisano_pi_m                  (lookup or computed)

graph:
  reachability adjacency       (256² normalized for largest m=120 single-graph;
                                use mini-batched subgraphs for training across m)
  σ-orbit length up to truncation (e.g. 16 steps)
  distance-to-singularity     (geodesic in σ ∪ μ ∪ λ_2 ∪ ν graph)
```

Graph features are computed per-modulus (one graph per `m`); training
over multiple `m` is graph-batched.

## Proposed file layout

```text
experiments/qa_ml/
  04_orbit_structure_discovery.py        # T1+T2 training + evaluation
  05_orbit_rule_extraction.py            # T3 extraction methods
  benchmark_protocol_v3.json             # QA_BENCHMARK_PROTOCOL.v1
  results_v3_structure_discovery.json    # T1+T2 metrics
  results_v3_extracted_rules.json        # T3 mined rules + rediscovery scores
  results_ledger_v3.jsonl                # qa_reproducibility log

tools/qa_ml/
  qa_features_v3.py        # extends qa_features with ψ_b, ψ_e, gcd*, factorization
  qa_graph_v3.py           # extends qa_graph with σ-orbit-length, dist-to-sing
  qa_rules.py              # RIPPER/BRL/decision-tree extraction wrappers

# No new cert family directory. No qa_alphageometry_ptolemy/ change.

docs/specs/
  QA_ML_V3_STRUCTURE_DISCOVERY_PLAN.md    # this file
  QA_ML_V3_FINDINGS.md                    # produced after experiment runs
```

`benchmark_protocol_v3.json` follows `QA_BENCHMARK_PROTOCOL.v1` (cert
[224]) so the existing benchmark linter catches missing fields. The
SOTA baseline is `qa_full_logreg` (carry-over from v1+v2). The ablation
is `swap graph adjacency to identity` (carry-over from v2).

## Evaluation protocol

| Metric | Computed on | Reported per |
|---|---|---|
| T1 macro F1 | M_test | model |
| T1 per-class recall | M_test, per failure mode | model × class |
| T2a MAE on `period(b, e, m)` | M_test (holds out m=75) | model |
| T2b macro F1 (5-class) | M_test | model |
| T3 rediscovery vs [277] | M_test ∩ `{m = 15k}` | model × extraction-method |
| T3 rediscovery vs [278] | M_test ∩ `{3 ∤ m, m ≠ 8}` | model × extraction-method |
| T3 m=8 boundary handling | single fixture | model × extraction-method |
| T3 m=75 anomaly handling | single fixture | model × extraction-method |
| Confidence under shuffle | shuffle generator labels in adjacency | per model — sanity check |

The **shuffle-generator-labels** test is the v3 analog of the v2
"shuffle adjacency" ablation: replace generator names with random
labels (so σ, μ, λ_2, ν edges become indistinguishable). A model that
extracts a true rule should drop in performance under this shuffle; a
model that learned a spurious correlation should not.

## Decision criteria for "v3 succeeded"

| Outcome | Criterion |
|---|---|
| **Strong success** | A T3 extraction method produces a rule with rediscover ≥ 0.95 against [277] AND a separate rule with rediscover ≥ 0.95 against [278], both on M_test. |
| **Partial success** | One certified rule rediscovered cleanly; the other near-rediscovered (≥ 0.70) but with structural differences worth investigating. |
| **Discovery candidate** | An extracted rule does NOT match either cert but rediscovers a non-trivial pattern (≥ 0.70 on a held-out subset) that is not yet certified. Trigger an independent verification before reporting as a finding — do NOT promote to cert directly. |
| **Null result** | Extraction methods produce no rule with rediscover ≥ 0.70 on either cert. The graph-learning advantage from [276] is not symbolically extractable; report honestly and move on. |

## Out-of-scope sub-questions (for future work)

- Why does `m=75` exhibit period-doubling? Structural derivation
  involving the A1 correction interaction with Fibonacci-mod-5²
  Pisano structure. Parked in `QA_ORBIT_PERIOD_SPECTRUM_CERT_DRAFT.md`.
- Does the graph-topology advantage extend to non-orbit prediction
  tasks? (e.g. predicting `qa_step` reachability between two states).
  Worth a separate v4 design once v3 settles.
- Can the rule-extraction method be turned into an **automated cert
  candidate generator** with a hostile auditor in the loop? Powerful
  but risks restarting the cert-cascade pattern; defer.

## Theorem NT compliance

All features are integer-derived; the GCN itself is observer-side float
arithmetic. Symbolic extraction operates over the model's float
predictions but the extracted rules themselves are integer-arithmetic
expressions. No float-to-integer cast in the QA layer. Boundary
crossing: integer fixture / dataset → torch tensor (input) → float
predictions / rules (output) → integer arithmetic in extracted rules.

## Open scope question (for review)

The current plan trains **one model per task** (separate models for T1
and T2). Two alternatives are worth weighing before code:

1. **Multi-task model**: shared GCN backbone, two heads. Lets the
   model share representations between failure-mode and period
   prediction. Risk: harder to extract rules from a multi-head model.

2. **Curriculum**: train T2 first (period prediction is the more
   primitive structural target), then freeze the backbone and train
   T1 on top. Risk: makes the experiment longer and the rule
   extraction harder to reason about.

Recommendation for v3.0: **separate models**. Multi-task and curriculum
are v3.1 / v3.2 if the separate-model run produces interesting findings
worth refining.

## Decision needed

This plan describes work that takes ~1-2 days of code + run time
(training across 35 moduli, extracting rules from ~4 models) plus a
synthesis pass. Before any code, two questions:

1. **Is the dataset split** (M_train = 35 moduli, M_test = 16 including
   m=8 and m=75) the right granularity? Smaller for faster iteration,
   larger for more structure?
2. **Which extraction method is the priority** — decision-tree
   distillation (interpretable, classical), rule mining (closer to
   [277] / [278] form), or symbolic regression (most powerful but
   noisiest)? Doing all four sequentially is fine if v3.0 is meant to
   be exploratory.

## References

- `docs/specs/QA_ML_ORBIT_DISCOVERY_SYNTHESIS.md` — synthesis closing
  the cert chain and motivating this plan.
- `tools/qa_ml/` — feature extractor, dataset, generators, graph
  builder (v1 + v2 modules to extend, not replace).
- Cert [276] (`qa_ml_orbit_topology_cert_v1`) — the result this v3
  plan refines.
- Cert [277] (`qa_orbit_pisano_5_factor_boundary_cert_v1`) — under-count
  rediscovery target.
- Cert [278] (`qa_orbit_no_3_divisor_overclaim_cert_v1`) — over-claim
  rediscovery target.
- Wall, D. D. (1960). *Fibonacci series modulo m*. Amer. Math. Monthly
  67(6), 525-532. DOI: 10.1080/00029890.1960.11989541.
- Kipf & Welling (2017) — arxiv:1609.02907.
- (Cohen-Welling 2016, ICML) — for graph-equivariant ML context (cert
  [247] family).
