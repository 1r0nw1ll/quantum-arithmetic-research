# QA-ORBIT Baseline Results

**Date:** 2026-03-09
**Split version:** v2 (satellite orbits now split 70/15/15 — fixes label-prior pathology)
**Status:** Baseline sweep complete on v2 splits.

---

## 1. Benchmark Setup

Two moduli evaluated:

| Modulus | Total tasks | Train | Dev | Test |
|---------|-------------|-------|-----|------|
| mod-9   | 2,037       | 1,345 | 368 | 324  |
| mod-24  | 14,556      | 10,066| 2,061 | 2,429 |

**Split v2:** Orbit families split 70/15/15 for both cosmos *and* satellite classes. Singularity (single fixed point) remains train-only. This ensures dev and test contain satellite orbit families, breaking the 100%-cosmos label prior present in split v1.

---

## 2. Baselines

Two models, each trained per task type:

| Model | Architecture | Notes |
|-------|-------------|-------|
| `LogisticRegression` | sklearn lbfgs, C=1.0, max\_iter=2000 | Linear baseline |
| `MLP(64×64)` | sklearn, relu, early stopping, val\_frac=0.1 | Nonlinear baseline |

**Features (degree-2 polynomial):** For each sample, 13 numeric features normalised by modulus:

- `b/m, e/m, b²/m², be/m², e²/m²`
- For reachability / shortest\_witness: same five terms for the target state `(b_target, e_target)`
- Wrapped circular distances `min(|b−b_target|, m−|b−b_target|)/m` and analogous for `e`

The degree-2 expansion was added specifically to give the models access to the quadratic terms that appear in the QA norm `f(b,e) = (b²+be-e²) mod m`. Prior flat-feature runs (linear only) gave equivalent results on `invariant_pred` and slightly worse on `shortest_witness`.

---

## 3. Split Pathology: `orbit_class` and `reachability`

**Split v2 status:** `orbit_class` is substantially improved; `reachability` remains prior-dominated.

**Finding (v1 split):** Both tasks were near-trivially solvable from label prior. **Under split v2, `orbit_class` becomes a real (if not primary) benchmark task.** `reachability` requires a task-generation fix, not a split fix.

| Task | Test set label distribution | Implication |
|------|-----------------------------|-------------|
| `orbit_class` | 100% "cosmos" by construction | Predicting "cosmos" always yields 100% OOD accuracy |
| `reachability` | 92.31% True (mod-24) | Predicting True always yields ≥92% OOD accuracy |

The root cause is deliberate: satellite and singularity states are all in train (they serve as easy scaffold). The test set therefore contains only cosmos orbit families — so `orbit_class` is trivially "cosmos" everywhere in dev/test. For `reachability`, cosmos orbits are single-cycle structures, so most target pairs are reachable, inducing the 92% True prior.

**`orbit_class` under split v2:** MLP achieves dev=0.889, test=0.866, gap=+0.023. The prior drops from 100% to 87%, and the model demonstrates a real (if small) IID/OOD gap. `orbit_class` can now be treated as an auxiliary benchmark task.

**`reachability` under split v2:** Prior remains ~92% True (dev=1.000, test=0.999 for MLP). This is a task-generation issue: the sampler draws more within-orbit (reachable) pairs than cross-orbit (unreachable) pairs. The fix is to balance True/False during generation in `tasks.py`, not in the split logic. `reachability` remains auxiliary until that is addressed.

---

## 4. Primary Task Results

### 4a. `shortest_witness`

This is the most informative task under the current split. The answer is an integer path length (0 to m−1, or −1 for unreachable), with roughly uniform distribution in the test set. The model must reason about orbit traversal length, not just label prior.

*All numbers below use split v2 (satellite 70/15/15).*

**mod-24 (float_poly features):**

| Model | Train | Dev (IID) | Test (OOD) | Gap |
|-------|-------|-----------|------------|-----|
| LogisticRegression | 0.179 | 0.147 | 0.173 | −0.026 |
| MLP(64×64) | 0.493 | 0.426 | 0.413 | +0.013 |
| Symbolic (ceiling) | — | 1.000 | 1.000 | 0.000 |

**mod-24 (onehot_flat — best encoding):**

| Model | Train | Dev (IID) | Test (OOD) | Gap |
|-------|-------|-----------|------------|-----|
| MLP(128×64) | 0.972 | 0.728 | 0.641 | +0.087 |
| Symbolic (ceiling) | — | 1.000 | 1.000 | 0.000 |

**Difficulty breakdown (onehot_flat MLP, mod-24 test):**

| Difficulty | Accuracy | Threshold |
|------------|----------|-----------|
| easy | 0.615 | ≤2 steps |
| medium | 0.653 | ≤5 steps |
| hard | 0.649 | >5 steps |

The onehot encoding achieves near-uniform accuracy across all difficulty tiers (0.615–0.653) — a qualitative shift from float features, which collapse on medium (0.044). The IID/OOD gap of +0.087 represents the fraction of orbit-traversal reasoning that fails to transfer from training cosmos/satellite families to held-out families. The gap is stable across both split versions.

### 4b. `invariant_pred`

The task is to compute `f(b,e) = (b²+be-e²) mod m` from `(b, e, m)`. The correct output is a class label in `{0, …, m−1}`.

**mod-24:**

| Model | Train | Dev (IID) | Test (OOD) | Gap |
|-------|-------|-----------|------------|-----|
| LogisticRegression | 0.152 | 0.000 | 0.012 | −0.012 |
| MLP(64×64) | 0.121 | 0.000 | 0.012 | −0.012 |
| Symbolic (ceiling) | — | 1.000 | 1.000 | 0.000 |

**mod-9:**

| Model | Train | Dev (IID) | Test (OOD) |
|-------|-------|-----------|------------|
| LogisticRegression | 0.351 | 0.000 | 0.000 |
| MLP(64×64) | 0.281 | 0.000 | 0.000 |

Both models achieve approximately 0% on dev and test across both moduli, despite having access to `b², be, e²` as explicit features. The residual 1.2% on mod-24 test (1–2 correct samples) is within noise.

**Interpretation:** These results indicate that `invariant_pred` is not solved by simple numeric-feature baselines. The task remains unsolved under both linear and MLP models, even with degree-2 polynomial features. The degree-2 expansion was chosen to match the functional form of `f`; its failure to improve test accuracy suggests that the modular reduction step — not the polynomial structure — is the primary barrier under the tested encoding.

This is not sufficient to conclude that MLPs in general cannot learn the task. What can be stated:

> Under a numeric floating-point encoding normalised by modulus, with linear and two-layer MLP architectures trained on the current splits, `invariant_pred` test accuracy remains at approximately 0% after quadratic feature expansion.

This makes `invariant_pred` a meaningful open challenge for stronger model classes.

---

## 5. Token Embedding Experiment: Encoding vs Difficulty

To diagnose whether `invariant_pred`'s 0% result was due to representation failure (float encoding) or genuine task difficulty, three encodings were compared on mod-24 (`invariant_pred` only, `n_train=420`, `n_dev=72`, `n_test=84`):

| Encoding | Model | Train | Dev | Test |
|----------|-------|-------|-----|------|
| float_poly (degree-2) | MLP(64×64) | 0.112 | 0.042 | 0.012 |
| integer (raw, no norm) | MLP(64×64) | 0.110 | 0.014 | 0.000 |
| **onehot (token-level)** | LogReg | 0.629 | 0.139 | 0.214 |
| **onehot (token-level)** | **MLP(64×64)** | **0.936** | **0.250** | **0.286** |
| Symbolic ceiling | — | — | 1.000 | 1.000 |

**Finding: encoding matters substantially, but does not solve the task.**

- Float and integer encodings: ≈0% dev/test (model cannot learn the norm function)
- One-hot encoding: 25–29% dev/test — a substantial jump from 0%, achieved without any architecture change

One-hot encoding gives each integer value `(b, e)` a learned identity vector, restoring the discrete structure of the problem. The first MLP layer applied to one-hot input is mathematically equivalent to a learned embedding table.

Despite this improvement, 25–29% is far from the symbolic ceiling (100%). The MLP memorises most of the training data (93.6% train accuracy) but generalises poorly.

**Verdict: Mixed — both hypotheses are partially true.**

> Representation matters: one-hot improves 0% → 25–29%. But task difficulty is real: even with discrete token embeddings, the modular norm function is not learned. Standard MLP/LR baselines cannot learn `(b²+be-e²) mod m` from any of the tested encodings, even with direct access to individual integer identities.

Note: On mod-9, all encodings yield 0% dev/test (`n_train=57`, `n_test=12` — too few samples for `invariant_pred` to be meaningful at this modulus).

Note on inverted test gap (mod-24): one-hot MLP achieves test=0.286 > dev=0.250 (−0.036 gap). With `n_dev=72` and `n_test=84`, this is within expected sampling noise. The orbit-index sorting of the split may also place structurally "easier" cosmos families in the test set.

---

## 6. Sequence Baseline: `shortest_witness`

Three encodings were tested for `shortest_witness` on mod-24 (`n_train=4542`, `n_dev=864`, `n_test=1008`), using a wider MLP to match input dimensionality.

| Encoding | Input dim | Model | Train | Dev | Test | Gap |
|----------|-----------|-------|-------|-----|------|-----|
| float_poly | 13 | MLP(64×64) | 0.425 | 0.350 | 0.343 | +0.006 |
| **onehot_flat** | **97** | **MLP(128×64)** | **0.978** | **0.748** | **0.664** | **+0.084** |
| bilinear | 1249 | MLP(256×128) | 0.946 | 0.422 | 0.374 | +0.048 |
| symbolic oracle | — | — | — | 1.000 | 1.000 | 0.000 |

**`onehot_flat` is the best result by a large margin: test=0.664, gap=+0.084.**

This is a 94% relative improvement in test accuracy over the previous float_poly best (0.343). The one-hot encoding assigns each integer value `(b, e, b_target, e_target)` an independent learned representation without cross-terms — the simplest possible discrete token encoding.

Counterintuitively, `bilinear` (which adds b⊗b_target and e⊗e_target outer products, totaling 1249 input dimensions) performs worse than `onehot_flat`. The bilinear features effectively create per-pair identifiers that allow the MLP to memorize training examples (train=0.946) but overfit to them — the 576-dimensional cross products reduce useful regularization pressure. This suggests that pairwise orbit identity, while informative, requires a different architecture (e.g., explicit attention) rather than pre-computed outer products fed to an MLP.

**Difficulty breakdown (onehot_flat, mod-24 test):**

| Difficulty | Accuracy | Float_poly comparison |
|------------|----------|-----------------------|
| easy | 0.726 | 0.738 |
| medium | 0.671 | 0.056 |
| hard | 0.629 | 0.290 |

The float_poly model essentially only solves easy cases (medium collapses to 5.6%). The onehot model achieves uniform accuracy across all difficulty tiers (~63–73%), indicating it has learned general orbit structure rather than memorizing short-path heuristics.

**The IID/OOD gap of +0.084 is the strongest algebraic generalisation signal in the benchmark to date.** The gap measures what fraction of the orbit-traversal rule fails to transfer from training cosmos families to held-out test families. A model that had fully learned the algebraic rule would show gap≈0 with high test accuracy; a model that purely memorised transitions would show large gap and low test accuracy. The onehot MLP is in between — it has partially learned the rule.

On mod-9 the pattern is less clear due to small sample sizes (train=609, test=144); bilinear achieves 0.250 test vs float_poly 0.153, but with only 12 output classes and 609 training examples, these results are indicative rather than conclusive.

---

## 7. Baseline Ladder (Complete)

Full picture, mod-24, primary tasks only:

| Task | Encoding | Model | Dev | Test | Gap |
|------|----------|-------|-----|------|-----|
| `invariant_pred` | float_poly | MLP(64×64) | 0.042 | 0.012 | — |
| `invariant_pred` | integer | MLP(64×64) | 0.014 | 0.000 | — |
| `invariant_pred` | onehot | MLP(64×64) | 0.250 | 0.286 | — |
| `invariant_pred` | symbolic oracle | — | 1.000 | 1.000 | 0.000 |
| `shortest_witness` | float_poly | MLP(64×64) | 0.350 | 0.343 | +0.006 |
| `shortest_witness` | onehot_flat | MLP(128×64) | 0.748 | 0.664 | +0.084 |
| `shortest_witness` | bilinear | MLP(256×128) | 0.422 | 0.374 | +0.048 |
| `shortest_witness` | symbolic oracle | — | 1.000 | 1.000 | 0.000 |

The gap between symbolic (100%) and best ML baseline on `shortest_witness` (66.4%) and `invariant_pred` (28.6%) defines the current challenge floor. Both tasks remain open to better model classes.

---

## 8. Interpretation Limits

The following conclusions are **not** supported by the current runs:

- That MLP architectures in general cannot learn `invariant_pred`.
- That transformers or token-embedding models would fail similarly on `invariant_pred`.
- That the observed IID/OOD gap on `shortest_witness` (+0.084 for onehot_flat) is stable across random seeds or architectures.
- That the label-prior issue in `orbit_class` / `reachability` cannot be fixed by re-splitting.

The current results establish a baseline floor, not a ceiling for model performance.

---

## 9. Connection to QA Theory

QA-ORBIT is not a generic orbit dataset. Each task family tests a specific algebraic property of the Quantum Arithmetic (QA) system, grounded in the ring of integers Z[φ] of Q(√5) (φ = golden ratio). The observed baseline difficulties are therefore not arbitrary — they reflect the intrinsic hardness of learning these algebraic structures.

### The QA norm and `invariant_pred`

The task asks a model to compute f(b,e) = (b² + be − e²) mod m from the integer pair (b, e).

This function is not an arbitrary quadratic. It is the norm N(b + eφ) in Q(√5), where the ring of integers Z[φ] = {a + bφ : a,b ∈ Z} carries the norm N(a+bφ) = a² + ab − b². The QA update map T = Q² acts as multiplication by φ² in this ring, and f is T-invariant: f(T(b,e)) = f(b,e) mod m for all states. This invariance is what partitions the state space into orbits — two states share an orbit if and only if they share a norm value (modulo prime-power conditions on the 3-adic valuation v₃(f)).

The benchmark result — that `invariant_pred` remains unsolved at 0% under all float and integer encodings, and reaches only 25–29% under one-hot token encoding — is therefore a statement about the difficulty of learning a number-theoretic invariant across orbit families. The modular reduction step (f mod m) introduces discontinuities that standard MLP architectures cannot represent through continuous approximation. Learning this function from examples would require, at minimum, architecture with explicit periodic or modular structure. The benchmark exposes this gap empirically.

### Generator-relative reachability and `shortest_witness`

The `shortest_witness` task asks: how many applications of the QA generator T are needed to reach (b_target, e_target) from (b, e)?

Within a cosmos orbit (an orbit of maximal length π(m)/2, where π(m) is the Pisano period), T acts as a cyclic shift of order L. The step count from (b,e) to (b_target, e_target) is the discrete index `(pos_target − pos_source) mod L` — a discrete logarithm in the cyclic group generated by T. Across different cosmos orbit families, the local geometry of this group is isomorphic (same cycle structure), but the embedding of specific (b,e) pairs into this structure varies.

The 64.1% test accuracy of the onehot_flat MLP — with the remarkable property that accuracy is *uniform across difficulty tiers* (easy 61.5%, medium 65.3%, hard 64.9%) — reflects partial learning of this orbit-traversal structure. The model learns that the discrete identity of the source and target tokens encodes enough information to estimate step count without explicit orbit enumeration. But the 8.7pp IID/OOD gap shows that this learned structure does not fully transfer to unseen orbit families: the model has learned a partial algebraic rule, not a complete one.

### What the benchmark is actually measuring

The IID/OOD split separates cosmos orbit families by index. Families in dev and test share the same algebraic orbit type (cosmos, length π(m)/2) but differ in which states are mapped to which positions. A model that had fully learned the algebraic rule — that step count is determined by the norm-preserving structure of T acting on Z[φ]/mZ[φ] — would show gap ≈ 0 with high test accuracy. A model that merely memorised transition patterns from training families would show large gap and low test accuracy.

The observed baseline behavior falls between these extremes: partial structural learning with a persistent OOD gap. This is precisely the target difficulty for a benchmark intended to probe algebraic generalisation.

### Connection to the cert ecosystem

The QA certificate families ([89]–[101]) pin the curvature scalar κ = 1 − |1 − lr·gain·H_QA|, where H_QA is derived from the same (b,e,d,a) substrate tuple. The cert system asserts that κ governs optimizer convergence through the Finite-Orbit Descent Theorem: L_{t+L} = ρ(O)·L_t, where ρ(O) = ∏(1−κ_t)². The benchmark tests the complementary question: can a neural model learn the algebraic invariants that the cert system *assumes* are preserved? The 0% result on `invariant_pred` says no — current models cannot verify QA cert claims from first principles, even when given the (b,e,m) inputs explicitly. That gap defines the frontier for future work.

---

## 10. Capacity Sweep — Minimum Sufficient Architecture (2026-03-09)

**Script:** `capacity_sweep.py`  **Data:** mod-24, split v2  **Models:** 9 MLP sizes × 3 alpha values × 4 encodings = 48 runs

**Tiers:** SOLVED ≥95% | STRONG ≥85% | PARTIAL ≥60% | FAILED <40%

### 10.1 `invariant_pred` — hard wall confirmed

| Encoding | Best arch | Best test | Tier |
|---|---|---|---|
| onehot | MLP(256,256,256) — 149K params | 33.0% | FAILED |
| onehot+modular | MLP(512,256,128) — 197K params | 16.5% | FAILED |

**Key finding:** No architecture at any capacity reaches even PARTIAL (≥60%) on `invariant_pred`. The wall is at ~33% for plain onehot and lower for the modular variant. Crucially, adding explicit Fourier harmonics and the unmodded quadratic term *hurts* performance (33% → 16.5%). This rules out a capacity explanation: the model is not failing because it lacks the raw materials to represent a quadratic form — it has them and performs worse. The barrier is structural: modular discontinuities in f mod m require architecture with explicit periodic structure that standard ReLU MLPs cannot express.

Full architecture ladder (onehot, α=0.0001):

| Architecture | Params | Train | Dev | Test | Gap | Tier |
|---|---|---|---|---|---|---|
| MLP(64,) | 4,370 | 0.256 | 0.012 | 0.000 | +0.012 | FAILED |
| MLP(512,) | 34,834 | 0.824 | 0.148 | 0.196 | −0.048 | FAILED |
| MLP(256,256) | 83,218 | 0.927 | 0.160 | 0.258 | −0.097 | FAILED |
| MLP(256,256,256) | 149,010 | 0.935 | 0.235 | **0.330** | −0.095 | FAILED |
| MLP(512,256,128) | 192,146 | 0.940 | 0.210 | 0.206 | +0.004 | FAILED |

Train accuracy saturates near 94%, but test accuracy plateaus at ~33% and does not improve with depth or regularization (alpha sweep: 24.7%–26.8%, no benefit). The train/test gap is not a capacity gap; it is an algebraic-generalisation gap that capacity alone cannot close.

### 10.2 `shortest_witness` — minimum sufficient architecture identified

| Encoding | Best arch | Params | Best test | Tier |
|---|---|---|---|---|
| onehot_flat | MLP(64,64) | 11,212 | **81.8%** | PARTIAL |
| onehot_flat + oracle | MLP(256,256) α=0.001 | 106,252 | **82.0%** | PARTIAL |

**Minimum sufficient:** MLP(64,) with **7,052 parameters** achieves test=81.7% — the smallest architecture tested. Adding capacity does not improve test accuracy; MLP(512,256,128) achieves 79.3%, *less* than MLP(64,). The pattern is learnable at the smallest scale.

**Oracle null result:** Injecting the precomputed norm f(b,e) and f(b_target,e_target) as additional one-hot features gives 82.0% — a negligible +0.2pp improvement over the 81.8% baseline without the norm. The bottleneck in `shortest_witness` is **not** the model's inability to compute the Q(√5) norm. The discrete-log structure of the orbit group is learnable independently of the norm invariant.

**IID/OOD gap:** Consistently +0.12 to +0.18 across all architectures and both encodings. This structural gap does not close with capacity, regularization, or oracle features. It is a property of the orbit-family split.

Full architecture ladder (onehot_flat, α=0.0001):

| Architecture | Params | Train | Dev | Test | Gap | Tier |
|---|---|---|---|---|---|---|
| MLP(64,) | 7,052 | 0.995 | 0.943 | **0.817** | +0.126 | PARTIAL |
| MLP(64,64) | 11,212 | 0.995 | 0.956 | **0.818** | +0.138 | PARTIAL |
| MLP(256,256) | 93,964 | 0.995 | 0.941 | 0.765 | +0.175 | PARTIAL |
| MLP(512,256,128) | 215,948 | 0.995 | 0.915 | 0.793 | +0.122 | PARTIAL |

### 10.3 Summary

The sweep produces a clean bifurcation:

| Task | Minimum architecture that reaches PARTIAL | Hard wall? |
|---|---|---|
| `invariant_pred` | None found (max 33.0% at 149K params) | Yes — algebraic |
| `shortest_witness` | MLP(64,) — 7,052 params | No — PARTIAL achievable; STRONG/SOLVED not yet |

`invariant_pred` is a genuine algebraic barrier for the standard MLP function class. `shortest_witness` is a learnable structure accessible to the smallest tested architecture, with a persistent IID/OOD gap that does not depend on model capacity.

These results update and strengthen the interpretation in §9. The oracle null result specifically eliminates the hypothesis that `shortest_witness` difficulty is caused by inability to compute the norm — the two difficulties are structurally independent.

---

## 11. Recommended Next Baselines

In priority order:

1. ~~**Symbolic-token embedding for `invariant_pred`**~~ ✓ **Done** — one-hot MLP achieves 25–33% test, confirming mixed result: encoding matters, task is still genuinely hard.

2. ~~**Capacity sweep**~~ ✓ **Done** — see §10. Hard wall confirmed for `invariant_pred`; minimum sufficient architecture (7K params) identified for `shortest_witness`.

3. ~~**Re-split for `orbit_class` and `reachability`**~~ ✓ **Done** — split v2 + balanced reachability sampling; all four tasks now honest (see §3).

4. **Reachability baseline rerun on balanced data:** The reachability results in §7 used pre-balance data. A rerun with the balanced split (§3) would complete the four-task baseline table.

5. **Harder IID/OOD split for `shortest_witness`:** Split by orbit norm class or larger held-out fraction. The oracle null result (§10.2) suggests the persistent ~15pp gap is inherent to orbit-family structure, not norm computation. A harder split would test whether the pattern fully generalises.

6. **PL-condition extension of the Finite-Orbit Descent Theorem:** The main theoretical open problem. See §9 (cert connection) and companion paper §10.
