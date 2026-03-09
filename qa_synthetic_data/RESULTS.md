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

## 9. Recommended Next Baselines

In priority order:

1. ~~**Symbolic-token embedding for `invariant_pred`**~~ ✓ **Done** — one-hot MLP achieves 25–29% test, confirming mixed result: encoding matters, task is still genuinely hard.

2. **Harder IID/OOD split for `shortest_witness`:** Rather than splitting by orbit index (adjacent families), split by orbit *norm class* or by a larger held-out fraction. This would widen the gap if the model is benefiting from structural similarity between adjacent cosmos families.

3. **Re-split for `orbit_class` and `reachability`:** Include a fraction of satellite/singularity states in dev/test to break the label-prior dominance. This requires modifying the split logic in `run_generator.py`.

4. **Sequence model for `shortest_witness`:** An MLP on flat features cannot represent the step-by-step orbit traversal structure. A recurrent or attention-based model over the orbit sequence may unlock significantly higher accuracy — and the resulting IID/OOD gap would be the most informative benchmark result to date.
