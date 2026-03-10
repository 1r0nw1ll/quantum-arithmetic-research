# QA-ORBIT: A Certifiable Benchmark for Algebraic Generalisation in Modular Dynamical Systems

**Abstract:** We introduce QA-ORBIT, a synthetic benchmark for algebraic generalisation grounded in the Quantum Arithmetic (QA) modular dynamical system. QA-ORBIT contains 20,394 tasks across four families—norm prediction, orbit classification, reachability, and shortest path—with deterministic answers and machine-verifiable certificates. The dataset is split by orbit family, forcing models to generalise algebraic rules to unseen families rather than interpolate memorised transitions. We evaluate a full architecture ladder (9 MLP sizes, 3 regularisation levels, 4 encodings), finding a clean bifurcation: (1) the QA norm function, a Q(√5) number-theoretic invariant, remains unsolved at any tested capacity—a 149K-parameter MLP reaches only 33.0% test accuracy, confirming a genuine algebraic barrier; (2) shortest-path prediction is learnable by the smallest tested architecture (7K parameters, 81.8% test accuracy), with a persistent 13.8pp IID/OOD gap that does not close with additional capacity. An oracle ablation—injecting the precomputed norm as a feature—yields no meaningful lift on shortest-path prediction, showing the two tasks have structurally independent difficulty. QA-ORBIT is released with a deterministic verifier and full baseline suite.

---

## 1. Introduction

While many neural reasoning benchmarks test compositional complexity or linguistic ambiguity, their difficulty is often coupled to dataset scale and statistical artefacts. We introduce QA-ORBIT, a benchmark whose difficulty derives from the algebraic structure of a specific mathematical system, where every answer is formally verifiable.

The underlying system is the QA update map, T: (b,e) → (b+e, b+2e) mod m, which acts as multiplication by φ² in the ring of integers Z[φ] of Q(√5). The map T preserves the norm f(b,e) = b² + be − e², partitioning the state space into three orbit types: cosmos (maximal-length cycles), satellite (shorter periodic orbits), and singularity (a fixed point).

QA-ORBIT probes whether neural models can learn this underlying structure. It comprises four tasks: predicting the algebraic invariant (`invariant_pred`), classifying the orbit type (`orbit_class`), determining state reachability (`reachability`), and finding the shortest path (`shortest_witness`). The benchmark is designed to test algebraic generalisation: can a model learn a rule from examples on seen orbit families and apply it to unseen families from the same algebraic system?

---

## 2. Dataset Construction

The dataset is generated from the state space {0,...,m−1}² under the update rule T(b,e) = (b+e, b+2e) mod m. Orbits are computed by forward simulation. Each of the 20,394 tasks includes a `sha256` hash for tamper-evident identification. All tasks are validated by a deterministic verifier; no task with `verifier_outcome=false` is included.

**Task Families:**

| Task | Input | Output | Verifier |
| :--- | :--- | :--- | :--- |
| `invariant_pred` | (b, e, m) | f(b,e) mod m | Closed-form: (b²+be−e²) mod m |
| `orbit_class` | (b, e, m) | cosmos/satellite/singularity | Orbit length vs. Pisano period |
| `reachability` | (b, e, b*, e*, m) | bool | Forward orbit membership check |
| `shortest_witness` | (b, e, b*, e*, m) | step count | Breadth-First Search (BFS) |

**Split Design:** To test generalisation, we split by orbit family, not by random state. Satellite and cosmos orbit families are each split 70/15/15 by their canonical root (sorted for determinism). The singularity (fixed point) is included in the training set only. This design ensures that dev and test sets contain unseen orbit families of both major types. For `reachability`, unreachable targets are sampled from external orbits to create an exact 50/50 class balance in all splits.

**Statistics (mod-24):**

| Split | Tasks | `orbit_class` (Cosmos %) | `reachability` (True %) |
| :--- | :--- | :--- | :--- |
| train | 14,104 | — | 50.0% |
| dev | 2,889 | 87.1% | 50.0% |
| test | 3,401 | 86.6% | 50.0% |

---

## 3. Baselines

We evaluate three encoding strategies and two model classes from scikit-learn for each task. The symbolic ceiling is a deterministic verifier that achieves 100% accuracy by construction.

*   **Encodings:**
    *   `integer`: Raw integer values for (b, e, m, ...).
    *   `float_poly`: 13 features including normalised state variables (b/m, e/m), quadratic terms (b², be, e²), and wrapped distances for target states.
    *   `onehot`: Independent one-hot encodings for source and target state variables.
*   **Models:**
    *   `LogisticRegression`: lbfgs solver, C=1.0.
    *   `MLPClassifier`: 64×64 hidden layers with ReLU activation and early stopping. Wider architectures (128×64, 256×128) were used for higher-dimensional one-hot encodings.

---

## 4. Results

### 4.1 `invariant_pred`: Learning the QA Norm

| Encoding | Model | Train | Dev | Test |
| :--- | :--- | :--- | :--- | :--- |
| float_poly | MLP(64×64) | 0.163 | 0.025 | 0.000 |
| integer | MLP(64×64) | 0.110 | 0.014 | 0.000 |
| **onehot** | **MLP(64×64)** | **0.936** | **0.250** | **0.286** |
| symbolic | — | 1.000 | 1.000 | 1.000 |

The one-hot encoding provides a significant lift over continuous representations, but the large gap between train (93.6%) and test (28.6%) accuracy indicates memorisation rather than generalisation of the modular norm function.

### 4.2 `shortest_witness`: Generator-Relative Orbit Traversal

| Encoding | Model | Train | Dev | Test | IID−OOD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| float_poly | MLP(64×64) | 0.492 | 0.404 | 0.388 | +0.016 |
| bilinear | MLP(256×128) | 0.944 | 0.415 | 0.359 | +0.056 |
| **onehot_flat** | **MLP(64×64)** | **0.995** | **0.956** | **0.818** | **+0.138** |
| symbolic | — | 1.000 | 1.000 | 1.000 | 0.000 |

The `onehot_flat` encoding achieves the best performance. The 13.8pp IID/OOD gap is the primary algebraic generalisation signal in the benchmark. Accuracy is uniform across easy (61.5%), medium (65.3%), and hard (64.9%) difficulty tiers, indicating the model learns a general orbit-position rule rather than a difficulty-specific shortcut.

### 4.3 Auxiliary Tasks

For `orbit_class`, an MLP with one-hot encoding achieves 0.889 dev and 0.866 test accuracy, a small generalisation gap of +0.023.

For `reachability` (balanced 50/50 True/False, mod-24):

| Encoding | Model | Train | Dev | Test |
| :--- | :--- | :--- | :--- | :--- |
| float_poly | LR | 0.954 | 0.994 | **1.000** |
| float_poly | MLP(64×64) | 0.986 | 1.000 | **1.000** |
| onehot_flat | LR | 0.986 | 1.000 | **1.000** |
| symbolic | — | 1.000 | 1.000 | 1.000 |

Once balanced, `reachability` is solved (100% test) by every tested model, including logistic regression on polynomial features. The reason is structural: the float_poly features include `b², be, e²` — the components of the Q(√5) norm f(b,e) = b² + be − e². A linear model can compute `f(b,e) − f(b_target, e_target)` as a linear combination of these terms without the modular reduction step. Orbit membership reduces to norm-class matching, which is a polynomial comparison, not a modular prediction. This explains the sharp contrast with `invariant_pred`: predicting f(b,e) mod m in absolute terms requires modular arithmetic; comparing whether two states share the same norm class does not.

### 4.4 Capacity Sweep: Minimum Sufficient Architecture

To determine whether the primary task failures are capacity-limited or structurally limited, we run a systematic sweep: 9 MLP architectures (4K–216K parameters), 3 regularisation levels (α ∈ {0.0001, 0.001, 0.01}), and two encoding variants per task, for 48 total runs.

**`invariant_pred` — algebraic barrier:**

| Architecture | Params | Train | Dev | Test |
| :--- | ---: | :--- | :--- | :--- |
| MLP(64,) | 4,370 | 0.256 | 0.012 | 0.000 |
| MLP(512,) | 34,834 | 0.824 | 0.148 | 0.196 |
| MLP(256,256) | 83,218 | 0.927 | 0.160 | 0.258 |
| **MLP(256,256,256)** | **149,010** | **0.935** | **0.235** | **0.330** |
| MLP(512,256,128) | 192,146 | 0.940 | 0.210 | 0.206 |

No architecture reaches even 40% test accuracy. Adding explicit inductive bias—Fourier harmonics of (b, e) and the unmodded quadratic scalar—worsens performance (best test 16.5% vs. 33.0%), ruling out a representation deficit. Regularisation has no effect (α sweep: 24.7%–26.8%). The barrier is structural: modular discontinuities in f mod m are not expressible by standard ReLU MLPs at any tested capacity.

**`shortest_witness` — minimum sufficient architecture:**

The orbit-position pattern is learnable by the smallest tested model: MLP(64,) with **7,052 parameters** achieves 81.7% test accuracy. Larger models do not improve this (MLP(512,256,128) achieves 79.3%). The IID/OOD gap (+0.12 to +0.18) is stable across all architectures and does not close with capacity.

**Oracle ablation:** Injecting the precomputed norms f(b,e) and f(b\*,e\*) as additional features yields 82.0% test accuracy—a negligible +0.2pp improvement. The two primary tasks have **structurally independent** difficulty: inability to compute the Q(√5) norm is not what limits `shortest_witness` performance.

---

## 5. Theoretical Grounding

The benchmark's difficulty is rooted in the algebraic structure of Z[φ]/mZ[φ], the finite quotient of the ring of integers of Q(√5).

The `invariant_pred` task requires the model to compute f(b,e) = N(b+eφ) mod m, where N is the field norm in Q(√5). This norm is T-invariant, and its 3-adic valuation v₃(f) determines the orbit structure. Learning f requires representing a modular quadratic form, a known challenge for standard neural architectures.

The `shortest_witness` task requires computing the discrete index of a target state relative to a source state within a cyclic group generated by T. For a cosmos orbit of length L, this is equivalent to finding the discrete logarithm `(pos_target − pos_source) mod L`. Generalising this function to unseen orbits requires learning the group structure itself, not merely local transition patterns. The oracle ablation (§4.4) confirms that this difficulty is independent of norm computation: providing f(b,e) directly as a feature does not close the IID/OOD gap, indicating the bottleneck is discrete-log generalisation rather than norm evaluation.

The `reachability` result sharpens this picture. Two states share an orbit if and only if they share a norm class (same v₃(f) profile). The float_poly features—which include b², be, e²—allow a linear model to compute the norm difference `f(b,e) − f(b*,e*)` directly, without the modular reduction step. This is why `reachability` is solved by logistic regression (100% test) while `invariant_pred` is not: relative norm comparison is a polynomial operation; absolute norm prediction modulo m introduces periodic discontinuities that polynomial models cannot represent across orbit families. The benchmark thus contains tasks of three distinct difficulty types, ordered by how directly they require the modular structure of f.

This benchmark directly probes the capabilities required to verify QA certificates [cf. 1], which rely on the Finite-Orbit Descent Theorem. The near-zero baseline performance on `invariant_pred` demonstrates empirically that standard models cannot learn the core algebraic invariant needed to verify these certificate claims from first principles.

---

## 6. Discussion

QA-ORBIT is not a test of general reasoning but a targeted probe of algebraic generalisation. Its value lies in its verifiable answers, interpretable difficulty, and an IID/OOD split designed to isolate and measure this specific capability.

**Minimum sufficient architecture.** The capacity sweep (§4.4) identifies a clean bifurcation between the two primary tasks. `invariant_pred` has a hard algebraic barrier: no tested architecture (up to 149K parameters, with or without explicit modular features) reaches even 40% test accuracy. `shortest_witness` is learnable at the smallest scale: a 7K-parameter MLP achieves 81.7% test accuracy, with no benefit from additional capacity. This contrast confirms that the benchmark probes two structurally distinct challenges within the same mathematical system.

**Limitations:** The current orbit-family split may place structurally similar orbits in the train and test sets. A more demanding split based on norm class or a larger held-out fraction could widen the IID/OOD gap and provide a stronger test of generalisation.

**Recommended Evaluation Protocol:** We recommend reporting accuracy per task and difficulty tier, along with the IID (dev) vs. OOD (test) accuracy gap, for `invariant_pred` and `shortest_witness`. Averaging across all four tasks is discouraged, as `orbit_class` and `reachability` are auxiliary.

---

## 7. Conclusion

QA-ORBIT provides a certifiable benchmark for algebraic generalisation in a modular dynamical system with deep theoretical grounding. A capacity sweep across 48 MLP configurations produces a clean bifurcation: the Q(√5) norm invariant (`invariant_pred`) is not learned by any tested architecture up to 149K parameters, with explicit modular inductive bias providing no benefit — a structural rather than capacity failure. The orbit traversal task (`shortest_witness`) is learnable by a 7K-parameter MLP (81.8% test accuracy), with a persistent 13.8pp IID/OOD gap that does not close with scale. An oracle ablation confirms the two difficulties are independent: norm access does not improve orbit traversal. These results establish a quantified and interpretable challenge floor — closing the IID/OOD gap on `shortest_witness` and any progress on `invariant_pred` beyond 33.0% are the concrete targets for future architectures.

---

## References

1.  QA Unified Curvature Paper (companion paper, this project).
2.  Wall, D. D. "Fibonacci series modulo m." *American Mathematical Monthly* 67.6 (1960): 525-532. (for Pisano Period).
3.  Alaca, S., & Williams, K. S. *Introductory algebraic number theory*. Cambridge University Press, 2004. (for Z[φ] and norm N(a+bφ) = a² + ab − b²).
4.  Finite-Orbit Descent Theorem: See §8.1 of companion paper [1].
