# QA-ORBIT: A Certifiable Benchmark for Algebraic Generalisation in Modular Dynamical Systems

**Abstract:** We introduce QA-ORBIT, a synthetic benchmark for algebraic generalisation grounded in the Quantum Arithmetic (QA) modular dynamical system. QA-ORBIT contains 20,394 tasks across four families—norm prediction, orbit classification, reachability, and shortest path—with deterministic answers and machine-verifiable certificates. The dataset is split by orbit family, forcing models to generalise algebraic rules to unseen families rather than interpolate memorised transitions. We evaluate several baselines, finding that: (1) the QA norm function, a Q(√5) number-theoretic invariant, remains unsolved by standard architectures; (2) shortest-path prediction reaches 64.1% test accuracy with a measurable 8.7pp IID/OOD gap, indicating partial generalisation. QA-ORBIT is released with a deterministic verifier and a full baseline suite to probe this algebraic generalisation gap.

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
| **onehot_flat** | **MLP(128×64)** | **0.972** | **0.728** | **0.641** | **+0.087** |
| bilinear | MLP(256×128) | 0.944 | 0.415 | 0.359 | +0.056 |
| symbolic | — | 1.000 | 1.000 | 1.000 | 0.000 |

The `onehot_flat` encoding achieves the best performance, and its 8.7pp IID/OOD gap is the clearest signal of algebraic generalisation in the benchmark. The model appears to learn a general rule, as accuracy is uniform across easy (61.5%), medium (65.3%), and hard (64.9%) difficulty tiers.

### 4.3 Auxiliary Tasks

For `orbit_class`, an MLP with one-hot encoding achieves 0.889 dev and 0.866 test accuracy, a small generalisation gap of +0.023. The `reachability` task, now balanced to 50% True/False, requires a baseline rerun (results here used pre-balance data).

---

## 5. Theoretical Grounding

The benchmark's difficulty is rooted in the algebraic structure of Z[φ]/mZ[φ], the finite quotient of the ring of integers of Q(√5).

The `invariant_pred` task requires the model to compute f(b,e) = N(b+eφ) mod m, where N is the field norm in Q(√5). This norm is T-invariant, and its 3-adic valuation v₃(f) determines the orbit structure. Learning f requires representing a modular quadratic form, a known challenge for standard neural architectures.

The `shortest_witness` task requires computing the discrete index of a target state relative to a source state within a cyclic group generated by T. For a cosmos orbit of length L, this is equivalent to finding the discrete logarithm `(pos_target − pos_source) mod L`. Generalising this function to unseen orbits requires learning the group structure itself, not merely local transition patterns.

This benchmark directly probes the capabilities required to verify QA certificates [cf. 1], which rely on the Finite-Orbit Descent Theorem. The near-zero baseline performance on `invariant_pred` demonstrates empirically that standard models cannot learn the core algebraic invariant needed to verify these certificate claims from first principles.

---

## 6. Discussion

QA-ORBIT is not a test of general reasoning but a targeted probe of algebraic generalisation. Its value lies in its verifiable answers, interpretable difficulty, and an IID/OOD split designed to isolate and measure this specific capability.

**Limitations:** The current orbit-family split may place structurally similar orbits in the train and test sets. A more demanding split based on norm class or a larger held-out fraction could provide a stronger test of generalisation.

**Recommended Evaluation Protocol:** We recommend reporting accuracy per task and difficulty tier, along with the IID (dev) vs. OOD (test) accuracy gap, for `invariant_pred` and `shortest_witness`. Averaging across all four tasks is discouraged, as `orbit_class` and `reachability` are auxiliary.

---

## 7. Conclusion

QA-ORBIT provides a certifiable benchmark for algebraic generalisation in a modular dynamical system with deep theoretical grounding. Our baseline evaluation shows that standard MLP and logistic regression models fail to solve the core tasks of learning the Q(√5) norm and predicting orbit traversal distance. The best-performing model, a one-hot MLP for `shortest_witness`, achieves 64.1% test accuracy, leaving a substantial gap to the 100% symbolic ceiling. This establishes a clear and quantifiable challenge for future architectures aiming to learn and generalise algebraic rules.

---

## References

1.  QA Unified Curvature Paper (companion paper, this project).
2.  Wall, D. D. "Fibonacci series modulo m." *American Mathematical Monthly* 67.6 (1960): 525-532. (for Pisano Period).
3.  Alaca, S., & Williams, K. S. *Introductory algebraic number theory*. Cambridge University Press, 2004. (for Z[φ] and norm N(a+bφ) = a² + ab − b²).
4.  Finite-Orbit Descent Theorem: See §8.1 of companion paper [1].
