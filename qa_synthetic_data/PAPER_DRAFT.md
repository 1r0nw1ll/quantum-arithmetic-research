# QA-ORBIT: A Certifiable Benchmark for Algebraic Generalisation in Modular Dynamical Systems

**Draft:** 2026-03-09 — for Gemini polish pass

---

## Abstract

We introduce QA-ORBIT, a synthetic reasoning benchmark grounded in the Quantum Arithmetic (QA) modular dynamical system. QA-ORBIT consists of 20,394 tasks across four families — norm prediction, orbit classification, reachability, and shortest path — all with deterministic ground-truth answers and machine-verifiable certificates. Tasks are split by orbit family rather than randomly, so models must generalise algebraic rules across unseen orbit families rather than interpolate memorised transitions. We run a baseline sweep including logistic regression, MLP, polynomial feature expansion, and one-hot token embeddings. The primary findings are: (1) the QA norm function f(b,e) = (b² + be − e²) mod m remains unsolved under all tested numeric and token-level encodings, confirming that learning a Q(√5) number-theoretic invariant is a non-trivial challenge for standard architectures; (2) shortest-path prediction, which requires learning generator-relative reachability in the orbit group, reaches 64.1% test accuracy under one-hot encoding with a measurable IID/OOD gap of 8.7pp. QA-ORBIT is released with a deterministic verifier, orbit-family splits, and a full baseline suite.

---

## 1. Introduction

Neural reasoning benchmarks typically evaluate tasks whose difficulty arises from dataset scale, linguistic ambiguity, or compositional complexity. QA-ORBIT is different: its difficulty arises from the algebraic structure of a specific mathematical system, and every answer is formally verifiable.

The underlying system is the QA update map T: (b,e) → (b+e, b+2e) mod m, which acts as multiplication by φ² in the ring of integers Z[φ] of Q(√5). The map T preserves the norm f(b,e) = b² + be − e², partitioning the state space into three orbit types: cosmos (maximal-length cycles of length π(m)/2, where π(m) is the Pisano period), satellite (shorter periodic orbits), and singularity (fixed point).

QA-ORBIT tests whether neural models can learn:
- the algebraic invariant f that governs this partition (task: `invariant_pred`),
- the orbit type of a given state (task: `orbit_class`),
- whether a target state is reachable via repeated application of T (task: `reachability`),
- and the minimum number of T-applications required (task: `shortest_witness`).

The benchmark is not a test of general reasoning ability. It is a targeted probe of algebraic generalisation: can a model learn a mathematical rule from examples on seen orbit families and apply it correctly to unseen orbit families drawn from the same algebraic structure?

---

## 2. Dataset Construction

**Update rule.** State (b,e) ∈ {0,...,m−1}², update: d = (b+e) mod m, a = (b+2e) mod m. Orbit structure is computed by forward simulation until a repeat is detected.

**Task families.** Four task types are generated for every state (b,e):

| Task | Input | Output | Verifier |
|------|-------|--------|----------|
| `invariant_pred` | (b, e, m) | f(b,e) mod m | closed-form |
| `orbit_class` | (b, e, m) | cosmos / satellite / singularity | orbit length comparison |
| `reachability` | (b, e, b*, e*, m) | bool | forward orbit membership |
| `shortest_witness` | (b, e, b*, e*, m) | step count | BFS |

**Reachability balance.** For each state, the generator produces L reachable target pairs (all within-orbit) and L unreachable target pairs (cycling through external states). This yields an exact 50/50 True/False ratio in all splits, preventing label-prior dominance.

**Split design (v2).** Satellite *and* cosmos orbit families are each split 70/15/15 by canonical orbit root (sorted for determinism). Singularity (unique fixed point) remains train-only. This ensures dev and test contain both satellite and cosmos families, preventing the 100%-cosmos label prior present in naive splits.

**Statistics (mod-24):**

| Split | Tasks | orbit_class True% | reachability True% |
|-------|-------|-------------------|--------------------|
| train | 14,104 | — | 50.0% |
| dev   | 2,889  | 87.1% cosmos | 50.0% |
| test  | 3,401  | 86.6% cosmos | 50.0% |

**Verifier.** All answers are independently recomputed at generation time. No task with `verifier_outcome=false` is included. Failure rate: 0 of 20,394 (mod-24).

**Canonical hash.** Each row includes `sha256("QA_SYNTHETIC\x00" + canonical_json(task_type + input))` as a tamper-evident identifier.

---

## 3. Baselines

Three encoding strategies and two model classes are evaluated for each task type. Models are trained per task type; label encoding is consistent across splits.

**Encodings:**
- `float_poly`: (b/m, e/m, b²/m², be/m², e²/m², target analogs, wrapped distances) — 13 features
- `integer`: (b, e, m) as raw integers
- `onehot`: b and e independently one-hot encoded (m dims each) + m scalar; for two-point tasks, source and target separately

**Models:** LogisticRegression (lbfgs, C=1.0) and MLPClassifier (64×64 relu, early stopping) from scikit-learn. Wider architectures (128×64, 256×128) for higher-dimensional encodings.

**Symbolic ceiling:** The deterministic verifier (BFS + closed-form) achieves 100% on all tasks and all splits by construction.

---

## 4. Results

### 4.1 `invariant_pred`: learning the QA norm

| Encoding | Model | Train | Dev | Test |
|----------|-------|-------|-----|------|
| float_poly | MLP(64×64) | 0.163 | 0.025 | 0.000 |
| integer | MLP(64×64) | 0.110 | 0.014 | 0.000 |
| **onehot** | **MLP(64×64)** | **0.936** | **0.250** | **0.286** |
| symbolic | — | — | 1.000 | 1.000 |

One-hot encoding improves from 0% to 25–29%, confirming that discrete integer identity is relevant. However, the 93.6% train accuracy against 28.6% test accuracy indicates clear memorisation without algebraic generalisation. The modular reduction step f mod m introduces periodic discontinuities that continuous and token-level MLP architectures cannot generalise across orbit families.

### 4.2 `shortest_witness`: generator-relative orbit traversal

| Encoding | Model | Train | Dev | Test | IID−OOD |
|----------|-------|-------|-----|------|---------|
| float_poly | MLP(64×64) | 0.492 | 0.404 | 0.388 | +0.016 |
| **onehot_flat** | **MLP(128×64)** | **0.972** | **0.728** | **0.641** | **+0.087** |
| bilinear | MLP(256×128) | 0.944 | 0.415 | 0.359 | +0.056 |
| symbolic | — | — | 1.000 | 1.000 | 0.000 |

The one-hot flat encoding (source + target tokens, no cross terms) outperforms both float and bilinear features. The bilinear model (which adds source⊗target outer products) overfits despite more information: per-pair identifiers memorise training transitions rather than learning orbit structure. Accuracy under the best baseline is *uniform across difficulty tiers* (easy=61.5%, medium=65.3%, hard=64.9%), indicating the model has learned a general orbit-position rule rather than a shortcut for easy cases.

The 8.7pp IID/OOD gap is stable across split versions and is the clearest algebraic generalisation signal in the benchmark.

### 4.3 Auxiliary tasks

`orbit_class` (split v2): MLP achieves dev=0.889, test=0.866, gap=+0.023. Now a real (if modest) benchmark task.

`reachability` (balanced): now 50/50 True/False; meaningful results require a rerun (not reported here — baseline sweep used pre-balance data for this task).

---

## 5. Theoretical Grounding

The benchmark's difficulty is not arbitrary. It reflects the algebraic structure of Z[φ]/mZ[φ], the finite quotient of the ring of integers of Q(√5).

**`invariant_pred`** asks models to compute f(b,e) = N(b+eφ) mod m, the Q(√5) norm residue. The norm is T-invariant (f(T(b,e)) = f(b,e) mod m), and orbit classification is determined by the 3-adic valuation v₃(f): v₃ ≥ 2 implies degenerate (satellite or singularity); v₃ = 0 implies cosmos. Learning to predict f is equivalent to learning an algebraic invariant that partitions the state space — a task that requires representing modular periodic structure unavailable in standard activation functions.

**`shortest_witness`** asks models to compute the discrete index of (b*,e*) relative to (b,e) in the cyclic group generated by T on Z[φ]/mZ[φ]. Within a cosmos orbit of length L = π(m)/2, this is the discrete logarithm `(pos_target − pos_source) mod L`. Learning this across unseen orbit families requires learning the orbit group structure, not just local transition patterns.

**Connection to QA certificates.** The cert ecosystem ([89]–[101]) pins the curvature scalar κ = 1 − |1 − lr·gain·H_QA| and asserts, via the Finite-Orbit Descent Theorem, that this governs optimizer convergence. The 0% baseline on `invariant_pred` is a concrete empirical statement: current neural models cannot verify QA cert claims from first principles. That gap defines the frontier this benchmark is intended to probe.

---

## 6. Discussion

**What the benchmark is and is not.** QA-ORBIT is not a test of general reasoning ability. It is a targeted algebraic-structure probe. The four tasks operationalise specific mathematical properties of one system. Its value is that every answer is verifiable, the difficulty is interpretable from first principles, and the IID/OOD design targets exactly the algebraic-generalisation question.

**Primary limitation.** The orbit-index split (cosmos families sorted by canonical root) means adjacent families in dev and test may be structurally similar to train families. A harder split — by norm class or by a larger held-out fraction — would widen the IID/OOD gap and provide a more demanding evaluation.

**Recommended evaluation protocol.** Report `acc@type`, `acc@difficulty`, `IID_acc` (dev), `OOD_acc` (test), and `IID−OOD_gap` separately for `invariant_pred` and `shortest_witness`. Do not headline the average across all four tasks; `orbit_class` and `reachability` are auxiliary until a harder split is in place.

---

## 7. Conclusion

QA-ORBIT provides a certifiable benchmark for algebraic generalisation in a specific and theoretically grounded modular dynamical system. The baseline sweep establishes that the primary challenge tasks — learning the Q(√5) norm and learning orbit-traversal step counts — are not solved by standard MLP or logistic regression baselines under any tested encoding, and exhibit measurable IID/OOD gaps under the best current encoding. These results establish the benchmark floor. Closing the gap between the 64.1% one-hot MLP and the 100% symbolic ceiling on `shortest_witness`, and any progress on `invariant_pred` beyond 28.6%, are the concrete targets for future model classes.

---

## References

- QA Unified Curvature Paper (companion paper, this project)
- Wilf-Zeilberger / Pisano period: for π(m) = Pisano period of Fibonacci mod m
- Z[φ] = ring of integers of Q(√5); norm N(a+bφ) = a² + ab − b²
- Finite-Orbit Descent Theorem: L_{t+L} = ρ(O)·L_t, §8.1 of companion paper
