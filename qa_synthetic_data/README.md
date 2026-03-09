# QA-ORBIT Benchmark

A certifiable synthetic reasoning benchmark generated from the Quantum Arithmetic (QA) modular dynamical system. Every task has a deterministic ground-truth answer and a machine-verifiable certificate.

---

## Overview

QA-ORBIT tests a model's ability to reason about modular dynamical systems: compute invariants, classify orbit structure, decide reachability, and find shortest paths. Unlike most synthetic benchmarks, every answer is **formally verifiable** — not filtered by a model judge.

The underlying system: state $(b, e) \in \{0,\ldots,m{-}1\}^2$, update rule:

```
d = (b + e) mod m
a = (b + 2e) mod m
```

Repeated application partitions the state space into orbits of three types:

| Class | Description | mod-24 count |
|---|---|---|
| **cosmos** | maximum-length orbits (length 12) | 504 states, 42 orbits |
| **satellite** | shorter periodic orbits (lengths 3, 4, 6) | 71 states |
| **singularity** | fixed point (length 1) | 1 state |

---

## Task Families

### 1. `invariant_pred`
**Input:** `(b, e, modulus)`
**Task:** Compute the QA norm $f(b,e) = (b^2 + be - e^2) \bmod m$
**Answer:** integer in $\{0,\ldots,m{-}1\}$
**Witness:** none (closed-form)
**Difficulty:** always `easy`

### 2. `orbit_class`
**Input:** `(b, e, modulus)`
**Task:** Classify the orbit containing $(b,e)$ as `cosmos`, `satellite`, or `singularity`
**Answer:** string label
**Witness:** orbit length (int)
**Difficulty:** `easy` = singularity, `medium` = satellite, `hard` = cosmos

### 3. `reachability`
**Input:** `(b, e, b_target, e_target, modulus)`
**Task:** Is $(b_\text{target}, e_\text{target})$ reachable from $(b, e)$ via QA steps?
**Answer:** bool
**Witness:** step count if reachable, `null` otherwise
**Difficulty:** `easy` ≤ 2 steps, `medium` ≤ 5, `hard` > 5

### 4. `shortest_witness`
**Input:** `(b, e, b_target, e_target, modulus)`
**Task:** Minimum QA steps from $(b,e)$ to $(b_\text{target}, e_\text{target})$
**Answer:** int (−1 if unreachable, 0 if same state)
**Witness:** list of intermediate states along the shortest path
**Difficulty:** same thresholds as `reachability`

---

## Dataset Statistics

| | mod-9 | mod-24 |
|---|---|---|
| **Total tasks** | 2,037 | 14,556 |
| invariant_pred | 81 | 576 |
| orbit_class | 81 | 576 |
| reachability | 978 | 6,990 |
| shortest_witness | 897 | 6,414 |
| easy | 645 | 4,605 |
| medium | 456 | 3,399 |
| hard | 936 | 6,552 |
| **train** | 1,389 | 10,344 |
| **dev** | 324 | 1,944 |
| **test** | 324 | 2,268 |
| Verifier failures | 0 | 0 |

---

## Split Philosophy

**Splits are by orbit family, not random.** This is the most important methodological choice.

A random 80/20 split would allow a model to memorise local transition patterns (e.g. "from state (3,5) the next state is (8,4)") and achieve high test accuracy without learning the underlying algebra. That would make the benchmark scientifically useless.

Instead:

- **singularity + all satellite states** → always in `train` (easy scaffold for basic structure)
- **cosmos orbits**: sorted deterministically by canonical root state, split **70 / 15 / 15** by orbit index

This means the **test set contains cosmos orbit families entirely unseen during training**. A model must generalise the algebraic rules across orbit structure, not memorise individual transitions.

| Split | Contents | Purpose |
|---|---|---|
| `train` | satellite + singularity + 70% of cosmos orbits | Learning |
| `dev` | 15% of cosmos orbits (adjacent to train) | IID evaluation |
| `test` | 15% of cosmos orbits (held out) | OOD evaluation |

The IID/OOD gap (dev accuracy − test accuracy) is the primary measure of algebraic generalisation.

---

## Row Format

Each file is JSONL (one JSON object per line):

```json
{
  "task_type": "reachability",
  "input": {"b": 3, "e": 7, "b_target": 11, "e_target": 2, "modulus": 24},
  "answer": true,
  "witness": 4,
  "difficulty": "medium",
  "canonical_hash": "a3f2...",
  "verifier_outcome": true
}
```

**Canonical hash:** `sha256("QA_SYNTHETIC\x00" + canonical_json(task_type + input))`
**`verifier_outcome`:** independently recomputed at generation time; all rows are `true`.

---

## Evaluation Protocol

**Recommended metrics:**

| Metric | Description |
|---|---|
| `acc@type` | Accuracy per task family |
| `acc@difficulty` | Accuracy per difficulty tier |
| `witness_validity` | Fraction of correct answers with a valid witness trace |
| `IID_acc` | Accuracy on dev split |
| `OOD_acc` | Accuracy on test split |
| `IID−OOD gap` | Primary generalisation measure |

**What a model may use:** input fields and modulus only. Lookup into training data at test time is not allowed.

---

## Files

```
data/
  QA_SYNTHETIC_mod9_all.jsonl      # full mod-9 dataset
  QA_SYNTHETIC_mod9_train.jsonl
  QA_SYNTHETIC_mod9_dev.jsonl
  QA_SYNTHETIC_mod9_test.jsonl
  QA_SYNTHETIC_mod24_all.jsonl     # full mod-24 dataset
  QA_SYNTHETIC_mod24_train.jsonl
  QA_SYNTHETIC_mod24_dev.jsonl
  QA_SYNTHETIC_mod24_test.jsonl
core.py          # QA arithmetic (parametrised by modulus)
tasks.py         # task generators
verify.py        # deterministic verifier
run_generator.py # regenerate datasets
baselines.py     # majority / random / symbolic baselines
```

## Regeneration

```bash
python run_generator.py                          # generates mod-9 and mod-24
python run_generator.py --modulus 9              # mod-9 only
python run_generator.py --modulus 9 --modulus 24 # explicit
```

---

## Baselines

| Baseline | Description | Expected accuracy |
|---|---|---|
| Majority class | Most common answer per task type (from train) | ~15–30% |
| Random | Uniform random valid answer | ~10–25% |
| Symbolic solver | Exact BFS (= the verifier) | 100% |

The symbolic solver is the performance ceiling and is implemented in `baselines.py`. It exists to confirm that tasks are solvable and to establish the upper bound.

A useful first ML baseline: a small sequence model (MLP or transformer) predicting `answer` from flattened input tokens, trained on `train`, evaluated on `dev` and `test` separately.

---

## Connection to Theory

QA-ORBIT is grounded in the Quantum Arithmetic convergence framework (see companion paper). The orbit structure arises from the algebraic properties of $\mathbb{Z}[\varphi]$ (the ring of integers of $\mathbb{Q}(\sqrt{5})$), and the norm invariant $f(b,e) = b^2 + be - e^2$ is preserved under the QA update map. Orbit classification by $f$-value is the discrete analogue of Lyapunov stability certification.
