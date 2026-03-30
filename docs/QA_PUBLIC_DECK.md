# QA: Two Spines, One Kernel

### A framework where arithmetic determines what is reachable — and what is impossible.

*Technical: a control and reachability theory over modular arithmetic state spaces.*

---

## Slide 1 — What QA Is

**One sentence:**

> QA is a framework where arithmetic structure directly determines reachability,
> and reachability directly determines search cost.

**The state space (derived from arithmetic — not chosen):**

```
- Singularity: fixed point (no outgoing transitions)
- Satellite:   periodic orbit (length 8)
- Cosmos:      larger periodic orbit (length 24)

All reachable states lie on one of these orbit classes.
The classification is determined by modular arithmetic over ℤ[φ] (φ = golden ratio).
```

```
          [107] QA_CORE_SPEC.v1
               (kernel)
                  │
        ┌─────────┴──────────┐
        │                    │
   Singularity           Satellite
   (fixed point)         (8-cycle)
        │                    │
        └─────────┬──────────┘
                  │
               Cosmos
              (24-cycle)
```

The orbit names are labels for arithmetic classes. A state belongs to cosmos, satellite,
or singularity because of its arithmetic properties — not because of anything we choose.

**What this is not:**

- Not new number theory
- Not a new logic
- Not a heuristic

**What it is:**

A framework where arithmetic structure directly determines reachability,
and reachability directly determines search cost.

---

## Slide 2 — The Core Claim

These two results show that **arithmetic structure determines both reachability and search cost.**

Two independently certified results. Both derived from the same kernel.

---

### Obstruction spine

> For any modulus m = pᵏ with p inert in ℤ[φ],
> the p-adic valuation condition vₚ(r) = 1 is a **complete arithmetic obstruction**:

```
vₚ(r) = 1
  ⟹  r is unreachable
  ⟹  planner expands 0 nodes
  ⟹  pruning_ratio = 1.0
```

Impossibility is **computable from arithmetic alone** — before any search begins.

---

### Control spine

> The compiler law is **domain-generic**:
> the orbit trajectory singularity → satellite → cosmos and path_length_k = 2
> are structural invariants that hold across physically distinct domains.

```
Cymatics    (flat → hexagons)       orbit: singularity→satellite→cosmos  k=2
Seismology  (quiet → surface_wave)  orbit: singularity→satellite→cosmos  k=2
```

The physics changes. The orbit does not.

---

## Slide 3 — Obstruction: One Concrete Example

**Target: r = 6, modulus m = 9 (p = 3, k = 2)**

#### Step 1 — Arithmetic verdict

```
v₃(6) = 1          ← 6 is divisible by 3 but not by 9
3 is inert in ℤ[φ]  ← 3 does not split in the ring of integers of ℚ(√5)

⟹  r = 6 is arithmetically forbidden
```

#### Step 2 — Planner behavior

```
Naive planner:            nodes_expanded = 47
Obstruction-aware planner: nodes_expanded = 0
```

The obstruction-aware planner does not search. It cannot reach r = 6.
There is nothing to try.

#### Step 3 — Efficiency

```
saved_nodes   = 47
pruning_ratio = 1.0   (100%)
```

This is not a good heuristic.
It is a **proof that no search is needed**.

The same guarantee holds for every r with vₚ(r) = 1, for every inert prime p.

---

## Slide 4 — Control: The Cross-Domain Result

Two domains. Different physics. Different state labels. Different moves.

| | Cymatics | Seismology |
|---|---|---|
| Initial state | flat | quiet |
| Intermediate | stripes | p_wave |
| Target state | hexagons | surface_wave |
| Moves | increase_amplitude, set_frequency | increase_gain, apply_lowpass |
| Governing physics | Faraday instability | Elastic wave propagation |

**What they share — from the arithmetic:**

```
orbit:   singularity → satellite → cosmos
k:       2
kernel:  QA_CORE_SPEC.v1
```

The compiler law does not know about Faraday patterns or seismic waves.
It governs the abstract orbit structure.
The domain provides only the concrete instantiation.

These systems share no governing equations, variables, or units.

> **A new domain is not a new theorem. It is a new witness of the same theorem.**

---

## Slide 5 — What Is Actually New

The key question: *what does QA give you that existing formalisms don't?*

---

### Classical mathematics

- Proves that solutions *exist* (existence theorems)
- Does not say how hard they are to find
- Does not eliminate search

### AI planning

- Estimates which branches are *unlikely* (heuristic pruning)
- Pruning is approximate — wrong pruning misses solutions
- Guarantees are statistical, not structural

### Machine learning

- Learns patterns in data
- Patterns are not guarantees
- Generalization is empirical, not provable

---

### QA gives you something different

> **QA turns impossibility into a first-class, computable object
> that eliminates search and transfers across domains.**

Specifically:

| Property | What it means |
|----------|---------------|
| **Deterministic impossibility certificates** | Not "unlikely" — provably unreachable, derived from vₚ |
| **Search elimination guarantees** | nodes_expanded = 0 is a theorem, not a heuristic |
| **Cross-domain invariant structure** | Same orbit holds in cymatics and seismology — verifiable, not assumed |
| **Machine-verified** | Every claim in this deck is backed by a certified fixture that a validator checks |

The last row matters. These are not informal claims.
They are checked by code, against canonical witnesses, every time the validator runs.

One more way to say it:

> **QA replaces search over states with classification of states via arithmetic invariants.**

---

## Slide 6 — Why It Matters

*Status: first two are directly certified; last two are immediate extensions.*

---

### ✓ Certified — AI planning and search

Obstruction-aware planners never expand forbidden nodes.
Search cost for arithmetically blocked targets is provably zero.
This is not a speedup. It is an **elimination**.

### ✓ Certified — Automated theorem proving

Proof search can be pruned before any inference step
for targets that fail the arithmetic obstruction check.
The pruning is **sound** — it cannot miss valid proofs.

### → Extension — Physics and dynamical systems

Orbit structure is derivable from arithmetic — not from simulation.
Shared orbit across cymatics and seismology suggests
a classification principle for physical pattern transitions
that is independent of the underlying PDE.

### → Extension — Structured machine learning

State spaces with QA orbit structure carry
provable algebraic invariants.
These can be used as hard constraints on embeddings and representations —
not as regularization, but as **guaranteed structure**.

---

## Slide 7 — What to Read Next

**Reading order: outermost → inward**

```
[120] QA_PUBLIC_OVERVIEW_DOC.v1          ← this deck
  └── [119] QA_DUAL_SPINE_UNIFICATION_REPORT.v1   ← formal unified map
        ├── [116] QA_OBSTRUCTION_STACK_REPORT.v1  ← obstruction spine (full)
        │         v_p(r)=1 → unreachable → pruned → ratio=1.0
        └── [118] QA_CONTROL_STACK_REPORT.v1      ← control spine (full)
                  orbit singularity→satellite→cosmos, k=2, domain-generic
```

**Each entry point is a validated document.**
The validator checks that the claims in the document match the underlying fixtures.
A document that contradicts its own theorem fails validation.

---

### The one sentence

> QA turns impossibility into a first-class, computable object
> that eliminates search and transfers across domains.

---

*Machine-verified. Dual-spine. Kernel-governed.*

`qa_alphageometry_ptolemy/` — `docs/families/120_qa_public_overview_doc.md`
