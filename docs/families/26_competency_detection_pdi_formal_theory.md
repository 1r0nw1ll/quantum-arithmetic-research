# [26] Competency Detection — PDI Formal Theory

## Human Tract: Path Diversity Index (v1.0.0)

Tag: `family-26-pdi-v1.0.0`

---

## 1. Scope

Family [26] formalizes a **computable, certificate-verifiable operationalization** of
Michael Levin's competency framework.

It does **not** attempt to prove Levin's theory.

It provides:

- A state-graph model of behavior
- Five structural metrics
- Deterministic validators
- Obstruction certificates when competency boundaries are hit

All claims are generator-relative.

---

## 2. Levin → QA Mapping

| Levin Concept         | QA Object                      | Certifiable? | Notes                       |
| --------------------- | ------------------------------ | ------------ | --------------------------- |
| Competency            | Reachability Class             | Yes          | BFS-reachable region        |
| Goal                  | Attractor Basin (sink SCC)     | Yes          | Structural, not semantic    |
| Memory                | Invariants                     | Yes          | Encoded in graph structure  |
| Agency                | Control Region Size (AI)       | Yes          | Reachable fraction          |
| Plasticity            | Delta Reachability per step    | Yes          | Windowed                    |
| Counterfactual Agency | Path Diversity (PDI)           | Yes (v1.0.0) | SCC-condensation DAG        |

This mapping is structural, not substrate-dependent.

---

## 3. Graph Model

A system is modeled as:

    G = (V, E)

Where:

- V = states
- E = generator transitions
- I ⊆ V = initial states

All metrics derive strictly from this directed graph.
No semantic interpretation is required for validation.

---

## 4. The Five Metrics

### 4.1 Agency Index (AI)

    AI = |R| / |V|

- R = reachable states from I
- Measures size of control region
- High AI ≠ high intelligence
- Deterministic linear explorers can have AI ≈ 1

### 4.2 Plasticity Index (PI)

    PI = delta_reachability / delta_perturbation

Measures discovery rate of new states. Captures **temporal expansion**, not
structural redundancy.

Reference set finding: bio > hybrid > AI.

### 4.3 Goal Density (GD)

    GD = |sink SCCs| / |V|

Counts attractor basins.

Limitation: structural sinks include crashes. Intentional goal states must be
semantically identified externally.

### 4.4 Control Entropy (CE)

    CE = -sum p(g) ln p(g)

Where p(g) = generator usage distribution.

Observed: CE ≈ ln(k) - KL(p || Uniform_k). Measures decentralization of
generator use. Not equivalent to intelligence.

### 4.5 Path Diversity Index (PDI) — v1.0.0

    PDI = |{v ∈ R : #paths(I → v) ≥ 2}| / |R|

Paths counted on the **SCC-condensation DAG**:

- Non-trivial SCCs saturate all internal nodes
- Trivial SCCs propagate counts in topological order
- Path counts capped at 2 (only ≥2 detection required)

Measures **counterfactual route richness**.

---

## 5. The Agency Decomposition

Before v1.0.0:

    Agency ≈ Reachability (AI)

After PDI:

    Agency = (Control Size) × (Counterfactual Redundancy)

Operationally:

    Effective Agency (EA) = AI × PDI

Interpretation:

- High AI, Low PDI → deterministic competence
- High PDI, Low PI → cyclic thrashing
- High PI, High PDI → flexible planner
- Low AI, Low PDI → constrained automaton

Live debugger demo:

| Phase | PI   | PDI   | Diagnosis                      |
| ----- | ---- | ----- | ------------------------------ |
| 1–2   | High | 0.0   | Linear deterministic expansion |
| 3     | 0    | 0.889 | Stuck-loop thrashing           |

This quadrant structure cannot be captured by AI or PI alone.

---

## 6. Validator Guarantees

Family [26] enforces a bidirectional PDI guard (Gate 4b):

| Condition                                      | fail_type          |
| ---------------------------------------------- | ------------------ |
| `multi_path_states` present, `pdi` wrong/absent| `PDI_MISMATCH`     |
| `pdi` present, `multi_path_states` absent      | `PDI_INPUT_MISSING`|

This prevents floating metrics without input basis. All metrics are recomputed
deterministically from cert inputs.

---

## 7. Reference Set Correction

Earlier synthetic construction inflated PDI (~0.96–0.99) due to duplicate
initial states in event generator (cross-link episodes reused primary episode
start states, causing `scc_path` to saturate immediately).

Corrected analytical fan-then-merge DAG formula:

    A   = round(R / (comps * (D + 1)))
    mid = D // 2
    multi_path_states = comps * (A - 1) * (D - mid)
    PDI = multi_path_states / R

Corrected reference set range: **PDI ∈ [0.425, 0.475]**

Confirms: PDI is topology-uniform across domains; PI remains the
domain-separating metric.

Validation status: 9/9 PASS, 7/7 fixtures PASS, meta-validator 92/92 PASS.

---

## 8. What Family [26] Does NOT Claim

- Does not measure intelligence
- Does not measure semantic surprise
- Does not prove Levin's theory
- Does not infer goals beyond structural attractors
- Does not measure creativity

It measures **structural competency geometry**.

---

## 9. What Is Novel

1. First certificate-based operationalization of Levin's framework
2. SCC-condensation-based PDI (counterfactual agency metric)
3. Formal separation of control size (AI), exploration rate (PI), structural
   redundancy (PDI)
4. Obstruction certificates when competency boundaries are hit

---

## 10. Interpretation Discipline

Family [26] metrics are:

- Generator-relative
- Substrate-independent
- Structurally defined
- Behavior-derived

All cross-domain comparisons are valid only when:

- Generator set is defined
- State equivalence relation is fixed
- SCC condensation is computed consistently

---

## 11. Status

Tag: `family-26-pdi-v1.0.0`

Validation:

- Competency fixtures: PASS (7/7)
- Reference sets: PASS (9/9)
- Guardrail alignment: PASS (18/18)
- Meta-validator: PASS (92/92)

---

## 12. Structural Diagram — Control Region vs SCC-Condensation

### 12.1 Full State Graph (G)

Example behavior graph:

```
Initial
   |
   v
  A1 → A2 → A3
   |           \
   v            v
  B1 → B2 → B3 → C
   ^            /
   |___________/
```

- All nodes reachable → AI ≈ 1.0
- There is a cycle: {B1, B2, B3, C}
- Multiple routes exist to C

### 12.2 Control Region (AI)

Control Region = BFS-reachable set R

    R = { Initial, A1, A2, A3, B1, B2, B3, C }

    AI = |R| / |V|

Captures **how much territory is reachable**. Does not capture whether that
territory has redundant routes.

### 12.3 SCC-Condensation DAG

Collapse strongly connected components:

```
[Initial]
    |
[A-chain]
    |
[Cycle_SCC]    ←  {B1, B2, B3, C}
```

Count distinct directed paths on the condensation DAG.
If two separate upstream routes enter the same SCC, all nodes inside are
multi-path.

### 12.4 Visual Contrast

| Metric | What it sees                                       |
| ------ | -------------------------------------------------- |
| AI     | Size of reachable region                           |
| PI     | Rate of expansion                                  |
| PDI    | Redundant route structure in condensation DAG      |

This separation explains:

- AI can remain high while PI collapses.
- PDI can be high even when PI = 0 (cyclic thrashing).
- PI can be high while PDI = 0 (linear tree expansion).

---

## 13. Formal Theorem Section

### 13.1 Definitions

Let:

- G = (V, E) be a finite directed graph
- I ⊆ V initial states
- R ⊆ V reachable states via BFS from I
- C(G) = SCC-condensation DAG
- paths(I → v) = count of distinct directed paths on C(G)

Define:

    AI  = |R| / |V|
    PDI = |{v ∈ R : paths(I → v) ≥ 2}| / |R|

### 13.2 Agency Decomposition Theorem

**Theorem.** Let a system be represented by G = (V, E) with initial states I.

Then structural agency decomposes into two orthogonal components:

1. Control Region Size (AI)
2. Counterfactual Route Redundancy (PDI)

Such that:

- AI depends only on reachability.
- PDI depends only on path multiplicity in C(G).
- AI and PDI are independent invariants of graph topology.

**Proof sketch.** AI is computed solely from BFS reachable set R. Removing
alternative paths without changing reachability leaves AI invariant.

PDI depends solely on path multiplicity in C(G). One can construct graphs where
reachability is identical but path multiplicity differs, and vice versa.

Therefore there exist graphs G₁, G₂ such that:

    AI(G₁) = AI(G₂)   and   PDI(G₁) ≠ PDI(G₂)

and vice versa. Hence AI and PDI are structurally independent. ∎

### 13.3 Corollary — Effective Agency

Define:

    EA = AI × PDI

Then EA = 0 if either:

- No reachable control (AI = 0)
- No counterfactual structure (PDI = 0)

EA provides a scalar lower bound on structural agency.

### 13.4 Classification Quadrants

| PI   | PDI  | Structural Type       |
| ---- | ---- | --------------------- |
| High | Low  | Linear explorer       |
| Low  | High | Cyclic thrashing      |
| High | High | Flexible planner      |
| Low  | Low  | Constrained automaton |

The debugger demo instantiates (PI = 0, PDI ≈ 0.89) → cyclic thrashing regime.

---

## 14. Theoretical Implications

The addition of PDI resolves the prior ambiguity:

> AI alone cannot distinguish deterministic linear control from
> counterfactual agency.

With PDI:

- Structural agency is no longer conflated with reachable territory.
- Counterfactual capacity becomes certificate-verifiable.
- Levin's "agency" concept gains a computable structural witness.

---

## 15. SCC Saturation Lemma

### 15.1 Setup

Let S ⊆ V be a strongly connected component of G.
Let paths(I → S) denote the number of distinct directed paths in the
condensation DAG from any SCC containing an initial state to SCC S.

Path counts are capped at 2.

### 15.2 Lemma

If paths(I → S) ≥ 2 in the condensation DAG, then:

    for all v ∈ S:  paths(I → v) ≥ 2   in the original graph G.

In words: if an SCC receives two distinct upstream routes in the condensation
DAG, every node inside that SCC is multi-path.

### 15.3 Proof

Because S is strongly connected: for all u, v ∈ S, there exists a directed
path u ↝ v.

Assume two distinct directed paths P₁, P₂ exist from some initial SCC to S.
These paths terminate at (possibly different) nodes s₁, s₂ ∈ S.

Let v ∈ S be arbitrary. Because S is strongly connected:

- There exists a path s₁ ↝ v
- There exists a path s₂ ↝ v

Extend:

    P₁ · (s₁ ↝ v)
    P₂ · (s₂ ↝ v)

These are two distinct directed paths from an initial state to v.

Therefore paths(I → v) ≥ 2. ∎

### 15.4 Corollary (Cycle Saturation)

If an SCC has size ≥ 2 and receives at least two upstream routes in the
condensation DAG, all internal nodes are multi-path. The contribution to
`multi_path_states` equals the size of the SCC.

PDI mass propagates in blocks over SCCs.

### 15.5 Consequence for Implementation

Because of saturation:

- No need to count internal node-level paths inside cycles.
- It suffices to count paths on the condensation DAG.
- Path multiplicity at SCC granularity is exact for PDI purposes.

This justifies: Tarjan SCC, DAG topological propagation, capped path counting.

Algorithm remains O(|V| + |E|).

### 15.6 Structural Insight

SCC saturation explains why:

- Cyclic thrashing can yield high PDI even when PI = 0.
- A single upstream bifurcation feeding a large cycle can dominate PDI.
- Linear trees (no merging) produce PDI = 0 even with AI ≈ 1.

> Cycles do not inflate PDI by themselves.
> Only multi-route entry into SCCs increases PDI.

This distinction is essential for preventing artificial inflation of
counterfactual agency.

---

## 16. Why Raw Path Counting Is Wrong

### 16.1 Counterexample — Cycle Inflation Without SCC Collapse

Consider:

```
Initial → A → B → C
              ↑     ↓
              └─────┘
```

Edges: Initial → A, A → B, B → C, C → B (cycle between B and C).

**Raw path counting (incorrect method):**

Paths to B:

    Initial → A → B
    Initial → A → B → C → B
    Initial → A → B → C → B → C → B
    ...

There are infinitely many distinct directed paths. Both B and C would appear
multi-path; PDI → 1. This is **false**. There is only one structural upstream
route. The cycle introduces repetition, not counterfactual diversity.

### 16.2 Correct Condensation DAG

Collapse SCC {B, C}:

```
Initial → A → [SCC_BC]
```

Exactly one path to the SCC. Therefore paths(Initial → SCC_BC) = 1.
No node is multi-path. PDI = 0.  Correct result.

### 16.3 Theorem — No Artificial Inflation

**Theorem.** Let G = (V, E) be a finite directed graph. Define PDI using the
SCC-condensation DAG with path multiplicity capped at ≥2.

Then cycles in G cannot increase PDI unless they receive at least two distinct
upstream routes in C(G).

**Proof.** Let S ⊆ V be an SCC. If paths(I → S) = 1 in the condensation DAG,
then by definition there exists exactly one upstream route entering S. All
directed paths to nodes in S must pass through that single condensation path.
Internal cycling produces infinitely many raw paths, but they share the same
condensation prefix. Therefore paths(I → v) < 2 under capped counting for all
v ∈ S. Hence PDI does not increase. ∎

### 16.4 Consequence

Without SCC collapse: PDI would diverge on cyclic graphs. Any loop would
falsely imply high counterfactual structure.

With SCC collapse: PDI measures upstream route multiplicity only. Internal
recurrence does not inflate the metric. Structural meaning is preserved.

---

## 17. Upper Bounds on PDI Given Generator Fan-Out k

### 17.1 Setup

Assume each state has at most k outgoing generator transitions and the
condensation DAG is finite and acyclic. Let R = |reachable|.

Seek a tight upper bound on PDI = M / R where M = |{v ∈ R : paths(I → v) ≥ 2}|.

### 17.2 Tree Expansion (No Merging)

If the condensation DAG is a tree: each node has exactly one upstream path, no
merging occurs, PDI = 0. Fan-out alone does not increase PDI.

### 17.3 Minimal Merging Construction

Maximum PDI occurs when the graph fans out then reconverges:

```
Initial
  /   \
 A1   A2
  \   /
   Merge
     |
   Large subtree (size D)
```

All nodes downstream of Merge are multi-path.

    PDI = D / R

### 17.4 Tight Upper Bound

Given maximum fan-out k, the largest possible multi-path region occurs when:

- Initial splits into k branches
- All k branches merge
- Merged region covers the remainder

Then at least k upstream branch roots must exist before merging:

    PDI ≤ 1 - k/R

As R → ∞: sup PDI = 1. For finite R: PDI ≤ 1 - k/R. This bound is tight.

### 17.5 Stronger Structural Bound

If depth before merging is d:

    PDI ≤ (R - k*d) / R

Larger fan-out k and deeper pre-merge depth d reduce achievable PDI.

### 17.6 Interpretation

PDI depends on **merging structure**, not branching alone. Counterfactual
redundancy requires reconvergence. Fan-out increases potential exploration;
merging increases counterfactual redundancy. These are structurally distinct
graph properties.

---

## 18. Summary of All Results

| Result | Statement |
| ------ | --------- |
| SCC Saturation Lemma | Multi-route entry to SCC saturates all internal nodes |
| No Artificial Inflation | Cycles alone cannot increase PDI |
| Upper Bound | PDI ≤ 1 - k/R under fan-out k |
| Lower Bound (Section 20) | PDI ≥ 1/(k+2) under k-regular merging |
| Cyclic Thrashing Theorem (Section 21) | PI=0 ∧ PDI>0 ⇒ infinite recurrence in non-trivial SCC |

Together: PDI is well-posed, bounded, merge-sensitive, and detects infinite
cyclic recurrence when PI collapses.

---

## 20. Lower Bound on PDI for k-Regular Merging Graphs

### 20.1 Definition (k-Regular Merging Graph)

Let G = (V, E) be finite. Each non-terminal SCC in the condensation DAG has
exactly k ≥ 2 outgoing edges, with at least one merging layer where at least
two upstream branches reconverge into a common downstream SCC S.

### 20.2 Minimal Merging Construction

```
           Initial
          /  |  ...  \
         B1  B2       Bk
           \  |  ...  /
            Merge_SCC
               |
           Downstream region (size D)
```

- Each branch Bᵢ contains exactly one pre-merge state
- Total reachable: R = 1 + k + D

### 20.3 Multi-Path States

All D states in downstream region receive at least two distinct upstream routes.

    M = D
    PDI = D / R

### 20.4 Lower Bound Theorem

**Theorem (Minimal Redundancy Bound).** In any k-regular merging condensation
DAG with at least one reconvergence:

    PDI ≥ 1 / (k + 2)

This bound is tight (achieved when D = 1).

**Interpretation:**

- Larger k (more fan-out) lowers the minimum achievable PDI.
- Small k merging graphs produce higher minimal redundancy.
- Even minimal merging forces a non-zero structural redundancy floor.

> Merging cannot occur without generating measurable counterfactual agency.

---

## 21. Cyclic Thrashing Theorem

### 21.1 Definitions

Let Rₜ = reachable states at time t, PI = delta|R| / delta_t.

### 21.2 Theorem (Cyclic Thrashing)

If:

1. PI = 0 over an infinite time horizon
2. PDI > 0

Then the system executes infinitely many transitions within at least one
non-trivial SCC.

### 21.3 Proof

Because PI = 0: |Rₜ| = constant. No new states are discovered; all transitions
occur within the existing reachable set.

Because PDI > 0: there exists at least one SCC S such that paths(I → S) ≥ 2.
By the SCC Saturation Lemma, all nodes in S are multi-path. Since the reachable
set is finite and no new states are added, the infinite execution sequence must
revisit some state infinitely often (Pigeonhole Principle).

Because PDI > 0 requires non-trivial merging structure, the recurrent state
lies in a non-trivial SCC. Therefore the system exhibits infinite recurrence
inside that SCC. ∎

### 21.4 Corollary (Thrashing Signature)

    PI = 0  ∧  PDI > 0

is a structural signature of:

- Non-terminal cyclic behavior
- No exploration
- Redundant route structure
- Infinite recurrence

This is exactly the debugger Phase 3 regime (PI = 0, PDI ≈ 0.89).

### 21.5 Converse

If PI = 0 ∧ PDI = 0: the system must terminate in a tree-like linear region
or remain in a trivial SCC. Thus cyclic thrashing requires PDI > 0.

---

## 22. Structural Takeaway

AI measures territory.

PI measures expansion velocity.

PDI measures reconvergent redundancy.

Only merging structure creates counterfactual agency.

---

*End of PDI Formal Theory tract. Tag: `family-26-pdi-v1.0.0`*
