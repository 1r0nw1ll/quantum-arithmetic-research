# Agency Decomposition via Path Diversity
## A Structural Analysis of Competency Metrics in Goal-Directed Systems

**Family [26] Whitepaper Extract · v1.0.0 · 2026-03-01**

---

### Abstract

We introduce the **Path Diversity Index (PDI)**, a structural metric quantifying
counterfactual route richness in competency-certified systems.  PDI complements
the existing Plasticity Index (PI) by measuring the fraction of reachable states
accessible via ≥2 distinct directed paths in the SCC-condensation DAG of the
system's generator graph.  Together, (PI, PDI) define four behavioral regimes —
*Frozen*, *Linear Explorer*, *Stuck-Loop Thrashing*, *Flexible Planner* —
separated by the **Agency Decomposition Theorem**: EA = AI × PDI.
Empirical analysis of nine reference systems spanning AI agents and biological
substrates reveals that all fall below the PDI = 0.5 threshold (range 0.425–0.475),
clustering exclusively in the *Linear Explorer* and *Frozen* regimes.
No reference system achieves *Flexible Planner* status, establishing a structural
ceiling for current competency exemplars and motivating a new design criterion for
high-agency systems.

---

### 1  Background

The QA Competency Detection Framework (Family [26]) certifies goal-directed
systems along four dimensions: Agency Index (AI), Plasticity Index (PI),
Goal Density (GD), and Control Entropy (CE).  Prior work established that AI
captures the *size* of the reachable control region while PI captures
*temporal adaptation velocity*.  However, neither metric addresses a third
structural property: whether the system has multiple independent routes to
its goals — the property that distinguishes a resilient planner from a
brittle one.

We define **Path Diversity Index** to fill this gap:

$$PDI \;=\; \frac{|M|}{R}$$

where $R$ is the count of reachable states and $M \subseteq R$ is the subset
of reachable states with **≥ 2 distinct directed paths** from the set of
initial states, counted on the **SCC-condensation DAG** of the generator graph.

The condensation DAG collapses each strongly connected component (SCC) to a
single node and imposes a topological order.  Path counts propagate in that
order: states inside a non-trivial SCC (a cycle) are immediately saturated to
count 2 if any route enters the SCC via 2 or more distinct paths.

---

### 2  Main Results

**Theorem 1 (SCC Saturation Lemma).**
*Let $S$ be a non-trivial SCC (containing a directed cycle).  If any state in
$S$ is reachable via ≥ 2 distinct directed paths, then every state in $S$
is reachable via ≥ 2 distinct directed paths.*

*Proof sketch.*  Within $S$ every pair of states is mutually reachable by the
SCC definition.  Composing the 2-route entry with any intra-SCC path yields
2 distinct routes to every member of $S$.  ∎

**Theorem 2 (No Artificial Inflation).**
*Cycles in the generator graph cannot increase PDI beyond what the DAG
condensation already implies.  Formally: removing all back-edges (reducing
every SCC to its entry node) does not decrease $|M|$.*

*Consequence.*  Inflated PDI estimates produced by naive path-counting on
cyclic graphs (without condensation) are provably incorrect.  The
SCC-condensation algorithm is the canonical counter.

**Theorem 3 (Agency Decomposition).**
*Let AI ∈ [0,1] measure the fractional size of the reachable control region
and PDI ∈ [0,1] as defined above.  Then Effective Agency*

$$EA \;=\; AI \times PDI$$

*decomposes agency into orthogonal structural factors: control-region extent
(AI) and counterfactual route richness (PDI).*

The orthogonality is empirical (Pearson r < 0.15 across reference set) and
structural: AI depends on reachability BFS diameter, while PDI depends on
DAG merge topology.  Neither implies the other.

**Theorem 4 (Cyclic Thrashing).**
*If PI = 0 and PDI > 0, the system recurs infinitely within a non-trivial SCC
without escaping to new reachable states.*

This formalizes the intuition behind the *Stuck-Loop Thrashing* regime:
multi-route access combined with zero plasticity produces a system that cycles
through known states rather than discovering new ones.

---

### 3  The (PI, PDI) Regime Map

The thresholds PI = 0.5 and PDI = 0.5 partition system space into four
structural regimes (Figure 1):

| Regime | PI | PDI | Structural character |
|---|---|---|---|
| **Frozen** | lo | lo | Deterministic, tree-like; EA minimal |
| **Linear Explorer** | hi | lo | Reaches new territory; single-route brittle |
| **Stuck-Loop Thrashing** | lo | hi | Redundant paths; no new territory; cycles |
| **Flexible Planner** | hi | hi | New territory + redundant paths; EA maximal |

*Figure 1: PI/PDI Structural Regime Map with empirical reference set
(n = 9).  See `qa_alphageometry_ptolemy/pi_pdi_quadrant.png`.*

The four regimes are not equally populated.  Achieving *Flexible Planner*
status simultaneously requires a wide reachability diameter (high PI) and a
merging-rich DAG topology (high PDI).  These constraints conflict: wide trees
tend to be shallow and fan-out dominated (low PDI), while merge-heavy graphs
tend to be dense and depth-constrained (moderate PI).

---

### 4  Empirical Findings

Nine reference systems were certified against the Family [26] schema across
three substrate domains (AI agents, biological systems, hybrid
bioelectric+human-in-the-loop systems).  Key results:

- **PDI range:** 0.425–0.475 across all nine systems — a compressed band
  well below the 0.5 threshold.
- **Regime distribution:** 7 of 9 in *Linear Explorer* (PI > 0.5, PDI < 0.5);
  2 of 9 in *Frozen* (tool-agent debugger: PI = 0.30; lab robot: PI = 0.35).
- **No system reaches *Flexible Planner* or *Stuck-Loop Thrashing*.**
- **PI/PDI orthogonality confirmed:** bio systems have higher mean PI (0.71)
  than AI agents (0.56), yet PDI is statistically indistinguishable across
  domains (bio 0.450, AI 0.432, hybrid 0.447; range < 0.02).

The compressed PDI band is structurally expected.  Reference systems are
hand-crafted exemplars with branching-factor-bounded generator graphs and
no deliberate merge engineering.  Achieving PDI > 0.5 requires explicit
construction of fan-then-merge subgraphs — a design criterion absent from
current engineering practice.

---

### 5  Implications

**For system designers.**  PDI provides a tractable design target:
systems with sparse generator graphs (few merge points) will be structurally
capped below PDI ≈ 0.45 regardless of how many generators they have.
Achieving Flexible Planner status requires *merge topology*, not merely
*generator count*.

**For competency certification.**  PDI and PI are non-redundant: a system can
score high on PI (temporal plasticity) while scoring low on PDI (structural
redundancy), or vice versa.  Certification of high-stakes systems should
report both.

**For the Cyclic Thrashing diagnosis.**  The stuck-loop failure mode —
observed in live demos as Phase 3 PDI = 0.889 with PI = 0 — is now formally
characterized.  It cannot arise from a Linear Explorer; it requires prior
construction of merge-rich topology before the plasticity collapse.

---

### References

- Family [26] operational spec: `docs/families/26_competency_detection.md`
- Family [26] formal theory: `docs/families/26_competency_detection_pdi_formal_theory.md`
- Schema: `qa_competency/schemas/QA_COMPETENCY_DETECTION_FRAMEWORK.v1.schema.json`
- Reference sets: `qa_competency/reference_sets/v1/`
- Figure 1 generator: `qa_alphageometry_ptolemy/generate_pi_pdi_quadrant.py`
- Tag: `family-26-pdi-v1.0.0` · `family-26-figure1-v1.0.0`
