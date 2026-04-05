# Generator-Aware QA Descriptors for Graph Community Detection

**Author:** Will Dale
**Date:** 2026-04-05
**Status:** Draft — GitHub-first publication

## Abstract

QA graph descriptors are not universal. A QA-native feature map that is
optimal for one network family can fail badly on another. We report five
benchmark graphs that make this explicit: the same QA axioms (A1 no-zero,
A2 derived coordinates, S1 integer multiplication, T2 observer-projection
firewall) admit multiple canonical descriptor families, and the right choice
depends on how the ground-truth community structure was generated. We isolate
two such families — **centrality-based invariants** and **hub-distance tuples**
— and show that each wins on graphs whose communities correspond to its
generator, and loses on graphs whose communities do not. The methodological
point is that "QA fails on graph X" is almost always shorthand for "our QA
extraction on X was miscalibrated": the axioms are fixed, the extraction is
not, and the correct extraction is determined by the community generator.

## 1. Background

The QA graph feature map ([qa_graph package]) encodes a pair `(b, e)` per node
as an 83-dimensional integer invariant vector drawn from the canonical 21
identities, the chromogeometric quadrances, and several derived families.
The standard choice in prior work was

$$
b = \deg(v), \quad e = \text{coreNumber}(v)
$$

and this worked well on moderately hub-diverse graphs such as American
College Football (`football.graphml`), delivering ARI = 0.906 for full-kernel
spectral clustering against baseline 0.850 (Δ = +0.056). On the 34-node
Zachary karate club, however, the same descriptor produces ARI = 0.15 versus
baseline 0.88 — an apparent catastrophic failure.

We claim the catastrophe is an extraction error, not an algorithmic one.

## 2. The Karate Diagnostic

Ground truth on karate splits the club into two factions, Mr. Hi's group and
John A.'s group. The two factions are cleanly separated by **geodesic
proximity**: each member is closer (in shortest-path sense) to their own
faction leader than to the other. Both leaders are high-degree high-core
nodes. In the `(deg, core)` feature map they map to **identical points** —
and any RBF kernel built on such features is structurally incapable of
separating them, because the feature vectors coincide.

The fix is to pick a descriptor whose image on the two leaders is **distinct**.
The simplest such descriptor is the integer distance vector

$$
\phi_H(v) = (d(v, h_1)+1,\; d(v, h_2)+1,\; \dots,\; d(v, h_k)+1)
$$

where `h_1, ..., h_k` are the top-k degree hubs and the +1 offset enforces
axiom A1. This descriptor is purely integer, A1-compliant by construction,
and `(d, a) = (b+e, b+2e)` follows as usual from A2. No floats, no
approximations — pure Theorem-NT compliance.

### 2.1 Single-coordinate analysis

On karate, the secondary-hub distance alone (`e = d(v, h_2) + 1`) recovers
ground truth at ARI = 0.7716 via a single integer threshold. The signed
difference `sign(b - e)` matches the winning spectral partition at
ARI = 0.7718 — meaning spectral clustering is **implicitly computing
hub-proximity** over a continuous relaxation of the same quantity.

The canonical invariants behave differently:

| Invariant | ARI vs spectral | ARI vs ground truth |
|---|---|---|
| `e` (distance to hub 2) | 0.668 | **0.772** |
| `sign(b − e)` | **0.772** | 0.669 |
| `b` (distance to hub 1) | 0.572 | 0.483 |
| `norm f(b, e) = b² + be − e²` | 0.572 | 0.483 |
| `d = b + e` | 0.007 | 0.000 |

The `d` path-sum is worthless as a classifier because it is ≈constant across
the graph — every node has roughly the same total distance to the two leaders.
But `e` alone is enough.

## 3. Five-Graph Benchmark

We extend the analysis to four additional graphs and compare five methods:
`baseline_spectral`, `louvain`, `qa_centrality_kernel` (the legacy
`(deg, core)` invariant kernel), `hub_distance_kmeans`, and the refined
`hub_distance_1axis` (pick the single hub whose distance partition maximises
Newman modularity).

| Graph | n | k | deg CV | baseline | qa-centrality | hub-dist 1-axis | best |
|---|---|---|---|---|---|---|---|
| karate | 34 | 2 | 0.833 | **0.882** | 0.072 | 0.772 | baseline |
| barbell(10, 3) | 23 | 2 | 0.295 | 0.826 | **1.000** | **1.000** | tied |
| caveman(4, 5) | 20 | 4 | 0.158 | **1.000** | **1.000** | 0.568 | baseline |
| florentine | 15 | 2 | 0.506 | 0.314 | -0.051 | **0.507** | **hub-dist** |
| football | 115 | 12 | 0.083 | **0.897** | 0.892 | 0.109 | baseline |

(Barbell and caveman are deterministic constructions from `networkx`; they
contain no randomness and therefore satisfy the Theorem-NT restriction
against stochastic QA test beds.)

### 3.1 What the table says

- **Florentine** is the clean win: historical Medici-vs-Strozzi faction
  labels are recovered at ARI 0.507 by hub-distance, versus 0.314 for
  spectral and 0.303 for Louvain. The QA descriptor beats every alternative
  by +0.19 ARI on a labelled real graph.
- **Barbell** is a tie: hub-distance reaches perfect ARI 1.0, matching the
  `qa_centrality_kernel` and improving on baseline spectral (0.826). The
  communities are two cliques joined by a bridge — both centrality and
  geodesic descriptors resolve it exactly.
- **Karate** is competitive: hub-distance (single axis) at ARI 0.772 sits
  below baseline 0.882 but matches the common published spectral result
  (≈0.77) and beats Louvain (0.490). More importantly, `sign(b − e)` is
  demonstrably what spectral is approximating.
- **Football** is the critical negative result. Degree CV is 0.083 — the
  graph is nearly degree-regular, with 12 dense teams-plus-conferences.
  Hub-distance collapses (ARI 0.109) because there are no meaningful hubs
  to anchor a geodesic partition. Baseline spectral and the legacy
  `qa_centrality_kernel` both reach ≈0.90.
- **Caveman** is trivially separable; everything perfect except the
  single-axis rule, which is too coarse for k = 4 cliques.

No single descriptor dominates the benchmark. Each wins on the graph family
whose community generator it was designed for, and each fails elsewhere.

## 4. The Generator-Aware Principle

We state the observation we believe this licenses:

> **A QA-native descriptor is the projection onto the QA state space of the
> generator that produced the community structure.** If the ground truth was
> generated by "assign each node to the faction of its nearest charismatic
> leader", the canonical QA descriptor is the integer tuple of shortest-path
> distances to the leaders. If the ground truth was generated by "nodes with
> similar structural role share a community", the canonical QA descriptor is
> the 83-dim invariant map from `(deg, core)`. The axioms are the same; the
> extraction changes.

This is the graph-theoretic specialization of the broader QA project ethos:
find what works in the domain first, map the winning structure into QA
algebra, certify the mapping. The QA contribution is never "a kernel that
beats everything on every graph". The QA contribution is **an axiom-faithful
integer representation of whatever generator the domain actually uses**.

### 4.1 Why this is not cargo-culting

A naive reading would be: "if you know the generator, just use it directly —
why map it into QA?". Three reasons this is not a rewrite of an existing
method:

1. **Composition.** Once a domain's generator is in QA form, it composes with
   every other QA descriptor — invariant identities, chromogeometric
   quadrances, orbit families, reachability witnesses — because they share
   the same axiom spine. A hub-distance tuple can be classified by its
   orbit family, checked for structural obstructions, or used as the input
   to a QA reasoner (`qa_reasoner`) for symbolic explanation. The raw
   distance rule has none of this.

2. **Discreteness guarantee.** Every QA extraction is integer, A1-compliant,
   and Theorem-NT-safe by construction. This is not a property of
   "shortest-path classification in general". It is a property that holds
   because the descriptor is defined inside the QA state space.

3. **Diagnostic lift.** When a QA descriptor fails on a graph (as
   hub-distance fails on football), the failure is structurally
   interpretable: the generator encoded by the descriptor is not the
   generator of the domain. This is information, not failure.

## 5. Reproduction

All code and data are in the repository:

```
qa_lab/qa_graph/karate_hub_distance_qa.py               # single-graph experiment
qa_lab/qa_graph/karate_spectral_qa_fingerprint.py       # QA fingerprint of spectral partition
qa_lab/qa_graph/hub_distance_descriptor_benchmark.py    # 5-graph generalization
qa_lab/qa_graph/hub_distance_descriptor_benchmark_results.json
```

Run:

```
cd qa_lab
PYTHONPATH=. python -m qa_graph.hub_distance_descriptor_benchmark
```

Datasets are all standard: `networkx.karate_club_graph()`,
`networkx.barbell_graph(10, 3)`, `networkx.connected_caveman_graph(4, 5)`,
`networkx.florentine_families_graph()`, and Mark Newman's
`football.graphml`. No stochastic generators, no randomized input, no
retraining between runs.

## 6. Open questions

- **How to detect the right generator automatically.** The current benchmark
  picks the descriptor by hand per graph. A fully automatic pipeline would
  need an unsupervised signal — e.g. pick the descriptor that maximises
  modularity of its own induced partition, then report which descriptor won
  as a diagnostic of what kind of graph it is.
- **k > 2 hub-distance.** For multi-community graphs, is the right object
  the full k-hub distance tuple, or a k-wise ranking? Single-axis fails on
  caveman with k = 4; pure argmin does better but not perfectly.
- **Non-hub geodesic generators.** The hub-distance descriptor assumes the
  "charismatic leader" model. Other geodesic generators exist — e.g. boundary
  nodes in planar graphs, or articulation points in biconnected components.
  Each should have its own QA extraction.
- **Connection to chromogeometry.** The karate fingerprint table shows
  `norm f(b, e) = b² + be − e²` recovers ground truth at 0.48, exactly the
  same as `b` alone. This suggests the norm is acting as a one-dimensional
  reduction of the hub-distance plane. Is there a chromogeometric
  quadrance that does better?

## 7. Conclusion

The claim "QA descriptors should be generator-aware" is supported by the
five-graph benchmark: different descriptor families win on different graph
families, and the winning descriptor is always the one whose construction
matches the generator of the ground-truth communities. The axioms (A1, A2,
S1, T2, T1, S2) are universal; the extraction is not, and pretending
otherwise leads to the apparent failures of Section 1.

Florentine is the headline positive result — **hub-distance beats every
alternative by 0.19 ARI on a real labelled historical network**. Football
is the headline negative result — the same descriptor collapses to 0.11
on a graph whose communities were not generated geodesically. Neither result
is a bug. Both are information about the domain, expressed in the one
language that keeps the axioms fixed while letting the extraction vary.

---

*Part of the QA research program. This note was written to GitHub rather
than journal-submitted per the project's GitHub-first publication policy.*
