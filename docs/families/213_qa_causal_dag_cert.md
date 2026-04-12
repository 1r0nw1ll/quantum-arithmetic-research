# Family [213] QA_CAUSAL_DAG_CERT.v1

## One-line summary

The A2 axiom (`d = b+e`, `a = b+2e`) IS the structural equation system of a 4-node Y-structure causal DAG with b, e exogenous and d, a endogenous colliders. Pair-invertibility theorem: all 6 unordered pairs bijective iff `gcd(2, m) = 1`; on S_9 all 6 bijective, on S_24 exactly the pair (b, a) is 2-to-1. Pearl's three causal hierarchy levels collapse because the SCM is deterministic — this is the SCM form of Theorem NT.

## Mathematical content

### The 4-node Y-structure DAG

```
          b             e
          |\           /|
          | \         / |
          |  \       /  |
          |   \     /   |
          |    \   /    |
          |     \ /     |
          |      X      |
          |     / \     |
          |    /   \    |
          |   /     \   |
          |  /       \  |
          | /         \ |
          |/           \|
          d             a
```

**Structure**: 4 nodes {b, e, d, a}, 4 directed edges {b→d, e→d, b→a, e→a}. Both d and a are colliders (two incoming arrows). Both are leaves (no children). b and e are exogenous (no parents). The graph is acyclic — topological order: b, e, d, a.

**Structural equations** (= A2 axiom):
```
d := b + e
a := b + 2e
```

### Theorem 1 (Pair Invertibility)

**Statement**: For modulus m, all 6 unordered pairs of {b, e, d, a} are bijective as maps from S_m = {1..m}² to their image iff `gcd(2, m) = 1`. When `gcd(2, m) = g > 1`, exactly the pair (b, a) is g-to-1; the other 5 pairs remain bijective.

**Proof sketch**: Given any 2 of (b, e, d, a), solve the 2×2 linear system from A2 over ℤ/mℤ.
- Pairs (b,e), (b,d), (e,d), (e,a), (d,a) have full-rank integer-matrix solves (determinant ±1 over ℤ): trivially invertible mod any m.
- Pair (b, a) requires solving `2e = a − b (mod m)`. This has a unique solution iff 2 is invertible mod m ⟺ gcd(2, m) = 1.
- When gcd(2, m) = g > 1: if (a − b) is divisible by g, there are exactly g solutions; otherwise none. Equivalently, the projection S_m → {(b,a)} is g-to-1 onto its image.

**Verified exhaustively**:

| Modulus | gcd(2, m) | (b,e) | (b,d) | (b,a) | (e,d) | (e,a) | (d,a) |
|---------|-----------|-------|-------|-------|-------|-------|-------|
| **S_9** (gcd=1) | 1 | 81/81 ✓ | 81/81 ✓ | 81/81 ✓ | 81/81 ✓ | 81/81 ✓ | 81/81 ✓ |
| **S_24** (gcd=2) | 2 | 576/576 ✓ | 576/576 ✓ | **288/576 (2-to-1)** | 576/576 ✓ | 576/576 ✓ | 576/576 ✓ |

### Theorem 2 (Pearl-Level Collapse)

**Statement**: Because the A2 structural equations are deterministic integer arithmetic (no stochastic noise term), Pearl's three causal hierarchy levels collapse to the same answer:

```
Level 1 (association):    P(d | b=b*, e=e*)          = δ(d − b* − e*)
Level 2 (intervention):   P(d | do(b=b*), e=e*)      = δ(d − b* − e*)
Level 3 (counterfactual): P(d_{b=b*} | b=b', e=e')   = δ(d − b* − e')
```

All three deliver identical values because there is no noise variable to marginalize over. This is a **degenerate but valid** SCM — it says the causal structure lives entirely in the discrete A2 identities.

### Theorem NT correspondence

Pearl-level collapse IS the SCM form of Theorem NT (observer projection firewall). The continuous measurement layer (e.g., `np.correlate(b, d)` over a time series) is Pearl Level 1 — observational association. It can NEVER enter the SCM as a causal input because the SCM is closed under A2. Theorem NT is the assertion that Level 1 (observer projection) cannot cross the discrete boundary; Levels 2 and 3 are internal to the discrete QA layer.

## Checks

| ID | Description |
|----|-------------|
| CDG_1     | schema_version == 'QA_CAUSAL_DAG_CERT.v1' |
| CDG_STRUCT | 4-node Y-DAG declared (nodes, edges, exogenous, colliders) |
| CDG_A2    | structural equations `d = b+e`, `a = b+2e` declared |
| CDG_PAIRS | pair-bijectivity table matches theorem on S_9 (all 6) and S_24 (5 of 6, (b,a) 2-to-1); recomputed independently |
| CDG_PEARL | Pearl-level collapse declared with deterministic-SCM reason |
| CDG_NT    | Theorem NT correspondence field references observer projection |
| CDG_191   | cross-reference to family 191 present |
| CDG_SRC   | source attribution to Judea Pearl present |
| CDG_WIT   | ≥ 4 witnesses (one per pair class) |
| CDG_F     | fail_ledger well-formed |

## Source grounding

- **Judea Pearl**, *Causality: Models, Reasoning, and Inference* (Cambridge University Press, 2nd ed. 2009) — structural causal models, do-calculus, three-level causal hierarchy
- **Sewall Wright**, "Correlation and Causation" (*Journal of Agricultural Research* 20, 1921, pp. 557–585) — original path analysis; direct ancestor of modern SCM
- Prerequisite: family [191] `qa_bateson_learning_levels_cert` — the Bateson filtration is the operator-level causal hierarchy
- Related: [150] `qa_septenary_unit_group` — 2 ∈ (ℤ/9ℤ)* (cyclic order 6, 2⁻¹ ≡ 5) is the exact reason pair (b,a) is bijective on S_9
- Related: [202] `qa_hebrew_mod9_identity` — Aiq Bekar dr supplies the A1 adjustment
- Verification module: `qa_lab/qa_graph/causal_dag.py`

## Connection to other families

- **[191]** Bateson Learning Levels — provides the operator-class filtration analogous to Pearl's causal hierarchy on QA dynamics
- **[150]** (ℤ/9ℤ)* unit group — unit-ness of 2 mod m is the gating condition for pair (b,a) bijectivity
- **[211]** Cayley-Bateson — same T dynamic as generator graph (structural view)
- **[212]** Fibonacci Hypergraph — same 4-tuple as Fibonacci window (dynamical view); causal DAG is the SCM view of the same object

Slots 1-4 of the graph types initiative (Cayley / Hypergraph / KG / Causal DAG) are **four duals of the same underlying QA 4-tuple**: structural components, sliding window, labelled multigraph, structural equations. Each view exposes a different property of the same arithmetic.

## Fixture files

- `fixtures/cdg_pass_y_structure.json` — PASS: declares full Y-structure, both pair-bijectivity tables, Pearl collapse, 4 witnesses across pair classes; validator independently recomputes pair-bijectivity on both S_9 and S_24
- `fixtures/cdg_fail_bad_structure.json` — FAIL: declares `a = b + 3e` (wrong) and empty exogenous list; validator must flag CDG_A2 and CDG_STRUCT
