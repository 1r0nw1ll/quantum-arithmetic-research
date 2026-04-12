# Family [212] QA_FIBONACCI_HYPERGRAPH_CERT.v1

## One-line summary

Every QA state (b,e) defines a length-4 Fibonacci window hyperedge (b, e, d, a). The resulting state-residue incidence hypergraph H(m) satisfies three structural theorems on S_9: (1) the QA dynamic T acts as a 1-step Fibonacci window slide, (2) every residue has degree exactly 4m = 36, (3) cosmos/satellite/singularity orbits collapse to (22, 22, 22, 4, 1) distinct multiset hyperedges.

## Mathematical content

### The QA Fibonacci hypergraph H(m)

Let S_m = {(b,e) : 1 ≤ b,e ≤ m} and V = {1, ..., m}. For each state (b,e) ∈ S_m define the **Fibonacci window hyperedge**

```
h(b,e) = (b, e, d, a)     where d = b+e, a = b+2e     (both mod m, A1-adjusted)
```

The state-residue incidence hypergraph H(m) has vertex set V and hyperedge set E = {h(b,e) : (b,e) ∈ S_m}, treated either as ordered tuples (m² distinct hyperedges) or as multisets (fewer, due to collisions).

### Theorem 1 (Sliding Window)

For every s = (b,e) ∈ S_m,

```
h(T(s)) = (e, d, a, (d + a) mod m)
```

Equivalently: T acts on hyperedges as the 1-step Fibonacci slide
`(F_0, F_1, F_2, F_3) → (F_1, F_2, F_3, F_4)` where `F_4 = F_2 + F_3`. Exhaustively verified on S_9: **81/81** states satisfy the formula.

*Proof*: Let T(s) = (e, b+e). Then d_{T(s)} = e + (b+e) = b+2e = a_s, and a_{T(s)} = e + 2(b+e) = 2b + 3e = (b+e) + (b+2e) = d_s + a_s. So h(T(s)) = (e, d_s, a_s, d_s + a_s). ∎

### Theorem 2 (Uniform Vertex Degree)

Every vertex v ∈ V has degree exactly **4m** in H(m). On S_9 this means every residue appears in exactly **36** hyperedges; total degree 4·81 = **324**.

*Proof*: Each of the 4 hyperedge positions (b, e, d, a) takes value v for exactly m of the m² states in S_m — one equation per slot, one solution per b or per e. Summing contributions across positions: 4m per vertex, independent of v. ∎

### Theorem 3 (Orbit-Multiset Collapse on S_9)

The five T-orbits on S_9 (sizes 24, 24, 24, 8, 1 per [191]) produce distinct multiset hyperedge counts (22, 22, 22, 4, 1):

| Orbit | Rep | Length | Family | Distinct multisets |
|-------|-----|--------|--------|---------------------|
| 0 | (1,1) | 24 | cosmos | 22 |
| 1 | (1,3) | 24 | cosmos | 22 |
| 2 | (1,4) | 24 | cosmos | 22 |
| 3 | (3,3) | 8 | satellite | 4 |
| 4 | (9,9) | 1 | singularity | 1 |

Cosmos orbits have 2 multiset collisions each (out of 24). The satellite orbit has a 2-to-1 collapse (8 states → 4 multisets) reflecting its internal period-2 symmetry. The singularity is the single multiset {9,9,9,9}.

## Checks

| ID | Description |
|----|-------------|
| HGR_1     | schema_version == 'QA_FIBONACCI_HYPERGRAPH_CERT.v1' |
| HGR_SLIDE | sliding window theorem 81/81 on S_9 (independently recomputed) |
| HGR_DEG   | vertex degree uniform at 4m = 36, total 324 (independently recomputed) |
| HGR_ORB   | orbit-multiset distribution (22, 22, 22, 4, 1) (independently recomputed) |
| HGR_FIB   | fibonacci_recurrence field declares F_{k+1} = F_{k-1} + F_k form |
| HGR_191   | cross-reference to family 191 present |
| HGR_SRC   | source attribution to Fibonacci and Berge present |
| HGR_WIT   | ≥ 3 witnesses (cosmos, satellite, singularity) |
| HGR_F     | fail_ledger well-formed |

## Source grounding

- **Fibonacci (Leonardo of Pisa)**, *Liber Abaci* (1202) — original recurrence F_{n+1} = F_n + F_{n-1}
- **Edouard Lucas**, "Théorie des Fonctions Numériques Simplement Périodiques" (*American Journal of Mathematics*, 1878) — periods of F_n mod m
- **D. D. Wall**, "Fibonacci Series Modulo m" (*American Mathematical Monthly* 67, 1960, pp. 525–532) — Pisano period function π(m)
- **Claude Berge**, *Hypergraphs: Combinatorics of Finite Sets* (North-Holland, 1989) — hypergraph theory foundations
- Prerequisite: family [191] `qa_bateson_learning_levels_cert` — orbit classification
- Related: [192] `qa_dual_extremality_24_cert` (orbit length = Pisano period)
- Related: [211] `qa_cayley_bateson_filtration_cert` (same T, graph view)
- Verification module: `qa_lab/qa_graph/hypergraph.py :: verify_all_theorems(9)`

## Connection to other families

- **[191]** orbit classification underpins the T-orbit partition used in Theorem 3
- **[192]** dual extremality: orbit length = Pisano period, 24 is the maximum on S_9
- **[211]** Cayley view of T — the structural companion to this dynamical view
- **[130]** origin of 24 — cosmos/satellite/singularity classification determines multiset collapse
- **[150]** (ℤ/9ℤ)* acts transitively on residues, supporting the vertex degree uniformity argument

## Fixture files

- `fixtures/hgr_pass_hypergraph.json` — PASS: declares all three theorems with correct constants; validator independently recomputes on S_9 and verifies equality
- `fixtures/hgr_fail_bad_degree.json` — FAIL: declares per_vertex=35 (wrong; correct is 36) and omits witnesses; validator must flag HGR_DEG and HGR_WIT
