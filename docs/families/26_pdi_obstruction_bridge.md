# PDI–Obstruction Bridge
## Formal Connection Between Path Diversity and Reachability Obstruction Classes

**Family [26] · Supplement v1.0.0 · 2026-03-01**
*Cross-references: Family [76] Failure Algebra, Family [86] Generator-Failure Unification*

---

### 1  Setup

Let a competency-certified system be represented as a **generator graph**
G = (V, E) where:

- V is the set of all states (total_states)
- E contains a directed edge (u, v, g) whenever generator g maps state u to
  state v
- I ⊆ V is the set of initial states
- R ⊆ V is the set of states reachable from I under any sequence of generators
- M ⊆ R is the set of states reachable via ≥ 2 distinct directed paths from I
  (counted on the SCC-condensation DAG of G)

The **Path Diversity Index** is PDI = |M| / |R|.

An **obstruction** O ⊆ E is a named set of edges removed from G by a structural
or operational barrier (e.g., resource exhaustion, invariant violation, domain
boundary).  The residual graph after applying O is G_O = (V, E \ O).

---

### 2  The Failure Algebra Carrier (Family [76])

The QA Failure Algebra Structure Cert defines a bounded join-semilattice over
a carrier of failure modes, with partial order:

```
        OUT_OF_DOMAIN          ← top (total blocking)
       /             \
 PARITY_BLOCK   INVARIANT_VIOLATION
       \             /
            OK                 ← bottom / unit (no failure)
```

Each generator g receives a **failure tag** τ(g) ∈ carrier that names the
worst-case failure mode g can produce.  Formally:

- τ(g) = **OK** — g succeeds in all reachable states; its edges are always
  present in G.
- τ(g) = **PARITY_BLOCK** — g is blocked in states where a parity constraint
  is active; its edges are conditionally absent.
- τ(g) = **INVARIANT_VIOLATION** — g is blocked where an invariant is
  breached; its edges are conditionally absent.
- τ(g) = **OUT_OF_DOMAIN** — g is structurally out of scope; its edges are
  permanently absent from G (or absent in a named subdomain).

The Family [86] generator-failure unification cert (cross-binding [76] + [80])
provides the concrete tagging for CAPS_TR systems:

| Generator      | Family tag | Failure tag          |
|----------------|------------|----------------------|
| fear_up        | fear       | PARITY_BLOCK         |
| fear_down      | fear       | OK                   |
| fear_lock      | fear       | INVARIANT_VIOLATION  |
| love_soothe    | love       | OK                   |
| love_support   | love       | OK                   |
| love_reframe   | love       | OK                   |

---

### 3  Obstruction Type Classification

**Definition.**  Let O be an obstruction and G_O the residual graph.
Define R_O = R(G_O) and M_O = M(G_O).  Obstruction O belongs to:

| Type | Condition | PDI effect |
|------|-----------|------------|
| **I — State-Annihilating** | R_O ⊊ R | Ambiguous: R shrinks; M shrinks by at least as much or more |
| **II — Route-Reducing** | R_O = R, M_O ⊊ M | PDI strictly decreases |
| **III — Topology-Preserving** | R_O = R, M_O = M | PDI unchanged |

*Note*: Type I obstructions can in principle increase PDI if they remove
low-diversity states from R while leaving M relatively intact, but this
requires a specially adversarial graph structure; for tree-spine graphs
(typical of reference exemplars) Type I obstructions decrease PDI.

**Completeness.**  Every obstruction O belongs to exactly one of Types I–III
(the conditions partition all (R_O, M_O) outcomes).

---

### 4  Bridge Theorem: Failure Tag → Obstruction Type

**Theorem B1 (Failure-Tag Obstruction Typing).**
*Let g be a generator with failure tag τ(g).  In the worst case:*

- τ(g) = OK → g produces no obstruction (Type III vacuously).
- τ(g) = PARITY_BLOCK → g produces a Type I or Type II obstruction depending
  on whether its blocked edges are the sole paths to some states (Type I) or
  merely redundant merge edges (Type II).
- τ(g) = INVARIANT_VIOLATION → same case split as PARITY_BLOCK; the
  distinction is semantic (parity vs. invariant boundary), not topological.
- τ(g) = OUT_OF_DOMAIN → g produces a Type I obstruction (permanent edge
  removal; affected states may become unreachable).

*Proof sketch.*  τ(g) = OK implies g's edges are always present, so O_g = ∅
and the residual graph is G itself (Type III).  For τ(g) ≥ PARITY_BLOCK, the
edges of g are absent in at least one state.  Whether any state in R loses
all its incoming paths (Type I) or only loses a redundant path (Type II)
depends on the graph topology, not the failure tag alone.  τ(g) = OUT_OF_DOMAIN
is definitionally a permanent removal; if g was the sole incoming generator for
any reachable state, that state falls out of R (Type I).  ∎

**Corollary B2 (OK-Generator PDI Safety).**
*If all generators with τ(g) ≠ OK are non-critical — i.e., every state
reachable via a non-OK generator is also reachable via a path using only
OK-tagged generators — then all active obstructions are Type III and PDI is
preserved.*

---

### 5  PDI-Criticality of Generators

**Definition.**  Generator g is **PDI-critical** if removing all edges of g
from G strictly decreases PDI(G_O) < PDI(G).

**Theorem B3 (Criticality Characterization).**
*Generator g is PDI-critical iff at least one edge (u, v, g) satisfies:*

1. *v ∈ M (v is a multi-path state in the baseline graph), and*
2. *removing (u, v, g) reduces the path count to v from ≥ 2 to exactly 1
   (so v exits M), and*
3. *no alternative incoming edge to v's SCC from a different generator
   preserves path count.*

*Equivalently: g is PDI-critical iff it is the sole provider of at least one
"merge edge" — an edge whose removal disconnects a merge point in the
condensation DAG.*

**Corollary B4.**
*A generator with τ(g) = OK can still be PDI-critical (if its edges provide
the only merge paths).  PDI-criticality is a graph property, not a failure-tag
property.  The failure tag only determines when criticality can be triggered.*

---

### 6  Effective Agency Preservation

**Theorem B5 (EA-Preservation Condition).**
*Effective Agency EA = AI × PDI is preserved under an obstruction O iff:*

1. *O is Type III (M_O = M, R_O = R), and*
2. *the agency index AI is computed over the same reachability region.*

*When O is Type I or II, EA degrades.  The degree of degradation is:*

$$\Delta EA(O) \;=\; AI \cdot \left(\frac{|M_O|}{|R_O|} - \frac{|M|}{|R|}\right)$$

*For Type II (R unchanged): $\Delta EA = AI \cdot (|M_O| - |M|) / |R| \leq 0$.*

*For Type I: both numerator and denominator change; ΔEA can be positive only
if low-diversity states are removed disproportionately.*

---

### 7  Reference Set Obstruction Analysis

Applying the classification to the 16 named obstructions across 9 reference
systems:

| Obstruction | System | Likely Type | Mechanism |
|---|---|---|---|
| `context_overflow` | tool_agent_debugger | I | Truncates reachable future states (context window is a hard state boundary) |
| `consensus_deadlock` | multi_agent_coordination | I | Locks the system into an absorbing state; downstream states unreachable |
| `message_loss` | multi_agent_coordination | II | Removes inter-agent coordination edges; some merge paths collapse |
| `empty_retrieval` | retrieval_agent_arag | II | Removes retrieval-success branches; surviving paths through fallback only |
| `contradictory_sources` | retrieval_agent_arag | II | Collapses multi-source merge points to single-source path |
| `nutrient_depletion` | organoid_self_repair | I | Eliminates repair-trajectory states below viability threshold |
| `necrotic_core_formation` | organoid_self_repair | I | Permanently removes core cell states; absorbing dead region |
| `energy_exhaustion` | planarian_regen | I | State-annihilating; regeneration halts |
| `incomplete_regeneration` | planarian_regen | II | Partial paths survive; merge points at late-stage patterning lost |
| `lethal_voltage_threshold` | xenopus_patterning | I | Hard absorbing state; downstream pattern states unreachable |
| `stage_restriction` | xenopus_patterning | II | Removes stage-transition edges; some parallel developmental paths collapse |
| `electrode_drift` | bioelectric_controller_with_llm | II | Degrades signal fidelity; merge points in closed-loop control collapse |
| `llm_hallucination` | bioelectric_controller_with_llm | II | Introduces spurious edges (net: route-reduction in the certified path set) |
| `human_timeout` | human_in_the_loop_protocol | I | Absorbs into timeout state; downstream consent states unreachable |
| `preference_reversal` | human_in_the_loop_protocol | II | Removes consistency-guaranteed merge edges; single surviving preference path |
| `consent_withdrawal` | human_in_the_loop_protocol | I | Terminal; all post-consent states become unreachable |
| `sample_contamination` | lab_robot_closed_loop | I | Contaminates run state; loop restart loses branching history |
| `tip_clog` | lab_robot_closed_loop | II | Removes dispensing-path edges; aspiration-only path survives |

**Summary:** 8 Type I, 9 Type II, 0 Type III in the reference set.  The absence
of Type III obstructions confirms that all named obstructions in certified
exemplars have measurable PDI impact — there are no "free" obstructions that
leave path diversity intact.

**Failure-tag correspondence (CAPS_TR example):**

| Generator failure tag | Corresponding obstruction type | Reference obstruction examples |
|---|---|---|
| OK | III (vacuous) | — |
| PARITY_BLOCK | II (conditional) | `message_loss`, `stage_restriction`, `tip_clog` |
| INVARIANT_VIOLATION | II or I | `electrode_drift`, `preference_reversal`, `energy_exhaustion` |
| OUT_OF_DOMAIN | I (permanent) | `lethal_voltage_threshold`, `necrotic_core_formation` |

---

### 8  Design Implication: The Merge-Engineer's Rule

Combining B1–B5 with the empirical finding that all 9 reference systems sit in
the PDI < 0.5 band yields a concrete design criterion:

> **To achieve PDI > 0.5, a system must have ≥ 2 independent OK-tagged
> generators that each provide a merge edge to every target state in M.**

"Independent" means: the two generators must traverse different intermediate
states (no shared dependency chain), so that blocking one does not block the
other.

In failure-algebra terms: the two merge-path generators must have
**join(τ(g₁), τ(g₂)) = OK** — i.e., both must be tagged OK.  Any pair where
join ≥ PARITY_BLOCK risks collapsing to a single route under the shared
blocking condition.

For the CAPS_TR domain: the love-family generators (all OK-tagged) are the
natural merge-path providers.  The fear-family (PARITY_BLOCK, INVARIANT_VIOLATION)
provides escalation routes but not merge-stable redundancy.

---

### 9  Connection to T2 / T3 (Family [86])

Theorems T2 and T3 in the generator-failure unification cert bound the SCC
structure and path propagation counts under the failure algebra.  The bridge
above extends these results to PDI:

- **T2 (SCC bounds)** gives the maximum SCC size under a generator set G; the
  SCC Saturation Lemma (Family [26] formal theory §15) then bounds |M| from
  above as a function of SCC entry route count.
- **T3 (Path propagation)** tracks path counts in topological order; combined
  with Theorem B3, T3's per-node path count directly determines which states
  enter M — giving a computable PDI certificate from the T3 witness table.

This establishes a **certification chain**:

```
Family [76] failure algebra
        ↓ generator tagging (Family [86])
Family [86] T2 SCC bounds + T3 path propagation
        ↓ merge-edge identification (Theorem B3)
Family [26] PDI (this document, Theorem B1–B5)
        ↓
Effective Agency EA = AI × PDI
```

No additional runtime computation is required beyond what T3 already produces.
PDI is derivable from the T3 witness without re-running BFS.

---

### References

- Family [26] formal theory: `docs/families/26_competency_detection_pdi_formal_theory.md`
- Family [76] failure algebra schema: `qa_failure_algebra_structure_cert_v1/schema.json`
- Family [86] generator-failure unification: `qa_generator_failure_unification_cert_v1/`
- CAPS_TR generator tagging: `qa_generator_failure_unification_cert_v1/fixtures/valid_caps_tr_fear_love.json`
- Reference set bundles: `qa_competency/reference_sets/v1/`
- Tags: `family-26-pdi-v1.0.0`, `family-86-energy-drift-v1` (implicit)
