# QA Geodesy — Synthesis & Navigation

**Status:** synthesis / navigation layer. **No new axioms. No new theorems. No new cert family.**
This document names a layer that already exists in the repo and points to the certs and theory docs that prove its claims. It is intentionally not a cert and contains no falsifiable assertions of its own.

If you're looking for a cert with a falsifiable claim about minimal-path structure, the right one is `[265]` qa_counterfactual_descent. The work below is consolidation, not addition.

**Companion files (primary references):** `docs/theory/QA_GENERATOR_REACHABILITY.md`, `docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md`, `docs/theory/QA_ORBIT_THEOREM_SYNTHESIS.md`, `docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md`, `docs/families/40_reachability_descent_run_cert.md`, `docs/families/265_qa_counterfactual_descent_cert.md`.

---

## What "QA geodesy" names

The study of **minimal legal generator paths on QA state graphs**.

Given:

- a finite QA state space `S = (Z/mZ)²` (or `Z²` when modular reduction is absent),
- a declared generator set `Γ ⊆ {σ (qa_step), μ_k (scalar_mult), ν (modulus_reduction), swap, const_(9,9), …}`,

a **QA geodesic** from `s` to `t` is a shortest sequence `g₁,…,gₙ ∈ Γ` with `gₙ∘…∘g₁(s)=t`. Distance:

```
d_Γ(s,t) = min n  if reachable,  ∞ otherwise.
```

The framing is **generator-relative**: distance, curvature, and obstruction are properties of `(S, Γ)`, not of states alone. This is the same generator-relative geometry developed in `QA_GENERATOR_REACHABILITY.md` and certified in [191] / [211].

### Naming clarification (read first)

There is already a cert family **`[168]` qa_ellipsoid_geodesic_cert_v1**, which certifies *continuous* geodesic properties of the WGS84 ellipsoid expressed in QA quantum-number arithmetic (`M/N = F/(d²-e²·s_φ)`, etc.). That cert is on the **observer-projection side** of Theorem NT — geodesics on a continuous surface, recovered through QN identities.

"QA geodesy" in this document refers to **discrete generator-path geometry on QA state graphs** — the QA-discrete side of Theorem NT. The two are distinct:

| Layer | Object | Cert anchor |
|---|---|---|
| QA-discrete (this doc) | Generator paths on `S_m` orbit graph | [40], [191], [211], [263], [265] |
| Observer projection ([168]) | Continuous geodesics on WGS84 surface | [168] (and [164] gnomonic, [166] loxodrome as projection-side path certs) |

Calling either layer "geodesy" alone is ambiguous. Use the qualifier when it matters.

---

## Prior art (what already exists)

### Generator-set component theory

| Concept | Anchor |
|---|---|
| Generator-relative reachability, CRT/p-adic stratification | `docs/theory/QA_GENERATOR_REACHABILITY.md` |
| Strict-filtration tier hierarchy `L_0 ⊂ L_1 ⊂ L_2a ⊂ L_2b ⊂ L_3` on `S_9`; 26 % of pairs L_1-reachable | `[191]` qa_bateson_learning_levels |
| Tiered classes are **exactly** connected components of nested undirected Cayley graphs `Γ_L1={T}`, `Γ_L2a`, `Γ_L2b` → component sizes (24,24,24,8,1)/(72,8,1)/(81) | `[211]` qa_cayley_bateson_filtration |
| Two-layer orbit stratification: `⟨σ,μ⟩`-orbits = content-ideal classes; `σ`-only refines by Legendre `(5\|p)`. `μ` is the collapse from Part II to Part I | `qa_orbit_stratification_cert_v1/` (family [261, pending registration]); `docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md`; `QA_ORBIT_THEOREM_SYNTHESIS.md` |
| Sliding-window incidence on Fibonacci hyperedges `(b,e,d,a)`; uniform vertex degree `4m` | `[212]` qa_fibonacci_hypergraph |
| A2 identities `d=b+e`, `a=b+2e` are the SCM of a 4-node Y-structure DAG | `[213]` qa_causal_dag |
| Signed cosmos orbits via Eisenstein form `f(b,e)=b²+be−e²`; `T²` preserves `f` mod m | `[214]` qa_norm_flip_signed_temporal |

### Distance, descent, and shortest paths

| Concept | Anchor |
|---|---|
| Deterministic greedy descent run on `Caps(N,N)` with policy `GREEDY_MIN_ENERGY_TIEBREAK_MOVE_NAME`, generators `{σ, μ, λ_k, ν}`, objective `L2_TO_TARGET`; 5-gate validator | `[40]` qa_reachability_descent_run_cert (`qa_reachability_descent_run_cert_v1/validator.py`) |
| Bounded-depth BFS reachability (`min_steps_within_k`) | `qa_reachability_descent_run_cert_v1/validator.py` |
| **Shortest-legal-generator-paths = exact counterfactuals.** BFS to first state where Boolean spec predicate flips; path length = counterfactual distance. Generators: `σ` (L_1), `μ_2` (L_2a), `μ_3` (L_2b) | `[265]` qa_counterfactual_descent (`docs/families/265_qa_counterfactual_descent_cert.md`) |
| Łojasiewicz-style monotone descent on orbit-energy | `[102]`, `[103]` qa_lojasiewicz_orbit |
| Path-shape classification (UNIFORM_A / UNIFORM_B / UNIFORM_C / MIXED) on Pythagorean tree | `[145]` qa_path_shape |
| Path-scale growth profiles (`G` ratio → 3+2√2 along UNIFORM_B) | `[146]` qa_path_scale |
| `T`-operator = Fibonacci shift `F=[[0,1],[1,1]]`, multiplication by φ in `Z[√5]/m`; orbit period = `ord(F)` in `GL₂(Z/mZ)` | `[126]` qa_red_group |
| Cosmos period = Pisano `π(m)`; `π(9)=24`, `π(7)=16` | `[128]` qa_spread_period |
| Joint extremality `π(24)=24`, `λ(24)=2`; `π(9)=24` bridges theoretical→applied modulus | `[192]` qa_dual_extremality_24 |

### Curvature and stability

| Concept | Anchor |
|---|---|
| Per-state stability margin `κ(t) = 1−|1−η_eff(t)|`; orbit certificate pins `κ_min = min_t κ(t)` as multi-step bottleneck | `[97]` qa_orbit_curvature |
| Branching/bottleneck structure of nested Cayley graphs (component counts at each level) | `[211]` |

### Obstruction and infinite distance

| Concept | Anchor |
|---|---|
| Native vs representation vs physical layer obstructions; representation debt ≠ physical-device failure | `[129]` qa_projection_obstruction |
| **Failure-density enumeration.** `p_fail = \|{s ∉ ψ}\| / \|S_m\|` exact; head-to-head against Kochenderfer 7.1 direct sampling with `\|err\| ≤ 4σ` envelope | `[263]` qa_failure_density_enumeration |
| Failure-algebra structure (atomic obstructions and their classification) | `[76]`, `[88]` |
| Generator-failure unification | `[86]` |
| Failure-compose operator (composition of obstructions) | `[87]` |
| Sympathetic oscillation = orbit co-membership; **discord = reachability obstruction**; triad concordance | `[185]` qa_keely_sympathetic_transfer |
| Cognitive light-cone radius = orbit radius (Singularity 0 / Satellite 8 / Cosmos 24); cancer = orbit shrinkage Cosmos→Satellite; 26 % structural CLC ceiling from [191] | `[193]` qa_levin_cognitive_lightcone |
| Morphospace voids are **algebraically necessary** (not unobserved); agency = `\|reachable\|/\|total\|` ∈ {1/81, 8/81, 72/81} | `[194]` qa_cognition_space_morphospace |

### Continuous projection of QA paths (observer-side)

These are *not* QA geodesics in this document's sense; they are continuous-surface geodesic images of QA-discrete `T`-walks, and live on the observer-projection side of Theorem NT. Listed for completeness because they are often confused with the discrete layer.

| Concept | Anchor |
|---|---|
| WGS84 ellipsoid `M/N` curvature ratio, axis ratio, eccentricity resonance in QN form | `[168]` qa_ellipsoid_geodesic |
| Gnomonic projection: great circles → straight lines; Berggren tree generators = discrete geodesic steps on cone | `[164]` qa_gnomonic_rt |
| Loxodrome (rhumb line) as `T`-operator constant-bearing path; period = `π(m)`; orbit-typed | `[166]` qa_loxodrome |
| Dead reckoning as exact `T`-operator iteration on mod-`m` lattice; zero drift | `[163]` qa_dead_reckoning |

### Theory docs

- `docs/theory/QA_GENERATOR_REACHABILITY.md` — primary generator-relative reachability theory (CRT factoring, content-ideal stratification, generator audits).
- `docs/theory/QA_ORBIT_STRATIFICATION_THEOREM.md` — proofs for the stratification cert.
- `docs/theory/QA_ORBIT_THEOREM_SYNTHESIS.md` — companion synthesis.
- `docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md` — sketch behind [191].
- `docs/families/40_reachability_descent_run_cert.md`, `…/265_qa_counterfactual_descent_cert.md` — closest existing cert-family writeups to the geodesy frame.

---

## What is already certified (no new claims)

Restating only results that have a cert anchor:

1. **Reachability classes are graph components.** ([211]) The `L_1`/`L_2a`/`L_2b` tiers of [191] are the connected components of the corresponding nested Cayley graph on `S_9` under the declared generator sets.
2. **Counterfactual distance = shortest-path BFS depth.** ([265]) For Boolean orbit-class specifications on `S_9` with declared generators `{σ, μ_2, μ_3}`, the minimum counterfactual is exactly the first BFS-depth at which the spec flips.
3. **Failure density is exact-enumerable on finite `S_m`.** ([263]) Variance is identically zero; corresponding Kochenderfer 7.1 direct-sampling estimator falls inside the `4σ` envelope at `N ∈ {100, 1000, 10000}`.
4. **Greedy descent on `Caps(N,N)` is well-defined and gate-checkable.** ([40]) Under `GREEDY_MIN_ENERGY_TIEBREAK_MOVE_NAME`, generator legality, monotonic improvement, and tiebreak determinism are validated per-step.
5. **Curvature has a stability reading.** ([97]) The orbit-wide minimum `κ_min` is the multi-step stability bottleneck; the cert exhausts the orbit and pins the minimum.
6. **Closed-orbit length = Pisano period.** ([126], [128], [192]) The cosmos period of the `T`-operator on `S_m` equals `π(m)`; this is `ord(F)` in `GL₂(Z/mZ)`.

Each of these has a corresponding cert validator with PASS/FAIL fixtures. None is restated or generalized here.

---

## Intentional non-claims

To keep this a navigation doc, the following are **not** asserted:

- No claim that "QA geodesy" is a single canonical geometry. It is a generator-relative graph layer; geometry depends on `Γ`.
- No weighted-geodesic theorem. A cost-per-generator Dijkstra reformulation is not currently certified.
- No curvature invariant beyond the existing `κ_min` ([97]). A scalar branching/SCC-curvature with stated bounds and falsifiable test is **future work**, not consolidated here.
- No generator-set classification theorem. [211] gives the component decomposition for the specific `Γ_L1 / Γ_L2a / Γ_L2b` ladder; a general taxonomy of generator subsets → reachability-class signature does not yet exist as a cert.
- No claim that cert family `[265]` "is" QA geodesy. It is the closest existing cert; this doc is a frame around it, not a rebranding.

---

## Open candidates for future cert work

These are cert-shaped only if the falsifiable claim is genuinely new (not a wrapper on [40]/[191]/[211]/[263]/[265]):

1. **Weighted-geodesic certification.** Declared cost function `w : Γ → ℕ⁺` with `w ≡ 1` reducing to BFS. Validator recomputes Dijkstra and rejects any non-minimal weighted path or any obstruction that is not closed-component independent of `w`. **Falsifier:** any declared minimum-weight path that Dijkstra beats.
2. **Generator-set classification theorem.** A discrete taxonomy `{Γ_i}` → reachability-class signature, with a lemma not implied by [191]+[211]. **Falsifier:** a declared signature that disagrees with exhaustive component enumeration.
3. **Branching-curvature scalar.** A scalar `κ_branch` defined on the orbit graph (e.g. local in/out-degree variance over BFS frontier) with a stated upper or lower bound across `S_9` orbits. **Falsifier:** a state violating the bound.
4. **Geodesic-compression identity.** A claim that minimum-description-length encoding `start ⊕ Γ-sequence` beats a declared baseline by a stated margin under a stated metric on a stated corpus. **Falsifier:** any sample where the baseline matches or wins.

Any of these could become a real cert. None is built here.

---

## Implementation pointer

All discrete-side mechanics are already in:

- `qa_reachability_descent_run_cert_v1/validator.py` (BFS, generator-legality, descent gates)
- `qa_alphageometry_ptolemy/qa_counterfactual_descent_cert_v1/qa_counterfactual_descent_cert_validate.py` (shortest-flip BFS)
- `qa_alphageometry_ptolemy/qa_cayley_bateson_filtration_cert_v1/` (component enumeration)
- `qa_alphageometry_ptolemy/qa_failure_density_enumeration_cert_v1/` (exact obstruction-density)

No new primitives are required to use the QA geodesy frame.

---

## References

Cited as primary sources behind the certs summarized above. Anchors are the certs themselves; this list documents the upstream literature.

- Bateson, G. (1972). *Steps to an Ecology of Mind*. ([191])
- Cayley, A. (1878). *Desiderata and suggestions: No. 2. The theory of groups: graphical representation*, Amer. J. Math. 1(2). ([211])
- Dehn, M. (1911). *Über unendliche diskontinuierliche Gruppen*, Math. Ann. 71. ([211])
- Eisenstein, G. (1844). Quadratic forms work underlying `f(b,e)=b²+be−e²`. ([214], [183], [133])
- Kochenderfer, M. J., Wheeler, T. A., Katz, S., Corso, A., & Moss, R. J. (Kochenderfer, 2026). *Algorithms for Validation*. MIT Press, CC-BY-NC-ND. Ch. 7, Ch. 11 §11.5. ([263], [265])
- Lang, S. (2002). *Algebra* (3rd ed.), Ch. III. Local-ring module classification via elementary divisors. (stratification cert)
- Levin, M. & Resnik, M. (Levin, 2026). *Mind Everywhere*, Biological Theory. ([193])
- Nielsen, J. (1924). Generation of `GL₂(Z)` by elementary matrices. (stratification cert)
- Pearl, J. (2009). *Causality* (2nd ed.). ([213])
- Solé, R., Seoane, L. F. et al. (Solé, 2026). *Cognition spaces*, arXiv:2601.12837. ([194])
- Wall, D. D. (1960). *Fibonacci series modulo m*, Amer. Math. Monthly 67. ([128], [192], [212])

---

*This file is a navigation layer. Edits should preserve the no-new-claims property; additions belong either in an existing cert or in a new cert with a falsifiable claim that does not duplicate [40]/[191]/[211]/[263]/[265].*
