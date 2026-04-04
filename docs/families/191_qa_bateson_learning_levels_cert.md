# Family [191] QA_BATESON_LEARNING_LEVELS_CERT.v1

## One-line summary

Gregory Bateson's Learning Levels (0/I/II/III) formalized as a strict invariant filtration on QA state spaces. Four invariants (orbit ⊂ family ⊂ modulus ⊂ ambient category) define five operator classes. Tiered Reachability Theorem exhaustively verified on S_9: only 26% of 6561 pairs are Level-I reachable.

## Mathematical content

### Invariant filtration

| Class | Invariant preserved | Invariant broken |
|-------|---------------------|------------------|
| L_0  | fixed point          | — |
| L_1  | orbit membership     | — |
| L_2a | family (cosmos/satellite/singularity) | orbit |
| L_2b | modulus              | family |
| L_3  | ambient category     | modulus |
| L_4  | (hypothetical)       | ambient category |

Each level is characterized by exactly which invariant its operators preserve; strict inclusion is witnessed by explicit operators that break the next invariant down.

### Tiered Reachability Theorem

For s_0, s_* ∈ S_m, the minimum tier τ(s_0, s_*) required to reach s_* from s_0 via operators of class L_0 ∪ ... ∪ L_k is:

- **τ = 0**  iff  s_0 = s_*
- **τ = 1**  iff  s_* ∈ Orbit(s_0) \ {s_0}
- **τ = 2a** iff  different orbit, same family
- **τ = 2b** iff  different family, same modulus
- **τ = 3**  iff  different modulus

**Key lemma**: R_1(s_0) = Orbit(s_0). The Level-I reachable set equals the T-orbit of s_0, because T ∈ L_1 generates the orbit and every L_1 operator preserves orbit membership. Hence any target outside Orbit(s_0) is Level-I unreachable — a formal double bind.

### Exhaustive verification on S_9 (6561 pairs)

| Tier | Count | Fraction | Witness operator |
|------|-------|----------|------------------|
| 0    | 81    | 1.23%    | identity         |
| 1    | 1712  | 26.09%   | T (qa_step)      |
| 2a   | 3456  | **52.67%** | scalar_mult k=2 |
| 2b   | 1312  | 20.00%   | scalar_mult k=3  |
| **total** | **6561** | **100.00%** | |

Structural formulas (all match exactly):
- Tier 0 = |S_9| = 81
- Tier 1 = Σ_orbits |O|·(|O|-1) = 3·24·23 + 8·7 + 0 = 1712
- Tier 2a = 72² - 3·24² = 3456 (cosmos same-family different-orbit; satellite has one orbit)
- Tier 2b = 2(72·8 + 72·1 + 8·1) = 1312 (cross-family pairs)

### Quantitative double bind

Only **26.09%** of ordered pairs in S_9 are Level-I reachable. The majority (**52.67%**) require Level-II-a promotion; **20.00%** require Level-II-b. Under the default QA dynamic T alone, the system is "stuck" (in the double-bind sense) for 73.91% of all target configurations.

This is a quantitative refinement of Bateson's clinical observation that most problems cannot be solved at the level they appear. Escape requires promotion to a higher operator class, and the required tier is determined by *which invariant* separates source from target.

### Concrete witnesses

| Tier | Operator | Source → target | Classification |
|------|----------|-----------------|----------------|
| 1    | T (qa_step) | (1,1) → (1,2)    | orbit-preserving |
| 2a   | ×2 (scalar mult, k=2) | (1,1) → (2,2) | orbit 0 → orbit 1, same family (cosmos) |
| 2a   | swap (b,e)↔(e,b)      | (1,2) → (2,1) | orbit 0 → orbit 1, same family |
| 2b   | ×3 (scalar mult, k=3) | (1,1) → (3,3) | cosmos → satellite |
| 2b   | constant (9,9)         | (1,1) → (9,9) | cosmos → singularity |
| 3    | modulus reduction m=3  | (4,5) in S_9 → (1,2) in S_3 | codomain S_3 ≠ S_9 |

The (ℤ/9ℤ)* scalar action gives a concrete 6-element group of invertible Level-II-a operators permuting the three cosmos orbits.

## Checks

| ID | Description |
|----|-------------|
| BLL_1       | schema_version == 'QA_BATESON_LEARNING_LEVELS_CERT.v1' |
| BLL_FILT    | invariant filtration well-formed (orbit, family, modulus, ambient_category) |
| BLL_TIER    | exhaustive tier count on S_9 matches structural prediction (81/1712/3456/1312) |
| BLL_L1      | ≥ 1 Level-I witness (orbit-preserving) |
| BLL_L2A     | ≥ 1 Level-II-a witness (orbit-changing, family-preserving) |
| BLL_L2B     | ≥ 1 Level-II-b witness (family-changing) |
| BLL_L3      | ≥ 1 Level-III witness (modulus-changing) |
| BLL_STRICT  | strict inclusions L_1 ⊊ L_2a ⊊ L_2b ⊊ L_3 witnessed |
| BLL_DB      | declared tier distribution matches theorem (sum=6561) |
| BLL_SRC     | source attribution to Bateson present |
| BLL_WITNESS | ≥ 4 witnesses (one per non-trivial tier) |
| BLL_F       | fail_ledger well-formed |

## Source grounding

- **Gregory Bateson**, *Steps to an Ecology of Mind* (1972) — Learning Levels hierarchy, logical types, double bind theory
- **Bertrand Russell & Alfred North Whitehead**, *Principia Mathematica* (1910–13) — theory of logical types
- **W. Ross Ashby**, *An Introduction to Cybernetics* (1956) — Law of Requisite Variety; ultrastability (Level 1/Level 2)
- **Paul Watzlawick, Janet Beavin, Don Jackson**, *Pragmatics of Human Communication* (1967) — first-order vs second-order change (MRI Palo Alto School)
- **Alfred Korzybski**, *Science and Sanity* (1933) — map/territory, levels of abstraction
- Full theoretical sketch: `docs/theory/QA_BATESON_LEARNING_LEVELS_SKETCH.md`
- Verification scripts: `tools/verify_bateson_level2.py`, `tools/verify_bateson_double_bind.py`

## Convergence map

This cert unifies multiple previously-disparate hierarchies under a single operator-stratification theorem:

| External hierarchy | Mapping |
|--------------------|---------|
| Bateson Learning 0/I/II/III | L_0 / L_1 / L_2 / L_3 |
| Ashby ultrastability (feedback / parameter change) | L_1 / L_2 |
| Watzlawick first-order / second-order change | L_1 / L_2 |
| Korzybski levels of abstraction | L_1 / L_2 / L_3 |
| NLP first-order / second-order change | L_1 / L_2a and L_2b |
| Dilts logical levels (environment→spirit) | L_1 / L_2a / L_2b / L_3 (approximate) |
| Stafford Beer VSM recursion | L_2 / L_3 |
| Levin morphogenesis (differentiation / metamorphosis) | L_2a / L_2b or L_3 |

Eight independent research programs converging on the same operator-class hierarchy.

## Connection to other families

- **[130] QA Origin of 24** — orbit classification via v_3(f) underpins the cosmos/satellite/singularity family distinction
- **[133] QA Eisenstein Norm** — norm f(b,e) = b²+be−e² determines orbit family
- **[140] QA Conic Discriminant** — I > 0 / I = 0 / I < 0 stratification is another invariant filtration on S_m (ring/horn/spindle torus)
- **[150] QA Septenary Unit Group** — (ℤ/9ℤ)* is the concrete 6-element group whose action on S_9 provides invertible L_2a operators
- **[153] QA Keely Triune** — triune (cosmos/satellite/singularity) = the family invariant preserved by L_2a, broken by L_2b

## Fixture files

- `fixtures/bll_pass_hierarchy.json` — PASS: witnesses at tiers 1, 2a, 2a, 2b, 2b, 3; exhaustive S_9 tier distribution
- `fixtures/bll_fail_bad_tier.json` — FAIL fixture for testing validator tier mismatch detection
