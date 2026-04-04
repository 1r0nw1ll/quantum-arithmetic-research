# Bateson's Learning Levels as Meta-Operator Hierarchy on QA State Space

**Status**: Theoretical sketch (pre-cert). Verify strictness claims before promoting to `QA_BATESON_LEARNING_LEVELS_CERT.v1`.
**Author**: Claude (main session), 2026-04-04
**Context**: NLP/cybernetics deep dive (OB: d6cec94b). Bateson's Learning 0/I/II/III, when stated formally, produces a strict hierarchy of operator classes on QA state spaces whose invariants are orbit, modulus, and algebraic ambient.

---

## 1. Motivation

Gregory Bateson, via Russell-Whitehead logical types, proposed a hierarchy of learning:

| Bateson Level | Informal description |
|---|---|
| Learning 0 | No change. Fixed response to a stimulus. |
| Learning I | Change *within* a fixed set of alternatives (classical conditioning). |
| Learning II | Change *of* the set of alternatives ("learning to learn", context/frame shift). |
| Learning III | Change of the *system* of sets of alternatives (rare, disorienting). |
| Learning IV | Evolutionary, trans-individual (Bateson's conjectural ceiling). |

Bateson's claim was that each level is a *meta-level* of the one below, and that confusing levels produces pathology (the double bind is a Level-II operator required in a Level-I-only vocabulary).

This sketch formalizes the hierarchy as strictly nested operator classes on QA state spaces, and identifies each level with existing QA machinery.

---

## 2. Setup

Let $m \in \mathbb{Z}_{\geq 2}$ be a modulus. Define the QA state space

$$S_m = \{(b,e) : b,e \in \{1,\dots,m\}\}$$

with derived coordinates $d = ((b+e-1) \bmod m) + 1$, $a = ((b+2e-1) \bmod m) + 1$, and the canonical dynamic

$$T_m : S_m \to S_m, \qquad T_m(b,e) = ((b,e) \text{ under qa\_step}).$$

The orbits of $T_m$ partition $S_m$. For $m=9$, we have three orbit classes (cosmos, satellite, singularity), distinguished by $v_3(f(b,e))$ where $f(b,e)=b^2+be-e^2$ (cert [130] and dynamics spine).

Let $\mathcal{C}$ denote the category whose objects are QA state spaces $S_m$ for all $m$, with morphisms the structure-preserving maps (to be defined per level below).

---

## 3. The Four Operator Classes

### Level 0 — Fixed Points

$$\mathcal{L}_0 = \{\mathrm{id}_{\{s\}} : s \in S_m, T_m(s) = s\}$$

The identity restricted to a fixed point. In $S_9$, the only fixed point is $(9,9)$ (the singularity). In $S_{24}$, there are analogous fixed points by CRT.

**QA identification**: Singularity orbit = Learning 0. A state that maps to itself is a system that has no learning to do.

### Level I — Intra-Orbit Operators

$$\mathcal{L}_1 = \{\phi : S_m \to S_m \mid \phi(\mathcal{O}) = \mathcal{O} \text{ for every orbit } \mathcal{O} \text{ of } T_m\}$$

Equivalently, $\mathcal{L}_1$ is the set of orbit-preserving self-maps of $S_m$. This includes all powers $T_m^n$ of the dynamic, and more generally any automorphism of $S_m$ that acts as a permutation within each orbit.

**QA identification**: Cycling within cosmos or satellite. "Doing the same thing better" — traversing the cycle differently, but staying inside the same orbit. The integer path time $T_1$ (from cert [150] and axiom T1) is a Level-I coordinate.

### Level II — Inter-Orbit (Modulus-Preserving) Operators

$$\mathcal{L}_2 = \{\phi : S_m \to S_m \mid \phi \text{ sends some orbit to a distinct orbit}\}$$

Operators that stay within $S_m$ (same modulus, same ambient algebra) but re-assign orbit membership. Composition with $\mathcal{L}_1$ operators remains in $\mathcal{L}_2 \cup \mathcal{L}_1$.

**Existence verified by direct computation on $S_9$** (`tools/verify_bateson_level2.py`, 2026-04-04). $S_9$ decomposes into 5 orbits: three cosmos orbits of length 24 (reps $(1,1), (1,3), (1,4)$), one satellite orbit of length 8 (rep $(3,3)$), and the singularity $(9,9)$. Six concrete Level-II operators were found, with a natural sub-stratification.

**Sub-stratification discovered during verification:**

$$\mathcal{L}_2 = \mathcal{L}_{2a} \cup \mathcal{L}_{2b}$$

- **Level II-a** (family-preserving, orbit-changing): operators that re-assign orbit membership but preserve the orbit FAMILY (cosmos/satellite/singularity).
- **Level II-b** (family-changing): operators that cross family boundaries.

**Concrete Level II-a witnesses on $S_9$:**

| Operator | Orbit crossings | Notes |
|---|---|---|
| $(b,e) \mapsto (2b, 2e) \bmod 9$ | 72/81 | **Invertible automorphism**: $2 \in (\mathbb{Z}/9\mathbb{Z})^\times$, order 6 |
| $(b,e) \mapsto (e, b)$ (swap) | 48/81 | Involution; permutes cosmos orbits |
| $(b,e) \mapsto (2e, b)$ (male→female analog) | 36/81 | Not a symmetry of orbits |

The scalar-multiplication group $(\mathbb{Z}/9\mathbb{Z})^\times \cong \mathbb{Z}/6\mathbb{Z}$ acts on $S_9$ by $(b,e) \mapsto (kb, ke)$ for $k \in \{1,2,4,5,7,8\}$. This gives a 6-element group of Level-II-a operators that permutes the three cosmos orbits without leaving the cosmos family.

**Concrete Level II-b witnesses on $S_9$:**

| Operator | Family changes | Notes |
|---|---|---|
| $(b,e) \mapsto (3,3)$ (constant) | 73/81 | Collapses 72 cosmos points to satellite |
| $(b,e) \mapsto (3b, 3e) \bmod 9$ | 80/81 | 3 is not a unit mod 9; image ⊂ satellite ∪ singularity |
| $(b,e) \mapsto (3 \cdot ((b{-}1)\%3{+}1), 3 \cdot ((e{-}1)\%3{+}1))$ | 80/81 | Reduce mod 3 + lift to satellite grid |

**Invariant structure**: Level I preserves orbit membership; Level II-a preserves family but not orbit; Level II-b preserves modulus but not family. Each sub-level is characterized by which invariant is broken.

**QA identification**: Level II-a corresponds to "reframing within context" (NLP terminology) — staying in cosmos (the dynamic realm) while shifting which cosmos orbit the system is on. Level II-b corresponds to "changing the context entirely" — moving between dynamic realms (cosmos ↔ satellite ↔ singularity).

### Level III — Modulus / Algebra Change

$$\mathcal{L}_3 = \{\phi : S_m \to S_n \mid m \neq n, \text{ or same } m \text{ but different ambient algebra}\}$$

Maps *between different state spaces* in $\mathcal{C}$. Examples:
- **Reduction**: $S_9 \to S_3$ via $(b,e) \bmod 3$.
- **Extension**: $S_9 \hookrightarrow S_{72}$ via CRT with $S_8$.
- **Ring change**: $\mathbb{Z}[\phi]/9 \to \mathbb{Z}[i]/5$ (moving between Fibonacci and Gaussian QA).

**QA identification**: Changing the modulus, the ring, or the axiom base. In our current work, mod-9 ↔ mod-24 transitions (theoretical vs. applied) are Level-III operators. Observer projection at the system boundary (QA → continuous output) is a Level-III functor into a fundamentally different category.

### Level IV — Category Change (Bateson's ceiling)

$$\mathcal{L}_4 = \{\text{functors } \mathcal{C} \to \mathcal{C}'\}$$

Maps between entire *categories* of state spaces. E.g., the passage from "QA over commutative rings" to "QA over Lie algebras" or "QA over categories with monoidal structure." This is speculative at present; Bateson himself marked Level IV as hypothetical.

---

## 4. Strictness Claim

**Theorem (Strict Hierarchy)**: $\mathcal{L}_0 \subsetneq \mathcal{L}_1 \subsetneq \mathcal{L}_2 \subsetneq \mathcal{L}_3 \subsetneq \mathcal{L}_4$.

Moreover, each inclusion is strict in the sense that there exists an operator at level $n+1$ that cannot be expressed as any composition of operators at levels $\leq n$.

**Proof sketch (each strictness is the same pattern — an invariant preserved below, broken above):**

| Inclusion | Invariant preserved at lower level | Breaks at upper level |
|---|---|---|
| $\mathcal{L}_0 \subsetneq \mathcal{L}_1$ | "Is the identity on a point" | $T_m^n$ for $n \not\equiv 0 \pmod{|\mathcal{O}|}$ moves state |
| $\mathcal{L}_1 \subsetneq \mathcal{L}_2$ | Orbit membership | Inter-orbit map changes orbit class |
| $\mathcal{L}_2 \subsetneq \mathcal{L}_3$ | Modulus $m$ (and ambient algebra) | $S_m \to S_n$ with $n \neq m$ changes codomain |
| $\mathcal{L}_3 \subsetneq \mathcal{L}_4$ | Ambient category $\mathcal{C}$ | Functor to $\mathcal{C}' \neq \mathcal{C}$ |

The key observation is that each level $n$ has a *defining invariant* $I_n$ such that every operator in $\mathcal{L}_n$ preserves $I_n$, and there is an operator in $\mathcal{L}_{n+1}$ that does not preserve $I_n$. This is enough for strict inclusion in one direction; the reverse ($\mathcal{L}_n \subseteq \mathcal{L}_{n+1}$) holds because any operator preserving $I_n$ trivially lies in $\mathcal{L}_{n+1}$ under the canonical inclusion (an endomorphism of $S_m$ is a morphism $S_m \to S_m$ in $\mathcal{C}$, etc.).

**What still needs to be verified rigorously:**
1. That Level-II operators *exist* on $S_9$ — i.e., exhibit an explicit $\phi : S_9 \to S_9$ that moves a cosmos element to a satellite element. The male→female transform is a candidate but needs to be checked against the orbit classification.
2. That the invariants are *actually* well-defined under composition closure at each level.
3. That Level-IV is non-empty and non-trivial (currently conjectural).

---

## 5. The Double Bind Theorem (Tiered Reachability)

**Verified exhaustively on $S_9$** via `tools/verify_bateson_double_bind.py` (2026-04-04). All 6561 ordered pairs $(s_0, s_*) \in S_9 \times S_9$ are classified by minimum promotion tier.

### 5.1 Definitions

For $s_0, s_* \in S_m$, the **minimum tier** $\tau(s_0, s_*)$ is the smallest $k \in \{0, 1, 2a, 2b, 3, 4\}$ such that $s_*$ is reachable from $s_0$ via operators in $\mathcal{L}_0 \cup \mathcal{L}_1 \cup \cdots \cup \mathcal{L}_k$.

The **Level-$k$ reachable set** from $s_0$ is
$$R_k(s_0) = \{\phi_n \circ \cdots \circ \phi_1 (s_0) : n \geq 0, \phi_i \in \mathcal{L}_0 \cup \cdots \cup \mathcal{L}_k\}.$$

### 5.2 The Level-I Reachable Set

**Lemma**: $R_1(s_0) = \mathcal{O}(s_0)$, the $T$-orbit of $s_0$.

**Proof**: $(\supseteq)$ Since $T \in \mathcal{L}_1$, iterating $T$ from $s_0$ covers the orbit. $(\subseteq)$ Every $\phi \in \mathcal{L}_1$ satisfies $\phi(s) \in \mathcal{O}(s)$ by definition of $\mathcal{L}_1$ (orbit-preserving). By induction, any composition $\phi_n \circ \cdots \circ \phi_1$ applied to $s_0$ remains in $\mathcal{O}(s_0)$. $\square$

### 5.3 The Double Bind Theorem

**Theorem (Tiered Reachability / Double Bind)**: Let $s_0, s_* \in S_m$. Then:
$$\tau(s_0, s_*) = \begin{cases}
0 & \text{if } s_0 = s_* \\
1 & \text{if } s_* \in \mathcal{O}(s_0) \setminus \{s_0\} \\
2a & \text{if } s_* \notin \mathcal{O}(s_0) \text{ and } \mathrm{fam}(s_*) = \mathrm{fam}(s_0) \\
2b & \text{if } \mathrm{fam}(s_*) \neq \mathrm{fam}(s_0), \text{ same modulus } m \\
3 & \text{if } s_*, s_0 \text{ in state spaces of different moduli}
\end{cases}$$

**Proof**:

*Lower bound* (each case requires at least the stated tier):
- $\tau \geq 1$ when $s_0 \neq s_*$: identity alone does not suffice.
- $\tau \geq 2$ when $s_* \notin \mathcal{O}(s_0)$: by the Lemma, $R_1(s_0) = \mathcal{O}(s_0) \not\ni s_*$, so no composition of Level-I operators reaches $s_*$.
- $\tau \geq 2b$ when $\mathrm{fam}(s_*) \neq \mathrm{fam}(s_0)$: every $\phi \in \mathcal{L}_{2a}$ preserves family by definition, so any $\mathcal{L}_1 \cup \mathcal{L}_{2a}$ composition preserves family. Crossing family requires an operator that breaks family, i.e., $\mathcal{L}_{2b}$.
- $\tau \geq 3$ when moduli differ: $\mathcal{L}_1 \cup \mathcal{L}_2$ consists of self-maps of a fixed $S_m$; crossing moduli requires $\mathcal{L}_3$.

*Upper bound* (each case is achievable at the stated tier):
- $\tau = 1$: $T^k(s_0) = s_*$ for some $k$ when $s_* \in \mathcal{O}(s_0)$.
- $\tau = 2a$: Define $\phi : S_m \to S_m$ by $\phi(s_0) = s_*$, $\phi(s) = s$ for $s \neq s_0$. This is family-preserving (only $s_0$ changes, and $\mathrm{fam}(s_*) = \mathrm{fam}(s_0)$) and orbit-changing (since $s_* \notin \mathcal{O}(s_0)$). Hence $\phi \in \mathcal{L}_{2a}$.
- $\tau = 2b$: Same piecewise construction; $\phi \in \mathcal{L}_{2b}$ since family changes.
- $\tau = 3$: Level-III operators (e.g., reduction or extension maps) exist between any $S_m$ and $S_n$ via CRT / ring homomorphisms. $\square$

### 5.4 Exhaustive Verification on $S_9$

All 6561 ordered pairs were classified:

| Tier | Count | Fraction | Example |
|---|---|---|---|
| 0 | 81 | 1.23% | $((1,1),(1,1))$ |
| 1 | 1712 | 26.09% | $((1,1),(1,2))$ |
| 2a | **3456** | **52.67%** | $((1,1),(1,3))$ |
| 2b | 1312 | 20.00% | $((1,1),(3,3))$ |
| **total** | **6561** | | |

Counts match the structural prediction exactly:
- Tier 0: $|S_9| = 81$ (diagonal)
- Tier 1: $\sum_{\mathcal{O}} |\mathcal{O}|(|\mathcal{O}|-1) = 3 \cdot 24 \cdot 23 + 8 \cdot 7 + 0 = 1712$
- Tier 2a: same-family different-orbit pairs = $72^2 - 3 \cdot 24^2 = 5184 - 1728 = 3456$ (cosmos only; satellite has one orbit so contributes 0)
- Tier 2b: cross-family pairs = $2(72 \cdot 8 + 72 \cdot 1 + 8 \cdot 1) = 2 \cdot 656 = 1312$

### 5.5 Concrete Witnesses (verified 2026-04-04)

| Tier | Witness operator | Source → target | Classification |
|---|---|---|---|
| 1 | $T$ (canonical dynamic) | $(1,1) \to (1,2)$ | orbit-preserving |
| 2a | $\phi_{\times 2}(b,e) = (2b, 2e) \bmod 9$ | $(1,1) \to (2,2)$ | orbit-changing, family-preserving |
| 2b | $\phi_{\times 3}(b,e) = (3b, 3e) \bmod 9$ | $(1,1) \to (3,3)$ | family-changing |

### 5.6 Quantitative Double Bind

**Corollary**: Only **26.09%** of ordered pairs in $S_9 \times S_9$ are Level-I reachable. The majority (**52.67%**) require Level-II-a promotion; **20.00%** require Level-II-b. Under the default dynamic $T$ alone, the system is "stuck" (in the double-bind sense) for nearly three-quarters of all target configurations.

This is a quantitative version of Bateson's clinical observation: most problems cannot be solved at the level they appear. Escape requires promotion to a higher operator class, and the required tier is determined by *which invariant* separates source from target.

### 5.7 Interpretations

**Therapy (Bateson / NLP)**: A client stuck at Level I is trying harder within the same frame. The therapist applies a Level II-a or II-b operator (reframing, context change, metaphorical intervention). Effectiveness requires the therapist to have access to higher-tier operators — Ashby's requisite variety, stratified by level. Most therapeutic impasses are II-a or II-b double binds.

**Control theory**: Level-I unreachability is the standard reachability-failure mode for LTI systems (target outside the reachable subspace). Level-II promotion corresponds to switching between linear models; Level-III corresponds to changing the model class entirely.

**Self-improvement (QA Lab track)**: A kernel improving itself within its own state space operates at Level II (same modulus, different orbit/family). A kernel changing its own state-space definition operates at Level III. The 26% Level-I reachability ratio suggests that most meaningful self-improvement requires at least Level II.

**Levin morphogenesis**: Cell differentiation = Level II-a (same organism, different cell fate within the dynamic realm). Metamorphosis = Level II-b or III (crossing developmental realms). The tier hierarchy gives a formal grammar for developmental transitions.

---

## 6. Self-Improvement Stability Criterion

**Setup**: A QA kernel is a pair $(S_m, T_m)$. Self-improvement means the kernel applies a Level-III operator $\Phi \in \mathcal{L}_3$ to *its own definition*, producing a new kernel $(S_{m'}, T_{m'}) = \Phi(S_m, T_m)$.

**Stability**: The self-improvement process is stable iff $\Phi$ has a fixed point — i.e., there exists $m^*$ such that $\Phi(S_{m^*}, T_{m^*}) = (S_{m^*}, T_{m^*})$.

**Interpretation**: A kernel that keeps "improving itself" without ever reaching a fixed point is in runaway meta-learning — formally analogous to Bateson's pathological Learning III (sometimes associated with psychosis in his model). A kernel that reaches a fixed point has found the $m^*$ at which further meta-improvement is redundant.

**Open question**: Does the QA orbit classification have a Level-III fixed point? Candidate: $m^* = 9$ (minimal nontrivial $v_3$ stratification) or $m^* = 24$ (maximal classical structure, Leech/monster connections).

**Connection to memory**: This formalizes the "QA Lab self-improvement" project directive. Self-improvement is Level III. Safe self-improvement requires fixed-point stability.

---

## 7. Unification with Existing Structures

Every previously-encountered "hierarchy" in our work collapses into this one:

| External hierarchy | Level | Notes |
|---|---|---|
| Dilts' Logical Levels (environment/behavior/capabilities/beliefs/identity/spirit) | I → II → III | Informal but directionally correct |
| Ashby's ultrastability (Level 1 feedback / Level 2 parameter change) | I / II | Ashby stopped at II; Bateson went further |
| Stafford Beer's VSM recursion | II / III | Each viable system contains viable subsystems (Level II); the whole recursion schema is Level III |
| Watzlawick's first-order / second-order change | I / II | First-order = intra-orbit, second-order = inter-orbit |
| Korzybski's levels of abstraction | I / II / III | Map-of-map-of-map structure |
| Learning 0/I/II/III (Bateson) | 0 / I / II / III | The direct source |
| Our three orbits (cosmos/satellite/singularity) | I (cosmos, satellite) + 0 (singularity) | Intra-level structure of $S_9$ |
| Our mod-9 ↔ mod-24 transitions | III | Modulus change |
| Levin morphogenesis (differentiation/metamorphosis) | II / III | Differentiation = orbit change; metamorphosis = state-space change |
| Human needs growth (ΔT) vs contribution (ΣT) | I vs II+ | Growth within frame vs growth of frame |
| NLP "first-order change" vs "second-order change" | I / II | Direct parallel |

**This is another convergence**. After the seven-path torus, the eight-path modeling synthesis (NLP), we have a nine-path hierarchy convergence. Same structure, multiple discoverers.

---

## 8. Open Items Before Cert Promotion

Before promoting to `QA_BATESON_LEARNING_LEVELS_CERT.v1`:

1. ~~**Verify Level-II existence on $S_9$**~~ **VERIFIED 2026-04-04** via `tools/verify_bateson_level2.py`. Six witnesses found, natural sub-stratification $\mathcal{L}_{2a} / \mathcal{L}_{2b}$ discovered (family-preserving vs. family-changing). The $(\mathbb{Z}/9\mathbb{Z})^\times$ scalar action gives a concrete 6-element group of Level-II-a automorphisms.
2. **Formalize the ambient category $\mathcal{C}$**: Objects, morphisms, and check that the Level-III maps are actually morphisms in a well-defined category.
3. **Strictness of $\mathcal{L}_2 \subsetneq \mathcal{L}_3$ via counterexample**: Explicit Level-III operator that cannot be decomposed into Level-II operators (easy candidate: $S_9 \to S_{24}$ via CRT embedding — the codomain is literally a different set).
4. **Check Dilts' logical levels mapping**: Dilts' hierarchy has six levels, ours has four plus a ceiling. Whether Dilts' extra levels are genuine refinements or informal splits of Level II/III.
5. ~~**Double bind theorem as an actual theorem**~~ **VERIFIED 2026-04-04**: Tiered Reachability Theorem proved (§5.3) and exhaustively verified on $S_9$ via `tools/verify_bateson_double_bind.py`. All 6561 pairs classified; structural counts match. Quantitative result: only 26% of pairs are Level-I reachable.
6. **Self-improvement fixed point**: Check whether $m = 9$ or $m = 24$ is a fixed point of any natural Level-III operator.

If items 1, 3, and 5 verify cleanly, promote to cert. If item 1 fails, collapse Levels I–II into a single level and rebuild the hierarchy as 3-level (not 4).

---

## 9. Why This Matters

This is not just a taxonomic exercise.

- **It formalizes meta-learning** on discrete state spaces, which is directly relevant to the QA Lab self-improvement track.
- **It gives a reachability-theoretic account of therapeutic/developmental change**, which matches the NLP modeling project's best intuitions but with explicit failure modes (unreachability, projection aliasing, non-existent Level-II operators).
- **It unifies nine previously-disparate hierarchies** (Bateson, Dilts, Ashby, Beer, Watzlawick, Korzybski, our orbits, our modulus changes, Levin) under a single operator-stratification theorem. Each discoverer found a piece of the same structure.
- **It provides a stability criterion for safe self-improvement** via Level-III fixed points. This is load-bearing for the QA Lab and Levin morphogenetic tracks.
- **It gives us the right formalization of NLP's modeling claim**: modeling excellence = generator reconstruction = recovering the level at which the exemplar operates, then finding or constructing the operator at that level in the learner's state space. Transfer succeeds iff the target trajectory is reachable at the same level; fails if it requires a higher-level promotion the learner cannot make.

---

*Next step: verify open item 1 (Level-II existence on $S_9$) by direct computation on the orbit classification.*
