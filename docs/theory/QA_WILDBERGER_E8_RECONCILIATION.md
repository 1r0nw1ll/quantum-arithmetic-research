# QA E8 Alignment ↔ Wildberger's Simply-Laced Combinatorial Construction

**Status:** reconciliation note, draft 2026-04-13
**Primary sources:**
- Wildberger, N.J. *A Combinatorial Construction for Simply-Laced Lie Algebras*. J. Lie Theory 13:155–165, 2003.
- Wildberger, N.J. *A Combinatorial Construction for Simply-Laced Lie Algebras*. Adv. Appl. Math. 30(1–2):385–396, 2003.
- Wildberger, N.J. *Minuscule Posets from Neighbourly Graph Sequences*. Eur. J. Combin. 24:741–757, 2003.

---

## 1. Background: QA's current E8 alignment

Per `CLAUDE.md §Core QA Architecture`:

> **E8 Alignment**: 4D QA tuples → 8D projection → cosine similarity to E8 root system (240 vectors)
> **Harmonic Index**: HI = E8_alignment × exp(−0.1 × loss)

Mechanism: a QA tuple (b, e, d, a) is lifted to R^8 via a chosen embedding, then compared to each of the 240 E8 roots by cosine similarity. The alignment score is continuous-valued.

**Theorem NT concern.** Cosine similarity is a continuous observer-projection statistic. Using it as the *alignment* score is legitimate (observer layer). Using it as a causal input to QA dynamics would be a T2 firewall violation. Current codebase treats it as observer-layer (as far as the linter has flagged). Whether a purely integer alignment mechanism is available is the question this note addresses.

## 2. Wildberger's construction — summary

Wildberger 2003 constructs the simply-laced Lie algebras **A_n, D_n, E_6, E_7** from their Dynkin diagrams using:

- A **distributive lattice** of order ideals on the minuscule poset attached to the diagram.
- **Raising and lowering operators** X_i^±, one per Dynkin node i, acting on the space of ideals.
- Commutation relations [X_i^+, X_j^−] = δ_{ij} H_i recovered combinatorially via the **numbers game** on the Dynkin diagram.

All operators act by integer coefficients on integer basis elements. No real numbers, no square roots, no continuous parameters.

**Scope of the method — verbatim from primary source.** Wildberger 2003 (Adv. Appl. Math., read locally) states explicitly in the introduction:

> "Our goal is to show how to (almost) uniformly construct the simply-laced Lie algebras using only graph theoretic ideas from the Dynkin diagrams. We will thus construct the Lie algebras corresponding to A_n, D_n, E_6 and E_7 using a method which is independent of type. **The only case not covered is that of E_8**, for which more sophisticated techniques must be used."

And Theorem 2.1 in that paper:

> "Let X be a simple graph for which there exists a maximal neighbourly X-heap F which is two-neighbourly. Then X is one of the graphs A_n (n ≥ 1), D_n (n ≥ 4), E_6 or E_7."

So E_8 is **structurally excluded** from the two-neighbourly maximal-heap classification. The mechanism: E_8 has no minuscule representation (Green 2013), which is equivalent to Wildberger's two-neighbourly maximality failing.

Consequence: **Wildberger's minuscule-poset construction does not produce E_8**. Any QA integer-E_8 alignment must come from elsewhere.

## 3. Reconciliation — what this means for QA

The naïve hope ("replace QA's cosine-sim E8 alignment with Wildberger's integer construction") **does not work** because the construction itself excludes E_8.

What *is* available, structurally:

| Construction                                    | Produces | Integer-only? | Direct QA fit? |
|-------------------------------------------------|----------|---------------|-----------------|
| Wildberger minuscule-poset (J. Lie Theory 2003) | A_n, D_n, E_6, E_7 | Yes | Yes — ideal lattices are order-theoretic |
| Wildberger "combinatorial construction" (Adv. Appl. Math. 2003) | A_n, D_n, E_6, E_7 — **not E_8** (Theorem 2.1) | Yes — distributive lattice of ideals | Yes for those types; no E_8 |
| QA cosine-sim to 240 E_8 roots                  | E_8 alignment score | **No** — continuous | Observer layer only |
| E_8 root lattice as integer Gram matrix         | E_8 itself           | Yes            | Direct but not yet integrated |

Wildberger's Adv. Appl. Math. 2003 paper may cover more ground than the J. Lie Theory version (different venue, shorter title, same year). If it includes an E_8-capable construction (e.g., via non-minuscule methods or via the adjoint representation), it becomes the candidate bridge. Acquiring the PDF and verifying this is the next action.

## 4. Alternative integer path to E_8 (independent of Wildberger 2003)

The E_8 root lattice has an explicit integer Gram matrix (the Cartan matrix times itself, or the standard E_8 Gram of norm-2 vectors). An integer alignment mechanism could be:

- Represent a QA tuple (b, e, d, a) as an integer vector in Z^8 via a fixed embedding.
- Compute inner products with E_8 root vectors as **integer** values (no cosine, no normalization).
- Identify the alignment family (root / co-root / Weyl chamber) by integer arithmetic on the Gram matrix.

This gives a Theorem-NT-clean alignment witness that does not depend on cosine similarity. It is a **discrete classification** rather than a continuous score — which matches QA's orbit/family structure in spirit.

## 5. Actions

- [x] Acquire and read Adv. Appl. Math. 30(1–2):385–396 — done 2026-04-13, Theorem 2.1 confirms E_8 exclusion.
- [x] Audit E_8 alignment code — `qa_representational_geometry.py:159–204`. Declaration at top: `'signal_injection': 'none (static algebraic analysis)'`. Observer-layer only. **Theorem NT clean.**
- [ ] Prototype a discrete integer-Gram alignment classifier on Satellite orbits (Codex task — python-write gated for Claude).
- [ ] If integer classifier discriminates comparably to cosine-sim, dual-run or replace.

## 6. 2026-04-14 CORRECTION — Mutation Game 2020 handles E_8

**This section corrects §2–5.** The 2003 minuscule-poset construction excluded E_8, but Wildberger's **2020 paper** *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems* (acquired 2026-04-14 from user's Downloads) extends the framework to cover **all** finite Coxeter types — including E_8.

**Primary source.** Wildberger, N.J. *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems* (≈ 79 pp; later published in Algebra Colloquium 27 (2020)). Local `/tmp/wild/mutation_game_2020.txt`.

**Theorem 0.1 (verbatim from primary source):** The finite Coxeter groups W correspond to Coxeter graphs Γ in the list: A_n (n ≥ 1), B_n (n ≥ 2), D_n (n ≥ 4), **E_6, E_7, E_8**, F_4, G_2, H_3, H_4, I_2(m). E_8 is explicitly included.

**Theorem 0.3** lists the reduced irreducible root systems: A_n, B_n, C_n, D_n, E_6, E_7, **E_8**, F_4, G_2.

**How the construction reaches E_8.** Not via minuscule posets (which still fail for E_8) but via **bidirected multigraphs and the Mutation Game**. Martians and anti-Martians occupy vertices of a directed multigraph; populations mutate by a specific integer-valued rule; the root populations R(X) are exactly the lattice produced from a single simple root by iterated mutations. For Coxeter graphs of type ADE (including E_8), mutations generate the full positive root poset R+(X, x).

**Inductive cascade (verbatim):** A_n → A_{n+1}, D_n → D_{n+1}, and the exceptional chain **E_6 → E_7 → E_8**, connected by ψ-complementation maps. The paper explicitly tabulates R+(E_8, 7) as a 120-element poset of integer-coefficient vectors (the 120 positive roots; with negatives, the full 240 E_8 roots).

**QA consequence.** Wildberger now has a Theorem-NT-clean **integer E_8 construction** via mutations. The QA codebase's current cosine-similarity E_8 alignment (`qa_representational_geometry.py:159–204`, still observer-layer-only and Theorem-NT-clean) can be supplemented or replaced by an integer mutation-based classifier:

- Represent a QA tuple (b, e, d, a) as an initial population on a chosen Coxeter graph.
- Apply integer mutations; the reachable populations are exactly the roots of the associated root system.
- Classify a QA orbit by which root-system region its T-iterates populate.

This is a substantive upgrade to the E_8 reconciliation situation. Not yet implemented; flagged as cert candidate **[244] QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1** for future session work.

## 7. Summary across 2003 and 2020 Wildberger integer Lie-algebra works

| Construction | Year | Types covered | QA applicability |
|--------------|------|---------------|-------------------|
| Minuscule-poset construction | 2003 | A_n, D_n, E_6, E_7 | Integer-clean; E_8 excluded |
| G_2 hexagon operators | 2003 | G_2 | Integer matrix entries {−2, −1, 0, 1, 2} |
| sl(3) diamond model | 2003 | sl(3) = A_2 | Integer polytope; cert [240] |
| **Mutation Game** | **2020** | **All finite Coxeter incl. E_8** | **Integer mutations; full 240-root E_8 explicitly** |

The earlier "E_8 excluded" conclusion was accurate for the 2003 method but is superseded by the 2020 Mutation Game paper. Wildberger's integer-Lie-algebra program now covers the full finite Coxeter classification, including E_8.

## 8. References

- Wildberger, N.J. *A combinatorial construction for simply-laced Lie algebras*. J. Lie Theory **13** (2003) 155–165.
- Wildberger, N.J. *A combinatorial construction for simply-laced Lie algebras*. Adv. Appl. Math. **30** (2003) 385–396.
- Wildberger, N.J. *Minuscule posets from neighbourly graph sequences*. Eur. J. Combin. **24** (2003) 741–757.
- **Wildberger, N.J.** *The Mutation Game, Coxeter–Dynkin Graphs, and Generalized Root Systems*. Algebra Colloquium **27** (2020). Primary source for E_8 via mutations; acquired 2026-04-14.
- Green, R.M. *Combinatorics of Minuscule Representations*. Cambridge Tracts in Math. 199, 2013 — confirms E_8 has no minuscule representation (relevant only to the 2003 method).
