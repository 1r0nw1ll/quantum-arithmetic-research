<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.6 Briddell/FST primary-source excerpt snapshot. Verbatim quotes from (a) Dale & Briddell 2026 v2 Completion Layer paper at qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex (1105 lines of LaTeX; Frontiers in Physics ms 1850870 revision v2); (b) Dale & Briddell 2026 Addendum at qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.tex (3pp); (c) Briddell 2020 Structural Physics monograph (external, referenced). Locators resolve against the fixture source_claims_briddell.json. Do not edit quote fields. -->

# Briddell FST Corpus — Phase 4.6 Primary-Source Excerpts (Corrected)

**Primary sources on disk:**
- `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.tex` — v2 completion-layer paper (Dale + Briddell, April 2026; Frontiers ms 1850870 revision v2)
- `qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.tex` — structural addendum (3pp, April 2026)
- `qa_alphageometry_ptolemy/qa_fst/qa_fst_completion_paper.pdf` — compiled PDF
- `qa_alphageometry_ptolemy/qa_fst/qa_fst_cert_bundle.json` + `qa_fst_validate.py` + `qa_fst_manifest.json` — v2 cert infrastructure

**Referenced external:**
- `briddellbook2020` = Don Briddell, *Structural Physics* (monograph). Not in repo; cited throughout the v2 paper's bibliography.

**Analytical context (OB captures):**
- 2026-04-11 FST v1→v2 re-map audit: fixed 5 v1 violations (hexagon witness unsourced, u/d ratio tautological, inverted cluster assignment, worse-mass-selection, Theorem NT firewall blur)
- 2026-04-13 STF↔QA b=e diagonal cross-validation: structural upgrade from postulate-and-certify to derivation-and-certify

---

## Dale & Briddell 2026 v2 — *A Quantum Arithmetic Completion Layer for FST* (Frontiers ms 1850870)

### #abstract-completion-layer (Abstract)

> "We present **Quantum Arithmetic (QA)** as a *completion layer*: a framework that formalizes state spaces, generators, invariants, and failure modes without introducing new physical assumptions, and applies them to the verification of an external target theory."

### #abstract-five-claims (Abstract)

> "We formalize five explicit claims from the FST primary text: (i) the Sierpinski Triangle Fractal (STF) decomposition of the proton's 1836 loops into integer clusters; (ii) the lambda-to-proton decay bookkeeping $2187 - 243 - 81 - 27 = 1836$ as exact integer arithmetic; (iii) a calibration postulate $P_1$ identifying one loop with one electron rest mass as the sanctioned bridge between the discrete QA layer and the continuous observer layer; (iv) a purely geometric three-cluster quark partition that does not assert Standard-Model flavor identification; and (v) the six-loop chiral pair structure of first-generation fermions."

### #abstract-firewall (Abstract)

> "All validation satisfies the QA Observer Projection Firewall (Theorem NT): the discrete state layer is integer-only, and the continuous observer layer is crossed exactly once per calibration row via the declared projection $\Pi$."

### #three-levels-engagement (§1.2)

> "We distinguish three levels of engagement with a theoretical proposal: (1) **Acceptance**: believing the theory is physically true. (2) **Verification**: confirming that stated claims are internally consistent and reproducible. (3) **Falsification**: identifying specific, bounded conditions under which a claim fails. Quantum Arithmetic, as used here, operates at levels (2) and (3). It does not address level (1) — physical truth is outside its scope."

### #plenum-core-elements (§2.1)

> "FST posits a non-local, space-filling structural potential called the **Plenum**, composed of continuous action loops. Particles form when these loops organize into stable structures through entanglement, rotation, and condensation. Two complementary loop types operate simultaneously: **Rspin** (counter-clockwise, anti-matter) and **Aspin** (clockwise, real-matter). Rspin loops remain deployed and dynamically rotating; Aspin loops may condense to form interaction boundaries."

### #structor-definition (§2.1)

> "The minimal space-defining unit is the **Structor**: a six-loop nucleus formed from three deployed Rspin loops enclosed by three condensed Aspin loops."

### #observer-projection-firewall (§3.2)

> "**Definition (Observer Projection Firewall).** An observer projection is a function $\Pi: S \to \mathbb{R}^n$ that maps an integer state in the QA layer to a real-valued output in the observer layer. The *observer projection firewall* (Theorem NT in the QA axiom block) asserts: $\Pi$ is the *only* direction of information flow between the two layers. Continuous values are never causal inputs to the QA layer; they may only be outputs. Any attempt to feed a float-typed value back into QA state, a QA generator, or a QA invariant is a Type-2b axiom violation."

### #postulate-p1 (§4)

> "**Postulate ($P_1$: Loop-mass calibration).** One loop (one 'Love') is calibrated to one electron rest mass: $1\;\text{Love} := m_e = 0.510998950\;69\;\mathrm{MeV}/c^2$. $P_1$ is declared, not derived. It is the sole bridge between the integer loop counts in the QA layer and MeV values in the observer layer."

### #cert1-proton-decomposition (§5)

> "Briddell's Fig. 2.64 presents the proton's 1836 loops as a Sierpinski-triangle decomposition: two bottom sub-triangles of 729 loops each, plus a top cluster of 378 loops, where $378 = 243 + 81 + 27 + 27$. The formal integer claim is: $1836 = 729 + 729 + 243 + 81 + 27 + 27$."

### #cert2-lambda-decay (§6)

> "The lambda hyperon is STF iteration 7 at 2187 loops. The proton is obtained from the lambda by removing three smaller nested STF sub-triangles corresponding to iterations 5, 4, and 3 (243, 81, and 27 loops respectively): $2187 - 243 - 81 - 27 = 1836$."

### #cert3-proton-mass-prediction (§7.4)

> "The direct-read proton-mass calibration $\Pi(1836) = 938.194\;\mathrm{MeV}$ differs from the PDG 2024 proton mass by $0.078\;\mathrm{MeV}$ ($0.0083\%$) ... the v2 direct-read $\Pi(1836) = 938.194\;\mathrm{MeV}$ has drift $0.078\;\mathrm{MeV}$, a $25\times$ improvement [over v1 subtraction-path], and has a cleaner epistemic status: it is the direct application of the declared postulate rather than an indirect chain of source numbers that propagates Briddell's own rounding."

### #cert4-no-sm-flavor (§8.3)

> "Under Postulate $P_1$, the Briddell clusters correspond to masses $\Pi(729) = 372.5\;\mathrm{MeV}$ and $\Pi(378) = 193.2\;\mathrm{MeV}$. These do not match the Standard Model current-quark masses ($m_u \approx 2.16\;\mathrm{MeV}$, $m_d \approx 4.67\;\mathrm{MeV}$ per PDG 2024) by three orders of magnitude, and they also do not match Standard Model constituent-quark masses (both approximately $336\;\mathrm{MeV}$ in non-relativistic quark models). The FST clusters are geometric constructs of Briddell's decomposition, not Standard-Model quark-flavor objects, and we do not certify any identification between them."

### #cert5-six-loop-chiral (§9)

> "Briddell states: 'A fermion mass is composed of six loops, three of which are Rspin real-loop quarks and three are Aspin anti-loops quarks' [line 795]. We formalize this as a type-level invariant on the FST state packet. Every first-generation fermion state must carry $|\text{Rspin}| = |\text{Aspin}| = 3$ as a type-level constraint."

### #what-qa-does-not-do (§13.1)

> "It is important to state explicitly what the QA completion layer does *not* claim: It does not propose alternative physics. The Plenum, loops, STF hierarchy, Rspin/Aspin sectors, and cloaking mechanism are all Briddell's constructs, used here exactly as stated. It does not reinterpret FST concepts. It does not 'correct' the theory. It does not certify Standard-Model flavor identifications that Briddell's own text does not support."

### #corrigendum-item-1 (§11)

> "**Corrigendum item 1: Fig. 2.64 caption formula (MS line 1286).** The current text reads 'The Proton mass, having 1836 loops $(729+729+243+27+27=1836)$.' The arithmetic in parentheses is missing the $81$ term; the actual sum of $729+729+243+27+27$ is $1755$, not $1836$. The STF-correct formula is $729+729+243+81+27+27 = 1836$. The *geometry* in the figure is correct as drawn; only the in-caption arithmetic has the omission."

### #corrigendum-item-2 (§11)

> "**Corrigendum item 2: Lambda decay product count (MS line 1292).** The current text reads 'Missing are 378 loops, which are the other decay products of the lambda.' The correct count is $351$ loops, not $378$. By direct subtraction, $2187 - 1836 = 351$, and $351$ decomposes cleanly in the STF basis as $243 + 81 + 27$ (iterations 5, 4, and 3). The value $378$ is the loop count of the down-quark cluster described earlier on the same page; the two quantities appear to have been accidentally interchanged."

---

## Dale & Briddell 2026 Addendum — *STF Iteration Sequence as the Canonical b=e Diagonal Orbit of QA*

### #addendum-structural-upgrade (Abstract)

> "This is a structural addendum to our companion paper ... The companion paper takes Briddell's STF iteration sequence $\{3, 9, 27, 81, 243, 729, 2187\}$ as a postulate inherited from the FST primary text and certifies the integer claims that depend on it. Here we observe that this sequence is not free: it is the unique self-similar orbit of the QA element rule $a = b + 2e$ along the diagonal $b = e$. Consequently every integer claim in Certificates I and II of the companion paper lives on a single QA orbit, and the choice of base 3 is forced — not selected — by QA structure."

### #addendum-diagonal-proposition (§2, Proposition 1)

> "The orbit of $(1,1)$ under $b_{n+1} = a_n$ on $D$ is $(1,1,2,3) \to (3,3,6,9) \to (9,9,18,27) \to (27,27,54,81) \to (81,81,162,243) \to (243,243,486,729) \to (729,729,1458,2187) \to \cdots$ The $a$-sequence is $a_n = 3^{n+1}$."

### #addendum-stf-identification (§2, Corollary 1)

> "The first seven $a$-values of the $b=e$ diagonal orbit are $\{3, 9, 27, 81, 243, 729, 2187\}$, which is the STF iteration sequence used in Section 2.3 of the companion paper, citing Briddell [Chap. 15]."

---

## Briddell 2020 *Structural Physics* — external referenced work

### #briddellbook2020-external

Don Briddell, *Structural Physics* (book-length monograph, 2020). Not on disk in this repo. Cited as `briddellbook2020` throughout the Dale-Briddell 2026 v2 paper. Relevant chapters:

- **Chapter 15**: STF hierarchy identification of the first seven iterations $\{3, 9, 27, 81, 243, 729, 2187\}$ with first-generation Standard Model particles.
- **Chapter 2**: Fig. 2.64 (proton STF decomposition) and the surrounding text at MS lines 1286, 1292, 1313–1326, 1580–1587, and line 795. Contains the two corrigendum-item typos (line 1286 and line 1292) flagged by the v2 paper §11.

External referenced only; all verbatim quotes in this file come from the Dale-Briddell 2026 v2 paper's own direct citations of `briddellbook2020`, which are reproduced within the v2 paper's source-locator anchors above.
