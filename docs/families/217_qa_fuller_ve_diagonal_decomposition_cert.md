# [217] QA Fuller VE Diagonal Decomposition Cert

**Status:** PASS (meta-validator)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Buckminster Fuller, *Synergetics* (1975)

## Claim

Fuller's cuboctahedral / vector-equilibrium shell count
$$ S_n = 10n^2 + 2, \qquad n \geq 1 $$
(sequence: 12, 42, 92, 162, 252, 362, 492, 642, 812, …) decomposes
across QA integer diagonals by the residue $n \bmod 3$:

- $n \not\equiv 0 \pmod 3$: $S_n$ sits on the $b=e$ diagonal $D_1$ with tuple $(S_n/3,\; S_n/3,\; 2S_n/3,\; S_n)$.
- $n \equiv 0 \pmod 3$: $S_n$ sits off $D_1$ on a sibling odd-divisor diagonal $D_k$ where $(2k+1) \mid S_n$.

**Proof.** $S_n \bmod 3 = (n^2 + 2) \bmod 3$. Since $n^2 \bmod 3 \in \{0, 1\}$ (1 when $n \not\equiv 0 \pmod 3$, 0 otherwise), $S_n \bmod 3 = 0$ iff $n \not\equiv 0 \pmod 3$. $\square$

## Significance

First documented physical hierarchy whose QA decomposition is **mixed across two diagonal classes**, not entirely on the canonical Sierpinski diagonal $D_1$. Complements the FST/STF paradigm (Briddell, entirely on $D_1$) by demonstrating that the diagonal-classification methodology can resolve structure within a hierarchy, not only between hierarchies.

The mod-3 selection is itself QA-natural — the triune residue partition on the canonical Sierpinski diagonal — giving a **2:1 density ratio** of on-diagonal to off-diagonal shells (6 of 9 in the first 9 shells).

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_fuller_ve_diagonal_decomposition_cert_v1/qa_fuller_ve_diagonal_decomposition_cert_validate.py`
- Fixtures: `fvdd_pass_shell_decomposition.json` (PASS), `fvdd_fail_wrong_mod3_classification.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| FVDD_1 | schema version matches |
| FVDD_FORMULA | shell counts for $n=1..9$ match $10n^2+2$ |
| FVDD_MOD3 | on-diagonal claim aligns with $n \bmod 3 \ne 0$ |
| FVDD_DIAGONAL | on-diagonal shells carry valid $b=e$ tuple |
| FVDD_OFFDIAGONAL | off-diagonal shells expose valid sibling $D_k$ candidate |
| FVDD_COMPUTATIONAL | exhaustive re-derivation for $n=1..9$ matches claims |
| FVDD_SRC | source attribution (Fuller + Will Dale) |
| FVDD_WITNESS | at least 3 witnesses covering on/off/mod3_structural |
| FVDD_F | fail_ledger well-formed |

## Cross-references

- Foundation: `docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md`
- FST paradigm: `qa_alphageometry_ptolemy/qa_fst/qa_fst_addendum_b_equals_e_diagonal.tex`
- Triune structure: `docs/theory/QA_SYNTAX_SVP_SEMANTICS.md`
- Sibling certs: FST family (entirely on $D_1$), [207] circle impossibility (role of observer projection), [208] quadrance product (role-distinction under value-equality).

## Open questions (not in this cert)

- Do higher shells ($n > 9$) maintain the 2:1 density ratio asymptotically? (Trivially yes by counting, but the multi-divisor structure of off-diagonal shells grows with $n$.)
- Which sibling diagonal $D_k$ is "canonical" for each off-diagonal shell when multiple odd divisors exist (e.g.\ $n=9$: $k \in \{3, 14, 101\}$)? The cert accepts any valid $(k, b)$ but does not yet pick one.
- Do other cuboctahedral / jitterbug derivatives (icosahedron with 12 vertices, VE-edge count $24n$, VE-face count $8n^2$) sit on $D_1$ or reveal further diagonal structure?
