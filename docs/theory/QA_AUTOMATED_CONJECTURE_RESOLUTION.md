# QA ↔ Automated Conjecture Resolution (Ju et al. 2026) Mapping

**Status:** structural mapping, draft 2026-04-15
**Primary source:** Ju, Gao, Jiang, Wu, Sun, Chen, Wang, Wang, Wang, He, Wu, Xiao, Liu, Dai, Dong (2026). *Automated Conjecture Resolution with Formal Verification*. arXiv:2604.03789, submitted 2026-04-04.
**Related:** `memory/feedback_map_best_to_qa.md` (methodology rule), cert families [223]/[224] (protocol certs), cert [122] (QA empirical observation verdict vocabulary), `docs/specs/QA_CODING_GUARDRAIL_ARCHITECTURE.md` (cert-submission boundary this mapping targets).

---

## 1. What the paper actually does

Ju et al. introduce a two-stage pipeline for autonomous mathematical research:

- **Rethlas** — informal reasoning agent that produces candidate proofs from a conjecture statement.
- **Archon** — formal verification agent that translates informal arguments into Lean 4 projects and closes proof gaps via retrieval + tactic search.
- **Retrieval layer** — Matlas (Mathlib retrieval) and LeanSearch (Lean theorem lookup) give both agents access to prior formal mathematics.

The paper demonstrates the pipeline by automatically resolving an open problem in commutative algebra with a machine-checkable Lean 4 proof. The architectural contribution is the **two-agent split**: informal reasoning first, formal closure second, with retrieval as shared substrate.

## 2. Why this is worth mapping to QA

QA already treats mathematical discovery as a reachability problem over theorem states. The canonical loop is:

```
conjecture → prove → certify → validate
```

Ju et al.'s pipeline is structurally isomorphic, but adds a rigorous formal endpoint (Lean 4) that QA currently lacks. QA certificate validators are Python + JSON with symbolic/exhaustive witnesses, not Lean proofs. The paper shows the state of the art for "machine-resolved conjecture" and makes it concrete what the gap is.

Per `memory/feedback_map_best_to_qa.md`: find what works best, extract the generator, map through QA. Ju et al. is the current best-performing general conjecture-resolver that produces machine-verified output. The QA contribution here is not reimplementing Rethlas+Archon; it is **typing the failure modes** that the paper elides into "proof not found."

## 3. Operator mapping (QA-native translation)

| Paper concept | QA concept | Notes |
|---|---|---|
| Rethlas.generate_candidate | QA conjecture generator over orbit/diagonal structure | QA's advantage: structured state space gives natural candidate classes (e.g. "fixed-d hyperbola," "sibling diagonal," "orbit invariant") |
| Archon.formalize | QA formal-proof bridge | Current QA bridges are Python validators + symbolic witnesses; Lean 4 is not wired in |
| Lean 4 proof closure | QA certificate validator `--self-test` passing | Different formal systems, same role: machine-checkable closure |
| Matlas/LeanSearch retrieval | `tools/qa_retrieval/query.py` A-RAG system | QA has its own retrieval over conversation corpus; cert-family registry is the analogue of Mathlib |
| Lean proof gap | QA **typed obstruction** | This is the key QA upgrade — see §4 |
| Proof-not-found | QA verdict `INCONCLUSIVE` or typed failure | Paper has one failure class; QA has four |

## 4. Where QA extends the paper — typed obstructions

The paper's failure model is binary: proof found or proof not found. QA's failure algebra requires distinguishing:

- **`formal_gap`** — Lean (or equivalent) cannot close the proof, but no QA-level invariant is violated. The conjecture may be true; the tools are insufficient.
- **`qa_obstruction`** — A QA invariant (orbit structure, parity, modulus, Theorem NT firewall) prevents the conjecture. The conjecture is false under QA axioms, independent of proof-search budget.
- **`generator_insufficient`** — The declared generator set does not span the statement's state space. The conjecture may be true, but this tool configuration cannot reach it.
- **`inconclusive`** — Budget exhausted without reaching any of the above.

Typed obstructions let downstream certs reason about *why* a conjecture resists proof, which is what converts failed attempts into usable information rather than noise. This is the QA extension of Ju et al.

## 5. Where QA does NOT match the paper

- **Lean 4 backend absent.** No `formal/lean/` tree exists in the repo. Cert validators are Python + JSON; symbolic/exhaustive, not proof-theoretic. A QA cert that declares `formal_backend: lean4` is aspirational until a Lean bridge is built.
- **Commutative-algebra domain absent.** The paper's demonstration problem (commutative algebra) is not a current QA domain. The architectural pattern maps; the specific problem does not.
- **"Interestingness / selection" policy differs.** The paper leaves cert-worthy-discovery selection implicit. QA has Gate 0 (mapping protocol), the meta-validator registry, and [223]/[224] design-contract certs. QA's selection policy is more explicit and more conservative (a cert must verify, not merely be plausible).
- **Retrieval scope differs.** Matlas/LeanSearch retrieve over formal mathematics; `tools/qa_retrieval` retrieves over the project's conversation corpus. Both serve candidate-generation, but the substrates are disjoint.

## 6. Proposed cert family scaffold

**Family:** `[248] QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1`
**Directory:** `qa_alphageometry_ptolemy/qa_formal_conjecture_resolution_cert_v1/`
**Gate 0:** `mapping_protocol_ref.json` (inline — the cert schema is self-contained)
**Validator:** `qa_formal_conjecture_resolution_cert_validate.py` with `--self-test`

### Cert record schema (fields)

- `schema_version`: `"QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1"`
- `conjecture_id`: string (e.g. `QA-2026-001`)
- `source_claim`: string — conjecture in natural language
- `qa_translation`: string — restatement in QA operators (orbit, diagonal, generator, T-operator)
- `candidate_kind`: enum `{orbit_invariant, diagonal_class, generator_chain, bridge, external}`
- `formal_backend`: enum `{qa_symbolic, qa_exhaustive, lean4_stub, python_proof}` — `lean4_stub` allowed to express aspiration; must declare so explicitly
- `proof_status`: enum `{proved, formal_gap, qa_obstruction, generator_insufficient, inconclusive}`
- `failure_mode`: string | null — **required non-null iff `proof_status != proved`**
- `generator_set`: array of strings (non-empty) — operators/axioms used
- `nt_compliance`: boolean — Theorem NT (observer-projection firewall) declared clean
- `witness_path`: string — path to fixture or symbolic witness file
- `verdict`: enum `{CONSISTENT, PARTIAL, CONTRADICTS, INCONCLUSIVE}` — matches [122] bridge vocabulary

### Validator checks (minimum)

1. `FCR_1` — schema_version matches
2. `FCR_2` — generator_set non-empty
3. `FCR_3` — if `proof_status != proved`, `failure_mode` is a non-empty string
4. `FCR_4` — `nt_compliance` is a boolean; if false, `failure_mode` must name the NT violation
5. `FCR_5` — `verdict` in the allowed set
6. `FCR_6` — `witness_path` exists or is marked `"symbolic_inline"`
7. `FCR_7` — if `formal_backend == lean4_stub`, record an `open_questions` field naming the Lean gap

### Fixtures (minimum set)

- `fcr_pass_proved.json` — a simple QA conjecture closed by exhaustive witness (e.g. "every tuple in S_9 with b=e satisfies d=2b")
- `fcr_pass_formal_gap.json` — conjecture with `formal_backend: lean4_stub`, `proof_status: formal_gap`, `failure_mode` explaining the gap
- `fcr_fail_missing_failure_label.json` — `proof_status: formal_gap` with `failure_mode: null` → must reject

## 7. Open questions for Will

1. **Lean bridge scope.** Is building an actual Lean 4 bridge (`formal/lean/` tree, compilation in meta-validator) in scope, or do we stub `formal_backend: lean4_stub` indefinitely and treat the cert as architectural placeholder?
2. **Protocol vs. witness family.** Should [248] be a protocol family (like [223]/[224], referenced by other certs), or a witness family (like [244] Mutation Game, standing on its own)?
3. **Retroactive application.** Should existing cert families retroactively populate [248] records (treating every existing cert as a conjecture-resolution event), or does [248] start fresh and only cover new resolutions?

## 8. Corrections to the ChatGPT precooked plan (recorded for provenance)

The initial mapping plan (ChatGPT, 2026-04-15) proposed paths that don't match repo conventions. Corrections:

- ✗ `qa_core/adapters/papers/…_adapter.py` — `qa_core/` does not exist (see MEMORY.md: "QA_Engine duplicated in ~50 files"). No adapter layer.
- ✗ `qa_alphageometry_ptolemy/certs/[2XX]_QA_FORMAL_CONJECTURE_RESOLUTION_CERT.v1/` — wrong convention. Cert dirs are `qa_alphageometry_ptolemy/qa_<name>_cert_v1/`; family numbers live in `qa_meta_validator.py` FAMILY_SWEEPS and `docs/families/N_slug.md`.
- ✗ `formal/lean/qa_paper_bridges/…` and `formal/tla/…` — `formal/` tree does not exist.
- ✗ `qa_alphageometry_ptolemy/cert_index.json` — no such file. Registry is FAMILY_SWEEPS.
- ✓ The architectural intuition (two-agent split, typed failures, cert-family target) is correct.
- ✓ The arXiv ID (2604.03789) is real, verified via WebFetch 2026-04-15.

This correction record is itself an example of the Layer 3 Codex-review failure class: plausible-looking paths that don't match the repo. The primary-source gate catches the missing citation class; Codex review at cert-submission catches the wrong-paths class.

## References

- Ju et al. (2026). *Automated Conjecture Resolution with Formal Verification.* arXiv:2604.03789. Submitted 2026-04-04. https://arxiv.org/abs/2604.03789
- `docs/specs/QA_CODING_GUARDRAIL_ARCHITECTURE.md` — three-layer guardrail that gates this mapping's cert-submission step.
- `memory/feedback_map_best_to_qa.md` — methodology rule: find best, extract generator, map through QA.
- Cert [122] `QA_EMPIRICAL_OBSERVATION_CERT` — verdict vocabulary precedent (CONSISTENT/PARTIAL/CONTRADICTS/INCONCLUSIVE).
- Certs [223]/[224] — design-contract family pattern (schema + validator + fixtures + meta-validator registration).
- Cert [244] `QA_MUTATION_GAME_ROOT_LATTICE_CERT.v1` — recent cert convention reference.
