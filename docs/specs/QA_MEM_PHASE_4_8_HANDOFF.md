<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-MEM Phase 4.8 cross-session handoff doc. Captures the open state of Phase 4.8 + the unresolved QA_MEM_TRANSITION_TO_MEMORY architectural-review forward so a fresh session can pick up without conversation context. Drafted 2026-04-17 by claude-main-1439 at session-end after Will's directive "make sure this can be picked up by a fresh session". -->

# QA-MEM Phase 4.8 — Cross-Session Handoff

**Status:** Phase 4.8 KICKOFF shipped 2026-04-17 (commit `dd04127`, pushed to `origin/main`). Body items deferred. One unresolved architectural-review forward (see §3) awaits source text + scope decision before any draft.

**Authoring session:** `claude-main-1439` / `corpus-heartmath`. Session done broadcast at 2026-04-17T19:04Z.

**Why this doc exists:** an architectural review proposed in a different session reached this session as a relayed quote ("ship as scope-doc proposal first, memory entry follows ... Phase 4.7 sequencing risk is the load-bearing finding"). The originating session's full review text was not recovered before context-end. Lesson Will surfaced: *the review needs an anchor in the repo or it dies in conversation context.* This file is that anchor.

---

## 1. What shipped (Phase 4.8 KICKOFF, commit `dd04127`)

HeartMath corpus ingestion. Routed through QA-MEM per Phase 4.6 Wildberger/Haramein pattern after Will's correction "why aren't you using QA-MEM?".

**Files added/modified in `dd04127`:**
- `Documents/heartmath_corpus/tomasino_1997_water_em_storage.pdf` (190KB)
- `Documents/heartmath_corpus/danielson_2014_hospital_wellness.pdf` (425KB)
- `Documents/heartmath_corpus/oschman_2015_heart_bidirectional_scalar_antenna.pdf` (510KB)
- `Documents/heartmath_corpus/edwards_2018_cfp_heartmath_psychology.pdf` (499KB)
- `tools/qa_kg/fixtures/source_claims_heartmath.json` (4 SourceWorks, `claims=[]`, `domain=""`)
- `docs/theory/heartmath_phase4_8_excerpts.md` (stub anchors per paper)
- `tools/qa_kg/CORPUS_INDEX.md` (new HeartMath section + cross-ref rows)
- `qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json` (rebootstrap)
- `qa_alphageometry_ptolemy/_meta_ledger.json` (regenerated, 210 entries)

**Origin search:** `scholar.google.com/scholar?start=130&q=heartmath&hl=en&as_sdt=4007` (results 131-140 of `q=heartmath`).

**Verification:** meta_validator exit 0, all gates PASS; QA axiom linter CLEAN; 38 qa_kg unit tests PASS; commit_intent broadcast, no veto received.

---

## 2. Phase 4.8 body — deferred items (in priority order)

Each item is independently committable. Sequence is recommended, not required. Cross-references in the §1 source files anchor each to the artifacts they touch.

1. **Verbatim excerpt extraction** — replace STUB sections in `docs/theory/heartmath_phase4_8_excerpts.md` with p-numbered quotes pulled from the four on-disk PDFs (`pdftotext` or `pypdf` first-page metadata for locators). Each PDF gets 3–5 anchors following the Phase 4.6 Haramein schema (`#<paper-id>-<topic-or-eq>`).
2. **Claims array population** — populate `claims` in `tools/qa_kg/fixtures/source_claims_heartmath.json` from the §2.1 anchors. Schema mirrors `source_claims_haramein.json`: `id`, `work`, `quote`, `source_locator` (`file:docs/theory/heartmath_phase4_8_excerpts.md#<anchor>`), `extraction_method: "manual"`, `confidence: 1.0`, `valid_from`, `domain`, `title`. After this, [228] determinism cert needs a rebootstrap (delete `expected_hash.json`, rerun meta_validator).
3. **Domain taxonomy extension** — add `psychophysiology` to `tools/qa_kg/domain_taxonomy.json` `domains` array + a description entry. Re-validate [254] R10 (closed-set domain check). Retro-stamp every HeartMath SourceWork + SourceClaim from `""` to `psychophysiology`. Decision point first: `psychophysiology` vs alternatives like `physiology`, `cardiology`, or splitting into multiple (Tomasino + Oschman → `svp`, Danielson + Edwards → `psychophysiology`). Recommend `psychophysiology` for all four to keep the corpus cohesive.
4. **Ingress-allowlist extension** — add `Documents/heartmath_corpus/` to `DOCUMENTS_PDF_INGRESS_PREFIXES` in `llm_qa_wrapper/cert_gate_hook.py` (line ~90) **AND** the case statement in `.claude/hooks/pretool_guard.sh` (line ~31). Both edits are `WRAPPER_SELF_MODIFICATION` — they route through Codex quarantine review. **Blocker: Codex bridge has been dead 7+ days as of 2026-04-17** (see `memory/feedback_cert_gate_bridge_health.md`). Until Codex bridge alive, future PDFs must be written via cwd-scoped commands (no `Documents/` substring in the Bash command body).
5. **Cert family candidate `qa_heartmath_coherence_cert_v1`** — grounds in Danielson 2014 + extracted HRV-coherence quotes + the OB 2026-03-25 research-agenda Thread 3 mapping (HI ↔ cardiac coherence ratio; brain–heart cross-coherence ↔ QA Markovian coupling; Schumann ↔ mod-9/24 orbit harmonics). Cert family registers in `qa_alphageometry_ptolemy/qa_meta_validator.py` FAMILY_SWEEPS list. Suggest using Phase 4.6 Haramein cert [218] as the structural template.
6. **Corpus expansion** — Scholar `start=130` is one slice. Pull the remaining pages (`start=0` through `start=120`, then `start=140+`) for the same query, plus McCraty's peer-reviewed GCI papers from `heartmath.org/research/research-library/` and Radin's IONS publications on presentiment / RDNG. Both McCraty + Radin are explicitly called out in the OB 2026-03-25 entry as load-bearing for Thread 3.

---

## 3. Open architectural-review forward (NOT YET DRAFTED)

**Proposal received via cross-session relay (sourceless quote):** ship a forward-looking architecture doc at `docs/specs/QA_MEM_TRANSITION_TO_MEMORY.md`, distinct from `docs/specs/QA_MEM_SCOPE.md`'s execution focus, then a one-line `MEMORY.md` pointer. Reasoning given: "Phase 4.7 sequencing risk is the load-bearing finding."

**What's missing before this can be drafted:**
- **Source text** of the original architectural review. The relayed snippet contained only the recommendation, not the analysis. Without the underlying review, any draft would be a fresh architectural pass under Claude's read, not a mirror of the original session's finding.
- **Scope decision** for the doc. Candidates: (a) OB-primary → QA-MEM-primary handover plan, (b) A-RAG-current → QA-MEM-current retrieval handover plan, (c) both, (d) something specific to the unstated Phase 4.7 sequencing risk.

**Possible recovery paths for a fresh session:**
1. Check OB thoughts from 2026-04-17 (after 19:04 UTC) for any capture from the originating session.
2. Check `qa_lab/logs/codex_bridge.log` mtime + recent entries — if the originating session was Codex-bridge-driven, the review may be in the bridge log.
3. Check the qa-collab event bus event log around the same window for related broadcasts (try `mcp__qa-collab__collab_read_events` filtered to `architectural_review` or similar topic).
4. Check `~/wt-papers/` and `~/wt-certs/` worktrees for in-flight Phase 4.7 sequencing scratch files — the originating session may have been one of those.
5. If none of the above recover the review text, ask Will to either re-paste the originating session's full message or authorize a fresh architectural pass (which would be Claude's read, explicitly flagged as such in the doc preamble).

**If/when drafted, the doc should at minimum cover:**
- The current state (Phase 4.6 + 4.7 + 4.8-kickoff: Wildberger/Haramein/Levin/Briddell/Iverson/Keely/Parker-Hull/Whittaker/Philomath/HeartMath corpora ingested as SourceWorks + SourceClaims; OB still in active use; A-RAG indexed at `tools/qa_retrieval/` over ~58k messages).
- The intended end-state (QA-MEM as primary memory and primary retrieval; OB and A-RAG demoted to legacy / archival).
- The transition-state invariants (which queries route where during transition; how authority-tiering interacts with OB recency; the cert gates that enforce no-double-counting between OB and QA-MEM during overlap).
- The Phase 4.7 sequencing risk (whatever it actually is — see "missing source text" above).
- Decision points and unresolved scope questions.

---

## 4. Outstanding tracked work — broader QA-MEM (not Phase 4.8 specific)

For full context the next session should be aware of these adjacent open items (all pre-existing as of `dd04127`):

- **Phase 4.7 backlog** in `tools/qa_kg/CORPUS_INDEX.md` (§ "Phase 4.7 hardening backlog"): Haramein 2008 SourceWork hole, Wildberger+Rubine 2025 ingestion, `Documents/levin_corpus/` + `Documents/whittaker/` + `Documents/heartmath_corpus/` ingress allowlist extension, `tools/qa_kg/tests/test_corpus_index.py` enforcing index ↔ fixture ↔ on-disk consistency, FST scaffolding-vs-primary audit, batch-download pipeline hardening.
- **QA-MEM glossary** (`memory/project_qa_mem_glossary_future_work.md`) — hand-curated compendium of terms + abbreviations as first-class QA-MEM data; cert-gated; disambiguation enforced; Will surfaced 2026-04-17 during Phase 4.6 correction. Sequenced after `test_corpus_index.py` + Levin title resolution.
- **QA-MEM Beta-A+B gate-6 head-to-head fail** (`memory/project_qa_mem_beta_testing.md`) — Phase 4.7 action items: graph-expansion + contradicts-aware authority relaxation. Distinct from §3 above; this is a known empirical finding with a known remediation path.

---

## 5. Pointers (single-source-of-truth files)

- **OB thought** — `608d2f66-1b54-47b8-a045-3d644d59500b` (2026-04-17T19:03Z), full Phase 4.8 KICKOFF observation.
- **Memory** — `memory/project_qa_mem_phase_4_8_heartmath.md` (project type, indexed in `MEMORY.md`).
- **Corpus index** — `tools/qa_kg/CORPUS_INDEX.md` (§ HeartMath corpus + cross-reference index rows).
- **Excerpts stub** — `docs/theory/heartmath_phase4_8_excerpts.md` (per-paper anchors + Phase 4.8 follow-up actions footer).
- **SourceWorks fixture** — `tools/qa_kg/fixtures/source_claims_heartmath.json`.
- **Determinism artifacts** — `qa_alphageometry_ptolemy/qa_kg_determinism_cert_v1/expected_hash.json` + `qa_alphageometry_ptolemy/_meta_ledger.json` (rebooted in `dd04127`; will need another reboot after §2 item 2 lands).
- **Commit body** — `git show dd04127` for the full deferred-items list and verification record.
- **Adjacent specs** — `docs/specs/QA_MEM_SCOPE.md` (execution focus), this file (Phase 4.8 handoff), planned future `docs/specs/QA_MEM_TRANSITION_TO_MEMORY.md` (forward-looking architecture; §3 above).
