# TLA Harness Remediation Plan

## 1. Incident
- Summary: A recent public-facing TLA+ contribution was rejected by external maintainers as incoherent, insufficiently grounded, and inappropriate for the target repository.
- Core failure: The harness let internal formal structure, TLC success, proof-ledger completeness, and internal “submittable/publication-ready” posture stand in for external admissibility. Formal artifacts such as `.tla`, `.cfg`, proof ledgers, manifests, and public-facing formal README/prose were not under the same review pressure as Python.
- Immediate lesson: Internal cert discipline and model-check success are necessary but not sufficient. Public formal-methods work needs explicit outsider translation, semantics-vs-bounds clarity, skeptical review, repository-fit review, and human approval.

## 2. Objectives
- Primary objective: Prevent recurrence of polished but externally incoherent public formal-methods submissions.
- Secondary objectives:
  - Separate internal readiness from public-submission readiness.
  - Add fail-closed controls for formal artifact classes, not just Python.
  - Detect vacuous formal structure and toy negative tests before submission.
  - Measure both formal quality and external admissibility under blind evaluation.

## 3. Pass 1 — Guardrails
- Goal: Ship the minimum fail-closed patch set that materially changes the public formal-methods ship path.
- Scope:
  - Expand quarantine/review coverage to:
    - `.tla`
    - `.cfg`
    - formal-methods `README.md`
    - `QARM_PROOF_LEDGER.md`
    - `ARTIFACT_MANIFEST.md`
  - Add a minimal formal-publication gate.
  - Block premature `publication-ready` / `submittable upstream` wording before gate pass.
  - Add only the highest-value lint checks first: tautological invariants, missing outsider translation, missing semantics-vs-bounds separation, toy negative tests.
  - Insert `external-fit-cleared` between `review-ready` and `submitted`.
- Required artifacts:
  - `audience_translation.md`
  - `semantics_boundary.md`
  - `repo_fit_review.json`
  - `skeptical_review.json`
  - `human_approval.json`
- Status: Complete
- Outputs:
  - hook and gate changes for formal artifact quarantine and commit/push blocking
  - regression tests for gate failures and premature publication claims
  - workflow/policy doc updates adding `external-fit-cleared`

## 4. Pass 2 — Bypass Audit
- Goal: Try to evade or spoof the new controls mechanically.
- Questions:
  - Can renamed or nested formal artifacts escape gating?
  - Can direct `git commit` / `git push` paths bypass review?
  - Can empty, generic, or templated review artifacts pass?
  - Are any enforcement paths documented but dead?
- Attack cases:
  - path and glob holes
  - renamed files
  - nested formal docs
  - empty JSON artifacts
  - fake skeptical review
  - fake human approval
  - submission prose outside expected locations
- Status: Complete
- Outputs:
  - bypass report
  - targeted fixes
  - regression tests for each closed bypass
  - findings from the first bypass round:
    - critical: `repo_fit_review.json`, `skeptical_review.json`, and `human_approval.json` accepted content-free or automated-looking payloads
    - major: blocked `publication-ready` / `submittable upstream` wording was only scanned in Markdown, so `submission.txt` passed
    - major: nested `README.md` above descendant `.tla` content was not always classified as formal-publication material
    - no bypass confirmed in this round for direct `git commit` / `git push` once formal paths were correctly classified

## 5. Pass 3 — Blind Evals
- Goal: Test whether the post-remediation harness can distinguish legitimate formalization from polished nonsense under uncertainty.
- Eval layers:
  - Blind generation
  - Blind review
  - Blind repair
- Required score axes:
  - `formal_validity_score`
  - `external_admissibility_score`
  - semantic adequacy
  - outsider comprehensibility
  - invariant non-vacuity
  - semantics-vs-bounds clarity
  - repository-fit plausibility
- Fixture classes:
  - known-good TLA+ style examples
  - polished but bad or jargon-heavy artifacts
  - QA-to-TLA translation tasks
  - repair cases where the correct answer is still reject
- Status: Complete
- Outputs:
  - benchmark set under `evals/tla_blind/`
  - rubric and score schema with mandatory `formal_validity_score` and `external_admissibility_score`
  - runner/scaffold for listing cases, showing model-visible prompts, and initializing blank scorecards
  - starter fixtures for generation, review, and repair
  - execution scaffold for scoring the current heuristic judgment layer against hidden labels in the starter review/repair corpus

## 6. Pass 4 — Post-Eval Tightening
- Goal: Use blind-eval failures to refine the gates, audits, and review logic.
- Focus:
  - false accepts
  - false rejects
  - polish-for-legitimacy confusion
  - weak outsider translation
  - weak repository-fit judgment
- Status: Not started

## 7. Open Risks
- Internal consistency may still be overweighted relative to external intelligibility.
- Skeptical review may degrade into template-filling instead of adversarial critique.
- Repo-fit review may become ceremonial.
- Human approval may become a rubber stamp.
- TLC success and non-vacuity bookkeeping may still bias judgment more than they should.
- QA-private terminology may continue to leak into public-facing formal artifacts without translation.

## 8. Decisions
- Confirmed:
  - Pass 1 is the minimal patch set, not the full idealized design.
  - Phase 1 will fail closed on a small required artifact set rather than waiting for a full source-grounding schema.
  - Pass 2 remains adversarial and mechanical.
  - Pass 3 remains blind and evaluative.
  - Blind evals must report `formal_validity_score` and `external_admissibility_score` separately.
  - Pass 3 will keep hidden labels and reference answers physically separate from model-visible prompts.
  - Pass 3 execution should treat `accept` as a high bar; artifacts with substantive clarity defects should default to `revise`, not `accept`.
  - TLC success alone must never dominate public-submission judgment.
  - Public formal-methods submission requires stricter controls than internal cert readiness.
  - Pass 2 hardened validators against content-free review artifacts instead of treating mere file presence as sufficient.
  - Pass 2 widened publication-claim scanning beyond Markdown to close submission-text bypasses.
  - Pass 2 widened README/formal-doc detection to cover nested formal trees rather than only sibling `.tla` / `.cfg` files.
  - Current repo findings supporting this split:
    - `qa_alphageometry_ptolemy/QARM_PROOF_LEDGER.md` labels `QAAxioms.tla` “submittable upstream”.
    - `qa_alphageometry_ptolemy/TLC_FAILURE_ANALYSIS_REPORT.md` labels a QA/QARM spec “publication-ready”.
    - `llm_qa_wrapper/cert_gate_hook.py` and its tests show strong Python-path gating, but no comparable formal-artifact gate.
    - `docs/specs/QA_CRITICAL_REVIEW_AGENT_SPEC.md` exists as policy but is not wired into the public formal-methods ship path.
- Pending:
  - Exact path set for “formal-methods README/docs”.
  - Whether `source_grounding.json` is Phase 1 or deferred to a later pass.
  - CI vs manual enforcement boundary for the blind eval harness.
  - Who is authorized to issue `human_approval.json`.

## 9. Remaining Gaps
- Phase 1 still uses heuristic outsider-translation and vacuity checks rather than a deeper semantic analyzer.
- No source-grounding artifact is enforced yet.
- Pass 1 does not yet distinguish templated skeptical review from genuinely adversarial review.
 - Pass 2 has not yet attacked renamed submission-prose filenames outside the current `submission/pr/cover_letter` set.
 - Pass 2 has not yet tested content-free artifacts that are syntactically rich but semantically hollow.
- Pass 3 is currently a starter corpus, not a large benchmark set.
- Pass 3 has scaffolded result handling but has not yet been wired into CI or pre-submission automation.
 - Pass 3 generation tasks still have no automated generator wired into blind execution; they are currently scaffold-only prompts.

## 10. Success Criteria
- A public-facing TLA+/formal-methods artifact cannot be committed or pushed toward submission without explicit external-fit clearance.
- Mechanical bypass attempts fail or are detected.
- Blind review can reliably reject polished nonsense.
- Blind generation improves outsider readability and semantics clarity.
- The harness no longer treats internal formal structure as a proxy for external admissibility.

## 11. Current Status Snapshot
- Current phase: Pass 4 — Post-Eval Tightening
- Last completed milestone: Starter blind corpus executed against the current heuristic judgment layer; the initial false-accept on semantics/TLC conflation was closed by tightening visible-review heuristics and the blind accept threshold.
- Next action: Expand the blind corpus and wire execution into CI or pre-submission review so future judgment regressions are visible.
- Owner / executor: Codex post-eval tightening pass.
