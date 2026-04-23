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
- Status: Not started
- Outputs:
  - hook and gate changes
  - regression tests
  - workflow/policy doc updates

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
- Status: Not started
- Outputs:
  - bypass report
  - targeted fixes
  - regression tests for each closed bypass

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
- Status: Not started
- Outputs:
  - benchmark set
  - rubric and score schema
  - runner/scaffold
  - starter result bundle

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
  - Pass 2 remains adversarial and mechanical.
  - Pass 3 remains blind and evaluative.
  - Blind evals must report `formal_validity_score` and `external_admissibility_score` separately.
  - TLC success alone must never dominate public-submission judgment.
  - Public formal-methods submission requires stricter controls than internal cert readiness.
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
- No live public-formal-methods gate.
- No formal-artifact quarantine path.
- No enforced skeptical-review artifact on the ship path.
- No enforced repository-fit artifact on the ship path.
- No anti-vacuity lint for TLA+ invariants and negative tests.
- No blind generation/review/repair benchmark set.
- No regression tests for the recent rejection pattern.

## 10. Success Criteria
- A public-facing TLA+/formal-methods artifact cannot be committed or pushed toward submission without explicit external-fit clearance.
- Mechanical bypass attempts fail or are detected.
- Blind review can reliably reject polished nonsense.
- Blind generation improves outsider readability and semantics clarity.
- The harness no longer treats internal formal structure as a proxy for external admissibility.

## 11. Current Status Snapshot
- Current phase: Planning / tracker established
- Last completed milestone: Incident audit completed; remediation sequence split into Guardrails, Bypass Audit, Blind Evals, and Post-Eval Tightening.
- Next action: Implement Pass 1 minimal patch set.
- Owner / executor: Codex implementation pass, followed by adversarial Codex audit and blind-eval build pass.
