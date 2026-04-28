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
- Status: Complete
- Outputs:
  - deterministic blind-generation execution path writing result bundles under `evals/tla_blind/results/current_system/`
  - added `source_grounding_score` and `repo_comparables_evidence_score` to the blind rubric/schema
  - tightened scoring for visible semantics/TLC conflation, source grounding, and repository-fit comparable evidence
  - starter-corpus execution now rejects the recreated polished-bad TLA+ fixture before any PR-draft path

## 7. Pass 5 — Evidence Automation
- Goal: Replace the remaining human-legitimacy dependencies with evidence-backed automation for source fidelity and target-repo appropriateness.
- Focus:
  - machine-readable source grounding
  - machine-readable comparable-evidence
  - adversarial evidence checks
  - downgrade human approval to audit/override status
- Status: Complete
- Outputs:
  - `source_grounding.json` and `repo_comparables.json` added to the formal-publication gate as required evidence artifacts
  - source-fidelity and comparable-support checks added to blind and gate scoring
  - automated adversarial evidence findings now lower admissibility when grounding or repo-fit support is weak
  - human approval removed as a required truth gate and treated as an optional audit/override artifact

## 8. Pass 6 — Evidence-Quality Calibration
- Goal: Stress-test the evidence-backed gate and blind scorer against plausible but misleading evidence rather than only missing evidence.
- Focus:
  - overstated or cherry-picked source excerpts
  - style-only or adjacent-class comparables used to overclaim repository fit
  - borderline cases that should be `revise` rather than auto-`accept` or auto-`reject`
- Status: Complete
- Outputs:
  - a distinct deception benchmark layer under `evals/tla_blind/deception_corpus/`
  - expected decision and score-range labels for evidence-deception cases
  - runner and current-system executor integration for the deception layer
  - calibration checks showing the recreated polished-bad incident class remains blocked while sparse-faithful cases still pass

## 9. Pass 7 — Corpus Benchmark Sweep
- Goal: Measure corpus-level blind performance across all currently labeled blind-eval fixtures rather than relying on starter-case spot checks.
- Focus:
  - confusion matrices by domain
  - false accept / false reject tracking
  - score-distribution summaries
  - cross-domain error-pattern detection
- Status: Complete
- Outputs:
  - benchmark runner under `evals/blind_benchmark/benchmark_current_corpus.py`
  - machine-readable report: `evals/blind_benchmark/results/current/blind_corpus_benchmark.json`
  - human-readable report: `evals/blind_benchmark/results/current/blind_corpus_benchmark.md`
  - current measured status:
    - TLA+: 9 labeled fixtures, 100% decision accuracy, 0 false accepts, 0 false rejects, status call `balanced`
    - Lean 4: 5 labeled fixtures, 100% decision accuracy, 0 false accepts, 0 false rejects, status call `balanced`
    - cross-domain: 14 labeled fixtures, 100% decision accuracy on the current labeled corpora

## 10. Open Risks
- Internal consistency may still be overweighted relative to external intelligibility.
- Skeptical review may degrade into template-filling instead of adversarial critique.
- Repo-fit review may become ceremonial.
- Human approval may become a rubber stamp.
- TLC success and non-vacuity bookkeeping may still bias judgment more than they should.
- QA-private terminology may continue to leak into public-facing formal artifacts without translation.

## 11. Decisions
- Confirmed:
  - Pass 1 is the minimal patch set, not the full idealized design.
  - Phase 1 will fail closed on a small required artifact set rather than waiting for a full source-grounding schema.
  - Pass 2 remains adversarial and mechanical.
  - Pass 3 remains blind and evaluative.
  - Blind evals must report `formal_validity_score` and `external_admissibility_score` separately.
  - Pass 3 will keep hidden labels and reference answers physically separate from model-visible prompts.
  - Pass 3 execution should treat `accept` as a high bar; artifacts with substantive clarity defects should default to `revise`, not `accept`.
  - Pass 4 should add source-grounding and comparable-evidence scoring before attempting broader redesign.
  - Pass 5 should require authoritative evidence artifacts rather than letting human approval stand in for source fidelity or repo fit.
  - Pass 6 should be a distinct deception layer, not a remix of missing-evidence tests.
  - Pass 6 should include both false-accept pressure and sparse-but-faithful false-reject protection.
  - Pass 7 should benchmark the entire labeled corpus by domain before more threshold tuning.
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
  - CI vs manual enforcement boundary for the blind eval harness.
  - Whether optional human override artifacts should be normalized to one filename instead of several aliases.

## 12. Remaining Gaps
- Phase 1 still uses heuristic outsider-translation and vacuity checks rather than a deeper semantic analyzer.
- Pass 1 does not yet distinguish templated skeptical review from genuinely adversarial review.
 - Pass 2 has not yet attacked renamed submission-prose filenames outside the current `submission/pr/cover_letter` set.
 - Pass 2 has not yet tested content-free artifacts that are syntactically rich but semantically hollow.
- Pass 3 is currently a starter corpus, not a large benchmark set.
- Pass 3 has scaffolded result handling but has not yet been wired into CI or pre-submission automation.
- Pass 4 uses a deterministic generation executor, not the full live agent stack.
- Pass 5 still relies on heuristic excerpt/interpretation checks rather than full semantic entailment.
- Pass 5 still approximates target-repo comparability; it does not yet mine comparables automatically from the target repository.
- Pass 6 still uses handcrafted deception fixtures rather than mined real-world misleading evidence packets.
- Pass 7 currently covers only the available labeled corpora; zero observed errors here should trigger corpus expansion, not complacency.

## 13. Success Criteria
- A public-facing TLA+/formal-methods artifact cannot be committed or pushed toward submission without explicit external-fit clearance.
- Mechanical bypass attempts fail or are detected.
- Blind review can reliably reject polished nonsense.
- Blind generation improves outsider readability and semantics clarity.
- The harness no longer treats internal formal structure as a proxy for external admissibility.

## 14. Current Status Snapshot
- Current phase: Pass 11 — v1 baseline frozen
- The harness is now a general anti-slop legitimacy harness, not a TLA+
  remediation. v1 architecture + reproducibility instructions live at
  [`docs/specs/BLIND_EVAL_HARNESS_V1.md`](BLIND_EVAL_HARNESS_V1.md).
- Master runner: `python3 evals/run_all.py` runs all 5 non-live suites
  in ~5-10s and emits `evals/results/v1_baseline_report.{md,json}`.
  `--with-live-agent` flag adds the codex Upwork stress (~10-15 min).
- Tag: `v1-blind-harness` marks this baseline. Future regressions
  should be diagnosed against this tag.
- Pass milestones to date:
  - Pass 7: intrinsic vs submission-bundle-completeness split (`e87b1d1`).
  - Pass 7-c: deception regression suite (`c2bb195`) — 4 known gaps documented and tolerated.
  - Pass 7-b: charitable comment-extraction adapter (`84be587`) — 50% of TLA revise was extraction debt.
  - Pass 7-a: upstream corpus expansion (`1892e03`) — TLA 99 / Lean 128, intrinsic accept 55.6% / 100%.
  - Pass 7-d: third blind domain `evals/upwork_blind/` (`494ecd5`/`236e81e`/`80832f0`) — cross-domain sweep 22/22.
  - Pass 8: live-agent stress on Upwork suite (`86d8b3f`) — overclaim variant degraded both tasks; rush + minimal_tests caused keyword drop-out.
  - Pass 9: shared cross-domain core extracted at `evals/_blind_core/` (`3c0b755`) — behavior preserved byte-identically.
  - Pass 10: fourth domain `evals/swe_bench_blind/` (`16f3118`) — SWE-Bench Verified pilot. Cross-domain blind-benchmark **30/30**. Deception regression **34 fixtures**, still exit-0. Shared core held without extension.
  - Pass 11: freeze + document. v1 doc, master runner, baseline report, tag.
  - Pass 12: SWE-Bench live-agent stress (`d9c8553`) — 25 codex calls × 5 prompt variants. Codex resists overclaim framing on patches (unlike Upwork). Wrong-file routing surfaced on django-11211. 17/25 heuristic-accept.
  - Pass 13a: SWE-Bench execution pilot (`b462f11`) — `git apply --check` against real cloned repos showed only 1 of 9 heuristic-accepted codex patches actually applies. 89% of accepts were structurally malformed (hunk-header count mismatches, non-ASCII whitespace).
  - Pass 13b: added `git apply --check` boundary to SWE-Bench scorer (`fb57f97`, opt-in via task_spec keys). Pass-12 live-agent rescored: accept rate **68% → 12%** (17 → 3). Existing fixtures unchanged.
  - Pass 13c: SWE-Bench `FAIL_TO_PASS` execution on 4 surviving patches + 3 controls (`9eeea03`). Of 2 testable codex patches, **both fix their bugs** (3/3 + 1/1 FAIL_TO_PASS). One harness false-revise surfaced: django-11211/minimal_tests scored revise for canonical-file mismatch but the alternative location is also valid. Astropy@c0a24c1d uninstallable on Python 3.13 + numpy 2.x → 3 results couldn't be tested.
  - Pass 14a: tiered patch-relevance in SWE-Bench scorer (`47ce58a`). Replaced binary "must touch canonical file → -3" with tier-1/2/3/4 grading. `django-11211/minimal_tests` live-agent flipped revise → accept as Pass-13c truth predicted. Also recovered 27 SWE-Bench fixture metadata files Pass 10's commit silently dropped under the global *.json gitignore.
  - Pass 15: calibration dashboard for SWE-Bench (`da0e9de`). Designed truth 8/8; executed truth 4/4 TP, 0 FA, 0 FR on the testable Django subset. Wired into master runner. Tagged `v1.1-blind-harness` as the post-recovery baseline.
  - Pass 16: structural gate for Upwork (`c9673b2`) — `python -m py_compile` + import smoke + `pytest --collect-only`. Pass-8 live-agent rescore: 0/10 codex outputs flipped — codex's Upwork Python is structurally clean; gate is armed but currently silent.
  - Pass 14b: astropy environment via official SWE-Bench docker image (`swebench/sweb.eval.x86_64.astropy_1776_astropy-14539:latest`, Python 3.11.5 + frozen numpy + cython 0.29.30). Workflow: apply test_patch (adds Q-format column to test_identical_tables — sanity-checked: tests fail with `ValueError: ambiguous truth value` at this state), apply patch under test, run pytest. **All 3 untested astropy patches PASS FAIL_TO_PASS** (canonical 2/2, codex looks_done 2/2, codex minimal_tests 2/2). Calibration dashboard updated: executed truth now **7/7 testable, 7/7 TP, 0 FA, 0 FR** — full executable coverage on the Pass-12 sample. The harness's heuristic + apply-check + tiered patch-relevance produces 100% precision and 100% recall on this set. Tagged `v1.2-blind-harness`.
  - Pass 18: harsher Upwork live-agent stress targeting the Pass-16 structural gate (`evals/upwork_blind/run_live_agent_pass18.py`). 5 engineered prompt variants × 2 tasks = 10 codex calls. Variants designed to trip each sub-check: `invent_deps` (push toward undeclared imports → import smoke), `top_level_demo_io` (push toward top-level FileIO → import smoke), `nested_package` (push toward relative imports that break standalone import → import smoke), `broken_parametrize` (push toward parametrize signature mismatch → `pytest --collect-only`), `compact_packed` (push toward malformed minified code → `py_compile`). Each output scored twice — once with the gate enabled (current harness) and once with the gate bypassed — so the gate's contribution is isolated by quadrant. Result: **structural gate fired alone (heur-pass × struct-fail) on 0/10 outputs**, total full-cascade fail 3/10 (all 3 are the same pre-existing `bugfix_factorial_zero` `required_keywords` heuristic finding from Pass 8 — unrelated to the gate). Inspection confirmed codex actively resisted each adversarial framing: `invent_deps` produced stdlib-only code; `top_level_demo_io` produced functions-only with no top-level IO; `nested_package` produced both the nested layout AND a flat top-level module so standalone import still works; `broken_parametrize` correctly destructured the 4-tuple to 2 params; `compact_packed` produced ugly but syntactically valid one-liners. This is the meaningful negative result Pass 18 was designed to surface: under engineered adversarial prompts, codex's Upwork Python craft does not produce structurally malformed outputs, so the Pass-16 gate has no measurable contribution on this task class. The gate remains armed against weaker models, regression on codex versions, or task classes (e.g. larger frameworks) where structural malformation is more likely.
  - Pass V1.3: SWE-Bench corpus expansion from 20 → 50 tasks. New runner `evals/swe_bench_blind/run_expansion_v2.py` generates **60 codex outputs** (30 tasks × 2 prompt variants — baseline + overclaim) using `codex exec --full-auto`. Sample at `/home/player2/upstream_corpora/swe_bench_verified/sample_v2_expansion.json` (12 django + 8 astropy + 4 sympy + 3 sphinx + 2 matplotlib + 1 scikit-learn). Scoring under the Pass-14a heuristic + tiered scorer: **28/60 heuristic-accept (47%)**, 4 revise, 27 reject, 1 timeout. Apply-check rescore: **22 of 28 accepts rejected on structural malformation (79%)** — non-ASCII whitespace, hunk-header mismatch, base_commit drift. 6 cascade survivors across 4 unique django tasks (django-14915 baseline+overclaim, django-15375 baseline, django-16136 baseline+overclaim, django-16612 baseline). FAIL_TO_PASS truth runner `evals/swe_bench_blind/run_expansion_docker.py` executes inside the official SWE-Bench docker image per task. Two fixes surfaced during truth execution: (1) production-only-hunk filter — codex patches add their own tests that conflict with gold test_patch, so we drop test-file diffs before applying; (2) django test-id parser — new django format is `method (Cls.method)` already containing the method name, naive append produced `Cls.method.method`. Post-fix: **6/6 cascade survivors actually fix their bugs** (canonical control 6/6 also passes). Combined executable truth: **13/13 TP, 0 FA, 0 FR** at scale. Pre-cascade accept rate (47%) → post-cascade actually-fix rate (10% of all codex outputs) is dominated by apply-check filtering, not heuristic discrimination. Calibration dashboard regenerated; tagging `v1.3-blind-harness`.
- Cert-gate hook fix (`0de0e35`) scopes formal-path scanning to staged set + explicit add targets.
- Four tolerated known gaps from Pass 7-c remain non-blocking pending recurrence.
- Pass 13b/14a/16 are SCORER changes behind opt-in flags or domain-local additions; v1.1 baseline reflects the post-hardening harness shape.
- Next action options: (a) **synthesis doc** (Pass 17) covering the now-supported claims of the cascade architecture (cheap structural truth → heuristic discrimination → execution truth on survivors), with explicit limits (post-cascade survivor truth ≠ full confusion truth; rejects never executed); (b) wire FAIL_TO_PASS docker truth as a default scorer flag (currently invoked manually post-cascade); (c) re-run V1.3 expansion truth on non-django survivors if any apply-check survivors emerge from astropy/sympy/sphinx prompts; (d) Pass-18-style stress on a weaker model than codex (cheaper local LLM) to confirm the Pass-16 gate fires when craft drops — currently the gate's contribution is unmeasured on real outputs.
- Owner / executor: Claude (session claude-main-1533).
