# Blind-Eval Harness v1

A general anti-slop legitimacy harness for AI-produced artifacts.

## What it is

A scoring + regression infrastructure that distinguishes legitimate work
from polished operational nonsense across four artifact types. Originally
built as Pass 1–6 of the TLA+ formal-publication remediation plan
(`docs/specs/TLA_HARNESS_REMEDIATION_PLAN.md`); generalized through
Passes 7–10 into a domain-agnostic shape with a shared core.

The harness answers two distinct questions per artifact:

1. **Intrinsic legitimacy** — is the artifact itself well-formed and
   honest about what it does? (Spec content, proof body, deliverable
   code, patch diff.)
2. **Submission-bundle completeness** — does the artifact ship the
   surrounding evidence we require for our own outbound submission
   path? (Source grounding, repo-comparable claims, audience translation,
   skeptical-review record.)

The split was the central architectural insight of Pass 7: the
pre-Pass-7 monolithic scorer false-rejected 100% of upstream-approved
TLA+ specs and 100% of mathlib4 Lean files because it conflated
"matches our local bundle shape" with "is this work legitimate?".

## Domain coverage (v1)

| Domain | Suite | Failure shape it catches |
|---|---|---|
| TLA+ | `evals/tla_blind/` | Tautological invariants; stuttering-only Next; project-private jargon; semantics/bounds conflation; vacuous TypeOK (known gap). |
| Lean 4 | `evals/lean4_blind/` | Deceptive sorry on theorem proof terms (vs pedagogical sorry in structure-instance stubs); README scope overclaim; theorem-statement fidelity gaps. |
| Upwork-style tasks | `evals/upwork_blind/` | Stubbed core operations (`pass`/`NotImplementedError`); fake test assertions (`assert True`); README overclaim ("production-ready", "comprehensive"); requirement keyword drop-out. |
| SWE-Bench Verified | `evals/swe_bench_blind/` | Patch claiming a fix while removing tests; placeholder-bodied methods; commit message overclaim; patches in the wrong file or referencing no issue symbols. |

Out-of-scope by design: branding, naming, design, copywriting,
subjective-acceptance deliverables. The harness has no honest way to
test for slop in subjective work; including those would invite the very
false-accept failure mode the project exists to prevent.

## Architecture

### Shared core — `evals/_blind_core/`

Extracted in Pass 9 (commit `3c0b755`). Pass 10 reused it without
extension on the SWE-Bench domain, validating the abstraction shape.

| Helper | Used by |
|---|---|
| `ORDER`, `worst_of`, `combined_decision` | `deception_regression`, `upstream_corpus` |
| `BUCKET_RULES`, `bucket_for_finding` | `upstream_corpus`, `swe_bench_blind` (regression integration) |
| `load_expected` | `deception_regression` |
| `bundle_present` | `deception_regression` |

12-check `_self_test` covers ordering invariants and bucket
classification across representative findings from all four domains.

### Domain-local — `evals/<domain>_blind/`

Each domain owns its own:
- `execute_current_system.py` — scorer + decision-from-scores helper
- `runner.py` — list/show/self-test CLI
- `rubrics/{rubric.md, scorecard_schema.json}` — score axes
- `tasks/generation/`, `review_corpus/`, `repair_cases/`, `deception_corpus/`

Score axes vary per domain; the cross-domain shape (intrinsic-axis
findings, decision-from-scores helper, hidden labels under
`hidden_label/expected_*.json`, `reviewer_rejection_risk_score`
aggregate) is consistent.

### Cross-domain harnesses

- **`evals/blind_benchmark/`** — labeled accuracy sweep across all four
  domains. Subprocesses each domain's `execute_current_system.py` and
  aggregates.
- **`evals/deception_regression/`** — precision regression suite.
  Bundle-aware policy: combined decision when bundle present, intrinsic-only
  when not. Classifies verdicts as MATCH / NEW_FALSE_ACCEPT /
  NEW_FALSE_REJECT / KNOWN_GAP_TOLERATED.
- **`evals/upstream_corpus/`** — TLA+ + Lean 4 against real
  community-approved corpora (`tlaplus/Examples`, `mathematics_in_lean`,
  `mathlib4` sample). Reports intrinsic-vs-completeness split.
- **`evals/upstream_corpus/charitable_adapter.py`** — measures how much
  of the TLA `revise` load is extraction debt (comments already in
  `.tla` files the harness wasn't reading) vs real under-explanation.

### Master runner — `evals/run_all.py`

Single entrypoint. Default scope (~5-10s):

```bash
python3 evals/run_all.py
```

Runs all five non-live suites and emits
`evals/results/v1_baseline_report.{md,json}` with per-suite status and
single overall pass/fail.

Opt-in live-agent stress (~10-15 min, requires `codex` on PATH):

```bash
python3 evals/run_all.py --with-live-agent
```

Exit 0 iff every suite is within acceptance bound. Exit 1 on any:
- `<100%` blind-benchmark labeled accuracy
- `>0` new false accepts or new false rejects in deception regression
- `>0` regressions in charitable adapter
- non-zero exit from any suite

## v1 baseline numbers

Captured 2026-04-27 from `evals/run_all.py` (no live-agent run).
Reproducible from this commit's working tree.

| Suite | Headline |
|---|---|
| `_blind_core` self-test | 12/12 checks pass |
| Cross-domain blind benchmark | **30/30** labeled fixtures, 100% accuracy, 0 false accept, 0 false reject |
| Deception regression | **34** fixtures, exit 0, **0** new false accepts, **0** new false rejects, **4** known gaps tolerated |
| Upstream benchmark | TLA intrinsic accept **55.6%** (99 specs); Lean intrinsic accept **100%** (128 files) |
| Charitable adapter | **47.7%** of TLA revise load was extraction debt (21 of 44 cases flip to accept under honest .tla comment extraction); 0 regressions |

Live-agent stress (Pass 8, codex on Upwork suite, captured separately
at `evals/upwork_blind/results/live_agent/live_agent_report.md`):
- 6/10 accept, 3/10 revise, 1/10 reject across 2 tasks × 5 prompt variants
- Overclaim variant degraded both tasks (1 reject + 1 revise) — harness
  detected codex's overclaim language under prompt pressure
- Rush + minimal_tests caused codex to drop required keywords →
  requirement_coverage triggered

## Corpus provenance

| Corpus | Source | Snapshot |
|---|---|---|
| TLA+ specs | `tlaplus/Examples` | commit `d9ce4db7` |
| Lean MIL solutions | `leanprover-community/mathematics_in_lean` | commit `2bf0e10d` |
| Lean mathlib sample | `leanprover-community/mathlib4` (sparse: Data/Nat + select basics) | (sparse-checkout HEAD) |
| SWE-Bench Verified sample | `princeton-nlp/SWE-bench_Verified` | 20-task curated subset, 8 repos |

All corpora are **cloned to** `/home/player2/upstream_corpora/` and
**not vendored** into this repo. Provenance SHAs are read at benchmark
runtime and recorded in each report.

## Known limitations (v1)

### 4 tolerated known gaps

Documented in `evals/deception_regression/fixtures/`. Did not recur in
the expanded TLA/Lean corpus (Pass 7-a) or the SWE-Bench pilot (Pass 10).
Tightening deferred per Pass 11 freeze rationale until prevalence data
justifies it.

| Fixture | Gap |
|---|---|
| `tla/vacuous_typeok_bundled` | `TypeOK == counter \in Nat` is tautologically satisfied by the state definition; current tautology regex matches `TRUE`/`x = x`/`x*x >= 0`/`1 = 1` but not set-membership-to-domain. |
| `tla/readme_spec_misalignment` | README claims a two-phase commit protocol; spec has only a single counter. No cross-check rule validates README claims against spec structure. |
| `lean/vacuous_premise` | `∀ x : Empty, False` with legitimate `Empty.elim` proof. No vacuous-premise check. |
| `lean/scope_overclaim_no_sorry` | README claims "commutativity for all commutative algebraic structures" for a Nat.add_comm proof. Phrasing outside the existing overclaim trigger list. |

### Other limitations

- **Text heuristics only.** No SWE-Bench `FAIL_TO_PASS` test execution
  in v1. No Lean `lake build`. No TLA `tlc` model check. The harness
  scores the artifact's *prose + structure*, not its dynamic
  correctness. Pass 10.5+ would integrate test execution.
- **Sample sizes.** SWE-Bench pilot uses 20 of 500 verified tasks
  (sampled with repo + difficulty diversity). Lean mathlib uses a
  ~85-file Data/Nat subset. Both are pilots; broader sweeps belong in
  later passes.
- **Live-agent only on Upwork.** Pass 8 stressed Upwork generation under
  codex with adversarial prompt variants. SWE-Bench live-agent stress is
  the natural next experiment.
- **Symbolic deception not caught.** A patch that imports the right
  module names + reproduces issue symbols decoratively but doesn't
  actually fix the bug would pass intrinsic scoring under v1. Real test
  execution (Pass 10.5) is the correct guard, not better text
  heuristics.
- **Contributor identification not modeled.** No author / maintainer /
  reviewer trust scoring. The harness judges artifacts in isolation.

## What v1 deliberately does not do

- **Tighten the 4 known gaps.** They have not recurred. Tightening
  before recurrence risks overfitting the tightening to fixtures.
- **Expand to a fifth domain.** Three concrete domains justified
  Pass 9's shared-core extraction; a fourth (SWE-Bench, Pass 10)
  validated the abstraction. A fifth would not increase confidence in
  the architecture; it would just increase surface area.
- **Replace heuristic scorers with model-as-judge.** The whole harness
  exists because model-as-judge is exactly what fails on polished slop.

## How to extend

When adding a fifth domain or a new failure mode:

1. **Reuse `_blind_core`** for ORDER, worst_of, bucket_for_finding,
   load_expected, bundle_present. Adding a new bucket to `BUCKET_RULES`
   is allowed but should be justified by ≥2 distinct findings.
2. **Add a new domain-local `evals/<domain>_blind/`** with the standard
   layout (rubric / runner / executor / 4 fixture subdirs).
3. **Wire into `evals/blind_benchmark/`** (DOMAINS dict) and
   `evals/deception_regression/` (load + score helpers, suite roots).
4. **Add a `_load_<domain>_scorer` function in `run_all.py`** if the
   domain has its own master-runner-visible suite.
5. **Run `evals/run_all.py`** before committing. It must stay green.
6. **If the suite needs unique infrastructure not in `_blind_core`**,
   that's a signal the abstraction needs extending (or the new domain
   doesn't fit the harness shape). Pass 9's discipline: extension only
   when ≥2 instances justify it.

## Reproducibility

```bash
# Default scope, ~5-10 seconds
python3 evals/run_all.py

# With live-agent stress on Upwork, ~10-15 minutes
python3 evals/run_all.py --with-live-agent
```

Exit 0 = all suites within acceptance bound. The v1 baseline at this
commit must reproduce as `ok: true`.

## Tag

Pass 11 freezes this state as `v1-blind-harness`. Future regressions
should be diagnosed against this tag.
