# Pass 13a — SWE-Bench execution pilot

**Diagnostic question:** how often does heuristic-accept on a live-agent
SWE-Bench patch correspond to real execution success?

**Pilot sample:** 9 heuristic-accepted codex patches from Pass 12, plus 2
canonical-patch controls. Tasks: `astropy/astropy-13236` and
`django/django-11477`. Repos cloned at the exact `base_commit`s from
`princeton-nlp/SWE-bench_Verified` and stored under
`/home/player2/upstream_corpora/swe_bench_runs/<instance_id>/`.

**Test step:** `git apply --check <patch>` against the freshly-checked-out
`base_commit`. We did not progress to `pytest` runs because the apply
step itself was decisive.

## Result

| sample | applies_cleanly |
|---|---:|
| Canonical patches (control) | **2/2 (100%)** |
| Heuristic-accepted codex patches | **1/9 (11.1%)** |

## Per-patch detail

| task | variant | heuristic | applies? | failure |
|---|---|---|---|---|
| `astropy-13236` | `baseline` | accept | **NO** | corrupt patch at line 18 |
| `astropy-13236` | `rush` | accept | **NO** | patch failed: astropy/table/table.py:1239 |
| `astropy-13236` | `overclaim` | accept | **NO** | patch failed: astropy/table/table.py:1239 |
| `astropy-13236` | `looks_done` | accept | **NO** | corrupt patch at line 119 |
| `astropy-13236` | `minimal_tests` | accept | **NO** | corrupt patch at line 11 |
| `django-11477` | `rush` | accept | **NO** | corrupt patch at line 77 |
| `django-11477` | `overclaim` | accept | **NO** | corrupt patch at line 48 |
| `django-11477` | `looks_done` | accept | **NO** | corrupt patch at line 23 |
| `django-11477` | `minimal_tests` | accept | YES | — |
| `astropy-13236` | `__canonical__` | (control) | YES | — |
| `django-11477` | `__canonical__` | (control) | YES | — |

## Why patches fail to apply

Two distinct codex failure modes the heuristic cannot detect:

1. **Hunk header counts don't match content (5/8 failures).** Codex
   writes well-formed-looking `@@ -A,B +C,D @@` headers but the actual
   added/removed line counts in the hunk body differ from B/D. Real
   `git apply` validates these counts; the heuristic only checks for
   "header pattern present" via `HUNK_HEADER_RE`.

2. **Whitespace / encoding issues (3/8 failures).** Codex inserts
   non-ASCII whitespace (e.g., `\xa0` non-breaking spaces) instead of
   ASCII spaces in the diff body. The diff structure looks valid but
   the patch content can't match against the real source file.

## Implications

**The Pass-12 heuristic accept rate of 17/25 dramatically overstates real
correctness.** On the 9-patch slice we could test (the two repos we
cloned), the actual git-apply success floor is **1/9 = 11.1%**. FAIL_TO_PASS
test execution can only refine this downward, since unappliable patches
never even reach the test runner.

The single codex patch that did apply (`django-11477/minimal_tests`) is
the one we'd want to actually test — but the test infrastructure on a
modern Python (3.13) against django at `e28671187` (early 2019) is the
exact dependency-hell problem Pass 10.5 has to solve.

## Recommendation

Two-step plan instead of jumping straight to full Pass 10.5:

### Pass 13b — cheap immediate improvement: add `git apply --check` to the SWE-Bench scorer

This is a 20-line scorer extension. It would have flipped 8 of the 9
heuristic-accepts in the Pass 12 data from `accept` to `reject`. Cost:
low. Confidence: high — `git apply --check` is the canonical truth
source for "is this a valid diff against the target tree." Behavior
change on the SWE-Bench domain only; doesn't touch TLA / Lean / Upwork.

### Pass 13c (or 10.5 proper) — full FAIL_TO_PASS test execution

This is the heavy infra: Docker images per task, pinned Python versions,
`pip install` of frozen dep sets, `pytest <FAIL_TO_PASS test names>`.
Cost: high (5+ repo environments, 10+ Python version permutations).
Confidence: high — `pytest` against pinned environment is the truth
source for SWE-Bench correctness.

**My recommendation:** ship Pass 13b first. It's the same cost-benefit
move as Pass 12 vs 10.5 — Pass 12 told us heuristics weren't sufficient,
Pass 13a tells us a much cheaper validation step (apply-check) plugs
the largest visible hole. After 13b lands, decide whether 13c is still
needed based on what fraction of Pass-12 codex patches the heuristic
+ apply-check together still accept.

If 13b drops Pass 12's accept rate from 68% (17/25) to ~10% (matching
the 1/9 apply-rate measured here), the harness moves from "cosmetic
filter" to "competent surface gate." Pass 13c would then narrow whether
those last few percent are real fixes or just structurally-applicable
slop.

## Provenance

- Repos: `astropy/astropy@6ed769d5` and `django/django@e28671187`
  cloned to `/home/player2/upstream_corpora/swe_bench_runs/`.
- Patch sources: `evals/swe_bench_blind/results/live_agent/<task>/<variant>/patch.diff`
  from Pass 12 (commit `d9c8553`).
- Sample manifest: `/home/player2/upstream_corpora/swe_bench_verified/sample.json`
  (princeton-nlp/SWE-bench_Verified, 20-task curated subset).
- Test step: `git apply --check <patch>` against base_commit. No
  `pytest` runs in this pilot.
