# Pass 21 — SWE-Bench Reject-Sample FAIL_TO_PASS Report

## Goal
Convert V1.3's survivor-only truth into a full confusion-matrix truth on the testable subset by running FAIL_TO_PASS against codex outputs the cascade *rejected* (heuristic non-accept). The interesting cell is `false_reject_recall_miss`: cascade said reject/revise but the patch actually fixes the bug.

## Headline
- Reject-sample size: **13** (heuristic non-accept ∧ patch.diff saved)
- Tested: **13** (image-pull failures: 0)

### Classifications on the reject sample
- True reject (structural — won't apply): **10**
- True reject (semantic — applies but tests fail): **0**
- **False reject (would have actually passed): 3**

## Full confusion matrix on the testable subset
| | actually_fixes | does_not_fix |
|---|---:|---:|
| **cascade=accept** (V1.3 survivors) | 6 (TP) | 0 (FA) |
| **cascade=reject/revise** (this pass) | 3 (FR) | 10 (TN) |

- Precision (TP / (TP + FA)) = **6/6 = 100.0%**
- Recall (TP / (TP + FR)) = **6/9 = 66.7%**

## False-reject root-cause

The 3 measured false-rejects map to 2 specific heuristic bugs in
`evals/swe_bench_blind/execute_current_system.py`. Both are
fixable; left for a follow-up pass so this pass remains a pure
measurement.

### Bug 1: unified-diff check too strict (2/3 false rejects)

`astropy-14096/baseline` and `django-15104/baseline` were rejected
with finding `Patch file is not a unified diff (missing 'diff --git'
or '@@' hunk header)`. Both patches contain valid `--- a/` / `+++ b/`
/ `@@ ... @@` unified diff format and `git apply --check` confirms
they apply cleanly. The `diff --git` header is *optional* — it is
emitted by `git diff` but valid unified diffs from other sources
(or from `diff -u`) omit it.

The check at line 192:
```python
if not has_diff_header or not has_hunk:
    findings.append("Patch file is not a unified diff ...")
```
should be relaxed to accept either `diff --git`+`@@` OR
`--- a/` / `+++ b/` + `@@`.

### Bug 2: placeholder detector ignores file context (1/3 false rejects)

`astropy-7166/baseline` was rejected for one `+pass` line. The line
is inside an added `@property def bar(self): "BAR"; pass` test-class
method in `astropy/utils/tests/test_misc.py` — a legitimate test
scaffold demonstrating docstring-inheritance behavior, not an
incomplete production fix. The production hunk in `misc.py` is a
real fix to the `inspect.isfunction(val)` predicate.

The placeholder check at line 232 does not distinguish test-file
hunks from production-file hunks. Fix: only count `+pass` /
`+raise NotImplementedError` / TODO / FIXME occurrences whose
enclosing hunk's `+++ b/` path is NOT in a test directory (use the
existing `TEST_PATH_RE`).

## Per-output detail
| instance_id | variant | heuristic | outcome | classification |
|---|---|---|---|---|
| `astropy__astropy-12907` | `baseline` | revise | apply_check_failed | true_reject_structural |
| `astropy__astropy-12907` | `overclaim` | reject | apply_check_failed | true_reject_structural |
| `astropy__astropy-13033` | `overclaim` | reject | apply_check_failed | true_reject_structural |
| `astropy__astropy-14096` | `baseline` | reject | actually_passes | false_reject_recall_miss |
| `astropy__astropy-14309` | `baseline` | reject | apply_check_failed | true_reject_structural |
| `astropy__astropy-14508` | `baseline` | reject | apply_check_failed | true_reject_structural |
| `astropy__astropy-14508` | `overclaim` | reject | apply_check_failed | true_reject_structural |
| `astropy__astropy-7166` | `baseline` | revise | actually_passes | false_reject_recall_miss |
| `astropy__astropy-7166` | `overclaim` | revise | apply_check_failed | true_reject_structural |
| `django__django-15098` | `baseline` | revise | apply_check_failed | true_reject_structural |
| `django__django-15098` | `overclaim` | reject | apply_check_failed | true_reject_structural |
| `django__django-15104` | `baseline` | reject | actually_passes | false_reject_recall_miss |
| `sympy__sympy-21379` | `overclaim` | reject | apply_check_failed | true_reject_structural |
