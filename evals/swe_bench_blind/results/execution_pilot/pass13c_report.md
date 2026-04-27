# Pass 13c — SWE-Bench FAIL_TO_PASS execution

**Diagnostic question:** of the 4 patches that survived Pass 13b's
heuristic + structural gate, how many actually fix their bugs (i.e.,
make `FAIL_TO_PASS` tests pass without breaking `PASS_TO_PASS`)?

**Method:** for each surviving patch, applied the task's `test_patch`
to add the FAIL_TO_PASS test scaffolding, then applied the patch under
test, ran the named FAIL_TO_PASS tests via the repo's own test runner,
and sampled a few PASS_TO_PASS tests as a regression sanity check.

## Result

| label | heuristic (Pass 13b) | applies | FAIL_TO_PASS | PASS_TO_PASS sample | mismatch |
|---|---|---|---:|---:|---|
| django-11477/canonical (control) | accept | YES | 3/3 | 3/3 | — |
| **django-11477/minimal_tests** (codex) | accept | YES | **3/3** | 3/3 | — |
| django-11211/canonical (control) | accept | YES | 1/1 | 3/3 | — |
| **django-11211/minimal_tests** (codex) | revise | YES | **1/1** | 3/3 | **`harness_false_revise`** |
| astropy-14539/canonical (control) | accept | YES | — | — | env unavailable |
| astropy-14539/looks_done (codex) | accept | YES | — | — | env unavailable |
| astropy-14539/minimal_tests (codex) | accept | YES | — | — | env unavailable |

## What this tells us

**Of 2 testable codex patches that survived the structural gate, both fix
their bugs.** No false accepts on the executable subset of Pass 13b's
output.

**One harness false-revise.** `django-11211/minimal_tests` was scored
`revise` by Pass 13b because it touched `django/contrib/contenttypes/fields.py`
instead of the canonical-path `django/db/models/fields/__init__.py`.
Real test execution shows the patch **does** make `FAIL_TO_PASS` pass
and doesn't break sampled `PASS_TO_PASS`. Codex's alternative fix site
is also valid: rather than add `get_prep_value` to `UUIDField` (the
canonical path), it normalizes object IDs in the
`GenericForeignKey.get_prefetch_queryset` path. Both achieve the same
effect for the failing test.

**The harness's `requirement_coverage_score` rule "patch must touch a
canonical file" is too strict.** It treats the canonical fix as the
only correct fix, when in practice multiple sites can resolve the same
bug. This is a real heuristic over-fit, surfaced by execution.

**Astropy environment is the bottleneck.** Astropy at `c0a24c1d` (2023)
fails `pip install -e .` on Python 3.13 + numpy 2.x. The 3 astropy
results from Pass 13b (1 looks_done accept, 1 minimal_tests accept,
1 canonical control) couldn't be tested without Docker / pinned-Python
infrastructure. That's the SWE-Bench community's documented reason for
shipping per-task Docker images.

## Calibration on the executable Django subset

|  | heuristic accept | heuristic revise | actual fix |
|---|---:|---:|---:|
| canonical controls | 2/2 | 0 | **2/2** ✓ |
| codex patches | 1/2 | 1/2 | **2/2** ✓ |

Heuristic + structural gate **floor** on this subset: 0 false accepts.
Heuristic **over-rejection** on this subset: 1 false revise (the
canonical-files-touched check was too strict).

## Mismatch classification

| mismatch class | count | example |
|---|---:|---|
| `harness_false_revise` (heuristic too strict; patch actually fixes bug) | 1 | django-11211/minimal_tests |
| `apply_fail` (caught by Pass 13b structural gate) | 14 | most Pass-12 codex outputs |
| `tests_fail` (heuristic accept but tests fail) | 0 | — |
| `partial_fix` (some FAIL_TO_PASS pass, others fail) | 0 | — |
| `true_fix` (heuristic accept + tests pass) | 1 | django-11477/minimal_tests |
| `env_unavailable` (couldn't test) | 3 | all astropy entries |

## Implications

**The heuristic harness + structural gate is a competent FLOOR on this
Django sample.** No tests-fail false accepts. The remaining issue is
the OPPOSITE direction: false revises for canonical-path overfit.

**Two follow-up paths:**

### Pass 14a — soften `requirement_coverage` (cheap)

The check currently treats `canonical_files_touched` as a hard
requirement. Replace with a softer signal:
- if patch touches a canonical file → strong positive
- if patch touches a file in the same module hierarchy → moderate positive
- if patch touches a file referenced in the issue text → weak positive
- if patch touches none of the above → keep the existing penalty

That would have flipped `django-11211/minimal_tests` from revise to
accept correctly, since `django/contrib/contenttypes/fields.py` is
mentioned in the issue's reproducer code.

Cost: ~30 lines in the SWE-Bench scorer. Doesn't change the structural
gate or the apply-check.

### Pass 14b — astropy environment (harder)

To test the 3 astropy results, need either:
- per-task Docker images (SWE-Bench's official approach — significant
  infrastructure)
- pyenv + pinned Python (3.10/3.11) + frozen requirements
- conda env per task with pinned numpy <2.0

The cost-benefit is worth measuring: how many of the Pass 13b accepts
are astropy/sympy/etc. tasks that need older Python? On the current
sample, 3/3 surviving accepts are astropy. So full Pass 10.5
infrastructure is needed to test those.

## Recommendation

**Do Pass 14a first** (the false-revise fix). It's the same kind of
cheap structural improvement Pass 13b was: a small scorer change that
addresses an empirically-observed harness gap.

**Pass 14b (astropy environment) is justified next** — but the cost is
higher and the data point we gained from Pass 13c on Django is already
informative: the heuristic + structural gate doesn't false-accept; its
remaining problem is over-rejection. If the same pattern holds on
astropy once we can test, the harness will have crossed from "competent
filter" to "calibrated filter."

## Provenance

- Test environment: Python 3.13.7 + system Django paths overridden via
  `PYTHONPATH=<repo_dir>`. Django 3.0.dev / 3.0 imports cleanly under
  Python 3.13 for the runtests.py path.
- Astropy: `pip install -e .` failed with build-system error against
  Python 3.13 + numpy 2.2.4. Documented as `env_unavailable` rather
  than retried with infrastructure work.
- Test runner: `python3 tests/runtests.py -v 0 <test_id>` with
  `PYTHONPATH=<repo_dir>`. Django's own runner; no pytest involvement.
- Sampled PASS_TO_PASS: first 3 from each task's PASS_TO_PASS list
  (151+ tests per task; 3 is a sanity check, not full regression).
- Repos: cloned shallow at exact base_commits under
  `/home/player2/upstream_corpora/swe_bench_runs/`.
