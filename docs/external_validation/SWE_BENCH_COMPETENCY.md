# SWE-bench Competency Validation

## What this is

This check validates that the QA competency detection framework
**meaningfully discriminates** between competent and incompetent behavior
on real-world code-change tasks from SWE-bench Lite.

For each real task, the harness derives two competency profiles:

- **Gold agent**: models the behavior that produced the correct gold patch
  (targeted edits, focused test fixes)
- **Null agent**: models an agent with no task-relevant knowledge
  (uniform random actions, minimal reachability, no goal convergence)

The gate checks that the gold agent beats the null agent on **at least one**
competency dimension per task: agency, plasticity, goal density, or control
entropy.

It is enforced by `qa_meta_validator.py` as test `[33]`
(computed as `FAMILY_SWEEPS[-1][0] + 3`).

## Source dataset

- Dataset: `princeton-nlp/SWE-bench_Lite`
- URL: <https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite>
- License: `mit`
- Local frozen subset (feature-only, no raw code redistributed):
  `qa_alphageometry_ptolemy/external_validation_data/swe_bench_lite_features.jsonl`
- Frozen manifest:
  `qa_alphageometry_ptolemy/external_validation_data/swe_bench_lite_features.MANIFEST.json`

30 task instances, 12 repos, `random.seed(42)` for reproducible selection.

### Vendoring policy

**Feature-only**: no raw patch text, test code, or problem statements are
redistributed. Each row contains:

- Metadata: `instance_id`, `repo`, `base_commit`, `created_at`, `version`
- Content hashes: `patch_sha256`, `test_patch_sha256`, `problem_statement_sha256`
- Pre-computed structural features: `patch_features` (files, hunks, additions,
  deletions), `test_patch_features`, `fail_to_pass_count`, `pass_to_pass_count`
- Provenance: `source_dataset`, `source_url`, `source_split`, `license`

This avoids redistributing third-party code under mixed upstream licenses.

## How to run

```bash
# Full report
python3 qa_alphageometry_ptolemy/external_validation_swe_bench_competency.py

# CI mode (single line)
python3 qa_alphageometry_ptolemy/external_validation_swe_bench_competency.py --ci
```

## Competency metrics

Four Levin-aligned metrics, derived from real patch structure:

| Metric             | Gold agent derivation                                   | Null agent  |
|--------------------|---------------------------------------------------------|-------------|
| Agency Index       | `(files * max(hunks,1)) / total_elements`               | `1 / total` |
| Plasticity Index   | `code_changes / (code_changes + test_surface)`          | `0.01/0.99` |
| Goal Density       | `max(fail_to_pass, 1) / total_elements`                 | `0 / total` |
| Control Entropy    | `-sum p(move) ln p(move)` over real action distribution | uniform 1.39|

## Separation property

The gate validates: on every task, the gold agent beats the null agent on
**at least 1 of 4 dimensions** with margin >= 0.01.

Empirical result on the frozen slice:
- Mean dimensions won: 3.0 / 4
- Agency alone separates on 36.7% of tasks (single-hunk patches tie)
- Plasticity, goal density, and entropy separate on nearly all tasks

## Pass criteria

- `total_tasks >= 20`
- `repos_represented >= 5`
- `separation_fail == 0` (every task must separate on at least one dimension)

Overridable via env vars: `QA_SWE_MIN_TASKS`, `QA_SWE_MIN_REPOS`,
`QA_SWE_MIN_SEPARATION`.

## Outputs

- `qa_alphageometry_ptolemy/external_validation_certs/swe_bench_competency_summary.json`
- `qa_alphageometry_ptolemy/external_validation_certs/swe_bench_competency_results.json`
- `qa_alphageometry_ptolemy/external_validation_certs/swe_bench_competency_violations.json`

Separation failures are emitted as typed obstruction witnesses with schema
`QA_SWE_BENCH_SEPARATION_FAILURE.v1`.

## Scope and limits

This validates that the competency framework's four metrics can discriminate
between "competent behavior" (modeled from the gold patch's structural
footprint) and "null behavior" (modeled as a uniform-action, zero-convergence
baseline) on real SWE-bench tasks.

It does **not** run live agents or measure end-to-end resolution rates.
The gold agent is an idealized model derived from the known-correct solution's
structural features.  The null agent is a deterministic adversarial baseline.

Future extensions can substitute real agent traces (from actual SWE-bench
evaluation runs) for the gold/null models.
