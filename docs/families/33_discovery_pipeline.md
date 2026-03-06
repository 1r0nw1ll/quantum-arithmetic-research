# Family [33]: QA Discovery Pipeline

## Purpose

Defines and validates the certified pipeline layer above families [31] (math
compiler stack) and [32] (conjecture-prove control loop). Treats discovery
as deterministic execution:

```
episode -> validate -> emit frontier -> emit return receipt -> (optional) seed
```

and bundles plans + runs + artifacts into ship-ready packs with Merkle lineage.

## Machine tract

| Artifact | Path |
|----------|------|
| Run schema | `qa_alphageometry_ptolemy/qa_discovery_pipeline/schemas/QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1.json` |
| Plan schema | `qa_alphageometry_ptolemy/qa_discovery_pipeline/schemas/QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1.json` |
| Bundle schema | `qa_alphageometry_ptolemy/qa_discovery_pipeline/schemas/QA_DISCOVERY_BUNDLE_SCHEMA.v1.json` |
| Validator | `qa_alphageometry_ptolemy/qa_discovery_pipeline/qa_discovery_pipeline_validator.py` |
| Batch runner | `qa_alphageometry_ptolemy/qa_discovery_pipeline/run_batch.py` |
| Valid run fixture | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/run_valid.json` |
| Neg run fixture (missing invariant_diff) | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/run_invalid_missing_invariant_diff.json` |
| Valid plan fixture | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/plan_valid.json` |
| Neg plan fixture (nondeterministic) | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/plan_invalid_nondeterministic.json` |
| Valid bundle fixture | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/bundle_valid.json` |
| Neg bundle fixture (bad hash chain) | `qa_alphageometry_ptolemy/qa_discovery_pipeline/fixtures/bundle_invalid_bad_chain.json` |

## Schemas

### Pipeline Run (`QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1`)

A single pipeline execution with ordered steps:

- **toolchain**: python version, OS, toolchain_id
- **inputs**: episode_ref, k bound, determinism declarations
- **execution.steps[]**: ordered pipeline ops (LOAD_EPISODE, VALIDATE_EPISODE, EMIT_FRONTIER, EMIT_RETURN_RECEIPT, EMIT_NEXT_SEED)
- **outputs.artifacts[]**: emitted artifacts with family + schema_id + path_or_hash
- **result**: `{status, fail_type?, invariant_diff?}` — typed on FAIL
- **merkle_parent**: Merkle linkage to prior run/plan

### Batch Plan (`QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1`)

Deterministic scheduler plan:

- **run_queue[]**: ordered list of `{run_id, episode_ref, k}`
- **determinism**: `{queue_ordering, seed_policy, canonical_json}` — canonical_json must be true
- **budget**: `{max_runs, max_seconds_total}`

### Bundle (`QA_DISCOVERY_BUNDLE_SCHEMA.v1`)

Ship artifact binding plan + runs + emitted artifacts:

- **plan_ref**: reference to the batch plan
- **run_refs[]**: references to each pipeline run record
- **artifact_refs[]**: references to all emitted conjecture-prove artifacts
- **summary**: `{runs_total, runs_success, fail_counts}`
- **hash_chain**: `{prev_bundle_hash, this_bundle_hash}` — self-hashing

## Typed failure algebra

| Fail type | Schema | Trigger |
|-----------|--------|---------|
| `SCHEMA_INVALID` | all | Missing or malformed required fields |
| `DETERMINISM_MISSING` | run, plan | Missing determinism declaration field |
| `DUPLICATE_STEP_INDEX` | run | Non-unique step indices |
| `RESULT_INCOMPLETE` | run | FAIL step/result without fail_type or invariant_diff |
| `MERKLE_PARENT_MISSING` | run, plan | Empty or missing merkle_parent |
| `NONDETERMINISTIC_PLAN` | plan | canonical_json is not true |
| `HASH_CHAIN_INVALID` | bundle | Empty this_bundle_hash |
| `SUMMARY_INVALID` | bundle | runs_success > runs_total |

## Batch runner

`run_batch.py` executes a batch plan by calling `qa_conjecture_prove/run_episode.py`
for each queued episode, emitting per-run pipeline run records and a final bundle:

```bash
python3 qa_alphageometry_ptolemy/qa_discovery_pipeline/run_batch.py \
  --plan qa_discovery_pipeline/fixtures/plan_valid.json \
  --out_dir qa_discovery_pipeline/out \
  [--toolchain_id lean4.12.0] \
  [--prev_bundle_hash 0000...] \
  [--created_utc 2026-02-11T00:00:00Z]
```

Emits:
- `out/run_<run_id>.json` (pipeline run records)
- `out/out_<run_id>/frontier_snapshot.json` (conjecture-prove artifacts)
- `out/out_<run_id>/bounded_return_receipt.json`
- `out/bundle_<hash>.json` (bundle with computed this_bundle_hash)

## Running

```bash
# Self-test (12 built-in checks)
python qa_alphageometry_ptolemy/qa_discovery_pipeline/qa_discovery_pipeline_validator.py --self-test

# Validate individual artifacts
python qa_alphageometry_ptolemy/qa_discovery_pipeline/qa_discovery_pipeline_validator.py run <file.json> --ci
python qa_alphageometry_ptolemy/qa_discovery_pipeline/qa_discovery_pipeline_validator.py plan <file.json> --ci
python qa_alphageometry_ptolemy/qa_discovery_pipeline/qa_discovery_pipeline_validator.py bundle <file.json> --ci
```
