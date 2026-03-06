# Family [32]: QA Conjecture-Prove Control Loop

## Purpose

Defines and validates the three-schema conjecture-prove control loop:

1. **Episode** (`QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1`): a complete
   conjecture-prove session with ordered steps, typed results, and trace refs.
2. **Frontier snapshot** (`QA_FRONTIER_SNAPSHOT_SCHEMA.v1`): the current
   exploration frontier with priority-scored states, visited set, and hash chain.
3. **Bounded return receipt** (`QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1`):
   a certificate of reachability analysis — either a witnessed return path
   or a typed obstruction (NO_RETURN_WITHIN_K / BUDGET_EXHAUSTED).

Together these schemas capture the full conjecture-prove loop: an agent
proposes conjectures, attempts proofs, tracks its frontier, and certifies
bounded return properties.

## Machine tract

| Artifact | Path |
|----------|------|
| Episode schema | `qa_alphageometry_ptolemy/qa_conjecture_prove/schemas/QA_CONJECTURE_PROVE_EPISODE_SCHEMA.v1.json` |
| Frontier schema | `qa_alphageometry_ptolemy/qa_conjecture_prove/schemas/QA_FRONTIER_SNAPSHOT_SCHEMA.v1.json` |
| Receipt schema | `qa_alphageometry_ptolemy/qa_conjecture_prove/schemas/QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1.json` |
| Validator | `qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py` |
| Episode harness | `qa_alphageometry_ptolemy/qa_conjecture_prove/run_episode.py` |
| Valid episode fixture | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/episode_valid.json` |
| Neg episode fixture (missing invariant_diff) | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/episode_invalid_missing_invariant_diff.json` |
| Valid frontier fixture | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/frontier_valid.json` |
| Neg frontier fixture (bad hash chain) | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/frontier_invalid_bad_hash_chain.json` |
| Valid receipt fixture | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/bounded_return_valid.json` |
| Neg receipt fixture (missing fail) | `qa_alphageometry_ptolemy/qa_conjecture_prove/fixtures/bounded_return_invalid_missing_fail.json` |

## Schemas

### Episode

Records a complete conjecture-prove session:

- **objective**: `{type, conjecture_statement}` — PROVE, REFUTE, or EXPLORE
- **initial_state**: `{layer, state_hash}` — starting point
- **steps[]**: ordered actions, each with `{step_index, action, trace_ref, input_hash, output_hash, result}`
- **result**: session-level outcome (`{status, fail_type?, invariant_diff?}`)

### Frontier Snapshot

Captures the exploration frontier at a point in time:

- **frontier[]**: `{state_hash, layer, priority}` — priority-scored candidate states
- **visited[]**: state hashes already explored
- **score_model**: weights for novelty, reuse, and obstruction diversity
- **hash_chain**: `{prev_snapshot_hash, this_snapshot_hash}` — chain integrity

### Bounded Return Receipt

Certifies a bounded return analysis result:

- **start_state / return_target_state**: endpoints of the return search
- **k**: bound on path length
- **search**: algorithm, budget, determinism parameters
- **result**: `{status, path?, fail_type?, invariant_diff?}`
  - `RETURN_FOUND` with witnessed path
  - `NO_RETURN_WITHIN_K` with typed obstruction
  - `BUDGET_EXHAUSTED` with search stats

## Typed failure algebra

| Fail type | Trigger |
|-----------|---------|
| `SCHEMA_INVALID` | Missing or malformed required fields |
| `MISSING_INVARIANT_DIFF` | Object without `invariant_diff` |
| `DUPLICATE_STEP_INDEX` | Non-unique or non-sequential step indices |
| `RESULT_INCOMPLETE` | FAIL result without fail_type or invariant_diff; NO_RETURN without fail_type |
| `INVALID_TRACE_REF` | Trace ref missing family prefix |
| `HASH_CHAIN_INVALID` | Empty or missing `this_snapshot_hash` |
| `DUPLICATE_FRONTIER_ENTRY` | Same state_hash appears twice in frontier |
| `PATH_EXCEEDS_K` | Witnessed return path longer than declared k |

## Validation order

### Episode gates
1. `invariant_diff` presence (hard gate)
2. Schema structural validation
3. Step ordering (unique, sequential)
4. Step result completeness (FAIL needs typed diff)
5. Trace ref family prefix validation

### Frontier gates
1. `invariant_diff` presence (hard gate)
2. Schema structural validation
3. Hash chain integrity (`this_snapshot_hash` must be non-empty)
4. Frontier uniqueness (no duplicate state_hash)

### Receipt gates
1. `invariant_diff` presence (hard gate)
2. Schema structural validation
3. Result consistency (NO_RETURN/BUDGET_EXHAUSTED requires fail_type + invariant_diff)
4. Path length check (path length <= k)

## Episode harness

`run_episode.py` replays a validated episode and emits:
1. A frontier snapshot (with self-hashing hash chain)
2. A bounded return-in-k receipt (deterministic BFS)

```bash
python qa_alphageometry_ptolemy/qa_conjecture_prove/run_episode.py \
  --episode <episode.json> --out_dir <dir> [--k 2] [--toolchain_id lean4.12.0]
```

## Running

```bash
# Self-test (11 built-in checks)
python qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py --self-test

# Validate an episode / frontier / receipt
python qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py episode <file.json>
python qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py frontier <file.json>
python qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py receipt <file.json>

# CI mode
python qa_alphageometry_ptolemy/qa_conjecture_prove/qa_conjecture_prove_validator.py episode <file.json> --ci
```
