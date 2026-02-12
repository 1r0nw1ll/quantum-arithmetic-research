# Family [31]: QA Math Compiler Stack

## Purpose

Implements a replay-first Lean autoformalization pipeline with typed failures:

1. `QA_FORMAL_TASK_SCHEMA.v1` for runnable Lean tasks,
2. `QA_MATH_COMPILER_TRACE_SCHEMA.v1` for per-step compilation traces,
3. `QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1` for deterministic replay gates,
4. `QA_HUMAN_FORMAL_PAIR_CERT.v1` for NL↔formal proof-backed pairings,
5. `QA_LEMMA_MINING_SCHEMA.v1` for compression-oriented lemma mining.

Legacy Family [31] artifacts (`QA_MATH_COMPILER_TRACE_SCHEMA.v1`,
`QA_COMPILER_PAIR_CERT_SCHEMA.v1`) remain supported for backward compatibility.

## Machine Tract

| Artifact | Path |
|----------|------|
| Validator | `qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py` |
| Task schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_FORMAL_TASK_SCHEMA.v1.json` |
| Trace schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_MATH_COMPILER_TRACE_SCHEMA.v1.json` |
| Replay schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1.json` |
| Pair v1 schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_HUMAN_FORMAL_PAIR_CERT.v1.json` |
| Lemma mining schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_LEMMA_MINING_SCHEMA.v1.json` |
| Legacy pair schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_COMPILER_PAIR_CERT_SCHEMA.v1.json` |

## CI Gates (Family Sweep)

`qa_meta_validator.py` runs all of the following for Family [31]:

- Legacy trace/pass fixture and trace/fail fixture (`RESULT_INCOMPLETE`),
- Legacy pair/pass fixture and pair/fail fixture (`HASH_SELF_BINDING`),
- Task pass fixture and task negative fixture (`SCHEMA_INVALID`),
- Replay pass fixture and replay negative fixture (`DETERMINISM_MISMATCH`),
- Pair v1 pass fixture and pair v1 negative fixture (`PROVED_PAIR_REPLAY_MISMATCH`),
- Lemma mining pass fixture and lemma mining negative fixture (`COMPRESSION_BELOW_TARGET`).

## Typed Failure Algebra (new gates)

| Fail type | Trigger |
|-----------|---------|
| `DETERMINISM_MISMATCH` | Replay hash differs from original hash for non-flake replay |
| `REPLAY_COUNTS_MISMATCH` | Declared replay metrics do not recompute from traces |
| `REPLAY_BELOW_THRESHOLD` | Replay success rate below configured minimum |
| `PROVED_PAIR_REPLAY_MISMATCH` | `status=PROVED` without successful trace replay |
| `COMPRESSION_METRIC_MISMATCH` | Declared median compression does not match recomputed median |
| `COMPRESSION_BELOW_TARGET` | Recomputed median compression below target |
| `NEEDS_PROOF_UNACCOUNTED` | `NEEDS_PROOF` lemma candidate has no typed failure record |

## Replay-First Done Milestone

Math Compiler v1 is considered done when a fixed benchmark suite satisfies all:

- deterministic replay gate meets configured replay threshold,
- Pair Pack v1 `PROVED` entries have successful replay-backed trace refs,
- Lemma pack reaches configured median trace compression target,
- every unresolved lemma is represented by typed failure (`fail_type`, `invariant_diff`).

## Running

```bash
# Full self-test (legacy + replay/task/pair_v1/lemma)
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py --self-test

# Validate individual artifacts
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py task qa_alphageometry_ptolemy/qa_math_compiler/fixtures/task_valid.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py replay qa_alphageometry_ptolemy/qa_math_compiler/fixtures/replay_valid.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py pair_v1 qa_alphageometry_ptolemy/qa_math_compiler/fixtures/pair_v1_valid_proved.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py lemma qa_alphageometry_ptolemy/qa_math_compiler/fixtures/lemma_mining_valid.json
```
