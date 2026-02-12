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

This family treats mathematical reasoning as a reproducible, machine-checkable
compilation pipeline rather than an opaque model output.

## Conceptual Model

Family [31] models mathematics as a compilable artifact chain.

A natural-language claim is compiled into Lean traces, then:

1. deterministically replayed,
2. paired back to the human statement with explicit alignment evidence,
3. compressed through mined reusable lemmas.

Every stage yields either a certified witness or a typed failure with
`fail_type` and `invariant_diff`.

## Pipeline Overview

```text
Natural Language Claim
  ->
Formal Task (Schema v1)
  ->
Compiler Trace
  ->
Replay Bundle
  ->
Pair Certificate
  ->
Lemma Mining
  ->
Compressed Proof Corpus
```

## Machine Tract

| Artifact | Path |
|----------|------|
| Validator | `qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py` |
| Task schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_FORMAL_TASK_SCHEMA.v1.json` |
| Trace schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_MATH_COMPILER_TRACE_SCHEMA.v1.json` |
| Replay schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_MATH_COMPILER_REPLAY_BUNDLE_SCHEMA.v1.json` |
| Pair v1 schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_HUMAN_FORMAL_PAIR_CERT.v1.json` |
| Lemma mining schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_LEMMA_MINING_SCHEMA.v1.json` |
| Demo pack index schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_MATH_COMPILER_DEMO_PACK_SCHEMA.v1.json` |
| Legacy pair schema | `qa_alphageometry_ptolemy/qa_math_compiler/schemas/QA_COMPILER_PAIR_CERT_SCHEMA.v1.json` |

## CI Gates (Family Sweep)

`qa_meta_validator.py` runs all of the following for Family [31]:

- Legacy trace/pass fixture and trace/fail fixture (`RESULT_INCOMPLETE`),
- Legacy pair/pass fixture and pair/fail fixture (`HASH_SELF_BINDING`),
- Task pass fixture and task negative fixture (`SCHEMA_INVALID`),
- Replay pass fixture and replay negative fixture (`DETERMINISM_MISMATCH`),
- Pair v1 pass fixture and pair v1 negative fixture (`PROVED_PAIR_REPLAY_MISMATCH`),
- Lemma mining pass fixture and lemma mining negative fixture (`COMPRESSION_BELOW_TARGET`),
- optional `demo_pack_v1` validation when `qa_math_compiler/demo_pack_v1/` is present.

## Typed Failure Algebra

| Fail type | Trigger |
|-----------|---------|
| `DETERMINISM_MISMATCH` | Replay hash differs from original hash for non-flake replay |
| `REPLAY_COUNTS_MISMATCH` | Declared replay metrics do not recompute from traces |
| `REPLAY_BELOW_THRESHOLD` | Replay success rate below configured minimum |
| `PROVED_PAIR_REPLAY_MISMATCH` | `status=PROVED` without successful trace replay |
| `COMPRESSION_METRIC_MISMATCH` | Declared median compression does not match recomputed median |
| `COMPRESSION_BELOW_TARGET` | Recomputed median compression below target |
| `NEEDS_PROOF_UNACCOUNTED` | `NEEDS_PROOF` lemma candidate has no typed failure record |
| `DEMO_PACK_MISSING_ARTIFACT` | Required demo-pack file is missing |
| `DEMO_PACK_INDEX_MISMATCH` | `demo_pack_v1/index.json` conflicts with discovered examples |
| `DEMO_PACK_EXAMPLE_INVALID` | Demo example artifact failed its delegated validator |
| `DEMO_PACK_LINKAGE_INVALID` | Demo example has broken pair/replay/trace linkage |
| `DEMO_PACK_BELOW_THRESHOLD` | Demo pack fails configured pack-level thresholds |

## Replay-First Done Milestone

Math Compiler v1 is considered done when a fixed benchmark suite satisfies all:

- deterministic replay gate meets configured replay threshold,
- Pair Pack v1 `PROVED` entries have successful replay-backed trace refs,
- Lemma pack reaches configured median trace compression target,
- every unresolved lemma is represented by typed failure (`fail_type`, `invariant_diff`).

## External Validation

Family [31] is externally reproducible via:

```bash
python qa_alphageometry_ptolemy/qa_meta_validator.py
```

This gate verifies both the family self-tests and the pass/fail fixtures wired
into the repository sweep.

## Running

```bash
# Full self-test (legacy + replay/task/pair_v1/lemma)
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py --self-test

# Validate individual artifacts
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py task qa_alphageometry_ptolemy/qa_math_compiler/fixtures/task_valid.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py replay qa_alphageometry_ptolemy/qa_math_compiler/fixtures/replay_valid.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py pair_v1 qa_alphageometry_ptolemy/qa_math_compiler/fixtures/pair_v1_valid_proved.json
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py lemma qa_alphageometry_ptolemy/qa_math_compiler/fixtures/lemma_mining_valid.json

# Validate optional demo pack
python qa_alphageometry_ptolemy/qa_math_compiler/qa_math_compiler_validator.py demo_pack qa_alphageometry_ptolemy/qa_math_compiler/demo_pack_v1
```

## Design Principle

No proof without replay.
No replay without determinism.
No compression without certification.
No unresolved failure without typed classification.
