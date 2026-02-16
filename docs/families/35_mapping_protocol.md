# [35] QA Mapping Protocol

## What this is

Defines **QA_MAPPING_PROTOCOL.v1**: the minimal machine-checkable contract for claiming that some external object/system has been formally mapped into QA.

This family is intentionally small and enforceable. It exists so every downstream “family” can be forced (by Gate 0) to declare:

- state manifold
- generators
- invariants
- failure taxonomy
- reachability view
- determinism contract (v1: unique successor or typed FAIL)

## Artifacts

| Artifact | Path |
|----------|------|
| Protocol schema | `qa_mapping_protocol/schema.json` |
| Gate validator | `qa_mapping_protocol/validator.py` |
| Canonical shared mapping object | `qa_mapping_protocol/canonical_mapping_protocol.json` |
| Minimal passing fixture | `qa_mapping_protocol/fixtures/valid_min.json` |
| Minimal failing fixture | `qa_mapping_protocol/fixtures/invalid_missing_determinism.json` |

All paths relative to repository root.

## How to run

```bash
# From repo root:
python qa_mapping_protocol/validator.py --self-test

# Or via meta-validator (runs as test [35])
cd qa_alphageometry_ptolemy
python qa_meta_validator.py
```

## Semantics (Gates)

- **Gate 1 — Schema Validity**: validates against `QA_MAPPING_PROTOCOL.v1`
- **Gate 2 — Invariant Diff Enforcement**: each generator must declare `invariant_effect`
- **Gate 3 — Failure Completeness**: `generators` and `failure_taxonomy` must both be non-empty
- **Gate 4 — Determinism Contract**: `invariant_diff_defined==true` and non-empty `nondeterminism_proof`
- **Gate 5 — State Constraint Closure**: `state_manifold.constraints` must be explicitly listed and non-empty

## Family-root granularity (Gate 0)

Gate 0 is enforced **per family root**, as specified by `family_root_rel` in
`qa_alphageometry_ptolemy/qa_meta_validator.py`’s `FAMILY_SWEEPS`.

- If multiple families share the same `family_root_rel` (many legacy entries use `"."`),
  they will share the same `mapping_protocol(_ref).json` and the gate result will be cached.
- For **new families**, prefer giving each family a dedicated root directory and setting
  `family_root_rel` to that directory. This makes the mapping protocol truly per-family
  without requiring per-certificate wiring.

## Changelog

- **v1** (2026-02-14): Initial schema + validator + fixtures; wired into meta-validator family sweeps.
