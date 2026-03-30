# [106] QA Plan-Control Compiler family

## What this is

Generic cert family proving a certifiable compilation relation between a QA planner witness and a QA control witness over a shared generator algebra. Cymatics is the first physical instantiation domain.

- **QA_PLAN_CONTROL_COMPILER_CERT.v1** — Certifies that a source planner cert and a target control cert share: the same initial state, the same target state, the same path length, the same ordered move sequence, and the same final QA orbit family. Both referenced certs are hash-pinned. The validator resolves certs by ID from their domain's fixtures directory and recomputes all nine claims independently.

## Core claim

> Whenever a QA planner cert emits a plan over a lawful generator algebra and a QA control cert instantiates that plan, a certifiable compilation edge exists between them: same generator sequence, same target invariant, same orbit-family witness, hash-pinned on both ends.

This is the abstraction upward from `[105]` cymatics: the search→plan→control→replay chain is a general QA law, not a cymatics-specific artifact.

## Validation checks

| Check | Description | Fail type |
|-------|-------------|-----------|
| CC1 | Source planner cert found in domain fixtures | SOURCE_CERT_MISSING |
| CC2 | Source cert hash matches declared hash | COMPILATION_HASH_MISMATCH |
| CC3 | Target control cert found in domain fixtures | TARGET_CERT_MISSING |
| CC4 | Target cert hash matches declared hash | COMPILATION_HASH_MISMATCH |
| CC5 | initial_pattern_class consistent: planner == control == claimed | TARGET_INVARIANT_MISMATCH |
| CC6 | target_pattern_class consistent: planner == control == claimed | TARGET_INVARIANT_MISMATCH |
| CC7 | path_length_k consistent: planner == control == claimed | PATH_LENGTH_MISMATCH |
| CC8 | move_sequence consistent: planner == control == claimed | GENERATOR_SEQUENCE_MISMATCH |
| CC9 | final_orbit_family consistent: planner == control == claimed | REPLAY_RESULT_MISMATCH |

## Artifacts

| Artifact | Path |
|----------|------|
| Compiler cert schema | `qa_alphageometry_ptolemy/qa_plan_control_compiler/schemas/qa_plan_control_compiler_cert.schema.json` |
| Validator | `qa_alphageometry_ptolemy/qa_plan_control_compiler/qa_plan_control_compiler_validate.py` |
| Fixtures (2) | `qa_alphageometry_ptolemy/qa_plan_control_compiler/fixtures/` |
| Mapping protocol ref | `qa_alphageometry_ptolemy/qa_plan_control_compiler/mapping_protocol_ref.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_plan_control_compiler
python qa_plan_control_compiler_validate.py --self-test    # JSON output for meta-validator
python qa_plan_control_compiler_validate.py --demo         # human-readable
python qa_plan_control_compiler_validate.py --cert fixtures/compiler_cert_pass_cymatics_hexagon.json
python qa_plan_control_compiler_validate.py --cert fixtures/compiler_cert_fail_sequence_mismatch.json
```

## Failure algebra

| fail_type | Trigger |
|-----------|---------|
| SOURCE_CERT_MISSING | source_planner_cert_id not found in source_domain fixtures |
| TARGET_CERT_MISSING | target_control_cert_id not found in target_domain fixtures |
| COMPILATION_HASH_MISMATCH | declared cert hash does not match actual SHA-256 of found cert |
| TARGET_INVARIANT_MISMATCH | initial or target pattern class inconsistent across planner/control/claims |
| PATH_LENGTH_MISMATCH | path_length_k inconsistent across planner/control/claims |
| GENERATOR_SEQUENCE_MISMATCH | move_sequence in compilation_claims does not match extracted moves from either cert |
| REPLAY_RESULT_MISMATCH | final_orbit_family inconsistent across planner/control/claims |
| NORMALIZATION_RULE_MISMATCH | reserved for future normalization-rule-aware move equivalence checks |

## Domain fixture resolution

The validator maps domain names to fixture directories under `qa_alphageometry_ptolemy/`:

| domain | fixture path |
|--------|-------------|
| `cymatics` | `qa_cymatics/fixtures/` |

Additional domains can be registered in `DOMAIN_TO_FIXTURES` in the validator.

## Cymatics instantiation

The first PASS cert (`compiler_cert_pass_cymatics_hexagon.json`) links:
- Source: `planner_cert_pass_replay_hexagon.v1` (BFS finds flat→stripes→hexagons in k=2)
- Target: `control_cert_pass_hexagon.v1` (flat→stripes→hexagons executed, final=hexagons/cosmos)
- Claims: initial=flat, target=hexagons, k=2, moves=[increase_amplitude, set_frequency], orbit=cosmos

The FAIL cert (`compiler_cert_fail_sequence_mismatch.json`) demonstrates `GENERATOR_SEQUENCE_MISMATCH`: both referenced certs agree on moves=[increase_amplitude, set_frequency] but compilation_claims declares the wrong second move (increase_frequency).

## Changelog

- **v1.0** (2026-03-21): Initial emission; 1 schema; 2 fixtures (1 PASS + 1 FAIL); cymatics as first domain; CC1–CC9 checks.
