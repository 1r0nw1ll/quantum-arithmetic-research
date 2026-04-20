# [257] QA Integer-State Pipeline Cert

## What this is

Certifies the two-boundary-crossing invariant of Theorem NT as a structural property of a QA-native pipeline. The observer/QA boundary is crossed exactly twice — once at input tokenization (continuous → integer tuples), once at output decoding (integer tuples → continuous) — and all interior state between the two crossings is pure integer tuples. No re-tokenization through a continuous intermediate is permitted inside the pipeline.

Structurally eliminates the GLM-5 TITO misalignment failure mode (arXiv:2602.15763, §4.1.2) by construction: the failure mode is re-tokenization on the learner side corrupting action↔reward alignment. With no continuous intermediate to re-project through, the failure mode has nothing to attach to.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_integer_state_pipeline_cert_v1/qa_integer_state_pipeline_cert_validate.py` |
| Pass fixture | `qa_integer_state_pipeline_cert_v1/fixtures/isp_pass_default.json` |
| Fail fixture | `qa_integer_state_pipeline_cert_v1/fixtures/isp_fail_continuous_intermediate.json` |
| Mapping ref | `qa_integer_state_pipeline_cert_v1/mapping_protocol_ref.json` |
| Reference prototype | `qa_lab/qa_orbit_resonance_attention.py` |
| Design doc | `docs/theory/QA_GLM5_ARCHITECTURE_MAPPING.md` §Failure-Mode-2 |

All paths relative to `qa_alphageometry_ptolemy/` unless otherwise noted.

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_integer_state_pipeline_cert_v1
python qa_integer_state_pipeline_cert_validate.py --self-test
```

## Semantics

- **ISP_1**: schema_version matches.
- **ISP_BOUND**: both `input_boundary.kind == "observer_projection_in"` + `direction == "continuous_to_integer"` AND `output_boundary.kind == "observer_projection_out"` + `direction == "integer_to_continuous"`.
- **ISP_NO_REPROJECT**: no interior `pipeline_stages` entry with `kind` containing `"retokeniz"` or `"decode_then_encode"`; every interior stage has `state_type == "integer_tuple"`.
- **ISP_INT**: every `interior_state_samples` entry is an integer tuple `(b, e)` in `{1..m}^2` — no floats, no Fractions, no np types.
- **ISP_DET**: declared `determinism.bitwise_identical == true`, `repeats >= 100`; validator independently recomputes the canonical trace twice and asserts bitwise equality.
- **ISP_SRC**: source attribution references `arXiv:2602.15763` and `TITO`.
- **ISP_WIT**: at least 3 interior-state witnesses.
- **ISP_F**: `fail_ledger` well-formed.

## Relation to GLM-5

GLM-5's TITO gateway is an engineering discipline layered on top of a continuous-learning substrate: it carries exact integer token IDs from rollout engine to learner to avoid the alignment corruption that text round-trips cause. It works because the team disciplines themselves not to re-tokenize. The structural guarantee — that there IS no continuous intermediate to re-project through — is what this cert certifies for QA-native pipelines. TITO becomes unnecessary when the two-boundary invariant is structural, not disciplinary.

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `ISP_NO_REPROJECT` | Interior stage re-projects through continuous intermediate | Remove the decode_then_encode stage; keep integer state throughout interior. |
| `ISP_INT` | Interior-state sample contains non-integer values | Trace is using float/Fraction state — convert to integer tuples. |
| `ISP_DET` | Declared determinism contradicts recomputation | Eliminate stochastic operations on QA path. |
| `ISP_BOUND` | One of the two boundaries missing or wrong direction | Add observer projections at both endpoints with correct direction. |

## Changelog

- **v1** (2026-04-19): Initial release. Together with [256] (attention) and [258] (training), completes the QA-native architecture cert triple for the GLM-5 failure-mode class.
