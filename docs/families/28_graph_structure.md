# [28] QA Graph Structure Bundle

## What this is

Certifies QA structural feature runs for graph community detection under paired baseline controls. This family encodes deterministic feature extraction, baseline parity, phase diagnostics, and metric-delta proofs in a failure-complete certificate.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__GRAPH_STRUCTURE.yaml` |
| Validator | `qa_graph_structure_validator_v1.py` |
| Bundle emitter/validator | `qa_graph_structure_bundle_v1.py` |
| Reference cert | `certs/QA_GRAPH_STRUCTURE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_GRAPH_STRUCTURE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_GRAPH_STRUCTURE_BUNDLE.v1.json` |
| Cert schema | `schemas/QA_GRAPH_STRUCTURE_CERT.v1.schema.json` |
| Bundle schema | `schemas/QA_GRAPH_STRUCTURE_BUNDLE.v1.schema.json` |
| Success example | `examples/graph_structure/graph_structure_success.json` |
| Failure example | `examples/graph_structure/graph_structure_parity_failure.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate individual cert
python qa_graph_structure_validator_v1.py --demo

# Validate with recompute level
python qa_graph_structure_validator_v1.py --level recompute examples/graph_structure/graph_structure_success.json

# Emit + check bundle
python qa_graph_structure_bundle_v1.py --emit --check

# Or via meta-validator (runs as family [28])
python qa_meta_validator.py
```

## Semantics

- **Generator set**: `{sigma_feat_extract, sigma_qa_embed, sigma_cluster, sigma_eval, sigma_phase_analyze}`
- **Paired baselines**: dataset/algorithm/seeds must match across baseline and QA traces
- **Delta integrity**: `delta_metrics[k] == qa_metrics[k] - baseline_metrics[k]`
- **Phase diagnostics**: phase-24 and phase-9 witnesses are replay-checked from trace terminal states
- **Hard invariants**: tuple consistency, feature determinism, eval reproducibility, trace completeness, baseline pairing

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `out_of_bounds` | Invalid graph size/community bounds | Correct graph input/config bounds |
| `invariant` | Determinism/reproducibility contract broken | Repair invariant-producing stage |
| `phase_violation` | Declared phase lock broken | Fix generator or phase mapping |
| `parity` | QA/baseline seed or config mismatch | Re-run with strictly matched pairing |
| `reduction` | Claim emitted without baseline evidence | Include baseline metrics/trace and re-emit |

## Changelog

- **v1.0.0** (2026-02-09): First graph-structure cert emission + bundle manifest + meta-validator gate.
