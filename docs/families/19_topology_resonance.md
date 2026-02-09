# [19] Topology Resonance Bundle

## What this is

Certifies generator-induced topological reachability in the QA state space. The canonical generator set `{sigma, mu, lambda2, nu}` produces SCC (strongly connected component) structures whose growth and phase coherence are certified as bundle artifacts. This is the geometric foundation — it proves the QA state space has non-trivial topology under its generators.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__TOPOLOGY_RESONANCE.yaml` |
| Validator | `qa_topology_resonance_validator_v1.py` |
| Bundle emitter/validator | `qa_topology_resonance_bundle_v1.py` |
| Reference cert | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.json` |
| Cert schema | `schemas/QA_TOPOLOGY_RESONANCE_CERT.v1.schema.json` |
| Bundle schema | `schemas/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.schema.json` |
| Success example | `examples/topology/topology_resonance_success.json` |
| Failure example | `examples/topology/topology_phase_break_failure.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate individual cert
python qa_topology_resonance_validator_v1.py --demo

# Validate with recompute level
python qa_topology_resonance_validator_v1.py --level recompute examples/topology/topology_resonance_success.json

# Emit + check bundle
python qa_topology_resonance_bundle_v1.py --emit --check

# Or via meta-validator (runs as test [19])
python qa_meta_validator.py
```

## Semantics

- **Generator set**: `{sigma, mu, lambda2, nu}` — the canonical QA topology generators
- **SCC monotonicity**: `scc_count_after >= scc_count_before` — connectivity never decreases
- **Phase lock**: `phase_24` and `phase_9` preserved across legal transitions
- **Resonance threshold**: `resonance_score >= resonance_threshold` for certification
- **Hard invariants**: `packet_conservation`, `no_reduction_axiom`, `connected_component_first_class`

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `phase_break` | Phase-24 or phase-9 changed across transition | Fix generator to preserve phase |
| `scc_drop` | SCC count decreased | Illegal generator — review transition |
| `resonance_below_threshold` | Score below declared threshold | Adjust generator set or threshold |
| `packet_drift` | Packet conservation violated | Fix packet handling |
| `invalid_generator` | Generator not in `{sigma, mu, lambda2, nu}` | Use only canonical generators |

## Example

**Passing** — resonance certified:
```json
{
  "generator_set": ["sigma", "mu"],
  "topology_witness": {
    "scc_count_before": 3,
    "scc_count_after": 4,
    "resonance_score": "7/10",
    "resonance_threshold": "1/2",
    "resonance_certified": true
  },
  "invariants": {
    "scc_monotone_non_decreasing": true,
    "packet_conservation": true
  }
}
```

**Failing** — phase break:
```json
{
  "success": false,
  "failure_mode": "phase_break",
  "failure_witness": {
    "phase_24_before": 5,
    "phase_24_after": 7,
    "phase_9_before": 2,
    "phase_9_after": 2
  }
}
```

## Changelog

- **v1.0.0** (2026-02-07): First cert emission + bundle manifest with validator.
