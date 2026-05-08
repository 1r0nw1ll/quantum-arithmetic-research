# [276] QA-ML Orbit Topology Cert

## What this is

Certifies that the QA generator reachability graph (sigma, mu, lambda_2, nu)
under symmetric GCN normalization lifts node-classification macro F1 by at
least +0.10 over an identity-adjacency ablation, holding node features,
architecture, seeds, and standardization fixed. Verified empirically across
nine moduli `m in {9, 12, 15, 18, 21, 24, 27, 30, 36}`.

## Primary source

- Kipf, T. N. & Welling, M. (2017). *Semi-Supervised Classification with Graph
  Convolutional Networks*. ICLR. arxiv:1609.02907.

## Artifacts

| Artifact | Path |
|---|---|
| Mapping protocol ref | `qa_ml_orbit_topology_cert_v1/mapping_protocol_ref.json` |
| Schema | `qa_ml_orbit_topology_cert_v1/schema.json` |
| Validator | `qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py` |
| Fixtures | `qa_ml_orbit_topology_cert_v1/fixtures/{pass_m9,pass_m24,fail_below_threshold,fail_missing_field}.json` |
| README | `qa_ml_orbit_topology_cert_v1/README.md` |
| SPEC | `qa_ml_orbit_topology_cert_v1/SPEC.md` |
| Generating experiment | `experiments/qa_ml/03_gnn_modulus_sweep.py` |
| Benchmark protocol | `experiments/qa_ml/benchmark_protocol_v2_modulus_sweep.json` |
| Sweep results | `experiments/qa_ml/results_gnn_modulus_sweep.json` |
| Reproducibility ledger | `experiments/qa_ml/results_ledger_v2_modulus_sweep.jsonl` |

## How to run

```bash
cd qa_alphageometry_ptolemy

# Structural validation
python qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py

# Plus smoke re-run on m=9
python qa_ml_orbit_topology_cert_v1/qa_ml_orbit_topology_cert_validate.py --smoke

# Or via meta-validator
python qa_meta_validator.py
```

## Semantics

- **ORBT_1**: schema conformance (required fields, fixture kind in {pass, fail}).
- **ORBT_2**: `graph_delta == with_graph_macro_f1 - without_graph_macro_f1` within tolerance.
- **ORBT_3**: PASS fixture has `graph_delta >= 0.10` and `passes_threshold = true`.
- **ORBT_4**: FAIL fixture trips its declared `expected_fail_type` (BELOW_THRESHOLD, MISSING_FIELD, ARITHMETIC).
- **SRC**: `mapping_protocol_ref.json` present with required fields.
- **F**: every FAIL fixture declares `expected_fail_type`.

## Failure modes

| fail_type | Meaning | Fix |
|---|---|---|
| `BELOW_THRESHOLD` | Reported `graph_delta` below 0.10 | Improve graph adjacency or revisit ablation matching |
| `MISSING_FIELD` | Required schema field absent | Add the field to fixture JSON |
| `ARITHMETIC` | `graph_delta != with - without` | Recompute and update fixture |

## Example

**Passing** (`fixtures/pass_m9.json`):

```json
{
  "schema_version": "QA_ML_ORBIT_TOPOLOGY_CERT.v1",
  "fixture_kind": "pass",
  "modulus": 9,
  "n_pairs": 81,
  "n_satellite": 8,
  "with_graph_macro_f1_mean": 1.000,
  "without_graph_macro_f1_mean": 0.696,
  "graph_delta": 0.304,
  "passes_threshold": true
}
```

**Failing** (`BELOW_THRESHOLD` in `fixtures/fail_below_threshold.json`):

```json
{
  "fixture_kind": "fail",
  "expected_fail_type": "BELOW_THRESHOLD",
  "with_graph_macro_f1_mean": 0.620,
  "without_graph_macro_f1_mean": 0.576,
  "graph_delta": 0.044,
  "passes_threshold": false
}
```

## References

- Kipf & Welling 2017, arxiv:1609.02907 (GCN spectral formulation).
- `qa_orbit_rules.py` — canonical orbit family and period via qa_step (A1-compliant).
- `tools/qa_ml/qa_generators.py` — sigma, mu, lambda_2, nu definitions.
- `tools/qa_ml/qa_graph.py` — `build_edges`, `dense_adjacency`, `gcn_normalize`.

## Boundary observation (NOT certified)

The `qa_orbit_rules.orbit_family` algebraic rule (`(m//3)|b AND (m//3)|e`)
under-counts period-8 pairs for `m in {15, 30}` (32 missed cases each).
Flagged for follow-up; does not affect the cert claim, which is grounded in
empirical orbit period.

## Changelog

- **v1.0.0** (2026-05-08): Initial release; nine-modulus sweep PASS; smoke ok.
