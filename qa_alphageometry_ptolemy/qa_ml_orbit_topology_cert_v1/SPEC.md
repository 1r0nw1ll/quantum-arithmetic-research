# [276] QA-ML Orbit Topology Cert — Spec

> Glossary: "Theorem NT" — i.e. the Observer Projection Firewall axiom (an invariant that bars float values from re-entering the QA discrete layer; see `docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md`).

## Primary source

- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph
  Convolutional Networks. *ICLR*. arxiv:1609.02907.

## Schema (fixture)

`schema.json` declares `QA_ML_ORBIT_TOPOLOGY_CERT.v1`. Required fields:

| Field | Meaning |
|---|---|
| `schema_version` | const `"QA_ML_ORBIT_TOPOLOGY_CERT.v1"` |
| `fixture_kind` | `"pass"` or `"fail"` |
| `modulus` | QA orbit grid modulus `m` (int >= 2) |
| `train_fraction` | fraction of labeled nodes in train mask (0..1) |
| `n_seeds` | number of stratified shuffle splits |
| `epochs` | GCN training epochs |
| `hidden` | GCN hidden width |
| `n_satellite` | empirical period-8 pair count for this `m` |
| `n_pairs` | total `m^2` pairs |
| `with_graph_macro_f1_mean` | macro F1 with QA-generator adjacency |
| `without_graph_macro_f1_mean` | macro F1 with identity adjacency (ablation) |
| `graph_delta` | with - without (must equal arithmetic within tolerance) |
| `passes_threshold` | declared bool: `graph_delta >= 0.10` |

Optional: `expected_fail_type` (only for fail fixtures), `tolerance`,
`raw_mlp_macro_f1_mean`, `qa_full_logreg_macro_f1_mean`, `primary_source`.

## Checks

- **ORBT_1** — schema conformance (required fields present, schema_version
  matches, fixture_kind in {pass, fail}).
- **ORBT_2** — `graph_delta` arithmetic equals `with - without` within
  declared tolerance.
- **ORBT_3** — PASS fixture has `graph_delta >= 0.10` and
  `passes_threshold == true`.
- **ORBT_4** — FAIL fixture trips its declared `expected_fail_type`:
  - `BELOW_THRESHOLD`: `graph_delta < 0.10` and `passes_threshold = false`.
  - `MISSING_FIELD`: schema check fails on at least one required field.
  - `ARITHMETIC`: declared `graph_delta` disagrees with `with - without`
    beyond tolerance.
- **SRC** — `mapping_protocol_ref.json` present with required fields.
- **F** — every FAIL fixture declares `expected_fail_type`.

## Smoke (--smoke flag)

Re-runs the GCN ablation on `m=9` with `n_seeds=3`, `epochs=80`,
`train_fraction=0.30`, asserts mean `graph_delta >= 0.10`. Skipped silently
if torch is not importable. ~3-5 seconds wall.

## Failure modes

| `expected_fail_type` | Meaning | Fixture |
|---|---|---|
| `BELOW_THRESHOLD` | Reported `graph_delta` below 0.10 | `fail_below_threshold.json` |
| `MISSING_FIELD` | Required schema field absent | `fail_missing_field.json` |
| `ARITHMETIC` | `graph_delta != with - without` (not currently fixtured) | (reserved) |

## Theorem NT compliance

QA-side state space is integer `(b, e) in {1,...,m}^2`. The reachability
adjacency is built from integer-valued generators (`tools.qa_ml.qa_generators`).
The GCN itself is float-side observer arithmetic. The Theorem NT firewall is
crossed exactly twice: integer feature packet -> torch tensor (input), GCN
logits -> argmax orbit class (output). No float feedback into the QA layer.

## Lineage

- Layer 1 (no cert): `experiments/qa_ml/01_baseline_orbit_classifier.py` and
  `benchmark_protocol.json` — sample-efficiency benchmark on mod-24 showing
  polynomial QA expansion is weak; modular-phase columns carry orbit signal.
- Layer 2 (no cert): `experiments/qa_ml/02_gnn_orbit_classifier.py` and
  `benchmark_protocol_v2.json` — first GCN result on mod-24 showing the
  +0.14..+0.23 graph contribution at `n_train in {40, 80, 160, 320}`.
- Layer 3 (this cert): `experiments/qa_ml/03_gnn_modulus_sweep.py` and
  `benchmark_protocol_v2_modulus_sweep.json` — boundary-hardening sweep
  across nine moduli; all PASS; cert grounds the empirical claim.
