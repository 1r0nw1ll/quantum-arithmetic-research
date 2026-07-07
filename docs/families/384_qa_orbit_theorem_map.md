# [384] QA Orbit Theorem Map Cert

**Family**: `qa_orbit_theorem_map_cert_v1`  
**Validator**: `qa_orbit_theorem_map_cert_v1/qa_orbit_theorem_map_validate.py`  
**Tool wrapper**: `tools/qa_orbit_theorem_map_validate.py`

## Purpose

This family validates exported QA orbit theorem-map JSON as a falsifiable finite artifact.

The validator does not trust the workbench that generated the map. It recomputes orbit-family labels from integer `qa_step` dynamics, recomputes exact multibase features, replays declared theorem-map leaf paths, and compares recomputed train/test confusion matrices against the declarations in the theorem map.

v1 covers the theorem-map artifact generated from moduli `{9,12,15,18,21,24,27,30}`. The per-prime-factor-regime cover validates all held-out rows with zero errors: `1142/1142`.

## Schema

| Field | Meaning |
|---|---|
| `kind` | Must be `qa_orbit_theorem_map`. |
| `config.primes` | Prime bases used for exact residue and valuation features. |
| `config.train_moduli` | Moduli used to build the declared theorem map. |
| `config.test_moduli` | Moduli used for declared evaluation. |
| `config.split_within_moduli` | Whether deterministic within-modulus stratified split is used. |
| `nodes` | Theorem-map nodes. Includes global tree, per-regime cover, and exact regime trees. |
| `edges` | Theorem-map edges connecting global tree to regime-cover decomposition. |
| `leaf_paths` | Exact feature predicates leading to each predicted leaf. |
| `train_eval` / `test_eval` | Declared confusion matrices and error counts. |

## Validator Checks

| Check | Description |
|---|---|
| `TM_SCHEMA` | Required theorem-map structure and config fields are present. |
| `TM_PATH` | Every recomputed row matches exactly one declared leaf path. |
| `TM_ROOT` | Each regime node's declared `root_predicate` matches its first leaf-path predicate. |
| `TM_EVAL` | Recomputed train/test confusion matrices exactly match declarations. |
| `TM_COVER` | Per-regime cover totals recompute to the declared overall held-out result. |
| `SRC` | Family has `mapping_protocol_ref.json`. |
| `F` | Fail fixtures must fail for their declared failure mode. |

## Fixtures

| Fixture | Kind | Purpose |
|---|---|---|
| `pass_orbit_theorem_map_v1.json` | PASS | Current theorem-map export; validates all declared global and per-regime metrics. |
| `fail_corrupt_235_root_path.json` | FAIL | Corrupts the `2^1*3^1*5^1` root predicate and must trigger `TM_ROOT`. |

## Key v1 Result

The theorem map contains a zero-error per-regime node for `2^1*3^1*5^1`.

Root predicate:

```text
qa.v2_gcd_ge_m_and_r5_line_3=1
```

Held-out result:

```text
rows=301, errors=0, accuracy=1/1
```

The root predicate expresses a 2-adic scale condition coupled to a mod-5 line condition. This is the exact feature that closes the previously leaking `m=30` regime.

## Family Relationships

| Related family / artifact | Relationship |
|---|---|
| `[276] QA-ML Orbit Topology` | Earlier orbit-classification experiments motivating interpretable orbit models. |
| `[277] QA Orbit Pisano 5-Factor Boundary` | Prior finite orbit boundary result in 5-factor regimes. |
| `[278] QA Orbit No-3-Divisor Overclaim` | Companion failure-surface cert for shortcut overclaims. |
| `[279] QA Orbit Access Theorem` | Mod-9 route-access theorem and orbit-family grounding. |
| `[281] QA Pisano-Orbit Correspondence` | Pisano-period explanation for mod-9 orbit classes. |
| `tools/qa_multibase_workbench.py` | Generates exact multibase features, trees, row explanations, and theorem-map exports. |
| `results/qa_orbit_theorem_map_v1.json` | Source theorem-map artifact mirrored by the PASS fixture. |

## Non-Claims

This cert does not prove infinite-modulus generalization. It does not certify the model-selection process that discovered the tree. It does not claim the global one-tree model is zero-error. It certifies only that the exported theorem-map JSON is internally exact for the declared finite orbit grids and declared deterministic split.

## Commands

```bash
python3 tools/qa_orbit_theorem_map_validate.py --self-test
python3 tools/qa_orbit_theorem_map_validate.py results/qa_orbit_theorem_map_v1.json
```

## Verification Note (2026-07-07)

Confirmed clean, no bugs. `python3 tools/qa_orbit_theorem_map_validate.py
--self-test` returns `{"ok": true, "errors": [], "pass_fixtures": 1,
"fail_fixtures": 1}`. The validator genuinely recomputes orbit labels
from integer `qa_step` dynamics and replays leaf paths against the
declared theorem-map JSON rather than trusting it — consistent with
the doc's stated non-claims (finite-grid only, no infinite-modulus
generalization claimed).
