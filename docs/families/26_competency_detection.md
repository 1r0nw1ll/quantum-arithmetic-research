# [26] QA Competency Detection

## What this is

A substrate-independent framework for detecting, measuring, and certifying
emergent competencies in biological, artificial, and hybrid systems.  Aligned
with Michael Levin's Platonic Space research programme: "we are still looking
for suites of tools to identify novel competencies."

QA operationalises competency as **reachable goal-directed transformation
under constraints**, making it formally verifiable and substrate-neutral.

## Levin-to-QA term mapping

| Levin term   | QA term               | Definition                    |
|--------------|-----------------------|-------------------------------|
| Competency   | Reachability Class    | Set of achievable states      |
| Goal         | Attractor Basin       | Stable terminal region        |
| Memory       | Invariant             | Preserved constraint          |
| Agency       | Control Region        | Navigable subspace            |
| Intelligence | Transform Capacity    | State-space volume            |
| Plasticity   | Generator Flexibility | Adaptation under perturbation |
| Morphospace  | State Manifold        | Embedded configuration space  |

## Artifacts

| Artifact | Path |
|----------|------|
| Bundle cert | `qa_competency/certs/QA_COMPETENCY_CERT_BUNDLE.v1.json` |
| Validator | `qa_competency/qa_competency_validator.py` |
| Metrics module | `qa_competency/qa_competency_metrics.py` |
| Framework schema | `qa_competency/schemas/QA_COMPETENCY_DETECTION_FRAMEWORK.v1.schema.json` |
| Bundle schema | `qa_competency/schemas/QA_COMPETENCY_CERT_BUNDLE.v1.schema.json` |
| Valid fixtures | `qa_competency/fixtures/valid/` |
| Invalid fixtures | `qa_competency/fixtures/invalid/` |

All paths relative to repository root.

## How to run

```bash
# Validate demo cert (self-test)
python qa_competency/qa_competency_validator.py --demo

# Run fixture assertions
python qa_competency/qa_competency_validator.py --fixtures

# Validate the bundle
python qa_competency/qa_competency_validator.py --validate-bundle \
    qa_competency/certs/QA_COMPETENCY_CERT_BUNDLE.v1.json

# Recompute manifest hashes
python qa_competency/qa_competency_validator.py --rehash

# Or via meta-validator
cd qa_alphageometry_ptolemy
python qa_meta_validator.py
```

## Semantics

### Competency metrics (four canonical measures)

- **Agency Index (AI)**: `|reachable_states| / |total_states|`.
  Measures internal control capacity. High = strong self-direction.

- **Plasticity Index (PI)**: `delta_reachability / delta_perturbation`.
  Measures adaptability under perturbation. High = developmental flexibility.

- **Goal Density (GD)**: `|attractor_basins| / |total_states|`.
  Measures how many stable outcomes exist. High = multiple achievable phenotypes.

- **Control Entropy (CE)**: `-sum p(move) ln p(move)` (natural log).
  Measures decision freedom. High = decentralised intelligence.

### Determinism rules

- Metrics are **recomputed** from `metric_inputs` and compared to
  `competency_metrics` (tolerance 1e-9). Mismatches fail with
  `METRIC_MISMATCH`.
- `graph_snapshot` is required: captures the state-graph hash and time window.
- `manifest` uses SHA-256 of canonical JSON (sorted keys, compact separators).
  Placeholder sentinel: 64 hex zeros.
- `reproducibility_seed` in `validation` block enables deterministic replay.

## Failure modes

| fail_type | Meaning | invariant_diff | Fix |
|-----------|---------|----------------|-----|
| `SCHEMA_VALIDATION_FAILED` | JSON Schema violation | `{"errors": [...]}` | Fix cert structure per schema |
| `MISSING_REQUIRED_BLOCK` | Required block absent | `{"missing": [...]}` | Add the missing block |
| `METRIC_MISMATCH` | Declared metric != recomputed | `{"field": "...", "declared": ..., "computed": ...}` | Correct metric or inputs |
| `MANIFEST_HASH_MISMATCH` | Manifest hash wrong | `{"claimed": "...", "computed": "..."}` | Run `--rehash` |

## Example

**Passing** (planarian regeneration):
```json
{
  "schema_id": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
  "system_metadata": {
    "domain": "developmental_biology",
    "substrate": "cell_collective",
    "description": "Planarian-like regeneration competency"
  },
  "metric_inputs": {
    "reachable_states": 710,
    "total_states": 1000,
    "attractor_basins": 120,
    "move_probabilities": {"cut": 0.5, "voltage_shift": 0.5},
    "delta_reachability": 83.0,
    "delta_perturbation": 100.0
  },
  "competency_metrics": {
    "agency_index": 0.71,
    "plasticity_index": 0.83,
    "goal_density": 0.12,
    "control_entropy": 0.6931471805599453
  }
}
```

**Failing** (`METRIC_MISMATCH`):
```json
{
  "metric_inputs": {
    "reachable_states": 710,
    "total_states": 1000
  },
  "competency_metrics": {
    "agency_index": 0.99
  }
}
```
Declared agency_index 0.99 but inputs give 710/1000 = 0.71.

## Reference sets

Curated exemplar bundles in `qa_competency/reference_sets/v1/` across three
domains (bio, AI, hybrid). Nine bundles total, each a valid
`QA_COMPETENCY_CERT_BUNDLE.v1` with 1-2 certs.

> **Disclaimer:** Reference sets are curated exemplars; they are not claims
> about specific wet-lab experiments, deployed AI systems, or clinical protocols.

| Domain | Bundle | Highlights |
|--------|--------|------------|
| bio | `planarian_regen` | 2 certs, regeneration + memory retention |
| bio | `xenopus_patterning` | High agency (0.88), low goal density (0.04) |
| bio | `organoid_self_repair` | 7 generators, high entropy (~1.94) |
| ai | `tool_agent_debugger` | Low plasticity (0.30), brittle agent |
| ai | `retrieval_agent_arag` | Citation-fidelity invariant |
| ai | `multi_agent_coordination` | 7 generators, 3 components, distributed |
| hybrid | `bioelectric_controller_with_llm` | Two-substrate bio+digital |
| hybrid | `lab_robot_closed_loop` | Low entropy deterministic protocol |
| hybrid | `human_in_the_loop_protocol` | Highest agency (0.91) |

Validate with:
```bash
python qa_competency/qa_competency_validator.py --reference-sets
```

Reference sets are also validated during the family sweep (`validate_all`).

## Changelog

- **v1.1.0** (2026-02-09): Add reference pack v1 (9 bundles across bio/AI/hybrid).
- **v1.0.0** (2026-02-09): Initial release. Levin-aligned competency framework.
