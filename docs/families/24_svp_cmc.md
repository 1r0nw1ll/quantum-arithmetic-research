# [24] QA SVP-CMC

## What this is

Certifies **Sympathetic Vibratory Physics — Cause Mechanics Core** (SVP-CMC) analyses. This is the first non-ML domain formalization in the QA certificate ecosystem: it applies the generator-controlled invariant framework to cause-first physics, where scalar configuration precedes all kinetic effects. Includes an 18-entry obstruction ledger for causal-language validation.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__SVP_CMC.yaml` |
| Semantics cert | `certs/QA_SVP_CMC_SEMANTICS_CERT.v1.json` |
| Witness pack | `certs/witness/QA_SVP_CMC_WITNESS_PACK.v1.json` |
| Counterexamples pack | `certs/counterexamples/QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1.json` |
| Analysis cert schema | `schemas/QA_SVP_CMC_ANALYSIS_CERT.v1.schema.json` |
| Semantics cert schema | `schemas/QA_SVP_CMC_SEMANTICS_CERT.v1.schema.json` |
| Witness schema | `schemas/QA_SVP_CMC_WITNESS_PACK.v1.schema.json` |
| Counterexamples schema | `schemas/QA_SVP_CMC_COUNTEREXAMPLES_PACK.v1.schema.json` |
| Validator | `qa_svp_cmc_validator.py` |
| Obstruction ledger | `qa_ledger__radionics_obstructions.v1.yaml` |
| Ledger sanity checker | `qa_radionics_ledger_sanity.py` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate demo certificate
python qa_svp_cmc_validator.py --demo

# Validate custom cert against ledger
python qa_svp_cmc_validator.py --cert path/to/cert.json --ledger qa_ledger__radionics_obstructions.v1.yaml

# Rehash a cert pack
python qa_svp_cmc_validator.py --rehash certs/witness/QA_SVP_CMC_WITNESS_PACK.v1.json

# Check ledger sanity
python qa_radionics_ledger_sanity.py --ledger qa_ledger__radionics_obstructions.v1.yaml

# Or via meta-validator (runs as test [24])
python qa_meta_validator.py
```

## Semantics

### Core principles

| Principle | Meaning |
|-----------|---------|
| **Geometric Action Principle** | All physical effects arise from geometric (scalar) configuration |
| **No Instantaneous Action** | Every transition requires positive path-length / latency |
| **Scalar-Kinetic Distinction** | Configuration (cause) is categorically distinct from motion (effect) |

### Scalar parameters (8)

`subdivision_frequency_band`, `phase_relationships`, `angle_orientation`, `amplitude_degree`, `polarity_arrangement`, `boundary_conditions`, `neutral_center_placement`, `disturbance_rate`

### Analysis procedure (7 steps)

| Step | Description |
|------|-------------|
| S1 | Identify subject type |
| S2 | Extract scalar configuration |
| S3 | Check forbidden causal patterns |
| S4 | Verify latency requirements |
| S5 | Verify neutral center placement |
| S6 | Check symmetry witnesses |
| S7 | Emit certified analysis |

### Forbidden causal patterns

1. Energy as cause
2. Force as cause
3. Transmission language (energy "sent" between objects)
4. Instantaneous action
5. Immediate effect
6. Frequency as cause (rates are identifiers, not physical frequencies)

### Subject types

`device`, `organism`, `phenomenon`, `state`, `question`, `protocol`

## Failure modes

The semantics cert declares 21 fail types. Key categories:

| Category | fail_types |
|----------|-----------|
| Causal model | `TRANSMISSION_MODEL_FORBIDDEN`, `INSTRUMENT_AS_SOURCE`, `ENERGY_AS_CAUSE` |
| Latency | `NO_INSTANT_ACTION`, `LATENCY_NOT_RECORDED`, `LATENCY_SECTION_MISSING` |
| Scalar | `OPERATOR_COHERENCE_IGNORED`, `TARGET_IDENTITY_UNDER_SPECIFIED`, `SCALAR_INCOMPLETE` |
| Category errors | `RATE_AS_FREQUENCY_ERROR`, `AMPLITUDE_AS_TYPE_ERROR` |
| Geometry | `NEUTRAL_CENTER_MISPLACED`, `SYMMETRY_NO_WITNESS` |
| Policy | `CONSERVATION_AS_CAUSE`, `KINETICS_AS_CAUSE_LANGUAGE` |
| Polarity | `POLARITY_CONTRADICTION` |

### Obstruction ledger

The ledger (`qa_ledger__radionics_obstructions.v1.yaml`) contains 18 named obstruction entries. Each entry specifies:
- Obstruction ID and description
- Which analysis step it blocks
- Corresponding fail_type
- Example trigger text

## Example

**Passing** — tuning fork resonance analysis:
```json
{
  "certificate_type": "SVP_CMC_ANALYSIS",
  "subject_type": "device",
  "subject_label": "tuning_fork_resonance",
  "scalar_configuration": {
    "subdivision_frequency_band": "A4_440Hz",
    "phase_relationships": "sympathetic_coupling",
    "neutral_center_placement": "fork_tine_midpoint"
  },
  "forbidden_pattern_check": {"passed": true, "violations": []},
  "latency_check": {"positive_path_length": true}
}
```

**Failing** — transmission model forbidden:
```json
{
  "forbidden_pattern_check": {
    "passed": false,
    "violations": ["energy transmitted from source to target"]
  },
  "expected_fail_type": "TRANSMISSION_MODEL_FORBIDDEN"
}
```

## Changelog

- **v1.4.0** (2026-02-08): Initial scaffold — validator + demo cert + ledger sanity.
- **v1.4.1** (2026-02-09): Added triplet pattern (witness + counterexamples packs), `--rehash` support, counterexample taxonomy.
