# QA Certificates

This directory contains the **QA Core System Stack** - a certified three-pillar framework establishing QA as a cross-domain coordination language.

## Core System Stack

The `QA_CORE_SYSTEM_STACK.v1.json` certificate defines three layers:

| Layer | Description | Witnesses |
|-------|-------------|-----------|
| **Agent OS** | Deterministic generators, regime selection, oracle-efficient control | claude_skills, context_geometry, rml_paper3 |
| **Model Substrate** | Resonance MoE, ellipse embeddings, tuple memory, topology learning | kimi_k2, kosmos, clara, qawm_paper2 |
| **Physical Substrate** | J/X/K energetics, QA memory cells | stat_mech, quantum_memory |

## Cross-Invariants

- `CI_TUPLE_CLOSURE` - All layers preserve (b,e,d,a) closure
- `CI_ELLIPSE_OPERATORS` - J,X,K maintain semantics across layers
- `CI_GENERATOR_DETERMINISM` - Replayable traces
- `CI_MODULAR_RESONANCE` - Mod-24/mod-9 govern all scales

## Running the Validator

```bash
# From repo root
python qa_alphageometry_ptolemy/certs/qa_core_stack_validate.py

# From qa_alphageometry_ptolemy/
python certs/qa_core_stack_validate.py
```

Expected output:
```
Certificate exists:     PASS
Schema valid:           PASS
Witnesses valid:        PASS
Cross-invariants valid: PASS
Layers valid:           PASS
Drift check:            PASS
Golden fixtures:        PASS

OVERALL: PASS
```

## Adding a New Witness

1. Create `witness/core_stack/<name>.witness.json` with required fields:
   ```json
   {
     "witness_id": "WITNESS__<NAME>__v1",
     "source": { "type": "...", "title": "...", "file": "..." },
     "claim": "...",
     "qa_mapping": { "layer": "AGENT_OS|MODEL_SUBSTRATE|PHYSICAL_SUBSTRATE", ... },
     "generators": ["sigma", "mu", ...],
     "invariants": [...],
     "failure_modes": [...]
   }
   ```

2. Add reference to `QA_CORE_SYSTEM_STACK.v1.json` witnesses array

3. Update `qa_core_stack_validate.py` name_map if needed

4. Run validator to confirm

## File Structure

```
certs/
├── QA_CORE_SYSTEM_STACK.v1.json    # Main certificate
├── qa_core_stack_validate.py        # Validator
├── expected_hashes.json             # Drift detection
├── README.md                        # This file
├── fixtures/
│   └── golden_fail_invalid_layer.json
└── witness/core_stack/
    ├── claude_skills.witness.json
    ├── context_geometry.witness.json
    ├── kimi_k2.witness.json
    ├── kosmos.witness.json
    ├── clara.witness.json
    ├── stat_mech.witness.json
    ├── quantum_memory.witness.json
    ├── qawm_paper2.witness.json      # Paper 2: QAWM topology learning
    └── rml_paper3.witness.json       # Paper 3: RML oracle-efficient control
```

## Trilogy Integration

The Core System Stack includes witnesses for the QA research trilogy:

| Paper | Title | Witness | Layer | Key Result |
|-------|-------|---------|-------|------------|
| Paper 1 | QA Transition System | (canonical reference) | Foundation | 21-element invariant packet, generator algebra |
| Paper 2 | QAWM | qawm_paper2.witness.json | MODEL_SUBSTRATE | 0.836 AUROC return-in-k, Cross-Caps generalization |
| Paper 3 | RML | rml_paper3.witness.json | AGENT_OS | 4.20 vs 2.97 normalized success, topology-over-constraints |

### Papers Status

- **Paper 1**: Complete (canonical reference at `qa_canonical.md`)
- **Paper 2**: Publication ready (QAWM world model learning)
- **Paper 3**: Writing complete (RML oracle-efficient control)

Full paper documentation in `Formalizing tuple drift in quantum-native learning/` directory (canonical) and mirrored in `gemini_qa_project/`.

## Version

- **Tag**: `qa-core-stack-v1.1.0`
- **Schema**: `schemas/QA_CORE_SYSTEM_STACK.v1.schema.json`
