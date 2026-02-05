# QA Certificates

This directory contains the **QA Core System Stack** - a certified three-pillar framework establishing QA as a cross-domain coordination language.

## Core System Stack

The `QA_CORE_SYSTEM_STACK.v1.json` certificate defines three layers:

| Layer | Description | Witnesses |
|-------|-------------|-----------|
| **Agent OS** | Deterministic generators, regime selection | claude_skills, context_geometry |
| **Model Substrate** | Resonance MoE, ellipse embeddings, tuple memory | kimi_k2, kosmos, clara |
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
└── witness/core_stack/
    ├── claude_skills.witness.json
    ├── context_geometry.witness.json
    ├── kimi_k2.witness.json
    ├── kosmos.witness.json
    ├── clara.witness.json
    ├── stat_mech.witness.json
    └── quantum_memory.witness.json
```

## Version

- **Tag**: `qa-core-stack-v1.0.0`
- **Schema**: `schemas/QA_CORE_SYSTEM_STACK.v1.schema.json`
