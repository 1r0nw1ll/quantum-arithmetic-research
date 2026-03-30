# QA Ontology Map (Semantic Core)

This file is a **pointer map** to the semantic core: schemas, certificate families, validators, invariants, and canonical mappings.

If you’re asking “what does QA *mean* / what is valid?”, start here.

## Canonical definitions

- Canonical QA definitions (verbatim usage guidance in `AGENTS.md`):
  - `Formalizing tuple drift in quantum-native learning/files/files(1)/qa_canonical.md`
- External-system mapping example (EBM reasoning / Kona podcast → QA):
  - `Documents/QA_MAP__EBM_REASONING_KONA_PODCAST.md`

## Core QA spine (ptolemy)

- Directory: `qa_alphageometry_ptolemy/`
  - Certificate spine docs: `qa_alphageometry_ptolemy/QA_DECISION_CERTIFICATE_SPINE.md`
  - Canonical mapping registry: `qa_alphageometry_ptolemy/QA_MAP_CANONICAL.md`
  - TLA+ specs (formal): `qa_alphageometry_ptolemy/QACertificateSpine.tla`, `qa_alphageometry_ptolemy/QARM_v02_*.tla`
  - Schemas: `qa_alphageometry_ptolemy/schemas/`
  - Datastore/view semantics + validators:
    - `qa_alphageometry_ptolemy/qa_datastore_validator.py`
    - `qa_alphageometry_ptolemy/qa_datastore_view_validator.py`

## Root QA axioms/theorems

- Axioms block: `QA_AXIOMS_BLOCK.md`
- Control theorems: `QA_CONTROL_THEOREMS.md`
- Pipeline drift (axioms/invariants): `QA_PIPELINE_AXIOM_DRIFT.md`

## “Ontology vs execution” note

Some files are executable *and* ontological (e.g., validators). For cartography purposes, treat validators as ontology: they define validity.
