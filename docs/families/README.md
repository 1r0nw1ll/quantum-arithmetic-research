# QA Certificate Family Index

Every QA certificate family is developed on **two tracts**:

1. **Machine tract**: schema, validator, cert bundle, counterexamples, meta-validator hook
2. **Human tract**: this documentation — what it is, how to run it, what breaks

A family **does not count as shipped** unless both tracts are present.

---

## Families

| ID | Name | Type | Status |
|----|------|------|--------|
| [18] | [QA Datastore](18_datastore.md) | Triplet (semantics + witness + counterexamples) | PASS |
| [19] | [Topology Resonance Bundle](19_topology_resonance.md) | Bundle manifest | PASS |
| [20] | [QA Datastore View](20_datastore_view.md) | Triplet | PASS |
| [21] | [QA A-RAG Interface](21_arag_interface.md) | Triplet | PASS |
| [22] | [QA Ingest->View Bridge](22_ingest_view_bridge.md) | Triplet | PASS |
| [23] | [QA Ingestion](23_ingestion.md) | Triplet | PASS |
| [24] | [QA SVP-CMC](24_svp_cmc.md) | Triplet + Ledger | PASS |
| [26] | [QA Competency Detection](26_competency_detection.md) | Bundle + Metrics | PASS |
| [27] | [QA Elliptic Correspondence](27_elliptic_correspondence.md) | Bundle + Deterministic Replay | PASS |
| [28] | [QA Graph Structure](28_graph_structure.md) | Bundle + Paired Deltas | PASS |

## Quick validation

```bash
# Run all families [1]-[28]
cd qa_alphageometry_ptolemy
python qa_meta_validator.py

# Fast mode (manifest integrity only)
python qa_meta_validator.py --fast
```

## Provenance chain

Families [18]-[23] form a certified provenance pipeline:

```
[23] Ingestion  -->  [22] Ingest->View Bridge  -->  [20] Datastore View  -->  [21] A-RAG Interface
                                                          |
                                                    [18] Datastore
```

[24] SVP-CMC is an independent domain family (cause-first physics).

[26] Competency Detection is a standalone portable module (`qa_competency/`)
aligned with Michael Levin's Platonic Space competency-detection programme.

## Two-tract checklist (for contributors)

Before shipping any family:

- [ ] Schema(s) committed
- [ ] Validator passes `--demo`
- [ ] Witness pack + counterexamples pack present
- [ ] Meta-validator hook wired in `qa_meta_validator.py`
- [ ] `docs/families/[NN]_<slug>.md` written
- [ ] This README index updated
- [ ] Release notes in `QA_MAP_CANONICAL.md` updated
