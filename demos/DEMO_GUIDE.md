# QA Cert Family Demo Guide

QA (Quantum Arithmetic) applies a universal invariant contract to radically different domains — geometry, cellular automata, energy models, governance — and classifies structural failure deterministically.

---

## Quick Start

```bash
python demos/qa_family_demo.py --family geogebra
python demos/qa_family_demo.py --family rule30
```

---

## What You Will See

### GeoGebra (Family [56])

Domain: exact geometric scenes exported from GeoGebra, using Z/Q typed coordinates
(`{"k":"Z","v":3}` for integers, `{"k":"Q","n":1,"d":4}` for rationals).

**PASS case:** A 3-4-5 right triangle with exact integer coordinates. The validator
computes spread invariants via cross-multiplication to integer equality (zero tolerance),
verifies the Pythagorean spread law, and confirms the step hash chain. All gates clear.

**FAIL case:** A point with a zero-denominator rational coordinate (`{"k":"Q","n":1,"d":0}`).
The `invariant_diff` field in the result records `ZERO_DENOMINATOR` — a typed obstruction
that identifies the structural impossibility precisely, not just a parse error.

### Rule 30 (Family [34])

Domain: Wolfram's Rule 30 elementary cellular automaton, certified for non-periodicity
over a finite observation window.

**PASS case:** A witnessed run where the center column aggregate hash matches the
certified witness. The cert pack establishes that no sub-period divides the run length,
confirming non-periodicity under the stated parameters.

**FAIL case:** A tampered or mismatched witness produces an `invariant_diff` recording
`AGGREGATE_MISMATCH` — the same field, the same structure, a different domain.

---

## Why This Matters

- The `invariant_diff` field is identical in structure whether the domain is a geometry
  scene, a cellular automaton run, an energy model, or an agent trace. Failure is
  classified, not described.

- This is not logging errors. It is classifying structural impossibility: a zero
  denominator is not a bad input — it is a proof that no valid geometric object can
  exist at that location.

- Exact substrates (families [50], [56]) eliminate floating-point tolerance tuning.
  Spread laws are verified by integer cross-multiplication; either the equality holds
  or it does not.

- The same Gate 0-3 pipeline (mapping protocol check, schema validation, invariant
  computation, hash-chain verification) applies uniformly across all 26 documented
  families. Adding a new domain means implementing the gates, not redesigning the
  contract.

- All 60 meta-validator tests pass simultaneously. The families are not isolated
  experiments — they share a single validator sweep that catches regressions across
  the entire cert ecosystem in one run.

---

## Full Family Map

| ID | Domain |
|----|--------|
| [18]–[24] | Cert-triplet families: semantics, witness, and counterexample bundles for seven core QA cert types |
| [26] | Competency detection: identifies structural competency signals in agent outputs |
| [27] | Elliptic correspondence: maps QA orbits to elliptic curve point structure |
| [28] | Graph structure: certifies graph-theoretic invariants under QA state transitions |
| [29] | Agent traces: schema, validator, and hash-chain fixtures for recorded agent execution traces |
| [30] | Agent trace competency: derives task-level competency metrics from trace->task->dominance chains |
| [31] | Math compiler stack: certified trace + pair cert for the QA math compiler pipeline |
| [32] | Conjecture-prove control loop: episode, frontier, and receipt for automated conjecture cycling |
| [33] | Discovery pipeline: run, plan, and bundle artifacts for the theorem discovery orchestrator |
| [34] | Rule 30 certified discovery: cert pack + witnesses for cellular automaton non-periodicity — **demo** |
| [35] | QA Mapping Protocol (inline): `mapping_protocol.json` schema and validator |
| [36] | QA Mapping Protocol REF: `mapping_protocol_ref.json` indirection layer |
| [37] | EBM Navigation Cert: energy-based model navigation episode certification |
| [38] | Energy-Capability Separation Cert: formal separation proof between energy and capability scores |
| [39] | EBM Verifier Bridge Cert: bridge between EBM cert emission and external verifier consumption |
| [44] | Rational Trig Type System: Wildberger rational trig + Martin-Lof type theory mapped to QA state manifolds |
| [45] | ARTexplorer Scene Adapter (float64): parses ARTexplorer JSON scenes, computes RT invariants, relative tolerance |
| [50] | ARTexplorer Scene Adapter v2 (exact): same adapter, integer-pair substrate, zero tolerance |
| [55] | Three.js Scene Adapter (float64): adapts Three.js JSON scene exports, float64 compute substrate |
| [56] | GeoGebra Scene Adapter (exact Z/Q): GeoGebra exports with typed integer/rational coordinates, LCM lift — **demo** |

Families [18]–[24] account for seven entries; the table above lists the 20 distinct
family roots that together cover all 26 documented families.

---

## Running All 60 Meta-Validator Tests

```bash
cd qa_alphageometry_ptolemy && python qa_meta_validator.py
```

All 60 tests are expected to pass. The sweep covers every documented family, all
mapping protocol gates, external validation contracts (SWE-bench, prompt injection),
and the human-tract doc gate (26 families documented).

To verify the mapping protocol validators independently:

```bash
python qa_mapping_protocol/validator.py --self-test
python qa_mapping_protocol_ref/validator.py --self-test
```
