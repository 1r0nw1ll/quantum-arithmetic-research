# [105] QA Cymatics Correspondence family

## What this is

Four-tier cert family mapping foundational cymatics scholarship to QA, realized as deterministic validators over a recognition cert, a reachability cert, a control cert, and a planner cert:

- **QA_CYMATIC_MODE_CERT.v1** — Chladni eigenmode witness: certifies that a plate/membrane at resonance occupies a specific (m,n) mode and that the nodal structure maps to a valid QA (b,e) state pair. The Chladni formula `a = m + 2n` is verified to echo QA's tuple derivation `a = b + 2e`.
- **QA_FARADAY_REACHABILITY_CERT.v1** — Faraday pattern-basin reachability: certifies legal transitions in a fluid Faraday setup and maps pattern classes (flat/stripes/hexagons) to QA orbit families (singularity/satellite/cosmos).
- **QA_CYMATIC_CONTROL_CERT.v1** — Cymatic control/programmability: certifies that a lawful generator sequence drives a cymatic system from an initial state to a target pattern, matching a QA orbit-family reachability claim.
- **QA_CYMATIC_PLANNER_CERT.v1** — Cymatic plan synthesis: certifies that a bounded search (BFS/DFS) over the legal generator alphabet either found a minimal plan reaching the target, or provably exhausted the bound. The plan is a replayable step-by-step witness; a no-plan cert documents the obstruction class. Optional `compiled_control_certificate_id` + `compiled_control_witness_hash` + `replay_consistent` fields close the compilation loop: the planner cert proves its plan compiles into the referenced control cert (hash-pinned integrity check). QA analogue: bounded orbit-graph reachability search from singularity to cosmos, plus a proof that the generator sequence corresponds to a pinned certified controller.

## Core claim

> Cymatics is the experimental study of how lawful resonance generators drive matter into visible, boundary-conditioned geometric states. QA is the formal study of how lawful arithmetic generators drive embedded structures into reachable geometric states.

## Artifacts

| Artifact | Path |
|----------|------|
| Correspondence ledger | `qa_alphageometry_ptolemy/qa_cymatics/qa_cymatics_correspondence_map.json` |
| Validator | `qa_alphageometry_ptolemy/qa_cymatics/qa_cymatics_validate.py` |
| Mode cert schema | `qa_alphageometry_ptolemy/qa_cymatics/schemas/qa_cymatic_mode_cert.schema.json` |
| Faraday cert schema | `qa_alphageometry_ptolemy/qa_cymatics/schemas/qa_faraday_reachability_cert.schema.json` |
| Control cert schema | `qa_alphageometry_ptolemy/qa_cymatics/schemas/qa_cymatic_control_cert.schema.json` |
| Planner cert schema | `qa_alphageometry_ptolemy/qa_cymatics/schemas/qa_cymatic_planner_cert.schema.json` |
| Fixtures (12) | `qa_alphageometry_ptolemy/qa_cymatics/fixtures/` |
| Mapping protocol ref | `qa_alphageometry_ptolemy/qa_cymatics/mapping_protocol_ref.json` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_cymatics
python qa_cymatics_validate.py --self-test    # JSON output for meta-validator
python qa_cymatics_validate.py --demo         # human-readable all fixtures
python qa_cymatics_validate.py --control fixtures/control_cert_pass_hexagon.json
python qa_cymatics_validate.py --control fixtures/control_cert_fail_illegal_transition.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_pass_shortest_hexagon.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_fail_no_plan_within_bound.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_pass_replay_hexagon.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_fail_replay_hash_mismatch.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_pass_minimality_hexagon.json
python qa_cymatics_validate.py --planner fixtures/planner_cert_fail_minimality_witness_incomplete.json
```

## Failure algebra

| fail_type | Cert family | Trigger |
|-----------|-------------|---------|
| OFF_RESONANCE | MODE | Drive frequency too far from eigenfrequency |
| BOUNDARY_MISMATCH | MODE / CONTROL | Symmetry group impossible for (m,n) or boundary |
| MODE_MIXING | MODE / FARADAY / CONTROL | Multiple modes coexist; state not unique |
| DAMPING_COLLAPSE | MODE / FARADAY / CONTROL | Amplitude below threshold |
| MEASUREMENT_ALIAS | MODE | Pattern artifact from capture method |
| TUPLE_FORMULA_VIOLATION | MODE | d ≠ b+e or a ≠ b+2e |
| ORBIT_CLASS_MISMATCH | MODE / CONTROL | orbit_family inconsistent with Q(√5) norm |
| NONLINEAR_ESCAPE | FARADAY / CONTROL | System in disordered/turbulent state |
| ILLEGAL_TRANSITION | FARADAY / CONTROL | Control move not in legal edge set |
| RETURN_PATH_NOT_FOUND | FARADAY / CONTROL | return_in_k=true but no valid return path |
| GOAL_NOT_REACHED | CONTROL | Final pattern ≠ target pattern |
| PATH_LENGTH_EXCEEDED | CONTROL | path_length_k > max_path_length_k |
| NO_PLAN_WITHIN_BOUND | PLANNER | BFS exhausted all paths within max_depth_k without reaching target |
| GOAL_NOT_REACHABLE | PLANNER | Target not reachable from initial under the declared bound |
| SEARCH_INCONSISTENCY | PLANNER | Algorithm unrecognized, or frontier stats inconsistent with plan |
| NONMINIMAL_PLAN | PLANNER | optimization_goal=shortest_path but plan is longer than BFS recomputed shortest |
| PLAN_CONTROL_MISMATCH | PLANNER | compiled control cert target/initial/final/moves inconsistent with planner plan |
| REPLAY_INCONSISTENCY | PLANNER | compiled_control_witness_hash does not match actual hash of referenced cert |
| COMPILED_CERT_MISSING | PLANNER | compiled_control_certificate_id not found in fixtures/ |
| MINIMALITY_WITNESS_INCOMPLETE | PLANNER | minimality_witness present but proved_no_path_shorter_than, excluded_shorter_lengths, or frontier_sizes inconsistent with path_length_k |

## Changelog

- **v1.3** (2026-03-21): Added minimality_witness as first-class cert field (proved_no_path_shorter_than + frontier_sizes + excluded_shorter_lengths); P11 validator check; MINIMALITY_WITNESS_INCOMPLETE fail type; 12 fixtures total.
- **v1.2** (2026-03-21): Added planner replay fields (compiled_control_certificate_id, compiled_control_witness_hash, replay_consistent); 2 new replay fixtures (pass + fail); 10 fixtures total; PLAN_CONTROL_MISMATCH/REPLAY_INCONSISTENCY/COMPILED_CERT_MISSING fail types.
- **v1.1** (2026-03-21): Added QA_CYMATIC_PLANNER_CERT.v1 (4th tier); 2 new fixtures; 4 schemas; 8 fixtures total; self-test 8/8.
- **v1.0** (2026-03-21): Initial 3-cert emission; 6 fixtures; all validators pass.
