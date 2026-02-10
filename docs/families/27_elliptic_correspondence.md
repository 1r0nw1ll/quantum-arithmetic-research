# [27] QA Elliptic Correspondence Bundle

## What this is

Certifies deterministic replay for the polynomial-ellipse correspondence induced by
`v^2 = u^3 + u` with the driver `v -> v^2 + v`.
The family models the six legal branch generators (2 sheet choices x 3 cubic-root choices)
and records either a success witness (invariant-preserving replay) or an obstruction witness
(e.g., ramification hit, cut crossing, escape).

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__ELLIPTIC_CORRESPONDENCE.yaml` |
| Certificate module | `qa_elliptic_correspondence_certificate.py` |
| Validator | `qa_elliptic_correspondence_validator_v3.py` |
| Bundle emitter/validator | `qa_elliptic_correspondence_bundle_v1.py` |
| Reference cert | `certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json` |
| Cert schema | `schemas/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.schema.json` |
| Bundle schema | `schemas/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.schema.json` |
| Success example | `examples/elliptic_correspondence/elliptic_correspondence_success.json` |
| Failure example | `examples/elliptic_correspondence/elliptic_correspondence_ramification_failure.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate demo certificates (success + failure)
python qa_elliptic_correspondence_validator_v3.py --demo

# Validate explicit certificates
python qa_elliptic_correspondence_validator_v3.py examples/elliptic_correspondence/elliptic_correspondence_success.json
python qa_elliptic_correspondence_validator_v3.py examples/elliptic_correspondence/elliptic_correspondence_ramification_failure.json

# Emit + check bundle
python qa_elliptic_correspondence_bundle_v1.py --emit --check

# Or via meta-validator (runs as family [27])
python qa_meta_validator.py
```

## Semantics

- **Generator set**: `{g_plus_r0, g_plus_r1, g_plus_r2, g_minus_r0, g_minus_r1, g_minus_r2}`
- **Branching contract**: `branching_factor_declared == len(generator_set) == 6`
- **Replay contract**: `trace_digest == sha256(canonical_json(transition_trace))`
- **Determinism contract**: repeated `(u_in, sheet_in, branch_in, generator)` must replay identical outcomes
- **Hard invariants**: `curve_constraint`, `determinism`, `cut_consistency`, `trace_complete`

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `NONFINITE_INPUT` | non-finite state/polynomial input | sanitize seed and numeric guards |
| `SQRT_BRANCH_UNDEFINED` | sqrt branch cannot be resolved under policy | tighten branch convention |
| `SQRT_CUT_CROSS_DISALLOWED` | forbidden cut crossing occurred | adjust generator or cut policy |
| `CUBIC_SOLVE_FAILED` | cubic inverse root solve failed | enforce deterministic root solver constraints |
| `RAMIFICATION_HIT` | entered ramification neighborhood | exclude critical neighborhoods or reroute generator path |
| `MULTIROOT_DEGENERATE` | cubic roots collided | increase precision or detect degeneracy earlier |
| `CUTSTATE_UPDATE_FAILED` | cut bookkeeping became ambiguous | harden cut-state transition logic |
| `MONODROMY_EVENT_DETECTED` | unexpected branch permutation event | tighten continuation policy |
| `ESCAPE` | state exceeded escape bound | lower step budget or enforce bounded control |
| `MAX_ITER_REACHED` | budget exhausted without decision | increase budget or emit obstruction certificate |

## Changelog

- **v1.0.0** (2026-02-10): Initial elliptic correspondence cert/validator/bundle + meta-validator gate.
