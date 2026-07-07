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
- **2026-07-06**: See Verification Note below — the shipped demo trace was
  mathematically fabricated and the validation pipeline never caught it;
  both are now fixed.

## Verification Note (2026-07-06)

**Major finding**: independently recomputed the shipped "success" demo
certificate's transition trace against the actual curve dynamics
(`v_in = sqrt_branch(u_in^3+u_in, sheet_in)`, driver `v_out = v_in^2+v_in`,
require `v_out^2 == u_out^3+u_out`). Step 1 (`u_in=1/2 -> u_out=2/5+i/10`
via `g_plus_r0`) gave `|v_out^2 - (u_out^3+u_out)| ≈ 1.5588`, wildly
nonzero, while the trace declared `curve_residual_abs: "0"`. The
certificate's numeric data did not satisfy the elliptic correspondence
it claims to certify at all — it was fabricated/hand-invented, not
computed.

**Root cause this went undetected**: the meta-validator's family [27]
check (`_validate_elliptic_bundle_if_present`) only ran
`qa_elliptic_correspondence_bundle_v1.py`'s bundle-manifest check, which
verifies file-hash *integrity* (that artifact files haven't changed
since the bundle was last emitted) — it never invoked
`qa_elliptic_correspondence_validator_v3.py`'s actual Level 1/2/3
validation on the certificate content at all. Separately,
`validator_v3.py`'s own "Level 3 (Recompute)" check
(`validate_recompute`) was pure validation theater: despite the
docstring calling it "deterministic trace replay," the code only checked
that `transition_trace` was a non-empty list — it never recomputed a
single step against the curve equation. This is a more severe instance
of the pattern found across the 2026-07 audit cycle (e.g. [184]-[188]
Keely family): here the *entire* mathematical claim of the cert was
unverified by any check in the pipeline, not just one sub-check.

**Fixes applied**:
1. `validator_v3.py`: `validate_recompute` now genuinely recomputes
   `v_in`, applies the driver, and checks `|v_out^2-(u_out^3+u_out)| <
   1e-6` per step, plus cross-checks the declared `curve_residual_abs`
   against the real recomputed residual (`recompute.step.N.curve_replay`,
   `recompute.step.N.residual_honesty`). Verified it rejects the
   original fabricated trace (residual 1.559 reported, correctly FAILED).
2. `qa_meta_validator.py`'s `_validate_elliptic_bundle_if_present` now
   also runs the real validator against the reference success/failure
   certs after the bundle-hash check passes, so family [27] can no
   longer pass on hash-integrity alone.
3. `qa_elliptic_correspondence_certificate.py`: replaced the fabricated
   4-step trace with a genuinely computed 2-step trace (`u_in=1/2`,
   generator `g_plus_r0` both steps), verified to residual ~1e-14/1e-15
   (float64 precision), staying within the declared `radius_bound_u/v=4`
   (max |u|≈2.347, max |v|≈3.419, no escape).
4. `schemas/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.schema.json`: the
   `scalar` definition only accepted exact integer/fraction strings,
   which is fundamentally incompatible with this correspondence's real
   dynamics — square/cube roots of a non-degenerate rational are
   irrational in general, so genuine trace values can essentially never
   come out as clean small fractions (this is likely *why* the original
   demo data was hand-invented rather than computed: the schema made
   honest data impossible to express). Extended `scalar` to also accept
   decimal and scientific-notation strings.
5. Regenerated `certs/QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.json`, its
   `.sha256` sidecar, `examples/elliptic_correspondence/
   elliptic_correspondence_success.json`, and re-emitted
   `certs/QA_ELLIPTIC_CORRESPONDENCE_BUNDLE.v1.json` so all hashes are
   consistent with the fixed data.

**Not fixed / follow-up noted**: the same "bundle manifest checks file
hashes only, never invokes the real content validator" pattern likely
exists in sibling bundle families [19] (Topology Resonance) and [28]
(Graph Structure) — flagged for a future audit pass, not fixed here to
keep this change scoped to family [27].
