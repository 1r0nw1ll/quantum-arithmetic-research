# [19] Topology Resonance Bundle

## What this is

Certifies generator-induced topological reachability in the QA state space. The canonical generator set `{sigma, mu, lambda2, nu}` produces SCC (strongly connected component) structures whose growth and phase coherence are certified as bundle artifacts. This is the geometric foundation — it proves the QA state space has non-trivial topology under its generators.

## Artifacts

| Artifact | Path |
|----------|------|
| Module spec (YAML) | `QA_MAP__TOPOLOGY_RESONANCE.yaml` |
| Validator | `qa_topology_resonance_validator_v1.py` |
| Bundle emitter/validator | `qa_topology_resonance_bundle_v1.py` |
| Reference cert | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.json` |
| Cert hash sidecar | `certs/QA_TOPOLOGY_RESONANCE_CERT.v1.sha256` |
| Bundle manifest | `certs/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.json` |
| Cert schema | `schemas/QA_TOPOLOGY_RESONANCE_CERT.v1.schema.json` |
| Bundle schema | `schemas/QA_TOPOLOGY_RESONANCE_BUNDLE.v1.schema.json` |
| Success example | `examples/topology/topology_resonance_success.json` |
| Failure example | `examples/topology/topology_phase_break_failure.json` |

All paths relative to `qa_alphageometry_ptolemy/`.

## How to run

```bash
cd qa_alphageometry_ptolemy

# Validate individual cert
python qa_topology_resonance_validator_v1.py --demo

# Validate with recompute level
python qa_topology_resonance_validator_v1.py --level recompute examples/topology/topology_resonance_success.json

# Emit + check bundle
python qa_topology_resonance_bundle_v1.py --emit --check

# Or via meta-validator (runs as test [19])
python qa_meta_validator.py
```

## Semantics

- **Generator set**: `{sigma, mu, lambda2, nu}` — the canonical QA topology generators
- **SCC monotonicity**: `scc_count_after >= scc_count_before` — connectivity never decreases
- **Phase lock**: `phase_24` and `phase_9` preserved across legal transitions
- **Resonance threshold**: `resonance_score >= resonance_threshold` for certification
- **Hard invariants**: `packet_conservation`, `no_reduction_axiom`, `connected_component_first_class`

## Failure modes

| fail_type | Meaning | Fix |
|-----------|---------|-----|
| `phase_break` | Phase-24 or phase-9 changed across transition | Fix generator to preserve phase |
| `scc_drop` | SCC count decreased | Illegal generator — review transition |
| `resonance_below_threshold` | Score below declared threshold | Adjust generator set or threshold |
| `packet_drift` | Packet conservation violated | Fix packet handling |
| `invalid_generator` | Generator not in `{sigma, mu, lambda2, nu}` | Use only canonical generators |

## Example

**Passing** — resonance certified:
```json
{
  "generator_set": ["sigma", "mu"],
  "topology_witness": {
    "scc_count_before": 3,
    "scc_count_after": 4,
    "resonance_score": "7/10",
    "resonance_threshold": "1/2",
    "resonance_certified": true
  },
  "invariants": {
    "scc_monotone_non_decreasing": true,
    "packet_conservation": true
  }
}
```

**Failing** — phase break:
```json
{
  "success": false,
  "failure_mode": "phase_break",
  "failure_witness": {
    "phase_24_before": 5,
    "phase_24_after": 7,
    "phase_9_before": 2,
    "phase_9_after": 2
  }
}
```

## Changelog

- **v1.0.0** (2026-02-07): First cert emission + bundle manifest with validator.
- **2026-07-06**: See Verification Note below — same class of gap as
  cert [27], plus a required honest `generator_grounding` caveat added.

## Verification Note (2026-07-06)

Follow-up to auditing sibling cert [27] (Elliptic Correspondence), which
had a fully fabricated demo trace undetected by two stacked gaps
(meta-validator only checked bundle file-hash integrity; the validator's
own "Level 3 recompute" was a stub). Checked whether family [19] has the
same gaps.

**Gap 1 (same as [27], confirmed and fixed)**: the meta-validator's
family [19] wrapper (`_validate_topology_bundle_if_present`) only ran
the bundle-manifest file-hash integrity check, never invoking
`qa_topology_resonance_validator_v1.py`'s real Level 1-3 validator on
certificate content. Fixed: now also runs the real validator against the
reference success/failure certs after the hash check passes.

**Gap 2 (different from [27], more fundamental, partially open)**:
unlike [27], this validator's own Level 3 "recompute" check is NOT a
stub — it genuinely re-derives `scc_count`/`phase_24`/`phase_9`/
`resonance_score` from the trace and cross-checks them against the
declared before/after witness fields (confirmed correct: `--demo` passes
all checks). **However**, the canonical generators `{sigma, mu, lambda2,
nu}` referenced throughout this cert family are never implemented
anywhere in the codebase as concrete operations on a real state space —
they are bare string labels checked only for set membership
(`VALID_GENERATORS = {"sigma", "mu", "lambda2", "nu"}`). `QA_MAP__TOPOLOGY_RESONANCE.yaml`'s
`source_paper` field is a vague "QA topology reachability notes (Caps(N,N)
program)" with no author/year/DOI, unlike essentially every other cert
in this repo. This means the existing recompute check can only ever
verify **internal trace self-consistency** (declared per-step deltas sum
to the declared endpoints) — it cannot and does not verify that the
declared numbers are a genuine consequence of applying a real generator
operation, because no such operation exists in code to check against.

This is architecturally different from [27]: [27] had a well-defined
target (the curve equation) that the data simply failed to satisfy;
[19]'s target (what `sigma`/`mu`/`lambda2`/`nu` concretely compute) was
never specified in the first place, so there is nothing to independently
recompute against beyond the trace's own internal arithmetic. Inventing
generator semantics now would be fabricating new mathematical content
under the guise of an audit fix, not verifying existing content — out of
scope for this pass.

**Fix applied instead**: added a required `generator_grounding` field
(`status`: `"abstract_symbolic"` or `"concrete_implementation"`, plus a
non-empty `caveat` string when abstract) to the schema and a new
`consistency.generator_grounding` validator check, so this cert can no
longer be mistaken for grounded the way [27] now is. Verified the check
rejects a missing/malformed grounding block. Updated the reference cert,
example fixture, sha256 sidecar, and bundle manifest accordingly.

**Not fixed / genuinely open**: defining concrete semantics for
`sigma`/`mu`/`lambda2`/`nu` (so their effect on SCC/phase/resonance could
be independently recomputed rather than merely self-consistency-checked)
would be new mathematical design work, not an audit fix. Left as an
honestly-flagged gap via the new `generator_grounding` caveat rather than
silently passing as if grounded.

**Follow-up**: family [28] Graph Structure bundle uses the identical
generic bundle-manifest pattern and was flagged in [27]'s audit as a
likely-same-gap candidate — not yet checked.
