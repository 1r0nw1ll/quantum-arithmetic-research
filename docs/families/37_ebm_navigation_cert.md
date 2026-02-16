# [37] QA EBM Navigation Cert (QA_EBM_NAVIGATION_CERT.v1)

This family hardens “energy-based reasoning / energy landscape navigation” into QA’s certified language:

- `S`: state manifold (explicit)
- `G`: generator set (explicit)
- `I`: invariants (enforced via invariant_diff + typed failures)
- `F`: failure taxonomy (failure-as-theorem)
- `R`: reachability witness (trace)
- `C`: determinism contract (total tie-break, exact scalars)

## Machine tract

Directory: `qa_ebm_navigation_cert/`

Files:

- `qa_ebm_navigation_cert/schema.json`
- `qa_ebm_navigation_cert/validator.py`
- `qa_ebm_navigation_cert/fixtures/`
- `qa_ebm_navigation_cert/mapping_protocol_ref.json` (Gate 0 intake)

### What it validates (gates)

- **Gate 1 — Schema validity**: draft-07 schema, no extra fields
- **Gate 2 — Canonical hash**: `digests.canonical_sha256` matches canonical compact JSON hash
- **Gate 3 — Exact energy + invariant_diff**: no floats; `delta_energy` and `delta_violations` must match
- **Gate 4 — Deterministic tie-break**: choose min-energy legal move, tie-break by lex `(generator, state_after)`
- **Gate 5 — Failure completeness**: typed failures require non-empty witnesses
- **Gate 6 — Verifier acceptance (optional)**: if `accepted_by_verifier=true`, require `verifier_bridge_ref` and
  digest-link it via `digests.refs`, enforce transitive validity by running the referenced bridge cert’s validator,
  and require the bridge cert to attest the navigation terminal `state_after` and `outcome.target_ref`

### Run

```bash
python qa_ebm_navigation_cert/validator.py --self-test
python qa_ebm_navigation_cert/validator.py qa_ebm_navigation_cert/fixtures/valid_min.json
```

## Human tract

Concept mapping (EBM reasoning / Kona podcast → QA):

- `Documents/QA_MAP__EBM_REASONING_KONA_PODCAST.md`
- `Documents/QA_MAPPING_PROTOCOL__EBM_REASONING_KONA_PODCAST.v1.json`

## Notes

This family intentionally enforces:

- **Energy is advisory** (navigation heuristic).
- **Invariants/verifiers are authoritative** (accept/reject).
- **Exact arithmetic discipline** at the witness layer (no float creep).

For verifier-gated acceptance, pair this with:

- `qa_ebm_verifier_bridge_cert/` (`QA_EBM_VERIFIER_BRIDGE_CERT.v1`)
