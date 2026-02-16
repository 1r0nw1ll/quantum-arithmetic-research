# [38] QA Energy–Capability Separation Cert (QA_ENERGY_CAPABILITY_SEPARATION_CERT.v1)

This family formalizes the hygiene theorem:

> `Energy ≠ Capability`  
> `Capability = Reachability(S, G, I)`  
> Energy policies are only *orderings* over legal successors; they do not expand generator closure.

It provides a **constructive separation witness** in a finite `Caps(N,N)` QA subuniverse:

- a target is reachable under the generator set (explicit witness path)
- a deterministic min-energy-legal policy fails to reach that target within the policy budget

## Machine tract

Directory: `qa_energy_capability_separation_cert/`

Files:

- `qa_energy_capability_separation_cert/schema.json`
- `qa_energy_capability_separation_cert/validator.py`
- `qa_energy_capability_separation_cert/fixtures/`
- `qa_energy_capability_separation_cert/mapping_protocol_ref.json` (Gate 0 intake)

### Gates (high-level)

- **Gate 1 — Schema validity**
- **Gate 2 — Canonical hash** (`digests.canonical_sha256`)
- **Gate 3 — Reachability witness legality** (path is legal in `Caps(N,N)` and reaches the target)
- **Gate 4 — Policy replay** (min-energy-legal + deterministic tie-break; exact energy deltas)
- **Gate 5 — Separation claim coherence** (reachable under generators, not reached by policy)
- **Gate 6 — invariant_diff coherence** (summary fields consistent)

### Run

```bash
python qa_energy_capability_separation_cert/validator.py --self-test
python qa_energy_capability_separation_cert/validator.py qa_energy_capability_separation_cert/fixtures/valid_min.json
```

## Notes

- Energy is treated as an **exact** scalar here (`E(b,e)=e`), to prevent float drift.
- The witness uses canonical QA generators `{sigma, mu}` as defined in `qa_canonical.md`.

