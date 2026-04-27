<!-- PRIMARY-SOURCE-EXEMPT: reason="Algorithm-database qa_mapping.md template (catalog field structure). Source citations + cert evidence are filled in per-entry. (Kochenderfer, 2026; Dale, 2026)." -->

# QA Mapping — `<algorithm_slug>`

## Status

`<established | candidate | open | rejected>`

## QA counterpart

- **Concrete artifact**: `<cert family / utility / bridge row / none>`
- **If `established`**: cite the cert that PASSes with this algorithm as its claim. Format: `cert [<NNN>] <cert_slug>` with link to `qa_alphageometry_ptolemy/<cert_slug>_cert_v1/`.
- **If `candidate`**: cite the existing artifact (cert / utility / spec) that already implements the same idea, plus the docs change needed to align vocabularies.
- **If `open`**: explain what a future cert would claim. Mirror the bridge spec's "Future cert claim" column. Add the candidate to `docs/specs/QA_KOCHENDERFER_BRIDGE.md` Standing Rule #2 if not already there.
- **If `rejected`**: state which Theorem NT firewall constraint blocks the mapping (e.g., "continuous-domain method; firewall-rejected as causal QA dynamics; observer-projection candidate at input boundary only").

## Evidence

For `established` entries only: cite the cert's evidence (e.g., "cert [263] `qa_failure_density_enumeration_cert_v1` reproduces cert [194]'s `1/81, 8/81, 72/81` ratios bit-exact via `tools/qa_kg/orbit_failure_enumeration.py` utility; Kochenderfer Algorithm 7.1 sampling at N ∈ {100, 1000, 10000} × seeds {42, 1337, 2024} (18 cases) lands inside `4σ` envelope per Kochenderfer eq. 7.3").

For `candidate`/`open`/`rejected`: this section is empty or omitted. The status field already conveys "no evidence yet."

## Theorem NT boundary note

Required field. Document:
- Which side of the Theorem NT firewall the classical algorithm lives on (continuous-domain / discrete-domain / hybrid)
- Which side the QA counterpart lives on (always discrete on the QA-causal side)
- How the boundary is crossed (input projection / output projection / no crossing required)
- Any input-noise / observer-projection considerations (mirror cert [264] `qa_runtime_odd_monitor` framing for ODD-style entries)

## Bridge spec row

Pointer: `docs/specs/QA_KOCHENDERFER_BRIDGE.md` §`<N.M>`. The bridge row is the canonical mapping documentation; this `qa_mapping.md` is the per-entry shorthand that points at it.

## Notes (free-form)

Anything else relevant: scope limitations, known gotchas, related entries, future work.
