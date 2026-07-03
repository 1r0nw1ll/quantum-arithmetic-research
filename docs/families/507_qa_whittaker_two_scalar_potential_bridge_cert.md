# [507] QA Whittaker Two-Scalar-Potential Bridge Certificate

**Schema**: `QA_WHITTAKER_TWO_SCALAR_POTENTIAL_BRIDGE_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_whittaker_two_scalar_potential_bridge_cert_v1/`
**Status**: PASS
**Added**: 2026-07-02
**Primary source**: E. T. Whittaker (1904), "On an expression of the electromagnetic field due to electrons by means of two scalar potential functions," Proc. London Math. Soc. s2-1:367-372, DOI 10.1112/plms/s2-1.1.367. On disk: `Documents/whittaker_corpus/whittaker_1904_two_scalar_potential_functions_electromagnetic_field.pdf`.

## Layer position

Layer 4 of the Whittaker → QA development ladder
(`docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md` §8), following
registered Layer 1 [266], Layer 2 [273], Layer 3 [274], and side-branch [498].

## Claim

Whittaker (1904) showed the six components of the electromagnetic field
(dielectric displacement `dx,dy,dz` and magnetic force `hx,hy,hz`) reduce to
derivatives of two scalar potentials, which the paper calls `F` and `G`. This
cert renames them `Phi, Psi` because `F = a·b = d²−e²` and `G = d²+e²` are
already QA-reserved chromogeometric invariants.

For a QA-rational plane-wave packet built from a registered [273] `S²`
direction `ω` and declared QA-rational `k, v, c`, the twelve raw
differential-operator coefficients of Whittaker's verbatim operator
`(Phi,Psi) → (dx,dy,dz,hx,hy,hz)` are exact `fractions.Fraction` values, and:

- `div(h)` (both channels) and `div(d)` `Psi`-channel vanish exactly and
  **unconditionally**.
- `div(d)` `Phi`-channel vanishes exactly when the packet satisfies the
  vacuum dispersion relation `v² = c²`, or when the direction is degenerate
  with `Kz=0`. For nonzero-`Kz` packets, dispersion is necessary and
  sufficient.

## A primary-source correction caught during construction

Before this cert was built, an AI-generated summary of Whittaker's paper
(supplied by the user, sourced from a different assistant) was checked
against the actual primary-source PDF. Five of six field equations matched
verbatim. The sixth (`hz`) did not: the summary guessed
`hz = ∂²Ψ/∂z² − (1/c²)∂²Ψ/∂t²` by false symmetry with the `dz` formula.
Working the exact-Fraction divergence algebra by hand (before opening the
PDF) already showed this guess breaks `div(h)=0` identically. Reading the
primary source confirmed the real formula is `hz = ∂²Ψ/∂x² + ∂²Ψ/∂y²` — a
genuinely different, non-obvious result of Whittaker's derivation (from an
auxiliary function `ψ` that "disappears automatically," per the paper's
§3) — and it is exactly the form that makes `div(h)=0` unconditional. The
corrected formula is what is implemented in the validator.

## Non-Claims

Does not prove Maxwell's equations (external definition, not derived), does
not claim QA derives electromagnetism, does not reconstruct any physical
field, does not claim Mie scattering, and does not claim scalar-wave-energy
physics. Does not claim Layer 5 (field reconstruction with explicit
discretization error) or Layer 6 (spherical/Bromwich/Mie mode indexing) —
see the ladder draft spec for those layers' mapped-but-unbuilt sharp-claim
forms.

## Cross-references

- [266] `qa_whittaker_rational_direction_s1_cert_v1` — Layer 1, `S¹` directions
- [273] `qa_whittaker_rational_direction_s2_cert_v1` — Layer 2, `S²` directions (hard dependency)
- [274] `qa_whittaker_scalar_angular_kernel_sampling_cert_v1` — Layer 3, kernel sampling
- [498] `qa_whittaker_phase_packet_algebra_cert_v1` — phase-packet algebra (lineage context)
- [497] `qa_steinmetz_whittaker_bridge_cert_v1` — separate Whittaker-1903/Steinmetz hysteresis bridge
- `docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md` §8 — full six-layer ladder, including Layer 5/6 mapping notes added alongside this cert

## Validator

`qa_alphageometry_ptolemy/qa_whittaker_two_scalar_potential_bridge_cert_v1/qa_whittaker_two_scalar_potential_bridge_cert_validate.py --self-test`
