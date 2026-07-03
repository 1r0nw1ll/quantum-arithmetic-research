# QA Whittaker Two-Scalar-Potential Bridge Cert v1

Candidate family ID: `[507]`

Layer 4 of the Whittaker -> QA development ladder
(`docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md` sec. 8).

## Purpose

Primary source: E. T. Whittaker (1904), "On an expression of the
electromagnetic field due to electrons by means of two scalar potential
functions," Proc. London Math. Soc. s2-1:367-372. DOI: 10.1112/plms/s2-1.1.367.
On disk: `Documents/whittaker_corpus/whittaker_1904_two_scalar_potential_functions_electromagnetic_field.pdf`.

Whittaker showed the six components of the electromagnetic field (dielectric
displacement `dx,dy,dz` and magnetic force `hx,hy,hz`) reduce to derivatives
of just two scalar potentials, which the paper itself calls `F` and `G`.

**Naming guardrail.** This cert renames Whittaker's potentials `Phi, Psi`
throughout. QA already reserves the letters `F = a*b = d^2-e^2` and
`G = d^2+e^2` for the chromogeometric triple; reusing Whittaker's letters
verbatim would silently collide with those.

## Claim (narrow)

For a QA-rational plane-wave packet built from a registered `[273]` `S^2`
direction `omega` and declared QA-rational `k, v, c`, the twelve raw
differential-operator coefficients of Whittaker's 1904 map
`(Phi,Psi) -> (dx,dy,dz,hx,hy,hz)` — reproduced verbatim from the primary
source (p.370, sec.3), **not** reconstructed by symmetry-guessing — are exact
`fractions.Fraction` values, and:

- `div(h)` (both `Phi`- and `Psi`-channel) vanishes exactly and
  **unconditionally**.
- `div(d)` `Psi`-channel vanishes exactly and **unconditionally**.
- `div(d)` `Phi`-channel vanishes exactly when the packet independently
  satisfies the vacuum dispersion relation `v*v = c*c`, or when the direction
  is degenerate with `Kz = 0`. For nonzero-`Kz` packets, dispersion is
  necessary and sufficient.

This is a linear-coefficient-algebra / exact-identity claim about
Whittaker's own published operators, instantiated at QA-rational directions.
It does **not** prove Maxwell's equations (they are the external definition
being checked against), does **not** claim QA derives electromagnetism, and
does **not** reconstruct any physical field.

## A primary-source correction caught during construction

An early hand-derived draft of this operator (written from a secondary
AI-generated summary, before the primary-source PDF was consulted) guessed
`hz = d^2(Psi)/dz^2 - (1/c^2) d^2(Psi)/dt^2` by false symmetry with the `dz`
formula. Working the exact-Fraction algebra showed that guess breaks
`div(h) = 0` identically. Reading the actual 1904 paper (p.370) showed the
real formula is `hz = d^2(Psi)/dx^2 + d^2(Psi)/dy^2` — a genuinely different,
non-obvious result of Whittaker's derivation, not a symmetry image of `dz`.
That is the formula implemented here, and it is exactly what makes
`div(h)=0` unconditional. This is recorded as a concrete instance of
"primary sources over consensus."

## Dependencies

- Hard dependency: registered `[273]` `qa_whittaker_rational_direction_s2_cert_v1`.
- Lineage context: registered `[498]` `qa_whittaker_phase_packet_algebra_cert_v1`
  (not a hard dependency in v1 — this cert does not reuse its phase_arg
  machinery directly, only its plane-wave-packet spirit).

## Gates

| Gate | Check |
| --- | --- |
| `WSPB_1` | dependency provenance: `[273]` present/registered; `omega_packet` is an actual member of `[273]`'s `D_m^(2)` |
| `WSPB_2` | packet declarations complete (`k`, `v`, `c`, `wave_equation_satisfied` present and well-typed) |
| `WSPB_3` | all twelve raw coefficients recomputed exactly from Whittaker's verbatim operator and checked against declared witnesses |
| `WSPB_4` | declared `wave_equation_satisfied` must equal the independently recomputed `v*v == c*c` |
| `WSPB_5` | divergence identities recomputed exactly; three vanish unconditionally; `div(d)_Phi` vanishes under dispersion or `Kz=0`, and is nonzero for nonzero-`Kz` packets without dispersion |
| `WSPB_6` | source attribution must cite Whittaker 1904, the DOI, and dependency `[273]` |
| `WSPB_7` | rejects trig evaluation, float pass/fail, numerical approximation, fitted coefficients |
| `WSPB_8` | rejects Maxwell-derivation / electromagnetism / physical-field / scalar-wave-energy / Mie-scattering / Layer-5-or-beyond overclaims |

## Fixtures

PASS:

- `fixtures/pass_wspb_z0_no_waveeq_m3.json` — `omega=(3,4,0)/5`, dispersion not satisfied, demonstrates the three unconditional identities with a zero `Kz`.
- `fixtures/pass_wspb_nonzero_z_waveeq_m3.json` — `omega=(175,500,88)/537`, dispersion satisfied (`v=c=1`), exercises the conditional `div(d)_Phi=0` identity with nonzero `Kz`.

FAIL:

- `fixtures/fail_wspb_false_wave_eq_claim.json` — same nonzero-`Kz` direction with `v=1,c=2` (dispersion **not** satisfied) but falsely claims `wave_equation_satisfied=true`; shows `div(d)_Phi` is genuinely nonzero (`-22/179`) in that case, i.e. the conditional gate is not vacuous.
- `fixtures/fail_wspb_nonzero_z_no_waveeq_zero_div_claim.json` — same nonzero-`Kz`, non-dispersion packet falsely declares `div(d)_Phi=0`; enforces the necessary-and-sufficient branch for nonzero `Kz`.
- `fixtures/fail_wspb_bad_coefficient_witness.json`
- `fixtures/fail_wspb_missing_source_doi.json`
- `fixtures/fail_wspb_overclaimed_maxwell_derivation.json`
- `fixtures/fail_wspb_float_trig_evaluation_claimed.json`
- `fixtures/fail_wspb_omega_not_in_dependency.json`

## Non-Claims

This cert does not claim numerical approximation, trigonometric evaluation,
Maxwell/EM derivation, electromagnetism, physical field reconstruction, Mie
scattering, or scalar-wave-energy physics. It does not claim Layer 5 (field
reconstruction / discretization error bounds) or Layer 6 (spherical /
Bromwich / Mie) results — see
`docs/specs/QA_WHITTAKER_RATIONAL_DIRECTION_CERT_DRAFT.md` sec. 8 for the
mapped-but-unbuilt sharp-claim forms of those layers.

## Run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_two_scalar_potential_bridge_cert_v1/qa_whittaker_two_scalar_potential_bridge_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`.
