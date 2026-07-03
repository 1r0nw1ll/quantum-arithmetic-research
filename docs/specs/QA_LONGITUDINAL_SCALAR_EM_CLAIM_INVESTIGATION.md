# QA Longitudinal / Scalar EM Claim Investigation

Status: source-audit and claim triage, not a registered cert.

Date: 2026-07-03

## Purpose

This document investigates the scalar/longitudinal EM claims that `[514]`
bounded but did not globally disprove:

- source-free scalar or longitudinal vacuum radiation modes;
- scalar/free-energy or over-unity claims;
- Bearden/Pond/SVP scalar-energy claims;
- scalar-potential physics;
- Dollard/Steinmetz/Whittaker/QA bridge claims.

The result is not "proved" or "disproved" as a blanket verdict. The result is
a claim map: which parts are already certified, which parts have a plausible
QA build path, which parts are blocked by missing evidence, and which specific
arguments are already refuted by source or algebra checks.

## Source Spine

Cert-grade local anchors:

- `[507]` `qa_whittaker_two_scalar_potential_bridge_cert_v1`: exact QA-rational
  coefficient algebra for Whittaker's 1904 two-scalar-potential operator.
- `[512]` `qa_inhomogeneous_maxwell_recovery_cert_v1`: exact finite recovery of
  `delta(starF) = J` under declared Hodge, source, sign, unit, and projection
  conventions.
- `[513]` `qa_maxwell_derivation_cert_v1`: bounded full-Maxwell assembly only
  within certified carrier, boundary, source, metric, unit, and observer
  projection conventions.
- `[514]` `qa_longitudinal_scalar_em_boundary_cert_v1`: representation boundary
  for scalar/longitudinal language; explicitly does not use Heaviside, Hertz,
  or Gibbs vector reductions as premises.
- `[24]` `qa_svp_cmc`: SVP-CMC scalar-first semantics and obstruction ledger.
- `[155]` `qa_bearden_phase_conjugate_cert_v1`: scaffold only; maps Bearden's
  "stress is a pumper" phrase to a QA/QCI structural analogy.
- `[197]` `qa_see_longitudinal_transverse_cert_v1`: historical/structural
  mapping of See's longitudinal/transverse duality to QA generator/observer
  duality.
- `[497]` `qa_steinmetz_whittaker_bridge_cert_v1`: calibrated-transform
  scaffold for hysteresis loop area vs QA curvature proxy; not a universal
  physical identity.

Primary / external anchors already used by the local certs:

- Maxwell, J.C. (1865), "A Dynamical Theory of the Electromagnetic Field,"
  Phil. Trans. R. Soc. 155:459-512, DOI 10.1098/rstl.1865.0008.
- Whittaker, E.T. (1904), "On an expression of the electromagnetic field due
  to electrons by means of two scalar potential functions," Proc. London Math.
  Soc. s2-1:367-372, DOI 10.1112/plms/s2-1.1.367.
- Steinmetz, C.P. (1892), "On the law of hysteresis," AIEE Trans. 9:3-64.
- Hatcher, A. (2002), Algebraic Topology, Ch. 2, ISBN 978-0-521-79540-1.
- Bossavit, A. (1998), Computational Electromagnetism,
  ISBN 978-0-12-118710-1.

Non-cert local notes used only as hypothesis sources:

- `field-structure-theory/03-integration-frameworks/whittaker-bearden-scalar/`
  notes on Dollard, Bearden, Whittaker, and Bruhn.
- `field-structure-theory/03-integration-frameworks/dollard-versor-whittaker-qa/DOLLARD_VERSOR_WHITTAKER_QA_SYNTHESIS.md`.
- `field-structure-theory/06-related-theories/bearden-scalar-em/bearden-scalar-em-comprehensive.txt`.
- `private/svp_propositions_qa_mapping.md` and
  `private/keely_40_laws_classification.md`.

These non-cert notes include imported assistant prose and must not be treated
as primary-source proof.

## Main Findings

1. Whittaker scalar-potential representation is real, source-anchored, and
   already has a narrow QA exact-algebra bridge in `[507]`.

2. `[507]` is not a Maxwell derivation and is not a scalar-energy proof. It
   imports Whittaker's EM operator from a physics primary source and certifies
   exact coefficient identities for QA-rational packets.

3. `[513]` now gives a bounded Maxwell assembly, but only inside its stated
   finite cochain carrier, Hodge/source, metric, unit, and observer projection
   conventions. That is not an unbounded claim that QA has derived all physical
   electromagnetism.

4. `[514]` correctly refuses to use Heaviside-Hertz-Gibbs vector packaging as
   the premise for the scalar/longitudinal question. It also correctly refuses
   to certify unsupported scalar-energy claims inside `[514]`. It does not
   globally disprove those claims.

5. The strongest negative result found locally is specific, not blanket:
   Bearden's "vector zero resultant" quaternion argument, as captured in the
   local Bruhn critique note, contains algebra/type errors and an energy
   bookkeeping failure for destructive interference. That refutes that
   argument as written. It does not refute every possible scalar-potential,
   longitudinal, or SVP claim.

6. The strongest positive research path is not "free energy." It is a bounded
   carrier/source/boundary investigation: scalar or longitudinal structure may
   be admissible as potential carrier, source component, boundary/media mode,
   gauge term, or observer projection, provided the future cert supplies exact
   source, Hodge/constitutive, boundary, unit, latency, and energy-bookkeeping
   evidence.

## Heaviside / Hertz / Gibbs Issue

The current bounded Maxwell program should not be described as "using
Heaviside's equations" as its foundation. Its cert-grade construction is in
finite exterior/cochain language:

```text
F = delta(A)
delta(F) = 0
starF = star_QA(F)
J = delta(starF)
delta(J) = 0
```

Observer-readable vector equations can be recovered only after projection and
sign/unit conventions are declared. That is different from taking Heaviside's
four-vectorized equations as axioms.

The "Hertz" objection is also respected: `[514]` forbids a transverse-only
Hertz premise for the scalar/longitudinal question. Longitudinal terms are not
ruled out by definition; they must appear as certified source, boundary, media,
gauge, carrier, or observer-projection structure.

The "Gibbs" label in `[514]` refers to vector-analysis packaging, not to using
Gibbs thermodynamics or statistical mechanics as the physical foundation.
Neither Gibbs vector notation nor thermodynamic randomness is allowed to decide
the scalar/longitudinal question inside QA logic.

## Claim Triage

| Claim | Current status | Why |
| --- | --- | --- |
| Whittaker two-scalar potentials represent EM field components | Supported as source-anchored representation; `[507]` certifies exact QA packet coefficient algebra | Whittaker 1904 is primary-source anchored; `[507]` implements the verbatim operator with corrected `hz` |
| QA derives Maxwell | Bounded only | `[513]` permits exactly: "QA derives Maxwell only within the stated carrier, boundary, source, metric, unit, and observer-projection conventions certified here." |
| QA cannot derive Maxwell | Not established | The Maxwell program explicitly rejects both "already derived unbounded Maxwell" and "cannot derive Maxwell" |
| Heaviside/Hertz/Gibbs reductions settle scalar EM | Rejected as premise | `[514]` requires scalar/longitudinal analysis without those reductions as foundations |
| Extra source-free vacuum scalar radiation mode | Not certified | Needs a QA carrier, source-free condition, Hodge/constitutive conventions, boundary conditions, and observer projection; `[514]` does not disprove it globally |
| Scalar potential equals physical field | Rejected inside current certs | Potential carriers are allowed, but field equivalence requires gauge/observable conventions and energy bookkeeping |
| Bearden/Pond/SVP scalar energy | Not certified; source-gated future work | `[155]` is only a scaffold; `[24]` requires scalar configuration, latency, boundary, neutral center, and no energy-as-cause errors |
| Bearden vector-zero-resultant free energy argument | Locally refuted as written | Bruhn critique note identifies quaternion product/type errors and destructive-interference energy redistribution |
| Dollard/Steinmetz/Whittaker bridge | Plausible calibrated-transform research path | `[497]` validates deterministic fixture consistency under fixed calibration constants; it does not prove a universal physical identity |
| Mie/Bromwich scalar or longitudinal scattering values | Not certified | Layer 6 map allows exact mode-lattice enumeration only; Bessel/Legendre/Mie values remain observer-side special functions |

## Source-Free Scalar Mode: What Would Count

A future positive claim cannot simply say "scalar waves exist" or "the scalar
potential radiates." It needs a specific witness:

- field carrier type: scalar, 1-cochain, 2-cochain, potential pair, or other
  exact finite object;
- source condition: exact statement of what "source-free" means in the QA
  complex, for example `J = 0` or an explicitly vanishing source cochain;
- Hodge/constitutive object: QA-native or declared observer boundary;
- boundary and media data: cavity, interface, material, or vacuum convention;
- gauge/projection convention: how potential data become observable data;
- energy bookkeeping: stored, dissipated, radiated, or redistributed energy,
  with no hidden source;
- latency/path-length evidence if the claim is framed through SVP-CMC.

Without those pieces, the claim remains unsupported, not disproven.

## Free-Energy / Over-Unity: Current Obstructions

Under SVP-CMC `[24]`, a free-energy claim is blocked unless it avoids these
specific obstruction classes:

- `ENERGY_AS_CAUSE`: energy output cannot be the cause; scalar configuration
  must be specified first.
- `INSTRUMENT_AS_SOURCE`: the instrument cannot be treated as the origin of the
  energy without source evidence.
- `TRANSMISSION_MODEL_FORBIDDEN`: SVP claims cannot be reduced to ordinary
  EM-like signal transmission as the primary cause.
- `NO_INSTANT_ACTION`: remote or nonlocal claims need response ordering and
  positive path-length/latency.
- `MISSING_BOUNDARY_CONDITIONS`: cavity, shell, interface, or target boundary
  conditions must be part of the scalar configuration.
- `MISSING_PHASE_COHERENCE`: claimed lock-in requires phase/coherence data.
- `KINETIC_AS_CAUSE`: heat, light, electrical output, or radiation are effects,
  not scalar-first causes.

This does not say "free energy is impossible." It says the current repo has no
cert-grade witness that survives these gates.

## Bearden / Pond / SVP Claims

The Bearden/Pond/SVP lane should be split into three claim classes:

1. Vocabulary and structural analogy.
   `[155]` lives here. It maps "stress is a pumper" to an empirical QCI
   opposite-sign pattern. It is a scaffold, not a physical scalar-energy cert.

2. Scalar-first causal semantics.
   `[24]` lives here. It supplies a useful QA/SVP gate: scalar configuration,
   boundary, phase, polarity, neutral center, and latency must be explicit.

3. Physical energy claim.
   No current cert lives here. A future cert would need primary Bearden/Pond
   passages, exact claim normalization, experimental or algebraic witnesses,
   and full energy/source bookkeeping.

Do not collapse these three classes. A structural analogy passing `[155]` does
not certify physical energy extraction.

## Dollard / Steinmetz / Whittaker Claim

The local Dollard/Steinmetz/Whittaker synthesis is strongest when framed as a
calibrated transform:

```text
measured hysteresis loop area <-> QA curvature proxy <-> Whittaker-compatible phase transport
```

`[497]` already guards this correctly:

- fixed QA tuple;
- fixed material/drive convention;
- fixed calibration constants;
- deterministic fixture consistency only;
- no universal physical identity claim.

The next meaningful investigation is empirical: hold calibration constants
fixed on one material/drive regime and test out-of-sample loops. If the bridge
generalizes, it becomes a serious measured-transform result. If it does not,
it remains a useful internal analogy.

## Mie / Bromwich / Spherical Modes

The QA-shaped part is discrete indexing, not field-value evaluation:

- scalar spherical lattice: `(n,m)`, `0 <= n <= N`, `-n <= m <= n`,
  count `(N + 1)*(N + 1)`;
- electromagnetic vector lattice: `(pol,n,m)`, `pol in {TE,TM}`,
  `1 <= n <= N`, `-n <= m <= n`, count `2*N*(N + 2)`.

Actual Mie scattering values require Bessel/Hankel/Legendre functions,
boundary matching, material parameters, and observer-side numerical
evaluation. That is not QA-exact Fraction arithmetic. A future exact cert can
cover the mode lattice; it cannot honestly claim scattering cross sections or
longitudinal/scalar energy values without a separate observer-projection
tool and source accounting.

## Investigation Verdict

The correct current position is:

```text
QA has a certified scalar-potential representation bridge and a bounded
Maxwell assembly. It has not globally disproven scalar/longitudinal EM,
free-energy, Bearden/Pond/SVP, or scalar-potential claims. It also has not
certified those positive claims. The next valid step is claim-specific:
source acquisition, normalization, QA carrier/source/Hodge/boundary
definition, energy bookkeeping, and either an exact algebra cert or a
pre-registered empirical observation cert.
```

## Positive Certification Path

For any future scalar/longitudinal EM claim, require:

1. Primary-source packet.
   Exact source passages, page/equation/timestamp, and claim owner.

2. Claim normalization.
   Separate representation claims, physical field claims, energy claims,
   device claims, and metaphysical vocabulary claims.

3. QA carrier model.
   Declare whether the carrier is a scalar, potential pair, cochain, source
   cochain, boundary condition, media/constitutive object, or projection.

4. Hodge/source/boundary evidence.
   Reuse `[510]`, `[511]`, `[512]`, and `[513]` where applicable, or explain why
   a different structure is needed.

5. SVP-CMC gate.
   Check scalar configuration, boundary, neutral center, phase, polarity,
   latency, and forbidden causal language.

6. Energy bookkeeping.
   Classify energy as stored, dissipated, redistributed, radiated, externally
   supplied, or observer-projected. Reject hidden-source accounting.

7. Witness type.
   Use an exact algebra cert for representation identities, a calibrated
   transform cert for hysteresis/bridge claims, or an empirical observation
   cert for physical/device claims.

## Immediate Next Work

- Build no new scalar-energy cert until primary Bearden/Pond/Dollard passages
  are collected.
- If a quick cert is desired, build a narrow claim-normalization cert that
  rejects category collapse between representation, physical field, energy,
  and device claims.
- For Dollard/Steinmetz/Whittaker, extend `[497]` with out-of-sample
  calibration tests rather than broadening the claim text.
- For Mie/Bromwich, build only the exact mode-lattice cert unless primary
  Bromwich source text and field-value boundary conditions are supplied.
