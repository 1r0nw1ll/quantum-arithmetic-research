# Dollard Versor Algebra, Whittaker Scalar Potentials, and QA Harmonic Curvature

## 1. Purpose and Scope

This note consolidates the existing repo material that connects Eric Dollard and Charles Steinmetz style operational electrical algebra, Tom Bearden and Whittaker scalar-potential interpretation, Don Briddell Field Structure Theory (FST) loop structure, and Quantum Arithmetic (QA) harmonic curvature.

The working thesis is conservative:

**Dollard/Steinmetz/versor algebra is not invalidated by Whittaker/Bearden scalar-potential theory. In QA terms, Dollard gives the 3D operational/projection algebra, Bearden/Whittaker give a 4D potential-space interpretation, and QA supplies a proposed harmonic modular transform connecting them.**

This is a reconciliation layer, not an identity claim. The sources below do not support saying that Dollard and Bearden are "saying the same thing." They support the more precise claim that their vocabularies can be treated as different projections of a shared field-process vocabulary when QA is used as the explicit bridge.

## 2. Source Spine

Primary source files:

1. `field-structure-theory/03-integration-frameworks/whittaker-bearden-scalar/Branch · Dollard Bearden conflict.md`
2. `field-structure-theory/01-core-theory/qa-fst-comparisons/FST and STF Analysis.md`
3. `field-structure-theory/01-core-theory/qa-fst-comparisons/QAFST Mathematical Review.md`
4. `docs/ai_chats/QA as Geometric Algebra.md`

Roles:

| File | Role |
|---|---|
| `Branch · Dollard Bearden conflict.md` | Main reconciliation thread: Dollard/Steinmetz hysteresis, Bearden/Whittaker scalar-potential interpretation, and the QA curvature-flux bridge. |
| `FST and STF Analysis.md` | Main QA-FST-Dollard mapping: canonical tuple `(1,1,2,3)`, Briddell loops, Sierpinski recursion, Dollard four-quadrant electricity, and versor-algebra language. |
| `QAFST Mathematical Review.md` | Formal QAFST field-theory layer: QA flux shells, Rauscher comparison, and Dollard quadrapolar versor interpretation. |
| `QA as Geometric Algebra.md` | Algebra bridge: QA as discrete geometric/Clifford algebra, with modular rotor steps converging to continuous GA rotors. |

## 3. Dollard / Steinmetz / Versor Layer

The Dollard/Steinmetz layer is operational and projection-first:

- It emphasizes measurable circuit behavior, magneto-dielectric media, coils, hysteresis, quadrature, and reproducible engineering procedure.
- In the conflict thread, Steinmetz hysteresis is framed as loop work in a B-H cycle:

\[
W_h = \oint H\,dB = 2\pi \hat H^2 \mu''(\omega)
\]

- The same file frames Dollard's approach as aligned with Steinmetz-style algebra and versor methods for circuits and waves.
- The FST/STF file links Dollard's four-quadrant electricity to QA's `2d = 4` polarity structure and to rotational invariance in complex space, described there as versor algebra.

For this synthesis, the Dollard layer should be treated as the 3D operational layer: it governs measured phase lag, loop area, reactive/real power transitions, and physical projection into circuit or material observables.

## 4. Bearden / Whittaker Scalar-Potential Layer

The Bearden/Whittaker layer is potential-space-first:

- The conflict thread describes Bearden as leaning on Whittaker-style scalar-potential decompositions and active-vacuum/plenum language.
- The same thread explicitly marks the "Dollard called Bearden a disinformation agent" claim as unverified hearsay unless a primary clip or transcript is found.
- Whittaker is presented there as a scalar-potential representation in which longitudinal plane-wave components can encode phase transport.

For this synthesis, Bearden/Whittaker should be treated as the 4D potential-space layer: it is useful for interpreting hidden phase transport and scalar-potential organization, but it must not be used to erase the Dollard/Steinmetz operational layer.

## 5. QA Harmonic Curvature Bridge

The repo's strongest bridge appears in `Branch · Dollard Bearden conflict.md`, where the Steinmetz loop area is mapped to QA curvature flux and Whittaker phase transport.

Preserve the QA invariant definitions exactly:

\[
J = b d,\qquad X = d e,\qquad K = d a,\qquad F = b a,\qquad C = 2 e d,\qquad G = e^2 + d^2.
\]

The bridge defines a QA sector phase and a scalar QA plenum potential:

\[
\Psi_{\mathrm{QA}}(t) = K e^{i\theta(t)} - J e^{-i\theta(t)} = R(t)e^{i\varphi(t)}.
\]

It then defines a QA field-symmetry 1-form and curvature:

\[
\mathcal{A}_{\mathrm{QA}} = \Pi\,d\theta,\qquad
\Pi = \alpha_X X + \alpha_J J + \alpha_K K,
\]

\[
\mathcal{F}_{\mathrm{QA}} = d\mathcal{A}_{\mathrm{QA}} = d\Pi \wedge d\theta.
\]

The calibration constants \(\alpha_X,\alpha_J,\alpha_K\) are dimension-balancing constants. They are fixed for a material/sample and drive convention; they should not be silently refit per result if the bridge is being used predictively.

## 6. Formal Bridge Equation

The central source equation is:

\[
\boxed{
\oint H\,dB
=
\iint_{\Sigma_T}\mathcal{F}_{\mathrm{QA}}
=
\Delta\int d\varphi
=
2\pi \hat H^2 \mu''(\omega)
}
\]

Interpretation:

| Term | Interpretation |
|---|---|
| \(\oint H\,dB\) | Steinmetz/Dollard operational loop work: the measured B-H hysteresis area. |
| \(\iint_{\Sigma_T}\mathcal{F}_{\mathrm{QA}}\) | QA curvature flux over the torus traced by \((\Pi,\theta)\). |
| \(\Delta\int d\varphi\) | Whittaker-compatible scalar phase transport through \(\Psi_{\mathrm{QA}}\). |
| \(2\pi \hat H^2 \mu''(\omega)\) | Single-tone Steinmetz loss expression under the stated drive convention. |

Guardrail: this equation should be read as a proposed calibrated transform:

**Steinmetz loop work = QA curvature flux = Whittaker phase transport under fixed QA seed, fixed material/drive convention, and fixed calibration constants.**

It should not be stated as a source-established universal physical identity.

## 7. FST Loop and Toroidal Interpretation

`FST and STF Analysis.md` supplies the FST embedding:

- The canonical QA tuple `(b,e,d,a) = (1,1,2,3)` is mapped to Sierpinski first-iteration structure, Briddell loop logic, and Dollard four-quadrant electricity.
- `a = 3` is interpreted as three field loops or primal helices.
- `d = 2` and `2d = 4` are used to connect QA nodal structure to four-polarity / four-quadrant electrical symmetry.
- A toroidal shell chart appears as:

\[
(x,y,z) = (a\cos\theta,\ d\sin\theta,\ e\sin\phi).
\]

In this synthesis, FST provides the loop/topology substrate:

| FST / STF item | QA item | Dollard/field interpretation |
|---|---|---|
| 3 primal loops | `a = 3` | Resonant loop units / output amplitude |
| Edge or nodal transition | `d = 2` | Phase node / torsion plane |
| Four polarity points | `2d = 4` | Four-quadrant electrical projection |
| Recursive Sierpinski growth | tuple iteration | Self-similar toroidal shell recursion |

## 8. Versor, Clifford, and QA Modular Rotor Bridge

`QA as Geometric Algebra.md` supplies the algebraic companion to the Dollard/versor files. The key bridge is:

\[
\text{GA rotor } e^{\theta e_{12}}
\quad\leftrightarrow\quad
\text{QA modular phase-step } \mathcal{R}_k,\qquad \theta = \frac{2\pi k}{24}.
\]

The same file gives a continuum-limit statement:

\[
\lim_{N\to\infty}\mathcal R_{N,k(N)} = e^{\theta e_{12}},
\qquad
\frac{2\pi k(N)}{N}\to\theta.
\]

It also ties the discrete Clifford unit to the quarter-turn:

\[
Q=\mathcal R_{N,N/4},\qquad Q^2=-1.
\]

The QA ellipse law gives the angle components from the integer tuple:

\[
C = 2de,\qquad F = ab,\qquad G = d^2 + e^2,\qquad C^2 + F^2 = G^2,
\]

\[
\sin\theta = \frac{2de}{d^2+e^2},\qquad
\cos\theta = \frac{ab}{d^2+e^2}.
\]

This is the clean algebraic bridge to Dollard's versor framing: QA modular phase steps act as discrete rotor operations; in a refinement limit they converge to continuous Clifford/GA rotors.

## 9. Translation Table

| Domain | Dollard / Steinmetz / Versor | Bearden / Whittaker | QA synthesis |
|---|---|---|---|
| Medium | Aether / magneto-dielectric continuum | Plenum / active vacuum | QA field manifold / space-counterspace shell |
| Algebra | Versor / Steinmetz operational algebra | Scalar-potential decomposition | Modular harmonic curvature algebra |
| Main observable | Circuit behavior, coils, hysteresis, quadrature | Longitudinal scalar-potential structure | Tuple-based shell dynamics |
| "Randomness" | Material hysteresis, statistical domain behavior | Hidden order / chaotic vacuum structure | Projection of deterministic modular dynamics |
| Energy loop | \(\oint H\,dB\) | Scalar phase transport | \(\iint \mathcal{F}_{\mathrm{QA}}\) |
| Rotation | Four-quadrant electricity / versor phase | Potential phase symmetry | Mod-24 modular rotor |
| QA bridge | Physical projection | Higher-dimensional potential | Proposed harmonic transform, pending validation |

## 10. Guardrails

Source-grounded:

- The repo contains an explicit QA-Steinmetz-Whittaker bridge equation in `Branch · Dollard Bearden conflict.md`.
- The repo contains a Dollard/FST/Sierpinski/QA mapping in `FST and STF Analysis.md`.
- The repo contains QAFST discussion linking QA flux shells to Dollard quadrapolar versor algebra in `QAFST Mathematical Review.md`.
- The repo contains a QA-to-GA/Clifford modular rotor bridge in `QA as Geometric Algebra.md`.

Interpretive synthesis:

- Treating Dollard/Steinmetz as the 3D operational projection and Bearden/Whittaker as the 4D potential-space layer is a reconciliation model assembled from the source spine.
- Treating random material hysteresis as a projection of deterministic modular dynamics is a QA interpretation, not an established external physics claim.
- Treating QA curvature flux as physically predictive requires fixed calibration constants and out-of-sample loop tests.

Do not claim:

- Dollard and Bearden are identical.
- Dollard's versor algebra is replaced by Bearden/Whittaker scalar-potential theory.
- Whittaker scalar potentials alone prove vacuum-energy claims.
- The "disinformation agent" claim is verified. It remains unverified until a primary source with a timestamp or transcript is found.
- The bridge equation is a universal physical identity before certs/simulations validate the transform.

Preferred language:

- "projection"
- "extension"
- "operational layer"
- "potential-space layer"
- "curvature bridge"
- "calibrated transform"
- "fixed material/drive convention"

Avoid:

- "Dollard and Bearden are saying the same thing"
- "QA proves Bearden"
- "Whittaker replaces Steinmetz"
- "hysteresis is vacuum energy"
- "validated" unless the validation artifact is named

## 11. Proposed Cert Targets

### QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1

Purpose: certify that a measured/projected hysteresis loop can be represented as QA curvature flux under fixed calibration constants, without claiming physical identity beyond the defined transform.

Minimum checks:

- Input contains measured or fixture `H(t), B(t)` loop data.
- Validator computes \(W_h = \oint H\,dB\).
- Validator computes QA invariants \(J=bd\), \(X=de\), \(K=da\), \(F=ba\), \(C=2ed\), \(G=e^2+d^2\).
- Calibration constants are declared once and reused.
- QA curvature integral matches the declared tolerance.
- Cert explicitly states that physical identity is not claimed beyond the transform.

### QA_DOLLARD_VERSOR_ROTATION_CERT.v1

Purpose: certify that a QA modular rotor step implements the corresponding discrete rotation and preserves the declared tuple/residue relations.

Minimum checks:

- Input declares modulus \(N\), step \(k\), and tuple `(b,e,d,a)`.
- Validator checks \(d=b+e\) and \(a=b+2e\), modulo \(N\) where appropriate.
- Validator verifies \(\mathcal R_{N,k_1}\circ\mathcal R_{N,k_2}=\mathcal R_{N,k_1+k_2}\).
- For \(N\) divisible by 4, validator verifies quarter-turn \(Q^2=-1\) in the 2D matrix representation.

### QA_FST_DOLLARD_QUADRATURE_MAP.v1

Purpose: certify the declared QA-FST-Dollard quadrature mapping for the canonical tuple and allowed recursive variants.

Minimum checks:

- Input declares tuple family and recurrence.
- Validator checks canonical `(1,1,2,3)` relations.
- Validator checks `2d = 4` for the canonical four-polarity mapping.
- Validator checks declared loop-count and node-count fields against the tuple.
- Cert labels FST/Dollard correspondences as interpretive mappings unless supported by a named source artifact.
