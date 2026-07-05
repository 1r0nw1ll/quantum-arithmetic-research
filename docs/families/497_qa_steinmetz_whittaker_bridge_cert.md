# [497] QA Steinmetz-Whittaker Bridge Cert

## What this is

A documentation-first empirical scaffold for the proposed bridge:

```
∮ H dB  ↔  ∬_Σ F_QA  ↔  Δ∫ dφ
```

The cert validates that for a fixed QA tuple `(b,e,d,a)`, fixed calibration constants, and explicit fixture data for `H(t)`, `B(t)`, `theta(t)`, the hysteresis loop integral `∮ H dB` equals the QA curvature proxy `∮ Π dθ` within declared tolerance.

## Claim (narrow)

Given:
- QA tuple `(b,e,d,a)` with `d = b+e` and `a = b+2e`
- Calibration constants `alpha_X`, `alpha_J`, `alpha_K`
- Fixture data: `H` (field intensity), `B` (flux density), `theta` (displacement angle)

The validator checks that the closed-loop hysteresis area matches the QA curvature proxy:

```
∮ H dB  ≈  ∮ Π dθ   where  Π = alpha_X·X + alpha_J·J + alpha_K·K
```

within the declared `tolerance`.

QA invariants used:

| Symbol | Formula |
|--------|---------|
| J | b·d |
| X | d·e |
| K | d·a |
| F | b·a |
| C | 2·e·d |
| G | e·e + d·d |

## Guardrail

**This cert validates deterministic transform consistency only.** It does not validate or prove a universal physical identity between Steinmetz, Whittaker, Dollard, Bearden, or QA.

Passing this cert means only that the fixture's declared QA bridge data is internally consistent under the validator's deterministic rules.

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_steinmetz_whittaker_bridge_cert_v1/validate.py` |
| Mapping ref | `qa_steinmetz_whittaker_bridge_cert_v1/mapping_protocol_ref.json` |
| PASS fixture (minimal) | `qa_steinmetz_whittaker_bridge_cert_v1/fixtures/pass_minimal_loop.json` |
| PASS fixture (variable π) | `qa_steinmetz_whittaker_bridge_cert_v1/fixtures/pass_variable_pi_loop.json` |
| FAIL fixture (changed calibration) | `qa_steinmetz_whittaker_bridge_cert_v1/fixtures/fail_changed_calibration.json` |
| Spec | `qa_steinmetz_whittaker_bridge_cert_v1/SPEC.md` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1
python3 validate.py --self-test
```

Expected: `{"ok":true}`

## Checks

| Check | Meaning |
|-------|---------|
| `cert_family` | Schema version must be `QA_STEINMETZ_WHITTAKER_BRIDGE_CERT.v1` |
| `tolerance` | Declared tolerance is a finite positive number |
| `guardrail` | Guardrail string matches canonical disclaimer |
| `declared_invariants` | J/X/K/F/C/G values match recomputed formulas |
| `calibration_match` | `calibration_constants` and `material_drive_convention` unchanged between calibration and evaluation sections |
| `hysteresis_area` | Closed-loop trapezoid integral of H dB |
| `curvature_proxy` | Closed-loop integral of Π dθ (constant Π or sampled `pi_series`) |
| `bridge_close` | `|hysteresis_area - curvature_proxy| ≤ tolerance` |

## QA Axiom Compliance

- **A1**: QA tuple `(b,e,d,a)` in {1,...,m}; never 0
- **A2**: `d = b+e` and `a = b+2e` — derived, not assigned independently; validator enforces this exactly
- **T2**: hysteresis/curvature integrals are observer outputs over fixed integer tuple; float arithmetic confined to the observer layer; tuple and invariants remain integer throughout
- **S1**: `G = e*e + d*d` (no `**2`)
- **S2**: `b`, `e` are integer inputs; no float state in the QA layer
- **T1**: no continuous time variable in QA logic; `H(t)`, `B(t)`, `theta(t)` are sampled observer sequences

## Primary Sources

- Steinmetz, C.P. (1892). On the law of hysteresis. *AIEE Trans.* 9:3–64.
- Whittaker, E.T. (1903). On the partial differential equations of mathematical physics. *Math. Annalen* 57:333–355. DOI 10.1007/BF01444290.
- Wildberger, N.J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. (QA tuple invariants, BEDA representation.)

## Relation to other certs

- **[498] `qa_whittaker_phase_packet_algebra_cert_v1`** — companion cert on the Whittaker side; [497] anchors the bridge operator (hysteresis ↔ curvature); [498] anchors the exact rational phase-packet structure on which the Whittaker ladder builds.
- **[273] `qa_whittaker_rational_direction_s2_cert_v1`** — provides the exact rational S2 directions that feed into [498]; [497] is independent of [273] but shares the broader Whittaker program.

## Scope boundary

**The cert does NOT:**
- Prove the physical equivalence of the Steinmetz hysteresis equation and the Whittaker differential equation
- Claim that any real material or device implements the bridge numerically
- Assert a causal relationship from the continuous integrals back into the QA tuple (Theorem NT)

**The cert DOES:**
- Verify that the QA tuple relations and invariant definitions are internally consistent
- Check that the hysteresis loop integral and curvature proxy agree within declared tolerance on the fixture data
- Enforce that calibration constants cannot change between the calibration and evaluation phases

## Verification Note (2026-07-05)

Independently checked both primary-source citations. **Steinmetz, C.P.
(1892) "On the Law of Hysteresis"** confirmed real — Transactions of the
American Institute of Electrical Engineers, volume IX(2), pages 3-64,
matching "AIEE Trans. 9:3-64" exactly. **Whittaker, E.T. (1903) "On the
partial differential equations of mathematical physics"** confirmed
real — *Mathematische Annalen* volume 57, pages 333-355, DOI
10.1007/BF01444290, exact match (same real paper independently confirmed
elsewhere in the project's Whittaker-ladder certs [266]/[273]/[274]).

Validator (`validate.py`) already performs genuine numeric computation
(`loop_integral_h_db` trapezoidal integration, `curvature_proxy`), not
fixture-trusting. No bugs found in the checked computation.

**Investigated whether the underlying bridge is real, not just whether
the cert hedges about it** — and then had to correct that investigation
once more (both corrections happened in the same audit session; see
`feedback_no_hedging_outcome_predictions.md` for the full trail). First
pass: claimed Dollard's "hysteresis of the aether" reinterpretation
rests on the classical luminiferous aether, which Michelson-Morley
(1887) ruled out experimentally — treating the question as settled.
That was too fast. Checking Dollard's own definition of "aether" more
carefully: he explicitly frames it as "counterspace... the so-called
vacuum or zero-point fulcrum of QED," described via permittivity and
permeability — i.e., he's relabeling real, mainstream QED vacuum/zero-
point concepts under nonstandard "aether" terminology, not directly
reasserting the specific 19th-century mechanical medium (with a
preferred rest frame and detectable "aether wind") that M-M falsified.

**Honest current state, not resolved either way**: whether Dollard's
specific claim — that Steinmetz's hysteresis law describes energy loss
in this reframed "vacuum medium," bridged via Whittaker's 1903
potential-theory decomposition — has any physical validity is genuinely
unverified, not confirmed and not conclusively false. It doesn't map
cleanly onto the disproven classical aether (so my first correction
overreached), and I have found no evidence it maps onto any accepted
or actively-studied physics framework either (mainstream QED vacuum
treatment, stochastic electrodynamics, or hidden-variable/pilot-wave
theories) — Dollard is not affiliated with, and as far as found here
not endorsed by, any of those research communities. This is an open,
unverified claim, not a settled one in either direction. The cert's own
narrow claim (internal transform consistency on fixture data) is
unaffected either way and remains valid as scoped.
