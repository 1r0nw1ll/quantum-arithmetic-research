# QA Steinmetz-Whittaker Bridge Cert v1

Documentation-first empirical cert scaffold for the proposed bridge:

\[
\oint H\,dB \leftrightarrow \iint_{\Sigma} F_{\mathrm{QA}} \leftrightarrow \Delta\int d\phi
\]

## Purpose

This cert validates deterministic transform consistency under:

- a fixed QA tuple `(b,e,d,a)`;
- a fixed material/drive convention;
- fixed calibration constants `alpha_X`, `alpha_J`, `alpha_K`;
- explicit fixture data for `H(t)`, `B(t)`, and `theta(t)`.

## Guardrail

This cert validates deterministic transform consistency only. It does not validate or prove a universal physical identity between Steinmetz, Whittaker, Dollard, Bearden, or QA.

Passing this cert means only that the fixture's declared QA bridge data is internally consistent under the validator's deterministic rules.

## Files

| File | Role |
|---|---|
| `SPEC.md` | Schema and validation contract. |
| `validate.py` | Pure Python stdlib validator. |
| `fixtures/pass_minimal_loop.json` | Minimal passing loop with reused calibration constants. |
| `fixtures/pass_variable_pi_loop.json` | Passing loop with sampled time-varying `pi_series`. |
| `fixtures/fail_changed_calibration.json` | Failing fixture where evaluation calibration mutates after calibration. |

## Validation Model

The validator checks tuple relations:

\[
d = b + e,\qquad a = b + 2e
\]

It checks canonical QA invariant definitions:

\[
J = b d,\quad X = d e,\quad K = d a,\quad F = b a,\quad C = 2 e d,\quad G = e^2 + d^2
\]

It computes the hysteresis loop area using a closed-loop trapezoid-style line integral:

\[
\oint H\,dB \approx \sum_i \frac{H_i + H_{i+1}}{2}(B_{i+1}-B_i)
\]

It computes the QA curvature proxy:

\[
\oint \Pi\,d\theta,\qquad \Pi=\alpha_X X+\alpha_J J+\alpha_K K
\]

For the minimal mode, \(\Pi\) is constant across the loop. If `evaluation.pi_series` is present, the validator instead treats it as sampled \(\Pi(t)\) and computes:

\[
\oint \Pi\,d\theta \approx \sum_i \frac{\Pi_i + \Pi_{i+1}}{2}(\theta_{i+1}-\theta_i)
\]

Calibration constants are fixture-declared in both `calibration` and `evaluation` sections and must match exactly.

## Usage

```bash
python3 qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/validate.py \
  qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/fixtures/pass_minimal_loop.json

python3 qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/validate.py \
  qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/fixtures/pass_variable_pi_loop.json

python3 qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/validate.py \
  qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/fixtures/fail_changed_calibration.json

python3 qa_alphageometry_ptolemy/qa_steinmetz_whittaker_bridge_cert_v1/validate.py --self-test
```
