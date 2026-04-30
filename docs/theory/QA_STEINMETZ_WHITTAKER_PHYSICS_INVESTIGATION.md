# QA Steinmetz-Whittaker Physics Investigation

## 1. Physical Question

Can measured B-H hysteresis loop work

\[
\oint H\,dB
\]

be represented or predicted by QA curvature

\[
\oint \Pi\,d\theta
\]

under fixed calibration?

The investigation is not whether QA proves Dollard, Bearden, Whittaker, or Steinmetz. The investigation is whether a fixed QA curvature coordinate can represent measured hysteresis loop work and then predict held-out loop work without refitting.

## 2. Physical Observables

Minimum observable set:

- \(H(t)\): magnetizing field over one or more closed drive cycles.
- \(B(t)\): magnetic flux density over the same samples.
- \(\oint H\,dB\): loop area, interpreted as energy lost per cycle per unit volume under standard B-H conventions.
- Drive frequency.
- Drive amplitude.
- Material or core type.
- Temperature, if available.

Useful secondary observables:

- Sample geometry and path length used to reconstruct \(H(t)\).
- Coil turns, excitation current, and pickup voltage if raw electromagnetic measurements are used.
- Saturation regime or maximum \(B\).
- Minor-loop versus major-loop status.
- Measurement uncertainty and sampling rate.

## 3. Baseline Physics

The measured side is standard hysteresis physics:

\[
W_h=\oint H\,dB.
\]

For sinusoidal or narrowband drives, this connects to complex permeability:

\[
W_h \approx 2\pi \hat H^2 \mu''(\omega)
\]

under the fixture's declared amplitude and convention.

Baseline models to compare against:

- Steinmetz law or generalized Steinmetz-style frequency/amplitude scaling.
- Complex permeability fits.
- Lossy elliptical B-H loop models.
- Preisach hysteresis models.
- Jiles-Atherton hysteresis models.

The QA bridge is evaluated against these baselines, not in isolation.

## 4. QA Hypothesis

The working QA coordinate is:

\[
\Pi(t)=\alpha_X X(t)+\alpha_J J(t)+\alpha_K K(t)
\]

with canonical QA definitions preserved:

\[
J=bd,\quad X=de,\quad K=da,\quad F=ba,\quad C=2ed,\quad G=e^2+d^2.
\]

The tuple relations remain:

\[
d=b+e,\qquad a=b+2e.
\]

QA does not replace B-H physics. It tests whether a harmonic curvature coordinate predicts loop work:

\[
\oint H\,dB \approx \oint \Pi\,d\theta.
\]

For the first empirical pass, \(\Pi(t)\) may be either:

- constant over a loop, using fixed tuple invariants and fixed \(\alpha\) calibration;
- sampled as \(\Pi(t)\), using a deterministic tuple/phase rule declared before evaluation.

The second mode is stronger because it tests a sampled curvature path rather than only a scalar loop-area match.

## 5. Experimental Protocol

1. Select a material/core and a fixed measurement convention.
2. Collect or load multiple B-H loops across amplitude, frequency, or repeat cycles.
3. Choose one calibration loop.
4. Fit \(\alpha_X,\alpha_J,\alpha_K\) only on the calibration loop.
5. Freeze the tuple rule, phase rule, and calibration constants.
6. Evaluate held-out loops without refitting.
7. Compare predicted \(\oint \Pi\,d\theta\) against measured \(\oint H\,dB\).
8. Report absolute error, relative error, parameter stability, and failure cases.

No held-out result should be accepted if \(\alpha_X,\alpha_J,\alpha_K\) are changed after calibration.

## 6. Baselines

Minimum baselines:

- Constant-\(\Pi\) fit.
- Sinusoidal or lossy elliptical B-H loop fit.
- Polynomial fit to loop area versus drive amplitude and frequency.
- Frequency-amplitude Steinmetz fit.

Useful stronger baselines:

- Jiles-Atherton fit with held-out loop prediction.
- Preisach-style fit with held-out loop prediction.
- Complex-permeability fit where \(\mu''(\omega)\) is estimated from calibration data and tested on held-out drives.

The QA bridge is interesting only if it improves prediction, compression, stability, or interpretability relative to simple baselines.

## 7. Evidence Levels

Weak evidence:

- One-loop fit.
- Post-hoc selection of tuple or calibration constants.
- Matching loop area without held-out prediction.

Useful evidence:

- Calibrate on one loop.
- Hold \(\alpha_X,\alpha_J,\alpha_K\) fixed.
- Predict held-out loops from the same material and drive family.
- Beat at least one simple baseline.

Strong evidence:

- Predict across multiple amplitudes and frequencies.
- Show stable calibration constants within a material class.
- Show structured failure modes.
- Generalize across related materials or core geometries.
- Outperform or compress standard hysteresis baselines without hidden refitting.

## 8. Dataset Plan

Preferred first target:

- Ferromagnetic-core B-H hysteresis loops, because Steinmetz loop work is directly defined on B-H cycles.

Data acquisition sequence:

1. Search for public digitized B-H hysteresis loop datasets for ferrite, silicon steel, transformer core material, or soft magnetic composites.
2. Prefer datasets with multiple loops from the same material at different drive amplitudes or frequencies.
3. Record units, measurement convention, sampling rate, material, geometry, and temperature if available.
4. If public digitized data is unavailable, create synthetic physically standard fixtures first.

Synthetic fixture options:

- Lossy elliptical B-H loop.
- Jiles-Atherton generated loops.
- Preisach generated loops.

Synthetic fixtures are scaffolding only. They can validate the analysis pipeline, but they do not establish physical evidence for QA.

## 9. Guardrails

- Do not claim universal physical identity.
- Do not claim that Whittaker proves vacuum energy.
- Do not claim that Dollard and Bearden are identical.
- Do not treat a one-loop fit as evidence of predictive physics.
- Do not refit calibration constants on held-out loops.
- Treat QA as a testable coordinate transform.
- Treat failures as informative, not as anomalies to tune away.

The core investigation remains:

\[
\text{measured loop work} \quad \leftrightarrow \quad \text{fixed-calibration QA curvature prediction}.
\]
