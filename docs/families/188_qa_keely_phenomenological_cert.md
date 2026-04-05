# Family [188] QA_KEELY_PHENOMENOLOGICAL_CERT.v1

## One-line summary

Keely's 17 phenomenological laws (Laws 13-15, 19-26, 30-32, 36, 38, 39) are all Theorem NT observer projections — continuous measurements (temperature, pressure, frequency, refractive index) that are EFFECTS of discrete QA structure, never causal inputs — Category 5 (42.5%) of the Vibes 5-category framework.

## Mathematical content

### Keely's phenomenological laws

| Law # | Name (svpwiki) | Observable type |
|-------|---------------|----------------|
| 13 | Law of Sono-Thermity | temperature (continuous) |
| 14 | Law of Oscillating Corpuscles | pressure amplitude (continuous) |
| 15 | Law of Vibrating Rotating Spheres | angular frequency (continuous) |
| 19 | Law of Dispersion | refractive index (continuous) |
| 20 | Law of Transmission of Gravity | force magnitude (continuous) |
| 21 | Law of Atomic Vibration | spectral frequency (continuous) |
| 22 | Law of Oscillating Atomic Substances | density (continuous) |
| 23 | Law of Acoustic Assimilation | acoustic impedance (continuous) |
| 24 | Law of Magnetic Assimilation | magnetic permeability (continuous) |
| 25 | Law of Etheric Attraction | field strength (continuous) |
| 26 | Law of Etheric Repulsion | field gradient (continuous) |
| 30 | Law of Electric Conductance | conductivity (continuous) |
| 31 | Law of Vibrating Atomic Substances | wavelength (continuous) |
| 32 | Law of Atomic Synthesis | energy (continuous) |
| 36 | Law of Sympathetic Vibration | amplitude envelope (continuous) |
| 38 | Law of Electric Induction | induced EMF (continuous) |
| 39 | Law of Cohesion | tensile modulus (continuous) |

### Category 5: phenomenological (Theorem NT)

This is the largest category (17/40 = 42.5%). Every law describes a **continuous physical measurement** — an observer projection of underlying discrete QA dynamics. By Theorem NT (Observer Projection Firewall):

1. Discrete QA dynamics produce orbit structure (cause)
2. Continuous observables are projected FROM this structure (effect)
3. The boundary is crossed exactly twice: input → QA layer, QA layer → output
4. **No continuous measurement ever feeds back as a QA causal input**

### Structural implication

These 17 laws confirm that Keely's experimental observations (sono-thermity, dispersion, conductance, etc.) are all consistent with being observer projections of a discrete vibratory substrate — exactly what QA predicts.

## Checks

| ID | Description |
|----|-------------|
| KPH_1 | schema_version == 'QA_KEELY_PHENOMENOLOGICAL_CERT.v1' |
| KPH_LAWS | all 17 law numbers present: {13,14,15,19,20,21,22,23,24,25,26,30,31,32,36,38,39} |
| KPH_NT | each law's observable is typed as continuous (Theorem NT compliant) |
| KPH_OBS | observer projection direction verified: QA → continuous, never reverse |
| KPH_DISC | underlying discrete QA state identified for each observable |
| KPH_W | ≥3 witnesses (distinct law→observable→QA mappings) |
| KPH_F | ≥1 falsifier (continuous→QA causal input rejected) |

## Source grounding

- **svpwiki.com**: Keely's 40 Laws of Vibratory Physics (Laws 13-15, 19-26, 30-32, 36, 38, 39)
- **Theorem NT**: Observer Projection Firewall (QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1)
- **Ben Iverson**: discrete QA dynamics as the causal substrate
- **Dale Pond / Vibes**: SVP consultant AI; Category 5 classification (2026-04-03)
- **[155] Bearden Phase Conjugate**: "stress is a pumper" = continuous observable of discrete QCI

## Connection to other families

- **[153] Keely Triune**: triune orbit structure is the discrete cause these laws observe
- **[184] Keely Structural Ratio**: structural invariants that phenomenological laws measure
- **[155] Bearden Phase Conjugate**: another Theorem NT observer projection family

## Fixture files

- `fixtures/kph_pass_observer.json` — 17 laws with observable type and QA source mapping
- `fixtures/kph_fail_no_nt.json` — falsifier with continuous value used as QA input (T2 violation)
