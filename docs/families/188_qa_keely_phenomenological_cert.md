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
| KPH_NT | each declared law's `theorem_nt_role` genuinely cross-checked against the top-level `all_laws_observer_projections` claim (hardened 2026-07-06 — previously only the two top-level booleans were trusted, so a law could secretly declare a non-observer-projection role without being caught) |
| KPH_OBS | every law declares a non-empty continuous observable (hardened 2026-07-06: promoted from warning to error) |
| KPH_DISC | every law declares a non-empty discrete QA source (hardened 2026-07-06: promoted from warning to error) |
| KPH_W | ≥3 witnesses AND the union of witnesses' law_refs covers all 17 required laws (hardened 2026-07-06 — previously only witness count was checked, so 3 witnesses repeating one law would still pass) |
| KPH_F | ≥1 falsifier (continuous→QA causal input rejected) |

## Source grounding

- **svpwiki.com**: Keely's 40 Laws of Vibratory Physics (Laws 13-15, 19-26, 30-32, 36, 38, 39)
- **Theorem NT**: Observer Projection Firewall (QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1)
- **Ben Iverson**: discrete QA dynamics as the causal substrate
- **Dale Pond / Vibes**: SVP consultant AI; Category 5 classification (2026-04-03)
- **[155] Bearden Phase Conjugate**: "stress is a pumper" = continuous observable of discrete QCI
- **Audit note (2026-07-04)**: Keely quotes spot-checked against live svpwiki.com (byte-match); underlying QA arithmetic is pre-existing invariant machinery. The category *classification judgment* itself rests on Vibes' (Dale Pond's AI tool) interpretation, not an independent falsifiable check — see `private/keely_40_laws_classification.md` provenance note.

## Connection to other families

- **[153] Keely Triune**: triune orbit structure is the discrete cause these laws observe
- **[184] Keely Structural Ratio**: structural invariants that phenomenological laws measure
- **[155] Bearden Phase Conjugate**: another Theorem NT observer projection family

## Fixture files

- `fixtures/kph_pass_observer.json` — 17 laws with observable type and QA source mapping
- `fixtures/kph_fail_no_nt.json` — falsifier with continuous value used as QA input (T2 violation)

## Verification Note (2026-07-06)

Independently reconfirmed all 17 required law numbers are declared with
non-empty `continuous_observable`/`discrete_qa_source` fields, and that
the 4 witness groups' `law_refs` union to exactly the 17-law set (13-15,
19-26, 30-32, 36/38/39) with no gaps. This cert is inherently
qualitative/classificatory (there is no (b,e) arithmetic to independently
recompute, unlike sibling certs [184]-[187]), so the strongest available
hardening is cross-checking declared per-law data against the top-level
claims rather than recomputing numeric identities.

**Found and hardened a real fixture-trusting gap of the same class as
[184]-[187]**: `KPH_NT` only checked two top-level boolean flags
(`all_laws_observer_projections`, `no_causal_feedback`) and never
cross-checked them against the per-law `theorem_nt_role` field — a
fixture could declare the top-level flags `true` while one or more laws
secretly carried a non-`observer_projection` role, and the check would
still pass. Hardened to iterate every declared law and require
`theorem_nt_role == "observer_projection"`, matching the top-level
claim. Also promoted `KPH_OBS`/`KPH_DISC` from warnings to hard errors
(a law missing its observable or discrete-source mapping is a genuine
gap in the classification claim, not merely informational), and
hardened `KPH_W` to require the union of all witnesses' `law_refs` to
cover every required law number, not just a bare count ≥3 (previously 3
witnesses all repeating the same law would have passed). Verified all
four hardened checks reject planted errors (sneaky non-observer role,
missing observable, missing discrete source, non-covering witness set)
while the real fixtures still pass.

This closes the Keely 5-category cluster audit: [184] Structural Ratio,
[185] Sympathetic Transfer, [186] Dominant Control, [187] Aggregation,
[188] Phenomenological — all five now have validators that genuinely
recompute or cross-check their certified claims rather than trusting
declared strings/flags.
