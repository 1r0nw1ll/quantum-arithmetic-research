# Family [209] QA_SIGNAL_GENERATOR_INFERENCE_CERT.v1

## One-line summary

For any m-valued time series, e_t = ((b_{t+1} - b_t - 1) % m) + 1 is the unique A1-compliant generator. The signal IS the orbit; the generator IS the dynamics. Cross-series generator synchrony measures coupling per [207]. Domain-general canonical observer projection for all signal analysis.

## Mathematical content

### Generator inference formula

For consecutive samples b_t, b_{t+1} in {1,...,m}, the QA step is:

    b_{t+1} = ((b_t + e_t - 1) % m) + 1

Inverting:

    e_t = ((b_{t+1} - b_t - 1) % m) + 1

**Properties:**
- **A1 closure**: e_t is always in {1,...,m}. Exhaustively verified for m=9 (81 pairs) and m=24 (576 pairs).
- **Uniqueness**: exactly one e_t per (b_t, b_{t+1}) pair. The QA step is a bijection in e for fixed b_t.
- **Identity**: e_t = m when b_{t+1} = b_t (no change = singularity generator).
- **Minimal step**: e_t = 1 when b_{t+1} = b_t + 1 (mod m, with A1 offset).

### Role distinction ([208])

- b = amplitude state (observed directly from quantized signal)
- e = transition generator (derived from consecutive amplitudes via QA step inversion)

Even when b_t = e_t numerically, their roles are structurally distinct per [208]: b is the state coordinate, e is the multiplier/generator.

### Generator synchrony ([207])

For N concurrent time series at time t:
- **Synchrony** = fraction of series sharing the modal generator
- Range: [1/m, 1.0]
- Synchrony = 1.0: all series have identical generator = singularity coupling (C=2de maximized per [207])
- Synchrony = 1/m: generators uniformly distributed = cosmos independence

### Supersedes

This method supersedes all hardcoded (b,e) lookup tables:
- MICROSTATE_STATES in eeg_orbit_classifier.py
- CMAP lookups in climate scripts (47, 48, 49)
- QUINTILE_TO_STATE in qa_finance_orbit_classifier.py

Generator inference derives (b,e) from signal evolution; hardcoded lookups assign (b,e) by analyst choice.

## Checks

| ID | Description |
|----|-------------|
| SGI_1 | schema_version == 'QA_SIGNAL_GENERATOR_INFERENCE_CERT.v1' |
| SGI_CLOSURE | A1 closure proof present + computationally verified (m=9, m=24) |
| SGI_UNIQUE | Uniqueness proof present + computationally verified |
| SGI_ROLE | Role distinction per [208] documented (b=state, e=generator) |
| SGI_SYNC | Generator synchrony definition with [207] connection |
| SGI_EMPIRICAL | Empirical validation with p < 0.05 |
| SGI_SUPERSEDE | List of superseded hardcoded mappings |
| SGI_SRC | Source attribution |
| SGI_WITNESS | >= 3 witnesses with valid generator sequences |
| SGI_F | fail_ledger well-formed |

## Empirical validation

**EEG seizure detection, corrected (CHB-MIT, 5 patients: chb01/02/03/05/06, 329 windows):**

The single-patient chb01-only headline this section originally reported
(ΔR² = +0.157, p = 0.0003) was an **off-by-one formula bug**, already
retracted in the fixture (`sgi_pass_default.json` →
`empirical_validation.correction`) but never propagated to this doc until
now. The corrected chb01-alone result is ΔR² = +0.051, p = 0.146 (**not
significant** on its own). The real finding is the multi-patient
combination:

- Per-patient ΔR² (delta-only baseline vs + basic [209] generator features): chb01 +0.051 (p=0.146, ns), chb02 +0.439 (p=0.0002, ***), chb03 +0.270 (p=0.0006, ***), chb05 +0.262 (p=0.0001, ***), chb06 +0.423 (p=0.0022, **)
- Mean ΔR² across 5 patients = **0.289**; Fisher's combined χ² = 76.24, p = 1×10⁻¹² (***); 5/5 patients positive direction
- The "beyond topographic Observer 3: ΔR²=+0.085, p=0.024" and "Combined R²=0.757 (vs delta-only 0.421)" figures from the original version of this section do not appear in the fixture's `empirical_validation` block at all and could not be traced to any comparison in `eeg_209_full_stack.py` or `eeg_combined_observer_test.py` on independent reproduction — removed as unsupported.

**Feature directions** (per the corrected fixture, not the retracted single-patient run):
- Generator entropy: seizure < baseline in most patients (more structured dynamics)
- Generator synchrony: **not** consistently seizure > baseline (only 1/5 patients) — distribution shape matters more than cross-channel coordination alone
- Discriminative power comes from the full feature set (singularity_frac, entropy, mean_f), not synchrony alone

## Connection to other families

- **[207] QA_CIRCLE_IMPOSSIBILITY_CERT.v1**: Singularity = maximum C = maximum coupling. Generator synchrony measures this directly.
- **[208] QA_QUADRANCE_PRODUCT_CERT.v1**: b and e are role-distinct factors. Generator inference produces this by construction.
- **[205] QA_GRID_CELL_RNS_CERT.v1**: Grid cells implement modular arithmetic; generator inference is the temporal analogue.
- **[191] QA_BATESON_LEARNING_LEVELS_CERT.v1**: Generator distribution = Level I dynamics; synchrony change = Level II.

## Source

Will Dale + Claude, 2026-04-08. Insight during EEG observer development: "QA applies to signal analysis. All data is generalizable to graphs. The signal IS the orbit."

## Status

- Validator: `qa_alphageometry_ptolemy/qa_signal_generator_inference_cert_v1/qa_signal_generator_inference_cert_validate.py`
- Fixtures: 1 PASS + 1 FAIL
- Self-test: PASS

## Verification Note (2026-07-07)

Independently confirmed the core algebraic claims by brute force:
`verify_a1_closure` and `verify_uniqueness` are already genuinely computed
by the validator over m=9 (81 pairs) and m=24 (576 pairs) — no
fixture-trusting gap.

For the empirical claim, downloaded the real CHB-MIT chb01 patient data
(all 7 seizure-containing files: 03/04/15/16/18/21/26, fetched fresh from
PhysioNet since it wasn't present locally or on LaCie) and re-ran the
actual `eeg_209_full_stack.py` script's "+ basic [209] (sing/synch/ent/f)"
nested-model test against the true delta-only baseline. Result: ΔR² =
+0.0512, p = 0.145978 — an essentially exact match to the fixture's own
already-declared corrected value (ΔR²=0.051, p=0.146). This independently
confirms the fixture's correction is genuine and the underlying
methodology (nested logistic regression, likelihood-ratio chi-square
test) is real, not fabricated.

Found that the human-tract doc (this file) had never been updated to
reflect that correction — it still stated the retracted, pre-correction
single-patient number (ΔR²=+0.157, p=0.0003) as the headline finding, and
included two further figures ("beyond Observer 3" ΔR²=+0.085, p=0.024;
combined R²=0.757) that do not appear anywhere in the fixture and could
not be traced to any comparison in either `eeg_209_full_stack.py` or
`eeg_combined_observer_test.py`. Rewrote the Empirical Validation section
to match the fixture's actual (corrected, multi-patient Fisher-combined)
claim and removed the untraceable figures.

Did not independently re-download/re-verify the other 4 patients
(chb02/03/05/06) in the multi-patient Fisher combination this pass — the
chb01 exact match gives strong confidence the correction narrative is
genuine, but the other 4 patients' individual numbers remain
fixture-only, not independently re-derived from raw data.
