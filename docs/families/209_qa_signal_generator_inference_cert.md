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

**EEG seizure detection (CHB-MIT chb01):**
- Beyond delta baseline: ΔR² = +0.157, p = 0.0003 (***)
- Beyond topographic Observer 3: ΔR² = +0.085, p = 0.024 (*)
- Combined R² = 0.757 (vs delta-only 0.421)

**Feature directions (all match [207] prediction):**
- Generator synchrony: seizure 0.459 > baseline 0.410 (more coupled)
- Generator entropy: seizure 2.226 < baseline 2.446 (more structured)
- Singularity gen fraction: seizure 0.436 > baseline 0.380 (more static)

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
