# Family [170] QA_CARDIAC_ARRHYTHMIA_CERT.v1

## One-line summary

QA orbit features as independent predictors of arrhythmia classification beyond R-R interval baseline using MIT-BIH Arrhythmia Database (48 records, 94536 beats).

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_cardiac_arrhythmia_cert_v1/qa_cardiac_arrhythmia_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-05)

Found the real backing script and data: `50_cardiac_preregistered.py`
(repo root) uses the standard `wfdb` PhysioNet library to download the
genuine MIT-BIH Arrhythmia Database — record list (100-124, 200-234
minus gaps) is the real, standard 48-record MIT-BIH numbering. The
script's pre-registration is properly done (Φ(D)=-1 prediction written
*before* seeing results, per its own header comment).

**Independently spot-checked live**: downloaded MIT-BIH record 100
directly from PhysioNet — signal shape (650000, 2) at fs=360 Hz (≈30.1
minutes, matching the claimed "30 min each"), 2274 real annotations with
plausible symbol distribution (2239 'N' normal, 33 'A' APB). Confirmed
the data source is genuine and live-accessible, not fabricated.

**Saved results JSON matches the cert's claims exactly**:
`50_cardiac_preregistered_results.json` shows `n_beats=94536`,
`records_used=48`, `delta_r2=0.03723` (rounds to the cert's "+0.037"),
`p_qa_add=0.0` (consistent with "p<10⁻⁶"), `n_surr_pass=2` (matches
"2/2 surrogates beaten"), `phi_confirmed=true`. Did not re-run the full
48-record download+train pipeline (would require downloading and
processing the complete dataset over the network) given the single-record
spot-check already confirms the data source and the saved results align
exactly with the cert. No bugs found; also fixed the stale doubled
"cert_cert" path above (same copy-paste artifact found in [172] ERA5's
doc during an earlier audit this cycle).

**One minor placeholder noted, not changed**: the fixture's declared
`p_value=1e-7` is a round-number placeholder, not derived from the
script's actual output (`p_qa_add=0.0`, i.e. underflowed below float64
precision — no more precise real number is recoverable without redoing
the statistical test symbolically). Unlike [171] EMG's stale p-value
(where a precise real replacement was available and substituted),
`1e-7` here is a defensible conservative bound rather than a
substantively wrong number — flagged for completeness, not fixed.
