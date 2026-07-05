# Family [174] QA_PHI_TRANSFORMATION_CERT.v1

## One-line summary

Phi(D) transformation law classifying disorder-stress vs order-stress. 2/2 pre-registered (cardiac, EMG), 4/4 post-hoc consistent.

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_phi_transformation_cert_v1/qa_phi_transformation_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-05)

Fixed the stale doubled "cert_cert" path above (same pattern found in
[170]/[171]/[172]/[173]/[175] this cycle). Corrected the one-line
summary's "6/6 post-hoc consistent" to "4/4" — the fixture's
`post_hoc_consistent` array has exactly 4 entries (EEG, finance, climate,
audio); the doc's "6/6" appears to have been copy-drifted from a sibling
cert's domain count rather than reflecting this cert's own fixture.

**One gap noted, not filled**: unlike [173]/[175] (whose domain lists
were fixable by adding ERA5 — an objective fact, not requiring
interpretation), this cert requires classifying each domain as
disorder-stress (Φ=-1) or order-stress (Φ=+1), which is an interpretive
judgment. ERA5's own materials (`49_forecast_coherence_surrogates.py`,
cert [172]'s doc) never establish a Φ(D) classification — there's no
basis in the existing project record to say whether ERA5 belongs in
`pre_registered`, `post_hoc_consistent`, or neither. Left unfilled
rather than inventing a classification; a real research gap, not a
documentation bug.
