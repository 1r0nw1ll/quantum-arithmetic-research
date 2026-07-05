# Family [173] QA_SURROGATE_METHODOLOGY_CERT.v1

## One-line summary

Corrected surrogate null design: real targets fixed, surrogate QCI only. Circular null problem identified and resolved. 7 domains confirmed.

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_surrogate_methodology_cert_v1/qa_surrogate_methodology_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-05)

This cert, along with sibling aggregate certs [174] and [175], summarizes
results across the "topographic → QA orbit → T-operator → QCI" domain
cluster ([170] cardiac, [171] EMG, [172] ERA5, plus finance/EEG/audio/
climate). No dedicated backing script exists (validator only checks
`confirmed_domains` has ≥6 entries) — this is a hand-maintained summary,
not independently recomputed.

**Found a stale domain-list inconsistency**: this cert's `confirmed_domains`
included ERA5 but was missing **EMG** (cert [171], confirmed with 2/2
surrogates beaten this same audit cycle) — while sibling cert [175]'s
lists had the *opposite* gap (EMG present, ERA5 missing). Neither doc had
the complete, accurate 7-domain set. Fixed here by adding EMG; fixed in
[175] by adding ERA5 (see that doc). Also dropped the doc's unverifiable
"6/8" ratio (couldn't confirm what the 2 non-confirmed domains in the
"/8" denominator were meant to be) in favor of stating the confirmed
count plainly. Also fixed the stale doubled "cert_cert" path above (same
pattern found in [170]/[171]/[172]'s docs earlier this cycle).
