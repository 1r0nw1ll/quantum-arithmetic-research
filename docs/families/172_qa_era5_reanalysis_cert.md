# Family [172] QA_ERA5_REANALYSIS_CERT.v1

## One-line summary

QCI as predictor of atmospheric variability using WeatherBench2 ERA5 data (3297 days x 15 channels, 500hPa).

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_era5_reanalysis_cert_v1/qa_era5_reanalysis_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-04)

Found the real backing pipeline in the repo root: `49_forecast_coherence_surrogates.py`
(the corrected v2) operating on `.era5_extracted.csv` — a real 3297-row,
15-column extraction from WeatherBench2's actual public ERA5 dataset
(5 grid points × 3 variables = 15 channels; row count and column count
both match the cert's claimed "3297 days x 15 channels" exactly).

**Important context found in the repo**: an earlier version,
`49_forecast_coherence_surrogates_v1_CIRCULAR_BUG.py`, had a real
methodological bug (its own docstring in the corrected version states:
"v1 had two issues: 1. Surrogate generated its own targets (circular)
2. Sign test was one-sided only"). Checked which version's results feed
this cert: **the cert correctly uses the corrected v2 results, not the
debunked v1 ones.** Confirmed by comparing both saved result JSONs
(`49_forecast_coherence_surrogate_results.json` = v1, buggy: all 4
surrogates show `"beats": false`, i.e. v1's own flawed methodology
didn't even support the claim; `49_forecast_coherence_surrogate_v2_results.json`
= v2, corrected: all 4 surrogates show `"beats": true` for both r and
partial_r) — exactly matching this cert's declared `surrogates_beaten: 4,
surrogates_total: 4`.

**Independently re-ran the corrected v2 script directly** (not just read
the saved JSON) against the real CSV: reproduced r=0.4620, partial_r=0.4266,
4/4 surrogates beaten on both metrics, bit-identical to the saved results
(same fixed seed=42). This is a genuine, non-fabricated, reproducible
result — no bug, and appropriately superseded a real prior methodological
flaw rather than quietly carrying it forward.

**Fixed**: the doc's "Machine tract" section had a stale double-"cert_cert"
path (`qa_era5_reanalysis_cert_cert_v1/...`) that didn't match the actual
directory (`qa_era5_reanalysis_cert_v1/...`) — likely a copy-paste
scaffold artifact. Corrected above.
