# Family [171] QA_EMG_PATHOLOGY_CERT.v1

## One-line summary

QA orbit features as independent predictors of EMG pathology classification beyond RMS using PhysioNet EMG Database (3 records, 1183 windows).

## Machine tract

- Validator: `qa_alphageometry_ptolemy/qa_emg_pathology_cert_v1/qa_emg_pathology_cert_validate.py`
- Fixtures: `pass_default.json`, `fail_missing_field.json`
- Meta-validator: registered in `qa_meta_validator.py` FAMILY_SWEEPS

## Status

PASS (self-test ok)

## Verification Note (2026-07-05)

Found the real backing script: `53_emg_preregistered.py` (repo root),
using `wfdb` to download the genuine PhysioNet EMG Database (`emgdb`,
3 real records: healthy/myopathy/neuropathy). Properly pre-registered
(Φ(D)=-1 prediction written before seeing results).

**Found and fixed two real bugs**, both metadata-only (the actual model
fit was always correct):

1. **Window count**: the script computed `n_normal`/`n_pathological`
   from the pre-alignment label array (1203 total), but saved them
   alongside `n_windows=len(y_aligned)` (1183, the post-alignment count
   actually used to fit the model) — an internally-inconsistent JSON
   (197+1006=1203 ≠ 1183). The aligned counts were already computed and
   printed to stdout (line 190-191 of the script) but never captured
   into the saved JSON. Fixed the script to save the aligned counts,
   reran it: `n_windows=1183, n_normal=177, n_pathological=1006`
   (177+1006=1183 ✓). All model results (ΔR²=0.6077, p=1.7e-132, both
   surrogates BEATS) are bit-identical before and after the fix,
   confirming the actual analysis was never affected — only the
   reported window/class breakdown was wrong.
2. **Stale p-value**: the cert's fixture declared `p_value=1e-7`, but
   the real script's output is `p_qa_add=1.73e-132` — ~125 orders of
   magnitude more significant than declared. The validator only checks
   `p_value < 0.05` (threshold, not exact match) so this never affected
   PASS/FAIL, and 1e-7 was never an overclaim (the true value is more
   significant, not less) — but it was still a stale placeholder, fixed
   to the real computed value.

Updated together: the script (`53_emg_preregistered.py`), the cert
fixture (`pass_default.json`, both `n_windows` and `p_value`), and this
doc (window count + fixed stale doubled "cert_cert" path). Cert
self-test reruns clean.
