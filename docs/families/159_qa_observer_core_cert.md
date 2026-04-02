# [159] QA Observer Core Cert

**Schema**: `QA_OBSERVER_CORE_CERT.v1`
**Status**: PASS (1 PASS + 1 FAIL fixture)

## What it certifies

The two foundational functions of the topographic observer pipeline, used across all 6 empirical domains.

### qa_mod(x, m)

**Formula**: `((int(x) - 1) % m) + 1`

**Axiom A1 compliance**: Output is ALWAYS in `{1, ..., m}`, never 0. Verified for all integers in `[-100, 200]` with `m = 24` and `m = 9`.

| Input | m=24 | Note |
|-------|------|------|
| 1 | 1 | Identity |
| 24 | 24 | Maximum |
| 25 | 1 | Wraps |
| 0 | 24 | Zero wraps to m |
| -1 | 23 | Negative handled |

### compute_qci(labels, cmap, m, window)

For each triple `(t, t+1, t+2)`, predicts `label[t+2]` via `qa_mod(cmap[label[t]] + cmap[label[t+1]], m)` and checks match. Returns rolling mean of match fractions.

- **Deterministic**: same input always produces same output
- **Output length**: `len(labels) - 2`
- **T2 compliant**: no float-to-int feedback; `int(x)` cast on input only

### 6 domain witnesses

1. Finance (scripts 30-33)
2. EEG (eeg_chbmit_scale.py)
3. Audio (qa_audio_residual_control.py)
4. Seismology (46_seismic_topographic_observer.py)
5. Climate (48_teleconnection_topographic_observer.py)
6. ERA5 (49_forecast_coherence_observer.py)

## How to run

```bash
# Unit tests (15 tests)
cd qa_lab && PYTHONPATH=. python -m pytest qa_observer/tests/ -v

# Validator self-test
cd qa_alphageometry_ptolemy/qa_observer_core_cert_v1
python qa_observer_core_cert_validate.py --self-test
```

## What breaks

- qa_mod returning 0 for any input (A1 violation)
- qa_mod output outside {1,...,m}
- compute_qci non-deterministic
- output_length != input_length - 2
- float-to-int feedback in qa_mod (T2 violation)
