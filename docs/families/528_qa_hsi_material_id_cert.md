# Family [528]: QA HSI Material Identification Cert

**Schema:** `QA_HSI_MATERIAL_ID_CERT.v1`  
**Root:** `qa_alphageometry_ptolemy/qa_hsi_material_id_cert_v1/`  
**Validator:** `qa_hsi_material_id_cert_validate.py --self-test`  
**Status:** Active; self-test passing; added to `FAMILY_SWEEPS`

## Purpose

This cert family gates hyperspectral imaging claims that identify materials or chemical-signature classes rather than ordinary land-cover labels. It validates the claim surface created by `tools/qa_hsi_material_identification.py`: wavelength grid, observer quantization boundary, material library hash, diagnostic absorption witnesses, mixture coverage, target detection, abundance-bin metrics, unknown rejection, and honest synthetic-vs-real scope.

The family exists to prevent the next roadmap step, real spectral-library ingestion with source hashes and sensor wavelength resampling, from becoming a loose experiment.

## Schema

| Field | Meaning |
|---|---|
| `schema_version` | Must be `QA_HSI_MATERIAL_ID_CERT.v1` |
| `cert_type` | Must be `qa_hsi_material_id_cert` |
| `source_type` | One of `synthetic`, `real_spectral_library`, `sensor_calibrated` |
| `scope_claim` | Declares whether real-data validation is claimed |
| `observer_boundary` | Declares one-time quantization and integer-only QA layer |
| `wavelength_grid_nm` | Positive ascending integer wavelength grid |
| `material_library` | Material spectra plus diagnostic absorption-band witnesses |
| `hashes` | Domain-separated SHA-256 hashes for grid and library |
| `sample_counts` | Pure, mixture, unknown, and known sample counts |
| `success_criteria` | Predeclared metric thresholds |
| `metrics` | Observed pure, mixture, target, abundance, and unknown-rejection metrics |
| `thresholds` | Unknown-rejection threshold and target rule |
| `result` | `PASS` or `FAIL` |
| `fail_ledger` | Declared failure ledger for failing certs |

## Validator Checks

| Gate | Check |
|---|---|
| Schema | Required top-level fields and fixed schema/type values |
| Source scope | Synthetic certs cannot claim real-data validation |
| Observer boundary | `quantized_once=true`, `qa_layer_integer_only=true`, nonempty quantization text |
| Wavelength grid | Nonempty positive ascending integer wavelengths |
| Material library | Every spectrum is 8-bit integer and matches grid length |
| Absorption witnesses | Every material declares absorption bands on the wavelength grid |
| Hashes | Recomputes `QA_HSI_WAVELENGTH_GRID.v1` and `QA_HSI_MATERIAL_LIBRARY.v1` domain-separated hashes |
| Sample coverage | Pure, mixture, unknown, and known counts are positive |
| Metrics | Pure top-1, mixture top-1, target F1, abundance-bin accuracy, and unknown FPR meet declared thresholds |
| Thresholds | Unknown distance threshold and target rule are declared |

## Fixtures

| Fixture | Expected | Purpose |
|---|---|---|
| `hmi_pass_synthetic_material_id.json` | PASS | Synthetic material-ID cert with valid hashes, metrics, and unknown-rejection evidence |
| `hmi_fail_bad_library_hash.json` | FAIL | Detects material-library hash mismatch |
| `hmi_fail_missing_unknown_rejection.json` | FAIL | Detects missing unknown rejection evidence and zero unknown samples |
| `hmi_fail_overclaim_real_data.json` | FAIL | Detects synthetic cert overclaiming real-data validation |

## Family Relationships

This family follows the HSI benchmark and correction work but changes the claim type:

- `tools/qa_hsi_material_identification.py`: generated the synthetic/library-first PASS benchmark.
- `results/qa_hsi_material_identification_001/`: contains the report and result JSON.
- Existing Indian Pines, Salinas, Pavia, and Houston HSI work remains land-cover or correction-focused.
- This family is the gate before real spectral-library ingestion for mineral, vegetation, and polymer spectra.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_hsi_material_id_cert_v1/qa_hsi_material_id_cert_validate.py --self-test
```

Expected result: `{"ok":true,...}` with one passing fixture and three correctly rejected fail fixtures.
