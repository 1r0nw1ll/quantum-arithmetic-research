# AIID Failure Algebra Composition Label v1

This is the additive Phase-II refinement after sample-50 validation:

- Keep F1-F6 as primitive classes.
- Represent ambiguous/compound incidents with `primary` + optional `secondary`.
- Make ambiguity explicit with machine-readable `strain_witness`.

Schema:
- `qa_alphageometry_ptolemy/schemas/QA_FAILURE_ALGEBRA_COMPOSITION_LABEL.v1.schema.json`

Dataset (current 4 strain incidents):
- `qa_alphageometry_ptolemy/external_validation_data/aiid_sample50_composition_labels.v1.json`

Validator:
- `qa_alphageometry_ptolemy/external_validation_aiid_failure_algebra_composition_v1.py`

## Field Model
- `schema_id`: must be `QA_FAILURE_ALGEBRA_COMPOSITION_LABEL.v1`
- `incident_id`: incident identifier
- `primary`: one of `F1..F6`
- `secondary`: optional list of `F1..F6` (max 3)
- `composition_form`: `serial | parallel | feedback | unknown`
- `confidence`: optional `high | medium | low`
- `strain_witness`:
  - `reason`: short cause of ambiguity
  - `competing_classes`: competing F-classes
  - `missing_fields`: concrete data fields needed to disambiguate
  - `notes`: <= 280 chars

## Run
```bash
python3 qa_alphageometry_ptolemy/external_validation_aiid_failure_algebra_composition_v1.py --self-test
python3 qa_alphageometry_ptolemy/external_validation_aiid_failure_algebra_composition_v1.py
```

## Fixture Set
- PASS: `qa_alphageometry_ptolemy/external_validation_fixtures/aiid_comp_v1_PASS.json`
- FAIL (missing primary): `qa_alphageometry_ptolemy/external_validation_fixtures/aiid_comp_v1_FAIL_missing_primary.json`
- FAIL (bad secondary enum): `qa_alphageometry_ptolemy/external_validation_fixtures/aiid_comp_v1_FAIL_bad_secondary_enum.json`

## Current 4-Record Table
| Incident ID | Primary | Secondary | Form | Confidence |
|---:|---|---|---|---|
| 21 | F4 | F1 | parallel | low |
| 39 | F1 | F6 | parallel | low |
| 41 | F1 | F4 | serial | low |
| 42 | F2 | F1 | serial | low |

## Example Record
```json
{
  "schema_id": "QA_FAILURE_ALGEBRA_COMPOSITION_LABEL.v1",
  "incident_id": 39,
  "primary": "F1",
  "secondary": ["F6"],
  "composition_form": "parallel",
  "confidence": "low",
  "strain_witness": {
    "reason": "Authenticity formalization gap co-occurs with representation-isolation effects.",
    "competing_classes": ["F1", "F6"],
    "missing_fields": ["intent", "downstream_harm_trace", "deception_exposure"],
    "notes": "Deepfake introduction case is mechanism-rich but harm realization details are sparse in the record."
  }
}
```
