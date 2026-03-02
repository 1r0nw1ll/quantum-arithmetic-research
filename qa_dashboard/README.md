# QA Auditor Dashboard (Prototype)

Minimal web UI to validate QA artifacts with a non-technical PASS/FAIL view.

## Run

```bash
python -m uvicorn qa_dashboard.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## What It Does

- Upload a JSON artifact (certificate or bundle)
- Select a validator
- Shows PASS/FAIL + a SHA-256 hash + a saved run folder
- Persists `input.json` + `result.json` under `QA_DASHBOARD_RUNS_DIR` (default: `/tmp/qa_dashboard_runs`)
- Appends a hash-chained audit log entry to `QA_AUDIT_LEDGER_PATH` (default: `/tmp/qa_dashboard_runs/audit_ledger.jsonl`)

## Validators Included

- `Decision Spine (qa_verify.py)` — `qa_alphageometry_ptolemy/qa_verify.py`
- `Mapping Protocol v1` — `qa_mapping_protocol/validator.py`
- `Fairness: Demographic Parity v1` — `qa_fairness_demographic_parity_cert_v1/validator.py`

## Configure

- `QA_DASHBOARD_RUNS_DIR=/some/path` (optional)
- `QA_AUDIT_LEDGER_PATH=/some/path/audit_ledger.jsonl` (optional)
