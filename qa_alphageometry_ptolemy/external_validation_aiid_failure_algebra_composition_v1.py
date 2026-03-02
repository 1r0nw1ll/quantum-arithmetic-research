#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA_PATH = REPO_ROOT / "qa_alphageometry_ptolemy" / "schemas" / "QA_FAILURE_ALGEBRA_COMPOSITION_LABEL.v1.schema.json"
DEFAULT_LABELS_PATH = REPO_ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_sample50_composition_labels.v1.json"
DEFAULT_INCIDENTS_PATH = REPO_ROOT / "qa_alphageometry_ptolemy" / "external_validation_data" / "aiid_sample50_incidents.json"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "qa_alphageometry_ptolemy" / "external_validation_certs" / "aiid_failure_algebra_composition_v1_summary.json"
FIXTURE_DIR = REPO_ROOT / "qa_alphageometry_ptolemy" / "external_validation_fixtures"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload
    raise TypeError("labels payload must be an object or array")


def validate_labels(
    labels_path: Path,
    schema_path: Path,
    incidents_path: Path,
) -> dict[str, Any]:
    schema = _load_json(schema_path)
    payload = _load_json(labels_path)
    records = _normalize_records(payload)
    incidents = _load_json(incidents_path)
    known_incident_ids = {int(row["incident_id"]) for row in incidents}

    validator = jsonschema.Draft202012Validator(schema)
    errors: list[dict[str, Any]] = []

    seen_ids: set[int] = set()
    duplicate_ids: list[int] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(
                {
                    "index": idx,
                    "type": "type_error",
                    "message": "record is not a JSON object",
                }
            )
            continue

        schema_errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
        for err in schema_errors:
            errors.append(
                {
                    "index": idx,
                    "type": "schema_error",
                    "message": err.message,
                    "path": "/".join(str(p) for p in err.path),
                }
            )

        incident_id = record.get("incident_id")
        if isinstance(incident_id, int):
            if incident_id not in known_incident_ids:
                errors.append(
                    {
                        "index": idx,
                        "type": "incident_lookup_error",
                        "message": f"incident_id {incident_id} not found in sample incidents",
                    }
                )
            if incident_id in seen_ids:
                duplicate_ids.append(incident_id)
            else:
                seen_ids.add(incident_id)

    summary = {
        "ok": len(errors) == 0,
        "labels_path": str(labels_path.relative_to(REPO_ROOT)),
        "schema_path": str(schema_path.relative_to(REPO_ROOT)),
        "incidents_path": str(incidents_path.relative_to(REPO_ROOT)),
        "record_count": len(records),
        "known_incident_count": len(known_incident_ids),
        "unique_incident_ids": len(duplicate_ids) == 0,
        "duplicate_incident_ids": sorted(set(duplicate_ids)),
        "error_count": len(errors),
        "errors": errors,
    }
    return summary


def run_self_test() -> int:
    pass_path = FIXTURE_DIR / "aiid_comp_v1_PASS.json"
    fail_missing_primary = FIXTURE_DIR / "aiid_comp_v1_FAIL_missing_primary.json"
    fail_bad_secondary = FIXTURE_DIR / "aiid_comp_v1_FAIL_bad_secondary_enum.json"

    pass_summary = validate_labels(pass_path, DEFAULT_SCHEMA_PATH, DEFAULT_INCIDENTS_PATH)
    fail_missing_summary = validate_labels(fail_missing_primary, DEFAULT_SCHEMA_PATH, DEFAULT_INCIDENTS_PATH)
    fail_secondary_summary = validate_labels(fail_bad_secondary, DEFAULT_SCHEMA_PATH, DEFAULT_INCIDENTS_PATH)
    dataset_summary = validate_labels(DEFAULT_LABELS_PATH, DEFAULT_SCHEMA_PATH, DEFAULT_INCIDENTS_PATH)

    checks = [
        (pass_summary["ok"] is True, "PASS fixture should validate"),
        (fail_missing_summary["ok"] is False, "FAIL missing-primary fixture should fail"),
        (fail_secondary_summary["ok"] is False, "FAIL bad-secondary fixture should fail"),
        (dataset_summary["ok"] is True, "sample50 composition labels should validate"),
    ]
    failed = [msg for ok, msg in checks if not ok]

    result = {
        "self_test_ok": len(failed) == 0,
        "checks": [msg for _, msg in checks],
        "failed_checks": failed,
        "fixtures": {
            "pass": pass_summary,
            "fail_missing_primary": fail_missing_summary,
            "fail_bad_secondary_enum": fail_secondary_summary,
            "sample50": dataset_summary,
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if len(failed) == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AIID composition labels (failure algebra v1)")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--incidents", default=str(DEFAULT_INCIDENTS_PATH))
    parser.add_argument("--out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    summary = validate_labels(Path(args.labels), Path(args.schema), Path(args.incidents))
    _dump_json(Path(args.out), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
