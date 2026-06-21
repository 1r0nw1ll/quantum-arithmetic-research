#!/usr/bin/env python3
"""Validate cert [492] from a hash-bound, network-free result artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


QA_COMPLIANCE = (
    "cert_validator -- hash-bound Open-Meteo ERA5 result artifact; "
    "anomaly rank bins floor(rank*27/N); a=b+2e derived raw; "
    "network-free deterministic self-test"
)

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = ROOT / "artifacts" / "certified_result.v1.json"
NEGATIVE_FIXTURE = ROOT / "fixtures" / "temperature_fail_tampered_hash.json"
SCHEMA_ID = "QA_WITT_TOWER_TEMPERATURE_PERSISTENCE_RESULT.v1"
HASH_DOMAIN = "qa.witt_tower.temperature_persistence.certified_result.v1"
EXPECTED_STATIONS = ("Chicago", "Minneapolis", "Seattle", "Miami")
RIVER_RATIO = Decimal("2.69")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def domain_hash(domain: str, value: object) -> str:
    payload = domain.encode("utf-8") + b"\x00" + canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"expected decimal-compatible value, got {value!r}")
    return Decimal(str(value))


def artifact_body(artifact: dict[str, Any]) -> dict[str, Any]:
    body = dict(artifact)
    body.pop("artifact_sha256", None)
    return body


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []

    if artifact.get("schema_id") != SCHEMA_ID:
        errors.append("SCHEMA_ID_MISMATCH")

    expected_hash = domain_hash(HASH_DOMAIN, artifact_body(artifact))
    if artifact.get("artifact_sha256") != expected_hash:
        errors.append("ARTIFACT_HASH_MISMATCH")

    source = artifact.get("source")
    if not isinstance(source, dict):
        errors.append("SOURCE_METADATA_MISSING")
        source = {}
    required_source = {
        "provider": "Open-Meteo",
        "dataset": "ERA5 historical archive",
        "endpoint": "https://archive-api.open-meteo.com/v1/archive",
        "start_date": "2000-01-01",
        "end_date": "2025-12-31",
        "variable": "temperature_2m_max",
        "timezone": "UTC",
    }
    if any(source.get(key) != value for key, value in required_source.items()):
        errors.append("SOURCE_METADATA_MISMATCH")

    requests = source.get("requests")
    if not isinstance(requests, list) or len(requests) != 4:
        errors.append("SOURCE_REQUEST_SET_INVALID")
        requests = []
    request_names = {
        row.get("station")
        for row in requests
        if isinstance(row, dict)
    }
    if request_names != set(EXPECTED_STATIONS):
        errors.append("SOURCE_REQUEST_SET_INVALID")

    method = artifact.get("method")
    expected_method = {
        "modulus": 27,
        "signal_rule": "b + 2*e <= 6",
        "target_offset_days": 2,
        "monthly_deseasonalization": True,
        "permutation_count": 5000,
        "seed": 42,
    }
    if method != expected_method:
        errors.append("METHOD_METADATA_MISMATCH")

    results = artifact.get("results")
    if not isinstance(results, dict) or set(results) != set(EXPECTED_STATIONS):
        errors.append("STATION_RESULT_SET_INVALID")
        results = {}

    total_signal = 0
    total_expected = Decimal(0)
    weighted_excess = Decimal(0)
    negative_count = 0
    all_positive_autocorr = True
    all_significant = True

    for station in EXPECTED_STATIONS:
        row = results.get(station)
        if not isinstance(row, dict):
            continue
        try:
            n_days = row["n_days"]
            n_signal = row["n_signal"]
            n_expected = decimal(row["n_expected"])
            autocorr = decimal(row["autocorr_lev"])
            excess = decimal(row["signal_excess_c"])
            persistence_p = decimal(row["persistence_p"])
        except (KeyError, ValueError):
            errors.append(f"STATION_RESULT_INVALID:{station}")
            continue
        if (
            isinstance(n_days, bool)
            or not isinstance(n_days, int)
            or n_days < 500
            or isinstance(n_signal, bool)
            or not isinstance(n_signal, int)
            or n_signal <= 0
            or n_expected <= 0
        ):
            errors.append(f"STATION_RESULT_INVALID:{station}")
            continue
        total_signal += n_signal
        total_expected += n_expected
        weighted_excess += Decimal(n_signal) * excess
        negative_count += int(excess < 0)
        all_positive_autocorr = all_positive_autocorr and autocorr > 0
        all_significant = all_significant and persistence_p < Decimal("0.001")

    aggregates = artifact.get("aggregates")
    if not isinstance(aggregates, dict):
        errors.append("AGGREGATES_MISSING")
        aggregates = {}
    if total_signal > 0 and total_expected > 0:
        recomputed_ratio = Decimal(total_signal) / total_expected
        recomputed_excess = weighted_excess / Decimal(total_signal)
        aggregate_checks = {
            "pooled_n_signal": aggregates.get("pooled_n_signal") == total_signal,
            "pooled_n_expected": decimal(aggregates.get("pooled_n_expected", 0))
            == total_expected,
            "pooled_n_signal_ratio": abs(
                decimal(aggregates.get("pooled_n_signal_ratio", 0))
                - recomputed_ratio
            )
            <= Decimal("0.001"),
            "pooled_excess_c": abs(
                decimal(aggregates.get("pooled_excess_c", 0))
                - recomputed_excess
            )
            <= Decimal("0.001"),
            "n_negative": aggregates.get("n_negative") == negative_count,
            "all_autocorr_positive": aggregates.get("all_autocorr_positive")
            is all_positive_autocorr,
        }
        if not all(aggregate_checks.values()):
            errors.append("AGGREGATE_RECOMPUTE_MISMATCH")
    else:
        recomputed_ratio = Decimal(0)
        recomputed_excess = Decimal(0)

    claim_checks = {
        "C1_all_autocorr_positive": all_positive_autocorr,
        "C2_pooled_excess_lt_neg1c": recomputed_excess < Decimal("-1"),
        "C3_n_negative_eq4": negative_count == 4,
        "C4_all_pers_p_lt001": all_significant,
        "C5_pooled_ratio_gt3": recomputed_ratio > Decimal("3"),
        "C6_ratio_exceeds_river": recomputed_ratio > RIVER_RATIO,
    }
    if not all(claim_checks.values()):
        errors.append("CLAIM_CHECK_FAILED")

    return {
        "ok": not errors,
        "errors": errors,
        "artifact_sha256": artifact.get("artifact_sha256"),
        "recomputed_sha256": expected_hash,
        "claim_checks": claim_checks,
    }


def apply_mutation(value: dict[str, Any], path: str, replacement: object) -> None:
    parts = path.split(".")
    current: Any = value
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = replacement


def self_test() -> dict[str, Any]:
    valid = validate_artifact(load_json(DEFAULT_ARTIFACT))
    fixture = load_json(NEGATIVE_FIXTURE)
    tampered = copy.deepcopy(load_json(DEFAULT_ARTIFACT))
    mutation = fixture.get("mutation", {})
    apply_mutation(
        tampered,
        str(mutation.get("path", "")),
        mutation.get("value"),
    )
    invalid = validate_artifact(tampered)
    expected_fail = fixture.get("expected_fail_type")
    checks = [
        {"name": "valid artifact accepted", "ok": valid["ok"]},
        {
            "name": "tampered artifact rejected",
            "ok": not invalid["ok"] and expected_fail in invalid["errors"],
        },
    ]
    return {"ok": all(row["ok"] for row in checks), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    result = self_test() if args.self_test else validate_artifact(load_json(args.artifact))
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
