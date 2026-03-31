#!/usr/bin/env python3
"""
QA_PRIME_BOUNDED_CERTIFICATE_SCALING_CERT.v1 validator.

Certifies the empirical scaling claim:
for tested intervals [2, N], the observed minimal passing bounded witness cap
matches the largest prime <= sqrt(N).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


ARTIFACT_HASH_DOMAIN = "QA_PRIME_BOUNDED_CERTIFICATE_SCALING.v1"


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


def _schema_path() -> Path:
    return Path(__file__).resolve().with_name("schema.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex_text(payload: str) -> str:
    return _sha256_hex_bytes(payload.encode("utf-8"))


def _domain_sha256(domain: str, payload: str) -> str:
    return _sha256_hex_bytes(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8"))


def _compute_cert_canonical_sha256(obj: dict[str, Any]) -> str:
    clone = json.loads(_canonical_json(obj))
    clone.setdefault("digests", {})
    clone["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex_text(_canonical_json(clone))


def _compute_artifact_canonical_hash(obj: dict[str, Any]) -> str:
    clone = dict(obj)
    clone.pop("canonical_hash", None)
    return _domain_sha256(ARTIFACT_HASH_DOMAIN, _canonical_json(clone))


def _validate_schema(obj: dict[str, Any]) -> None:
    import jsonschema

    jsonschema.validate(instance=obj, schema=_load_json(_schema_path()))


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _primes_up_to(limit: int) -> list[int]:
    return [value for value in range(2, limit + 1) if _is_prime(value)]


def _largest_prime_leq(limit: int) -> Optional[int]:
    primes = _primes_up_to(limit)
    return None if not primes else primes[-1]


def validate_cert(obj: dict[str, Any]) -> list[GateResult]:
    results: list[GateResult] = []

    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as exc:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL, f"Schema invalid: {exc}"))
        return results

    declared_hash = obj.get("digests", {}).get("canonical_sha256", "")
    computed_hash = _compute_cert_canonical_sha256(obj)
    if declared_hash == "0" * 64:
        results.append(
            GateResult(
                "gate_2_canonical_hash",
                GateStatus.FAIL,
                "canonical_sha256 is placeholder",
                {"computed": computed_hash},
            )
        )
        return results
    if declared_hash != computed_hash:
        results.append(
            GateResult(
                "gate_2_canonical_hash",
                GateStatus.FAIL,
                "canonical_sha256 mismatch",
                {"declared": declared_hash, "computed": computed_hash},
            )
        )
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS, "canonical_sha256 matches"))

    artifact_path = Path(obj["evidence"]["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = _repo_root() / artifact_path
    if not artifact_path.exists():
        results.append(
            GateResult(
                "gate_3_artifact_integrity",
                GateStatus.FAIL,
                f"Referenced artifact does not exist: {obj['evidence']['artifact_path']}",
            )
        )
        return results

    artifact = _load_json(artifact_path)
    artifact_declared_hash = artifact.get("canonical_hash", "")
    artifact_computed_hash = _compute_artifact_canonical_hash(artifact)
    if artifact_declared_hash != artifact_computed_hash:
        results.append(
            GateResult(
                "gate_3_artifact_integrity",
                GateStatus.FAIL,
                "Artifact canonical_hash mismatch",
                {
                    "artifact_path": obj["evidence"]["artifact_path"],
                    "declared": artifact_declared_hash,
                    "computed": artifact_computed_hash,
                },
            )
        )
        return results
    if obj["evidence"]["artifact_canonical_hash"] != artifact_declared_hash:
        results.append(
            GateResult(
                "gate_3_artifact_integrity",
                GateStatus.FAIL,
                "Cert artifact_canonical_hash does not match artifact canonical_hash",
                {
                    "cert": obj["evidence"]["artifact_canonical_hash"],
                    "artifact": artifact_declared_hash,
                },
            )
        )
        return results

    mismatches: dict[str, Any] = {}
    evidence = obj["evidence"]
    for key in ("experiment_id", "hypothesis", "success_criteria"):
        if evidence[key] != artifact.get(key):
            mismatches[key] = {"cert": evidence[key], "artifact": artifact.get(key)}
    if obj["result"] != artifact.get("result"):
        mismatches["result"] = {"cert": obj["result"], "artifact": artifact.get("result")}
    if obj["rows"] != artifact.get("rows"):
        mismatches["rows"] = {"cert": obj["rows"], "artifact": artifact.get("rows")}
    if mismatches:
        results.append(
            GateResult(
                "gate_3_artifact_integrity",
                GateStatus.FAIL,
                "Cert fields do not match referenced artifact",
                mismatches,
            )
        )
        return results
    results.append(
        GateResult(
            "gate_3_artifact_integrity",
            GateStatus.PASS,
            "Referenced artifact exists, hashes cleanly, and matches cert fields",
            {"artifact_path": obj["evidence"]["artifact_path"]},
        )
    )

    rows = obj["rows"]
    ends = [row["end"] for row in rows]
    if ends != sorted(ends) or len(set(ends)) != len(ends):
        results.append(
            GateResult(
                "gate_4_endpoint_ordering",
                GateStatus.FAIL,
                "Endpoints must be strictly increasing and unique",
                {"ends": ends},
            )
        )
        return results
    results.append(
        GateResult(
            "gate_4_endpoint_ordering",
            GateStatus.PASS,
            f"{len(ends)} endpoints are strictly increasing and unique",
            {"ends": ends},
        )
    )

    row_issues = []
    for row in rows:
        end = row["end"]
        sqrt_floor = math.isqrt(end)
        expected_caps = _primes_up_to(max(2, sqrt_floor))
        expected_predicted = _largest_prime_leq(sqrt_floor)
        observed = row["observed_minimal_pass_prime_max"]
        if row["sqrt_floor"] != sqrt_floor:
            row_issues.append({"end": end, "field": "sqrt_floor", "declared": row["sqrt_floor"], "expected": sqrt_floor})
        if row["candidate_caps"] != expected_caps:
            row_issues.append({"end": end, "field": "candidate_caps", "declared": row["candidate_caps"], "expected": expected_caps})
        if row["predicted_largest_prime_leq_sqrt_end"] != expected_predicted:
            row_issues.append(
                {
                    "end": end,
                    "field": "predicted_largest_prime_leq_sqrt_end",
                    "declared": row["predicted_largest_prime_leq_sqrt_end"],
                    "expected": expected_predicted,
                }
            )
        if observed not in expected_caps:
            row_issues.append(
                {
                    "end": end,
                    "field": "observed_minimal_pass_prime_max",
                    "declared": observed,
                    "expected_membership": expected_caps,
                }
            )
    if row_issues:
        results.append(
            GateResult(
                "gate_5_row_recomputation",
                GateStatus.FAIL,
                "One or more rows disagree with recomputed predictor inputs",
                {"issues": row_issues},
            )
        )
        return results
    results.append(
        GateResult(
            "gate_5_row_recomputation",
            GateStatus.PASS,
            "All rows agree with recomputed sqrt-floor, candidate caps, and predicted bound",
        )
    )

    flag_issues = []
    for row in rows:
        expected_match = row["observed_minimal_pass_prime_max"] == row["predicted_largest_prime_leq_sqrt_end"]
        if row["matches_prediction"] != expected_match:
            flag_issues.append({"end": row["end"], "declared": row["matches_prediction"], "expected": expected_match})
    if flag_issues:
        results.append(
            GateResult(
                "gate_6_match_flags",
                GateStatus.FAIL,
                "matches_prediction flags are not honest summaries of row data",
                {"issues": flag_issues},
            )
        )
        return results
    results.append(
        GateResult(
            "gate_6_match_flags",
            GateStatus.PASS,
            "matches_prediction flags agree with observed vs predicted values on every row",
        )
    )

    expected_result = "PASS" if all(row["matches_prediction"] for row in rows) else "FAIL"
    if obj["result"] != expected_result:
        results.append(
            GateResult(
                "gate_7_result_consistency",
                GateStatus.FAIL,
                "result does not honestly summarize the row-level outcomes",
                {"declared": obj["result"], "expected": expected_result},
            )
        )
        return results
    results.append(
        GateResult(
            "gate_7_result_consistency",
            GateStatus.PASS,
            f"result={obj['result']} honestly summarizes the row-level outcomes",
        )
    )

    return results


def _report_ok(results: list[GateResult]) -> bool:
    return all(result.status == GateStatus.PASS for result in results)


def _print_human(results: list[GateResult]) -> None:
    for result in results:
        print(f"[{result.status.value}] {result.gate}: {result.message}")


def _print_json(results: list[GateResult]) -> None:
    print(json.dumps({"ok": _report_ok(results), "results": [result.to_dict() for result in results]}, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    expected = {
        "pass_scaling_100_1000.json": True,
        "fail_scaling_100_500.json": True,
    }
    fixture_rows = []
    overall_ok = True
    for name, expected_ok in expected.items():
        path = fixtures_dir / name
        results = validate_cert(_load_json(path))
        ok = _report_ok(results)
        if ok != expected_ok:
            overall_ok = False
        fixture_rows.append(
            {
                "fixture": name,
                "ok": ok,
                "expected_ok": expected_ok,
                "result": _load_json(path)["result"],
                "gates": [result.to_dict() for result in results],
            }
        )
    payload = {"ok": overall_ok, "fixtures": fixture_rows}
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in fixture_rows:
            print(f"[{'PASS' if row['ok'] else 'FAIL'}] {row['fixture']} expected_ok={row['expected_ok']}")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA prime bounded certificate scaling certs.")
    parser.add_argument("--file", type=Path, help="Validate one cert file")
    parser.add_argument("--self-test", action="store_true", help="Run fixture self-test")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    if args.self_test:
        return self_test(args.json)
    if args.file:
        results = validate_cert(_load_json(args.file))
        if args.json:
            _print_json(results)
        else:
            _print_human(results)
        return 0 if _report_ok(results) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
