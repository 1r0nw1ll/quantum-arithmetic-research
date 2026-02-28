#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema


def canonical_json_bytes(obj: Any) -> bytes:
    # Stable canonical form for hash binding and diff checks.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def discriminant_short_weierstrass_mod_p(a_coeff: int, b_coeff: int, prime: int) -> int:
    # Delta = -16 * (4A^3 + 27B^2), reduced mod p.
    a3 = pow(a_coeff, 3, prime)
    b2 = pow(b_coeff, 2, prime)
    return (-16 * (4 * a3 + 27 * b2)) % prime


def count_points_fp_short_weierstrass(a_coeff: int, b_coeff: int, prime: int) -> int:
    # Deterministic brute-force count over F_p plus point at infinity.
    point_count = 1
    a_mod = a_coeff % prime
    b_mod = b_coeff % prime
    for x_val in range(prime):
        rhs = (pow(x_val, 3, prime) + a_mod * x_val + b_mod) % prime
        for y_val in range(prime):
            if (y_val * y_val - rhs) % prime == 0:
                point_count += 1
    return point_count


def ap_from_point_count(point_count_fp: int, prime: int) -> int:
    return (prime + 1) - point_count_fp


FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_RECOMPUTE_MISMATCH = "RECOMPUTE_MISMATCH"
FAIL_REDUCTION_TYPE_MISMATCH = "REDUCTION_TYPE_MISMATCH"


@dataclass
class Result:
    ok: bool
    fail_type: Optional[str] = None
    invariant_diff: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    value: Optional[Dict[str, Any]] = None


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Result]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
        return None
    except jsonschema.ValidationError as exc:
        return Result(
            ok=False,
            fail_type=FAIL_SCHEMA,
            invariant_diff={},
            details={"error": str(exc)},
        )


def recompute(cert: Dict[str, Any]) -> Dict[str, Any]:
    a_coeff = int(cert["curve"]["model"]["A"])
    b_coeff = int(cert["curve"]["model"]["B"])
    prime = int(cert["prime"])

    delta_mod_p = discriminant_short_weierstrass_mod_p(a_coeff, b_coeff, prime)
    is_good_reduction = (delta_mod_p % prime) != 0
    point_count_fp = count_points_fp_short_weierstrass(a_coeff, b_coeff, prime)
    ap_value = ap_from_point_count(point_count_fp, prime)

    return {
        "delta_mod_p": int(delta_mod_p),
        "is_good_reduction": bool(is_good_reduction),
        "point_count_fp": int(point_count_fp),
        "ap": int(ap_value),
    }


def gate_2_3_checks(cert: Dict[str, Any], recomputed: Dict[str, Any]) -> Optional[Result]:
    claimed = cert["claimed"]
    invariant_diff: Dict[str, Any] = {}

    if int(claimed["point_count_fp"]) != int(recomputed["point_count_fp"]):
        invariant_diff["point_count_fp"] = {
            "claimed": int(claimed["point_count_fp"]),
            "recomputed": int(recomputed["point_count_fp"]),
        }

    if int(claimed["ap"]) != int(recomputed["ap"]):
        invariant_diff["ap"] = {
            "claimed": int(claimed["ap"]),
            "recomputed": int(recomputed["ap"]),
        }

    if "delta_mod_p" in claimed and int(claimed["delta_mod_p"]) != int(recomputed["delta_mod_p"]):
        invariant_diff["delta_mod_p"] = {
            "claimed": int(claimed["delta_mod_p"]),
            "recomputed": int(recomputed["delta_mod_p"]),
        }

    if "is_good_reduction" in claimed and bool(claimed["is_good_reduction"]) != bool(recomputed["is_good_reduction"]):
        invariant_diff["is_good_reduction"] = {
            "claimed": bool(claimed["is_good_reduction"]),
            "recomputed": bool(recomputed["is_good_reduction"]),
        }

    if invariant_diff:
        return Result(
            ok=False,
            fail_type=FAIL_RECOMPUTE_MISMATCH,
            invariant_diff=invariant_diff,
            details={"recomputed": recomputed},
        )

    if "reduction_type" in claimed:
        reduction_type = claimed["reduction_type"]
        if recomputed["is_good_reduction"] and reduction_type != "GOOD":
            return Result(
                ok=False,
                fail_type=FAIL_REDUCTION_TYPE_MISMATCH,
                invariant_diff={"reduction_type": {"claimed": reduction_type, "expected": "GOOD"}},
                details={"recomputed": recomputed},
            )
        if (not recomputed["is_good_reduction"]) and reduction_type == "GOOD":
            return Result(
                ok=False,
                fail_type=FAIL_REDUCTION_TYPE_MISMATCH,
                invariant_diff={"reduction_type": {"claimed": reduction_type, "expected": "BAD_*"}},
                details={"recomputed": recomputed},
            )

    return None


def compute_cert_sha256(cert: Dict[str, Any]) -> str:
    return sha256_hex_bytes(canonical_json_bytes(cert))


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    schema_result = gate_1_schema(cert, schema)
    if schema_result is not None:
        return schema_result.__dict__

    recomputed = recompute(cert)
    invariant_result = gate_2_3_checks(cert, recomputed)
    if invariant_result is not None:
        return invariant_result.__dict__

    return Result(
        ok=True,
        value={
            "curve_id": cert["curve"]["curve_id"],
            "prime": int(cert["prime"]),
            "recomputed": recomputed,
            "cert_sha256": compute_cert_sha256(cert),
        },
    ).__dict__


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_self_test() -> int:
    base_dir = Path(__file__).resolve().parent
    schema_path = base_dir / "schema.json"
    pass_p5_path = base_dir / "fixtures" / "pass_good_p5.json"
    pass_p7_v1_1_path = base_dir / "fixtures" / "pass_good_p7_v1_1.json"
    fail_path = base_dir / "fixtures" / "fail_wrong_ap.json"

    schema = load_json(str(schema_path))
    pass_p5_cert = load_json(str(pass_p5_path))
    pass_p7_v1_1_cert = load_json(str(pass_p7_v1_1_path))
    fail_cert = load_json(str(fail_path))

    pass_p5_result = validate_cert(pass_p5_cert, schema)
    pass_p7_v1_1_result = validate_cert(pass_p7_v1_1_cert, schema)
    fail_result = validate_cert(fail_cert, schema)

    checks = [
        (pass_p5_result.get("ok") is True, "p5 pass fixture must validate"),
        (pass_p7_v1_1_result.get("ok") is True, "p7 v1.1 pass fixture must validate"),
        (fail_result.get("ok") is False, "fail fixture must fail"),
        (fail_result.get("fail_type") == FAIL_RECOMPUTE_MISMATCH, "fail fixture must hit RECOMPUTE_MISMATCH"),
        (
            pass_p7_v1_1_result.get("value", {}).get("recomputed", {}).get("point_count_fp") == 12,
            "p7 recompute point_count_fp must be 12",
        ),
        (
            pass_p7_v1_1_result.get("value", {}).get("recomputed", {}).get("ap") == -4,
            "p7 recompute ap must be -4",
        ),
        (
            pass_p7_v1_1_result.get("value", {}).get("recomputed", {}).get("delta_mod_p") == 2,
            "p7 recompute delta_mod_p must be 2",
        ),
        (
            pass_p7_v1_1_result.get("value", {}).get("recomputed", {}).get("is_good_reduction") is True,
            "p7 recompute is_good_reduction must be true",
        ),
    ]
    failed_checks = [msg for ok, msg in checks if not ok]

    summary = {
        "self_test_ok": len(failed_checks) == 0,
        "checks": [msg for _, msg in checks],
        "failed_checks": failed_checks,
        "pass_p5_result": pass_p5_result,
        "pass_p7_v1_1_result": pass_p7_v1_1_result,
        "fail_result": fail_result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failed_checks else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema")
    parser.add_argument("--cert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())
    if not args.schema or not args.cert:
        parser.error("either use --self-test, or provide both --schema and --cert")

    schema = load_json(args.schema)
    cert = load_json(args.cert)
    print(json.dumps(validate_cert(cert, schema), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
