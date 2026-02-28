#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def discriminant_short_weierstrass_mod_p(a_coeff: int, b_coeff: int, prime: int) -> int:
    a3 = pow(a_coeff, 3, prime)
    b2 = pow(b_coeff, 2, prime)
    return (-16 * (4 * a3 + 27 * b2)) % prime


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = math.isqrt(n)
    factor = 3
    while factor <= limit:
        if n % factor == 0:
            return False
        factor += 2
    return True


def count_points_fp_short_weierstrass_p2(a_coeff: int, b_coeff: int) -> int:
    point_count = 1
    for x_val in range(2):
        rhs = (x_val ** 3 + a_coeff * x_val + b_coeff) % 2
        for y_val in range(2):
            if (y_val * y_val - rhs) % 2 == 0:
                point_count += 1
    return point_count


def legendre_symbol(rhs: int, prime: int) -> int:
    if rhs % prime == 0:
        return 0
    value = pow(rhs, (prime - 1) // 2, prime)
    if value == 1:
        return 1
    if value == prime - 1:
        return -1
    # For odd primes this branch should be unreachable.
    return 0


def count_points_fp_short_weierstrass(a_coeff: int, b_coeff: int, prime: int) -> int:
    if not is_prime(prime):
        raise ValueError(f"prime field modulus must be prime, got {prime}")

    if prime == 2:
        return count_points_fp_short_weierstrass_p2(a_coeff, b_coeff)

    # O(p) point count using Euler's criterion / Legendre symbol.
    point_count = 1
    a_mod = a_coeff % prime
    b_mod = b_coeff % prime
    for x_val in range(prime):
        rhs = (pow(x_val, 3, prime) + a_mod * x_val + b_mod) % prime
        ls = legendre_symbol(rhs, prime)
        if ls == 0:
            point_count += 1
        elif ls == 1:
            point_count += 2
    return point_count


def ap_from_point_count(point_count_fp: int, prime: int) -> int:
    return (prime + 1) - point_count_fp


FAIL_SCHEMA = "SCHEMA_INVALID"
FAIL_RECOMPUTE_MISMATCH = "RECOMPUTE_MISMATCH"
FAIL_MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
FAIL_DUPLICATE_PRIME = "DUPLICATE_PRIME"
FAIL_NON_PRIME_INPUT = "NON_PRIME_INPUT"
FAIL_TRACE_LENGTH_MISMATCH = "TRACE_LENGTH_MISMATCH"
FAIL_TRACE_MONOTONICITY_VIOLATION = "TRACE_MONOTONICITY_VIOLATION"
FAIL_TRACE_INCONSISTENT = "TRACE_INCONSISTENT"


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
        return Result(ok=False, fail_type=FAIL_SCHEMA, invariant_diff={}, details={"error": str(exc)})


def recompute_for_prime(a_coeff: int, b_coeff: int, prime: int) -> Dict[str, Any]:
    delta_mod_p = discriminant_short_weierstrass_mod_p(a_coeff, b_coeff, prime)
    is_good_reduction = (delta_mod_p % prime) != 0
    point_count_fp = count_points_fp_short_weierstrass(a_coeff, b_coeff, prime)
    ap_val = ap_from_point_count(point_count_fp, prime)
    return {
        "point_count_fp": int(point_count_fp),
        "ap": int(ap_val),
        "delta_mod_p": int(delta_mod_p),
        "is_good_reduction": bool(is_good_reduction),
    }


def normalize_claimed_record_hashes(record_hashes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = [{"prime": int(item["prime"]), "sha256": str(item["sha256"]).lower()} for item in record_hashes]
    return sorted(normalized, key=lambda item: item["prime"])


def build_manifest(curve_model: Dict[str, Any], recomputed_by_prime: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    a_coeff = int(curve_model["A"])
    b_coeff = int(curve_model["B"])
    record_hashes: List[Dict[str, Any]] = []
    for prime in sorted(recomputed_by_prime):
        payload = {
            "curve": {"A": a_coeff, "B": b_coeff},
            "prime": int(prime),
            "recomputed": recomputed_by_prime[prime],
        }
        record_hashes.append({"prime": int(prime), "sha256": sha256_hex_bytes(canonical_json_bytes(payload))})
    batch_sha256 = sha256_hex_bytes(canonical_json_bytes(record_hashes))
    return {"record_hashes": record_hashes, "batch_sha256": batch_sha256}


def gate_2_recompute_checks(source_batch: Dict[str, Any], recomputed_by_prime: Dict[int, Dict[str, Any]]) -> Optional[Result]:
    primes = [int(entry["prime"]) for entry in source_batch["records"]]
    if len(set(primes)) != len(primes):
        duplicates = sorted({prime for prime in primes if primes.count(prime) > 1})
        return Result(ok=False, fail_type=FAIL_DUPLICATE_PRIME, invariant_diff={"duplicate_primes": duplicates}, details={})

    invariant_diff: Dict[str, Any] = {}
    for entry in source_batch["records"]:
        prime = int(entry["prime"])
        claimed = entry["claimed"]
        recomputed = recomputed_by_prime[prime]
        prefix = f"prime_{prime}"

        if int(claimed["point_count_fp"]) != int(recomputed["point_count_fp"]):
            invariant_diff[f"{prefix}.point_count_fp"] = {"claimed": int(claimed["point_count_fp"]), "recomputed": int(recomputed["point_count_fp"])}
        if int(claimed["ap"]) != int(recomputed["ap"]):
            invariant_diff[f"{prefix}.ap"] = {"claimed": int(claimed["ap"]), "recomputed": int(recomputed["ap"])}

        if "delta_mod_p" in claimed and int(claimed["delta_mod_p"]) != int(recomputed["delta_mod_p"]):
            invariant_diff[f"{prefix}.delta_mod_p"] = {"claimed": int(claimed["delta_mod_p"]), "recomputed": int(recomputed["delta_mod_p"])}
        if "is_good_reduction" in claimed and bool(claimed["is_good_reduction"]) != bool(recomputed["is_good_reduction"]):
            invariant_diff[f"{prefix}.is_good_reduction"] = {"claimed": bool(claimed["is_good_reduction"]), "recomputed": bool(recomputed["is_good_reduction"])}
        if "reduction_type" in claimed:
            reduction_type = str(claimed["reduction_type"])
            if recomputed["is_good_reduction"] and reduction_type != "GOOD":
                invariant_diff[f"{prefix}.reduction_type"] = {"claimed": reduction_type, "expected": "GOOD"}
            if (not recomputed["is_good_reduction"]) and reduction_type == "GOOD":
                invariant_diff[f"{prefix}.reduction_type"] = {"claimed": reduction_type, "expected": "BAD_*"}

    if invariant_diff:
        return Result(ok=False, fail_type=FAIL_RECOMPUTE_MISMATCH, invariant_diff=invariant_diff, details={"recomputed_by_prime": recomputed_by_prime})
    return None


def gate_3_manifest_checks(source_batch: Dict[str, Any], recomputed_manifest: Dict[str, Any]) -> Optional[Result]:
    claimed_manifest = source_batch["claimed_manifest"]
    claimed_record_hashes = normalize_claimed_record_hashes(claimed_manifest["record_hashes"])
    recomputed_record_hashes = normalize_claimed_record_hashes(recomputed_manifest["record_hashes"])
    claimed_batch_sha256 = str(claimed_manifest["batch_sha256"]).lower()
    recomputed_batch_sha256 = str(recomputed_manifest["batch_sha256"]).lower()

    invariant_diff: Dict[str, Any] = {}
    if claimed_record_hashes != recomputed_record_hashes:
        invariant_diff["record_hashes"] = {"claimed": claimed_record_hashes, "recomputed": recomputed_record_hashes}
    if claimed_batch_sha256 != recomputed_batch_sha256:
        invariant_diff["batch_sha256"] = {"claimed": claimed_batch_sha256, "recomputed": recomputed_batch_sha256}

    if invariant_diff:
        return Result(ok=False, fail_type=FAIL_MANIFEST_MISMATCH, invariant_diff=invariant_diff, details={"recomputed_manifest": recomputed_manifest})
    return None


def build_exact_proxy(source_batch: Dict[str, Any], recomputed_by_prime: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    primes_sorted = sorted(int(entry["prime"]) for entry in source_batch["records"])
    numerator_factors = [int(recomputed_by_prime[prime]["point_count_fp"]) for prime in primes_sorted]
    denominator_factors = [int(prime) for prime in primes_sorted]

    numerator = 1
    for value in numerator_factors:
        numerator *= value
    denominator = 1
    for value in denominator_factors:
        denominator *= value

    return {
        "numerator_factors": numerator_factors,
        "denominator_factors": denominator_factors,
        "numerator": int(numerator),
        "denominator": int(denominator),
    }


def gate_4_proxy_checks(claimed_proxy: Dict[str, Any], recomputed_proxy: Dict[str, Any]) -> Optional[Result]:
    invariant_diff: Dict[str, Any] = {}
    for key in ["numerator_factors", "denominator_factors", "numerator", "denominator"]:
        if claimed_proxy.get(key) != recomputed_proxy.get(key):
            invariant_diff[key] = {"claimed": claimed_proxy.get(key), "recomputed": recomputed_proxy.get(key)}
    if invariant_diff:
        return Result(ok=False, fail_type=FAIL_RECOMPUTE_MISMATCH, invariant_diff=invariant_diff, details={"recomputed_proxy": recomputed_proxy})
    return None


def gate_5_trace_checks(rank_trace: Dict[str, Any]) -> Optional[Result]:
    lower_bounds = [int(value) for value in rank_trace["lower_bounds"]]
    upper_bounds = [int(value) for value in rank_trace["upper_bounds"]]
    squeeze_closed = bool(rank_trace["squeeze_closed"])

    if len(lower_bounds) != len(upper_bounds):
        return Result(
            ok=False,
            fail_type=FAIL_TRACE_LENGTH_MISMATCH,
            invariant_diff={"trace_lengths": {"lower": len(lower_bounds), "upper": len(upper_bounds)}},
            details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
        )

    for idx in range(1, len(lower_bounds)):
        if lower_bounds[idx] < lower_bounds[idx - 1]:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_MONOTONICITY_VIOLATION,
                invariant_diff={
                    "lower_bounds": {
                        "index": idx,
                        "previous": lower_bounds[idx - 1],
                        "current": lower_bounds[idx],
                    }
                },
                details={"lower_bounds": lower_bounds},
            )
    for idx in range(1, len(upper_bounds)):
        if upper_bounds[idx] > upper_bounds[idx - 1]:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_MONOTONICITY_VIOLATION,
                invariant_diff={
                    "upper_bounds": {
                        "index": idx,
                        "previous": upper_bounds[idx - 1],
                        "current": upper_bounds[idx],
                    }
                },
                details={"upper_bounds": upper_bounds},
            )

    equality_steps: List[int] = []
    for idx, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
        if low > high:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_INCONSISTENT,
                invariant_diff={"interval_crossing": {"index": idx, "lower": low, "upper": high}},
                details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
            )
        if low == high:
            equality_steps.append(idx)

    if squeeze_closed:
        if not equality_steps:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_INCONSISTENT,
                invariant_diff={"squeeze_closed": {"claimed": True, "observed": False}},
                details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
            )
        first_close_step = equality_steps[0]
        if "closing_step" in rank_trace and int(rank_trace["closing_step"]) != first_close_step:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_INCONSISTENT,
                invariant_diff={
                    "closing_step": {
                        "claimed": int(rank_trace["closing_step"]),
                        "recomputed": first_close_step,
                    }
                },
                details={"equality_steps": equality_steps},
            )
        if "final_rank" in rank_trace:
            final_rank = int(rank_trace["final_rank"])
            if lower_bounds[-1] != upper_bounds[-1] or final_rank != lower_bounds[-1]:
                return Result(
                    ok=False,
                    fail_type=FAIL_TRACE_INCONSISTENT,
                    invariant_diff={
                        "final_rank": {
                            "claimed": final_rank,
                            "recomputed": lower_bounds[-1],
                            "upper_final": upper_bounds[-1],
                        }
                    },
                    details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
                )
    else:
        if equality_steps:
            return Result(
                ok=False,
                fail_type=FAIL_TRACE_INCONSISTENT,
                invariant_diff={"squeeze_closed": {"claimed": False, "observed": True, "equality_steps": equality_steps}},
                details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
            )
        if "final_rank" in rank_trace:
            final_rank = int(rank_trace["final_rank"])
            if final_rank < lower_bounds[-1] or final_rank > upper_bounds[-1]:
                return Result(
                    ok=False,
                    fail_type=FAIL_TRACE_INCONSISTENT,
                    invariant_diff={
                        "final_rank_interval": {
                            "claimed": final_rank,
                            "lower_final": lower_bounds[-1],
                            "upper_final": upper_bounds[-1],
                        }
                    },
                    details={"lower_bounds": lower_bounds, "upper_bounds": upper_bounds},
                )

    return None


def build_rank_summary(rank_trace: Dict[str, Any]) -> Dict[str, Any]:
    lower_bounds = [int(value) for value in rank_trace["lower_bounds"]]
    upper_bounds = [int(value) for value in rank_trace["upper_bounds"]]
    equality_steps = [idx for idx, (low, high) in enumerate(zip(lower_bounds, upper_bounds)) if low == high]
    return {
        "steps": len(lower_bounds),
        "squeeze_closed": bool(rank_trace["squeeze_closed"]),
        "first_closing_step": equality_steps[0] if equality_steps else None,
        "lower_final": lower_bounds[-1],
        "upper_final": upper_bounds[-1],
    }


def compute_cert_sha256(cert: Dict[str, Any]) -> str:
    return sha256_hex_bytes(canonical_json_bytes(cert))


def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    gate_1 = gate_1_schema(cert, schema)
    if gate_1 is not None:
        return gate_1.__dict__

    source_batch = cert["source_batch"]
    a_coeff = int(source_batch["curve"]["model"]["A"])
    b_coeff = int(source_batch["curve"]["model"]["B"])

    recomputed_by_prime: Dict[int, Dict[str, Any]] = {}
    for entry in source_batch["records"]:
        prime = int(entry["prime"])
        try:
            recomputed_by_prime[prime] = recompute_for_prime(a_coeff, b_coeff, prime)
        except ValueError as exc:
            return Result(
                ok=False,
                fail_type=FAIL_NON_PRIME_INPUT,
                invariant_diff={"prime": prime},
                details={"error": str(exc)},
            ).__dict__

    gate_2 = gate_2_recompute_checks(source_batch, recomputed_by_prime)
    if gate_2 is not None:
        return gate_2.__dict__

    recomputed_manifest = build_manifest(source_batch["curve"]["model"], recomputed_by_prime)
    gate_3 = gate_3_manifest_checks(source_batch, recomputed_manifest)
    if gate_3 is not None:
        return gate_3.__dict__

    recomputed_proxy = build_exact_proxy(source_batch, recomputed_by_prime)
    gate_4 = gate_4_proxy_checks(cert["claimed_proxy"], recomputed_proxy)
    if gate_4 is not None:
        return gate_4.__dict__

    gate_5 = gate_5_trace_checks(cert["rank_trace"])
    if gate_5 is not None:
        return gate_5.__dict__

    rank_summary = build_rank_summary(cert["rank_trace"])
    return Result(
        ok=True,
        value={
            "curve_id": source_batch["curve"]["curve_id"],
            "num_primes": len(source_batch["records"]),
            "primes_sorted": [int(entry["prime"]) for entry in sorted(source_batch["records"], key=lambda x: int(x["prime"]))],
            "recomputed_by_prime": {str(prime): value for prime, value in sorted(recomputed_by_prime.items())},
            "recomputed_manifest": recomputed_manifest,
            "recomputed_proxy": recomputed_proxy,
            "rank_summary": rank_summary,
            "cert_sha256": compute_cert_sha256(cert),
        },
    ).__dict__


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(str(base / "schema.json"))
    pass_closed = load_json(str(base / "fixtures" / "pass_closed_p5_p7.json"))
    pass_open = load_json(str(base / "fixtures" / "pass_open_p5_p11.json"))
    fail_trace = load_json(str(base / "fixtures" / "fail_bad_trace_crossing.json"))
    fail_proxy = load_json(str(base / "fixtures" / "fail_wrong_proxy_denominator.json"))
    fail_ap = load_json(str(base / "fixtures" / "fail_wrong_ap_p7.json"))

    result_closed = validate_cert(pass_closed, schema)
    result_open = validate_cert(pass_open, schema)
    result_fail = validate_cert(fail_trace, schema)
    result_fail_proxy = validate_cert(fail_proxy, schema)
    result_fail_ap = validate_cert(fail_ap, schema)

    checks = [
        (result_closed.get("ok") is True, "closed squeeze fixture must validate"),
        (result_open.get("ok") is True, "open squeeze fixture must validate"),
        (result_fail.get("ok") is False, "invalid trace fixture must fail"),
        (result_fail_proxy.get("ok") is False, "invalid proxy fixture must fail"),
        (result_fail_ap.get("ok") is False, "invalid ap fixture must fail"),
        (result_fail.get("fail_type") == FAIL_TRACE_INCONSISTENT, "invalid trace must hit TRACE_INCONSISTENT"),
        (result_fail_proxy.get("fail_type") == FAIL_RECOMPUTE_MISMATCH, "invalid proxy must hit RECOMPUTE_MISMATCH"),
        (result_fail_ap.get("fail_type") == FAIL_RECOMPUTE_MISMATCH, "invalid ap must hit RECOMPUTE_MISMATCH"),
        (result_closed.get("value", {}).get("rank_summary", {}).get("first_closing_step") == 1, "closed fixture first closing step must be 1"),
        (result_closed.get("value", {}).get("rank_summary", {}).get("lower_final") == 1, "closed fixture final lower rank must be 1"),
        (result_open.get("value", {}).get("rank_summary", {}).get("squeeze_closed") is False, "open fixture must remain open"),
    ]
    failed_checks = [msg for ok, msg in checks if not ok]

    print(
        json.dumps(
            {
                "self_test_ok": len(failed_checks) == 0,
                "checks": [msg for _, msg in checks],
                "failed_checks": failed_checks,
                "pass_closed_result": result_closed,
                "pass_open_result": result_open,
                "fail_trace_result": result_fail,
                "fail_proxy_result": result_fail_proxy,
                "fail_ap_result": result_fail_ap,
            },
            indent=2,
            sort_keys=True,
        )
    )
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
