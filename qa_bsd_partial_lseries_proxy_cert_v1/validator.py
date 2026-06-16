#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jsonschema  # type: ignore
except ModuleNotFoundError:
    jsonschema = None


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def discriminant_short_weierstrass_mod_p(a_coeff: int, b_coeff: int, prime: int) -> int:
    a3 = pow(a_coeff, 3, prime)
    b2 = pow(b_coeff, 2, prime)
    return (-16 * (4 * a3 + 27 * b2)) % prime


def count_points_fp_short_weierstrass(a_coeff: int, b_coeff: int, prime: int) -> int:
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
FAIL_MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
FAIL_DUPLICATE_PRIME = "DUPLICATE_PRIME"


@dataclass
class Result:
    ok: bool
    fail_type: Optional[str] = None
    invariant_diff: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    value: Optional[Dict[str, Any]] = None


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Result]:
    try:
        if jsonschema is None:
            validate_schema_minimal(cert, schema)
        else:
            jsonschema.Draft202012Validator(schema).validate(cert)
        return None
    except Exception as exc:
        return Result(ok=False, fail_type=FAIL_SCHEMA, invariant_diff={}, details={"error": str(exc)})


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expect_type(value: Any, expected_type: type, path: str) -> None:
    expect(isinstance(value, expected_type), f"{path} must be {expected_type.__name__}")


def expect_required(obj: Dict[str, Any], fields: list[str], path: str) -> None:
    missing = [field for field in fields if field not in obj]
    expect(not missing, f"{path} missing required fields: {', '.join(missing)}")


def expect_int(value: Any, path: str, min_value: Optional[int] = None,
               max_value: Optional[int] = None) -> None:
    expect(isinstance(value, int) and not isinstance(value, bool), f"{path} must be integer")
    if min_value is not None:
        expect(value >= min_value, f"{path} below minimum {min_value}")
    if max_value is not None:
        expect(value <= max_value, f"{path} above maximum {max_value}")


def expect_hash64(value: Any, path: str) -> None:
    expect(
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value),
        f"{path} must be 64 lowercase hex chars",
    )


def validate_curve(curve: Dict[str, Any], path: str) -> None:
    expect_type(curve, dict, path)
    expect_required(curve, ["curve_id", "model"], path)
    expect_type(curve["curve_id"], str, f"{path}.curve_id")
    expect(len(curve["curve_id"]) >= 1, f"{path}.curve_id empty")
    model = curve["model"]
    expect_type(model, dict, f"{path}.model")
    expect_required(model, ["type", "A", "B"], f"{path}.model")
    expect(model["type"] == "short_weierstrass", f"{path}.model.type invalid")
    expect_int(model["A"], f"{path}.model.A")
    expect_int(model["B"], f"{path}.model.B")


def validate_claimed(claimed: Dict[str, Any], path: str) -> None:
    expect_type(claimed, dict, path)
    expect_required(claimed, ["point_count_fp", "ap"], path)
    expect_int(claimed["point_count_fp"], f"{path}.point_count_fp", 1)
    expect_int(claimed["ap"], f"{path}.ap")
    if "delta_mod_p" in claimed:
        expect_int(claimed["delta_mod_p"], f"{path}.delta_mod_p")
    if "is_good_reduction" in claimed:
        expect_type(claimed["is_good_reduction"], bool, f"{path}.is_good_reduction")
    if "ap_source" in claimed:
        expect(claimed["ap_source"] in {"point_count", "table"}, f"{path}.ap_source invalid")
    if "reduction_type" in claimed:
        expect(
            claimed["reduction_type"] in {
                "GOOD",
                "BAD_UNSPECIFIED",
                "BAD_MULTIPLICATIVE_SPLIT",
                "BAD_MULTIPLICATIVE_NONSPLIT",
                "BAD_ADDITIVE",
            },
            f"{path}.reduction_type invalid",
        )


def validate_manifest(manifest: Dict[str, Any], path: str) -> None:
    expect_type(manifest, dict, path)
    expect_required(manifest, ["record_hashes", "batch_sha256"], path)
    hashes = manifest["record_hashes"]
    expect(isinstance(hashes, list) and len(hashes) >= 1, f"{path}.record_hashes must be nonempty")
    for idx, entry in enumerate(hashes):
        e_path = f"{path}.record_hashes[{idx}]"
        expect_type(entry, dict, e_path)
        expect_required(entry, ["prime", "sha256"], e_path)
        expect_int(entry["prime"], f"{e_path}.prime", 2, 100000)
        expect_hash64(entry["sha256"], f"{e_path}.sha256")
    expect_hash64(manifest["batch_sha256"], f"{path}.batch_sha256")


def validate_schema_minimal(cert: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Dependency-free checker for the concrete partial L-series proxy schema."""

    expect_type(cert, dict, "$")
    expect_required(cert, schema.get("required", []), "$")
    expect(cert.get("schema_id") == "QA_BSD_PARTIAL_LSERIES_PROXY_CERT", "schema_id invalid")
    expect(cert.get("schema_version") == "v1", "schema_version invalid")

    source = cert.get("source_batch")
    expect_type(source, dict, "source_batch")
    expect_required(source, ["curve", "records", "claimed_manifest"], "source_batch")
    validate_curve(source["curve"], "source_batch.curve")
    records = source["records"]
    expect(isinstance(records, list) and len(records) >= 1, "source_batch.records must be nonempty")
    for idx, record in enumerate(records):
        path = f"source_batch.records[{idx}]"
        expect_type(record, dict, path)
        expect_required(record, ["prime", "claimed"], path)
        expect_int(record["prime"], f"{path}.prime", 2, 100000)
        validate_claimed(record["claimed"], f"{path}.claimed")
    validate_manifest(source["claimed_manifest"], "source_batch.claimed_manifest")

    proxy = cert.get("claimed_proxy")
    expect_type(proxy, dict, "claimed_proxy")
    expect_required(proxy, ["numerator_factors", "denominator_factors", "numerator", "denominator"],
                    "claimed_proxy")
    for field in ("numerator_factors", "denominator_factors"):
        values = proxy[field]
        expect(isinstance(values, list) and len(values) >= 1, f"claimed_proxy.{field} must be nonempty")
        for idx, value in enumerate(values):
            expect_int(value, f"claimed_proxy.{field}[{idx}]", 1)
    expect_int(proxy["numerator"], "claimed_proxy.numerator", 1)
    expect_int(proxy["denominator"], "claimed_proxy.denominator", 1)
    if "meta" in cert:
        expect_type(cert["meta"], dict, "meta")


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
    normalized = [{"prime": int(x["prime"]), "sha256": str(x["sha256"]).lower()} for x in record_hashes]
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
        duplicates = sorted({p for p in primes if primes.count(p) > 1})
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
    # Non-reduced exact product: Π (#E(F_p)/p)
    primes_sorted = sorted(int(entry["prime"]) for entry in source_batch["records"])
    numerator_factors = [int(recomputed_by_prime[p]["point_count_fp"]) for p in primes_sorted]
    denominator_factors = [int(p) for p in primes_sorted]

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
        recomputed_by_prime[prime] = recompute_for_prime(a_coeff, b_coeff, prime)

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

    return Result(
        ok=True,
        value={
            "curve_id": source_batch["curve"]["curve_id"],
            "num_primes": len(source_batch["records"]),
            "primes_sorted": [int(entry["prime"]) for entry in sorted(source_batch["records"], key=lambda x: int(x["prime"]))],
            "recomputed_by_prime": {str(prime): value for prime, value in sorted(recomputed_by_prime.items())},
            "recomputed_manifest": recomputed_manifest,
            "recomputed_proxy": recomputed_proxy,
            "cert_sha256": compute_cert_sha256(cert),
        },
    ).__dict__


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_self_test() -> int:
    base = Path(__file__).resolve().parent
    schema = load_json(str(base / "schema.json"))
    pass_a = load_json(str(base / "fixtures" / "pass_proxy_p5_p7.json"))
    pass_b = load_json(str(base / "fixtures" / "pass_proxy_p5_p11.json"))
    fail = load_json(str(base / "fixtures" / "fail_wrong_proxy_denominator.json"))

    result_a = validate_cert(pass_a, schema)
    result_b = validate_cert(pass_b, schema)
    result_f = validate_cert(fail, schema)

    checks = [
        (result_a.get("ok") is True, "pass p5/p7 must validate"),
        (result_b.get("ok") is True, "pass p5/p11 must validate"),
        (result_f.get("ok") is False, "fail must fail"),
        (result_f.get("fail_type") == FAIL_RECOMPUTE_MISMATCH, "fail must hit RECOMPUTE_MISMATCH"),
        (result_a.get("value", {}).get("recomputed_proxy", {}).get("numerator") == 72, "p5/p7 numerator must be 72"),
        (result_a.get("value", {}).get("recomputed_proxy", {}).get("denominator") == 35, "p5/p7 denominator must be 35"),
    ]
    failed_checks = [msg for ok, msg in checks if not ok]
    print(
        json.dumps(
            {
                "self_test_ok": len(failed_checks) == 0,
                "checks": [msg for _, msg in checks],
                "failed_checks": failed_checks,
                "pass_a_result": result_a,
                "pass_b_result": result_b,
                "fail_result": result_f,
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
