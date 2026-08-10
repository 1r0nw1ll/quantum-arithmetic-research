#!/usr/bin/env python3
"""QA HSI Material Identification Cert validator.

Certifies the claim surface for hyperspectral material-identification
experiments: wavelength grid, observer quantization boundary, material-library
hash, diagnostic absorption witnesses, mixture/unknown coverage, target
detection, abundance bins, and synthetic-vs-real scope honesty.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "QA_HSI_MATERIAL_ID_CERT.v1"
CERT_TYPE = "qa_hsi_material_id_cert"
KNOWN_SOURCE_TYPES = {"synthetic", "real_spectral_library", "sensor_calibrated"}
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def domain_hash(domain: str, obj: Any) -> str:
    return sha256(domain.encode("utf-8") + b"\x00" + canonical_json(obj)).hexdigest()


class _Out:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.detected_fails: set[str] = set()

    def fail(self, code: str, msg: str) -> None:
        self.detected_fails.add(code)
        self.errors.append(f"{code}: {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_hsi_material_id_cert(cert: dict[str, Any]) -> dict[str, Any]:
    out = _Out()

    if cert.get("schema_version") != SCHEMA_VERSION:
        out.fail("SCHEMA_VERSION", f"schema_version must be {SCHEMA_VERSION!r}")
    if cert.get("cert_type") != CERT_TYPE:
        out.fail("CERT_TYPE", f"cert_type must be {CERT_TYPE!r}")

    required = [
        "certificate_id",
        "title",
        "created_utc",
        "source_type",
        "scope_claim",
        "observer_boundary",
        "wavelength_grid_nm",
        "material_library",
        "hashes",
        "sample_counts",
        "success_criteria",
        "metrics",
        "thresholds",
        "result",
        "fail_ledger",
    ]
    for field in required:
        if field not in cert:
            out.fail("MISSING_FIELD", f"missing required field {field!r}")
    if out.errors:
        return _reconcile(cert, out)

    source_type = cert.get("source_type")
    if source_type not in KNOWN_SOURCE_TYPES:
        out.fail("BAD_SOURCE_TYPE", f"unknown source_type {source_type!r}")

    scope_claim = cert.get("scope_claim", {})
    if source_type == "synthetic" and scope_claim.get("claims_real_data_validation") is True:
        out.fail("SYNTHETIC_REAL_OVERCLAIM", "synthetic cert cannot claim real-data validation")

    boundary = cert.get("observer_boundary", {})
    if boundary.get("quantized_once") is not True:
        out.fail("BAD_OBSERVER_BOUNDARY", "observer_boundary.quantized_once must be true")
    if boundary.get("qa_layer_integer_only") is not True:
        out.fail("BAD_OBSERVER_BOUNDARY", "observer_boundary.qa_layer_integer_only must be true")
    if not isinstance(boundary.get("quantization"), str) or not boundary["quantization"].strip():
        out.fail("BAD_OBSERVER_BOUNDARY", "observer_boundary.quantization must be a nonempty string")

    wavelengths = cert.get("wavelength_grid_nm", [])
    if not isinstance(wavelengths, list) or not wavelengths:
        out.fail("BAD_WAVELENGTH_GRID", "wavelength_grid_nm must be a nonempty list")
    elif any(not isinstance(x, int) or x <= 0 for x in wavelengths):
        out.fail("BAD_WAVELENGTH_GRID", "all wavelengths must be positive integers")
    elif wavelengths != sorted(wavelengths):
        out.fail("BAD_WAVELENGTH_GRID", "wavelength_grid_nm must be sorted ascending")

    library = cert.get("material_library", {})
    if not isinstance(library, dict) or not library:
        out.fail("BAD_MATERIAL_LIBRARY", "material_library must be a nonempty object")
    elif isinstance(wavelengths, list):
        wavelength_set = set(wavelengths)
        for name, entry in library.items():
            if not isinstance(entry, dict):
                out.fail("BAD_MATERIAL_LIBRARY", f"material {name!r} must be an object")
                continue
            spectrum = entry.get("spectrum")
            bands = entry.get("absorption_bands_nm")
            if not isinstance(spectrum, list) or len(spectrum) != len(wavelengths):
                out.fail("BAD_MATERIAL_LIBRARY", f"material {name!r} spectrum length must match wavelength grid")
            elif any(not isinstance(v, int) or v < 1 or v > 255 for v in spectrum):
                out.fail("BAD_MATERIAL_LIBRARY", f"material {name!r} spectrum must contain 8-bit positive integers")
            if not isinstance(bands, list) or not bands:
                out.fail("MISSING_ABSORPTION_WITNESS", f"material {name!r} has no diagnostic absorption bands")
            elif any(b not in wavelength_set for b in bands):
                out.fail("BAD_ABSORPTION_WITNESS", f"material {name!r} absorption bands must lie on wavelength grid")

    hashes = cert.get("hashes", {})
    if isinstance(hashes, dict):
        declared_wl = hashes.get("wavelength_grid_sha256")
        declared_lib = hashes.get("material_library_sha256")
        if not isinstance(declared_wl, str) or not HEX64_RE.match(declared_wl):
            out.fail("BAD_HASH_FORMAT", "wavelength_grid_sha256 must be 64 lowercase hex chars")
        elif declared_wl != domain_hash("QA_HSI_WAVELENGTH_GRID.v1", wavelengths):
            out.fail("WAVELENGTH_HASH_MISMATCH", "wavelength_grid_sha256 does not recompute")
        if not isinstance(declared_lib, str) or not HEX64_RE.match(declared_lib):
            out.fail("BAD_HASH_FORMAT", "material_library_sha256 must be 64 lowercase hex chars")
        elif declared_lib != domain_hash("QA_HSI_MATERIAL_LIBRARY.v1", library):
            out.fail("MATERIAL_LIBRARY_HASH_MISMATCH", "material_library_sha256 does not recompute")
    else:
        out.fail("BAD_HASHES", "hashes must be an object")

    counts = cert.get("sample_counts", {})
    for key in ("pure_total", "mixture_total", "unknown_total", "known_total"):
        if not isinstance(counts.get(key), int) or counts[key] <= 0:
            out.fail("BAD_SAMPLE_COUNTS", f"sample_counts.{key} must be a positive integer")

    criteria = cert.get("success_criteria", {})
    metrics = cert.get("metrics", {})
    gate_specs = [
        ("pure_top1_accuracy", "pure_top1_accuracy_min", ">="),
        ("mixture_dominant_top1_accuracy", "mixture_dominant_top1_accuracy_min", ">="),
        ("target_detection_f1", "target_detection_f1_min", ">="),
        ("abundance_bin_accuracy", "abundance_bin_accuracy_min", ">="),
        ("unknown_rejection_fpr", "unknown_rejection_fpr_max", "<="),
    ]
    for metric_key, criterion_key, op in gate_specs:
        metric = metrics.get(metric_key)
        criterion = criteria.get(criterion_key)
        if not _is_number(metric):
            out.fail("MISSING_METRIC", f"metrics.{metric_key} must be numeric")
            continue
        if not _is_number(criterion):
            out.fail("MISSING_CRITERION", f"success_criteria.{criterion_key} must be numeric")
            continue
        if op == ">=" and metric < criterion:
            out.fail("METRIC_GATE_FAIL", f"{metric_key}={metric} is below {criterion_key}={criterion}")
        if op == "<=" and metric > criterion:
            out.fail("METRIC_GATE_FAIL", f"{metric_key}={metric} is above {criterion_key}={criterion}")

    thresholds = cert.get("thresholds", {})
    if not isinstance(thresholds, dict):
        out.fail("BAD_THRESHOLDS", "thresholds must be an object")
    else:
        if not isinstance(thresholds.get("unknown_nearest_distance_threshold"), int):
            out.fail("BAD_THRESHOLDS", "unknown_nearest_distance_threshold must be an integer")
        if not isinstance(thresholds.get("target_rule"), str) or not thresholds["target_rule"].strip():
            out.fail("BAD_THRESHOLDS", "target_rule must be a nonempty string")

    return _reconcile(cert, out)


def _reconcile(cert: dict[str, Any], out: _Out) -> dict[str, Any]:
    declared_result = cert.get("result")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {
        item.get("fail_type")
        for item in declared_ledger
        if isinstance(item, dict) and isinstance(item.get("fail_type"), str)
    }

    if declared_result == "PASS":
        if out.detected_fails:
            for code in sorted(out.detected_fails):
                if not any(err.startswith(f"{code}:") for err in out.errors):
                    out.errors.append(f"{code}: cert declares PASS but validator detected failure")
    elif declared_result == "FAIL":
        missing = out.detected_fails - declared_fail_types
        for code in sorted(missing):
            out.warn(f"detected {code} but fail_ledger does not declare it")
    else:
        out.fail("BAD_RESULT", "result must be PASS or FAIL")

    ok = declared_result == "PASS" and not out.errors
    if declared_result == "FAIL" and out.detected_fails:
        ok = False

    return {
        "ok": ok,
        "label": "PASS" if ok else "FAIL",
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(out.detected_fails),
    }


def validate_file(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return validate_hsi_material_id_cert(json.load(f))


def self_test() -> dict[str, Any]:
    fixtures = Path(__file__).parent / "fixtures"
    expected = {
        "hmi_pass_synthetic_material_id.json": True,
        "hmi_fail_bad_library_hash.json": False,
        "hmi_fail_missing_unknown_rejection.json": False,
        "hmi_fail_overclaim_real_data.json": False,
    }
    results = []
    all_ok = True
    for name, expect_ok in expected.items():
        path = fixtures / name
        if not path.exists():
            results.append({"fixture": name, "ok": False, "error": "missing fixture"})
            all_ok = False
            continue
        result = validate_file(path)
        matched = result["ok"] == expect_ok
        all_ok = all_ok and matched
        results.append({
            "fixture": name,
            "ok": matched,
            "validator_ok": result["ok"],
            "label": result["label"],
            "detected_fails": result["detected_fails"],
            "errors": result["errors"],
        })
    return {"ok": all_ok, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="QA HSI Material Identification Cert validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
