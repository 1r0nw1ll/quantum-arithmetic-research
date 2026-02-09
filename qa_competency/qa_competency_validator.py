#!/usr/bin/env python3
"""
qa_competency_validator.py

Validator for QA_COMPETENCY_DETECTION_FRAMEWORK.v1 certs and bundles.

Aligned with Michael Levin's Platonic Space competency-detection programme.
Enforces deterministic metric recomputation, manifest integrity, and
structured failure reporting (fail_type + invariant_diff).

Usage:
    python qa_competency_validator.py --demo
    python qa_competency_validator.py --rehash
    python qa_competency_validator.py --fixtures
    python qa_competency_validator.py --validate-bundle path/to/bundle.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------

def canonical_json_compact(obj: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, full Unicode."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    """SHA256 of canonical JSON representation (64-hex)."""
    return hashlib.sha256(
        canonical_json_compact(obj).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Manifest helpers (HEX64_ZERO placeholder convention)
# ---------------------------------------------------------------------------

HEX64_ZERO = "0" * 64


def _manifest_hashable_copy(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return deep copy with manifest.canonical_json_sha256 zeroed for hashing."""
    out = copy.deepcopy(obj)
    if "manifest" in out and isinstance(out["manifest"], dict):
        out["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    return out


def _enforce_manifest(obj: Dict[str, Any], label: str) -> None:
    """Verify manifest hash matches canonical JSON of content."""
    manifest = obj.get("manifest", {})
    claimed = manifest.get("canonical_json_sha256", "")
    if claimed == HEX64_ZERO:
        return  # sentinel: not yet hashed, skip enforcement
    hashable = _manifest_hashable_copy(obj)
    computed = sha256_canonical(hashable)
    if claimed != computed:
        raise ValidationError(
            fail_type="MANIFEST_HASH_MISMATCH",
            invariant_diff={
                "label": label,
                "claimed": claimed,
                "computed": computed,
            },
        )


def _update_manifest(obj: Dict[str, Any]) -> str:
    """Update manifest hash in place and return the new hash."""
    if "manifest" not in obj:
        obj["manifest"] = {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        }
    hashable = _manifest_hashable_copy(obj)
    computed = sha256_canonical(hashable)
    obj["manifest"]["canonical_json_sha256"] = computed
    return computed


# ---------------------------------------------------------------------------
# Structured failure
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Structured validation failure carrying fail_type + invariant_diff."""

    def __init__(self, fail_type: str, invariant_diff: Dict[str, Any],
                 message: str = ""):
        self.fail_type = fail_type
        self.invariant_diff = invariant_diff
        super().__init__(message or f"{fail_type}: {invariant_diff}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fail_type": self.fail_type,
            "invariant_diff": self.invariant_diff,
        }


@dataclass
class ValidationResult:
    ok: bool
    fail_type: str = ""
    invariant_diff: Dict[str, Any] = field(default_factory=dict)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"ok": self.ok}
        if not self.ok:
            d["fail_type"] = self.fail_type
            d["invariant_diff"] = self.invariant_diff
            if self.details:
                d["details"] = self.details
        return d


# ---------------------------------------------------------------------------
# Schema validation (uses jsonschema if available, else structural checks)
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_schema_path(schema_name: str) -> str:
    """Find schema file relative to this module."""
    return str(Path(__file__).parent / "schemas" / schema_name)


def _jsonschema_validate(instance: Any, schema: Dict[str, Any]) -> None:
    """Validate instance against JSON Schema. Falls back to structural check."""
    try:
        from jsonschema import Draft7Validator
        v = Draft7Validator(schema)
        errors = sorted(v.iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            lines = []
            for e in errors[:20]:
                loc = ".".join(str(p) for p in e.path) if e.path else "<root>"
                lines.append(f"{loc}: {e.message}")
            raise ValidationError(
                fail_type="SCHEMA_VALIDATION_FAILED",
                invariant_diff={"errors": lines},
            )
    except ImportError:
        _structural_check(instance, schema)


def _structural_check(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Minimal structural validation when jsonschema is not installed."""
    required = schema.get("required", [])
    missing = [k for k in required if k not in instance]
    if missing:
        raise ValidationError(
            fail_type="MISSING_REQUIRED_BLOCK",
            invariant_diff={"missing": missing},
        )


# ---------------------------------------------------------------------------
# Metric recomputation
# ---------------------------------------------------------------------------

def _recompute_metrics(cert: Dict[str, Any], tol: float = 1e-9) -> None:
    """Recompute metrics from metric_inputs and compare to declared values."""
    try:
        from qa_competency.qa_competency_metrics import compute_competency_metrics
    except ImportError:
        from qa_competency_metrics import compute_competency_metrics

    mi = cert.get("metric_inputs")
    if mi is None:
        raise ValidationError(
            fail_type="MISSING_REQUIRED_BLOCK",
            invariant_diff={"missing": "metric_inputs"},
        )

    computed = compute_competency_metrics(
        reachable_states=int(mi.get("reachable_states", 0)),
        total_states=int(mi.get("total_states", 0)),
        attractor_basins=int(mi.get("attractor_basins", 0)),
        move_probabilities=mi.get("move_probabilities") or {},
        delta_reachability=float(mi.get("delta_reachability", 0.0)),
        delta_perturbation=float(mi.get("delta_perturbation", 0.0)),
    ).as_dict()

    declared = cert.get("competency_metrics", {})
    for key, cv in computed.items():
        dv = float(declared.get(key, 0.0))
        if abs(dv - cv) > tol:
            raise ValidationError(
                fail_type="METRIC_MISMATCH",
                invariant_diff={
                    "field": key,
                    "declared": dv,
                    "computed": cv,
                },
            )


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------

REQUIRED_CERT_BLOCKS = [
    "schema_id", "system_metadata", "state_space", "generators",
    "invariants", "reachability", "graph_snapshot", "metric_inputs",
    "competency_metrics", "validation", "examples", "manifest",
]


def _semantic_checks(cert: Dict[str, Any]) -> None:
    """Verify required blocks and schema_id."""
    if cert.get("schema_id") != "QA_COMPETENCY_DETECTION_FRAMEWORK.v1":
        raise ValidationError(
            fail_type="SCHEMA_VALIDATION_FAILED",
            invariant_diff={
                "field": "schema_id",
                "expected": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
                "actual": cert.get("schema_id"),
            },
        )
    missing = [k for k in REQUIRED_CERT_BLOCKS if k not in cert]
    if missing:
        raise ValidationError(
            fail_type="MISSING_REQUIRED_BLOCK",
            invariant_diff={"missing": missing},
        )
    r = cert.get("reachability", {})
    if int(r.get("components", 0)) < 0 or int(r.get("diameter", 0)) < 0:
        raise ValidationError(
            fail_type="SCHEMA_VALIDATION_FAILED",
            invariant_diff={"issue": "reachability values must be non-negative"},
        )


# ---------------------------------------------------------------------------
# Single cert validation
# ---------------------------------------------------------------------------

def validate_cert(
    cert: Dict[str, Any],
    *,
    schema_path: Optional[str] = None,
    verify_manifest: bool = True,
    metric_tol: float = 1e-9,
) -> ValidationResult:
    """Validate a single QA_COMPETENCY_DETECTION_FRAMEWORK.v1 cert."""
    try:
        if schema_path:
            schema = _load_json(schema_path)
            _jsonschema_validate(cert, schema)
        _semantic_checks(cert)
        _recompute_metrics(cert, tol=metric_tol)
        if verify_manifest:
            _enforce_manifest(cert, "cert")
        return ValidationResult(ok=True)
    except ValidationError as e:
        return ValidationResult(
            ok=False,
            fail_type=e.fail_type,
            invariant_diff=e.invariant_diff,
            details=str(e),
        )


# ---------------------------------------------------------------------------
# Bundle validation
# ---------------------------------------------------------------------------

def validate_bundle(
    bundle: Dict[str, Any],
    *,
    bundle_schema_path: Optional[str] = None,
    cert_schema_path: Optional[str] = None,
    verify_manifest: bool = True,
    metric_tol: float = 1e-9,
) -> ValidationResult:
    """Validate a QA_COMPETENCY_CERT_BUNDLE.v1."""
    try:
        if bundle.get("schema_id") != "QA_COMPETENCY_CERT_BUNDLE.v1":
            raise ValidationError(
                fail_type="SCHEMA_VALIDATION_FAILED",
                invariant_diff={
                    "field": "schema_id",
                    "expected": "QA_COMPETENCY_CERT_BUNDLE.v1",
                    "actual": bundle.get("schema_id"),
                },
            )

        if bundle_schema_path:
            schema = _load_json(bundle_schema_path)
            _jsonschema_validate(bundle, schema)

        certs = bundle.get("certs", [])
        if not isinstance(certs, list) or len(certs) == 0:
            raise ValidationError(
                fail_type="SCHEMA_VALIDATION_FAILED",
                invariant_diff={"issue": "certs array empty or missing"},
            )

        for idx, cert in enumerate(certs):
            if not isinstance(cert, dict):
                raise ValidationError(
                    fail_type="SCHEMA_VALIDATION_FAILED",
                    invariant_diff={"issue": f"cert[{idx}] is not an object"},
                )
            _semantic_checks(cert)
            _recompute_metrics(cert, tol=metric_tol)
            if verify_manifest:
                _enforce_manifest(cert, f"cert[{idx}]")

        if verify_manifest:
            _enforce_manifest(bundle, "bundle")

        return ValidationResult(ok=True)
    except ValidationError as e:
        return ValidationResult(
            ok=False,
            fail_type=e.fail_type,
            invariant_diff=e.invariant_diff,
            details=str(e),
        )


# ---------------------------------------------------------------------------
# validate_all — entry point for family sweep
# ---------------------------------------------------------------------------

def validate_all(*, bundle_path: str) -> None:
    """
    Validate the competency bundle end-to-end.
    Raises on any failure (family sweep contract).
    """
    bundle = _load_json(bundle_path)
    base = Path(__file__).parent

    bundle_schema = str(base / "schemas" / "QA_COMPETENCY_CERT_BUNDLE.v1.schema.json")
    cert_schema = str(base / "schemas" / "QA_COMPETENCY_DETECTION_FRAMEWORK.v1.schema.json")

    result = validate_bundle(
        bundle,
        bundle_schema_path=bundle_schema if os.path.exists(bundle_schema) else None,
        cert_schema_path=cert_schema if os.path.exists(cert_schema) else None,
    )
    if not result.ok:
        raise RuntimeError(
            f"Bundle validation failed: {result.fail_type} — {result.invariant_diff}"
        )

    n_certs = len(bundle.get("certs", []))
    print(f"[Competency] Validated bundle: {n_certs} cert(s), metrics recomputed, manifest ok")


# ---------------------------------------------------------------------------
# Fixture runner
# ---------------------------------------------------------------------------

def run_fixtures() -> Dict[str, Any]:
    """Run fixture assertions. Returns {ok, passed, failed, details}."""
    base = Path(__file__).parent
    cert_schema = str(base / "schemas" / "QA_COMPETENCY_DETECTION_FRAMEWORK.v1.schema.json")

    results: List[Dict[str, Any]] = []

    # Valid fixtures
    valid_dir = base / "fixtures" / "valid"
    if valid_dir.is_dir():
        for f in sorted(valid_dir.glob("*.json")):
            cert = _load_json(str(f))
            r = validate_cert(cert, schema_path=cert_schema)
            results.append({
                "file": f.name,
                "expected": "pass",
                "actual": "pass" if r.ok else "fail",
                "passed": r.ok,
                "details": r.to_dict(),
            })

    # Invalid fixtures
    invalid_dir = base / "fixtures" / "invalid"
    if invalid_dir.is_dir():
        for f in sorted(invalid_dir.glob("*.json")):
            cert = _load_json(str(f))
            # Read expected fail_type from _expected metadata if present
            expected_ft = cert.pop("_expected_fail_type", None)
            r = validate_cert(cert, schema_path=cert_schema)
            fixture_passed = not r.ok  # should fail
            if expected_ft and r.fail_type != expected_ft:
                fixture_passed = False
            results.append({
                "file": f.name,
                "expected": "fail",
                "expected_fail_type": expected_ft,
                "actual": "fail" if not r.ok else "pass",
                "actual_fail_type": r.fail_type,
                "passed": fixture_passed,
                "details": r.to_dict(),
            })

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    return {
        "ok": failed == 0 and len(results) > 0,
        "passed": passed,
        "failed": failed,
        "tests": results,
    }


# ---------------------------------------------------------------------------
# Rehash
# ---------------------------------------------------------------------------

def rehash_all() -> None:
    """Recompute manifest hashes for all cert files."""
    base = Path(__file__).parent
    cert_files = list((base / "certs").glob("*.json"))
    for p in sorted(cert_files):
        obj = _load_json(str(p))

        # If bundle, rehash embedded certs first
        if isinstance(obj.get("certs"), list):
            for cert in obj["certs"]:
                if isinstance(cert, dict) and "manifest" in cert:
                    h = _update_manifest(cert)
                    print(f"  [REHASH] embedded cert: {h[:16]}...")

        h = _update_manifest(obj)
        with open(p, "w", encoding="utf-8") as f:
            f.write(canonical_json_compact(obj))
            f.write("\n")
        print(f"[REHASH] {p.name}: {h[:16]}...")


# ---------------------------------------------------------------------------
# Demo cert
# ---------------------------------------------------------------------------

def make_demo_cert() -> Dict[str, Any]:
    """Create a minimal valid competency cert for demo/test purposes."""
    return {
        "schema_id": "QA_COMPETENCY_DETECTION_FRAMEWORK.v1",
        "system_metadata": {
            "domain": "demo",
            "substrate": "synthetic",
            "description": "Minimal demo cert for validator self-test",
        },
        "state_space": {
            "dimension": 2,
            "coordinates": ["x", "y"],
            "constraints": [],
        },
        "generators": [
            {"id": "move_x", "description": "Shift along x", "action": "increment_x"},
        ],
        "invariants": [],
        "reachability": {
            "components": 1,
            "diameter": 4,
            "obstructions": [],
        },
        "graph_snapshot": {
            "hash_sha256": HEX64_ZERO,
            "time_window": {
                "start_utc": "2026-02-09T00:00:00Z",
                "end_utc": "2026-02-09T00:00:00Z",
            },
            "edge_semantics": "generator-edges: apply generator id to state yields next-state",
        },
        "metric_inputs": {
            "reachable_states": 5,
            "total_states": 10,
            "attractor_basins": 2,
            "move_probabilities": {"move_x": 1.0},
            "delta_reachability": 3.0,
            "delta_perturbation": 5.0,
        },
        "competency_metrics": {
            "agency_index": 0.5,
            "plasticity_index": 0.6,
            "goal_density": 0.2,
            "control_entropy": 0.0,  # single move -> entropy = 0
        },
        "validation": {
            "validator": "qa_competency_validator.py",
            "hash": "sha256:" + HEX64_ZERO,
            "reproducibility_seed": 42,
        },
        "examples": ["demo_self_test"],
        "manifest": {
            "manifest_version": 1,
            "hash_alg": "sha256_canonical",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA Competency Detection validator")
    ap.add_argument("--demo", action="store_true", help="Validate a demo cert")
    ap.add_argument("--rehash", action="store_true", help="Recompute manifest hashes")
    ap.add_argument("--fixtures", action="store_true", help="Run fixture assertions")
    ap.add_argument("--validate-bundle", metavar="PATH", help="Validate a bundle file")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args(argv)

    if args.rehash:
        print("[Competency] Rehashing manifest hashes...")
        rehash_all()
        return 0

    if args.demo:
        print("[Competency] Running demo validation...")
        cert = make_demo_cert()
        r = validate_cert(cert, verify_manifest=False)
        if r.ok:
            print("[Competency] Demo cert: PASS")
            return 0
        else:
            print(f"[Competency] Demo cert: FAIL ({r.fail_type})")
            if args.json:
                print(json.dumps(r.to_dict(), indent=2))
            return 1

    if args.fixtures:
        print("[Competency] Running fixture tests...")
        result = run_fixtures()
        if result["ok"]:
            print(f"[Competency] Fixtures: PASS ({result['passed']} passed)")
        else:
            print(f"[Competency] Fixtures: FAIL ({result['failed']} failed)")
            for t in result["tests"]:
                if not t["passed"]:
                    print(f"  {t['file']}: expected={t['expected']}, "
                          f"actual={t['actual']}, fail_type={t.get('actual_fail_type', 'n/a')}")
        if args.json:
            print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1

    if args.validate_bundle:
        print(f"[Competency] Validating bundle: {args.validate_bundle}")
        try:
            validate_all(bundle_path=args.validate_bundle)
            return 0
        except Exception as e:
            print(f"[Competency] FAIL: {e}")
            return 1

    print("Usage: qa_competency_validator.py --demo | --rehash | --fixtures | --validate-bundle PATH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
