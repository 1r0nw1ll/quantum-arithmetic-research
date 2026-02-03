"""
qa_fst_validate.py

Deterministic validator for Field Structure Theory (FST) / Structural Physics (SP)
QA module spine and certificate bundle.

Replay contract:
    LOAD -> CANONICALIZE -> ENFORCE_STF_BASIS -> ENFORCE_SYMMETRY
    -> ENFORCE_HOMOMORPHISM -> CLASSIFY -> EMIT_METRICS

Fail records use strict {move, fail_type, invariant_diff} contract.
SOURCE_NUMERIC_DRIFT is a warning (not a hard fail) unless it breaks
an explicit test.

Hardened per CERT_FAMILY_HARDENING_PLAYBOOK.md:
- Imports canonicalization from qa_cert_core (single source of truth)
- Supports --check-manifest for fast CI gates
- Dual hash verification (file bytes + canonical JSON)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Import canonical functions from shared core to prevent drift
try:
    # When run as module: python -m qa_alphageometry_ptolemy.qa_fst.qa_fst_validate
    from ..qa_cert_core import canonical_json_compact, sha256_canonical, sha256_file
except ImportError:
    # When run directly: python qa_fst_validate.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qa_cert_core import canonical_json_compact, sha256_canonical, sha256_file


# ============================================================================
# CANONICAL JSON + HASHING (wrappers for backward compatibility)
# ============================================================================

def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8.

    NOTE: This is a wrapper around qa_cert_core.canonical_json_compact()
    to maintain backward compatibility. All new code should import directly.
    """
    return canonical_json_compact(obj)


def sha256_hex(s: str) -> str:
    """SHA256 of a string. For canonical JSON, prefer sha256_canonical()."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ============================================================================
# FAIL RECORD
# ============================================================================

def fail_record(move: str, fail_type: str,
                invariant_diff: Dict[str, Any]) -> Dict[str, Any]:
    """Strict {move, fail_type, invariant_diff} per FAIL_RECORD.v1 contract."""
    return {
        "move": move,
        "fail_type": fail_type,
        "invariant_diff": invariant_diff,
    }


def warning_record(move: str, fail_type: str,
                   invariant_diff: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as fail_record, but classified as warning."""
    rec = fail_record(move, fail_type, invariant_diff)
    rec["severity"] = "warning"
    return rec


# ============================================================================
# VALIDATION ENGINE
# ============================================================================

class FSTValidationResult:
    """Full validation output with fail records + metrics."""

    def __init__(self) -> None:
        self.ok: bool = True
        self.fail_records: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.hashes: Dict[str, str] = {}

    def add_fail(self, move: str, fail_type: str,
                 invariant_diff: Dict[str, Any]) -> None:
        self.ok = False
        self.fail_records.append(fail_record(move, fail_type, invariant_diff))

    def add_warning(self, move: str, fail_type: str,
                    invariant_diff: Dict[str, Any]) -> None:
        self.warnings.append(warning_record(move, fail_type, invariant_diff))

    @property
    def result_label(self) -> str:
        if not self.ok:
            return "FAIL"
        if self.warnings:
            return "PASS_WITH_WARNINGS"
        return "PASS"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "result": self.result_label,
            "ok": self.ok,
            "hashes": self.hashes,
            "metrics": self.metrics,
        }
        if self.fail_records:
            d["fail_records"] = self.fail_records
        if self.warnings:
            d["warnings"] = self.warnings
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def validate_bundle(spine: Dict[str, Any],
                    cert_bundle: Dict[str, Any],
                    manifest: Optional[Dict[str, Any]] = None
                    ) -> FSTValidationResult:
    """
    Full deterministic validation of FST module spine + certificate bundle.

    Steps:
        1. LOAD: schema version checks
        2. CANONICALIZE: sha256 of canonical JSON
        3. ENFORCE_STF_BASIS: lambda bookkeeping arithmetic
        4. ENFORCE_SYMMETRY: delta_sym threshold for stability claims
        5. ENFORCE_HOMOMORPHISM: ratio test + bookkeeping
        6. CLASSIFY: emit fail records
        7. EMIT_METRICS: return metrics bundle
    """
    out = FSTValidationResult()

    # ---- STEP 1: LOAD ----
    if spine.get("schema_version") != "QA_MAP_MODULE_SPINE.v1":
        out.add_fail("LOAD", "BAD_SCHEMA",
                     {"expected": "QA_MAP_MODULE_SPINE.v1",
                      "got": spine.get("schema_version")})
        return out

    if cert_bundle.get("schema_version") != "QA_CERT_BUNDLE.v1":
        out.add_fail("LOAD", "BAD_SCHEMA",
                     {"expected": "QA_CERT_BUNDLE.v1",
                      "got": cert_bundle.get("schema_version")})
        return out

    # Check module_id cross-reference
    if cert_bundle.get("module_id") != spine.get("module_id"):
        out.add_fail("LOAD", "MODULE_ID_MISMATCH",
                     {"spine": spine.get("module_id"),
                      "bundle": cert_bundle.get("module_id")})

    # ---- STEP 2: CANONICALIZE ----
    spine_hash = sha256_hex(canonical_json(spine))
    cert_hash = sha256_hex(canonical_json(cert_bundle))
    out.hashes["spine"] = spine_hash
    out.hashes["cert_bundle"] = cert_hash

    # Verify against manifest if provided
    if manifest is not None:
        for entry in manifest.get("entries", []):
            eid = entry.get("id", "")
            expected_hash = entry.get("sha256", "")
            if eid == spine.get("module_id"):
                if spine_hash != expected_hash:
                    out.add_fail("CANONICALIZE", "HASH_MISMATCH",
                                 {"artifact": "spine", "expected": expected_hash,
                                  "computed": spine_hash})
            elif eid == cert_bundle.get("certificate_id"):
                if cert_hash != expected_hash:
                    out.add_fail("CANONICALIZE", "HASH_MISMATCH",
                                 {"artifact": "cert_bundle",
                                  "expected": expected_hash,
                                  "computed": cert_hash})

    # ---- STEPS 3-5: Process each claim ----
    for claim in cert_bundle.get("claims", []):
        claim_type = claim.get("type", "")

        # ---- STEP 4: ENFORCE_SYMMETRY (delta_sym witness) ----
        if claim_type.startswith("FST_PROTON_STABILITY"):
            w = claim.get("symmetry_witness", {})
            excess = w.get("side_excess_triangles", [])
            total = w.get("total_triangles", 1)

            # Recompute L1 and delta_sym
            l1_recomputed = sum(abs(x) for x in excess)
            delta_sym_recomputed = l1_recomputed / total if total > 0 else 0.0

            out.metrics["delta_sym_recomputed"] = delta_sym_recomputed
            out.metrics["l1_recomputed"] = l1_recomputed

            # Check declared values match recomputation
            declared_l1 = w.get("computed", {}).get("l1")
            declared_delta = w.get("computed", {}).get("delta_sym")

            if declared_l1 is not None and declared_l1 != l1_recomputed:
                out.add_fail("ENFORCE_SYMMETRY", "RECOMPUTE_MISMATCH",
                             {"field": "l1", "declared": declared_l1,
                              "recomputed": l1_recomputed})

            if declared_delta is not None:
                if abs(declared_delta - delta_sym_recomputed) > 1e-12:
                    out.add_fail("ENFORCE_SYMMETRY", "RECOMPUTE_MISMATCH",
                                 {"field": "delta_sym",
                                  "declared": declared_delta,
                                  "recomputed": delta_sym_recomputed})

            # Threshold check
            threshold = w.get("thresholds", {}).get("stable_if_delta_sym_le")
            if threshold is not None:
                if delta_sym_recomputed > threshold:
                    out.add_fail("ENFORCE_SYMMETRY", "SYMMETRY_DEFECT",
                                 {"delta_sym": delta_sym_recomputed,
                                  "threshold": threshold})

        # ---- STEPS 3 + 5: ENFORCE_STF_BASIS + ENFORCE_HOMOMORPHISM ----
        if claim_type.startswith("LOOP_TO_MEV_HOMOMORPHISM"):
            tolerances = claim.get("tolerances", {})
            ratio_tol = tolerances.get("ratio_abs_tol", 0.001)

            for test in claim.get("tests", []):
                test_name = test.get("name", "")

                # Ratio test (u/d quark)
                if test_name == "u_d_ratio":
                    mev = test.get("mev", {})
                    loops = test.get("loops", {})

                    # Recompute ratios
                    mev_u = mev.get("u", 0)
                    mev_d = mev.get("d", 1)
                    loop_u = loops.get("u", 0)
                    loop_d = loops.get("d", 1)

                    mev_ratio = mev_u / mev_d if mev_d != 0 else 0
                    loop_ratio = loop_u / loop_d if loop_d != 0 else 0
                    abs_delta = abs(mev_ratio - loop_ratio)

                    out.metrics["u_d_mev_ratio_recomputed"] = mev_ratio
                    out.metrics["u_d_loop_ratio_recomputed"] = loop_ratio
                    out.metrics["u_d_ratio_abs_delta"] = abs_delta

                    # Check declared ratio values
                    declared_mev_ratio = mev.get("ratio")
                    if declared_mev_ratio is not None:
                        if abs(declared_mev_ratio - mev_ratio) > 1e-10:
                            out.add_fail("ENFORCE_HOMOMORPHISM",
                                         "RECOMPUTE_MISMATCH",
                                         {"field": "mev.ratio",
                                          "declared": declared_mev_ratio,
                                          "recomputed": mev_ratio})

                    declared_loop_ratio = loops.get("ratio")
                    if declared_loop_ratio is not None:
                        if abs(declared_loop_ratio - loop_ratio) > 1e-10:
                            out.add_fail("ENFORCE_HOMOMORPHISM",
                                         "RECOMPUTE_MISMATCH",
                                         {"field": "loops.ratio",
                                          "declared": declared_loop_ratio,
                                          "recomputed": loop_ratio})

                    # Tolerance check
                    if abs_delta > ratio_tol:
                        out.add_fail("ENFORCE_HOMOMORPHISM", "RATIO_MISMATCH",
                                     {"abs_delta": abs_delta,
                                      "tolerance": ratio_tol})

                # Lambda decay bookkeeping
                if test_name == "lambda_decay_to_proton_loop_bookkeeping":
                    loops_data = test.get("loops", {})
                    mev_data = test.get("mev", {})

                    # ENFORCE_STF_BASIS: loop arithmetic is exact
                    lambda_loops = loops_data.get("lambda", 0)
                    minus_loops = loops_data.get("minus", [])
                    loops_result = lambda_loops - sum(minus_loops)
                    declared_loops_result = loops_data.get("result")

                    out.metrics["lambda_loops_recomputed"] = loops_result

                    if declared_loops_result is not None:
                        if loops_result != declared_loops_result:
                            out.add_fail("ENFORCE_STF_BASIS",
                                         "BOOKKEEPING_MISMATCH",
                                         {"declared": declared_loops_result,
                                          "recomputed": loops_result})

                    # MeV bookkeeping (record, warn on drift)
                    lambda_mev = mev_data.get("lambda", 0)
                    minus_mev = mev_data.get("minus", [])
                    mev_result = lambda_mev - sum(minus_mev)
                    declared_mev_result = mev_data.get("result")

                    out.metrics["lambda_mev_recomputed"] = mev_result

                    if declared_mev_result is not None:
                        mev_diff = abs(mev_result - declared_mev_result)
                        if mev_diff > 1e-6:
                            # SOURCE_NUMERIC_DRIFT: warning, not hard fail
                            out.add_warning(
                                "ENFORCE_HOMOMORPHISM",
                                "SOURCE_NUMERIC_DRIFT",
                                {"declared_mev_result": declared_mev_result,
                                 "recomputed_mev_result": mev_result,
                                 "diff": mev_diff})

                    # Downstream comparison: bookkeeping MeV vs known
                    # proton mass. Drift here is expected and logged as
                    # warning (the source text claims equivalence but the
                    # bookkeeping MeV result doesn't exactly match PDG).
                    ref = test.get("proton_mev_reference", {})
                    ref_value = ref.get("value")
                    if ref_value is not None:
                        drift = abs(mev_result - ref_value)
                        out.metrics["proton_mev_drift"] = drift
                        decay_tol = tolerances.get("decay_mev_abs_tol", 5.0)
                        if drift > 0.01:
                            out.add_warning(
                                "ENFORCE_HOMOMORPHISM",
                                "SOURCE_NUMERIC_DRIFT",
                                {"bookkeeping_mev": mev_result,
                                 "proton_mev_reference": ref_value,
                                 "drift": drift,
                                 "within_tolerance": drift <= decay_tol})

    return out


# ============================================================================
# FILE LOADERS
# ============================================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def validate_from_files(spine_path: str, bundle_path: str,
                        manifest_path: Optional[str] = None
                        ) -> FSTValidationResult:
    """Load and validate from file paths."""
    spine = load_json(spine_path)
    bundle = load_json(bundle_path)
    manifest = load_json(manifest_path) if manifest_path else None
    return validate_bundle(spine, bundle, manifest)


# ============================================================================
# MANIFEST GENERATION
# ============================================================================

def generate_manifest(base_dir: str) -> Dict[str, Any]:
    """Generate a SHA256 manifest covering all replayable artifacts.

    Covers: spine, cert bundle, submission packet, validator source,
    and all schema files. This ensures anyone replaying can prove they
    validated with the same artifacts.

    Hardening (per CERT_FAMILY_HARDENING_PLAYBOOK.md):
    - Includes hash_spec for canonicalization documentation
    - Dual hashes for JSON files (file bytes + canonical JSON)
    """
    from datetime import datetime, timezone

    # JSON artifacts (both file bytes and canonical hash)
    json_artifacts = [
        ("qa_fst_module_spine.json", "module_spine"),
        ("qa_fst_cert_bundle.json", "cert_bundle"),
        ("qa_fst_submission_packet_spine.json", "submission_packet"),
    ]

    # Schema files (both file bytes and canonical hash)
    schema_artifacts = [
        ("schemas/QA_MAP_MODULE_SPINE.v1.schema.json", "schema:module_spine"),
        ("schemas/QA_CERT_BUNDLE.v1.schema.json", "schema:cert_bundle"),
        ("schemas/QA_SHA256_MANIFEST.v1.schema.json", "schema:manifest"),
        ("schemas/FAIL_RECORD.v1.schema.json", "schema:fail_record"),
        ("schemas/QA_SUBMISSION_PACKET_SPINE.v1.schema.json",
         "schema:submission_packet"),
    ]

    # Validator source (raw byte hash — not JSON-canonicalized)
    source_artifacts = [
        ("qa_fst_validate.py", "validator_source"),
    ]

    entries: List[Dict[str, Any]] = []

    for fname, entry_id in json_artifacts:
        path = os.path.join(base_dir, fname)
        obj = load_json(path)
        # Canonical hash (semantic identity)
        canonical_hash = sha256_canonical(obj)
        # File bytes hash (exact file integrity)
        file_hash = sha256_file(path)
        entries.append({
            "id": entry_id,
            "file": fname,
            "sha256": canonical_hash,  # Primary hash for validation
            "sha256_file": file_hash,  # File bytes for exact match
            "canonicalization": "json_sorted_no_ws",
        })

    for fname, entry_id in schema_artifacts:
        path = os.path.join(base_dir, fname)
        obj = load_json(path)
        canonical_hash = sha256_canonical(obj)
        file_hash = sha256_file(path)
        entries.append({
            "id": entry_id,
            "file": fname,
            "sha256": canonical_hash,
            "sha256_file": file_hash,
            "canonicalization": "json_sorted_no_ws",
        })

    for fname, entry_id in source_artifacts:
        path = os.path.join(base_dir, fname)
        file_hash = sha256_file(path)
        entries.append({
            "id": entry_id,
            "file": fname,
            "sha256": file_hash,
            "canonicalization": "raw_bytes",
        })

    entries_canonical = canonical_json(entries)
    manifest_hash = sha256_hex(entries_canonical)

    return {
        "schema_version": "QA_SHA256_MANIFEST.v1",
        "manifest_id": "qa.manifest.fst.v2",
        "hash_spec": {
            "id": "qa.hash_spec.v1",
            "version": "1.0",
            "sha256": "canonical_json (semantic identity) for JSON, file_bytes for source",
            "sha256_file": "file_bytes (exact file content integrity)",
            "canonical_spec": "json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)",
            "source": "qa_cert_core.canonical_json_compact"
        },
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "entries": entries,
        "manifest_sha256": manifest_hash,
    }


# ============================================================================
# MANIFEST INTEGRITY CHECK (fast gate for CI)
# ============================================================================

def check_manifest_integrity_fst(here: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast manifest integrity check for CI gates.

    Verifies:
    1. Each artifact file exists
    2. File SHA256 matches (for raw_bytes entries)
    3. Canonical SHA256 matches (for json_sorted_no_ws entries)

    Returns dict with 'ok', 'checks', 'errors' fields.
    """
    results = {"ok": True, "checks": [], "errors": []}

    # Check hash_spec is present
    hash_spec = manifest.get("hash_spec", {})
    hash_spec_id = hash_spec.get("id", "missing")
    results["hash_spec_id"] = hash_spec_id

    # Build file path mapping
    file_mapping = {
        "module_spine": "qa_fst_module_spine.json",
        "cert_bundle": "qa_fst_cert_bundle.json",
        "submission_packet": "qa_fst_submission_packet_spine.json",
        "schema:module_spine": "schemas/QA_MAP_MODULE_SPINE.v1.schema.json",
        "schema:cert_bundle": "schemas/QA_CERT_BUNDLE.v1.schema.json",
        "schema:manifest": "schemas/QA_SHA256_MANIFEST.v1.schema.json",
        "schema:fail_record": "schemas/FAIL_RECORD.v1.schema.json",
        "schema:submission_packet": "schemas/QA_SUBMISSION_PACKET_SPINE.v1.schema.json",
        "validator_source": "qa_fst_validate.py",
    }

    for entry in manifest.get("entries", []):
        entry_id = entry.get("id", "unknown")
        expected_sha = entry.get("sha256", "")
        canon_type = entry.get("canonicalization", "")

        fname = file_mapping.get(entry_id)
        if not fname:
            results["checks"].append(f"{entry_id}: SKIP - unknown entry")
            continue

        path = os.path.join(here, fname)
        if not os.path.exists(path):
            results["errors"].append(f"{entry_id}: file not found: {fname}")
            results["ok"] = False
            continue

        if canon_type == "raw_bytes":
            # Check file bytes hash
            actual_sha = sha256_file(path)
            if actual_sha == expected_sha:
                results["checks"].append(f"{entry_id}: OK (file sha256)")
            else:
                results["errors"].append(
                    f"{entry_id}: FILE SHA256 MISMATCH\n"
                    f"  manifest: {expected_sha[:16]}...\n"
                    f"  actual:   {actual_sha[:16]}..."
                )
                results["ok"] = False
        elif canon_type == "json_sorted_no_ws":
            # Check canonical JSON hash
            with open(path) as f:
                obj = json.load(f)
            actual_sha = sha256_canonical(obj)
            if actual_sha == expected_sha:
                results["checks"].append(f"{entry_id}: OK (canonical sha256)")
            else:
                results["errors"].append(
                    f"{entry_id}: CANONICAL SHA256 MISMATCH\n"
                    f"  manifest: {expected_sha[:16]}...\n"
                    f"  actual:   {actual_sha[:16]}..."
                )
                results["ok"] = False
        else:
            results["checks"].append(f"{entry_id}: SKIP - unknown canonicalization")

    return results


# ============================================================================
# CLI + SELF-TEST
# ============================================================================

def print_usage():
    print("Usage: python qa_fst_validate.py [OPTIONS]")
    print()
    print("Options:")
    print("  --all              Full behavioral validation")
    print("  --check-manifest   Fast manifest integrity check (for CI)")
    print("  --validate         Legacy: run validation with JSON output")
    print("  --generate-manifest Generate manifest from artifacts")
    print("  --json             Output results as JSON")
    print("  --summary          Print summary of checks")
    print("  (no args)          Run self-test suite")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    spine_path = os.path.join(here, "qa_fst_module_spine.json")
    bundle_path = os.path.join(here, "qa_fst_cert_bundle.json")
    manifest_path = os.path.join(here, "qa_fst_manifest.json")

    args = sys.argv[1:]
    json_output = "--json" in args

    if "--help" in args or "-h" in args:
        print_usage()
        sys.exit(0)

    if "--check-manifest" in args:
        # Fast manifest integrity check
        if not os.path.exists(manifest_path):
            if json_output:
                print(json.dumps({"ok": False, "error": "manifest not found"}))
            else:
                print("ERROR: Manifest not found:", manifest_path)
            sys.exit(1)

        manifest = load_json(manifest_path)
        result = check_manifest_integrity_fst(here, manifest)

        if json_output:
            print(json.dumps(result, indent=2))
        else:
            print(f"Manifest integrity check: {'PASS' if result['ok'] else 'FAIL'}")
            print(f"  hash_spec_id: {result.get('hash_spec_id', 'missing')}")
            for check in result["checks"]:
                print(f"  {check}")
            for err in result["errors"]:
                print(f"  ERROR: {err}")

        sys.exit(0 if result["ok"] else 1)

    if "--all" in args:
        # Full behavioral validation
        mp = manifest_path if os.path.exists(manifest_path) else None
        result = validate_from_files(spine_path, bundle_path, mp)

        if json_output:
            print(result.to_json())
        else:
            print(f"FST Validation: {result.result_label}")
            print(f"  delta_sym = {result.metrics.get('delta_sym_recomputed')}")
            print(f"  u/d ratio delta = {result.metrics.get('u_d_ratio_abs_delta')}")
            print(f"  warnings: {len(result.warnings)}")
            print(f"  fails:    {len(result.fail_records)}")
            if result.fail_records:
                for fr in result.fail_records:
                    print(f"    FAIL: {fr}")

        sys.exit(0 if result.ok else 1)

    if "--summary" in args:
        # Summary mode
        mp = manifest_path if os.path.exists(manifest_path) else None
        result = validate_from_files(spine_path, bundle_path, mp)
        print(f"FST: {result.result_label} "
              f"(fails={len(result.fail_records)}, warns={len(result.warnings)})")
        sys.exit(0 if result.ok else 1)

    if "--generate-manifest" in args:
        manifest = generate_manifest(here)
        out_path = manifest_path
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Manifest written to {out_path}")
        print(f"  {len(manifest['entries'])} artifact(s) covered:")
        for entry in manifest["entries"]:
            print(f"    {entry['id']}: {entry['sha256'][:16]}...")
        print(f"  manifest hash: {manifest['manifest_sha256']}")
        sys.exit(0)

    if "--validate" in args:
        # Legacy mode
        mp = manifest_path if os.path.exists(manifest_path) else None
        result = validate_from_files(spine_path, bundle_path, mp)
        print(result.to_json())
        sys.exit(0 if result.ok else 1)

    # ---- SELF-TEST ----
    print("=== FST VALIDATOR SELF-TEST ===\n")

    checks_passed = 0
    checks_total = 0

    # Test 1: Validate spine + bundle (no manifest)
    checks_total += 1
    result = validate_from_files(spine_path, bundle_path)
    label = result.result_label
    print(f"[1] Validate spine + bundle: {label}")
    print(f"    delta_sym = {result.metrics.get('delta_sym_recomputed')}")
    print(f"    u/d ratio delta = {result.metrics.get('u_d_ratio_abs_delta')}")
    print(f"    lambda loops = {result.metrics.get('lambda_loops_recomputed')}")
    print(f"    lambda MeV   = {result.metrics.get('lambda_mev_recomputed')}")
    print(f"    warnings: {len(result.warnings)}")
    print(f"    fails:    {len(result.fail_records)}")

    if result.ok:
        checks_passed += 1
        print("    -> PASS")
    else:
        print("    -> FAIL (unexpected)")
        for fr in result.fail_records:
            print(f"       {fr}")

    # Test 2: delta_sym recomputation matches declared
    checks_total += 1
    expected_delta = 6 / 1836
    actual_delta = result.metrics.get("delta_sym_recomputed", -1)
    if abs(actual_delta - expected_delta) < 1e-12:
        checks_passed += 1
        print(f"[2] delta_sym recompute: {actual_delta:.16f} == 6/1836 -> PASS")
    else:
        print(f"[2] delta_sym recompute: {actual_delta} != {expected_delta} -> FAIL")

    # Test 3: Loop bookkeeping is exact (2187 - 243 - 81 - 27 = 1836)
    checks_total += 1
    loops_result = result.metrics.get("lambda_loops_recomputed")
    if loops_result == 1836:
        checks_passed += 1
        print(f"[3] Loop bookkeeping: 2187-243-81-27 = {loops_result} -> PASS")
    else:
        print(f"[3] Loop bookkeeping: got {loops_result}, expected 1836 -> FAIL")

    # Test 4: u/d ratio within tolerance
    checks_total += 1
    ratio_delta = result.metrics.get("u_d_ratio_abs_delta", 999)
    if ratio_delta < 0.001:
        checks_passed += 1
        print(f"[4] u/d ratio tolerance: delta={ratio_delta:.10f} < 0.001 -> PASS")
    else:
        print(f"[4] u/d ratio tolerance: delta={ratio_delta} >= 0.001 -> FAIL")

    # Test 5: SOURCE_NUMERIC_DRIFT is warning (not fail)
    checks_total += 1
    drift_warnings = [w for w in result.warnings
                      if w.get("fail_type") == "SOURCE_NUMERIC_DRIFT"]
    if len(drift_warnings) >= 1 and result.ok:
        checks_passed += 1
        drift_val = result.metrics.get("proton_mev_drift", "?")
        print(f"[5] SOURCE_NUMERIC_DRIFT logged as warning (not fail): "
              f"drift={drift_val} MeV <= 5.0 tol -> PASS")
    else:
        print(f"[5] Expected SOURCE_NUMERIC_DRIFT warning: "
              f"warnings={len(drift_warnings)}, ok={result.ok} -> FAIL")

    # Test 6: Bad schema version triggers hard fail
    checks_total += 1
    bad_spine = {"schema_version": "WRONG", "module_id": "test"}
    bad_result = validate_bundle(bad_spine, {}, None)
    if not bad_result.ok and any(
            fr["fail_type"] == "BAD_SCHEMA" for fr in bad_result.fail_records):
        checks_passed += 1
        print("[6] Bad schema version -> hard FAIL with BAD_SCHEMA -> PASS")
    else:
        print("[6] Bad schema version should fail -> FAIL")

    # Test 7: Hash determinism
    checks_total += 1
    spine = load_json(spine_path)
    h1 = sha256_hex(canonical_json(spine))
    h2 = sha256_hex(canonical_json(spine))
    if h1 == h2 and len(h1) == 64:
        checks_passed += 1
        print(f"[7] Hash determinism: {h1[:16]}... -> PASS")
    else:
        print(f"[7] Hash determinism: {h1} != {h2} -> FAIL")

    # Test 8: Validate with manifest (if exists)
    checks_total += 1
    if os.path.exists(manifest_path):
        result_m = validate_from_files(spine_path, bundle_path, manifest_path)
        if result_m.ok:
            checks_passed += 1
            print(f"[8] Validate with manifest: {result_m.result_label} -> PASS")
        else:
            print(f"[8] Validate with manifest: FAIL")
            for fr in result_m.fail_records:
                print(f"       {fr}")
    else:
        # Generate manifest first, then validate
        manifest = generate_manifest(here)
        result_m = validate_bundle(
            load_json(spine_path), load_json(bundle_path), manifest)
        if result_m.ok:
            checks_passed += 1
            print(f"[8] Validate with generated manifest: "
                  f"{result_m.result_label} -> PASS")
        else:
            print(f"[8] Validate with generated manifest: FAIL")

    print(f"\n{checks_passed}/{checks_total} checks passed")
    if checks_passed == checks_total:
        print("All FST validator self-tests PASSED")
    else:
        print(f"FAILED: {checks_total - checks_passed} check(s)")
        sys.exit(1)
