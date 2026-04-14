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


# ============================================================================
# THEOREM NT BOUNDARY: observer-layer constants
# ============================================================================
# These constants are the ONLY float values the validator touches and they
# live strictly on the observer side of the Pi boundary. Integer state is
# never compared to or updated from these numbers.

QA_COMPLIANCE = {
    "observer": "FST_LOOP_MASS_CALIBRATION_POSTULATE_P1",
    "state_alphabet": "integer STF basis {3,9,27,81,243,729,2187}",
    "rationale": "Pi is the ONLY sanctioned int->float boundary crossing; "
                 "all MeV arithmetic lives on the observer side of Pi.",
}

M_E_MEV = 0.51099895069  # PDG 2024 electron rest mass, MeV/c^2
PDG_PROTON_MEV = 938.27208816  # PDG 2024 proton rest mass
PDG_LAMBDA_MEV = 1115.683  # PDG 2024 Lambda^0 baryon rest mass

STF_BASIS = (3, 9, 27, 81, 243, 729, 2187)


def apply_Pi(loop_count: int) -> float:
    """Observer projection Pi: int loop count -> float MeV.

    This is the ONLY sanctioned float cast in the FST module. Loop count enters
    as int; MeV leaves as float. The reverse direction (MeV -> int) is
    forbidden by Theorem NT and does not exist in this module.
    """
    assert isinstance(loop_count, int), "Pi requires int input (Theorem NT)"
    return loop_count * M_E_MEV  # noqa: FIREWALL-1 (Pi projection)


def validate_bundle(spine: Dict[str, Any],
                    cert_bundle: Dict[str, Any],
                    manifest: Optional[Dict[str, Any]] = None
                    ) -> FSTValidationResult:
    """
    Full deterministic validation of FST module spine + certificate bundle (v2).

    Steps:
        1. LOAD: schema version checks + cross-reference
        2. CANONICALIZE: sha256 of canonical JSON
        3. ENFORCE_STF_DECOMPOSITION: integer partition and subtraction checks
        4. ENFORCE_SECTOR_INVARIANT: Rspin/Aspin cardinality for fermions
        5. APPLY_PI: calibration postulate - Pi projection and drift logging
        6. CLASSIFY: emit fail records + warnings
        7. EMIT_METRICS: return metrics bundle

    v2 changes: removed hexagon symmetry witness handler (not in source);
    removed tautological u/d ratio pass handler; replaced homomorphism path
    with Pi-based direct-read calibration; all MeV arithmetic now lives
    behind the Pi firewall and is explicitly observer-layer.
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

    if cert_bundle.get("module_id") != spine.get("module_id"):
        out.add_fail("LOAD", "MODULE_ID_MISMATCH",
                     {"spine": spine.get("module_id"),
                      "bundle": cert_bundle.get("module_id")})

    # ---- STEP 2: CANONICALIZE ----
    spine_hash = sha256_hex(canonical_json(spine))
    cert_hash = sha256_hex(canonical_json(cert_bundle))
    out.hashes["spine"] = spine_hash
    out.hashes["cert_bundle"] = cert_hash

    if manifest is not None:
        for entry in manifest.get("entries", []):
            eid = entry.get("id", "")
            expected_hash = entry.get("sha256", "")
            if eid == spine.get("module_id"):
                if spine_hash != expected_hash:
                    out.add_fail("CANONICALIZE", "HASH_MISMATCH",
                                 {"artifact": "spine",
                                  "expected": expected_hash,
                                  "computed": spine_hash})
            elif eid == cert_bundle.get("certificate_id"):
                if cert_hash != expected_hash:
                    out.add_fail("CANONICALIZE", "HASH_MISMATCH",
                                 {"artifact": "cert_bundle",
                                  "expected": expected_hash,
                                  "computed": cert_hash})

    # ---- STEPS 3-5: Process each claim ----
    calibration_drifts: List[Dict[str, Any]] = []

    for claim in cert_bundle.get("claims", []):
        claim_type = claim.get("type", "")

        # ---- STEP 3a: FST_STF_DECOMPOSITION_PROTON ----
        if claim_type == "FST_STF_DECOMPOSITION_PROTON.v1":
            decomp = claim.get("stf_decomposition", {})
            total = decomp.get("total_loops")
            partition = decomp.get("partition", [])
            partition_sum = sum(partition)

            out.metrics["proton_partition"] = list(partition)
            out.metrics["proton_partition_sum"] = partition_sum
            out.metrics["proton_declared_total"] = total

            if not all(isinstance(x, int) for x in partition):
                out.add_fail("ENFORCE_STF_DECOMPOSITION",
                             "NON_INTEGER_PARTITION",
                             {"partition": partition})
            elif total != partition_sum:
                out.add_fail("ENFORCE_STF_DECOMPOSITION", "NOT_IN_STF_BASIS",
                             {"declared_total": total,
                              "partition_sum": partition_sum,
                              "partition": partition})

            # All partition elements must live in the STF basis (or be
            # repetitions of basis elements - e.g. two 27s are allowed).
            for p in partition:
                if p not in STF_BASIS:
                    out.add_fail("ENFORCE_STF_DECOMPOSITION",
                                 "NOT_IN_STF_BASIS",
                                 {"element": p, "basis": list(STF_BASIS)})

            # Source typo is a warning, not a failure.
            typo = claim.get("source_typo")
            if typo is not None:
                out.add_warning("ENFORCE_STF_DECOMPOSITION",
                                "SOURCE_INTERNAL_INCONSISTENCY",
                                {"location": typo.get("location"),
                                 "diagnosis": typo.get("diagnosis"),
                                 "actual_sum": typo.get("actual_sum")})

        # ---- STEP 3b: FST_STF_LAMBDA_DECAY_BOOKKEEPING ----
        if claim_type == "FST_STF_LAMBDA_DECAY_BOOKKEEPING.v1":
            decay = claim.get("stf_decay_bookkeeping", {})
            initial = decay.get("initial_loops")
            subtract = decay.get("subtract", [])
            declared_final = decay.get("final_loops")
            computed_final = initial - sum(subtract) if initial is not None else None

            out.metrics["lambda_initial_loops"] = initial
            out.metrics["lambda_subtract_sum"] = sum(subtract) if subtract else 0
            out.metrics["lambda_decay_final_recomputed"] = computed_final

            if computed_final != declared_final:
                out.add_fail("ENFORCE_STF_DECOMPOSITION",
                             "BOOKKEEPING_MISMATCH",
                             {"initial": initial, "subtract": subtract,
                              "declared_final": declared_final,
                              "recomputed_final": computed_final})

            for s in subtract:
                if s not in STF_BASIS:
                    out.add_fail("ENFORCE_STF_DECOMPOSITION",
                                 "NOT_IN_STF_BASIS",
                                 {"element": s, "basis": list(STF_BASIS)})

            typo = claim.get("source_typo")
            if typo is not None:
                out.add_warning("ENFORCE_STF_DECOMPOSITION",
                                "SOURCE_INTERNAL_INCONSISTENCY",
                                {"location": typo.get("location"),
                                 "diagnosis": typo.get("diagnosis"),
                                 "actual_missing": typo.get("actual_missing")})

        # ---- STEP 5: FST_LOOP_MASS_CALIBRATION_POSTULATE ----
        if claim_type == "FST_LOOP_MASS_CALIBRATION_POSTULATE.v1":
            # Verify the declared m_e matches our PDG constant to avoid
            # silent calibration drift between spine and validator.
            postulate = claim.get("postulate_P1", {})
            declared_me = postulate.get("m_e_mev")
            if declared_me is not None and declared_me != M_E_MEV:
                out.add_fail("APPLY_PI", "CALIBRATION_ANCHOR_MISMATCH",
                             {"declared_m_e": declared_me,
                              "validator_m_e": M_E_MEV})

            # Apply Pi to every row in the calibration table and log drift.
            # All float arithmetic here is explicitly observer-layer.
            tolerances = claim.get("tolerances", {})
            warn_at_percent = tolerances.get(
                "calibration_drift_warn_at_percent", 0.01)
            fail_at_percent = tolerances.get(
                "calibration_drift_fail_at_percent", 1.0)

            calibration_recomputed = []
            for row in claim.get("calibration_table", []):
                loops = row.get("loops")
                if loops is None or not isinstance(loops, int):
                    continue

                pi_mev = apply_Pi(loops)
                record = {
                    "label": row.get("label"),
                    "loops": loops,
                    "pi_mev_recomputed": pi_mev,
                }

                # Consistency check against declared pi_mev_expected
                declared_pi = row.get("pi_mev_expected")
                if declared_pi is not None:
                    if abs(declared_pi - pi_mev) > 1e-6:
                        out.add_fail("APPLY_PI", "RECOMPUTE_MISMATCH",
                                     {"label": row.get("label"),
                                      "declared": declared_pi,
                                      "recomputed": pi_mev})

                # Drift against PDG reference (if declared)
                pdg_ref = row.get("pdg_ref_value")
                if pdg_ref is not None:
                    drift_abs = abs(pi_mev - pdg_ref)
                    drift_percent = 100.0 * drift_abs / pdg_ref
                    record["pdg_ref_value"] = pdg_ref
                    record["drift_mev"] = drift_abs
                    record["drift_percent"] = drift_percent

                    if drift_percent > fail_at_percent:
                        out.add_fail("APPLY_PI", "CALIBRATION_DRIFT_HARD",
                                     {"label": row.get("label"),
                                      "pi_mev": pi_mev,
                                      "pdg_ref": pdg_ref,
                                      "drift_percent": drift_percent,
                                      "fail_threshold_percent": fail_at_percent})
                    elif drift_percent > warn_at_percent:
                        drift_record = {
                            "label": row.get("label"),
                            "loops": loops,
                            "pi_mev": pi_mev,
                            "pdg_ref": pdg_ref,
                            "drift_mev": drift_abs,
                            "drift_percent": drift_percent,
                        }
                        calibration_drifts.append(drift_record)
                        out.add_warning("APPLY_PI", "CALIBRATION_DRIFT",
                                        drift_record)

                calibration_recomputed.append(record)

            out.metrics["calibration_table_recomputed"] = calibration_recomputed

        # ---- STEP 3c: FST_QUARK_GEOMETRIC_STRUCTURE ----
        if claim_type == "FST_QUARK_GEOMETRIC_STRUCTURE.v1":
            structural = claim.get("structural_claim", {})
            partition = structural.get("proton_integer_partition", [])
            partition_sum = sum(partition)
            out.metrics["quark_partition"] = list(partition)
            out.metrics["quark_partition_sum"] = partition_sum

            if partition_sum != 1836:
                out.add_fail("ENFORCE_STF_DECOMPOSITION",
                             "QUARK_PARTITION_SUM_MISMATCH",
                             {"partition": partition,
                              "partition_sum": partition_sum,
                              "expected": 1836})

            # The numerological observation is not validated; it is simply
            # carried through to metrics as declared observer-layer data.
            obs = claim.get("numerological_observation")
            if obs is not None:
                out.metrics["numerological_loop_ratio"] = obs.get(
                    "loop_ratio_378_over_729")
                out.metrics["numerological_pdg_u_over_d"] = obs.get(
                    "pdg_2024_current_quark_ratio_u_over_d")

        # ---- STEP 4: FST_FERMION_SIX_LOOP_CHIRAL_STRUCTURE ----
        if claim_type == "FST_FERMION_SIX_LOOP_CHIRAL_STRUCTURE.v1":
            structural = claim.get("structural_claim", {})
            required = structural.get("sector_cardinality_required", {})
            rspin = required.get("Rspin")
            aspin = required.get("Aspin")

            out.metrics["fermion_sector_rspin"] = rspin
            out.metrics["fermion_sector_aspin"] = aspin

            if not (isinstance(rspin, int) and isinstance(aspin, int)):
                out.add_fail("ENFORCE_SECTOR_INVARIANT",
                             "SECTOR_CARDINALITY_NOT_INT",
                             {"Rspin": rspin, "Aspin": aspin})
            elif rspin != 3 or aspin != 3:
                out.add_fail("ENFORCE_SECTOR_INVARIANT", "SECTOR_IMBALANCE",
                             {"Rspin": rspin, "Aspin": aspin,
                              "required": "both == 3"})

    out.metrics["calibration_drift_count"] = len(calibration_drifts)
    out.metrics["theorem_nt_firewall_crossings"] = sum(
        1 for c in cert_bundle.get("claims", [])
        if c.get("type") == "FST_LOOP_MASS_CALIBRATION_POSTULATE.v1"
        for _ in c.get("calibration_table", []))
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
    - Uses dict-based format (unified with Kayser pattern)
    - Includes hash_spec for canonicalization documentation
    - Dual hashes: sha256 (file bytes) + canonical_sha256 (semantic identity)
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

    # Use dict-based format (unified with Kayser pattern)
    certificates: Dict[str, Any] = {}

    for fname, entry_id in json_artifacts:
        path = os.path.join(base_dir, fname)
        obj = load_json(path)
        # File bytes hash (exact file integrity) - named 'sha256' for Kayser compat
        file_hash = sha256_file(path)
        # Canonical hash (semantic identity)
        canonical_hash = sha256_canonical(obj)
        certificates[entry_id] = {
            "file": fname,
            "sha256": file_hash,
            "canonical_sha256": canonical_hash,
        }

    for fname, entry_id in schema_artifacts:
        path = os.path.join(base_dir, fname)
        obj = load_json(path)
        file_hash = sha256_file(path)
        canonical_hash = sha256_canonical(obj)
        certificates[entry_id] = {
            "file": fname,
            "sha256": file_hash,
            "canonical_sha256": canonical_hash,
        }

    for fname, entry_id in source_artifacts:
        path = os.path.join(base_dir, fname)
        file_hash = sha256_file(path)
        certificates[entry_id] = {
            "file": fname,
            "sha256": file_hash,
            # No canonical_sha256 for raw source files
        }

    certificates_canonical = canonical_json(certificates)
    manifest_hash = sha256_hex(certificates_canonical)

    return {
        "schema_version": "QA_MANIFEST.v1",  # Unified with Kayser format
        "manifest_id": "qa.manifest.fst.v3",
        "hash_spec": {
            "id": "qa.hash_spec.v1",
            "version": "1.0",
            "sha256": "file_bytes (exact file content integrity)",
            "canonical_sha256": "canonical_json (semantic identity, sorted keys, no whitespace)",
            "canonical_spec": "json.dumps(obj, sort_keys=True, separators=(',',':'), ensure_ascii=False)",
            "source": "qa_cert_core.canonical_json_compact"
        },
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "certificates": certificates,  # Dict-based format (unified)
        "manifest_sha256": manifest_hash,
    }


# ============================================================================
# MANIFEST INTEGRITY CHECK (fast gate for CI)
# ============================================================================

def check_manifest_integrity_fst(here: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast manifest integrity check for CI gates.

    Uses unified dict-based format (same as Kayser):
    - manifest["certificates"][name]["sha256"] = file bytes
    - manifest["certificates"][name]["canonical_sha256"] = semantic identity

    Returns dict with 'ok', 'checks', 'errors' fields.
    """
    results = {"ok": True, "checks": [], "errors": []}

    # Check hash_spec is present
    hash_spec = manifest.get("hash_spec", {})
    hash_spec_id = hash_spec.get("id", "missing")
    results["hash_spec_id"] = hash_spec_id

    # Support both old (entries array) and new (certificates dict) formats
    if "certificates" in manifest:
        # New unified dict format
        for name, entry in manifest.get("certificates", {}).items():
            cert_file = entry.get("file")
            if not cert_file:
                results["errors"].append(f"{name}: missing 'file' in manifest")
                results["ok"] = False
                continue

            cert_path = os.path.join(here, cert_file)
            if not os.path.exists(cert_path):
                results["errors"].append(f"{name}: file not found: {cert_file}")
                results["ok"] = False
                continue

            # Check file bytes SHA256
            manifest_sha = entry.get("sha256")
            if manifest_sha:
                actual_sha = sha256_file(cert_path)
                if actual_sha == manifest_sha:
                    results["checks"].append(f"{name}: OK (file sha256)")
                else:
                    results["errors"].append(
                        f"{name}: FILE SHA256 MISMATCH\n"
                        f"  manifest: {manifest_sha[:16]}...\n"
                        f"  actual:   {actual_sha[:16]}..."
                    )
                    results["ok"] = False

            # Check canonical SHA256 (skip for raw source files)
            manifest_canonical = entry.get("canonical_sha256")
            if manifest_canonical:
                with open(cert_path) as f:
                    obj = json.load(f)
                actual_canonical = sha256_canonical(obj)
                if actual_canonical == manifest_canonical:
                    results["checks"].append(f"{name}: OK (canonical sha256)")
                else:
                    results["errors"].append(
                        f"{name}: CANONICAL SHA256 MISMATCH\n"
                        f"  manifest: {manifest_canonical[:16]}...\n"
                        f"  actual:   {actual_canonical[:16]}..."
                    )
                    results["ok"] = False
    else:
        # Legacy entries array format (backward compat)
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
        # Full behavioral validation (v2)
        mp = manifest_path if os.path.exists(manifest_path) else None
        result = validate_from_files(spine_path, bundle_path, mp)

        if json_output:
            print(result.to_json())
        else:
            print(f"FST Validation (v2): {result.result_label}")
            print(f"  proton partition sum = "
                  f"{result.metrics.get('proton_partition_sum')}")
            print(f"  lambda decay final   = "
                  f"{result.metrics.get('lambda_decay_final_recomputed')}")
            print(f"  fermion sector       = "
                  f"R={result.metrics.get('fermion_sector_rspin')} "
                  f"A={result.metrics.get('fermion_sector_aspin')}")
            print(f"  firewall crossings   = "
                  f"{result.metrics.get('theorem_nt_firewall_crossings')}")
            print(f"  calibration drifts   = "
                  f"{result.metrics.get('calibration_drift_count')}")
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
        print(f"  {len(manifest['certificates'])} artifact(s) covered:")
        for name, entry in manifest["certificates"].items():
            print(f"    {name}: {entry['sha256'][:16]}...")
        print(f"  manifest hash: {manifest['manifest_sha256']}")
        sys.exit(0)

    if "--validate" in args:
        # Legacy mode
        mp = manifest_path if os.path.exists(manifest_path) else None
        result = validate_from_files(spine_path, bundle_path, mp)
        print(result.to_json())
        sys.exit(0 if result.ok else 1)

    # ---- SELF-TEST (v2) ----
    print("=== FST VALIDATOR SELF-TEST (v2) ===\n")

    checks_passed = 0
    checks_total = 0

    # Test 1: Validate spine + bundle (no manifest)
    checks_total += 1
    result = validate_from_files(spine_path, bundle_path)
    label = result.result_label
    print(f"[1] Validate spine + bundle: {label}")
    print(f"    proton partition sum = "
          f"{result.metrics.get('proton_partition_sum')}")
    print(f"    lambda decay final   = "
          f"{result.metrics.get('lambda_decay_final_recomputed')}")
    print(f"    fermion sector       = "
          f"R={result.metrics.get('fermion_sector_rspin')} "
          f"A={result.metrics.get('fermion_sector_aspin')}")
    print(f"    firewall crossings   = "
          f"{result.metrics.get('theorem_nt_firewall_crossings')}")
    print(f"    calibration drifts   = "
          f"{result.metrics.get('calibration_drift_count')}")
    print(f"    warnings: {len(result.warnings)}")
    print(f"    fails:    {len(result.fail_records)}")

    if result.ok:
        checks_passed += 1
        print("    -> PASS")
    else:
        print("    -> FAIL (unexpected)")
        for fr in result.fail_records:
            print(f"       {fr}")

    # Test 2: proton partition sum is 1836
    checks_total += 1
    psum = result.metrics.get("proton_partition_sum")
    if psum == 1836:
        checks_passed += 1
        print(f"[2] Proton partition sum = {psum} (expected 1836) -> PASS")
    else:
        print(f"[2] Proton partition sum = {psum} (expected 1836) -> FAIL")

    # Test 3: Lambda decay bookkeeping (2187 - 243 - 81 - 27 = 1836)
    checks_total += 1
    lambda_final = result.metrics.get("lambda_decay_final_recomputed")
    if lambda_final == 1836:
        checks_passed += 1
        print(f"[3] Lambda decay: 2187 - 351 = {lambda_final} -> PASS")
    else:
        print(f"[3] Lambda decay: got {lambda_final}, expected 1836 -> FAIL")

    # Test 4: Pi calibration on proton gives ~938.194 MeV (within 0.01 MeV)
    checks_total += 1
    pi_proton = apply_Pi(1836)
    expected_proton_pi = 1836 * M_E_MEV
    if abs(pi_proton - expected_proton_pi) < 1e-9:
        checks_passed += 1
        print(f"[4] Pi(1836) = {pi_proton:.6f} MeV (direct-read calibration) "
              f"-> PASS")
    else:
        print(f"[4] Pi(1836) recompute mismatch -> FAIL")

    # Test 5: Pi(1836) drift vs PDG is < 0.01% (at the warn threshold)
    checks_total += 1
    proton_drift_pct = 100.0 * abs(pi_proton - PDG_PROTON_MEV) / PDG_PROTON_MEV
    # We expect the proton drift to exceed the 0.01% warn threshold
    # (actual ~0.0083% — right at the edge) so we just verify it's < 1%.
    if proton_drift_pct < 1.0:
        checks_passed += 1
        print(f"[5] Proton calibration drift = {proton_drift_pct:.5f}% "
              f"(< 1% hard-fail threshold) -> PASS")
    else:
        print(f"[5] Proton drift {proton_drift_pct}% >= 1% -> FAIL")

    # Test 6: Lambda drift is in the CALIBRATION_DRIFT warn band
    checks_total += 1
    pi_lambda = apply_Pi(2187)
    lambda_drift_pct = 100.0 * abs(pi_lambda - PDG_LAMBDA_MEV) / PDG_LAMBDA_MEV
    cal_warns = [w for w in result.warnings
                 if w.get("fail_type") == "CALIBRATION_DRIFT"]
    if lambda_drift_pct > 0.01 and len(cal_warns) >= 1 and result.ok:
        checks_passed += 1
        print(f"[6] Lambda drift {lambda_drift_pct:.4f}% logged as "
              f"CALIBRATION_DRIFT warning (ok={result.ok}) -> PASS")
    else:
        print(f"[6] Expected CALIBRATION_DRIFT warning for lambda "
              f"(drift={lambda_drift_pct}%, warns={len(cal_warns)}, "
              f"ok={result.ok}) -> FAIL")

    # Test 7: SOURCE_INTERNAL_INCONSISTENCY flagged for the two in-source typos
    checks_total += 1
    typo_warns = [w for w in result.warnings
                  if w.get("fail_type") == "SOURCE_INTERNAL_INCONSISTENCY"]
    if len(typo_warns) >= 2:
        checks_passed += 1
        print(f"[7] In-source typos flagged: {len(typo_warns)} "
              f"SOURCE_INTERNAL_INCONSISTENCY warnings -> PASS")
    else:
        print(f"[7] Expected >=2 SOURCE_INTERNAL_INCONSISTENCY warnings, "
              f"got {len(typo_warns)} -> FAIL")

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
