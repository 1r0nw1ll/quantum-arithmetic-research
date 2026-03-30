#!/usr/bin/env python3
"""QA Security Competency Cert family [124] validator — QA_SECURITY_COMPETENCY_CERT.v1

Extends [123] (QA Agent Competency) with immune system structure.
Certifies that a security algorithm profile is structurally valid and
quantum-resilience-complete.

Schema: QA_SECURITY_COMPETENCY_CERT.v1
Fields:
  schema_version       "QA_SECURITY_COMPETENCY_CERT.v1"
  algorithm_name       str — unique algorithm identifier
  security_role        str — identity | membrane | integrity | self_nonself | healing | collective
  immune_function      str — detection | containment | recovery
  goal                 str — what the algorithm achieves (≥10 chars)
  orbit_signature      str — cosmos | satellite | singularity | mixed
  levin_cell_type      str — stem | progenitor | differentiated
  pq_readiness         str — fips_final | in_progress | classical_only | hybrid_transitional
  pq_migration_path    str — migration target (required when security_role=identity/membrane
                             AND pq_readiness=classical_only; may be "" otherwise)
  nist_fips            str — FIPS designation if fips_final (e.g. "FIPS 203"), else ""
  failure_modes        list[str] — ≥1 structural failure condition
  composition_rules    list[str] — ≥1 composition constraint
  parent_cert          str — parent cert this profile maps to
  result               "PASS" | "FAIL"

Validator checks:
  SC1  schema_version matches exactly                          → SCHEMA_VERSION_MISMATCH
  SC2  security_role is a known value                         → UNKNOWN_SECURITY_ROLE
  SC3  immune_function is a known value                       → UNKNOWN_IMMUNE_FUNCTION
  SC4  pq_readiness is a known value                          → UNKNOWN_PQ_READINESS
  SC5  identity/membrane + classical_only → pq_migration_path → PQ_MIGRATION_REQUIRED
  SC6  fips_final → nist_fips non-empty                       → MISSING_FIPS_DESIGNATION
  SC7  failure_modes is nonempty list                         → EMPTY_FAILURE_MODES
  SC8  composition_rules is nonempty list                     → EMPTY_COMPOSITION_RULES
  SC9  orbit_signature matches levin_cell_type                → CELL_ORBIT_MISMATCH
       (differentiated ↔ cosmos, progenitor ↔ satellite/mixed, stem ↔ singularity)
  SC10 goal is at least 10 characters                         → GOAL_TOO_SHORT
  SC11 result matches actual validation outcome               → RESULT_MISMATCH

Usage:
  python qa_security_competency_cert_validate.py --self-test
  python qa_security_competency_cert_validate.py --file fixtures/scc_pass_ml_kem.json
"""

QA_COMPLIANCE = "cert_validator — validates cert JSON structure, no empirical QA state machine"


import json
import sys
import argparse
from pathlib import Path

SCHEMA_VERSION = "QA_SECURITY_COMPETENCY_CERT.v1"

KNOWN_SECURITY_ROLES  = frozenset(["identity", "membrane", "integrity",
                                    "self_nonself", "healing", "collective"])
KNOWN_IMMUNE_FUNCTIONS = frozenset(["detection", "containment", "recovery"])
KNOWN_PQ_READINESS    = frozenset(["fips_final", "in_progress",
                                    "classical_only", "hybrid_transitional"])
KNOWN_ORBIT_SIGNATURES = frozenset(["cosmos", "satellite", "singularity", "mixed"])
KNOWN_LEVIN_CELL_TYPES = frozenset(["stem", "progenitor", "differentiated"])

# Roles that require a PQ migration path if pq_readiness = classical_only
PQ_SENSITIVE_ROLES = frozenset(["identity", "membrane"])

# Cell type → allowed orbit signatures (inherited from [123])
CELL_ORBIT_RULES = {
    "differentiated": frozenset(["cosmos"]),
    "progenitor":     frozenset(["satellite", "mixed"]),
    "stem":           frozenset(["singularity"]),
}


def validate_cert(cert: dict) -> list[str]:
    """Return list of error codes. Empty list = PASS."""
    errors = []

    # SC1 — schema version
    if cert.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"SCHEMA_VERSION_MISMATCH: expected '{SCHEMA_VERSION}', "
                      f"got '{cert.get('schema_version')}'")

    # SC2 — security_role
    role = cert.get("security_role", "")
    if role not in KNOWN_SECURITY_ROLES:
        errors.append(f"UNKNOWN_SECURITY_ROLE: '{role}' not in {sorted(KNOWN_SECURITY_ROLES)}")

    # SC3 — immune_function
    immune = cert.get("immune_function", "")
    if immune not in KNOWN_IMMUNE_FUNCTIONS:
        errors.append(f"UNKNOWN_IMMUNE_FUNCTION: '{immune}' not in {sorted(KNOWN_IMMUNE_FUNCTIONS)}")

    # SC4 — pq_readiness
    pq = cert.get("pq_readiness", "")
    if pq not in KNOWN_PQ_READINESS:
        errors.append(f"UNKNOWN_PQ_READINESS: '{pq}' not in {sorted(KNOWN_PQ_READINESS)}")

    # SC5 — PQ migration required for classical identity/membrane
    if role in PQ_SENSITIVE_ROLES and pq == "classical_only":
        migration = cert.get("pq_migration_path", "")
        if not migration or not migration.strip():
            errors.append(
                f"PQ_MIGRATION_REQUIRED: security_role='{role}' + pq_readiness='classical_only' "
                f"requires a non-empty pq_migration_path (quantum resilience invariant)"
            )

    # SC6 — FIPS designation required for fips_final
    if pq == "fips_final":
        fips = cert.get("nist_fips", "")
        if not fips or not fips.strip():
            errors.append(
                "MISSING_FIPS_DESIGNATION: pq_readiness='fips_final' requires "
                "non-empty nist_fips (e.g. 'FIPS 203')"
            )

    # SC7 — failure_modes
    fm = cert.get("failure_modes", [])
    if not isinstance(fm, list) or len(fm) == 0:
        errors.append("EMPTY_FAILURE_MODES: failure_modes must be a non-empty list")

    # SC8 — composition_rules
    cr = cert.get("composition_rules", [])
    if not isinstance(cr, list) or len(cr) == 0:
        errors.append("EMPTY_COMPOSITION_RULES: composition_rules must be a non-empty list")

    # SC9 — cell/orbit consistency
    orbit = cert.get("orbit_signature", "")
    cell  = cert.get("levin_cell_type", "")
    if orbit not in KNOWN_ORBIT_SIGNATURES:
        errors.append(f"UNKNOWN_ORBIT_SIGNATURE: '{orbit}'")
    if cell not in KNOWN_LEVIN_CELL_TYPES:
        errors.append(f"UNKNOWN_LEVIN_CELL_TYPE: '{cell}'")
    if orbit in KNOWN_ORBIT_SIGNATURES and cell in KNOWN_LEVIN_CELL_TYPES:
        allowed = CELL_ORBIT_RULES.get(cell, frozenset())
        if orbit not in allowed:
            errors.append(
                f"CELL_ORBIT_MISMATCH: levin_cell_type='{cell}' requires "
                f"orbit_signature ∈ {sorted(allowed)}, got '{orbit}'"
            )

    # SC10 — goal length
    goal = cert.get("goal", "")
    if len(goal) < 10:
        errors.append(f"GOAL_TOO_SHORT: goal must be ≥10 chars, got {len(goal)}")

    return errors


def check_result_field(cert: dict, errors: list[str]) -> list[str]:
    """SC11 — result field must match actual validation outcome."""
    expected = "PASS" if not errors else "FAIL"
    declared = cert.get("result", "")
    if declared != expected:
        # Only report if no other errors (avoid double-reporting FAIL fixtures)
        if declared == "PASS" and errors:
            return errors + [f"RESULT_MISMATCH: declared PASS but {len(errors)} error(s) found"]
        if declared == "FAIL" and not errors:
            return [f"RESULT_MISMATCH: declared FAIL but cert validates cleanly"]
    return errors


def validate_file(path: Path) -> dict:
    cert   = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_cert(cert)
    errors = check_result_field(cert, errors)
    ok     = len(errors) == 0
    return {"ok": ok, "errors": errors, "file": str(path), "algorithm": cert.get("algorithm_name", "?")}


def self_test() -> dict:
    """Run all fixtures; return JSON with ok=true/false."""
    fixture_dir = Path(__file__).parent / "fixtures"
    fixtures    = sorted(fixture_dir.glob("*.json"))

    if not fixtures:
        return {"ok": False, "error": "no fixtures found"}

    results = []
    all_ok  = True

    for f in fixtures:
        cert      = json.loads(f.read_text(encoding="utf-8"))
        errors    = validate_cert(cert)
        errors    = check_result_field(cert, errors)
        declared  = cert.get("result", "")
        actual    = "PASS" if not errors else "FAIL"
        fixture_ok = (declared == actual)

        results.append({
            "file":      f.name,
            "algorithm": cert.get("algorithm_name", "?"),
            "declared":  declared,
            "actual":    actual,
            "ok":        fixture_ok,
            "errors":    errors,
        })

        if not fixture_ok:
            all_ok = False

    return {"ok": all_ok, "fixtures": results, "total": len(fixtures)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="QA Security Competency Cert [124] validator")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true", help="Run all fixtures")
    group.add_argument("--file",      type=Path,           help="Validate a single JSON file")
    args = parser.parse_args(argv)

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)
    else:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
