#!/usr/bin/env python3
"""QA Empirical Observation Cert family [122] validator — QA_EMPIRICAL_OBSERVATION_CERT.v1

Bridges captured experimental observations (Open Brain, experiment scripts, paper results)
to the cert ecosystem. Certifies the verdict of an empirical finding against a named
parent cert claim.

Validator checks:
  V1  observation.source is a known source type        → UNKNOWN_OBSERVATION_SOURCE
  V2  parent_cert.schema_version is a non-empty string → INVALID_PARENT_CERT_REF
  V3  verdict is one of the four valid verdicts         → INVALID_VERDICT
  V4  verdict==CONTRADICTS implies nonempty fail_ledger → CONTRADICTS_WITHOUT_FAIL_LEDGER
  V5  evidence list is nonempty                         → EMPTY_EVIDENCE

Usage:
  python qa_empirical_observation_cert_validate.py --self-test
  python qa_empirical_observation_cert_validate.py --file fixtures/eoc_pass_audio_orbit_consistent.json
"""

import json
import sys
import argparse
from pathlib import Path


KNOWN_SOURCES = frozenset(["open_brain", "experiment_script", "paper_result", "external_dataset"])
VALID_VERDICTS = frozenset(["CONSISTENT", "CONTRADICTS", "PARTIAL", "INCONCLUSIVE"])
KNOWN_FAIL_TYPES = frozenset([
    "UNKNOWN_OBSERVATION_SOURCE",
    "INVALID_PARENT_CERT_REF",
    "INVALID_VERDICT",
    "CONTRADICTS_WITHOUT_FAIL_LEDGER",
    "EMPTY_EVIDENCE",
])


class _Out:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def fail(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def validate_empirical_observation_cert(cert: dict) -> dict:
    out = _Out()
    detected_fails: set = set()

    # ── schema_version / cert_type ──────────────────────────────────────────
    if cert.get("schema_version") != "QA_EMPIRICAL_OBSERVATION_CERT.v1":
        out.fail(f"schema_version must be 'QA_EMPIRICAL_OBSERVATION_CERT.v1', got {cert.get('schema_version')!r}")
    if cert.get("cert_type") != "qa_empirical_observation_cert":
        out.fail(f"cert_type must be 'qa_empirical_observation_cert', got {cert.get('cert_type')!r}")

    # ── required top-level fields ───────────────────────────────────────────
    for field in ["certificate_id", "title", "created_utc",
                  "observation", "parent_cert", "verdict",
                  "evidence", "validation_checks", "fail_ledger", "result"]:
        if field not in cert:
            out.fail(f"missing required field: {field!r}")

    if out.errors:
        return _reconcile(cert, out, detected_fails)

    # ── V1: observation.source is known ─────────────────────────────────────
    obs = cert.get("observation", {})
    source = obs.get("source", "")
    if source not in KNOWN_SOURCES:
        detected_fails.add("UNKNOWN_OBSERVATION_SOURCE")

    # ── V2: parent_cert.schema_version is a non-empty string ────────────────
    pc = cert.get("parent_cert", {})
    pc_sv = pc.get("schema_version", "")
    if not isinstance(pc_sv, str) or not pc_sv.strip():
        detected_fails.add("INVALID_PARENT_CERT_REF")

    # ── V3: verdict is one of the four valid values ─────────────────────────
    verdict = cert.get("verdict", "")
    if verdict not in VALID_VERDICTS:
        detected_fails.add("INVALID_VERDICT")

    # ── V4: CONTRADICTS requires nonempty fail_ledger ───────────────────────
    fail_ledger = cert.get("fail_ledger", [])
    if verdict == "CONTRADICTS" and not fail_ledger:
        detected_fails.add("CONTRADICTS_WITHOUT_FAIL_LEDGER")

    # ── V5: evidence is nonempty ─────────────────────────────────────────────
    evidence = cert.get("evidence", [])
    if not evidence:
        detected_fails.add("EMPTY_EVIDENCE")

    return _reconcile(cert, out, detected_fails)


def _reconcile(cert: dict, out: _Out, detected_fails: set) -> dict:
    declared_result = cert.get("result", "")
    declared_ledger = cert.get("fail_ledger", [])
    declared_fail_types = {e.get("fail_type") for e in declared_ledger if isinstance(e, dict)}

    # Only enforce KNOWN_FAIL_TYPES when result=FAIL (validator-generated failures).
    # When result=PASS and verdict=CONTRADICTS, fail_ledger documents domain-specific
    # contradictions which may have any fail_type — no warning needed.
    if declared_result == "FAIL":
        for ft in declared_fail_types:
            if ft not in KNOWN_FAIL_TYPES:
                out.warn(f"unrecognised fail_type in fail_ledger: {ft!r}")

    if declared_result == "PASS":
        if detected_fails:
            for ft in sorted(detected_fails):
                out.fail(f"cert declares PASS but detected: {ft}")
    elif declared_result == "FAIL":
        missing_from_ledger = detected_fails - declared_fail_types
        for ft in sorted(missing_from_ledger):
            out.warn(f"detected {ft} but not declared in fail_ledger")
        phantom = declared_fail_types - detected_fails
        for ft in sorted(phantom):
            out.warn(f"fail_ledger declares {ft} but validator did not detect it")
    else:
        out.fail(f"result must be 'PASS' or 'FAIL', got {declared_result!r}")

    ok = len(out.errors) == 0
    label = "PASS" if ok else "FAIL"
    if ok and out.warnings:
        label = "PASS_WITH_WARNINGS"

    return {
        "ok": ok,
        "label": label,
        "certificate_id": cert.get("certificate_id", "(unknown)"),
        "errors": out.errors,
        "warnings": out.warnings,
        "detected_fails": sorted(detected_fails),
    }


def validate_file(path: Path) -> dict:
    with open(path) as f:
        cert = json.load(f)
    return validate_empirical_observation_cert(cert)


def self_test() -> dict:
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = {
        "eoc_pass_audio_orbit_consistent.json":  True,
        "eoc_pass_finance_contradicts.json":      True,
        "eoc_pass_qa_matched_generator_compression_discrete_consistent.json": True,
        "eoc_pass_qa_segmented_compression_consistent.json": True,
        "eoc_pass_prime_bounded_scaling_consistent.json": True,
        "eoc_fail_empty_evidence.json":           True,
    }

    results = []
    all_ok = True

    for fname, expect_ok in expected.items():
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        r = validate_file(fpath)
        passed = r["ok"] == expect_ok
        if not passed:
            all_ok = False
        results.append({
            "fixture": fname,
            "ok": passed,
            "label": r["label"],
            "errors": r["errors"],
            "warnings": r["warnings"],
        })

    return {"ok": all_ok, "results": results}


def main():
    parser = argparse.ArgumentParser(description="QA Empirical Observation Cert [122] validator")
    parser.add_argument("--self-test", action="store_true",
                        help="Run self-test against fixture suite")
    parser.add_argument("--file", type=Path,
                        help="Validate a single cert file")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    if args.file:
        result = validate_file(args.file)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
