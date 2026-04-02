"""
QA PHILOMATH Bridge Validator
=============================

Checks the narrow bridge from PHILOMATH top-5 targets into QA-native reduction rules.

Validator contract:
- --self-test exits 0 and prints {"ok": true, ...} on success
- accepts a single cert JSON path in normal mode
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "QA_PHILOMATH_BRIDGE.v1"
CERT_TYPE = "qa_philomath_bridge"

EXPECTED_SOURCE_MANIFEST = "qa_ingestion_sources/qa_philomath_corpus_manifest.json"
EXPECTED_SOURCE_QUEUE = "qa_ingestion_sources/qa_philomath_ingestion_queue.json"
EXPECTED_SOURCE_LEDGER = "qa_ingestion_sources/qa_philomath_claim_ledger.json"
EXPECTED_SOURCE_CROSSWALK = "qa_ingestion_sources/qa_philomath_top5_crosswalk.json"

EXPECTED_ORDER = [
    "The Digital Root",
    "Prime Numbers",
    "Reciprocity of Numbers and Prime Factorization",
    "Number Classification",
    "Semiprime Factorization - The Geometric Solutions",
]

EXPECTED_ENTRY_RULES = {
    "The Digital Root": {
        "claim_class": "modular",
        "promotion_test_type": "exact_reduction",
    },
    "Prime Numbers": {
        "claim_class": "modular",
        "promotion_test_type": "residue_filter_verification",
    },
    "Reciprocity of Numbers and Prime Factorization": {
        "claim_class": "structural_analogy",
        "promotion_test_type": "factor_recovery",
    },
    "Number Classification": {
        "claim_class": "modular",
        "promotion_test_type": "membership_testability",
    },
    "Semiprime Factorization - The Geometric Solutions": {
        "claim_class": "structural_analogy",
        "promotion_test_type": "invertible_geometry",
    },
}


def validate(cert: Dict[str, Any]) -> Tuple[bool, List[str]]:
    detected: List[str] = []

    if cert.get("schema_version") != SCHEMA_VERSION:
        detected.append("INVALID_KERNEL_REFERENCE")
    if cert.get("cert_type") != CERT_TYPE:
        detected.append("INVALID_KERNEL_REFERENCE")

    if cert.get("source_manifest_ref") != EXPECTED_SOURCE_MANIFEST:
        detected.append("SOURCE_REF_MISMATCH")
    if cert.get("source_queue_ref") != EXPECTED_SOURCE_QUEUE:
        detected.append("SOURCE_REF_MISMATCH")
    if cert.get("source_claim_ledger_ref") != EXPECTED_SOURCE_LEDGER:
        detected.append("SOURCE_REF_MISMATCH")
    if cert.get("source_crosswalk_ref") != EXPECTED_SOURCE_CROSSWALK:
        detected.append("SOURCE_REF_MISMATCH")

    top5_order = cert.get("top5_order")
    if top5_order != EXPECTED_ORDER:
        detected.append("TOP5_ORDER_MISMATCH")

    chapter_bridges = cert.get("chapter_bridges")
    if not isinstance(chapter_bridges, list) or len(chapter_bridges) != 5:
        detected.append("BRIDGE_ENTRY_MISSING")
        chapter_bridges = []

    seen_chapters = set()
    for entry in chapter_bridges:
        if not isinstance(entry, dict):
            detected.append("BRIDGE_ENTRY_INVALID")
            continue
        chapter = entry.get("chapter")
        if chapter not in EXPECTED_ENTRY_RULES:
            detected.append("BRIDGE_ENTRY_INVALID")
            continue
        seen_chapters.add(chapter)
        expected = EXPECTED_ENTRY_RULES[chapter]
        if entry.get("claim_class") != expected["claim_class"]:
            detected.append("PROMOTION_RULE_MISMATCH")
        if entry.get("promotion_test_type") != expected["promotion_test_type"]:
            detected.append("PROMOTION_RULE_MISMATCH")
        if not isinstance(entry.get("priority"), int):
            detected.append("BRIDGE_ENTRY_INVALID")
        if len(str(entry.get("qa_native_rewrite", "")).strip()) < 20:
            detected.append("BRIDGE_ENTRY_INVALID")
        if len(str(entry.get("promotion_condition", "")).strip()) < 20:
            detected.append("BRIDGE_ENTRY_INVALID")
        outputs = entry.get("candidate_outputs")
        if not isinstance(outputs, list) or len(outputs) < 1:
            detected.append("BRIDGE_ENTRY_INVALID")

    if seen_chapters != set(EXPECTED_ORDER):
        detected.append("BRIDGE_ENTRY_MISSING")

    pass_witness = cert.get("canonical_pass_witness", {})
    if pass_witness.get("bridge_result") != "top5_bridge_verified":
        detected.append("WITNESS_MISMATCH")

    fail_witness = cert.get("canonical_fail_witness", {})
    fail_types = fail_witness.get("fail_types")
    if not isinstance(fail_types, list) or len(fail_types) == 0:
        detected.append("WITNESS_MISMATCH")

    seen = set()
    fail_list: List[str] = []
    for item in detected:
        if item not in seen:
            seen.add(item)
            fail_list.append(item)
    return (len(fail_list) == 0, fail_list)


def _run_self_test() -> Dict[str, Any]:
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    results = []
    all_ok = True

    pass_path = os.path.join(fixtures_dir, "philomath_bridge_pass_canonical.json")
    try:
        with open(pass_path, "r", encoding="utf-8") as handle:
            cert = json.load(handle)
        passed, fail_types = validate(cert)
        ok = passed and len(fail_types) == 0
        results.append(
            {
                "fixture": "philomath_bridge_pass_canonical.json",
                "expected": "PASS",
                "got": "PASS" if ok else f"FAIL({fail_types})",
                "ok": ok,
            }
        )
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({"fixture": "philomath_bridge_pass_canonical.json", "error": str(exc), "ok": False})
        all_ok = False

    fail_path = os.path.join(fixtures_dir, "philomath_bridge_fail_wrong_order.json")
    try:
        with open(fail_path, "r", encoding="utf-8") as handle:
            cert = json.load(handle)
        passed, fail_types = validate(cert)
        expected_fail = {"TOP5_ORDER_MISMATCH"}
        ok = (not passed) and expected_fail.issubset(set(fail_types))
        results.append(
            {
                "fixture": "philomath_bridge_fail_wrong_order.json",
                "expected": "FAIL(TOP5_ORDER_MISMATCH)",
                "got": f"FAIL({fail_types})" if not passed else "PASS",
                "ok": ok,
            }
        )
        if not ok:
            all_ok = False
    except Exception as exc:
        results.append({"fixture": "philomath_bridge_fail_wrong_order.json", "error": str(exc), "ok": False})
        all_ok = False

    return {"ok": all_ok, "results": results}


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = _run_self_test()
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        sys.exit(0 if result["ok"] else 1)

    if len(sys.argv) != 2:
        print("Usage: python qa_philomath_bridge_validate.py <cert.json>")
        print("       python qa_philomath_bridge_validate.py --self-test")
        sys.exit(1)

    cert_path = sys.argv[1]
    with open(cert_path, "r", encoding="utf-8") as handle:
        cert = json.load(handle)
    passed, fail_types = validate(cert)
    if passed:
        print(f"PASS: {cert_path}")
        sys.exit(0)
    print(f"FAIL: {cert_path}")
    for fail_type in fail_types:
        print(f"- {fail_type}")
    sys.exit(1)
