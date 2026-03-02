from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

from qa_alphageometry_ptolemy import qa_verify as spine_verify
from qa_mapping_protocol import validator as mapping_protocol_v1
from qa_fairness_demographic_parity_cert_v1 import validator as dp_v1


ValidatorId = Literal["decision_spine", "mapping_protocol_v1", "fairness_demographic_parity_v1"]


def _result_dict(status: str, check_name: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "check_name": check_name,
        "message": message,
        "details": details,
    }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, int]:
    passed = sum(1 for r in results if r.get("status") in ("PASSED", "PASS"))
    failed = sum(1 for r in results if r.get("status") in ("FAILED", "FAIL"))
    warnings = sum(1 for r in results if r.get("status") in ("WARNING", "WARN", "SKIP"))
    return {"passed": passed, "failed": failed, "warnings": warnings}


def _spine_result_to_dict(r: spine_verify.VerifyResult) -> Dict[str, Any]:
    return {
        "status": r.status.name,
        "check_name": r.check_name,
        "message": r.message,
        "details": r.details if r.details else None,
    }


def _run_decision_spine(obj: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    if "manifest" in obj:
        manifest = obj.get("manifest", {}) if isinstance(obj.get("manifest"), dict) else {}
        coherence = obj.get("coherence", {}) if isinstance(obj.get("coherence"), dict) else {}
        certificates = obj.get("certificates", {}) if isinstance(obj.get("certificates"), dict) else {}

        if manifest.get("bundle_hash"):
            results.append(_result_dict("PASSED", "bundle.has_hash", f"Bundle hash: {manifest['bundle_hash']}"))
        else:
            results.append(_result_dict("WARNING", "bundle.no_hash", "Bundle missing tamper-evident hash"))

        if coherence.get("valid") is True:
            results.append(_result_dict(
                "PASSED",
                "bundle.coherence",
                f"Bundle is coherent ({coherence.get('checks', 0)} checks)",
            ))
        else:
            results.append(_result_dict(
                "FAILED",
                "bundle.coherence",
                f"Bundle coherence failed: {coherence.get('violations', [])}",
            ))

        for cert_type, cert_list in certificates.items():
            if not isinstance(cert_list, list):
                continue
            for cert in cert_list:
                if not isinstance(cert, dict):
                    continue
                _, res = spine_verify.verify_certificate(cert)
                results.extend([_spine_result_to_dict(x) for x in res])

    elif "bundle_id" in obj:
        res = spine_verify.verify_bundle_coherence(obj)
        results.extend([_spine_result_to_dict(x) for x in res])

        for cert_type in ["policy", "mcts", "exploration", "inference", "filter", "rl", "imitation"]:
            key = f"{cert_type}_certificates"
            if key not in obj:
                continue
            cert_list = obj.get(key)
            if not isinstance(cert_list, list):
                continue
            for cert in cert_list:
                if not isinstance(cert, dict):
                    continue
                _, res2 = spine_verify.verify_certificate(cert)
                results.extend([_spine_result_to_dict(x) for x in res2])

    else:
        if not isinstance(obj, dict):
            return {
                "ok": False,
                "validator_id": "decision_spine",
                "results": [_result_dict("FAILED", "input.type", "Expected a JSON object at top-level")],
                "passed": 0,
                "failed": 1,
                "warnings": 0,
            }
        _, res = spine_verify.verify_certificate(obj)
        results.extend([_spine_result_to_dict(x) for x in res])

    counts = _summarize(results)
    return {
        "ok": counts["failed"] == 0,
        "validator_id": "decision_spine",
        "results": results,
        **counts,
    }


def _run_mapping_protocol_v1(obj: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    try:
        gate_results = mapping_protocol_v1.validate_mapping(obj)
    except Exception as e:
        results.append(_result_dict("FAIL", "gate_1_schema_validity", f"Validation crashed: {e}"))
        counts = _summarize(results)
        return {
            "ok": False,
            "validator_id": "mapping_protocol_v1",
            "results": results,
            **counts,
        }

    for gr in gate_results:
        results.append({
            "status": gr.status.value,
            "check_name": gr.gate,
            "message": gr.message,
            "details": gr.details,
        })

    counts = _summarize(results)
    return {
        "ok": counts["failed"] == 0,
        "validator_id": "mapping_protocol_v1",
        "results": results,
        **counts,
    }

def _run_fairness_demographic_parity_v1(obj: Dict[str, Any]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    try:
        gate_results = dp_v1.validate(obj)
    except Exception as e:
        results.append(_result_dict("FAIL", "gate_0_runtime", f"Validation crashed: {e}"))
        counts = _summarize(results)
        return {
            "ok": False,
            "validator_id": "fairness_demographic_parity_v1",
            "results": results,
            **counts,
        }

    for gr in gate_results:
        results.append({
            "status": gr.status.value,
            "check_name": gr.gate,
            "message": gr.message,
            "details": gr.details,
        })

    counts = _summarize(results)
    return {
        "ok": counts["failed"] == 0,
        "validator_id": "fairness_demographic_parity_v1",
        "results": results,
        **counts,
    }


def run_validator(*, validator_id: ValidatorId, obj: Dict[str, Any]) -> Dict[str, Any]:
    if validator_id == "decision_spine":
        return _run_decision_spine(obj)
    if validator_id == "mapping_protocol_v1":
        return _run_mapping_protocol_v1(obj)
    if validator_id == "fairness_demographic_parity_v1":
        return _run_fairness_demographic_parity_v1(obj)
    return {
        "ok": False,
        "validator_id": validator_id,
        "results": [_result_dict("FAILED", "validator.unknown", f"Unknown validator_id: {validator_id}")],
        "passed": 0,
        "failed": 1,
        "warnings": 0,
    }
