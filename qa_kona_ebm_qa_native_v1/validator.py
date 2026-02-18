#!/usr/bin/env python3
"""
validator.py

QA_KONA_EBM_QA_NATIVE_CERT.v1 validator.

Five gates:
  Gate 1 -- Schema: required fields, types, algorithm constant
  Gate 2 -- Config sanity: n_visible==784, range checks, lr > 0
  Gate 3 -- Orbit map integrity: recompute orbit_map_hash, verify match
  Gate 4 -- Deterministic replay: re-run train_qa_rbm, verify trace_hash
  Gate 5 -- Invariant_diff contract: GRADIENT_EXPLOSION -> non-null; CONVERGED/STALLED -> null

CLI:
    python validator.py <fixture.json>
    python validator.py <fixture.json> --json
    python validator.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from qa_orbit_map import orbit_map_hash as compute_orbit_map_hash
from rbm_qa_native_train import train_qa_rbm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(gate: str, fail_type: str, target_path: str, reason: str) -> dict:
    return {
        "gate": gate,
        "status": "FAIL",
        "details": {
            "invariant_diff": {
                "fail_type": fail_type,
                "target_path": target_path,
                "reason": reason,
            }
        },
    }


def _pass(gate: str, details: Optional[dict] = None) -> dict:
    return {"gate": gate, "status": "PASS", "details": details or {}}


# ---------------------------------------------------------------------------
# Gate 1 -- Schema
# ---------------------------------------------------------------------------

_REQUIRED_TOP = ["cert_type", "schema_version", "cert_id", "issued_utc",
                 "model_config", "result", "trace"]
_REQUIRED_MODEL_CONFIG = ["n_visible", "n_samples", "n_epochs", "lr", "seed", "algorithm"]
_REQUIRED_RESULT = ["status", "energy_per_epoch", "reconstruction_error_per_epoch",
                    "grad_norm_per_epoch", "final_weights_norm", "invariant_diff", "orbit_analysis"]
_REQUIRED_ORBIT_ANALYSIS = ["orbit_class_alignment", "orbit_coherence_score",
                             "orbit_dominant_class", "orbit_map_hash"]
_ORBIT_TYPES = ["COSMOS", "SATELLITE", "SINGULARITY"]
_REQUIRED_TRACE = ["trace_hash"]
_VALID_STATUSES = {"CONVERGED", "STALLED", "GRADIENT_EXPLOSION"}


def gate1_schema(cert: Any) -> Tuple[dict, bool]:
    gate = "Gate1_schema"

    if not isinstance(cert, dict):
        return _fail(gate, "SCHEMA_ERROR", ".", "cert must be a JSON object"), False

    for field in _REQUIRED_TOP:
        if field not in cert:
            return _fail(gate, "SCHEMA_ERROR", field,
                         f"required top-level field '{field}' is missing"), False

    if cert.get("cert_type") != "QA_KONA_EBM_QA_NATIVE_CERT.v1":
        return _fail(gate, "SCHEMA_ERROR", "cert_type",
                     f"cert_type must be 'QA_KONA_EBM_QA_NATIVE_CERT.v1', got {cert.get('cert_type')!r}"), False

    if cert.get("schema_version") != 1:
        return _fail(gate, "SCHEMA_ERROR", "schema_version",
                     f"schema_version must be 1, got {cert.get('schema_version')!r}"), False

    if not isinstance(cert.get("cert_id"), str) or not cert["cert_id"]:
        return _fail(gate, "SCHEMA_ERROR", "cert_id", "cert_id must be a non-empty string"), False

    if not isinstance(cert.get("issued_utc"), str) or not cert["issued_utc"]:
        return _fail(gate, "SCHEMA_ERROR", "issued_utc", "issued_utc must be a non-empty string"), False

    # model_config
    mc = cert.get("model_config")
    if not isinstance(mc, dict):
        return _fail(gate, "SCHEMA_ERROR", "model_config", "model_config must be an object"), False
    for field in _REQUIRED_MODEL_CONFIG:
        if field not in mc:
            return _fail(gate, "SCHEMA_ERROR", f"model_config.{field}",
                         f"required field '{field}' missing from model_config"), False
    if mc.get("algorithm") != "rbm_qa_native_cd1_numpy":
        return _fail(gate, "SCHEMA_ERROR", "model_config.algorithm",
                     f"algorithm must be 'rbm_qa_native_cd1_numpy', got {mc.get('algorithm')!r}"), False
    for int_field in ["n_visible", "n_samples", "n_epochs", "seed"]:
        if not isinstance(mc.get(int_field), int):
            return _fail(gate, "SCHEMA_ERROR", f"model_config.{int_field}",
                         f"{int_field} must be an integer"), False
    if not isinstance(mc.get("lr"), (int, float)):
        return _fail(gate, "SCHEMA_ERROR", "model_config.lr", "lr must be a number"), False

    # result
    res = cert.get("result")
    if not isinstance(res, dict):
        return _fail(gate, "SCHEMA_ERROR", "result", "result must be an object"), False
    for field in _REQUIRED_RESULT:
        if field not in res:
            return _fail(gate, "SCHEMA_ERROR", f"result.{field}",
                         f"required field '{field}' missing from result"), False
    if res.get("status") not in _VALID_STATUSES:
        return _fail(gate, "SCHEMA_ERROR", "result.status",
                     f"status must be one of {sorted(_VALID_STATUSES)}, got {res.get('status')!r}"), False
    for arr_field in ["energy_per_epoch", "reconstruction_error_per_epoch", "grad_norm_per_epoch"]:
        if not isinstance(res.get(arr_field), list):
            return _fail(gate, "SCHEMA_ERROR", f"result.{arr_field}",
                         f"{arr_field} must be an array"), False
    if not isinstance(res.get("final_weights_norm"), (int, float)):
        return _fail(gate, "SCHEMA_ERROR", "result.final_weights_norm",
                     "final_weights_norm must be a number"), False

    # invariant_diff
    idiff = res.get("invariant_diff")
    if idiff is not None:
        if not isinstance(idiff, dict):
            return _fail(gate, "SCHEMA_ERROR", "result.invariant_diff",
                         "invariant_diff must be null or an object"), False
        for k in ["fail_type", "target_path", "reason"]:
            if k not in idiff:
                return _fail(gate, "SCHEMA_ERROR", f"result.invariant_diff.{k}",
                             f"invariant_diff missing required key '{k}'"), False

    # orbit_analysis
    oa = res.get("orbit_analysis")
    if not isinstance(oa, dict):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis",
                     "orbit_analysis must be an object"), False
    for field in _REQUIRED_ORBIT_ANALYSIS:
        if field not in oa:
            return _fail(gate, "SCHEMA_ERROR", f"result.orbit_analysis.{field}",
                         f"required field '{field}' missing from orbit_analysis"), False

    # orbit_class_alignment: each orbit type must have 10-element array
    oca = oa.get("orbit_class_alignment")
    if not isinstance(oca, dict):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_class_alignment",
                     "orbit_class_alignment must be an object"), False
    for otype in _ORBIT_TYPES:
        if otype not in oca:
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_class_alignment.{otype}",
                         f"orbit_class_alignment missing key '{otype}'"), False
        if not isinstance(oca[otype], list) or len(oca[otype]) != 10:
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_class_alignment.{otype}",
                         f"{otype} must be a 10-element array"), False

    # orbit_coherence_score
    ocs = oa.get("orbit_coherence_score")
    if not isinstance(ocs, dict):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_coherence_score",
                     "orbit_coherence_score must be an object"), False
    for otype in _ORBIT_TYPES:
        if otype not in ocs:
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_coherence_score.{otype}",
                         f"orbit_coherence_score missing key '{otype}'"), False
        if not isinstance(ocs[otype], (int, float)):
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_coherence_score.{otype}",
                         f"{otype} must be a number"), False

    # orbit_dominant_class
    odc = oa.get("orbit_dominant_class")
    if not isinstance(odc, dict):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_dominant_class",
                     "orbit_dominant_class must be an object"), False
    for otype in _ORBIT_TYPES:
        if otype not in odc:
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_dominant_class.{otype}",
                         f"orbit_dominant_class missing key '{otype}'"), False
        v = odc[otype]
        if not isinstance(v, int) or v < 0 or v > 9:
            return _fail(gate, "SCHEMA_ERROR",
                         f"result.orbit_analysis.orbit_dominant_class.{otype}",
                         f"{otype} must be an integer in [0,9]"), False

    # orbit_map_hash
    omh = oa.get("orbit_map_hash")
    if not isinstance(omh, str) or len(omh) != 64:
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_map_hash",
                     "orbit_map_hash must be a 64-character hex string"), False
    if not all(c in "0123456789abcdef" for c in omh):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_map_hash",
                     "orbit_map_hash must contain only lowercase hex characters"), False

    # trace
    tr = cert.get("trace")
    if not isinstance(tr, dict):
        return _fail(gate, "SCHEMA_ERROR", "trace", "trace must be an object"), False
    for field in _REQUIRED_TRACE:
        if field not in tr:
            return _fail(gate, "SCHEMA_ERROR", f"trace.{field}",
                         f"required field '{field}' missing from trace"), False
    th = tr.get("trace_hash")
    if not isinstance(th, str) or len(th) != 64:
        return _fail(gate, "SCHEMA_ERROR", "trace.trace_hash",
                     "trace_hash must be a 64-character hex string"), False
    if not all(c in "0123456789abcdef" for c in th):
        return _fail(gate, "SCHEMA_ERROR", "trace.trace_hash",
                     "trace_hash must contain only lowercase hex characters"), False

    return _pass(gate, {"cert_type": cert["cert_type"], "cert_id": cert["cert_id"],
                        "status": res["status"]}), True


# ---------------------------------------------------------------------------
# Gate 2 -- Config sanity
# ---------------------------------------------------------------------------

def gate2_config(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate2_config"
    mc = cert["model_config"]

    if mc["n_visible"] != 784:
        return _fail(gate, "CONFIG_INVALID", "model_config.n_visible",
                     f"n_visible must be 784 (MNIST), got {mc['n_visible']}"), False

    if not (1 <= mc["n_samples"] <= 60000):
        return _fail(gate, "CONFIG_INVALID", "model_config.n_samples",
                     f"n_samples must be in [1, 60000], got {mc['n_samples']}"), False

    if not (1 <= mc["n_epochs"] <= 100):
        return _fail(gate, "CONFIG_INVALID", "model_config.n_epochs",
                     f"n_epochs must be in [1, 100], got {mc['n_epochs']}"), False

    if mc["lr"] <= 0:
        return _fail(gate, "CONFIG_INVALID", "model_config.lr",
                     f"lr must be > 0, got {mc['lr']}"), False

    return _pass(gate, {
        "n_visible": mc["n_visible"],
        "n_hidden": 81,
        "n_samples": mc["n_samples"],
        "n_epochs": mc["n_epochs"],
        "lr": mc["lr"],
        "seed": mc["seed"],
    }), True


# ---------------------------------------------------------------------------
# Gate 3 -- Orbit map integrity
# ---------------------------------------------------------------------------

def gate3_orbit_map(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate3_orbit_map_integrity"

    expected_hash = cert["result"]["orbit_analysis"]["orbit_map_hash"]
    actual_hash = compute_orbit_map_hash()

    if actual_hash != expected_hash:
        return _fail(
            gate,
            "ORBIT_MAP_VIOLATION",
            "result.orbit_analysis.orbit_map_hash",
            "orbit assignment does not match canonical QA state enumeration",
        ), False

    return _pass(gate, {"orbit_map_hash": actual_hash}), True


# ---------------------------------------------------------------------------
# Gate 4 -- Deterministic replay
# ---------------------------------------------------------------------------

def gate4_replay(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate4_deterministic_replay"
    mc = cert["model_config"]

    replayed = train_qa_rbm(
        n_samples=mc["n_samples"],
        n_epochs=mc["n_epochs"],
        lr=mc["lr"],
        seed=mc["seed"],
    )

    expected_hash = cert["trace"]["trace_hash"]
    actual_hash = replayed["trace_hash"]

    if actual_hash != expected_hash:
        return _fail(gate, "TRACE_HASH_MISMATCH", "trace.trace_hash",
                     "deterministic replay produced different trace hash"), False

    expected_energy = cert["result"]["energy_per_epoch"]
    actual_energy = replayed["energy_per_epoch"]
    if [round(e, 6) for e in expected_energy] != [round(e, 6) for e in actual_energy]:
        return _fail(gate, "TRACE_HASH_MISMATCH", "result.energy_per_epoch",
                     "deterministic replay produced different energy_per_epoch"), False

    if cert["result"]["status"] != replayed["status"]:
        return _fail(gate, "TRACE_HASH_MISMATCH", "result.status",
                     f"cert status {cert['result']['status']!r} != replay status {replayed['status']!r}"), False

    return _pass(gate, {"trace_hash": actual_hash, "replay_status": replayed["status"]}), True


# ---------------------------------------------------------------------------
# Gate 5 -- Invariant_diff contract
# ---------------------------------------------------------------------------

def gate5_invariant_diff(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate5_invariant_diff"
    status = cert["result"]["status"]
    idiff = cert["result"]["invariant_diff"]

    if status in ("CONVERGED", "STALLED"):
        if idiff is not None:
            return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                         "result.invariant_diff",
                         f"invariant_diff must be null when status is {status!r}"), False
        return _pass(gate, {"status": status, "invariant_diff": "null (correct)"}), True

    if status == "GRADIENT_EXPLOSION":
        if idiff is None:
            return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                         "result.invariant_diff",
                         "invariant_diff must be present when status is 'GRADIENT_EXPLOSION'"), False
        for k in ["fail_type", "target_path", "reason"]:
            if k not in idiff:
                return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                             f"result.invariant_diff.{k}",
                             f"invariant_diff missing required key '{k}'"), False
        return _fail(gate, "GRADIENT_EXPLOSION",
                     idiff.get("target_path", "result.grad_norm_per_epoch"),
                     idiff.get("reason", "training exploded")), False

    return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION", "result.status",
                 f"unknown status {status!r}"), False


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate(cert: Any) -> dict:
    results: List[dict] = []

    g1, ok = gate1_schema(cert)
    results.append(g1)
    if not ok:
        cert_id = cert.get("cert_id", "<unknown>") if isinstance(cert, dict) else "<unknown>"
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    cert_id = cert["cert_id"]

    g2, ok = gate2_config(cert)
    results.append(g2)
    if not ok:
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    g3, ok = gate3_orbit_map(cert)
    results.append(g3)
    if not ok:
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    g4, ok = gate4_replay(cert)
    results.append(g4)
    if not ok:
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    g5, ok = gate5_invariant_diff(cert)
    results.append(g5)
    if not ok:
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    return {"cert_id": cert_id, "status": "PASS", "results": results}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_FIXTURES_DIR = os.path.join(_DIR, "fixtures")

_SELF_TEST_CASES = [
    ("valid_qa_rbm_run.json",            "PASS"),
    ("invalid_orbit_map_violation.json", "FAIL"),
    ("invalid_trace_mismatch.json",      "FAIL"),
]


def self_test() -> bool:
    all_ok = True
    print("Running self-test...")
    for fname, expected_status in _SELF_TEST_CASES:
        fpath = os.path.join(_FIXTURES_DIR, fname)
        try:
            with open(fpath) as f:
                cert = json.load(f)
        except Exception as e:
            print(f"  FAIL  {fname}: could not load: {e}")
            all_ok = False
            continue

        result = validate(cert)
        actual = result["status"]
        ok = actual == expected_status
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark}  {fname}: expected={expected_status} got={actual}")
        if not ok:
            all_ok = False
            for gate_res in result["results"]:
                if gate_res["status"] == "FAIL":
                    print(f"        failing gate: {gate_res['gate']}")
                    print(f"        details: {json.dumps(gate_res['details'], indent=8)}")

    if all_ok:
        print("Self-test PASSED.")
    else:
        print("Self-test FAILED.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QA_KONA_EBM_QA_NATIVE_CERT.v1 validator")
    parser.add_argument("fixture", nargs="?", help="Path to cert JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--self-test", action="store_true", dest="self_test",
                        help="Run self-test against built-in fixtures")
    args = parser.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)

    if not args.fixture:
        parser.error("provide a fixture path or --self-test")

    with open(args.fixture) as f:
        cert = json.load(f)

    result = validate(cert)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"cert_id : {result['cert_id']}")
        print(f"status  : {result['status']}")
        for gate_res in result["results"]:
            print(f"  [{gate_res['status']}] {gate_res['gate']}")
            if gate_res["status"] == "FAIL":
                idiff = gate_res["details"].get("invariant_diff", {})
                print(f"         fail_type  : {idiff.get('fail_type')}")
                print(f"         target_path: {idiff.get('target_path')}")
                print(f"         reason     : {idiff.get('reason')}")

    sys.exit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
