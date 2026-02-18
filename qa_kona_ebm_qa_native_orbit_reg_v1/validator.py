#!/usr/bin/env python3
"""
validator.py

QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1 validator.

Six gates:
  Gate 1 -- Schema: required fields, types, algorithm constant
  Gate 2 -- Config sanity: n_visible==784, ranges, lr > 0, lambda_orbit >= 0
  Gate 3 -- Orbit map integrity: recompute orbit_map_hash, verify match.
             Also verifies coherence_gap_stats present.
             If result.generator_curvature is present: recomputes
             kappa_hat_per_epoch = [1-|1-lr_ep*lambda_orbit| for each epoch],
             checks kappa_hash, min_kappa_hat/epoch consistency, and fails
             NEGATIVE_GENERATOR_CURVATURE if any recomputed kappa < 0, or
             CURVATURE_RECOMPUTE_MISMATCH if values don't match cert.
  Gate 4 -- Deterministic replay: re-run train_qa_orbit_reg_rbm, verify
             trace_hash, reg_trace_hash, and status.
  Gate 5 -- Invariant_diff contract:
             REGULARIZER_NUMERIC_INSTABILITY or GRADIENT_EXPLOSION ->
               invariant_diff non-null with fail_type/target_path/reason
             CONVERGED/STALLED -> invariant_diff must be null
  Gate 6 -- LR schedule contract (conditional; only runs when
             result.lr_per_epoch is present):
             If model_config.lr_schedule is present and non-null:
               - Verify type == "step"
               - Verify steps[0].epoch == 1
               - Verify lr_per_epoch length == len(result.energy_per_epoch)
               - Verify each lr_per_epoch[i] matches the schedule for epoch i+1
             If lr_schedule is absent or null:
               - Verify all values in lr_per_epoch == model_config.lr
             Failure type: LR_SCHEDULE_INVALID

CLI:
    python validator.py <fixture.json>
    python validator.py <fixture.json> --json
    python validator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, List, Optional, Tuple

_DIR = os.path.dirname(os.path.abspath(__file__))
_FAMILY63 = os.path.join(os.path.dirname(_DIR), "qa_kona_ebm_qa_native_v1")
sys.path.insert(0, _FAMILY63)
sys.path.insert(0, _DIR)

from qa_orbit_map import orbit_map_hash as compute_orbit_map_hash
from rbm_qa_orbit_reg_train import train_qa_orbit_reg_rbm, _build_lr_lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(gate: str, fail_type: str, target_path: str, reason: str) -> dict:
    return {
        "gate": gate,
        "status": "FAIL",
        "details": {
            "invariant_diff": {
                "fail_type":   fail_type,
                "target_path": target_path,
                "reason":      reason,
            }
        },
    }


def _pass(gate: str, details: Optional[dict] = None) -> dict:
    return {"gate": gate, "status": "PASS", "details": details or {}}


# ---------------------------------------------------------------------------
# Gate 1 -- Schema
# ---------------------------------------------------------------------------

_REQUIRED_TOP = [
    "cert_type", "schema_version", "cert_id", "issued_utc",
    "model_config", "result", "trace"
]
_REQUIRED_MODEL_CONFIG = [
    "n_visible", "n_samples", "n_epochs", "lr", "lambda_orbit", "seed", "algorithm"
]
_REQUIRED_RESULT = [
    "status", "energy_per_epoch", "reconstruction_error_per_epoch",
    "grad_norm_per_epoch", "reg_norm_per_epoch", "final_weights_norm",
    "invariant_diff", "orbit_analysis", "reg_trace_hash"
]
_REQUIRED_ORBIT_ANALYSIS = [
    "orbit_class_alignment", "orbit_coherence_score",
    "orbit_dominant_class", "orbit_map_hash", "coherence_gap_stats"
]
_ORBIT_TYPES = ["COSMOS", "SATELLITE", "SINGULARITY"]
_REQUIRED_TRACE = ["trace_hash"]
_VALID_STATUSES = {
    "CONVERGED", "STALLED", "GRADIENT_EXPLOSION", "REGULARIZER_NUMERIC_INSTABILITY"
}
_CERT_TYPE = "QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1"
_ALGORITHM  = "rbm_qa_native_orbit_reg_cd1_numpy"


def gate1_schema(cert: Any) -> Tuple[dict, bool]:
    gate = "Gate1_schema"

    if not isinstance(cert, dict):
        return _fail(gate, "SCHEMA_ERROR", ".", "cert must be a JSON object"), False

    for field in _REQUIRED_TOP:
        if field not in cert:
            return _fail(gate, "SCHEMA_ERROR", field,
                         f"required top-level field '{field}' is missing"), False

    if cert.get("cert_type") != _CERT_TYPE:
        return _fail(gate, "SCHEMA_ERROR", "cert_type",
                     f"cert_type must be '{_CERT_TYPE}', got {cert.get('cert_type')!r}"), False

    if cert.get("schema_version") != 1:
        return _fail(gate, "SCHEMA_ERROR", "schema_version",
                     f"schema_version must be 1, got {cert.get('schema_version')!r}"), False

    if not isinstance(cert.get("cert_id"), str) or not cert["cert_id"]:
        return _fail(gate, "SCHEMA_ERROR", "cert_id", "cert_id must be a non-empty string"), False

    if not isinstance(cert.get("issued_utc"), str) or not cert["issued_utc"]:
        return _fail(gate, "SCHEMA_ERROR", "issued_utc", "issued_utc must be a non-empty string"), False

    mc = cert.get("model_config")
    if not isinstance(mc, dict):
        return _fail(gate, "SCHEMA_ERROR", "model_config", "model_config must be an object"), False
    for field in _REQUIRED_MODEL_CONFIG:
        if field not in mc:
            return _fail(gate, "SCHEMA_ERROR", f"model_config.{field}",
                         f"required field '{field}' missing from model_config"), False
    if mc.get("algorithm") != _ALGORITHM:
        return _fail(gate, "SCHEMA_ERROR", "model_config.algorithm",
                     f"algorithm must be '{_ALGORITHM}', got {mc.get('algorithm')!r}"), False
    for int_field in ["n_visible", "n_samples", "n_epochs", "seed"]:
        if not isinstance(mc.get(int_field), int):
            return _fail(gate, "SCHEMA_ERROR", f"model_config.{int_field}",
                         f"{int_field} must be an integer"), False
    for num_field in ["lr", "lambda_orbit"]:
        if not isinstance(mc.get(num_field), (int, float)):
            return _fail(gate, "SCHEMA_ERROR", f"model_config.{num_field}",
                         f"{num_field} must be a number"), False

    # Validate lr_schedule if present (optional field)
    if "lr_schedule" in mc:
        lrs = mc["lr_schedule"]
        if lrs is not None:
            if not isinstance(lrs, dict):
                return _fail(gate, "SCHEMA_ERROR", "model_config.lr_schedule",
                             "lr_schedule must be null or an object"), False
            if lrs.get("type") != "step":
                return _fail(gate, "SCHEMA_ERROR", "model_config.lr_schedule.type",
                             f"lr_schedule.type must be 'step', got {lrs.get('type')!r}"), False
            steps = lrs.get("steps")
            if not isinstance(steps, list) or len(steps) == 0:
                return _fail(gate, "SCHEMA_ERROR", "model_config.lr_schedule.steps",
                             "lr_schedule.steps must be a non-empty array"), False
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    return _fail(gate, "SCHEMA_ERROR",
                                 f"model_config.lr_schedule.steps[{i}]",
                                 f"steps[{i}] must be an object"), False
                for k in ["epoch", "lr"]:
                    if k not in step:
                        return _fail(gate, "SCHEMA_ERROR",
                                     f"model_config.lr_schedule.steps[{i}].{k}",
                                     f"steps[{i}] missing required key '{k}'"), False
                if not isinstance(step["epoch"], int) or step["epoch"] < 1:
                    return _fail(gate, "SCHEMA_ERROR",
                                 f"model_config.lr_schedule.steps[{i}].epoch",
                                 f"steps[{i}].epoch must be a positive integer"), False
                if not isinstance(step["lr"], (int, float)) or step["lr"] <= 0:
                    return _fail(gate, "SCHEMA_ERROR",
                                 f"model_config.lr_schedule.steps[{i}].lr",
                                 f"steps[{i}].lr must be a positive number"), False

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
    for arr_field in ["energy_per_epoch", "reconstruction_error_per_epoch",
                      "grad_norm_per_epoch", "reg_norm_per_epoch"]:
        if not isinstance(res.get(arr_field), list):
            return _fail(gate, "SCHEMA_ERROR", f"result.{arr_field}",
                         f"{arr_field} must be an array"), False
    # lr_per_epoch is optional; validate type if present
    if "lr_per_epoch" in res:
        if not isinstance(res["lr_per_epoch"], list):
            return _fail(gate, "SCHEMA_ERROR", "result.lr_per_epoch",
                         "lr_per_epoch must be an array"), False
        for i, v in enumerate(res["lr_per_epoch"]):
            if not isinstance(v, (int, float)):
                return _fail(gate, "SCHEMA_ERROR", f"result.lr_per_epoch[{i}]",
                             f"lr_per_epoch[{i}] must be a number"), False

    if not isinstance(res.get("final_weights_norm"), (int, float)):
        return _fail(gate, "SCHEMA_ERROR", "result.final_weights_norm",
                     "final_weights_norm must be a number"), False
    rth = res.get("reg_trace_hash")
    if not isinstance(rth, str) or len(rth) != 64:
        return _fail(gate, "SCHEMA_ERROR", "result.reg_trace_hash",
                     "reg_trace_hash must be a 64-character hex string"), False
    if not all(c in "0123456789abcdef" for c in rth):
        return _fail(gate, "SCHEMA_ERROR", "result.reg_trace_hash",
                     "reg_trace_hash must contain only lowercase hex characters"), False

    idiff = res.get("invariant_diff")
    if idiff is not None:
        if not isinstance(idiff, dict):
            return _fail(gate, "SCHEMA_ERROR", "result.invariant_diff",
                         "invariant_diff must be null or an object"), False
        for k in ["fail_type", "target_path", "reason"]:
            if k not in idiff:
                return _fail(gate, "SCHEMA_ERROR", f"result.invariant_diff.{k}",
                             f"invariant_diff missing required key '{k}'"), False

    oa = res.get("orbit_analysis")
    if not isinstance(oa, dict):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis",
                     "orbit_analysis must be an object"), False
    for field in _REQUIRED_ORBIT_ANALYSIS:
        if field not in oa:
            return _fail(gate, "SCHEMA_ERROR", f"result.orbit_analysis.{field}",
                         f"required field '{field}' missing from orbit_analysis"), False

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

    omh = oa.get("orbit_map_hash")
    if not isinstance(omh, str) or len(omh) != 64:
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_map_hash",
                     "orbit_map_hash must be a 64-character hex string"), False
    if not all(c in "0123456789abcdef" for c in omh):
        return _fail(gate, "SCHEMA_ERROR", "result.orbit_analysis.orbit_map_hash",
                     "orbit_map_hash must contain only lowercase hex characters"), False

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
    if mc["lambda_orbit"] < 0:
        return _fail(gate, "CONFIG_INVALID", "model_config.lambda_orbit",
                     f"lambda_orbit must be >= 0, got {mc['lambda_orbit']}"), False

    return _pass(gate, {
        "n_visible":    mc["n_visible"],
        "n_hidden":     81,
        "n_samples":    mc["n_samples"],
        "n_epochs":     mc["n_epochs"],
        "lr":           mc["lr"],
        "lambda_orbit": mc["lambda_orbit"],
        "seed":         mc["seed"],
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

    status = cert["result"]["status"]
    if status not in ("GRADIENT_EXPLOSION", "REGULARIZER_NUMERIC_INSTABILITY"):
        oa = cert["result"]["orbit_analysis"]
        cgs = oa.get("coherence_gap_stats")
        if not isinstance(cgs, dict):
            return _fail(
                gate,
                "MISSING_COHERENCE_GAP_STATS",
                "result.orbit_analysis.coherence_gap_stats",
                f"coherence_gap_stats must be present and an object when status is {status!r}",
            ), False
        for ot in ["COSMOS", "SATELLITE", "SINGULARITY"]:
            if ot not in cgs:
                return _fail(
                    gate,
                    "MISSING_COHERENCE_GAP_STATS",
                    f"result.orbit_analysis.coherence_gap_stats.{ot}",
                    f"coherence_gap_stats missing orbit type '{ot}'",
                ), False
            stat = cgs[ot]
            for key in ["c_real", "c_perm_mean", "c_perm_std", "z_score", "p_value"]:
                if key not in stat:
                    return _fail(
                        gate,
                        "MISSING_COHERENCE_GAP_STATS",
                        f"result.orbit_analysis.coherence_gap_stats.{ot}.{key}",
                        f"coherence_gap_stats.{ot} missing required key '{key}'",
                    ), False

    # --- Curvature invariant check (conditional on generator_curvature present) ---
    gc = cert["result"].get("generator_curvature")
    if gc is not None:
        import hashlib as _hashlib
        lambda_orbit = cert["model_config"]["lambda_orbit"]
        lpe = cert["result"].get("lr_per_epoch")
        if not lpe:
            n_ep = len(cert["result"]["energy_per_epoch"])
            lpe = [float(cert["model_config"]["lr"])] * n_ep

        recomputed = [round(1.0 - abs(1.0 - float(lr_ep) * lambda_orbit), 8)
                      for lr_ep in lpe]

        # Check kappa_hash
        kappa_payload = json.dumps(recomputed, separators=(",", ":")).encode()
        recomputed_hash = _hashlib.sha256(kappa_payload).hexdigest()
        if recomputed_hash != gc.get("kappa_hash", ""):
            return _fail(
                gate, "CURVATURE_RECOMPUTE_MISMATCH",
                "result.generator_curvature.kappa_hash",
                f"recomputed kappa_hash {recomputed_hash} != cert {gc.get('kappa_hash')}",
            ), False

        # Check individual values
        cert_kappas = gc.get("kappa_hat_per_epoch", [])
        if len(cert_kappas) != len(recomputed):
            return _fail(
                gate, "CURVATURE_RECOMPUTE_MISMATCH",
                "result.generator_curvature.kappa_hat_per_epoch",
                f"length mismatch: cert={len(cert_kappas)} recomputed={len(recomputed)}",
            ), False
        for i, (ck, rk) in enumerate(zip(cert_kappas, recomputed)):
            if round(float(ck), 8) != rk:
                return _fail(
                    gate, "CURVATURE_RECOMPUTE_MISMATCH",
                    f"result.generator_curvature.kappa_hat_per_epoch[{i}]",
                    f"cert={ck} recomputed={rk} at epoch {i+1}",
                ), False

        # Check for negative curvature (structural instability)
        for i, k in enumerate(recomputed):
            if k < 0.0:
                lr_ep = float(lpe[i])
                return _fail(
                    gate, "NEGATIVE_GENERATOR_CURVATURE",
                    f"result.generator_curvature.kappa_hat_per_epoch[{i}]",
                    f"kappa_hat={k} < 0 at epoch {i+1}: "
                    f"lr*lambda_orbit={lr_ep*lambda_orbit:.6f} > 2 "
                    f"(lr={lr_ep}, lambda_orbit={lambda_orbit})",
                ), False

        # Check min_kappa_hat and min_kappa_epoch
        actual_min = min(recomputed)
        actual_min_ep = next(
            i + 1 for i, v in enumerate(recomputed) if v == actual_min
        )
        if round(float(gc.get("min_kappa_hat", 0)), 8) != round(actual_min, 8):
            return _fail(
                gate, "CURVATURE_RECOMPUTE_MISMATCH",
                "result.generator_curvature.min_kappa_hat",
                f"cert={gc.get('min_kappa_hat')} recomputed={actual_min}",
            ), False
        if gc.get("min_kappa_epoch") != actual_min_ep:
            return _fail(
                gate, "CURVATURE_RECOMPUTE_MISMATCH",
                "result.generator_curvature.min_kappa_epoch",
                f"cert={gc.get('min_kappa_epoch')} recomputed={actual_min_ep}",
            ), False

    return _pass(gate, {"orbit_map_hash": actual_hash}), True


# ---------------------------------------------------------------------------
# Gate 4 -- Deterministic replay
# ---------------------------------------------------------------------------

def gate4_replay(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate4_deterministic_replay"
    mc = cert["model_config"]

    replayed = train_qa_orbit_reg_rbm(
        n_samples=mc["n_samples"],
        n_epochs=mc["n_epochs"],
        lr=mc["lr"],
        lambda_orbit=mc["lambda_orbit"],
        seed=mc["seed"],
        lr_schedule=mc.get("lr_schedule"),
    )

    expected_hash = cert["trace"]["trace_hash"]
    actual_hash = replayed["trace_hash"]
    if actual_hash != expected_hash:
        return _fail(gate, "TRACE_HASH_MISMATCH", "trace.trace_hash",
                     "deterministic replay produced different trace hash"), False

    expected_reg_hash = cert["result"]["reg_trace_hash"]
    actual_reg_hash = replayed["reg_trace_hash"]
    if actual_reg_hash != expected_reg_hash:
        return _fail(gate, "TRACE_HASH_MISMATCH", "result.reg_trace_hash",
                     "deterministic replay produced different reg_trace_hash"), False

    gc = cert["result"].get("generator_curvature")
    if gc is not None:
        import hashlib as _hashlib
        lambda_orbit = cert["model_config"]["lambda_orbit"]
        replayed_lpe = replayed.get("lr_per_epoch", [])
        recomputed_kappas = [round(1.0 - abs(1.0 - float(lr_ep) * lambda_orbit), 8)
                             for lr_ep in replayed_lpe]
        kappa_payload = json.dumps(recomputed_kappas, separators=(",", ":")).encode()
        replayed_kappa_hash = _hashlib.sha256(kappa_payload).hexdigest()
        if replayed_kappa_hash != gc.get("kappa_hash", ""):
            return _fail(gate, "TRACE_HASH_MISMATCH",
                         "result.generator_curvature.kappa_hash",
                         "kappa_hash from replay does not match cert"), False

    if cert["result"]["status"] != replayed["status"]:
        return _fail(gate, "TRACE_HASH_MISMATCH", "result.status",
                     f"cert status {cert['result']['status']!r} != replay status {replayed['status']!r}"), False

    return _pass(gate, {
        "trace_hash":     actual_hash,
        "reg_trace_hash": actual_reg_hash,
        "replay_status":  replayed["status"],
    }), True


# ---------------------------------------------------------------------------
# Gate 5 -- Invariant_diff contract
# ---------------------------------------------------------------------------

_FAIL_STATUSES = {"GRADIENT_EXPLOSION", "REGULARIZER_NUMERIC_INSTABILITY"}


def gate5_invariant_diff(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate5_invariant_diff"
    status = cert["result"]["status"]
    idiff  = cert["result"]["invariant_diff"]

    if status in ("CONVERGED", "STALLED"):
        if idiff is not None:
            return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                         "result.invariant_diff",
                         f"invariant_diff must be null when status is {status!r}"), False
        return _pass(gate, {"status": status, "invariant_diff": "null (correct)"}), True

    if status in _FAIL_STATUSES:
        if idiff is None:
            return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                         "result.invariant_diff",
                         f"invariant_diff must be present when status is {status!r}"), False
        for k in ["fail_type", "target_path", "reason"]:
            if k not in idiff:
                return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION",
                             f"result.invariant_diff.{k}",
                             f"invariant_diff missing required key '{k}'"), False
        return _fail(gate, status,
                     idiff.get("target_path", "result.status"),
                     idiff.get("reason", f"training failed with status {status}")), False

    return _fail(gate, "INVARIANT_DIFF_CONTRACT_VIOLATION", "result.status",
                 f"unknown status {status!r}"), False


# ---------------------------------------------------------------------------
# Gate 6 -- LR schedule contract (conditional on lr_per_epoch presence)
# ---------------------------------------------------------------------------

def gate6_lr_schedule(cert: dict) -> Tuple[dict, bool]:
    gate = "Gate6_lr_schedule"
    res = cert["result"]
    mc = cert["model_config"]

    # Gate 6 is only active when lr_per_epoch is present in the result
    if "lr_per_epoch" not in res:
        return _pass(gate, {"skipped": "lr_per_epoch not present in result"}), True

    lr_per_epoch = res["lr_per_epoch"]
    n_epochs_result = len(res["energy_per_epoch"])
    lr_schedule = mc.get("lr_schedule")

    # Verify length: lr_per_epoch must have the same number of entries as energy_per_epoch
    if len(lr_per_epoch) != n_epochs_result:
        return _fail(
            gate,
            "LR_SCHEDULE_INVALID",
            "result.lr_per_epoch",
            f"lr_per_epoch length {len(lr_per_epoch)} does not match "
            f"energy_per_epoch length {n_epochs_result}",
        ), False

    if lr_schedule is None or not isinstance(lr_schedule, dict):
        # Fixed-lr path: all entries must equal model_config.lr
        fixed_lr = float(mc["lr"])
        for i, v in enumerate(lr_per_epoch):
            expected = round(fixed_lr, 8)
            if round(float(v), 8) != expected:
                return _fail(
                    gate,
                    "LR_SCHEDULE_INVALID",
                    f"result.lr_per_epoch[{i}]",
                    f"lr_per_epoch[{i}]={v} but lr_schedule is null/absent so all "
                    f"values must equal model_config.lr={fixed_lr}",
                ), False
        return _pass(gate, {"mode": "fixed_lr", "lr": fixed_lr,
                             "n_epochs": n_epochs_result}), True

    # Step-schedule path: validate structure then verify each entry
    if lr_schedule.get("type") != "step":
        return _fail(
            gate,
            "LR_SCHEDULE_INVALID",
            "model_config.lr_schedule.type",
            f"lr_schedule.type must be 'step', got {lr_schedule.get('type')!r}",
        ), False

    steps = lr_schedule.get("steps", [])
    if not steps or steps[0].get("epoch") != 1:
        return _fail(
            gate,
            "LR_SCHEDULE_INVALID",
            "model_config.lr_schedule.steps[0].epoch",
            "lr_schedule.steps[0].epoch must be 1",
        ), False

    # Build expected lr list using the same helper as the trainer
    # n_epochs here is the full configured n_epochs (model_config), but
    # lr_per_epoch may be shorter if training halted early.  We compute
    # the full table and slice to len(lr_per_epoch).
    n_epochs_config = mc["n_epochs"]
    expected_lrs = _build_lr_lookup(lr_schedule, n_epochs_config, float(mc["lr"]))
    expected_slice = expected_lrs[:len(lr_per_epoch)]

    for i, (actual, expected) in enumerate(zip(lr_per_epoch, expected_slice)):
        if round(float(actual), 8) != round(float(expected), 8):
            return _fail(
                gate,
                "LR_SCHEDULE_INVALID",
                f"result.lr_per_epoch[{i}]",
                f"lr_per_epoch[{i}]={actual} but schedule says epoch {i+1} "
                f"should use lr={expected}",
            ), False

    return _pass(gate, {
        "mode": "step_schedule",
        "steps": steps,
        "n_epochs_checked": len(lr_per_epoch),
    }), True


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

    g6, ok = gate6_lr_schedule(cert)
    results.append(g6)
    if not ok:
        return {"cert_id": cert_id, "status": "FAIL", "results": results}

    return {"cert_id": cert_id, "status": "PASS", "results": results}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_FIXTURES_DIR = os.path.join(_DIR, "fixtures")

_SELF_TEST_CASES = [
    ("valid_orbit_reg_stable_run.json",             "PASS"),
    ("invalid_regularizer_instability.json",        "FAIL"),
    ("invalid_trace_mismatch.json",                 "FAIL"),
    ("valid_orbit_reg_lr_decay_stable_run.json",    "PASS"),
    ("invalid_lr_schedule.json",                    "FAIL"),
    ("valid_orbit_reg_kappa_stable.json",           "PASS"),
    ("invalid_negative_generator_curvature.json",   "FAIL"),
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
    parser = argparse.ArgumentParser(
        description="QA_KONA_EBM_QA_NATIVE_ORBIT_REG_CERT.v1 validator"
    )
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
