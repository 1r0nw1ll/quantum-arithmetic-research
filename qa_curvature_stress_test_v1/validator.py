#!/usr/bin/env python3
"""
validator.py

QA_CURVATURE_STRESS_TEST_BUNDLE.v1 validator (Machine Tract).

Family [71]: Cross-family curvature stress-test bundle.

Gates:
  1) Schema validity — required fields, schema_id, types
  2) Entry anchor coherence — entry_id pattern, model_config ranges,
     generator_curvature required fields
  3) κ recompute integrity — recompute kappa_hat_per_epoch from (lr, lambda_orbit),
     verify kappa_hash, min_kappa_hat, min_kappa_epoch; fail NEGATIVE_GENERATOR_CURVATURE
     or CURVATURE_RECOMPUTE_MISMATCH
  4) κ sign prediction alignment — expected_sign from recomputed min_kappa_hat vs
     predicted.kappa_sign; fail KAPPA_SIGN_MISMATCH
  5) Monoidal bottleneck law — TENSOR compositions must satisfy
     kappa_hat_composed == min(component kappas); fail BOTTLE_NECK_VIOLATION or
     COMPOSITION_REF_MISSING

CLI:
    python validator.py <bundle.json>
    python validator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical JSON + SHA256
# ---------------------------------------------------------------------------

def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _kappa_hash(kappas: list) -> str:
    return _sha256_hex(_canonical_json(kappas))


# ---------------------------------------------------------------------------
# Gate result structures
# ---------------------------------------------------------------------------

class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class Diff:
    gate: int
    fail_type: str
    path: str
    reason: str


@dataclass
class GateResult:
    gate_id: int
    status: GateStatus
    message: str
    diffs: List[Diff] = field(default_factory=list)


def _pass(gate_id: int, msg: str) -> GateResult:
    return GateResult(gate_id, GateStatus.PASS, msg)


def _fail(gate_id: int, fail_type: str, path: str, reason: str) -> GateResult:
    d = Diff(gate=gate_id, fail_type=fail_type, path=path, reason=reason)
    return GateResult(gate_id, GateStatus.FAIL, f"{fail_type} @ {path} — {reason}", [d])


# ---------------------------------------------------------------------------
# Gate 1: Schema validity
# ---------------------------------------------------------------------------

_ENTRY_ID_RE = re.compile(r"^[a-z0-9_\-]{3,64}$")
_KAPPA_HASH_RE = re.compile(r"^[a-f0-9]{64}$")


def _gate1_schema(cert: Any) -> GateResult:
    """Gate 1: schema_id, required top-level fields, result structure."""
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "cert must be a JSON object")

    schema_id = cert.get("schema_id")
    if schema_id != "QA_CURVATURE_STRESS_TEST_BUNDLE.v1":
        return _fail(1, "SCHEMA_ID_MISMATCH", "schema_id",
                     f"expected 'QA_CURVATURE_STRESS_TEST_BUNDLE.v1', got {schema_id!r}")

    for key in ("cert_id", "created_utc", "subject", "result"):
        if key not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", key,
                         f"required top-level field '{key}' missing")

    if not isinstance(cert["cert_id"], str) or len(cert["cert_id"]) < 6:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "cert_id",
                     "cert_id must be a string with minLength 6")

    subject = cert["subject"]
    if not isinstance(subject, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject", "subject must be an object")
    for key in ("bundle_version", "purpose"):
        if key not in subject:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"subject.{key}",
                         f"required field '{key}' missing in subject")
    if subject.get("bundle_version") != "v1":
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.bundle_version",
                     f"bundle_version must be 'v1', got {subject.get('bundle_version')!r}")

    result = cert["result"]
    if not isinstance(result, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result", "result must be an object")
    for key in ("status", "entries", "compositions"):
        if key not in result:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"result.{key}",
                         f"required field '{key}' missing in result")

    if result["status"] not in ("OK", "FAIL"):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.status",
                     f"status must be 'OK' or 'FAIL', got {result['status']!r}")

    entries = result["entries"]
    if not isinstance(entries, list) or len(entries) < 1:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.entries",
                     "entries must be a non-empty array")

    compositions = result["compositions"]
    if not isinstance(compositions, list):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.compositions",
                     "compositions must be an array")

    return _pass(1, f"schema valid — {len(entries)} entries, {len(compositions)} compositions")


# ---------------------------------------------------------------------------
# Gate 2: Entry anchor coherence
# ---------------------------------------------------------------------------

def _gate2_entry_coherence(cert: Any) -> GateResult:
    """Gate 2: validate each entry's structure and config."""
    entries = cert["result"]["entries"]

    for i, entry in enumerate(entries):
        path_prefix = f"result.entries[{i}]"

        if not isinstance(entry, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", path_prefix, "entry must be an object")

        for key in ("entry_id", "family_id", "family_name", "model_config", "result"):
            if key not in entry:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{path_prefix}.{key}",
                             f"required field '{key}' missing in entry")

        entry_id = entry["entry_id"]
        if not isinstance(entry_id, str) or not _ENTRY_ID_RE.match(entry_id):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{path_prefix}.entry_id",
                         f"entry_id must match ^[a-z0-9_\\-]{{3,64}}$, got {entry_id!r}")

        mc = entry["model_config"]
        if not isinstance(mc, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{path_prefix}.model_config",
                         "model_config must be an object")
        for key in ("lr", "lambda_orbit"):
            if key not in mc:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING",
                             f"{path_prefix}.model_config.{key}",
                             f"required field '{key}' missing in model_config")
        if not isinstance(mc["lr"], (int, float)) or mc["lr"] <= 0:
            return _fail(2, "CONFIG_INVALID", f"{path_prefix}.model_config.lr",
                         f"lr must be a positive number, got {mc['lr']!r}")
        if not isinstance(mc["lambda_orbit"], (int, float)) or mc["lambda_orbit"] < 0:
            return _fail(2, "CONFIG_INVALID", f"{path_prefix}.model_config.lambda_orbit",
                         f"lambda_orbit must be >= 0, got {mc['lambda_orbit']!r}")

        entry_result = entry["result"]
        if not isinstance(entry_result, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{path_prefix}.result",
                         "entry result must be an object")
        for key in ("generator_curvature", "predicted", "observed"):
            if key not in entry_result:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING",
                             f"{path_prefix}.result.{key}",
                             f"required field '{key}' missing in entry result")

        gc = entry_result["generator_curvature"]
        if not isinstance(gc, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature",
                         "generator_curvature must be an object")
        for key in ("definition", "kappa_hat_per_epoch", "min_kappa_hat",
                    "min_kappa_epoch", "kappa_hash", "max_dev_norm", "max_dev_epoch"):
            if key not in gc:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING",
                             f"{path_prefix}.result.generator_curvature.{key}",
                             f"required field '{key}' missing in generator_curvature")

        kappas = gc["kappa_hat_per_epoch"]
        if not isinstance(kappas, list) or len(kappas) < 1:
            return _fail(2, "SCHEMA_TYPE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.kappa_hat_per_epoch",
                         "kappa_hat_per_epoch must be a non-empty array")
        if not all(isinstance(k, (int, float)) for k in kappas):
            return _fail(2, "SCHEMA_TYPE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.kappa_hat_per_epoch",
                         "kappa_hat_per_epoch items must be numbers")

    return _pass(2, f"entry anchor coherence OK — {len(entries)} entries")


# ---------------------------------------------------------------------------
# Gate 3: κ recompute integrity
# ---------------------------------------------------------------------------

def _recompute_kappas(entry: Dict[str, Any]) -> List[float]:
    """Recompute kappa_hat_per_epoch from model_config and optional lr_per_epoch."""
    mc = entry["model_config"]
    lr_const = float(mc["lr"])
    lambda_orbit = float(mc["lambda_orbit"])
    cert_kappas = entry["result"]["generator_curvature"]["kappa_hat_per_epoch"]
    n_epochs = len(cert_kappas)

    lr_per_epoch_cert = entry["result"].get("lr_per_epoch")
    if lr_per_epoch_cert is not None:
        lr_list = [float(x) for x in lr_per_epoch_cert]
    else:
        lr_list = [lr_const] * n_epochs

    return [round(1.0 - abs(1.0 - lr_i * lambda_orbit), 8) for lr_i in lr_list]


def _gate3_kappa_recompute(cert: Any) -> Tuple[GateResult, Dict[str, float]]:
    """Gate 3: κ recompute integrity. Returns (result, {entry_id: min_kappa})."""
    entries = cert["result"]["entries"]
    min_kappas: Dict[str, float] = {}

    for i, entry in enumerate(entries):
        path_prefix = f"result.entries[{i}]"
        entry_id = entry["entry_id"]
        mc = entry["model_config"]
        entry_result = entry["result"]
        gc = entry_result["generator_curvature"]
        cert_kappas = gc["kappa_hat_per_epoch"]
        n_epochs = len(cert_kappas)

        # Validate lr_per_epoch length if present
        lr_per_epoch_cert = entry_result.get("lr_per_epoch")
        if lr_per_epoch_cert is not None:
            if not isinstance(lr_per_epoch_cert, list):
                return _fail(3, "SCHEMA_TYPE_MISMATCH",
                             f"{path_prefix}.result.lr_per_epoch",
                             "lr_per_epoch must be an array"), {}
            if len(lr_per_epoch_cert) != n_epochs:
                return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                             f"{path_prefix}.result.lr_per_epoch",
                             f"lr_per_epoch length {len(lr_per_epoch_cert)} != "
                             f"kappa_hat_per_epoch length {n_epochs}"), {}

        recomputed = _recompute_kappas(entry)

        # Check for negative kappas
        for ep_idx, k in enumerate(recomputed):
            if k < 0:
                return _fail(3, "NEGATIVE_GENERATOR_CURVATURE",
                             f"{path_prefix}.result.generator_curvature.kappa_hat_per_epoch[{ep_idx}]",
                             f"recomputed kappa {k} < 0 at epoch {ep_idx + 1}"), {}

        # Check each recomputed value matches cert
        for ep_idx, (r_k, c_k) in enumerate(zip(recomputed, cert_kappas)):
            if round(r_k, 8) != round(float(c_k), 8):
                return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                             f"{path_prefix}.result.generator_curvature.kappa_hat_per_epoch[{ep_idx}]",
                             f"recomputed {r_k} != cert {c_k} at epoch {ep_idx + 1}"), {}

        # Check kappa_hash
        expected_hash = _kappa_hash(recomputed)
        cert_hash = gc["kappa_hash"]
        if not _KAPPA_HASH_RE.match(str(cert_hash)):
            return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.kappa_hash",
                         "kappa_hash is not a valid 64-char hex string"), {}
        if expected_hash != cert_hash:
            return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.kappa_hash",
                         f"recomputed hash {expected_hash} != cert {cert_hash}"), {}

        # Check min_kappa_hat
        min_k = min(recomputed)
        cert_min = round(float(gc["min_kappa_hat"]), 8)
        if round(min_k, 8) != cert_min:
            return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.min_kappa_hat",
                         f"recomputed min {min_k} != cert {gc['min_kappa_hat']}"), {}

        # Check min_kappa_epoch (1-indexed, first occurrence)
        argmin_1idx = recomputed.index(min_k) + 1
        cert_argmin = int(gc["min_kappa_epoch"])
        if argmin_1idx != cert_argmin:
            return _fail(3, "CURVATURE_RECOMPUTE_MISMATCH",
                         f"{path_prefix}.result.generator_curvature.min_kappa_epoch",
                         f"recomputed argmin epoch {argmin_1idx} != cert {cert_argmin}"), {}

        min_kappas[entry_id] = min_k

    return _pass(3, f"κ recompute integrity OK — {len(entries)} entries"), min_kappas


# ---------------------------------------------------------------------------
# Gate 4: κ sign prediction alignment
# ---------------------------------------------------------------------------

def _gate4_sign_prediction(cert: Any, min_kappas: Dict[str, float]) -> GateResult:
    """Gate 4: predicted.kappa_sign must match recomputed min_kappa_hat."""
    entries = cert["result"]["entries"]

    for i, entry in enumerate(entries):
        path_prefix = f"result.entries[{i}]"
        entry_id = entry["entry_id"]

        if entry_id not in min_kappas:
            # Should not happen if Gate 3 passed, but be safe
            return _fail(4, "KAPPA_SIGN_MISMATCH",
                         f"{path_prefix}.result.predicted.kappa_sign",
                         f"no recomputed min_kappa for entry {entry_id!r}")

        min_k = min_kappas[entry_id]
        if min_k > 1e-8:
            expected_sign = "POS"
        elif min_k < -1e-8:
            expected_sign = "NEG"
        else:
            expected_sign = "ZERO"

        predicted = entry["result"]["predicted"]
        cert_sign = predicted.get("kappa_sign")
        if cert_sign != expected_sign:
            return _fail(4, "KAPPA_SIGN_MISMATCH",
                         f"{path_prefix}.result.predicted.kappa_sign",
                         f"expected {expected_sign!r} (min_kappa={min_k}), "
                         f"got {cert_sign!r}")

    return _pass(4, f"κ sign prediction alignment OK — {len(entries)} entries")


# ---------------------------------------------------------------------------
# Gate 5: Monoidal bottleneck law
# ---------------------------------------------------------------------------

def _gate5_bottleneck(cert: Any, min_kappas: Dict[str, float]) -> GateResult:
    """Gate 5: TENSOR compositions must satisfy kappa_hat_composed == min(component kappas)."""
    compositions = cert["result"].get("compositions", [])

    for j, comp in enumerate(compositions):
        path_prefix = f"result.compositions[{j}]"
        comp_id = comp.get("composition_id", f"[{j}]")

        op = comp.get("op")
        if op != "TENSOR":
            return _fail(5, "COMPOSITION_OP_INVALID",
                         f"{path_prefix}.op",
                         f"op must be 'TENSOR', got {op!r}")

        components = comp.get("components", [])
        if not isinstance(components, list) or len(components) < 2:
            return _fail(5, "COMPOSITION_REF_MISSING",
                         f"{path_prefix}.components",
                         "compositions must have at least 2 components")

        component_kappas: List[float] = []
        for ref in components:
            if ref not in min_kappas:
                return _fail(5, "COMPOSITION_REF_MISSING",
                             f"{path_prefix}.components",
                             f"component entry_id {ref!r} not found in bundle entries")
            component_kappas.append(min_kappas[ref])

        expected_composed = round(min(component_kappas), 8)
        cert_composed = round(float(comp["kappa_hat_composed"]), 8)
        if cert_composed != expected_composed:
            return _fail(5, "BOTTLE_NECK_VIOLATION",
                         f"{path_prefix}.kappa_hat_composed",
                         f"expected min({[round(k,8) for k in component_kappas]})={expected_composed}, cert={cert_composed}")

    return _pass(5, f"monoidal bottleneck law OK — {len(compositions)} compositions")


# ---------------------------------------------------------------------------
# Full validation pipeline
# ---------------------------------------------------------------------------

def validate_bundle(cert: Any) -> List[GateResult]:
    """Run all 5 gates. Stops at first failing gate."""
    results: List[GateResult] = []

    g1 = _gate1_schema(cert)
    results.append(g1)
    if g1.status == GateStatus.FAIL:
        return results

    g2 = _gate2_entry_coherence(cert)
    results.append(g2)
    if g2.status == GateStatus.FAIL:
        return results

    g3, min_kappas = _gate3_kappa_recompute(cert)
    results.append(g3)
    if g3.status == GateStatus.FAIL:
        return results

    g4 = _gate4_sign_prediction(cert, min_kappas)
    results.append(g4)
    if g4.status == GateStatus.FAIL:
        return results

    g5 = _gate5_bottleneck(cert, min_kappas)
    results.append(g5)

    return results


def _print_results(results: List[GateResult]) -> bool:
    """Print gate results. Returns True if all PASS."""
    ok = True
    for gr in results:
        if gr.status == GateStatus.PASS:
            print(f"[PASS] Gate {gr.gate_id}: {gr.message}")
        else:
            ok = False
            print(f"[FAIL] Gate {gr.gate_id}: {gr.message}")
    return ok


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_SELF_TEST_CASES = [
    ("valid_cross_family_bundle.json",           "PASS"),
    ("valid_real_cross_family_bundle.json",       "PASS"),
    ("invalid_missing_family.json",              "FAIL"),
    ("invalid_kappa_sign_mismatch.json",         "FAIL"),
    ("invalid_bottleneck_violation.json",        "FAIL"),
]


def run_self_test(fixtures_dir: str) -> bool:
    """Run self-test against all 4 fixture files."""
    all_ok = True
    for fixture_name, expected in _SELF_TEST_CASES:
        fixture_path = os.path.join(fixtures_dir, fixture_name)
        if not os.path.exists(fixture_path):
            print(f"[FAIL] self-test: fixture not found: {fixture_path}")
            all_ok = False
            continue

        with open(fixture_path, "r", encoding="utf-8") as fh:
            try:
                cert = json.load(fh)
            except json.JSONDecodeError as exc:
                print(f"[FAIL] self-test {fixture_name}: JSON parse error: {exc}")
                all_ok = False
                continue

        results = validate_bundle(cert)
        passed = all(r.status == GateStatus.PASS for r in results)
        actual = "PASS" if passed else "FAIL"

        if actual == expected:
            print(f"[PASS] self-test {fixture_name}: got {actual} (expected {expected})")
        else:
            print(f"[FAIL] self-test {fixture_name}: got {actual} (expected {expected})")
            for gr in results:
                tag = "[PASS]" if gr.status == GateStatus.PASS else "[FAIL]"
                print(f"       {tag} Gate {gr.gate_id}: {gr.message}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QA Curvature Stress-Test Bundle (v1) validator"
    )
    parser.add_argument("bundle", nargs="?", help="path to bundle JSON file")
    parser.add_argument("--self-test", action="store_true",
                        help="run self-test against fixture directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(script_dir, "fixtures")

    if args.self_test:
        ok = run_self_test(fixtures_dir)
        sys.exit(0 if ok else 1)

    if not args.bundle:
        parser.error("provide a bundle JSON file or use --self-test")

    with open(args.bundle, "r", encoding="utf-8") as fh:
        cert = json.load(fh)

    results = validate_bundle(cert)
    ok = _print_results(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
