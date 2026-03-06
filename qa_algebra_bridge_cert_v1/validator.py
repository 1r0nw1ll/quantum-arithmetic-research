#!/usr/bin/env python3
"""
validator.py

QA_ALGEBRA_BRIDGE_CERT.v1 validator (Machine tract).

Gates:
  1) Schema shape / type checks
  2) Deterministic generator probe recomputation
  3) Word convention / order probes
  4) Component-bridge probes (word + scale)
  5) Semantics hash binding + invariant_diff claim verification

CLI:
  python validator.py <cert.json>
  python validator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import qa_structural_algebra as qsa
from qa_algebra_bridge_cert_v1.semantics_anchor import (
    BRIDGE_CORE_PROPERTIES,
    BRIDGE_GENERATOR_DEFS,
    BRIDGE_GENERATOR_SEMANTICS_REF,
    BRIDGE_SEMANTICS_ID,
    BRIDGE_SEMANTICS_SHA256,
    BRIDGE_WORD_APPLICATION_ORDER,
    build_bridge_semantics_payload,
)


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


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _pass(gate_id: int, message: str) -> GateResult:
    return GateResult(gate_id, GateStatus.PASS, message)


def _fail(gate_id: int, fail_type: str, path: str, reason: str) -> GateResult:
    return GateResult(gate_id, GateStatus.FAIL, f"{fail_type} @ {path} -- {reason}", [Diff(gate_id, fail_type, path, reason)])


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_int2(v: Any) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v)


def _semantics_payload(cert: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "semantics_id": cert["subject"]["semantics_id"],
        "generator_semantics_ref": cert["subject"]["generator_semantics_ref"],
        "word_application_order": cert["subject"]["word_application_order"],
        "generator_defs": cert["semantics"]["generator_defs"],
        "core_properties": cert["semantics"]["core_properties"],
    }


def _gate1_schema(cert: Any) -> GateResult:
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "certificate must be object")

    required = ["schema_version", "created_utc", "subject", "semantics", "claims", "fixtures", "result"]
    for key in required:
        if key not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", key, "required field missing")

    if cert.get("schema_version") != "QA_ALGEBRA_BRIDGE_CERT.v1":
        return _fail(
            1,
            "SCHEMA_VERSION_MISMATCH",
            "schema_version",
            f"expected 'QA_ALGEBRA_BRIDGE_CERT.v1', got {cert.get('schema_version')!r}",
        )

    subject = cert.get("subject")
    if not isinstance(subject, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject", "subject must be object")
    for key in ("semantics_id", "generator_semantics_ref", "word_application_order"):
        if key not in subject:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"subject.{key}", "required")

    if subject.get("semantics_id") != BRIDGE_SEMANTICS_ID:
        return _fail(1, "SEMANTICS_ID_MISMATCH", "subject.semantics_id", f"must be {BRIDGE_SEMANTICS_ID}")
    if subject.get("generator_semantics_ref") != BRIDGE_GENERATOR_SEMANTICS_REF:
        return _fail(1, "SEMANTICS_REF_MISMATCH", "subject.generator_semantics_ref", f"must be {BRIDGE_GENERATOR_SEMANTICS_REF}")
    if subject.get("word_application_order") != BRIDGE_WORD_APPLICATION_ORDER:
        return _fail(1, "WORD_CONVENTION_MISMATCH", "subject.word_application_order", f"must be {BRIDGE_WORD_APPLICATION_ORDER}")

    semantics = cert.get("semantics")
    if not isinstance(semantics, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "semantics", "semantics must be object")
    for key in ("generator_defs", "core_properties", "semantics_sha256"):
        if key not in semantics:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"semantics.{key}", "required")

    gdefs = semantics.get("generator_defs")
    if not isinstance(gdefs, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "semantics.generator_defs", "must be object")
    for key in ("sigma", "mu", "R", "lambda_k", "nu"):
        if key not in gdefs or not isinstance(gdefs[key], str):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"semantics.generator_defs.{key}", "must be string")
    if not isinstance(semantics.get("core_properties"), list) or len(semantics["core_properties"]) < 1:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "semantics.core_properties", "must be non-empty list")
    if not isinstance(semantics.get("semantics_sha256"), str):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "semantics.semantics_sha256", "must be string")

    claims = cert.get("claims")
    if not isinstance(claims, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "claims", "claims must be object")
    for key in ("generator_behavior", "word_convention", "component_bridge", "semantics_hash_binding"):
        if not isinstance(claims.get(key), bool):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"claims.{key}", "must be boolean")

    fixtures = cert.get("fixtures")
    if not isinstance(fixtures, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "fixtures", "fixtures must be object")
    for key in ("generator_probes", "word_probes", "component_probes"):
        if not isinstance(fixtures.get(key), list) or len(fixtures[key]) < 1:
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"fixtures.{key}", "must be non-empty list")

    result = cert.get("result")
    if not isinstance(result, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result", "result must be object")
    for key in ("ok", "failures", "invariant_diff_map"):
        if key not in result:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"result.{key}", "required")
    idm = result.get("invariant_diff_map")
    if not isinstance(idm, dict) or "entries" not in idm or "rollup_sha256" not in idm:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map", "must include entries and rollup_sha256")

    return _pass(1, "schema shape valid")


def _gate2_generator_probes(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["generator_behavior"]:
        return _pass(2, "generator behavior claim disabled")

    fixtures = cert["fixtures"]
    for i, sample in enumerate(fixtures["generator_probes"]):
        p = f"fixtures.generator_probes[{i}]"
        if not isinstance(sample, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("op", "state", "expected_ok", "expected_fail_type"):
            if key not in sample:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        if not _is_int2(sample["state"]):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{p}.state", "must be int[2]")

        op = sample["op"]
        state = (sample["state"][0], sample["state"][1])
        expected_ok = sample["expected_ok"]

        if op == "sigma":
            res = qsa.sigma(state)
        elif op == "mu":
            res = qsa.mu(state)
        elif op == "R":
            res = qsa.R(state)
        elif op == "lambda_k":
            if "k" not in sample:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.k", "required for lambda_k")
            res = qsa.lambda_k(state, sample["k"])
        elif op == "nu":
            res = qsa.nu(state)
        else:
            return _fail(2, "GENERATOR_PROBE_UNKNOWN_OP", f"{p}.op", f"unsupported op {op!r}")

        if expected_ok:
            if not res["ok"]:
                return _fail(2, "GENERATOR_PROBE_EXPECTED_OK", p, f"expected success, got {res.get('fail_type')}")
            if "expected_value" not in sample or not _is_int2(sample["expected_value"]):
                return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{p}.expected_value", "must be int[2] for expected_ok=true")
            exp_v = (sample["expected_value"][0], sample["expected_value"][1])
            if res["value"] != exp_v:
                return _fail(2, "GENERATOR_PROBE_VALUE_MISMATCH", f"{p}.expected_value", f"expected {exp_v}, recomputed {res['value']}")
        else:
            if res["ok"]:
                return _fail(2, "GENERATOR_PROBE_EXPECTED_FAIL", p, f"expected failure, got {res['value']}")
            exp_ft = sample["expected_fail_type"]
            if exp_ft and res["fail_type"] != exp_ft:
                return _fail(2, "GENERATOR_PROBE_FAILTYPE_MISMATCH", f"{p}.expected_fail_type", f"expected {exp_ft}, recomputed {res['fail_type']}")

    return _pass(2, f"generator probes passed ({len(fixtures['generator_probes'])} samples)")


def _gate3_word_probes(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["word_convention"]:
        return _pass(3, "word convention claim disabled")

    fixtures = cert["fixtures"]

    # Explicit convention anchors for left-to-right application
    rl = qsa.word_to_state("RL", seed=(1, 1))
    lr = qsa.word_to_state("LR", seed=(1, 1))
    if (not rl["ok"]) or (not lr["ok"]) or rl["value"] != (2, 3) or lr["value"] != (3, 2):
        return _fail(
            3,
            "WORD_ORDER_CONVENTION_MISMATCH",
            "subject.word_application_order",
            f"{BRIDGE_WORD_APPLICATION_ORDER} anchor checks failed",
        )

    for i, sample in enumerate(fixtures["word_probes"]):
        p = f"fixtures.word_probes[{i}]"
        if not isinstance(sample, dict):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("word", "seed", "expected_state"):
            if key not in sample:
                return _fail(3, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        if not _is_int2(sample["seed"]) or not _is_int2(sample["expected_state"]):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", p, "seed and expected_state must be int[2]")

        word = sample["word"]
        seed = (sample["seed"][0], sample["seed"][1])
        exp_state = sample["expected_state"]

        res = qsa.word_to_state(word, seed=seed)
        if not res["ok"]:
            return _fail(3, res["fail_type"], p, "word_to_state failed")

        got = list(res["value"])
        if got != exp_state:
            return _fail(3, "WORD_PROBE_STATE_MISMATCH", f"{p}.expected_state", f"expected {exp_state}, recomputed {got}")

    return _pass(3, f"word probes passed ({len(fixtures['word_probes'])} samples)")


def _gate4_component_probes(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["component_bridge"]:
        return _pass(4, "component bridge claim disabled")

    fixtures = cert["fixtures"]
    for i, sample in enumerate(fixtures["component_probes"]):
        p = f"fixtures.component_probes[{i}]"
        if not isinstance(sample, dict):
            return _fail(4, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("state", "expected_scale", "expected_normalized", "expected_word"):
            if key not in sample:
                return _fail(4, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        if not _is_int2(sample["state"]) or not _is_int2(sample["expected_normalized"]):
            return _fail(4, "SCHEMA_TYPE_MISMATCH", p, "state and expected_normalized must be int[2]")

        state = (sample["state"][0], sample["state"][1])
        expected_scale = sample["expected_scale"]
        expected_normalized = (sample["expected_normalized"][0], sample["expected_normalized"][1])
        expected_word = sample["expected_word"]

        sws = qsa.state_to_word_with_scale(state)
        if not sws["ok"]:
            return _fail(4, sws["fail_type"], p, "state_to_word_with_scale failed")

        got = sws["value"]
        if got["scale"] != expected_scale or tuple(got["normalized"]) != expected_normalized or got["word"] != expected_word:
            return _fail(
                4,
                "COMPONENT_PROBE_MISMATCH",
                p,
                f"expected scale={expected_scale}, normalized={expected_normalized}, word={expected_word!r}; recomputed {got}",
            )

        rebuild = qsa.word_to_state(expected_word, seed=(expected_scale, expected_scale))
        if not rebuild["ok"]:
            return _fail(4, rebuild["fail_type"], p, "word_to_state on scaled seed failed")
        if rebuild["value"] != state:
            return _fail(4, "COMPONENT_SCALED_ROUNDTRIP_MISMATCH", p, f"expected {state}, recomputed {rebuild['value']}")

    return _pass(4, f"component probes passed ({len(fixtures['component_probes'])} samples)")


def _gate5_bindings(cert: Dict[str, Any], computed_failures: List[Dict[str, Any]]) -> GateResult:
    diffs: List[Diff] = []

    if cert["claims"]["semantics_hash_binding"]:
        payload = _semantics_payload(cert)
        canonical_payload = build_bridge_semantics_payload()
        if payload != canonical_payload:
            diffs.append(
                Diff(
                    5,
                    "SEMANTICS_PAYLOAD_MISMATCH",
                    "semantics",
                    "claimed semantics payload differs from canonical bridge payload",
                )
            )

        claimed_hash = cert["semantics"].get("semantics_sha256")
        recomputed_hash = _sha256_hex(_canonical_json(payload))
        if claimed_hash != recomputed_hash:
            diffs.append(
                Diff(
                    5,
                    "SEMANTICS_HASH_MISMATCH",
                    "semantics.semantics_sha256",
                    f"claimed {claimed_hash} != recomputed_from_claim {recomputed_hash}",
                )
            )
        if claimed_hash != BRIDGE_SEMANTICS_SHA256:
            diffs.append(
                Diff(
                    5,
                    "SEMANTICS_HASH_MISMATCH",
                    "semantics.semantics_sha256",
                    f"claimed {claimed_hash} != canonical {BRIDGE_SEMANTICS_SHA256}",
                )
            )

    result = cert.get("result", {})
    idm = result.get("invariant_diff_map", {}) if isinstance(result, dict) else {}
    entries = idm.get("entries")
    rollup_claimed = idm.get("rollup_sha256")

    if entries != computed_failures:
        diffs.append(
            Diff(
                5,
                "INVARIANT_DIFF_MAP_CLAIM_MISMATCH",
                "result.invariant_diff_map.entries",
                "claimed entries differ from recomputed gate failures",
            )
        )

    recomputed_rollup = _sha256_hex(_canonical_json(computed_failures))
    if rollup_claimed != recomputed_rollup:
        diffs.append(
            Diff(
                5,
                "INVARIANT_DIFF_MAP_CLAIM_MISMATCH",
                "result.invariant_diff_map.rollup_sha256",
                f"claimed {rollup_claimed} != recomputed {recomputed_rollup}",
            )
        )

    if diffs:
        return GateResult(
            gate_id=5,
            status=GateStatus.FAIL,
            message="BINDING_MISMATCH @ semantics/result invariant maps -- claim does not match recomputed truth",
            diffs=diffs,
        )

    return _pass(5, f"semantics hash + invariant_diff claims verified (entries={len(computed_failures)})")


def validate_cert(cert: Dict[str, Any]) -> Dict[str, Any]:
    gate_log: List[GateResult] = []
    failures: List[Dict[str, Any]] = []

    g1 = _gate1_schema(cert)
    gate_log.append(g1)
    if g1.status == GateStatus.FAIL:
        failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g1.diffs)
    else:
        g2 = _gate2_generator_probes(cert)
        gate_log.append(g2)
        if g2.status == GateStatus.FAIL:
            failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g2.diffs)
        else:
            g3 = _gate3_word_probes(cert)
            gate_log.append(g3)
            if g3.status == GateStatus.FAIL:
                failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g3.diffs)
            else:
                g4 = _gate4_component_probes(cert)
                gate_log.append(g4)
                if g4.status == GateStatus.FAIL:
                    failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g4.diffs)

    g5 = _gate5_bindings(cert, failures)
    gate_log.append(g5)
    if g5.status == GateStatus.FAIL:
        failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g5.diffs)

    rollup = _sha256_hex(_canonical_json(failures))

    ok = len(failures) == 0
    return {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "gates": [{"gate_id": g.gate_id, "status": g.status.value, "message": g.message} for g in gate_log],
        "failures": failures,
        "invariant_diff_map": {
            "entries": failures,
            "rollup_sha256": rollup,
        },
    }


def run_self_test(base_dir: str) -> int:
    fixtures = [
        ("valid_min.json", True),
        ("invalid_bad_word_probe.json", False),
    ]

    ok_all = True
    for name, expected_ok in fixtures:
        path = os.path.join(base_dir, "fixtures", name)
        if not os.path.isfile(path):
            print(f"[FAIL] self-test {name}: missing fixture")
            ok_all = False
            continue
        cert = _load_json(path)
        result = validate_cert(cert)
        actual_ok = bool(result["ok"])
        if actual_ok == expected_ok:
            print(f"[PASS] self-test {name}: got {actual_ok} (expected {expected_ok})")
        else:
            print(f"[FAIL] self-test {name}: got {actual_ok} (expected {expected_ok})")
            print(json.dumps(result, indent=2, sort_keys=True))
            ok_all = False

    if ok_all:
        print("[PASS] all self-tests")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA_ALGEBRA_BRIDGE_CERT.v1 certificates")
    parser.add_argument("file", nargs="?", help="certificate JSON file")
    parser.add_argument("--self-test", action="store_true", help="run self-tests against fixtures")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(BASE_DIR)

    if not args.file:
        parser.error("provide a certificate JSON file or use --self-test")

    cert = _load_json(args.file)
    result = validate_cert(cert)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
