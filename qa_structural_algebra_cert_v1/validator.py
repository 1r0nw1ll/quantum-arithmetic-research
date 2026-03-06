#!/usr/bin/env python3
"""
validator.py

QA_STRUCTURAL_ALGEBRA_CERT.v1 validator (Machine tract).

Gates:
  1) Schema shape / type checks
  2) Deterministic recomputation of sample expectations
  3) Bounded uniqueness + normal-form audit up to N
  4) Scaling component checks vs gcd (+ optional scaled reachability audit)
  5) Verify invariant_diff_map claim against recomputed failures

CLI:
  python validator.py <cert.json>
  python validator.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# Ensure repo-root imports work when validator is run from family directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import qa_structural_algebra as qsa

try:
    from qa_algebra_bridge_cert_v1.semantics_anchor import BRIDGE_SEMANTICS_SHA256
    BRIDGE_SOURCE_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only when bridge module missing/broken
    BRIDGE_SEMANTICS_SHA256 = None
    BRIDGE_SOURCE_ERROR = str(exc)

try:
    from qa_failure_algebra_structure_cert_v1.failure_algebra_anchor import (
        FAILURE_ALGEBRA_ANCHOR_REF,
        FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256,
        FAILURE_TYPES,
        UNIT,
        join as failure_join,
    )
    from qa_failure_algebra_structure_cert_v1.projection_maps import (
        choose_projection_map,
        choose_projection_map_with_reason,
        FAILURE_PROJECTION_MAP_V1_STRUCTURAL,
        PROJECTION_MAP_NAME_STRUCTURAL,
        FAILURE_PROJECTION_MAP_VERSION,
    )
    FAILURE_ALGEBRA_SOURCE_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only when failure algebra module missing/broken
    FAILURE_ALGEBRA_ANCHOR_REF = None
    FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256 = None
    FAILURE_TYPES = ()
    UNIT = "OK"
    failure_join = None
    choose_projection_map = None
    choose_projection_map_with_reason = None
    FAILURE_PROJECTION_MAP_V1_STRUCTURAL = {}
    PROJECTION_MAP_NAME_STRUCTURAL = "unknown"
    FAILURE_PROJECTION_MAP_VERSION = "UNAVAILABLE"
    FAILURE_ALGEBRA_SOURCE_ERROR = str(exc)


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


def _is_int2(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, int) for x in value)


def _gate1_schema(cert: Any) -> GateResult:
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "certificate must be a JSON object")

    required = ["schema_version", "created_utc", "subject", "claims", "fixtures", "result"]
    for key in required:
        if key not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", key, f"required field '{key}' missing")

    if cert.get("schema_version") != "QA_STRUCTURAL_ALGEBRA_CERT.v1":
        return _fail(
            1,
            "SCHEMA_VERSION_MISMATCH",
            "schema_version",
            f"expected 'QA_STRUCTURAL_ALGEBRA_CERT.v1', got {cert.get('schema_version')!r}",
        )

    subject = cert.get("subject")
    if not isinstance(subject, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject", "subject must be object")
    for key in ("seed", "N", "generators", "algebra_bridge_semantics_sha256"):
        if key not in subject:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"subject.{key}", "required")
    if not _is_int2(subject.get("seed")):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.seed", "must be int[2]")
    if not isinstance(subject.get("N"), int) or subject["N"] <= 0:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.N", "must be integer >= 1")
    if not isinstance(subject.get("generators"), list) or len(subject["generators"]) < 1:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.generators", "must be non-empty list")
    if not isinstance(subject.get("algebra_bridge_semantics_sha256"), str):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.algebra_bridge_semantics_sha256", "must be string")
    if BRIDGE_SEMANTICS_SHA256 is None:
        return _fail(
            1,
            "BRIDGE_SEMANTICS_SOURCE_MISSING",
            "subject.algebra_bridge_semantics_sha256",
            f"unable to load bridge semantics source: {BRIDGE_SOURCE_ERROR}",
        )
    if subject["algebra_bridge_semantics_sha256"] != BRIDGE_SEMANTICS_SHA256:
        return _fail(
            1,
            "BRIDGE_SEMANTICS_HASH_MISMATCH",
            "subject.algebra_bridge_semantics_sha256",
            f"expected {BRIDGE_SEMANTICS_SHA256}, got {subject['algebra_bridge_semantics_sha256']}",
        )
    if "failure_algebra_anchor_ref" in subject:
        if not isinstance(subject.get("failure_algebra_anchor_ref"), str):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.failure_algebra_anchor_ref", "must be string")
        if FAILURE_ALGEBRA_ANCHOR_REF is None:
            return _fail(
                1,
                "FAILURE_ALGEBRA_SOURCE_MISSING",
                "subject.failure_algebra_anchor_ref",
                f"unable to load failure algebra source: {FAILURE_ALGEBRA_SOURCE_ERROR}",
            )
        if subject["failure_algebra_anchor_ref"] != FAILURE_ALGEBRA_ANCHOR_REF:
            return _fail(
                1,
                "FAILURE_ALGEBRA_ANCHOR_REF_MISMATCH",
                "subject.failure_algebra_anchor_ref",
                f"expected {FAILURE_ALGEBRA_ANCHOR_REF}, got {subject['failure_algebra_anchor_ref']}",
            )
    if "failure_algebra_anchor_rollup_sha256" in subject:
        if not isinstance(subject.get("failure_algebra_anchor_rollup_sha256"), str):
            return _fail(
                1,
                "SCHEMA_TYPE_MISMATCH",
                "subject.failure_algebra_anchor_rollup_sha256",
                "must be string",
            )
        if "failure_algebra_anchor_ref" not in subject:
            return _fail(
                1,
                "SCHEMA_VALUE_INVALID",
                "subject.failure_algebra_anchor_rollup_sha256",
                "requires subject.failure_algebra_anchor_ref",
            )
        if FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256 is None:
            return _fail(
                1,
                "FAILURE_ALGEBRA_SOURCE_MISSING",
                "subject.failure_algebra_anchor_rollup_sha256",
                f"unable to load failure algebra source: {FAILURE_ALGEBRA_SOURCE_ERROR}",
            )
        if subject["failure_algebra_anchor_rollup_sha256"] != FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256:
            return _fail(
                1,
                "FAILURE_ALGEBRA_ANCHOR_ROLLUP_MISMATCH",
                "subject.failure_algebra_anchor_rollup_sha256",
                f"expected {FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256}, got {subject['failure_algebra_anchor_rollup_sha256']}",
            )

    claims = cert.get("claims")
    if not isinstance(claims, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "claims", "claims must be object")
    for key in ("normal_form", "uniqueness", "scaling_components", "guarded_contraction"):
        if not isinstance(claims.get(key), bool):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"claims.{key}", "must be boolean")

    fixtures = cert.get("fixtures")
    if not isinstance(fixtures, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "fixtures", "fixtures must be object")
    for key in ("roundtrip_samples", "nu_guard_samples", "scale_samples"):
        if not isinstance(fixtures.get(key), list) or len(fixtures[key]) < 1:
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"fixtures.{key}", "must be non-empty list")

    result = cert.get("result")
    if not isinstance(result, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result", "result must be object")
    for key in ("ok", "failures", "invariant_diff_map"):
        if key not in result:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"result.{key}", "required")
    if not isinstance(result.get("ok"), bool):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.ok", "must be boolean")
    if not isinstance(result.get("failures"), list):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.failures", "must be array")
    idm = result.get("invariant_diff_map")
    if not isinstance(idm, dict) or "entries" not in idm or "rollup_sha256" not in idm:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map", "must include entries and rollup_sha256")

    return _pass(1, "schema shape valid")


def _gate2_samples(cert: Dict[str, Any]) -> GateResult:
    fixtures = cert["fixtures"]
    seed = tuple(cert["subject"]["seed"])
    if seed != (1, 1):
        return _fail(2, "SEED_UNSUPPORTED", "subject.seed", "current cert semantics require seed=[1,1]")

    for i, sample in enumerate(fixtures["roundtrip_samples"]):
        p = f"fixtures.roundtrip_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("state", "expected_word", "expected_roundtrip"):
            if key not in sample:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        state = sample["state"]
        if not _is_int2(state):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{p}.state", "must be int[2]")

        state_t = (state[0], state[1])
        res_word = qsa.state_to_word(state_t)
        if not res_word["ok"]:
            return _fail(2, res_word["fail_type"], f"{p}.state", "state_to_word failed")

        if res_word["value"] != sample["expected_word"]:
            return _fail(
                2,
                "ROUNDTRIP_WORD_MISMATCH",
                f"{p}.expected_word",
                f"expected {sample['expected_word']!r}, recomputed {res_word['value']!r}",
            )

        res_state = qsa.word_to_state(sample["expected_word"], seed=seed)
        if not res_state["ok"]:
            return _fail(2, res_state["fail_type"], f"{p}.expected_word", "word_to_state failed")

        got_roundtrip = list(res_state["value"])
        if got_roundtrip != sample["expected_roundtrip"]:
            return _fail(
                2,
                "ROUNDTRIP_STATE_MISMATCH",
                f"{p}.expected_roundtrip",
                f"expected {sample['expected_roundtrip']}, recomputed {got_roundtrip}",
            )

    for i, sample in enumerate(fixtures["nu_guard_samples"]):
        p = f"fixtures.nu_guard_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("state", "expected_ok", "expected_fail_type"):
            if key not in sample:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        if not _is_int2(sample["state"]):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", f"{p}.state", "must be int[2]")

        res = qsa.nu((sample["state"][0], sample["state"][1]))
        expected_ok = sample["expected_ok"]
        if expected_ok is True:
            if not res["ok"]:
                return _fail(2, "NU_GUARD_EXPECTED_OK", p, f"expected success, got {res.get('fail_type')}")
            if "expected_value" in sample and sample["expected_value"] is not None:
                got = list(res["value"])
                if got != sample["expected_value"]:
                    return _fail(2, "NU_VALUE_MISMATCH", f"{p}.expected_value", f"expected {sample['expected_value']}, got {got}")
        else:
            if res["ok"]:
                return _fail(2, "NU_GUARD_EXPECTED_FAIL", p, f"expected failure, got value {res['value']}")
            expected_ft = sample["expected_fail_type"]
            if expected_ft and res["fail_type"] != expected_ft:
                return _fail(2, "NU_FAILTYPE_MISMATCH", f"{p}.expected_fail_type", f"expected {expected_ft}, got {res['fail_type']}")

    return _pass(2, "sample recomputation checks passed")


def _gate3_uniqueness(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["uniqueness"]:
        return _pass(3, "uniqueness claim disabled")

    N = cert["subject"]["N"]
    seed = tuple(cert["subject"]["seed"])
    if seed != (1, 1):
        return _fail(3, "SEED_UNSUPPORTED", "subject.seed", "uniqueness audit expects seed=[1,1]")

    enum_res = qsa.reachable_up_to(N, allow_scaling=False)
    if not enum_res["ok"]:
        return _fail(3, enum_res["fail_type"], "reachable_up_to", "bounded uniqueness enumeration failed")

    states_set, mapping = enum_res["value"]
    if len(states_set) != len(mapping):
        return _fail(3, "NOT_UNIQUE", "reachable_up_to", "state set and mapping cardinality mismatch")

    # Full bounded normal-form audit for all coprime states <= N.
    for b in range(1, N + 1):
        for e in range(1, N + 1):
            if math.gcd(b, e) != 1:
                continue
            state = (b, e)
            sw = qsa.state_to_word(state)
            if not sw["ok"]:
                return _fail(3, sw["fail_type"], "state_to_word", f"failed at state {state}")
            wr = qsa.word_to_state(sw["value"], seed=seed)
            if not wr["ok"]:
                return _fail(3, wr["fail_type"], "word_to_state", f"failed at state {state}")
            if wr["value"] != state:
                return _fail(3, "ROUNDTRIP_STATE_MISMATCH", "word_to_state", f"state {state} roundtrip mismatch")
            if state not in mapping:
                return _fail(3, "NORMAL_FORM_INCOMPLETE", "reachable_up_to", f"missing coprime state {state}")
            if mapping[state] != sw["value"]:
                return _fail(
                    3,
                    "NORMAL_FORM_MISMATCH",
                    "reachable_up_to",
                    f"state {state}: mapping word {mapping[state]!r} != computed {sw['value']!r}",
                )

    return _pass(3, f"bounded uniqueness + normal-form audit passed (N={N}, states={len(states_set)})")


def _gate4_scaling(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["scaling_components"]:
        return _pass(4, "scaling claim disabled")

    fixtures = cert["fixtures"]
    N = cert["subject"]["N"]

    for i, sample in enumerate(fixtures["scale_samples"]):
        p = f"fixtures.scale_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(4, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in ("state", "expected_normalized", "expected_scale"):
            if key not in sample:
                return _fail(4, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")
        if not _is_int2(sample["state"]) or not _is_int2(sample["expected_normalized"]):
            return _fail(4, "SCHEMA_TYPE_MISMATCH", p, "state and expected_normalized must be int[2]")

        state = (sample["state"][0], sample["state"][1])
        exp_norm = tuple(sample["expected_normalized"])
        exp_scale = sample["expected_scale"]

        cg = qsa.component_gcd(state)
        if not cg["ok"]:
            return _fail(4, cg["fail_type"], f"{p}.state", "component_gcd failed")
        if cg["value"] != exp_scale:
            return _fail(4, "SCALE_GCD_MISMATCH", f"{p}.expected_scale", f"expected {exp_scale}, gcd={cg['value']}")

        norm = qsa.normalize_to_coprime(state)
        if not norm["ok"]:
            return _fail(4, norm["fail_type"], f"{p}.state", "normalize_to_coprime failed")
        got_norm, got_scale = norm["value"]
        if got_norm != exp_norm or got_scale != exp_scale:
            return _fail(
                4,
                "SCALE_NORMALIZATION_MISMATCH",
                p,
                f"expected normalized={exp_norm}, scale={exp_scale}; got normalized={got_norm}, scale={got_scale}",
            )

        sws = qsa.state_to_word_with_scale(state)
        if not sws["ok"]:
            return _fail(4, sws["fail_type"], f"{p}.state", "state_to_word_with_scale failed")
        rec = sws["value"]
        if tuple(rec["normalized"]) != exp_norm or rec["scale"] != exp_scale:
            return _fail(
                4,
                "SCALE_WORD_WITH_SCALE_MISMATCH",
                p,
                f"expected normalized={exp_norm}, scale={exp_scale}; got {rec}",
            )

        rb = rec["normalized"][0] * rec["scale"]
        re = rec["normalized"][1] * rec["scale"]
        if (rb, re) != state:
            return _fail(4, "SCALE_RECONSTRUCTION_MISMATCH", p, f"reconstructed {(rb, re)} != {state}")

    # Optional scaled enumerator audit: with scaling enabled all states <= N should appear.
    scaled = qsa.reachable_up_to(N, allow_scaling=True)
    if not scaled["ok"]:
        return _fail(4, scaled["fail_type"], "reachable_up_to(allow_scaling=True)", "scaled reachability failed")
    states_set, mapping = scaled["value"]
    expected_total = N * N
    if len(states_set) != expected_total:
        return _fail(
            4,
            "SCALED_REACHABILITY_INCOMPLETE",
            "reachable_up_to(allow_scaling=True)",
            f"expected {expected_total} states, got {len(states_set)}",
        )
    for v in mapping.values():
        if not (isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)):
            return _fail(4, "SCALED_MAPPING_SHAPE_INVALID", "reachable_up_to(allow_scaling=True)", "mapping values must be (word, scale)")

    return _pass(4, f"scaling/gcd checks passed ({len(fixtures['scale_samples'])} samples, N={N})")


def _gate5_invariant_diff_claim(cert: Dict[str, Any], computed_failures: List[Dict[str, Any]]) -> GateResult:
    result = cert.get("result")
    if not isinstance(result, dict):
        return _fail(5, "SCHEMA_TYPE_MISMATCH", "result", "result must be object")

    idm = result.get("invariant_diff_map")
    if not isinstance(idm, dict):
        return _fail(5, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map", "must be object")

    entries = idm.get("entries")
    if not isinstance(entries, list):
        return _fail(5, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map.entries", "must be array")

    rollup_claimed = idm.get("rollup_sha256")
    if not isinstance(rollup_claimed, str):
        return _fail(5, "SCHEMA_TYPE_MISMATCH", "result.invariant_diff_map.rollup_sha256", "must be string")

    diffs: List[Diff] = []
    if entries != computed_failures:
        diffs.append(
            Diff(
                5,
                "INVARIANT_DIFF_MAP_CLAIM_MISMATCH",
                "result.invariant_diff_map.entries",
                "claimed entries differ from recomputed gate failures",
            )
        )

    rollup_recomputed = _sha256_hex(_canonical_json(computed_failures))
    if rollup_claimed != rollup_recomputed:
        diffs.append(
            Diff(
                5,
                "INVARIANT_DIFF_MAP_CLAIM_MISMATCH",
                "result.invariant_diff_map.rollup_sha256",
                f"claimed {rollup_claimed} != recomputed {rollup_recomputed}",
            )
        )

    if diffs:
        return GateResult(
            gate_id=5,
            status=GateStatus.FAIL,
            message="INVARIANT_DIFF_MAP_CLAIM_MISMATCH @ result.invariant_diff_map -- claim does not match recomputed failures",
            diffs=diffs,
        )

    return _pass(5, f"invariant_diff_map claim verified (entries={len(entries)})")


def _derive_failure_join_summary(cert: Dict[str, Any], failures: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Optional, non-blocking audit field:
    - emitted only when soft-bridge anchor ref is present and the anchor is loadable
    - computes join across known [76]-carrier fail types
    - preserves unknown fail types explicitly (no PASS/FAIL impact)
    """
    subject = cert.get("subject")
    if not isinstance(subject, dict):
        return None
    if "failure_algebra_anchor_ref" not in subject:
        return None
    if FAILURE_ALGEBRA_ANCHOR_REF is None or failure_join is None:
        return None

    known_types = set(FAILURE_TYPES)
    raw_fail_types = [f.get("fail_type", "") for f in failures if isinstance(f, dict)]
    known_fail_types: List[str] = [t for t in raw_fail_types if t in known_types]
    unknown_fail_types: List[str] = [t for t in raw_fail_types if t not in known_types]

    join_complete = len(unknown_fail_types) == 0
    if not known_fail_types:
        joined_fail_type: Optional[str] = UNIT
    else:
        acc = known_fail_types[0]
        for t in known_fail_types[1:]:
            acc = failure_join(acc, t)
        joined_fail_type = acc

    projection_map_name = PROJECTION_MAP_NAME_STRUCTURAL
    projection_map = FAILURE_PROJECTION_MAP_V1_STRUCTURAL
    projection_selector_reason = "default structural_schema"
    projection_selector_hits: List[str] = []
    if choose_projection_map_with_reason is not None:
        projection_map_name, projection_map, projection_selector_reason, projection_selector_hits = choose_projection_map_with_reason(
            raw_fail_types
        )
    elif choose_projection_map is not None:
        projection_map_name, projection_map = choose_projection_map(raw_fail_types)

    projected_fail_types: List[str] = []
    projection_unknown_fail_types: List[str] = []
    for t in raw_fail_types:
        projected = projection_map.get(t)
        if projected is None:
            projection_unknown_fail_types.append(t)
            continue
        if projected in known_types:
            projected_fail_types.append(projected)
        else:
            projection_unknown_fail_types.append(t)

    projection_complete = len(projection_unknown_fail_types) == 0
    if not projected_fail_types:
        joined_fail_type_projected: Optional[str] = UNIT
    else:
        acc_projected = projected_fail_types[0]
        for t in projected_fail_types[1:]:
            acc_projected = failure_join(acc_projected, t)
        joined_fail_type_projected = acc_projected
    projection_mapped_count = len(projected_fail_types)
    projection_unmapped_count = len(projection_unknown_fail_types)
    projection_total_count = projection_mapped_count + projection_unmapped_count
    failure_signature_v1 = (
        f"{FAILURE_PROJECTION_MAP_VERSION}:{projection_map_name}:{joined_fail_type_projected}:"
        f"{str(projection_complete).lower()}:{projection_mapped_count}/{projection_total_count}"
    )
    failure_signature_v1_sha256 = _sha256_hex(failure_signature_v1)

    return {
        "anchor_ref": FAILURE_ALGEBRA_ANCHOR_REF,
        "anchor_rollup_sha256": FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256,
        "join_complete": join_complete,
        "joined_fail_type": joined_fail_type,
        "known_fail_types": known_fail_types,
        "unknown_fail_types": unknown_fail_types,
        "known_fail_types_count": len(known_fail_types),
        "unknown_fail_types_count": len(unknown_fail_types),
        "projected_fail_types": projected_fail_types,
        "projection_mapped_count": projection_mapped_count,
        "projection_unknown_fail_types": projection_unknown_fail_types,
        "projection_unmapped_count": projection_unmapped_count,
        "projection_coverage": {
            "mapped": projection_mapped_count,
            "total": projection_total_count,
        },
        "joined_fail_type_projected": joined_fail_type_projected,
        "projection_complete": projection_complete,
        "projection_map_name": projection_map_name,
        "projection_selector_reason": projection_selector_reason,
        "projection_selector_hits": projection_selector_hits,
        "projection_map_version": FAILURE_PROJECTION_MAP_VERSION,
        "failure_signature_v1": failure_signature_v1,
        "failure_signature_v1_sha256": failure_signature_v1_sha256,
    }


def validate_cert(cert: Dict[str, Any]) -> Dict[str, Any]:
    gate_log: List[GateResult] = []
    failures: List[Dict[str, Any]] = []

    g1 = _gate1_schema(cert)
    gate_log.append(g1)
    if g1.status == GateStatus.FAIL:
        failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g1.diffs)
    else:
        g2 = _gate2_samples(cert)
        gate_log.append(g2)
        if g2.status == GateStatus.FAIL:
            failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g2.diffs)
        else:
            g3 = _gate3_uniqueness(cert)
            gate_log.append(g3)
            if g3.status == GateStatus.FAIL:
                failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g3.diffs)
            else:
                g4 = _gate4_scaling(cert)
                gate_log.append(g4)
                if g4.status == GateStatus.FAIL:
                    failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g4.diffs)

    # Gate 5: verify fixture-provided invariant_diff_map claim against recomputed truth.
    g5 = _gate5_invariant_diff_claim(cert, failures)
    gate_log.append(g5)
    if g5.status == GateStatus.FAIL:
        failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g5.diffs)

    # Build validator output invariant diff map from recomputed failures.
    rollup = _sha256_hex(_canonical_json(failures))

    ok = len(failures) == 0
    out = {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "gates": [{"gate_id": g.gate_id, "status": g.status.value, "message": g.message} for g in gate_log],
        "failures": failures,
        "invariant_diff_map": {
            "entries": failures,
            "rollup_sha256": rollup,
        },
    }
    summary = _derive_failure_join_summary(cert, failures)
    if summary is not None:
        out["failure_join_summary"] = summary
    return out


def _run_partial_projection_probe(base_dir: str) -> bool:
    """Deterministic non-gating probe for partial projection coverage."""
    if FAILURE_ALGEBRA_ANCHOR_REF is None or FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256 is None:
        print("[PASS] projection coverage probe: skipped (failure algebra anchor unavailable)")
        return True

    fixture_path = os.path.join(base_dir, "fixtures", "invalid_bridge_hash_mismatch.json")
    if not os.path.isfile(fixture_path):
        print("[PASS] projection coverage probe: skipped (fixture missing)")
        return True

    cert = _load_json(fixture_path)
    subject = cert.setdefault("subject", {})
    subject["failure_algebra_anchor_ref"] = FAILURE_ALGEBRA_ANCHOR_REF
    subject["failure_algebra_anchor_rollup_sha256"] = FAILURE_ALGEBRA_ANCHOR_ROLLUP_SHA256

    result = validate_cert(cert)
    failures_probe = list(result.get("failures", []))
    failures_probe.append(
        {
            "gate": 0,
            "fail_type": "NOT_UNIQUE",
            "path": "probe.synthetic",
            "reason": "synthetic unmapped fail tag for partial projection coverage probe",
        }
    )
    summary = _derive_failure_join_summary(cert, failures_probe)
    if not isinstance(summary, dict):
        print("[FAIL] projection coverage probe: missing summary")
        return False

    mapped = int(summary.get("projection_mapped_count", -1))
    unmapped = int(summary.get("projection_unmapped_count", -1))
    coverage = summary.get("projection_coverage")
    complete = summary.get("projection_complete")
    expected_coverage = {"mapped": 1, "total": 2}
    if mapped == 1 and unmapped == 1 and coverage == expected_coverage and complete is False:
        print("[PASS] projection coverage probe: partial coverage detected (mapped=1,total=2,complete=false)")
        return True

    print("[FAIL] projection coverage probe: unexpected summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return False


def run_self_test(base_dir: str) -> int:
    fixtures = [
        ("valid_min.json", True),
        ("invalid_bad_expected_word.json", False),
        ("invalid_bridge_hash_mismatch.json", False),
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
        ok_all = _run_partial_projection_probe(base_dir) and ok_all

    if ok_all:
        print("[PASS] all self-tests")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA_STRUCTURAL_ALGEBRA_CERT.v1 certificates")
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
