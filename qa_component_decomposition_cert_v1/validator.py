#!/usr/bin/env python3
"""
validator.py

QA_COMPONENT_DECOMPOSITION_CERT.v1 validator (Machine tract).

Gates:
  1) Schema shape / type checks
  2) Deterministic decomposition sample recomputation
  3) Deterministic nu-power characterization sample checks
  4) Bounded theorem sweep up to N
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
from typing import Any, Dict, List, Tuple


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


def _v2(n: int) -> int:
    if n <= 0:
        return 0
    c = 0
    while (n % 2) == 0:
        n //= 2
        c += 1
    return c


def _gate1_schema(cert: Any) -> GateResult:
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "certificate must be object")

    required = ["schema_version", "created_utc", "subject", "claims", "fixtures", "result"]
    for key in required:
        if key not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", key, "required field missing")

    if cert.get("schema_version") != "QA_COMPONENT_DECOMPOSITION_CERT.v1":
        return _fail(
            1,
            "SCHEMA_VERSION_MISMATCH",
            "schema_version",
            f"expected 'QA_COMPONENT_DECOMPOSITION_CERT.v1', got {cert.get('schema_version')!r}",
        )

    subject = cert.get("subject")
    if not isinstance(subject, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject", "subject must be object")
    for key in ("seed_unit", "N", "generators", "algebra_bridge_semantics_sha256"):
        if key not in subject:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", f"subject.{key}", "required")
    if not _is_int2(subject.get("seed_unit")):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject.seed_unit", "must be int[2]")
    if tuple(subject.get("seed_unit")) != (1, 1):
        return _fail(1, "SEED_UNSUPPORTED", "subject.seed_unit", "must be [1,1]")
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
    for key in (
        "component_decomposition",
        "scaled_seed_roundtrip",
        "nu_power2_characterization",
        "bounded_theorem_sweep",
    ):
        if not isinstance(claims.get(key), bool):
            return _fail(1, "SCHEMA_TYPE_MISMATCH", f"claims.{key}", "must be boolean")

    fixtures = cert.get("fixtures")
    if not isinstance(fixtures, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "fixtures", "fixtures must be object")
    for key in ("decomposition_samples", "nu_samples"):
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


def _gate2_decomposition_samples(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["component_decomposition"]:
        return _pass(2, "component decomposition claim disabled")

    fixtures = cert["fixtures"]
    for i, sample in enumerate(fixtures["decomposition_samples"]):
        p = f"fixtures.decomposition_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in (
            "state",
            "expected_gcd",
            "expected_normalized",
            "expected_word",
            "expected_scaled_seed_roundtrip",
        ):
            if key not in sample:
                return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")

        if not _is_int2(sample["state"]) or not _is_int2(sample["expected_normalized"]) or not _is_int2(sample["expected_scaled_seed_roundtrip"]):
            return _fail(2, "SCHEMA_TYPE_MISMATCH", p, "state/expected_normalized/expected_scaled_seed_roundtrip must be int[2]")

        state = (sample["state"][0], sample["state"][1])
        expected_g = sample["expected_gcd"]
        expected_norm = (sample["expected_normalized"][0], sample["expected_normalized"][1])
        expected_word = sample["expected_word"]
        expected_scaled_roundtrip = (sample["expected_scaled_seed_roundtrip"][0], sample["expected_scaled_seed_roundtrip"][1])

        res_g = qsa.component_gcd(state)
        if not res_g["ok"]:
            return _fail(2, res_g["fail_type"], f"{p}.state", "component_gcd failed")
        if res_g["value"] != expected_g:
            return _fail(2, "DECOMP_GCD_MISMATCH", f"{p}.expected_gcd", f"expected {expected_g}, recomputed {res_g['value']}")

        res_norm = qsa.normalize_to_coprime(state)
        if not res_norm["ok"]:
            return _fail(2, res_norm["fail_type"], f"{p}.state", "normalize_to_coprime failed")
        got_norm, got_scale = res_norm["value"]
        if got_norm != expected_norm or got_scale != expected_g:
            return _fail(
                2,
                "DECOMP_NORMALIZED_MISMATCH",
                f"{p}.expected_normalized",
                f"expected normalized={expected_norm}, scale={expected_g}; recomputed normalized={got_norm}, scale={got_scale}",
            )

        res_word = qsa.state_to_word(expected_norm)
        if not res_word["ok"]:
            return _fail(2, res_word["fail_type"], f"{p}.expected_normalized", "state_to_word failed")
        if res_word["value"] != expected_word:
            return _fail(2, "DECOMP_WORD_MISMATCH", f"{p}.expected_word", f"expected {expected_word!r}, recomputed {res_word['value']!r}")

        seed_scaled = (expected_g, expected_g)
        res_scaled = qsa.word_to_state(expected_word, seed=seed_scaled)
        if not res_scaled["ok"]:
            return _fail(2, res_scaled["fail_type"], f"{p}.expected_word", "word_to_state on scaled seed failed")
        if res_scaled["value"] != expected_scaled_roundtrip:
            return _fail(
                2,
                "DECOMP_SCALED_SEED_ROUNDTRIP_MISMATCH",
                f"{p}.expected_scaled_seed_roundtrip",
                f"expected {expected_scaled_roundtrip}, recomputed {res_scaled['value']}",
            )

    return _pass(2, f"decomposition sample checks passed ({len(fixtures['decomposition_samples'])} samples)")


def _gate3_nu_samples(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["nu_power2_characterization"]:
        return _pass(3, "nu characterization claim disabled")

    fixtures = cert["fixtures"]
    for i, sample in enumerate(fixtures["nu_samples"]):
        p = f"fixtures.nu_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for key in (
            "state",
            "expected_v2_g",
            "expected_after_v2",
            "expected_reaches_normalized",
            "expected_next_ok",
            "expected_next_fail_type",
        ):
            if key not in sample:
                return _fail(3, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{key}", "required")

        if not _is_int2(sample["state"]) or not _is_int2(sample["expected_after_v2"]):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", p, "state and expected_after_v2 must be int[2]")

        state = (sample["state"][0], sample["state"][1])
        expected_after = (sample["expected_after_v2"][0], sample["expected_after_v2"][1])
        expected_v2 = sample["expected_v2_g"]

        g = math.gcd(state[0], state[1])
        got_v2 = _v2(g)
        if got_v2 != expected_v2:
            return _fail(3, "NU_V2_MISMATCH", f"{p}.expected_v2_g", f"expected {expected_v2}, recomputed {got_v2}")

        cur = state
        for step in range(got_v2):
            res_nu = qsa.nu(cur)
            if not res_nu["ok"]:
                return _fail(3, "NU_PREMATURE_BLOCK", p, f"nu blocked at step {step} from state {cur}: {res_nu.get('fail_type')}")
            cur = res_nu["value"]

        if cur != expected_after:
            return _fail(3, "NU_AFTER_V2_MISMATCH", f"{p}.expected_after_v2", f"expected {expected_after}, recomputed {cur}")

        res_norm = qsa.normalize_to_coprime(state)
        if not res_norm["ok"]:
            return _fail(3, res_norm["fail_type"], p, "normalize_to_coprime failed")
        normalized, _scale = res_norm["value"]
        reaches = cur == normalized
        if reaches != sample["expected_reaches_normalized"]:
            return _fail(
                3,
                "NU_REACH_FLAG_MISMATCH",
                f"{p}.expected_reaches_normalized",
                f"expected {sample['expected_reaches_normalized']}, recomputed {reaches}",
            )

        next_res = qsa.nu(cur)
        expected_next_ok = sample["expected_next_ok"]
        if expected_next_ok:
            if not next_res["ok"]:
                return _fail(3, "NU_NEXT_EXPECTED_OK", p, f"expected next nu success, got {next_res.get('fail_type')}")
            if "expected_next_value" in sample and sample["expected_next_value"] is not None:
                if not _is_int2(sample["expected_next_value"]):
                    return _fail(3, "SCHEMA_TYPE_MISMATCH", f"{p}.expected_next_value", "must be int[2]")
                expected_next_value = (sample["expected_next_value"][0], sample["expected_next_value"][1])
                if next_res["value"] != expected_next_value:
                    return _fail(
                        3,
                        "NU_NEXT_VALUE_MISMATCH",
                        f"{p}.expected_next_value",
                        f"expected {expected_next_value}, recomputed {next_res['value']}",
                    )
        else:
            if next_res["ok"]:
                return _fail(3, "NU_NEXT_EXPECTED_FAIL", p, f"expected next nu failure, got {next_res['value']}")
            expected_ft = sample["expected_next_fail_type"]
            if expected_ft and next_res["fail_type"] != expected_ft:
                return _fail(
                    3,
                    "NU_NEXT_FAILTYPE_MISMATCH",
                    f"{p}.expected_next_fail_type",
                    f"expected {expected_ft}, recomputed {next_res['fail_type']}",
                )

    return _pass(3, f"nu characterization sample checks passed ({len(fixtures['nu_samples'])} samples)")


def _gate4_bounded_theorem_sweep(cert: Dict[str, Any]) -> GateResult:
    if not cert["claims"]["bounded_theorem_sweep"]:
        return _pass(4, "bounded theorem sweep claim disabled")

    N = cert["subject"]["N"]

    for b in range(1, N + 1):
        for e in range(1, N + 1):
            state = (b, e)
            g = math.gcd(b, e)
            normalized = (b // g, e // g)

            # Decomposition theorem: normalized is coprime and reconstructs state via scaled seed.
            if math.gcd(normalized[0], normalized[1]) != 1:
                return _fail(4, "BOUNDED_NORMALIZED_NOT_COPRIME", "bounded_sweep", f"state {state}, normalized {normalized}")

            sw = qsa.state_to_word(normalized)
            if not sw["ok"]:
                return _fail(4, sw["fail_type"], "bounded_sweep", f"state_to_word failed for normalized {normalized}")

            wr = qsa.word_to_state(sw["value"], seed=(g, g))
            if not wr["ok"]:
                return _fail(4, wr["fail_type"], "bounded_sweep", f"word_to_state failed for state {state}")
            if wr["value"] != state:
                return _fail(4, "BOUNDED_SCALED_ROUNDTRIP_MISMATCH", "bounded_sweep", f"state {state}, recomputed {wr['value']}")

            # nu characterization theorem: exactly v2(g) successful contractions.
            v2g = _v2(g)
            cur = state
            for step in range(v2g):
                res_nu = qsa.nu(cur)
                if not res_nu["ok"]:
                    return _fail(4, "BOUNDED_NU_PREMATURE_BLOCK", "bounded_sweep", f"state {state}, step {step}, at {cur}")
                cur = res_nu["value"]

            odd_part = g // (2 ** v2g)
            reaches_normalized = (cur == normalized)
            if (odd_part == 1) != reaches_normalized:
                return _fail(
                    4,
                    "BOUNDED_POWER2_REACH_MISMATCH",
                    "bounded_sweep",
                    f"state {state}, odd_part {odd_part}, after_v2 {cur}, normalized {normalized}",
                )

            # After v2(g) contractions, next nu must fail (odd factor exposed).
            next_res = qsa.nu(cur)
            if next_res["ok"]:
                return _fail(4, "BOUNDED_NU_EXPECTED_BLOCK", "bounded_sweep", f"state {state}, post-v2 state {cur}")
            if next_res["fail_type"] != "ODD_BLOCK":
                return _fail(
                    4,
                    "BOUNDED_NU_FAILTYPE_MISMATCH",
                    "bounded_sweep",
                    f"state {state}, expected ODD_BLOCK, got {next_res['fail_type']}",
                )

    return _pass(4, f"bounded component theorem sweep passed (N={N}, states={N * N})")


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


def _derive_failure_join_summary(cert: Dict[str, Any], failures: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Optional, non-blocking audit field:
    - emitted only when soft-bridge anchor ref is present and the anchor is loadable
    - computes join across known [76]-carrier fail types
    - reports unknown fail types explicitly and leaves pass/fail unchanged
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
        joined_fail_type: str | None = UNIT
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
        joined_fail_type_projected: str | None = UNIT
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
        g2 = _gate2_decomposition_samples(cert)
        gate_log.append(g2)
        if g2.status == GateStatus.FAIL:
            failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g2.diffs)
        else:
            g3 = _gate3_nu_samples(cert)
            gate_log.append(g3)
            if g3.status == GateStatus.FAIL:
                failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g3.diffs)
            else:
                g4 = _gate4_bounded_theorem_sweep(cert)
                gate_log.append(g4)
                if g4.status == GateStatus.FAIL:
                    failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g4.diffs)

    g5 = _gate5_invariant_diff_claim(cert, failures)
    gate_log.append(g5)
    if g5.status == GateStatus.FAIL:
        failures.extend({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason} for d in g5.diffs)

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
        ("invalid_bad_decomposition_word.json", False),
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
    parser = argparse.ArgumentParser(description="Validate QA_COMPONENT_DECOMPOSITION_CERT.v1 certificates")
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
