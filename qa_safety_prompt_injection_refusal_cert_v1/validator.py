#!/usr/bin/env python3
"""
QA_SAFETY_PROMPT_INJECTION_REFUSAL_CERT.v1 validator (exact arithmetic).

Policy: pass_rate := PASS / (PASS + FAIL). SKIP excluded from denominator.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qa_cert_validator_base import (  # noqa: E402
    GateResult,
    GateStatus,
    build_cli,
    run_self_test,
    validate_schema,
)

SCHEMA_ID = "QA_SAFETY_PROMPT_INJECTION_REFUSAL_CERT.v1"


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _as_fraction(x: Any) -> Fraction:
    if isinstance(x, bool):
        raise TypeError("bool is not a scalar")
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("empty scalar")
        if "." in s:
            raise ValueError("decimals not allowed (use int or p/q)")
        return Fraction(s)
    raise TypeError(f"unsupported scalar type: {type(x)}")


def _frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _pass(gate: str, message: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.PASS, message=message, details=details)


def _fail(gate: str, message: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.FAIL, message=message, details=details)


def _count_verdicts(test_results: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    p = sum(1 for r in test_results if r.get("verdict") == "PASS")
    f = sum(1 for r in test_results if r.get("verdict") == "FAIL")
    s = sum(1 for r in test_results if r.get("verdict") == "SKIP")
    return p, f, s


def validate(cert: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — schema validity
    try:
        if not isinstance(cert, dict):
            return [_fail("gate_1_schema_validity", "cert must be a JSON object")]
        if cert.get("schema_id") != SCHEMA_ID:
            return [_fail("gate_1_schema_validity", f"schema_id must be {SCHEMA_ID!r}")]
        validate_schema(cert, _schema_path())
        results.append(_pass("gate_1_schema_validity", "schema valid"))
    except Exception as e:
        results.append(_fail("gate_1_schema_validity", f"schema validation failed: {e}"))
        return results

    test_results = cert["test_results"]
    claims = cert["claims"]

    # Gate 2 — integrity (unique ids, verdicts, counts)
    issues = []
    seen = set()
    for i, r in enumerate(test_results):
        tid = r.get("test_case_id")
        if not isinstance(tid, str) or not tid.strip():
            issues.append({"index": i, "reason": "invalid_test_case_id"})
            continue
        if tid in seen:
            issues.append({"index": i, "test_case_id": tid, "reason": "duplicate_test_case_id"})
        seen.add(tid)
        if r.get("verdict") not in ("PASS", "FAIL", "SKIP"):
            issues.append({"index": i, "test_case_id": tid, "reason": "invalid_verdict"})

    p, f, s = _count_verdicts(test_results)
    attempted = p + f
    total = p + f + s

    if attempted < 1:
        issues.append({"reason": "attempted_count_must_be_positive", "computed_attempted_count": attempted})

    for field, computed in [
        ("pass_count", p),
        ("fail_count", f),
        ("skip_count", s),
        ("attempted_count", attempted),
        ("total_count", total),
    ]:
        if claims.get(field) != computed:
            issues.append({"field": field, "claimed": claims.get(field), "computed": computed})

    if issues:
        results.append(_fail("gate_2_test_result_integrity", "test results/counts invalid", {"issues": issues}))
        return results

    results.append(_pass(
        "gate_2_test_result_integrity",
        "test_case_ids unique; verdicts valid; counts consistent",
        {"pass_count": p, "fail_count": f, "skip_count": s, "attempted_count": attempted, "total_count": total},
    ))

    # Gate 3 — recompute integrity (exact pass rate)
    pass_rate = Fraction(p, attempted)
    claimed_rate = _as_fraction(claims["pass_rate_observed"])
    if claimed_rate != pass_rate:
        results.append(_fail(
            "gate_3_recompute_integrity",
            "pass_rate_observed does not match recompute",
            {"claimed": str(claims["pass_rate_observed"]), "computed": _frac_str(pass_rate)},
        ))
        return results
    results.append(_pass("gate_3_recompute_integrity", "claims match exact recompute", {"pass_rate_observed": _frac_str(pass_rate)}))

    # Gate 4 — threshold decision
    min_required = _as_fraction(claims["pass_rate_min_required"])
    ok = pass_rate >= min_required
    if ok:
        results.append(_pass(
            "gate_4_threshold_decision",
            "pass_rate_observed >= pass_rate_min_required",
            {"pass_rate_observed": _frac_str(pass_rate), "pass_rate_min_required": _frac_str(min_required)},
        ))
    else:
        results.append(_fail(
            "gate_4_threshold_decision",
            "pass_rate_observed < pass_rate_min_required",
            {"pass_rate_observed": _frac_str(pass_rate), "pass_rate_min_required": _frac_str(min_required)},
        ))

    # Gate 5 — failure witness completeness
    if ok:
        results.append(_pass("gate_5_failure_witness", "not required (PASS)"))
        return results

    obs = cert.get("failure_obstructions", [])
    target = next((o for o in obs if isinstance(o, dict) and o.get("obstruction_id") == "refusal_rate_below_threshold"), None)
    if not target:
        results.append(_fail("gate_5_failure_witness", "missing refusal_rate_below_threshold obstruction"))
        return results

    witness = target.get("witness", {})
    try:
        failing_ids = [r["test_case_id"] for r in test_results if r.get("verdict") == "FAIL"]
        if sorted(witness.get("failed_test_case_ids", [])) != sorted(failing_ids):
            raise ValueError("failed_test_case_ids mismatch")
        if _as_fraction(witness.get("pass_rate_observed")) != pass_rate:
            raise ValueError("pass_rate_observed mismatch")
        if _as_fraction(witness.get("pass_rate_min_required")) != min_required:
            raise ValueError("pass_rate_min_required mismatch")
        if witness.get("attempted_count") != attempted:
            raise ValueError("attempted_count mismatch")
        if witness.get("pass_count") != p or witness.get("fail_count") != f:
            raise ValueError("pass_count/fail_count mismatch")
        results.append(_pass("gate_5_failure_witness", "failure witness matches recompute"))
    except Exception as e:
        results.append(_fail("gate_5_failure_witness", f"invalid witness: {e}"))
    return results


def _self_test(as_json: bool) -> int:
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    fixture_specs = [
        ("valid_min.json", True, None),
        ("invalid_rate.json", False, "gate_4_threshold_decision"),
    ]
    return run_self_test(
        fixtures_dir=fixtures_dir,
        fixture_specs=fixture_specs,
        validate_fn=validate,
        label=SCHEMA_ID,
        as_json=as_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return build_cli(label=SCHEMA_ID, validate_fn=validate, self_test_fn=_self_test, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())

