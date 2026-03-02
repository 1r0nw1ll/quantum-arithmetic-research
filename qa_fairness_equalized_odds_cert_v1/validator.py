#!/usr/bin/env python3
"""
validator.py

QA_FAIRNESS_EQUALIZED_ODDS_CERT.v1 validator (Machine tract).

Equalized odds gaps (exact arithmetic):
  TPR(g) = TP / (TP + FN)
  FPR(g) = FP / (FP + TN)

Claims:
  tpr_gap_observed = max_g TPR(g) - min_g TPR(g)
  fpr_gap_observed = max_g FPR(g) - min_g FPR(g)
  tpr_gap_observed <= tpr_gap_max
  fpr_gap_observed <= fpr_gap_max

Gates:
  Gate 1 — Schema validity (jsonschema)
  Gate 2 — Count integrity (nonnegative; denominators positive; unique groups)
  Gate 3 — Recompute integrity (exact rates + gaps match claims)
  Gate 4 — Threshold decision (both gaps within max)
  Gate 5 — Failure witness completeness (if FAIL, obstruction matches recompute)

CLI:
  python validator.py <cert.json>
  python validator.py --self-test
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

# Allow running this validator as a script from within its own directory tree.
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


SCHEMA_ID = "QA_FAIRNESS_EQUALIZED_ODDS_CERT.v1"


def _schema_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "schema.json")


def _as_fraction(x: Any) -> Fraction:
    if isinstance(x, bool):
        raise TypeError("bool is not a scalar")
    if isinstance(x, int):
        return Fraction(x, 1)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError("empty scalar string")
        if "." in s:
            raise ValueError("decimal strings not allowed (use int or p/q)")
        return Fraction(s)
    raise TypeError(f"unsupported scalar type: {type(x)}")


def _frac_str(f: Fraction) -> str:
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def _pass(gate: str, message: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.PASS, message=message, details=details)


def _fail(gate: str, message: str, details: Optional[Dict[str, Any]] = None) -> GateResult:
    return GateResult(gate=gate, status=GateStatus.FAIL, message=message, details=details)


def _rates(groups: List[Dict[str, Any]]) -> Tuple[Dict[str, Fraction], Dict[str, Fraction]]:
    tpr: Dict[str, Fraction] = {}
    fpr: Dict[str, Fraction] = {}
    for g in groups:
        label = g["group"]
        tp = g["tp"]
        fp = g["fp"]
        tn = g["tn"]
        fn = g["fn"]
        tpr[label] = Fraction(tp, tp + fn)
        fpr[label] = Fraction(fp, fp + tn)
    return tpr, fpr


def _gap(rate_by_group: Dict[str, Fraction]) -> Tuple[Fraction, str, str]:
    items = list(rate_by_group.items())
    min_g, min_v = min(items, key=lambda kv: kv[1])
    max_g, max_v = max(items, key=lambda kv: kv[1])
    return max_v - min_v, min_g, max_g


def validate(cert: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
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

    groups = cert["data"]["groups"]

    # Gate 2 — Count integrity
    seen = set()
    issues = []
    for i, g in enumerate(groups):
        label = g.get("group")
        if label in seen:
            issues.append({"index": i, "group": label, "reason": "duplicate_group"})
        seen.add(label)
        for field in ["tp", "fp", "tn", "fn"]:
            v = g.get(field)
            if not isinstance(v, int):
                issues.append({"index": i, "group": label, "field": field, "reason": "non_integer"})
            elif v < 0:
                issues.append({"index": i, "group": label, "field": field, "reason": "negative"})

        tp = g.get("tp")
        fn = g.get("fn")
        fp = g.get("fp")
        tn = g.get("tn")
        if isinstance(tp, int) and isinstance(fn, int) and (tp + fn) <= 0:
            issues.append({"index": i, "group": label, "reason": "tp_plus_fn_must_be_positive"})
        if isinstance(fp, int) and isinstance(tn, int) and (fp + tn) <= 0:
            issues.append({"index": i, "group": label, "reason": "fp_plus_tn_must_be_positive"})

    if issues:
        results.append(_fail("gate_2_count_integrity", "group counts invalid", {"issues": issues}))
        return results
    results.append(_pass("gate_2_count_integrity", "group counts consistent", {"group_count": len(groups)}))

    # Gate 3 — Recompute integrity
    try:
        computed_tpr, computed_fpr = _rates(groups)
        computed_tpr_gap, tpr_min_g, tpr_max_g = _gap(computed_tpr)
        computed_fpr_gap, fpr_min_g, fpr_max_g = _gap(computed_fpr)

        mismatches = []
        claimed_tpr = cert["claims"]["tpr_by_group"]
        claimed_fpr = cert["claims"]["fpr_by_group"]

        for label, v in computed_tpr.items():
            if label not in claimed_tpr:
                mismatches.append({"metric": "tpr", "group": label, "reason": "missing_claimed_rate"})
                continue
            if _as_fraction(claimed_tpr[label]) != v:
                mismatches.append({"metric": "tpr", "group": label, "claimed": str(claimed_tpr[label]), "computed": _frac_str(v)})

        for label, v in computed_fpr.items():
            if label not in claimed_fpr:
                mismatches.append({"metric": "fpr", "group": label, "reason": "missing_claimed_rate"})
                continue
            if _as_fraction(claimed_fpr[label]) != v:
                mismatches.append({"metric": "fpr", "group": label, "claimed": str(claimed_fpr[label]), "computed": _frac_str(v)})

        if _as_fraction(cert["claims"]["tpr_gap_observed"]) != computed_tpr_gap:
            mismatches.append({"field": "tpr_gap_observed", "claimed": str(cert["claims"]["tpr_gap_observed"]), "computed": _frac_str(computed_tpr_gap)})
        if _as_fraction(cert["claims"]["fpr_gap_observed"]) != computed_fpr_gap:
            mismatches.append({"field": "fpr_gap_observed", "claimed": str(cert["claims"]["fpr_gap_observed"]), "computed": _frac_str(computed_fpr_gap)})

        if mismatches:
            results.append(_fail(
                "gate_3_recompute_integrity",
                "claimed rates/gaps do not match recompute",
                {
                    "mismatches": mismatches,
                    "computed": {
                        "tpr_by_group": {k: _frac_str(v) for k, v in computed_tpr.items()},
                        "fpr_by_group": {k: _frac_str(v) for k, v in computed_fpr.items()},
                        "tpr_gap_observed": _frac_str(computed_tpr_gap),
                        "fpr_gap_observed": _frac_str(computed_fpr_gap),
                        "argmins_argmaxes": {
                            "tpr_min_group": tpr_min_g,
                            "tpr_max_group": tpr_max_g,
                            "fpr_min_group": fpr_min_g,
                            "fpr_max_group": fpr_max_g,
                        },
                    },
                },
            ))
            return results

        results.append(_pass(
            "gate_3_recompute_integrity",
            "claims match exact recompute",
            {
                "computed": {
                    "tpr_by_group": {k: _frac_str(v) for k, v in computed_tpr.items()},
                    "fpr_by_group": {k: _frac_str(v) for k, v in computed_fpr.items()},
                    "tpr_gap_observed": _frac_str(computed_tpr_gap),
                    "fpr_gap_observed": _frac_str(computed_fpr_gap),
                }
            },
        ))
    except Exception as e:
        results.append(_fail("gate_3_recompute_integrity", f"recompute failed: {e}"))
        return results

    # Gate 4 — Threshold decision
    tpr_gap_max = _as_fraction(cert["claims"]["tpr_gap_max"])
    fpr_gap_max = _as_fraction(cert["claims"]["fpr_gap_max"])
    tpr_gap_observed = _as_fraction(cert["claims"]["tpr_gap_observed"])
    fpr_gap_observed = _as_fraction(cert["claims"]["fpr_gap_observed"])

    ok_tpr = tpr_gap_observed <= tpr_gap_max
    ok_fpr = fpr_gap_observed <= fpr_gap_max
    if ok_tpr and ok_fpr:
        results.append(_pass(
            "gate_4_threshold_decision",
            "tpr_gap_observed <= tpr_gap_max and fpr_gap_observed <= fpr_gap_max",
            {
                "tpr_gap_observed": _frac_str(tpr_gap_observed),
                "tpr_gap_max": _frac_str(tpr_gap_max),
                "fpr_gap_observed": _frac_str(fpr_gap_observed),
                "fpr_gap_max": _frac_str(fpr_gap_max),
            },
        ))
    else:
        results.append(_fail(
            "gate_4_threshold_decision",
            "one or more gaps exceed max thresholds",
            {
                "tpr_ok": ok_tpr,
                "fpr_ok": ok_fpr,
                "tpr_gap_observed": _frac_str(tpr_gap_observed),
                "tpr_gap_max": _frac_str(tpr_gap_max),
                "fpr_gap_observed": _frac_str(fpr_gap_observed),
                "fpr_gap_max": _frac_str(fpr_gap_max),
            },
        ))

    # Gate 5 — Failure witness completeness
    if ok_tpr and ok_fpr:
        results.append(_pass("gate_5_failure_witness", "not required (PASS)"))
        return results

    obs = cert.get("failure_obstructions", [])
    target = None
    for o in obs:
        if isinstance(o, dict) and o.get("obstruction_id") == "equalized_odds_gap_exceeds_threshold":
            target = o
            break
    if not target:
        results.append(_fail("gate_5_failure_witness", "missing equalized_odds_gap_exceeds_threshold obstruction"))
        return results

    witness = target.get("witness", {})
    try:
        computed_tpr, computed_fpr = _rates(groups)
        computed_tpr_gap, tpr_min_g, tpr_max_g = _gap(computed_tpr)
        computed_fpr_gap, fpr_min_g, fpr_max_g = _gap(computed_fpr)

        w_tpr = witness.get("tpr", {})
        w_fpr = witness.get("fpr", {})

        def _check_side(side: Dict[str, Any], *, metric: str, rate_by_group: Dict[str, Fraction], gap: Fraction, gap_max: Fraction, min_g: str, max_g: str) -> None:
            if side.get("group_min") != min_g or side.get("group_max") != max_g:
                raise ValueError(f"{metric} group_min/group_max mismatch")
            if _as_fraction(side.get("rate_min")) != rate_by_group[min_g]:
                raise ValueError(f"{metric} rate_min mismatch")
            if _as_fraction(side.get("rate_max")) != rate_by_group[max_g]:
                raise ValueError(f"{metric} rate_max mismatch")
            if _as_fraction(side.get("gap")) != gap:
                raise ValueError(f"{metric} gap mismatch")
            if _as_fraction(side.get("gap_max")) != gap_max:
                raise ValueError(f"{metric} gap_max mismatch")

        _check_side(
            w_tpr,
            metric="tpr",
            rate_by_group=computed_tpr,
            gap=computed_tpr_gap,
            gap_max=tpr_gap_max,
            min_g=tpr_min_g,
            max_g=tpr_max_g,
        )
        _check_side(
            w_fpr,
            metric="fpr",
            rate_by_group=computed_fpr,
            gap=computed_fpr_gap,
            gap_max=fpr_gap_max,
            min_g=fpr_min_g,
            max_g=fpr_max_g,
        )

        results.append(_pass("gate_5_failure_witness", "failure witness matches recompute"))
    except Exception as e:
        results.append(_fail("gate_5_failure_witness", f"invalid witness: {e}"))

    return results


def _self_test(as_json: bool) -> int:
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    fixture_specs = [
        ("valid_min.json", True, None),
        ("invalid_gap.json", False, "gate_4_threshold_decision"),
    ]
    return run_self_test(
        fixtures_dir=fixtures_dir,
        fixture_specs=fixture_specs,
        validate_fn=validate,
        label="QA_FAIRNESS_EQUALIZED_ODDS_CERT.v1",
        as_json=as_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return build_cli(
        label="QA_FAIRNESS_EQUALIZED_ODDS_CERT.v1",
        validate_fn=validate,
        self_test_fn=_self_test,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

