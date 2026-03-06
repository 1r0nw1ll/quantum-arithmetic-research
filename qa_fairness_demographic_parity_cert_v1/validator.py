#!/usr/bin/env python3
"""
validator.py

QA_FAIRNESS_DEMOGRAPHIC_PARITY_CERT.v1 validator (Machine tract).

Claim:
  demographic parity gap <= dp_gap_max

Gates:
  Gate 1 — Schema validity (jsonschema)
  Gate 2 — Count integrity (nonnegative; n_pos_pred <= n_total; unique groups)
  Gate 3 — Recompute integrity (exact rates + dp_gap match claims)
  Gate 4 — Threshold decision (dp_gap_observed <= dp_gap_max)
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

from qa_cert_validator_base import (
    GateResult,
    GateStatus,
    build_cli,
    report_ok,
    validate_schema,
)


SCHEMA_ID = "QA_FAIRNESS_DEMOGRAPHIC_PARITY_CERT.v1"


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


def _compute_rates_and_gap(groups: List[Dict[str, Any]]) -> Tuple[Dict[str, Fraction], Fraction]:
    rates: Dict[str, Fraction] = {}
    for g in groups:
        label = g["group"]
        n_total = g["n_total"]
        n_pos = g["n_pos_pred"]
        rates[label] = Fraction(n_pos, n_total)
    gap = max(rates.values()) - min(rates.values())
    return rates, gap


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
    bad = []
    for i, g in enumerate(groups):
        label = g.get("group")
        if label in seen:
            bad.append({"index": i, "group": label, "reason": "duplicate_group"})
        seen.add(label)
        n_total = g.get("n_total")
        n_pos = g.get("n_pos_pred")
        if not isinstance(n_total, int) or not isinstance(n_pos, int):
            bad.append({"index": i, "group": label, "reason": "non_integer_counts"})
            continue
        if n_total <= 0:
            bad.append({"index": i, "group": label, "reason": "n_total_must_be_positive"})
        if n_pos < 0:
            bad.append({"index": i, "group": label, "reason": "n_pos_pred_must_be_nonnegative"})
        if n_pos > n_total:
            bad.append({"index": i, "group": label, "reason": "n_pos_pred_exceeds_n_total"})

    if bad:
        results.append(_fail("gate_2_count_integrity", "group counts invalid", {"issues": bad}))
        return results
    results.append(_pass("gate_2_count_integrity", "group counts consistent", {"group_count": len(groups)}))

    # Gate 3 — Recompute integrity
    try:
        computed_rates, computed_gap = _compute_rates_and_gap(groups)
        claimed_rates = cert["claims"]["selection_rates"]
        mismatches = []
        for label, rate in computed_rates.items():
            if label not in claimed_rates:
                mismatches.append({"group": label, "reason": "missing_claimed_rate"})
                continue
            cr = _as_fraction(claimed_rates[label])
            if cr != rate:
                mismatches.append({"group": label, "claimed": str(claimed_rates[label]), "computed": _frac_str(rate)})

        claimed_gap = _as_fraction(cert["claims"]["dp_gap_observed"])
        if claimed_gap != computed_gap:
            mismatches.append({"field": "dp_gap_observed", "claimed": str(cert["claims"]["dp_gap_observed"]), "computed": _frac_str(computed_gap)})

        if mismatches:
            results.append(_fail(
                "gate_3_recompute_integrity",
                "claimed rates/gap do not match recompute",
                {"mismatches": mismatches, "computed_rates": {k: _frac_str(v) for k, v in computed_rates.items()}, "computed_gap": _frac_str(computed_gap)},
            ))
            return results

        results.append(_pass(
            "gate_3_recompute_integrity",
            "claims match exact recompute",
            {"computed_rates": {k: _frac_str(v) for k, v in computed_rates.items()}, "computed_gap": _frac_str(computed_gap)},
        ))
    except Exception as e:
        results.append(_fail("gate_3_recompute_integrity", f"recompute failed: {e}"))
        return results

    # Gate 4 — Threshold decision
    dp_gap_max = _as_fraction(cert["claims"]["dp_gap_max"])
    dp_gap_observed = _as_fraction(cert["claims"]["dp_gap_observed"])
    ok = dp_gap_observed <= dp_gap_max
    if ok:
        results.append(_pass(
            "gate_4_threshold_decision",
            "dp_gap_observed <= dp_gap_max",
            {"dp_gap_observed": _frac_str(dp_gap_observed), "dp_gap_max": _frac_str(dp_gap_max)},
        ))
    else:
        results.append(_fail(
            "gate_4_threshold_decision",
            "dp_gap_observed > dp_gap_max",
            {"dp_gap_observed": _frac_str(dp_gap_observed), "dp_gap_max": _frac_str(dp_gap_max)},
        ))

    # Gate 5 — Failure witness completeness
    if ok:
        results.append(_pass("gate_5_failure_witness", "not required (PASS)"))
        return results

    obs = cert.get("failure_obstructions", [])
    target = None
    for o in obs:
        if isinstance(o, dict) and o.get("obstruction_id") == "dp_gap_exceeds_threshold":
            target = o
            break
    if not target:
        results.append(_fail("gate_5_failure_witness", "missing dp_gap_exceeds_threshold obstruction"))
        return results

    witness = target.get("witness", {})
    try:
        group_min = witness.get("group_min")
        group_max = witness.get("group_max")
        if group_min not in cert["claims"]["selection_rates"] or group_max not in cert["claims"]["selection_rates"]:
            raise ValueError("witness groups must exist in selection_rates")
        rate_min = _as_fraction(witness.get("rate_min"))
        rate_max = _as_fraction(witness.get("rate_max"))
        gap_w = _as_fraction(witness.get("dp_gap_observed"))
        max_w = _as_fraction(witness.get("dp_gap_max"))

        computed_rates, computed_gap = _compute_rates_and_gap(groups)
        if computed_rates[group_min] != rate_min:
            raise ValueError("rate_min mismatch")
        if computed_rates[group_max] != rate_max:
            raise ValueError("rate_max mismatch")
        if computed_gap != gap_w:
            raise ValueError("dp_gap_observed mismatch")
        if dp_gap_max != max_w:
            raise ValueError("dp_gap_max mismatch")

        results.append(_pass("gate_5_failure_witness", "failure witness matches recompute"))
    except Exception as e:
        results.append(_fail("gate_5_failure_witness", f"invalid witness: {e}"))

    return results


def _self_test(as_json: bool) -> int:
    fixtures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    from qa_cert_validator_base import run_self_test

    fixture_specs = [
        ("valid_min.json", True, None),
        ("invalid_gap.json", False, "gate_4_threshold_decision"),
    ]
    return run_self_test(
        fixtures_dir=fixtures_dir,
        fixture_specs=fixture_specs,
        validate_fn=validate,
        label="QA_FAIRNESS_DEMOGRAPHIC_PARITY_CERT.v1",
        as_json=as_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    return build_cli(
        label="QA_FAIRNESS_DEMOGRAPHIC_PARITY_CERT.v1",
        validate_fn=validate,
        self_test_fn=_self_test,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
