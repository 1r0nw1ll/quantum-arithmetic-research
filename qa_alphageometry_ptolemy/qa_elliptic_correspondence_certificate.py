#!/usr/bin/env python3
"""
qa_elliptic_correspondence_certificate.py

Certificate dataclasses and deterministic demo builders for
QA_ELLIPTIC_CORRESPONDENCE_CERT.v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List


GENERATOR_SET = [
    "g_plus_r0",
    "g_plus_r1",
    "g_plus_r2",
    "g_minus_r0",
    "g_minus_r1",
    "g_minus_r2",
]

FAILURE_MODES = [
    "NONFINITE_INPUT",
    "SQRT_BRANCH_UNDEFINED",
    "SQRT_CUT_CROSS_DISALLOWED",
    "CUBIC_SOLVE_FAILED",
    "RAMIFICATION_HIT",
    "MULTIROOT_DEGENERATE",
    "CUTSTATE_UPDATE_FAILED",
    "MONODROMY_EVENT_DETECTED",
    "ESCAPE",
    "MAX_ITER_REACHED",
]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def trace_digest(trace: List[Dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(trace).encode("utf-8")).hexdigest()


@dataclass
class EllipticCorrespondenceCertificate:
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return self.payload

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, sort_keys=True)


def _success_transition_trace() -> List[Dict[str, Any]]:
    """Genuinely computed via the curve dynamics (2026-07-06 fix).

    Each step applies v_in = sqrt_branch(u_in^3+u_in, sheet_in), then the
    driver v_out = v_in^2+v_in, then solves u_out^3+u_out = v_out^2 for
    u_out (cubic root selected by branch_index under lex_re_im_abs_arg
    ordering). Values are inherently irrational (square/cube roots of a
    non-degenerate rational), so they are recorded to 15 significant
    digits rather than as exact fractions -- the previous exact-fraction
    values in this trace were fabricated placeholders that did not
    satisfy the curve equation at all (residual ~1.56 on a claimed exact
    0; see docs/families/27_elliptic_correspondence.md Verification Note).
    curve_residual_abs values below are the genuine |v_out^2 -
    (u_out^3+u_out)| from float64 arithmetic, not a fabricated exact "0".
    """
    return [
        {
            "step_index": 1,
            "generator": "g_plus_r0",
            "u_in_re": "0.5",
            "u_in_im": "0",
            "sheet_in": "plus",
            "branch_index_in": 0,
            "u_out_re": "-0.500479251467299",
            "u_out_im": "-1.32341922437594",
            "sheet_out": "plus",
            "branch_index_out": 0,
            "curve_residual_abs": "5.18e-15",
            "status": "ok",
            "fail_type": "",
            "monodromy_event": False,
            "cut_crossed": False,
        },
        {
            "step_index": 2,
            "generator": "g_plus_r0",
            "u_in_re": "-0.500479251467299",
            "u_in_im": "-1.32341922437594",
            "sheet_in": "plus",
            "branch_index_in": 0,
            "u_out_re": "-1.06152423212589",
            "u_out_im": "-2.09296466433895",
            "sheet_out": "plus",
            "branch_index_out": 0,
            "curve_residual_abs": "1.65e-14",
            "status": "ok",
            "fail_type": "",
            "monodromy_event": False,
            "cut_crossed": False,
        },
    ]


def build_demo_success_certificate() -> EllipticCorrespondenceCertificate:
    trace = _success_transition_trace()
    payload = {
        "certificate_id": "elliptic_correspondence_demo_001",
        "certificate_type": "ELLIPTIC_CORRESPONDENCE_CERT",
        "timestamp": "2026-02-10T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_ELLIPTIC_CORRESPONDENCE_CERT.v1",
        "success": True,
        "generator_set": GENERATOR_SET,
        "state_descriptor": {
            "state_space": "S=(u,sheet,branch_index,cut_state)",
            "sqrt_branch": "principal",
            "cubic_root_order": "lex_re_im_abs_arg",
            "radius_bound_u": "4",
            "radius_bound_v": "4",
            "cut_state": {
                "sqrt_cut": "negative_real_axis",
                "cubic_label": "principal"
            }
        },
        "topology_witness": {
            "branching_factor_declared": 6,
            "steps_executed": 2,
            "ramification_hit_count": 0,
            "monodromy_event_count": 0,
            "cut_cross_count": 0,
            "escape_observed": False,
            "max_norm_u": "2.34677113957921",
            "max_norm_v": "3.41940618384472"
        },
        "invariants": {
            "curve_constraint": True,
            "determinism": True,
            "cut_consistency": True,
            "trace_complete": True
        },
        "recompute_inputs": {
            "trace_schema": "QA_ELLIPTIC_CORRESPONDENCE_TRACE.v1",
            "trace_digest": trace_digest(trace),
            "initial_state": {
                "u_re": "0.5",
                "u_im": "0",
                "sheet": "plus",
                "branch_index": 0
            },
            "transition_trace": trace
        },
        "qa_interpretation": {
            "success_type": "ELLIPTIC_CORRESPONDENCE_CERTIFIED",
            "note": "Deterministic branch replay holds under fixed cut conventions"
        }
    }
    return EllipticCorrespondenceCertificate(payload=payload)


def build_demo_failure_certificate() -> EllipticCorrespondenceCertificate:
    payload = {
        "certificate_id": "elliptic_correspondence_ramification_failure_001",
        "certificate_type": "ELLIPTIC_CORRESPONDENCE_CERT",
        "timestamp": "2026-02-10T00:00:00Z",
        "version": "1.0.0",
        "schema": "QA_ELLIPTIC_CORRESPONDENCE_CERT.v1",
        "success": False,
        "generator_set": GENERATOR_SET,
        "failure_mode": "RAMIFICATION_HIT",
        "failure_witness": {
            "reason": "|P'(u)| dropped below epsilon near critical locus",
            "critical_points": ["0+0.577350269i", "0-0.577350269i"],
            "u_value": {
                "re": "0",
                "im": "433/750"
            },
            "p_prime_abs": "1/5000",
            "epsilon": "1/1000",
            "first_bad_step": 3,
            "recommended_action": [
                "Increase ramification exclusion radius",
                "Switch generator from g_minus_r2 to g_plus_r1 before step 3"
            ]
        },
        "qa_interpretation": {
            "failure_class": "RAMIFICATION_HIT",
            "obstruction_type": "branch_degeneracy",
            "note": "The correspondence hit cubic branch collision neighborhood"
        }
    }
    return EllipticCorrespondenceCertificate(payload=payload)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Emit demo elliptic correspondence certificates")
    parser.add_argument("--success", action="store_true", help="Print success demo certificate")
    parser.add_argument("--failure", action="store_true", help="Print failure demo certificate")
    parser.add_argument("--json", action="store_true", help="Alias for printing JSON output")
    args = parser.parse_args()

    if not args.success and not args.failure:
        args.success = True
        args.failure = True

    payload: Dict[str, Any] = {
        "generated_utc": _utc_now_iso(),
    }
    if args.success:
        payload["success_certificate"] = build_demo_success_certificate().to_dict()
    if args.failure:
        payload["failure_certificate"] = build_demo_failure_certificate().to_dict()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
