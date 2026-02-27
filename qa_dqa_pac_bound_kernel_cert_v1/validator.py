#!/usr/bin/env python3
"""
validator.py

QA_DQA_PAC_BOUND_KERNEL_CERT.v1 validator (Machine Tract) — Family [85].

Single source of truth for the QA_DQA PAC bound kernel.
formula_id: PAC_BAYES_QA_DQA_LOGDELTA_V1

Kernel formula (locked by this cert):
    L     = ln(1/delta)
    slack = sqrt((D_QA + L) / (2*m))
    bound = risk_hat + slack
    bound_clipped = min(1.0, max(0.0, bound))
    bound_percent_unrounded = 100.0 * bound_clipped
    bound_percent = ROUND_HALF_UP(bound_percent_unrounded, decimals=1)

Gates:
  1. JSON schema validity
  2. Canonical SHA-256 + kernel_block_sha256 + schema_sha256 integrity
  3. Kernel definition lock — name/version/formula_id/log_term/rounding/tolerance exact
  4. Per-case recomputation — witness intermediates + expected.bound_percent verified
  5. Cross-case monotonic sanity (log-term + D_QA monotonicity)

Failure modes:
  SCHEMA_INVALID, DIGEST_MISMATCH, KERNEL_DEFINITION_DRIFT,
  CASE_RECOMPUTE_MISMATCH, KERNEL_SANITY_VIOLATION
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Locked kernel constants (must match kernel.definition.computation)
# ---------------------------------------------------------------------------
_KERNEL_NAME        = "QA_DQA_PAC_BOUND_KERNEL"
_KERNEL_VERSION     = "v1"
_FORMULA_ID         = "PAC_BAYES_QA_DQA_LOGDELTA_V1"
_LOG_TERM           = "ln(1/delta)"
_ROUNDING_MODE      = "ROUND_HALF_UP"
_ROUNDING_DECIMALS  = 1
_TOLERANCE_ABS      = 0.1        # percent points
_WITNESS_LOG_TOL    = 1e-12      # absolute on log_1_over_delta
_WITNESS_UNRND_TOL  = 1e-9       # absolute on bound_unrounded_percent


# ---------------------------------------------------------------------------
# Gate scaffolding
# ---------------------------------------------------------------------------

class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class GateResult:
    gate: str
    status: GateStatus
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256_of_obj(obj: Any) -> str:
    return _sha256_hex(_canonical_compact(obj).encode("utf-8"))


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    """SHA-256 over canonical JSON with digests.canonical_sha256 zeroed."""
    copy = json.loads(_canonical_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_compact(copy).encode("utf-8"))


def _compute_schema_sha256() -> str:
    """SHA-256 of the schema.json file bytes."""
    with open(_schema_path(), "rb") as f:
        return _sha256_hex(f.read())


def _round_half_up(x: float, decimals: int = 1) -> float:
    """Round x to `decimals` decimal places using ROUND_HALF_UP (not banker's)."""
    factor = 10 ** decimals
    return math.floor(x * factor + 0.5) / factor


def _kernel_compute(D_QA: float, m: int, delta: float, risk_hat: float):
    """Return (log_1_over_delta, bound_unrounded_percent, bound_percent)."""
    L     = math.log(1.0 / delta)
    slack = math.sqrt((D_QA + L) / (2.0 * m))
    bound = risk_hat + slack
    bound_clipped = min(1.0, max(0.0, bound))
    unrounded = 100.0 * bound_clipped
    rounded   = _round_half_up(unrounded, _ROUNDING_DECIMALS)
    return L, unrounded, rounded


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # ------------------------------------------------------------------
    # Gate 1 — Schema validity
    # ------------------------------------------------------------------
    try:
        import jsonschema
        schema = _load_json(_schema_path())
        jsonschema.validate(instance=obj, schema=schema)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Schema valid; cert_version=QA_DQA_PAC_BOUND_KERNEL_CERT.v1",
        ))
    except Exception as exc:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"SCHEMA_INVALID: {exc}",
        ))
        return results

    # ------------------------------------------------------------------
    # Gate 2 — Digest integrity (canonical + kernel_block + schema)
    # ------------------------------------------------------------------
    digests = obj.get("digests", {})

    # 2a: canonical_sha256
    want_canonical = digests.get("canonical_sha256", "")
    got_canonical  = _compute_canonical_sha256(obj)
    if want_canonical == "0" * 64:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 is placeholder",
            {"got": got_canonical},
        ))
        return results
    if want_canonical != got_canonical:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 mismatch",
            {"want": want_canonical, "got": got_canonical},
        ))
        return results

    # 2b: kernel_block_sha256
    want_kernel_block = digests.get("kernel_block_sha256", "")
    got_kernel_block  = _canonical_sha256_of_obj(obj["kernel"])
    if want_kernel_block != got_kernel_block:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: kernel_block_sha256 mismatch",
            {"want": want_kernel_block, "got": got_kernel_block},
        ))
        return results

    # 2c: schema_sha256
    want_schema = digests.get("schema_sha256", "")
    got_schema  = _compute_schema_sha256()
    if want_schema != got_schema:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: schema_sha256 mismatch",
            {"want": want_schema, "got": got_schema},
        ))
        return results

    results.append(GateResult(
        "gate_2_digest_integrity", GateStatus.PASS,
        "canonical_sha256 + kernel_block_sha256 + schema_sha256 all verified",
    ))

    # ------------------------------------------------------------------
    # Gate 3 — Kernel definition lock
    # ------------------------------------------------------------------
    kern  = obj["kernel"]
    defn  = kern["definition"]
    comp  = defn["computation"]
    errors: List[Dict] = []

    if kern["name"] != _KERNEL_NAME:
        errors.append({"path": "kernel.name",
                        "expected": _KERNEL_NAME, "got": kern["name"]})
    if kern["kernel_version"] != _KERNEL_VERSION:
        errors.append({"path": "kernel.kernel_version",
                        "expected": _KERNEL_VERSION, "got": kern["kernel_version"]})
    if kern["formula_id"] != _FORMULA_ID:
        errors.append({"path": "kernel.formula_id",
                        "expected": _FORMULA_ID, "got": kern["formula_id"]})
    if comp.get("uses_log_term") is not True:
        errors.append({"path": "kernel.definition.computation.uses_log_term",
                        "expected": True, "got": comp.get("uses_log_term")})
    if comp.get("log_term") != _LOG_TERM:
        errors.append({"path": "kernel.definition.computation.log_term",
                        "expected": _LOG_TERM, "got": comp.get("log_term")})
    if comp.get("units") != "percent":
        errors.append({"path": "kernel.definition.computation.units",
                        "expected": "percent", "got": comp.get("units")})
    rnd = comp.get("rounding", {})
    if rnd.get("mode") != _ROUNDING_MODE:
        errors.append({"path": "kernel.definition.computation.rounding.mode",
                        "expected": _ROUNDING_MODE, "got": rnd.get("mode")})
    if rnd.get("decimals") != _ROUNDING_DECIMALS:
        errors.append({"path": "kernel.definition.computation.rounding.decimals",
                        "expected": _ROUNDING_DECIMALS, "got": rnd.get("decimals")})
    tol = comp.get("tolerance", {})
    if abs(float(tol.get("abs", -1)) - _TOLERANCE_ABS) > 1e-12:
        errors.append({"path": "kernel.definition.computation.tolerance.abs",
                        "expected": _TOLERANCE_ABS, "got": tol.get("abs")})

    if errors:
        results.append(GateResult(
            "gate_3_kernel_definition_lock", GateStatus.FAIL,
            "KERNEL_DEFINITION_DRIFT",
            {"invariant_diff": {
                "gate": 3, "fields": errors,
                "case_id": None, "delta": None, "tolerance_abs": None,
            }},
        ))
        return results

    results.append(GateResult(
        "gate_3_kernel_definition_lock", GateStatus.PASS,
        (f"Kernel locked: name={_KERNEL_NAME} v={_KERNEL_VERSION} "
         f"formula={_FORMULA_ID} log_term={_LOG_TERM} "
         f"rounding={_ROUNDING_MODE}/{_ROUNDING_DECIMALS}dp tol={_TOLERANCE_ABS}"),
    ))

    # ------------------------------------------------------------------
    # Gate 4 — Per-case numeric recomputation
    # ------------------------------------------------------------------
    case_results = []
    for case in obj["cases"]:
        cid = case["case_id"]
        inp = case["inputs"]
        exp = case["expected"]
        wit = case["witness"]

        try:
            L_calc, unr_calc, pct_calc = _kernel_compute(
                float(inp["D_QA"]), int(inp["m"]),
                float(inp["delta"]), float(inp["risk_hat"]),
            )
        except Exception as exc:
            results.append(GateResult(
                "gate_4_case_recompute", GateStatus.FAIL,
                f"CASE_RECOMPUTE_MISMATCH: computation error for case '{cid}': {exc}",
                {"case_id": cid},
            ))
            return results

        inter = wit["intermediates"]
        L_wit   = float(inter["log_1_over_delta"])
        unr_wit = float(inter["bound_unrounded_percent"])
        pct_exp = float(exp["bound_percent"])

        field_errors = []
        # Witness log term
        if abs(L_calc - L_wit) > _WITNESS_LOG_TOL:
            field_errors.append({
                "path": f"cases[case_id={cid}].witness.intermediates.log_1_over_delta",
                "expected": L_calc, "got": L_wit,
                "delta": abs(L_calc - L_wit), "tolerance_abs": _WITNESS_LOG_TOL,
                "case_id": cid,
            })
        # Witness unrounded percent
        if abs(unr_calc - unr_wit) > _WITNESS_UNRND_TOL:
            field_errors.append({
                "path": f"cases[case_id={cid}].witness.intermediates.bound_unrounded_percent",
                "expected": unr_calc, "got": unr_wit,
                "delta": abs(unr_calc - unr_wit), "tolerance_abs": _WITNESS_UNRND_TOL,
                "case_id": cid,
            })
        # Expected bound_percent
        if abs(pct_calc - pct_exp) > _TOLERANCE_ABS:
            field_errors.append({
                "path": f"cases[case_id={cid}].expected.bound_percent",
                "expected": pct_calc, "got": pct_exp,
                "delta": abs(pct_calc - pct_exp), "tolerance_abs": _TOLERANCE_ABS,
                "case_id": cid,
            })
        # rounding_applied must be True
        if not wit.get("rounding_applied"):
            field_errors.append({
                "path": f"cases[case_id={cid}].witness.rounding_applied",
                "expected": True, "got": wit.get("rounding_applied"),
                "delta": None, "tolerance_abs": None, "case_id": cid,
            })

        if field_errors:
            results.append(GateResult(
                "gate_4_case_recompute", GateStatus.FAIL,
                f"CASE_RECOMPUTE_MISMATCH: case '{cid}'",
                {"invariant_diff": {
                    "gate": 4,
                    "path": field_errors[0]["path"],
                    "expected": field_errors[0]["expected"],
                    "observed": field_errors[0]["got"],
                    "delta": field_errors[0].get("delta"),
                    "tolerance_abs": field_errors[0].get("tolerance_abs"),
                    "case_id": cid,
                },
                "recomputed": {
                    "log_1_over_delta": L_calc,
                    "bound_unrounded_percent": unr_calc,
                    "bound_percent": pct_calc,
                },
                "all_field_errors": field_errors},
            ))
            return results

        case_results.append({
            "case_id": cid, "L": L_calc, "unr": unr_calc, "pct": pct_calc,
            "D_QA": float(inp["D_QA"]), "m": int(inp["m"]),
            "delta": float(inp["delta"]), "risk_hat": float(inp["risk_hat"]),
        })

    results.append(GateResult(
        "gate_4_case_recompute", GateStatus.PASS,
        f"All {len(case_results)} cases recomputed correctly",
        {"cases": [{"case_id": c["case_id"], "bound_percent": c["pct"]}
                   for c in case_results]},
    ))

    # ------------------------------------------------------------------
    # Gate 5 — Cross-case monotonic sanity
    # ------------------------------------------------------------------
    n = len(case_results)
    sanity_errors = []

    for i in range(n):
        ci = case_results[i]
        for j in range(n):
            if i == j:
                continue
            cj = case_results[j]
            # Log-term monotonicity: same (D_QA, m, risk_hat), smaller delta → larger bound
            if (abs(ci["D_QA"] - cj["D_QA"]) < 1e-9 and
                ci["m"] == cj["m"] and
                abs(ci["risk_hat"] - cj["risk_hat"]) < 1e-9):
                if ci["delta"] < cj["delta"] and ci["unr"] < cj["unr"] - 1e-9:
                    sanity_errors.append({
                        "type": "LOG_TERM_MONOTONICITY",
                        "case_a": ci["case_id"],
                        "case_b": cj["case_id"],
                        "detail": (f"delta({ci['case_id']})={ci['delta']} < delta({cj['case_id']})={cj['delta']} "
                                   f"but bound({ci['case_id']})={ci['unr']:.4f} < bound({cj['case_id']})={cj['unr']:.4f}"),
                    })
            # D_QA monotonicity: same (m, delta, risk_hat), larger D_QA → larger bound
            if (ci["m"] == cj["m"] and
                abs(ci["delta"] - cj["delta"]) < 1e-9 and
                abs(ci["risk_hat"] - cj["risk_hat"]) < 1e-9):
                if ci["D_QA"] > cj["D_QA"] + 1e-9 and ci["unr"] < cj["unr"] - 1e-9:
                    sanity_errors.append({
                        "type": "DQA_MONOTONICITY",
                        "case_a": ci["case_id"],
                        "case_b": cj["case_id"],
                        "detail": (f"D_QA({ci['case_id']})={ci['D_QA']} > D_QA({cj['case_id']})={cj['D_QA']} "
                                   f"but bound({ci['case_id']})={ci['unr']:.4f} < bound({cj['case_id']})={cj['unr']:.4f}"),
                    })

    if sanity_errors:
        results.append(GateResult(
            "gate_5_cross_case_sanity", GateStatus.FAIL,
            f"KERNEL_SANITY_VIOLATION: {len(sanity_errors)} monotonicity violation(s)",
            {"invariant_diff": {
                "gate": 5, "path": "cases",
                "expected": "monotonic in D_QA and ln(1/delta)",
                "observed": f"{len(sanity_errors)} violations",
                "delta": None, "tolerance_abs": None, "case_id": None,
            },
            "violations": sanity_errors},
        ))
        return results

    results.append(GateResult(
        "gate_5_cross_case_sanity", GateStatus.PASS,
        f"Cross-case monotonicity verified across {n} cases",
    ))

    return results


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------

def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json_out(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx   = os.path.join(base, "fixtures")
    fixtures = [
        ("valid_dqa_pac_bound_kernel_v1.json",   True,  None),
        ("invalid_digest_mismatch.json",          False, "gate_2_digest_integrity"),
        ("invalid_wrong_log_term.json",           False, "gate_4_case_recompute"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({
                "fixture": name, "ok": None,
                "expected_ok": should_pass, "failed_gates": [], "note": "MISSING",
            })
            ok = False
            continue
        obj = _load_json(path)
        res = validate_cert(obj)
        passed = _report_ok(res)
        if should_pass != passed:
            ok = False
        fail_gates = [r.gate for r in res if r.status == GateStatus.FAIL]
        if (not should_pass) and expected_fail_gate and expected_fail_gate not in fail_gates:
            ok = False
        details.append({
            "fixture": name, "ok": passed,
            "expected_ok": should_pass, "failed_gates": fail_gates,
        })

    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_DQA_PAC_BOUND_KERNEL_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QA_DQA_PAC_BOUND_KERNEL_CERT.v1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON file to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(as_json=args.json)
    if not args.file:
        ap.print_help()
        return 2
    obj = _load_json(args.file)
    results = validate_cert(obj)
    if args.json:
        _print_json_out(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
