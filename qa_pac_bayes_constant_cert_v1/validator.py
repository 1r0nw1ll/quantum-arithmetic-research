#!/usr/bin/env python3
"""
validator.py

QA_PAC_BAYES_CONSTANT_CERT.v1.1 validator (Machine Tract) — Family [84].

Locks the Phase-1 PAC-Bayes constants and bound tables into a
CI-recomputable certificate, and binds to the canonical QA_DQA kernel cert
(Family [85]) via pac_kernel_ref.

Gates:
  1. JSON schema validity (cert_version == QA_PAC_BAYES_CONSTANT_CERT.v1.1)
  2. Canonical SHA-256 digest integrity (self-referential)
  3. K1 recomputation — verify K1 = 2*C^2*N*(M/2)^2 against constants fields
  4. PAC bound recomputation + improvement ratio consistency
  5. Kernel reference binding — pac_kernel_ref.formula_id + kernel_block_sha256 must
     match the locked Family [85] kernel block values
  6. DPI scope declaration — dpi.claim must be "structured_only"

Bound variant QA_DQA formula:
  bound = risk_hat + sqrt((K1 * D_QA + log(1/delta)) / m)
  bound_percent = round(bound * 100, percent_round_dp)

Failure modes:
  SCHEMA_INVALID, DIGEST_MISMATCH, K1_RECOMPUTE_MISMATCH,
  PAC_BOUND_RECOMPUTE_MISMATCH, KERNEL_REF_MISMATCH, DPI_SCOPE_VIOLATION
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
# Locked kernel reference constants (Family [85] binding)
# ---------------------------------------------------------------------------
_KERNEL_FORMULA_ID       = "PAC_BAYES_QA_DQA_LOGDELTA_V1"
_KERNEL_BLOCK_SHA256     = "553b7588ebf8fd1b10bddcc34a03387d85a0e8089ad3319fdaa234aa7f674676"
_KERNEL_CERT_VERSION     = "QA_DQA_PAC_BOUND_KERNEL_CERT.v1"


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


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    """SHA-256 over canonical JSON with digests.canonical_sha256 zeroed."""
    copy = json.loads(_canonical_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_compact(copy))


def _compute_pac_bound_percent(
    risk_hat: float,
    D_QA: float,
    m: int,
    delta: float,
    K1: float,
    bound_variant: str,
    percent_round_dp: int,
) -> float:
    """Recompute PAC bound percent from raw inputs.

    QA_DQA variant (Phase-1 formula):
        bound = risk_hat + sqrt((K1*D_QA + log(1/delta)) / m)
    MCCALISTER variant (standard McAllister):
        bound = risk_hat + sqrt((K1*D_QA + log(m/delta)) / m)
    """
    if bound_variant == "QA_DQA":
        complexity = math.sqrt((K1 * D_QA + math.log(1.0 / delta)) / m)
    elif bound_variant == "MCCALISTER":
        complexity = math.sqrt((K1 * D_QA + math.log(float(m) / delta)) / m)
    elif bound_variant == "CATONI":
        complexity = math.sqrt((K1 * D_QA + math.log(float(m) / delta)) / m)
    else:
        raise ValueError(f"Unknown bound_variant: {bound_variant!r}")
    return round((risk_hat + complexity) * 100.0, percent_round_dp)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_cert(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []
    k1_calc: Optional[float] = None
    bound_calc: Optional[float] = None
    ratio_calc: Optional[float] = None

    # ------------------------------------------------------------------
    # Gate 1 — Schema validity
    # ------------------------------------------------------------------
    try:
        import jsonschema
        schema = _load_json(_schema_path())
        jsonschema.validate(instance=obj, schema=schema)
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.PASS,
            "Schema valid; cert_version=QA_PAC_BAYES_CONSTANT_CERT.v1.1",
        ))
    except Exception as exc:
        results.append(GateResult(
            "gate_1_schema_validity", GateStatus.FAIL,
            f"SCHEMA_INVALID: {exc}",
        ))
        return results

    # ------------------------------------------------------------------
    # Gate 2 — Canonical hash integrity
    # ------------------------------------------------------------------
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got  = _compute_canonical_sha256(obj)
    if want == "0" * 64:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 is placeholder",
            {"got": got},
        ))
        return results
    if want != got:
        results.append(GateResult(
            "gate_2_digest_integrity", GateStatus.FAIL,
            "DIGEST_MISMATCH: canonical_sha256 does not match",
            {"want": want, "got": got},
        ))
        return results
    results.append(GateResult(
        "gate_2_digest_integrity", GateStatus.PASS,
        "canonical_sha256 verified",
    ))

    # ------------------------------------------------------------------
    # Gate 3 — K1 recomputation
    # ------------------------------------------------------------------
    sys_  = obj["system"]
    N, M, C = sys_["N"], sys_["M"], float(sys_["C"])
    M_is_even_declared = sys_["M_is_even"]

    errors: List[Dict] = []

    if M_is_even_declared != (M % 2 == 0):
        errors.append({
            "path": "system.M_is_even",
            "expected": (M % 2 == 0),
            "got": M_is_even_declared,
        })
    if M % 2 != 0:
        errors.append({
            "path": "system.M",
            "expected": "even integer (M/2 must be an integer)",
            "got": M,
        })

    if not errors:
        k1_calc = 2.0 * (C ** 2) * N * ((M / 2.0) ** 2)
        consts  = obj["constants"]
        tol     = float(consts["tolerance_abs"])
        formula = consts["K1_formula"]

        if formula != "2*C^2*N*(M/2)^2":
            errors.append({
                "path": "constants.K1_formula",
                "expected": "2*C^2*N*(M/2)^2",
                "got": formula,
            })
        if abs(k1_calc - float(consts["K1_expected"])) > tol:
            errors.append({
                "path": "constants.K1_expected",
                "expected": k1_calc,
                "got": consts["K1_expected"],
            })
        if abs(k1_calc - float(consts["K1_recomputed"])) > tol:
            errors.append({
                "path": "constants.K1_recomputed",
                "expected": k1_calc,
                "got": consts["K1_recomputed"],
            })

    if errors:
        results.append(GateResult(
            "gate_3_k1_recompute", GateStatus.FAIL,
            "K1_RECOMPUTE_MISMATCH",
            {"invariant_diff": {
                "gate": 3,
                "fields": errors,
                "recomputed": {
                    "K1_calc": k1_calc,
                    "bound_percent_calc": None,
                    "improvement_ratio_calc": None,
                },
            }},
        ))
        return results

    results.append(GateResult(
        "gate_3_k1_recompute", GateStatus.PASS,
        f"K1={k1_calc:.1f} verified (N={N}, M={M}, C={C}, formula=2*C^2*N*(M/2)^2)",
    ))

    # ------------------------------------------------------------------
    # Gate 4 — PAC bound recomputation + improvement ratio
    # ------------------------------------------------------------------
    pac_in  = obj["pac_inputs"]
    pac_out = obj["pac_outputs"]
    rp      = pac_out["rounding_policy"]
    pdp     = rp["percent_round_dp"]
    rdp     = rp["ratio_round_dp"]

    try:
        bound_calc = _compute_pac_bound_percent(
            risk_hat=float(pac_in["risk_hat"]),
            D_QA=float(pac_in["D_QA"]),
            m=int(pac_in["m"]),
            delta=float(pac_in["delta"]),
            K1=k1_calc,
            bound_variant=pac_in["bound_variant"],
            percent_round_dp=pdp,
        )
    except Exception as exc:
        results.append(GateResult(
            "gate_4_pac_bound_recompute", GateStatus.FAIL,
            f"PAC_BOUND_RECOMPUTE_MISMATCH: bound computation error: {exc}",
        ))
        return results

    initial_pct = float(pac_out["initial_bound_percent"])
    tight_pct   = float(pac_out["tight_bound_percent"])
    ratio_calc  = round(initial_pct / tight_pct, rdp)

    errors = []
    if bound_calc != float(pac_out["bound_percent"]):
        errors.append({
            "path": "pac_outputs.bound_percent",
            "expected": bound_calc,
            "got": pac_out["bound_percent"],
        })
    if bound_calc != float(pac_out["bound_percent_recomputed"]):
        errors.append({
            "path": "pac_outputs.bound_percent_recomputed",
            "expected": bound_calc,
            "got": pac_out["bound_percent_recomputed"],
        })
    if tight_pct >= initial_pct:
        errors.append({
            "path": "pac_outputs.tight_bound_percent",
            "expected": f"< initial_bound_percent ({initial_pct})",
            "got": tight_pct,
        })
    if ratio_calc != float(pac_out["improvement_ratio"]):
        errors.append({
            "path": "pac_outputs.improvement_ratio",
            "expected": ratio_calc,
            "got": pac_out["improvement_ratio"],
        })
    if ratio_calc != float(pac_out["improvement_ratio_recomputed"]):
        errors.append({
            "path": "pac_outputs.improvement_ratio_recomputed",
            "expected": ratio_calc,
            "got": pac_out["improvement_ratio_recomputed"],
        })

    if errors:
        results.append(GateResult(
            "gate_4_pac_bound_recompute", GateStatus.FAIL,
            "PAC_BOUND_RECOMPUTE_MISMATCH",
            {"invariant_diff": {
                "gate": 4,
                "fields": errors,
                "recomputed": {
                    "K1_calc": k1_calc,
                    "bound_percent_calc": bound_calc,
                    "improvement_ratio_calc": ratio_calc,
                },
            }},
        ))
        return results

    results.append(GateResult(
        "gate_4_pac_bound_recompute", GateStatus.PASS,
        (f"bound_percent={bound_calc} ({pac_in['bound_variant']}), "
         f"improvement_ratio={ratio_calc} "
         f"({initial_pct:.0f}%→{tight_pct:.0f}%)"),
    ))

    # ------------------------------------------------------------------
    # Gate 5 — Kernel reference binding ([84]→[85] link)
    # ------------------------------------------------------------------
    ref = obj["pac_kernel_ref"]
    ref_errors = []

    if ref.get("formula_id") != _KERNEL_FORMULA_ID:
        ref_errors.append({
            "path": "pac_kernel_ref.formula_id",
            "expected": _KERNEL_FORMULA_ID,
            "got": ref.get("formula_id"),
        })
    if ref.get("kernel_block_sha256") != _KERNEL_BLOCK_SHA256:
        ref_errors.append({
            "path": "pac_kernel_ref.kernel_block_sha256",
            "expected": _KERNEL_BLOCK_SHA256,
            "got": ref.get("kernel_block_sha256"),
        })
    if ref.get("cert_version") != _KERNEL_CERT_VERSION:
        ref_errors.append({
            "path": "pac_kernel_ref.cert_version",
            "expected": _KERNEL_CERT_VERSION,
            "got": ref.get("cert_version"),
        })

    if ref_errors:
        results.append(GateResult(
            "gate_5_kernel_ref_binding", GateStatus.FAIL,
            "KERNEL_REF_MISMATCH: pac_kernel_ref does not match locked Family [85] values",
            {"invariant_diff": {
                "gate": 5,
                "fields": ref_errors,
                "expected_formula_id": _KERNEL_FORMULA_ID,
                "expected_kernel_block_sha256": _KERNEL_BLOCK_SHA256,
                "expected_cert_version": _KERNEL_CERT_VERSION,
            }},
        ))
        return results

    results.append(GateResult(
        "gate_5_kernel_ref_binding", GateStatus.PASS,
        (f"pac_kernel_ref bound to Family [85]: "
         f"formula_id={_KERNEL_FORMULA_ID} "
         f"kernel_block_sha256={_KERNEL_BLOCK_SHA256[:16]}..."),
    ))

    # ------------------------------------------------------------------
    # Gate 6 — DPI scope declaration
    # ------------------------------------------------------------------
    dpi   = obj["dpi"]
    claim = dpi["claim"]
    if claim != "structured_only":
        results.append(GateResult(
            "gate_6_dpi_scope", GateStatus.FAIL,
            f"DPI_SCOPE_VIOLATION: dpi.claim must be 'structured_only', got '{claim}'",
            {"invariant_diff": {
                "gate": 6,
                "fields": [{"path": "dpi.claim", "expected": "structured_only", "got": claim}],
                "recomputed": {
                    "K1_calc": k1_calc,
                    "bound_percent_calc": bound_calc,
                    "improvement_ratio_calc": ratio_calc,
                },
            }},
        ))
        return results

    vr = float(dpi["random_trials"]["violation_rate"])
    results.append(GateResult(
        "gate_6_dpi_scope", GateStatus.PASS,
        (f"dpi.claim='structured_only'; violation_rate={vr:.3f} "
         f"({int(round(vr*100))}% of {dpi['random_trials']['n_trials']} random trials; "
         f"seed={dpi['random_trials']['seed_used_for_structured_demo']})"),
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
        ("valid_pac_bayes_constant_v1_1.json",    True,  None),
        ("invalid_k1_mismatch.json",               False, "gate_3_k1_recompute"),
        ("invalid_kernel_ref_mismatch.json",       False, "gate_5_kernel_ref_binding"),
        ("invalid_dpi_claim_universal.json",       False, "gate_6_dpi_scope"),
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
        print("=== QA_PAC_BAYES_CONSTANT_CERT.v1.1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="QA_PAC_BAYES_CONSTANT_CERT.v1.1 validator")
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
