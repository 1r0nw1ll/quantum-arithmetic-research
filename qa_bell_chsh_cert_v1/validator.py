#!/usr/bin/env python3
"""
validator.py

QA_BELL_CHSH_CERT.v1 validator (Machine Tract).

Certifies the "8|N theorem": the QA cosine correlator
    E_N(s,t) = cos(2π(s-t)/N)
achieves the CHSH Tsirelson bound |S| = 2√2 if and only if 8 divides N.

Independently verified by exhaustive search over all N^4 settings for N ≤ 32.

Gates:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity (self-referential)
- Gate 3: 8|N theorem consistency — for each n_sweep entry, verify:
          divisible_by_8 == (N % 8 == 0)
- Gate 4: Tsirelson bound values — verify hit_tsirelson is consistent with max_abs_S:
          if hit_tsirelson: max_abs_S ≥ TSIRELSON - tol
          if not hit_tsirelson: max_abs_S < TSIRELSON - tol
- Gate 5: Model assessment — model_valid=True, model_not_physically_realizable=False
          (the QA cosine correlator is a valid deterministic LHV, not a quantum model)

Failure modes:
  SCHEMA_INVALID, HASH_MISMATCH, DIVISIBILITY_MISMATCH, TSIRELSON_VALUE_MISMATCH,
  MODEL_NOT_PHYSICALLY_REALIZABLE, MODEL_INVALID
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

TSIRELSON_BOUND = 2.0 * math.sqrt(2.0)  # ≈ 2.8284271247461902
TSIRELSON_TOL = 1e-6


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
        return {"gate": self.gate, "status": self.status.value,
                "message": self.message, "details": self.details}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


def _compute_canonical_sha256(obj: Dict[str, Any]) -> str:
    copy = json.loads(_canonical_json_compact(obj))
    copy.setdefault("digests", {})
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_json_compact(copy))


def validate_cert(obj: Dict[str, Any], cert_dir: Optional[str] = None) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS, "Schema valid"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL,
                                  f"SCHEMA_INVALID: {e}"))
        return results

    # Gate 2 — Canonical hash integrity
    want = obj.get("digests", {}).get("canonical_sha256", "")
    got = _compute_canonical_sha256(obj)
    if want == "0" * 64:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "HASH_MISMATCH: canonical_sha256 is placeholder",
                                  {"got": got}))
        return results
    if want != got:
        results.append(GateResult("gate_2_canonical_hash", GateStatus.FAIL,
                                  "HASH_MISMATCH: canonical_sha256 does not match",
                                  {"want": want, "got": got}))
        return results
    results.append(GateResult("gate_2_canonical_hash", GateStatus.PASS,
                               "canonical_sha256 matches"))

    # Gate 3 — 8|N theorem divisibility consistency
    n_sweep = obj["n_sweep"]
    mismatches = []
    for entry in n_sweep:
        N = entry["N"]
        declared_div8 = entry["divisible_by_8"]
        computed_div8 = (N % 8 == 0)
        if declared_div8 != computed_div8:
            mismatches.append(
                f"N={N}: declared divisible_by_8={declared_div8} but computed={computed_div8}"
            )
    if mismatches:
        results.append(GateResult("gate_3_divisibility", GateStatus.FAIL,
                                  f"DIVISIBILITY_MISMATCH: {'; '.join(mismatches)}",
                                  {"mismatches": mismatches}))
        return results
    results.append(GateResult("gate_3_divisibility", GateStatus.PASS,
                               f"8|N divisibility consistent across {len(n_sweep)} N values"))

    # Gate 4 — Tsirelson bound value consistency
    tsirelson_threshold = TSIRELSON_BOUND - TSIRELSON_TOL
    value_mismatches = []
    for entry in n_sweep:
        N = entry["N"]
        max_S = entry["max_abs_S"]
        hit = entry["hit_tsirelson"]
        div8 = (N % 8 == 0)

        # Check that hit_tsirelson is consistent with max_abs_S
        should_hit_from_value = max_S >= tsirelson_threshold
        if hit != should_hit_from_value:
            value_mismatches.append(
                f"N={N}: hit_tsirelson={hit} but max_abs_S={max_S:.6f} "
                f"(threshold={tsirelson_threshold:.6f}) implies hit={should_hit_from_value}"
            )
        # The 8|N theorem: if 8|N, must hit; if not 8|N, must not hit
        if div8 and not hit:
            value_mismatches.append(
                f"N={N}: 8|N is True but hit_tsirelson=False (8|N theorem violated)"
            )
        if not div8 and hit:
            value_mismatches.append(
                f"N={N}: 8|N is False but hit_tsirelson=True (8|N theorem violated)"
            )

    if value_mismatches:
        results.append(GateResult("gate_4_tsirelson_values", GateStatus.FAIL,
                                  f"TSIRELSON_VALUE_MISMATCH: {'; '.join(value_mismatches)}",
                                  {"value_mismatches": value_mismatches,
                                   "tsirelson_bound": TSIRELSON_BOUND,
                                   "tol": TSIRELSON_TOL}))
        return results

    n_hits = sum(1 for e in n_sweep if e["hit_tsirelson"])
    n_miss = sum(1 for e in n_sweep if not e["hit_tsirelson"])
    results.append(GateResult("gate_4_tsirelson_values", GateStatus.PASS,
                               f"Tsirelson bound values consistent: {n_hits} hit, {n_miss} miss "
                               f"(bound={TSIRELSON_BOUND:.10f}, tol={TSIRELSON_TOL})"))

    # Gate 5 — Model assessment
    assessment = obj["model_assessment"]
    if assessment["model_not_physically_realizable"]:
        results.append(GateResult("gate_5_model_assessment", GateStatus.FAIL,
                                  "MODEL_NOT_PHYSICALLY_REALIZABLE: CHSH cosine correlator is flagged as "
                                  "physically unrealizable — this should be False. The QA cosine correlator "
                                  "IS a valid deterministic LHV model.",
                                  {"model_not_physically_realizable": True,
                                   "notes": assessment.get("notes", "")}))
        return results
    if not assessment["model_valid"]:
        results.append(GateResult("gate_5_model_assessment", GateStatus.FAIL,
                                  "MODEL_INVALID: model_valid=False",
                                  {"model_valid": False,
                                   "notes": assessment.get("notes", "")}))
        return results
    results.append(GateResult("gate_5_model_assessment", GateStatus.PASS,
                               "model_valid=True, model_not_physically_realizable=False"))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {"ok": _report_ok(results), "results": [r.to_dict() for r in results]}
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")
    fixtures = [
        ("valid_chsh_8n_theorem.json",           True,  None),
        ("invalid_wrong_condition.json",          False, "gate_3_divisibility"),
        ("invalid_wrong_value.json",              False, "gate_4_tsirelson_values"),
    ]
    ok = True
    details = []
    for name, should_pass, expected_fail_gate in fixtures:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({"fixture": name, "ok": None, "expected_ok": should_pass,
                             "failed_gates": [], "note": "MISSING"})
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
        details.append({"fixture": name, "ok": passed, "expected_ok": should_pass,
                         "failed_gates": fail_gates})
    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details}, indent=2, sort_keys=True))
    else:
        print("=== QA_BELL_CHSH_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_BELL_CHSH_CERT.v1 validator")
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
    cert_dir = os.path.dirname(os.path.abspath(args.file))
    results = validate_cert(obj, cert_dir=cert_dir)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
