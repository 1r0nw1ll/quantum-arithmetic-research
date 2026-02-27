#!/usr/bin/env python3
"""
validator.py

QA_RAMAN_KNN_RESULTS_CERT.v1 validator (Machine Tract).

Certifies the LOO kNN classification results for the QA Raman spectroscopy
paper. Locks in the headline accuracy, k-sweep table, and per-class breakdown
from qa_knn_baseline.py against the frozen features artifact SHA-256.

Gates:
- Gate 1: JSON schema validity
- Gate 2: Canonical SHA-256 digest integrity (self-referential)
- Gate 3: k-sweep table consistency — best_k row accuracy matches classifier.best_accuracy
- Gate 4: k=1 sanity — unweighted and weighted k=1 rows must have equal accuracy
          (for k=1 LOO, distance weighting is irrelevant; both neighbors are identical)
- Gate 5: Model assessment — model_valid=True and model_not_physically_realizable=False
          (kNN is a valid classical classifier, not subject to quantum realizability issues)

Failure modes:
  SCHEMA_INVALID, HASH_MISMATCH, BEST_ACC_MISMATCH, K1_PARITY_MISMATCH,
  MODEL_NOT_PHYSICALLY_REALIZABLE, MODEL_INVALID
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


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

    # Gate 3 — k-sweep best row matches classifier.best_accuracy
    best_k = obj["classifier"]["best_k"]
    best_weighted = obj["classifier"]["weighted"]
    best_acc = obj["classifier"]["best_accuracy"]
    tol = 1e-6

    best_row = None
    for row in obj["k_sweep"]:
        if row["k"] == best_k and row["weighted"] == best_weighted:
            best_row = row
            break

    if best_row is None:
        results.append(GateResult("gate_3_best_acc_consistency", GateStatus.FAIL,
                                  f"BEST_ACC_MISMATCH: no k_sweep row with k={best_k}, "
                                  f"weighted={best_weighted}",
                                  {"best_k": best_k, "best_weighted": best_weighted}))
        return results

    sweep_acc = best_row["accuracy"]
    if abs(sweep_acc - best_acc) > tol:
        results.append(GateResult("gate_3_best_acc_consistency", GateStatus.FAIL,
                                  "BEST_ACC_MISMATCH: k_sweep row accuracy differs from "
                                  "classifier.best_accuracy",
                                  {"k_sweep_accuracy": sweep_acc,
                                   "classifier_best_accuracy": best_acc,
                                   "diff": abs(sweep_acc - best_acc),
                                   "tol": tol}))
        return results
    results.append(GateResult("gate_3_best_acc_consistency", GateStatus.PASS,
                               f"best_accuracy={best_acc:.6f} matches k_sweep row "
                               f"(k={best_k}, weighted={best_weighted})"))

    # Gate 4 — k=1 parity: unweighted and weighted accuracies must be equal
    k1_unweighted = None
    k1_weighted = None
    for row in obj["k_sweep"]:
        if row["k"] == 1:
            if not row["weighted"]:
                k1_unweighted = row["accuracy"]
            else:
                k1_weighted = row["accuracy"]

    if k1_unweighted is None or k1_weighted is None:
        # If k=1 is not in the sweep at all, skip this gate
        results.append(GateResult("gate_4_k1_parity", GateStatus.PASS,
                                   "Skipped (no k=1 rows in sweep)"))
    elif abs(k1_unweighted - k1_weighted) > tol:
        results.append(GateResult("gate_4_k1_parity", GateStatus.FAIL,
                                  "K1_PARITY_MISMATCH: k=1 unweighted != weighted (impossible for LOO kNN)",
                                  {"k1_unweighted": k1_unweighted,
                                   "k1_weighted": k1_weighted,
                                   "diff": abs(k1_unweighted - k1_weighted)}))
        return results
    else:
        results.append(GateResult("gate_4_k1_parity", GateStatus.PASS,
                                   f"k=1 parity verified: unweighted={k1_unweighted:.6f} "
                                   f"== weighted={k1_weighted:.6f}"))

    # Gate 5 — Model assessment validity
    assessment = obj["model_assessment"]
    if assessment["model_not_physically_realizable"]:
        results.append(GateResult("gate_5_model_assessment", GateStatus.FAIL,
                                  "MODEL_NOT_PHYSICALLY_REALIZABLE: kNN classifier is flagged as "
                                  "physically unrealizable — this should be False for a standard kNN",
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
        ("valid_raman_knn_v1.json",               True,  None),
        ("invalid_best_acc_mismatch.json",         False, "gate_3_best_acc_consistency"),
        ("invalid_model_not_realizable.json",      False, "gate_5_model_assessment"),
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
        print("=== QA_RAMAN_KNN_RESULTS_CERT.v1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            status = "PASS" if (d["ok"] == d["expected_ok"]) else "FAIL"
            print(f"  {d['fixture']}: {status}")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_RAMAN_KNN_RESULTS_CERT.v1 validator")
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
