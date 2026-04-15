#!/usr/bin/env python3
"""
validator.py

QA_EXPERIMENT_PROTOCOL.v1 validator (Machine Tract).

Gates:
  Gate 1 — Schema Validity
  Gate 2 — Null Independence Defined (non-empty independence_argument)
  Gate 3 — Pre-Registration Complete (seed + date_utc + n_trials>=1)
  Gate 4 — Decision Rules Complete (accept + reject + on_unsupportive enum)
  Gate 5 — Observer Projection Declared (description + state_alphabet non-empty)

Authority: EXPERIMENT_AXIOMS_BLOCK.md (Part A, E1-E6; Part C, N1-N3).
Mirrors qa_mapping_protocol/validator.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


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


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(
        instance=obj,
        schema=schema,
        format_checker=jsonschema.FormatChecker(),
    )


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _nonempty_str_list(x: Any) -> bool:
    return isinstance(x, list) and bool(x) and all(_is_nonempty_str(v) for v in x)


def _is_iso_datetime(value: Any) -> bool:
    if not _is_nonempty_str(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


_REAL_DATA_SENTINELS = frozenset({"pending", "synthetic_only"})


def _real_data_status_ok(value: Any, base_dir: Optional[str]) -> bool:
    if not _is_nonempty_str(value):
        return False
    if value in _REAL_DATA_SENTINELS:
        return True
    path = os.path.expanduser(value)
    if not os.path.isabs(path):
        path = os.path.join(base_dir or os.getcwd(), path)
    return os.path.exists(path)


def _resolve_path(value: str, base_dir: Optional[str]) -> str:
    path = os.path.expanduser(value)
    if not os.path.isabs(path):
        path = os.path.join(base_dir or os.getcwd(), path)
    return os.path.abspath(path)


def _source_mapping_ok(value: Any, base_dir: Optional[str]) -> tuple[bool, Dict[str, Any]]:
    if not isinstance(value, dict):
        return False, {"reason": "source_mapping must be an object"}

    theory_doc = value.get("theory_doc")
    primary_source = value.get("primary_source")
    rationale = value.get("mapping_rationale")
    fields_ok = all(_is_nonempty_str(v) for v in (theory_doc, primary_source, rationale))
    if not fields_ok:
        return False, {"reason": "theory_doc, primary_source, and mapping_rationale must be non-empty"}

    theory_path = _resolve_path(theory_doc, base_dir)
    if not os.path.exists(theory_path):
        return False, {"reason": "theory_doc does not exist", "theory_doc": theory_path}

    with open(theory_path, "r", encoding="utf-8", errors="replace") as handle:
        haystack = handle.read()
    if primary_source not in haystack:
        return False, {
            "reason": "primary_source is not present in theory_doc",
            "theory_doc": theory_path,
            "primary_source": primary_source,
        }

    return True, {"theory_doc": theory_path, "primary_source": primary_source}


def _ablation_ok(value: Any) -> tuple[bool, Dict[str, Any]]:
    if not isinstance(value, dict):
        return False, {"reason": "ablation must be an object"}
    callable_ok = _is_nonempty_str(value.get("callable"))
    destroyed_ok = _is_nonempty_str(value.get("destroyed_structure"))
    direction_ok = _is_nonempty_str(value.get("expected_direction"))
    return (
        callable_ok and destroyed_ok and direction_ok,
        {"callable": callable_ok, "destroyed_structure": destroyed_ok, "expected_direction": direction_ok},
    )


def _reproducibility_ok(value: Any) -> tuple[bool, Dict[str, Any]]:
    if not isinstance(value, dict):
        return False, {"reason": "reproducibility must be an object"}
    seed_ok = isinstance(value.get("seed"), int) and value.get("seed") >= 0
    data_sha = value.get("data_sha256")
    data_ok = (
        data_sha in _REAL_DATA_SENTINELS
        or (_is_nonempty_str(data_sha) and bool(re.fullmatch(r"[0-9a-f]{64}", data_sha)))
    )
    packages = value.get("package_versions")
    packages_ok = (
        isinstance(packages, dict)
        and bool(packages)
        and all(_is_nonempty_str(k) and _is_nonempty_str(v) for k, v in packages.items())
    )
    ledger_ok = _is_nonempty_str(value.get("results_ledger"))
    return (
        seed_ok and data_ok and packages_ok and ledger_ok,
        {"seed": seed_ok, "data_sha256": data_ok, "package_versions": packages_ok, "results_ledger": ledger_ok},
    )


def validate_experiment(obj: Dict[str, Any], *, base_dir: Optional[str] = None) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema Validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS,
                                  "Valid QA_EXPERIMENT_PROTOCOL.v1 schema"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL,
                                  f"Schema validation failed: {e}"))
        return results

    # Gate 2 — Null Design Defined
    null = obj.get("null_model", {})
    null_ok = (
        isinstance(null, dict)
        and _is_nonempty_str(null.get("generating_process"))
        and _nonempty_str_list(null.get("held_fixed"))
        and _nonempty_str_list(null.get("permuted"))
        and _is_nonempty_str(null.get("independence_argument"))
    )
    if null_ok:
        results.append(GateResult("gate_2_null_design", GateStatus.PASS,
                                  "Null generating process, held-fixed set, permuted set, and independence argument present"))
    else:
        results.append(GateResult("gate_2_null_design", GateStatus.FAIL,
                                  "Null design incomplete — declare generating_process, held_fixed, permuted, and independence_argument",
                                  {
                                      "generating_process": isinstance(null, dict) and _is_nonempty_str(null.get("generating_process")),
                                      "held_fixed": isinstance(null, dict) and _nonempty_str_list(null.get("held_fixed")),
                                      "permuted": isinstance(null, dict) and _nonempty_str_list(null.get("permuted")),
                                      "independence_argument": isinstance(null, dict) and _is_nonempty_str(null.get("independence_argument")),
                                  }))

    # Gate 3 — Pre-Registration Complete
    pre = obj.get("pre_registration", {})
    seed_ok = isinstance(pre, dict) and isinstance(pre.get("seed"), int) and pre.get("seed") >= 0
    date_ok = isinstance(pre, dict) and _is_iso_datetime(pre.get("date_utc"))
    trials_ok = (isinstance(pre, dict)
                 and isinstance(pre.get("n_trials"), int)
                 and pre.get("n_trials", 0) >= 1)
    if seed_ok and date_ok and trials_ok:
        results.append(GateResult("gate_3_pre_registration", GateStatus.PASS,
                                  "Pre-registration fields complete"))
    else:
        results.append(GateResult("gate_3_pre_registration", GateStatus.FAIL,
                                  "Pre-registration incomplete",
                                  {"seed_ok": seed_ok, "date_ok": date_ok,
                                   "trials_ok": trials_ok}))

    # Gate 4 — Decision Rules Complete
    dr = obj.get("decision_rules", {})
    accept_ok = isinstance(dr, dict) and _is_nonempty_str(dr.get("accept_criterion"))
    reject_ok = isinstance(dr, dict) and _is_nonempty_str(dr.get("reject_criterion"))
    on_null_ok = isinstance(dr, dict) and dr.get("on_unsupportive") in (
        "investigate_observer", "investigate_implementation",
        "pre_registered_accept",
    )
    if accept_ok and reject_ok and on_null_ok:
        results.append(GateResult("gate_4_decision_rules", GateStatus.PASS,
                                  "Decision rules complete"))
    else:
        results.append(GateResult("gate_4_decision_rules", GateStatus.FAIL,
                                  "Decision rules incomplete",
                                  {"accept": accept_ok, "reject": reject_ok,
                                   "on_unsupportive": on_null_ok}))

    # Gate 5 — Observer Projection Declared
    op = obj.get("observer_projection", {})
    desc_ok = isinstance(op, dict) and _is_nonempty_str(op.get("description"))
    alpha_ok = isinstance(op, dict) and _is_nonempty_str(op.get("state_alphabet"))
    if desc_ok and alpha_ok:
        results.append(GateResult("gate_5_observer_projection", GateStatus.PASS,
                                  "Observer projection declared"))
    else:
        results.append(GateResult("gate_5_observer_projection", GateStatus.FAIL,
                                  "Observer projection incomplete",
                                  {"description": desc_ok, "state_alphabet": alpha_ok}))

    # Gate 6 — Real Data Status
    real_data_status = obj.get("real_data_status")
    real_data_ok = _real_data_status_ok(real_data_status, base_dir)
    if real_data_ok:
        results.append(GateResult("gate_6_real_data_status", GateStatus.PASS,
                                  "real_data_status is pending, synthetic_only, or an existing path"))
    else:
        results.append(GateResult("gate_6_real_data_status", GateStatus.FAIL,
                                  "real_data_status must be 'pending', 'synthetic_only', "
                                  "or a path that exists",
                                  {"real_data_status": real_data_status,
                                   "base_dir": base_dir or os.getcwd()}))

    # Gate 7 — Source Mapping Cross-Reference
    source_ok, source_details = _source_mapping_ok(obj.get("source_mapping"), base_dir)
    if source_ok:
        results.append(GateResult("gate_7_source_mapping", GateStatus.PASS,
                                  "source_mapping.primary_source is present in source_mapping.theory_doc",
                                  source_details))
    else:
        results.append(GateResult("gate_7_source_mapping", GateStatus.FAIL,
                                  "source_mapping must point at a theory_doc containing the declared primary_source",
                                  source_details))

    # Gate 8 — Ablation Declared
    ablation_ok, ablation_details = _ablation_ok(obj.get("ablation"))
    if ablation_ok:
        results.append(GateResult("gate_8_ablation", GateStatus.PASS,
                                  "Ablation callable and destroyed QA structure declared"))
    else:
        results.append(GateResult("gate_8_ablation", GateStatus.FAIL,
                                  "ablation must declare callable, destroyed_structure, and expected_direction",
                                  ablation_details))

    # Gate 9 — Reproducibility Manifest Declared
    repro_ok, repro_details = _reproducibility_ok(obj.get("reproducibility"))
    if repro_ok:
        results.append(GateResult("gate_9_reproducibility", GateStatus.PASS,
                                  "Reproducibility seed, data hash status, package versions, and ledger path declared"))
    else:
        results.append(GateResult("gate_9_reproducibility", GateStatus.FAIL,
                                  "reproducibility manifest incomplete",
                                  repro_details))

    return results


def _report_ok(results: List[GateResult]) -> bool:
    return all(r.status == GateStatus.PASS for r in results)


def _print_human(results: List[GateResult]) -> None:
    for r in results:
        print(f"[{r.status.value}] {r.gate}: {r.message}")


def _print_json(results: List[GateResult]) -> None:
    payload = {
        "ok": _report_ok(results),
        "results": [r.to_dict() for r in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    valid = _load_json(os.path.join(base, "fixtures", "valid_min.json"))
    invalid = _load_json(os.path.join(base, "fixtures", "invalid_missing_null_independence.json"))

    vr = validate_experiment(valid, base_dir=base)
    ir = validate_experiment(invalid, base_dir=base)

    ok = _report_ok(vr) and not _report_ok(ir)
    failed_gates = {r.gate for r in ir if r.status == GateStatus.FAIL}
    expected = {"gate_1_schema_validity"}
    ok = ok and expected.issubset(failed_gates)

    if as_json:
        print(json.dumps({
            "ok": ok,
            "valid_ok": _report_ok(vr),
            "invalid_ok": _report_ok(ir),
            "invalid_failed_gates": sorted(failed_gates),
        }, indent=2, sort_keys=True))
    else:
        print("=== QA_EXPERIMENT_PROTOCOL.v1 SELF-TEST ===")
        print(f"valid_min.json:  {'PASS' if _report_ok(vr) else 'FAIL'}")
        print(f"invalid_missing_null_independence.json:  "
              f"{'PASS' if _report_ok(ir) else 'FAIL'} (expected FAIL)")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_EXPERIMENT_PROTOCOL.v1 validator")
    ap.add_argument("file", nargs="?", help="Experiment JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_experiment(obj, base_dir=os.path.dirname(os.path.abspath(args.file)))
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
