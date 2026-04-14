#!/usr/bin/env python3
"""
validator.py

QA_BENCHMARK_PROTOCOL.v1 validator (Machine Tract).

Gates:
  Gate 1 — Schema Validity
  Gate 2 — Baseline Parity (all parity flags True, baselines non-empty)
  Gate 3 — Calibration Provenance (procedure + domain_of_origin non-empty)
  Gate 4 — Framework Inheritance (mode declared; prior_cert required if not novel)
  Gate 5 — Metrics Non-Empty

Authority: EXPERIMENT_AXIOMS_BLOCK.md (Part B, B1-B4).
Mirrors qa_mapping_protocol/validator.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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
    jsonschema.validate(instance=obj, schema=schema)


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


_DOTTED_REF = re.compile(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$')


def _is_importable_dotted_ref(value: Any) -> bool:
    if not _is_nonempty_str(value) or not _DOTTED_REF.fullmatch(value.strip()):
        return False
    top_level = value.strip().split(".", 1)[0]
    return importlib.util.find_spec(top_level) is not None


def validate_benchmark(obj: Dict[str, Any]) -> List[GateResult]:
    results: List[GateResult] = []

    # Gate 1 — Schema Validity
    try:
        _validate_schema(obj)
        results.append(GateResult("gate_1_schema_validity", GateStatus.PASS,
                                  "Valid QA_BENCHMARK_PROTOCOL.v1 schema"))
    except Exception as e:
        results.append(GateResult("gate_1_schema_validity", GateStatus.FAIL,
                                  f"Schema validation failed: {e}"))
        return results

    # Gate 2 — Baseline Parity
    parity = obj.get("parity_contract", {})
    baselines = obj.get("baselines", [])
    parity_ok = (isinstance(parity, dict)
                 and parity.get("same_seed_all_methods") is True
                 and parity.get("same_data_split") is True
                 and parity.get("same_preprocessing") is True)
    baselines_ok = (
        isinstance(baselines, list)
        and len(baselines) >= 1
        and all(
            isinstance(b, dict)
            and _is_nonempty_str(b.get("name"))
            and _is_importable_dotted_ref(b.get("implementation_ref"))
            for b in baselines
        )
    )
    if parity_ok and baselines_ok:
        results.append(GateResult("gate_2_baseline_parity", GateStatus.PASS,
                                  f"Parity enforced over {len(baselines)} baseline(s)"))
    else:
        results.append(GateResult("gate_2_baseline_parity", GateStatus.FAIL,
                                  "Baseline parity contract incomplete",
                                  {"parity_contract": parity,
                                   "baseline_count": len(baselines) if isinstance(baselines, list) else None}))

    # Gate 3 — Calibration Provenance
    cal = obj.get("calibration_provenance", {})
    proc_ok = isinstance(cal, dict) and _is_nonempty_str(cal.get("procedure"))
    dom_ok = isinstance(cal, dict) and _is_nonempty_str(cal.get("domain_of_origin"))
    learned_ok = isinstance(cal, dict) and _is_nonempty_str(cal.get("learned_on"))
    if proc_ok and dom_ok and learned_ok:
        results.append(GateResult("gate_3_calibration_provenance", GateStatus.PASS,
                                  "Calibration provenance declared"))
    else:
        results.append(GateResult("gate_3_calibration_provenance", GateStatus.FAIL,
                                  "Calibration provenance incomplete — "
                                  "see EXPERIMENT_AXIOMS_BLOCK.md B3 (cmap-tuned-for-finance lesson)",
                                  {"procedure": proc_ok, "domain_of_origin": dom_ok,
                                   "learned_on": learned_ok}))

    # Gate 4 — Framework Inheritance
    fi = obj.get("framework_inheritance", {})
    mode = fi.get("mode") if isinstance(fi, dict) else None
    if mode in ("inherit", "ported"):
        prior_ok = _is_nonempty_str(fi.get("prior_cert"))
        if prior_ok:
            results.append(GateResult("gate_4_framework_inheritance", GateStatus.PASS,
                                      f"Framework {mode} from prior_cert"))
        else:
            results.append(GateResult("gate_4_framework_inheritance", GateStatus.FAIL,
                                      f"mode='{mode}' requires prior_cert"))
    elif mode == "novel":
        results.append(GateResult("gate_4_framework_inheritance", GateStatus.PASS,
                                  "Framework declared novel"))
    else:
        results.append(GateResult("gate_4_framework_inheritance", GateStatus.FAIL,
                                  "framework_inheritance.mode not declared",
                                  {"mode": mode}))

    # Gate 5 — Metrics Non-Empty
    metrics = obj.get("metrics", [])
    if isinstance(metrics, list) and len(metrics) >= 1 and all(_is_nonempty_str(m) for m in metrics):
        results.append(GateResult("gate_5_metrics", GateStatus.PASS,
                                  f"{len(metrics)} metric(s) declared"))
    else:
        results.append(GateResult("gate_5_metrics", GateStatus.FAIL,
                                  "metrics array missing/empty"))

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
    invalid = _load_json(os.path.join(base, "fixtures", "invalid_missing_calibration_provenance.json"))

    vr = validate_benchmark(valid)
    ir = validate_benchmark(invalid)

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
        print("=== QA_BENCHMARK_PROTOCOL.v1 SELF-TEST ===")
        print(f"valid_min.json:  {'PASS' if _report_ok(vr) else 'FAIL'}")
        print(f"invalid_missing_calibration_provenance.json:  "
              f"{'PASS' if _report_ok(ir) else 'FAIL'} (expected FAIL)")
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_BENCHMARK_PROTOCOL.v1 validator")
    ap.add_argument("file", nargs="?", help="Benchmark JSON file to validate")
    ap.add_argument("--self-test", action="store_true", help="Run validator self-test")
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    results = validate_benchmark(obj)
    if args.json:
        _print_json(results)
    else:
        _print_human(results)
        print(f"\nRESULT: {'PASS' if _report_ok(results) else 'FAIL'}")
    return 0 if _report_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
