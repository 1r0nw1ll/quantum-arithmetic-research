from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qa_dashboard.ledger import ledger


REPORT_SCHEMA_ID = "QA_DASHBOARD_AUDIT_REPORT.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\\-]+$")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _runs_root() -> Path:
    root = os.environ.get("QA_DASHBOARD_RUNS_DIR", "/tmp/qa_dashboard_runs")
    return Path(root)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise ValueError("decimal strings not allowed")
        return Fraction(s)
    raise TypeError(f"unsupported scalar type: {type(x)}")


def _frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _summarize_safety_prompt_injection(input_obj: Dict[str, Any], *, max_ids: int = 10) -> Dict[str, Any]:
    obs = input_obj.get("failure_obstructions", [])
    target = None
    for o in obs:
        if isinstance(o, dict) and o.get("obstruction_id") == "refusal_rate_below_threshold":
            target = o
            break
    if not isinstance(target, dict):
        return {"kind": "safety_prompt_injection_refusal", "summary": "No failure obstruction found."}
    witness = target.get("witness", {}) if isinstance(target.get("witness"), dict) else {}
    failed_ids = witness.get("failed_test_case_ids", [])
    if not isinstance(failed_ids, list):
        failed_ids = []
    failed_ids_s = [str(x) for x in failed_ids[:max_ids]]
    return {
        "kind": "safety_prompt_injection_refusal",
        "summary": "Pass rate below threshold.",
        "failed_test_case_ids_head": failed_ids_s,
        "failed_test_case_ids_total": len(failed_ids) if isinstance(failed_ids, list) else None,
        "pass_rate_observed": witness.get("pass_rate_observed"),
        "pass_rate_min_required": witness.get("pass_rate_min_required"),
        "attempted_count": witness.get("attempted_count"),
        "pass_count": witness.get("pass_count"),
        "fail_count": witness.get("fail_count"),
    }


def _summarize_fairness_demographic_parity(input_obj: Dict[str, Any]) -> Dict[str, Any]:
    obs = input_obj.get("failure_obstructions", [])
    target = None
    for o in obs:
        if isinstance(o, dict) and o.get("obstruction_id") == "dp_gap_exceeds_threshold":
            target = o
            break
    if not isinstance(target, dict):
        return {"kind": "fairness_demographic_parity", "summary": "No failure obstruction found."}
    w = target.get("witness", {}) if isinstance(target.get("witness"), dict) else {}
    return {
        "kind": "fairness_demographic_parity",
        "summary": "DP gap exceeds threshold.",
        "group_min": w.get("group_min"),
        "group_max": w.get("group_max"),
        "rate_min": w.get("rate_min"),
        "rate_max": w.get("rate_max"),
        "dp_gap_observed": w.get("dp_gap_observed"),
        "dp_gap_max": w.get("dp_gap_max"),
    }


def _summarize_fairness_equalized_odds(input_obj: Dict[str, Any]) -> Dict[str, Any]:
    obs = input_obj.get("failure_obstructions", [])
    target = None
    for o in obs:
        if isinstance(o, dict) and o.get("obstruction_id") == "equalized_odds_gap_exceeds_threshold":
            target = o
            break
    if not isinstance(target, dict):
        return {"kind": "fairness_equalized_odds", "summary": "No failure obstruction found."}
    w = target.get("witness", {}) if isinstance(target.get("witness"), dict) else {}
    tpr = w.get("tpr", {}) if isinstance(w.get("tpr"), dict) else {}
    fpr = w.get("fpr", {}) if isinstance(w.get("fpr"), dict) else {}

    # Pick the single worst violation by (gap - gap_max), if parseable.
    best = None
    for metric, side in [("TPR", tpr), ("FPR", fpr)]:
        try:
            gap = _as_fraction(side.get("gap"))
            gap_max = _as_fraction(side.get("gap_max"))
            delta = gap - gap_max
            cand = (delta, metric, side)
            if best is None or cand[0] > best[0]:
                best = cand
        except Exception:
            continue

    worst = None
    if best is not None:
        _, metric, side = best
        worst = {
            "metric": metric,
            "group_min": side.get("group_min"),
            "group_max": side.get("group_max"),
            "rate_min": side.get("rate_min"),
            "rate_max": side.get("rate_max"),
            "gap": side.get("gap"),
            "gap_max": side.get("gap_max"),
        }

    return {
        "kind": "fairness_equalized_odds",
        "summary": "Equalized odds gap exceeds threshold.",
        "worst_violation": worst,
    }


def _generic_failure_summary(input_obj: Dict[str, Any], *, max_items: int = 5) -> Dict[str, Any]:
    obs = input_obj.get("failure_obstructions", [])
    if not isinstance(obs, list) or not obs:
        return {"kind": "generic", "summary": "No failure_obstructions in input."}
    head = []
    for o in obs[:max_items]:
        if isinstance(o, dict):
            head.append({
                "obstruction_id": o.get("obstruction_id"),
                "conclusion": o.get("conclusion"),
            })
    return {"kind": "generic", "summary": "See failure_obstructions in input.json/result.json.", "obstructions_head": head}


def generate_report(run_id: str) -> Dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise ValueError("invalid run_id format")

    run_dir = _runs_root() / run_id
    meta_path = run_dir / "meta.json"
    input_path = run_dir / "input.json"
    result_path = run_dir / "result.json"

    meta = _load_json(meta_path) if meta_path.exists() else {}
    input_obj = _load_json(input_path) if input_path.exists() else {}
    result_obj = _load_json(result_path) if result_path.exists() else {}

    ok = bool(result_obj.get("ok"))
    validator_id = str(meta.get("validator_id") or result_obj.get("validator_id") or "")

    ledger_ok, ledger_details = ledger.verify()
    ledger_chain_hash = meta.get("ledger_chain_hash")

    # Summarize failure witness in an auditor-legible way.
    witness_summary = None
    if not ok:
        if validator_id == "safety_prompt_injection_refusal_v1":
            witness_summary = _summarize_safety_prompt_injection(input_obj)
        elif validator_id == "fairness_demographic_parity_v1":
            witness_summary = _summarize_fairness_demographic_parity(input_obj)
        elif validator_id == "fairness_equalized_odds_v1":
            witness_summary = _summarize_fairness_equalized_odds(input_obj)
        else:
            witness_summary = _generic_failure_summary(input_obj)

    report = {
        "schema_id": REPORT_SCHEMA_ID,
        "report_created_utc": _utc_now_compact(),
        "run": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "filename": meta.get("filename"),
            "validator_id": validator_id,
            "ok": ok,
            "passed": result_obj.get("passed"),
            "failed": result_obj.get("failed"),
            "warnings": result_obj.get("warnings"),
        },
        "hashes": {
            "uploaded_sha256": meta.get("uploaded_sha256"),
            "canonical_sha256": meta.get("canonical_sha256"),
            "ledger_chain_hash": ledger_chain_hash,
        },
        "ledger": {
            "path": str(ledger.path),
            "ok": ledger_ok,
            **(ledger_details if isinstance(ledger_details, dict) else {}),
        },
        "witness_summary": witness_summary,
        "artifacts": {
            "input_json_path": str(input_path),
            "result_json_path": str(result_path),
            "meta_json_path": str(meta_path),
        },
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate an auditor-facing report for a dashboard run_id")
    ap.add_argument("--run-id", required=True, help="Run ID under QA_DASHBOARD_RUNS_DIR")
    ap.add_argument("--out", default="", help="Optional output path (writes JSON). If omitted, prints to stdout.")
    args = ap.parse_args(argv)

    rep = generate_report(args.run_id)
    payload = json.dumps(rep, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

