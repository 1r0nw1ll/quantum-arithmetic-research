#!/usr/bin/env python3
# noqa: DECL-1 (eval execution scaffold — not empirical QA code)
"""
Execute the blind TLA+ starter corpus against the current heuristic judgment
layer and compare results to hidden labels where available.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from qa_formal_publication_gate import score_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _decision_from_scores(scores: dict[str, int]) -> str:
    if scores["external_admissibility_score"] <= 0 or scores["reviewer_rejection_risk_score"] >= 3:
        return "reject"
    if (
        scores["formal_validity_score"] < 3
        or scores["external_admissibility_score"] < 3
        or scores["semantic_adequacy_score"] < 3
        or scores["semantics_vs_bounds_clarity_score"] < 2
        or scores["repository_fit_plausibility_score"] < 2
        or scores["reviewer_rejection_risk_score"] > 0
    ):
        return "revise"
    return "accept"


def _evaluate_review_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_scorecard.json")
    report = score_bundle(case_dir / "artifact", require_artifacts=False)
    result = {
        "layer": "review",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["decision"],
        "expected_scores": label["scores"],
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    return result


def _evaluate_repair_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_outcome.json")
    report = score_bundle(case_dir / "input", require_artifacts=False)
    result = {
        "layer": "repair",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["final_decision"],
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    return result


def _generation_stub(case_dir: Path) -> dict[str, Any]:
    task = _load_json(case_dir / "task.json")
    return {
        "layer": "generation",
        "case_id": task["case_id"],
        "title": task["title"],
        "status": "not_executed",
        "reason": "Current post-remediation system has no automated blind-generation executor wired into the harness.",
    }


def execute() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case_dir in sorted((ROOT / "tasks" / "generation").iterdir()):
        if case_dir.is_dir():
            results.append(_generation_stub(case_dir))
    for case_dir in sorted((ROOT / "review_corpus").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_review_case(case_dir))
    for case_dir in sorted((ROOT / "repair_cases").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_repair_case(case_dir))

    executed = [row for row in results if "scores" in row]
    fv = Counter(row["scores"]["formal_validity_score"] for row in executed)
    ea = Counter(row["scores"]["external_admissibility_score"] for row in executed)
    mismatches = [row for row in executed if not row["matches_expected_decision"]]
    return {
        "ok": True,
        "results": results,
        "formal_validity_distribution": {str(k): fv[k] for k in sorted(fv)},
        "external_admissibility_distribution": {str(k): ea[k] for k in sorted(ea)},
        "mismatches": [
            {
                "case_id": row["case_id"],
                "layer": row["layer"],
                "decision": row["decision"],
                "expected_decision": row["expected_decision"],
                "errors": row["errors"],
            }
            for row in mismatches
        ],
    }


def main() -> int:
    payload = execute()
    print(_json_dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
