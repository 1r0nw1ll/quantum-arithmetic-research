#!/usr/bin/env python3
# noqa: DECL-1 (eval execution scaffold — not empirical QA code)
"""
Execute the blind TLA+ starter corpus against the current heuristic judgment
layer and compare results to hidden labels where available.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from qa_formal_publication_gate import score_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current_system"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _decision_from_scores(scores: dict[str, int]) -> str:
    if (
        (scores["external_admissibility_score"] <= 0 and (
            scores["formal_validity_score"] <= 1
            or scores["outsider_comprehensibility_score"] <= 1
        ))
        or (
            scores["reviewer_rejection_risk_score"] >= 3
            and (scores["formal_validity_score"] <= 1 or scores["outsider_comprehensibility_score"] <= 1)
        )
    ):
        return "reject"
    if (
        scores["formal_validity_score"] < 3
        or scores["external_admissibility_score"] < 3
        or scores["semantic_adequacy_score"] < 3
        or scores["source_grounding_score"] < 2
        or scores["semantics_vs_bounds_clarity_score"] < 2
        or scores["repository_fit_plausibility_score"] < 2
        or scores["repo_comparables_evidence_score"] < 2
        or scores["reviewer_rejection_risk_score"] > 0
    ):
        return "revise"
    return "accept"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_bundle(case_dir: Path) -> Path:
    task = _load_json(case_dir / "task.json")
    spec = task["generation_spec"]
    out_dir = RESULTS_ROOT / "generation" / task["case_id"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    module_name = spec["module_name"]
    variable_names = [item["name"] for item in spec["variables"]]
    variable_line = "VARIABLE " + ", ".join(variable_names) if len(variable_names) == 1 else "VARIABLES " + ", ".join(variable_names)
    tla_lines = [f"---- MODULE {module_name} ----", "EXTENDS Naturals", "", variable_line, ""]
    tla_lines.extend(spec["init_lines"])
    tla_lines.append("")
    tla_lines.extend(spec["action_defs"])
    tla_lines.append("")
    for invariant in spec["invariants"]:
        tla_lines.append(invariant)
    tla_lines.append("")
    tla_lines.append("====")
    _write_text(out_dir / f"{module_name}.tla", "\n".join(tla_lines) + "\n")

    readme_lines = [
        f"# {task['title']}",
        "",
        "## What is modeled",
        f"This artifact models the system described in the visible task prompt for `{task['case_id']}`.",
        "",
        "## Variable/action justification",
    ]
    for item in spec["variables"]:
        readme_lines.append(f"- `{item['name']}`: {item['purpose']}")
    for item in spec["actions"]:
        readme_lines.append(f"- `{item['name']}`: {item['summary']}")
    readme_lines.extend([
        "",
        "## Intrinsic semantics",
        *[f"- {line}" for line in spec["intrinsic_semantics"]],
        "",
        "## TLC bounds",
        *[f"- {line}" for line in spec["tlc_bounds"]],
        "",
        "## Source grounding",
        *[f"- {line}" for line in spec["source_grounding"]],
        "",
        "## Repository fit",
        f"This aims at `{task['target_repo']}`.",
        "Comparable examples:",
        *[f"- similar to {item}" for item in spec["repo_comparables"]],
        "",
    ])
    _write_text(out_dir / "README.md", "\n".join(readme_lines))
    _write_text(
        out_dir / "bundle_manifest.json",
        json.dumps(
            {
                "case_id": task["case_id"],
                "module_name": module_name,
                "generated_by": "current_system_template_generator",
                "files": [f"{module_name}.tla", "README.md"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    return out_dir


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


def _evaluate_generation_case(case_dir: Path) -> dict[str, Any]:
    task = _load_json(case_dir / "task.json")
    bundle_dir = _generate_bundle(case_dir)
    report = score_bundle(bundle_dir, require_artifacts=False)
    return {
        "layer": "generation",
        "case_id": task["case_id"],
        "title": task["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "result_bundle": str(bundle_dir.relative_to(REPO_ROOT)),
    }


def execute() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted((ROOT / "tasks" / "generation").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_generation_case(case_dir))
    for case_dir in sorted((ROOT / "review_corpus").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_review_case(case_dir))
    for case_dir in sorted((ROOT / "repair_cases").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_repair_case(case_dir))

    executed = [row for row in results if "scores" in row]
    fv = Counter(row["scores"]["formal_validity_score"] for row in executed)
    ea = Counter(row["scores"]["external_admissibility_score"] for row in executed)
    mismatches = [row for row in executed if row.get("matches_expected_decision") is False]
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
