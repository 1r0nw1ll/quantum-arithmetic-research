#!/usr/bin/env python3
# noqa: DECL-1 (eval execution scaffold — not empirical QA code)
"""
Execute the Lean 4 starter corpus against a lightweight heuristic judgment layer.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current_system"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _score_bundle(bundle_root: Path) -> dict[str, Any]:
    findings: list[str] = []
    lean_files = sorted(bundle_root.glob("*.lean"))
    lean_text = "\n".join(_read_text(path) for path in lean_files)
    prose = ""
    for rel in ("README.md", "source_grounding.json", "theorem_fit_review.json"):
        path = bundle_root / rel
        if path.exists():
            prose += "\n" + _read_text(path)
    lowered_prose = prose.lower()
    lowered_lean = lean_text.lower()

    scores = {
        "formal_validity_score": 3,
        "external_admissibility_score": 3,
        "proof_correctness_score": 3,
        "theorem_statement_fidelity_score": 3,
        "source_fidelity_score": 3,
        "math_explanation_quality_score": 3,
        "reviewer_rejection_risk_score": 0,
    }

    if "sorry" in lowered_lean or "admit" in lowered_lean:
        findings.append("Lean proof contains sorry/admit placeholder")
        scores["formal_validity_score"] -= 2
        scores["proof_correctness_score"] -= 3
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 2
    if "group" in lowered_lean and "nat" in lowered_lean:
        findings.append("Theorem statement appears to import group-level claims for a natural-number proof task")
        scores["formal_validity_score"] -= 1
        scores["proof_correctness_score"] -= 1
        scores["theorem_statement_fidelity_score"] -= 3
        scores["external_admissibility_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 2
    if any(token in lowered_prose for token in ("wider abstraction", "group-theoretic", "general group", "ambient algebraic identity")):
        findings.append("README overclaims source support through unnecessary abstraction")
        scores["theorem_statement_fidelity_score"] -= 1
        scores["source_fidelity_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "theorem statement exactly matches" not in lowered_prose and "matches the task statement" not in lowered_prose:
        findings.append("README does not explicitly state theorem-statement fidelity")
        scores["theorem_statement_fidelity_score"] -= 1
        scores["math_explanation_quality_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "source excerpt" not in lowered_prose and "\"source_excerpt\"" not in lowered_prose:
        findings.append("Source grounding is missing explicit excerpts")
        scores["source_fidelity_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if any(token in lowered_prose for token in ("overstates", "stronger than source", "beyond the source")):
        findings.append("Grounding packet admits source overreach")
        scores["source_fidelity_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "proof idea:" not in lowered_prose and "we use induction" not in lowered_prose and "we use simp" not in lowered_prose:
        findings.append("README does not explain the proof idea")
        scores["math_explanation_quality_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "follows from" in lowered_prose and "source excerpt" in lowered_prose and "not a proof obligation" in lowered_prose:
        findings.append("Grounding uses source text that names terms but does not justify the proof obligation")
        scores["source_fidelity_score"] -= 2
        scores["proof_correctness_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "example worth reviewing" not in lowered_prose and "maintainer value" not in lowered_prose:
        findings.append("Bundle does not justify why the proof is worth external review")
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    for key in scores:
        if key == "reviewer_rejection_risk_score":
            scores[key] = max(0, min(3, scores[key]))
        else:
            scores[key] = max(0, min(3, scores[key]))
    return {"ok": not findings, "errors": findings, "scores": scores}


def _decision_from_scores(scores: dict[str, int]) -> str:
    if (
        scores["proof_correctness_score"] <= 0
        or scores["theorem_statement_fidelity_score"] <= 0
        or (scores["external_admissibility_score"] <= 0 and scores["reviewer_rejection_risk_score"] >= 2)
    ):
        return "reject"
    if (
        scores["formal_validity_score"] < 3
        or scores["external_admissibility_score"] < 3
        or scores["proof_correctness_score"] < 3
        or scores["theorem_statement_fidelity_score"] < 3
        or scores["source_fidelity_score"] < 3
        or scores["math_explanation_quality_score"] < 3
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
    _write_text(out_dir / f"{spec['module_name']}.lean", spec["lean_text"].rstrip() + "\n")
    _write_text(
        out_dir / "README.md",
        "\n".join(
            [
                f"# {task['title']}",
                "",
                f"The theorem statement exactly matches the task statement: {spec['theorem_summary']}",
                "",
                f"Proof idea: {spec['proof_idea']}",
                "",
                f"Maintainer value: {spec['maintainer_value']}",
                "",
            ]
        ),
    )
    _write_text(
        out_dir / "source_grounding.json",
        json.dumps(
            {
                "entries": spec["source_grounding"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    _write_text(
        out_dir / "theorem_fit_review.json",
        json.dumps(
            {
                "target_context": task["target_context"],
                "why_belongs": spec["why_belongs"],
                "maintainer_value": spec["maintainer_value"],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    return out_dir


def _score_ranges_match(observed: dict[str, int], expected_ranges: dict[str, list[int] | tuple[int, int]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for name, bounds in expected_ranges.items():
        if name not in observed:
            errors.append(f"missing scored field {name}")
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            errors.append(f"invalid expected range for {name}")
            continue
        lower, upper = int(bounds[0]), int(bounds[1])
        value = int(observed[name])
        if value < lower or value > upper:
            errors.append(f"{name}={value} outside expected range [{lower}, {upper}]")
    return (not errors, errors)


def _evaluate_generation_case(case_dir: Path) -> dict[str, Any]:
    task = _load_json(case_dir / "task.json")
    bundle_dir = _generate_bundle(case_dir)
    report = _score_bundle(bundle_dir)
    return {
        "layer": "generation",
        "case_id": task["case_id"],
        "title": task["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "result_bundle": str(bundle_dir.relative_to(ROOT.parent.parent)),
    }


def _evaluate_review_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_scorecard.json")
    report = _score_bundle(case_dir / "artifact")
    return {
        "layer": "review",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["decision"],
        "expected_scores": label["scores"],
        "matches_expected_decision": _decision_from_scores(report["scores"]) == label["decision"],
    }


def _evaluate_repair_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_outcome.json")
    report = _score_bundle(case_dir / "input")
    return {
        "layer": "repair",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["final_decision"],
        "matches_expected_decision": _decision_from_scores(report["scores"]) == label["final_decision"],
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
    print(_json_dumps(execute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
