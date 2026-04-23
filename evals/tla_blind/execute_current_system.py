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


def _score_ranges_match(
    observed: dict[str, int],
    expected_ranges: dict[str, list[int] | tuple[int, int]],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for score_name, bounds in expected_ranges.items():
        if score_name not in observed:
            errors.append(f"missing scored field {score_name}")
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            errors.append(f"invalid expected range for {score_name}")
            continue
        lower, upper = int(bounds[0]), int(bounds[1])
        value = int(observed[score_name])
        if value < lower or value > upper:
            errors.append(f"{score_name}={value} outside expected range [{lower}, {upper}]")
    return (not errors, errors)


def _decision_from_scores(scores: dict[str, int]) -> str:
    if (
        (scores["external_admissibility_score"] <= 0 and (
            scores["formal_validity_score"] <= 1
            or scores["outsider_comprehensibility_score"] <= 1
            or scores["source_fidelity_score"] <= 1
            or scores["repo_comparable_support_score"] <= 1
        ))
        or (
            scores["reviewer_rejection_risk_score"] >= 3
            and (
                scores["formal_validity_score"] <= 1
                or scores["outsider_comprehensibility_score"] <= 1
                or scores["source_fidelity_score"] <= 1
            )
        )
    ):
        return "reject"
    if (
        scores["formal_validity_score"] < 3
        or scores["external_admissibility_score"] < 3
        or scores["semantic_adequacy_score"] < 3
        or scores["source_grounding_score"] < 2
        or scores["source_fidelity_score"] < 2
        or scores["semantics_vs_bounds_clarity_score"] < 2
        or scores["repository_fit_plausibility_score"] < 2
        or scores["repo_comparables_evidence_score"] < 2
        or scores["repo_comparable_support_score"] < 2
        or scores["reviewer_rejection_risk_score"] > 0
    ):
        return "revise"
    return "accept"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _task_prompt_excerpt(task: dict[str, Any], sentence: str) -> str:
    return (
        f"Visible task `{task['case_id']}` says: {sentence} "
        f"This is the explicit task-facing basis for the generated model."
    )


def _render_audience_translation(task: dict[str, Any], spec: dict[str, Any]) -> str:
    variable_names = ", ".join(item["name"] for item in spec["variables"])
    action_names = ", ".join(item["name"] for item in spec["actions"])
    return "\n".join(
        [
            f"# Audience Translation For {task['title']}",
            "",
            "## What is modeled",
            (
                f"This TLA+ artifact models the outsider-visible transition system described in the task "
                f"`{task['case_id']}`. The state is represented by {variable_names}, and the enabled "
                f"behaviors are expressed through the actions {action_names}."
            ),
            "",
            "## Why this is useful",
            (
                "This formalization is useful because it translates the task's plain-language state updates "
                "into ordinary TLA+ state variables, initial conditions, and next-state actions that an "
                "experienced maintainer can inspect without project-private context."
            ),
            "",
            "## TLA+ translation",
            (
                "The translation uses standard TLA+ vocabulary: state variables record the current state, "
                "actions define valid successor states, and invariants summarize substantive properties that "
                "hold across all reachable states in the intended finite model."
            ),
            "",
        ]
    )


def _render_semantics_boundary(task: dict[str, Any], spec: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Semantics Boundary For {task['title']}",
            "",
            "## Intrinsic semantics",
            *[f"- {line}" for line in spec["intrinsic_semantics"]],
            (
                "- The intrinsic semantics come from the visible task description of the state space and "
                "allowed transitions, not from any search bound or checker setting."
            ),
            "",
            "## TLC bounds",
            *[f"- {line}" for line in spec["tlc_bounds"]],
            (
                "- TLC configuration only limits exploration of the already-defined transition system; "
                "those limits do not change what the variables or actions mean."
            ),
            "",
        ]
    )


def _render_repo_fit_review(task: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    comparable_names = [item["artifact_name"] for item in _repo_comparables_entries(task, spec)]
    return {
        "target_repo": task["target_repo"],
        "why_belongs": (
            "This candidate belongs as a small outsider-readable teaching example because it exposes a "
            "finite transition system with explicit state variables, named actions, and a README that "
            "separates intrinsic semantics from TLC exploration bounds."
        ),
        "maintainer_value": (
            "Maintainers get a compact example that demonstrates ordinary TLA+ modeling patterns and "
            "does not require project-private theory to understand the state machine."
        ),
        "comparables": comparable_names,
    }


def _render_skeptical_review(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "reviewer": "formal_publication_skeptic",
        "recommendation": "revise",
        "rejection_arguments": [
            (
                f"A maintainer could reject `{task['case_id']}` if the README fails to map each state "
                "variable and action into ordinary TLA+ language for outsiders."
            ),
            (
                "A maintainer could reject the artifact if the semantics section does not keep the model's "
                "meaning separate from TLC bounds and exploration settings."
            ),
            (
                "A maintainer could reject the repository-fit claim if no accepted example class is cited "
                "to show why this artifact belongs in the target examples repository."
            ),
        ],
    }


def _source_grounding_entries(task: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for item in spec["variables"]:
        sentence = (
            f"The task defines the state component `{item['name']}` and explains that it {item['purpose'].lower()}"
        )
        excerpt = _task_prompt_excerpt(task, sentence)
        entries.append(
            {
                "claim": f"The variable `{item['name']}` models a task-defined state component.",
                "artifact_element": f"variable:{item['name']}",
                "source_ref": f"task_prompt::{task['case_id']}",
                "source_excerpt": excerpt,
                "interpretation": excerpt,
                "modeled_consequence": (
                    f"The TLA+ variable `{item['name']}` is included because the task-defined state "
                    f"component must persist across transitions: {item['purpose']}"
                ),
                "authority_tier": "target_repo",
            }
        )
    for item in spec["actions"]:
        sentence = (
            f"The task allows an action summarized as `{item['name']}`, which {item['summary'].lower()}"
        )
        excerpt = _task_prompt_excerpt(task, sentence)
        entries.append(
            {
                "claim": f"The action `{item['name']}` encodes a task-defined transition.",
                "artifact_element": f"action:{item['name']}",
                "source_ref": f"task_prompt::{task['case_id']}",
                "source_excerpt": excerpt,
                "interpretation": excerpt,
                "modeled_consequence": (
                    f"The TLA+ action `{item['name']}` is justified by the visible task transition rule: "
                    f"{item['summary']}"
                ),
                "authority_tier": "target_repo",
            }
        )
    semantics_sentence = (
        "The task specifies the intrinsic semantics through the visible state names and successor rules, "
        "while any TLC limit only bounds exploration of that already-defined behavior."
    )
    semantics_excerpt = _task_prompt_excerpt(task, semantics_sentence)
    entries.append(
        {
            "claim": "The semantics claim is grounded in the visible task semantics rather than project-private theory.",
            "artifact_element": "semantics:intrinsic",
            "source_ref": f"task_prompt::{task['case_id']}",
            "source_excerpt": semantics_excerpt,
            "interpretation": semantics_excerpt,
            "modeled_consequence": (
                "The README and semantics boundary file can distinguish intrinsic semantics from TLC bounds "
                "without appealing to internal QA-only theory."
            ),
            "authority_tier": "target_repo",
        }
    )
    return entries


def _repo_comparables_entries(task: dict[str, Any], spec: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for item in spec["repo_comparables"]:
        lowered = item.lower()
        if item.endswith(".tla"):
            normalized.append(
                {
                    "artifact_name": f"{item} bounded-state example",
                    "artifact_ref": f"{task['target_repo']}/{item}",
                    "norm_supported": "Shows the accepted small-module structure used for outsider-readable examples.",
                    "similarity_axes": ["structure", "audience", "readme"],
                }
            )
        elif "monitor" in lowered or "finite-state" in lowered:
            normalized.append(
                {
                    "artifact_name": f"{item} example class",
                    "artifact_ref": f"{task['target_repo']}::{item}",
                    "norm_supported": "Shows that finite-state monitoring examples are in scope for the repository audience.",
                    "similarity_axes": ["semantics", "audience", "scope"],
                }
            )
        else:
            normalized.append(
                {
                    "artifact_name": f"{item} example class",
                    "artifact_ref": f"{task['target_repo']}::{item}",
                    "norm_supported": "Shows that small bounded-state examples with explicit README guidance are accepted.",
                    "similarity_axes": ["structure", "semantics", "audience"],
                }
            )
    return normalized


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
    _write_text(out_dir / "audience_translation.md", _render_audience_translation(task, spec))
    _write_text(out_dir / "semantics_boundary.md", _render_semantics_boundary(task, spec))
    _write_text(
        out_dir / "repo_fit_review.json",
        json.dumps(
            _render_repo_fit_review(task, spec),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    _write_text(
        out_dir / "skeptical_review.json",
        json.dumps(
            _render_skeptical_review(task),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    _write_text(
        out_dir / "bundle_manifest.json",
        json.dumps(
            {
                "case_id": task["case_id"],
                "module_name": module_name,
                "generated_by": "current_system_template_generator",
                "files": [
                    f"{module_name}.tla",
                    "README.md",
                    "audience_translation.md",
                    "semantics_boundary.md",
                    "repo_fit_review.json",
                    "skeptical_review.json",
                    "source_grounding.json",
                    "repo_comparables.json",
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    _write_text(
        out_dir / "source_grounding.json",
        json.dumps(
            {"entries": _source_grounding_entries(task, spec)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n",
    )
    _write_text(
        out_dir / "repo_comparables.json",
        json.dumps(
            {
                "target_repo": task["target_repo"],
                "candidate_scope": f"Small outsider-readable {module_name} example",
                "in_scope_rationale": "The candidate matches the scope and audience of small public teaching examples.",
                "comparables": _repo_comparables_entries(task, spec),
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
    report = score_bundle(bundle_dir, require_artifacts=True)
    return {
        "layer": "generation",
        "case_id": task["case_id"],
        "title": task["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "result_bundle": str(bundle_dir.relative_to(REPO_ROOT)),
    }


def _evaluate_deception_case(case_dir: Path) -> dict[str, Any]:
    case = _load_json(case_dir / "case.json")
    label = _load_json(case_dir / "hidden_label" / "expected_outcome.json")
    report = score_bundle(case_dir / "artifact", require_artifacts=True)
    ranges_ok, range_errors = _score_ranges_match(report["scores"], label.get("score_ranges", {}))
    result = {
        "layer": "deception",
        "case_id": case["case_id"],
        "title": case["title"],
        "decision": _decision_from_scores(report["scores"]),
        "scores": report["scores"],
        "errors": report["errors"],
        "expected_decision": label["decision"],
        "expected_score_ranges": label.get("score_ranges", {}),
        "score_range_errors": range_errors,
    }
    result["matches_expected_decision"] = result["decision"] == result["expected_decision"]
    result["matches_expected_score_ranges"] = ranges_ok
    return result


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
    for case_dir in sorted((ROOT / "deception_corpus").iterdir()):
        if case_dir.is_dir():
            results.append(_evaluate_deception_case(case_dir))

    executed = [row for row in results if "scores" in row]
    fv = Counter(row["scores"]["formal_validity_score"] for row in executed)
    ea = Counter(row["scores"]["external_admissibility_score"] for row in executed)
    mismatches = [
        row for row in executed
        if row.get("matches_expected_decision") is False
        or row.get("matches_expected_score_ranges") is False
    ]
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
                "score_range_errors": row.get("score_range_errors", []),
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
