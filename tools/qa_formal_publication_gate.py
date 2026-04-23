#!/usr/bin/env python3
# noqa: DECL-1 (guardrail tool — not empirical QA code)
"""
Fail-closed gate for public-facing formal-methods artifacts.

Phase 1 is intentionally minimal: it blocks commit/push when formal artifacts
are changed without the outsider-facing explanation and approval files needed
to keep internal structure from being mistaken for external admissibility.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "audience_translation.md",
    "semantics_boundary.md",
    "repo_fit_review.json",
    "skeptical_review.json",
    "human_approval.json",
)
BLOCKED_PUBLICATION_CLAIMS = (
    "publication-ready",
    "submittable upstream",
)
PLACEHOLDER_TOKENS = {
    "tbd",
    "todo",
    "n/a",
    "na",
    "none",
    "unknown",
    "placeholder",
    "fill me in",
    "coming soon",
}
SKEPTICAL_KEYWORDS = {
    "reject",
    "maintainer",
    "coherence",
    "semantic",
    "invariant",
    "fit",
    "repository",
    "outsider",
    "grounding",
    "vacuous",
}
TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".adoc", ".json", ".yaml", ".yml", ".tla", ".cfg"}
PROJECT_PRIVATE_JARGON = (
    "observer projection firewall",
    "qa legality",
    "theorem nt",
    "theorem-facing spine",
    "observer-ready",
    "lane-two",
)
COMPARABLE_EVIDENCE_MARKERS = (
    "similar to",
    "in the style of",
    "comparable to",
    "compare to",
    "comparable examples",
    "examples like",
)
TAUTOLOGY_PATTERNS = (
    re.compile(r"^\s*TRUE\s*$", re.I | re.M),
    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\s*$", re.I | re.M),
    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*\1\s*>=\s*0\s*$", re.I | re.M),
    re.compile(r"^\s*1\s*=\s*1\s*$", re.I | re.M),
)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _bundle_root_for_path(repo_root: Path, rel_path: str) -> Path:
    candidate = (repo_root / rel_path).resolve(strict=False)
    if rel_path.startswith("qa_alphageometry_ptolemy/"):
        return (repo_root / "qa_alphageometry_ptolemy").resolve(strict=False)
    return candidate.parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _is_placeholder_text(value: Any, *, min_length: int = 12) -> bool:
    if not isinstance(value, str):
        return True
    lowered = " ".join(value.lower().split())
    if len(lowered) < min_length:
        return True
    return lowered in PLACEHOLDER_TOKENS or any(token in lowered for token in ("auto-generated", "template", "lorem ipsum"))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("JSON artifact must be an object")
    return data


def _validate_audience_translation(path: Path) -> list[str]:
    text = _read_text(path).strip()
    errors: list[str] = []
    if len(text) < 120:
        errors.append("audience_translation.md is too short for outsider grounding")
    lowered = text.lower()
    if "what is modeled" not in lowered and "modeled" not in lowered:
        errors.append("audience_translation.md must explain what is being modeled")
    if "why" not in lowered or "useful" not in lowered:
        errors.append("audience_translation.md must explain why the model is useful")
    if "tla+" not in lowered and "formal" not in lowered:
        errors.append("audience_translation.md must translate into formal-methods language")
    return errors


def _validate_semantics_boundary(path: Path) -> list[str]:
    text = _read_text(path).strip()
    lowered = text.lower()
    errors: list[str] = []
    if len(text) < 120:
        errors.append("semantics_boundary.md is too short to separate semantics from bounds")
    if "intrinsic semantics" not in lowered:
        errors.append("semantics_boundary.md must name the intrinsic semantics section")
    if "tlc" not in lowered and "model checking" not in lowered:
        errors.append("semantics_boundary.md must name TLC or model checking bounds")
    if "bound" not in lowered and "cap" not in lowered:
        errors.append("semantics_boundary.md must distinguish bounds/caps from semantics")
    return errors


def _validate_repo_fit(path: Path) -> list[str]:
    data = _load_json(path)
    errors: list[str] = []
    for key in ("target_repo", "why_belongs", "maintainer_value"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"repo_fit_review.json missing non-empty {key}")
        elif _is_placeholder_text(value, min_length=18):
            errors.append(f"repo_fit_review.json {key} is content-free")
    comparables = data.get("comparables")
    if not isinstance(comparables, list) or len(comparables) < 2:
        errors.append("repo_fit_review.json must list at least 2 target-repo comparables")
    elif not all(isinstance(item, str) and not _is_placeholder_text(item, min_length=4) for item in comparables):
        errors.append("repo_fit_review.json comparables must be specific non-placeholder entries")
    return errors


def _validate_skeptical_review(path: Path) -> list[str]:
    data = _load_json(path)
    errors: list[str] = []
    recommendation = data.get("recommendation")
    if recommendation not in {"accept", "revise", "reject"}:
        errors.append("skeptical_review.json recommendation must be accept/revise/reject")
    arguments = data.get("rejection_arguments")
    if not isinstance(arguments, list) or not arguments or not all(isinstance(x, str) and x.strip() for x in arguments):
        errors.append("skeptical_review.json must include non-empty rejection_arguments")
    else:
        if len(arguments) < 2:
            errors.append("skeptical_review.json must include at least 2 rejection_arguments")
        substantive_arguments = [arg for arg in arguments if len(arg.strip()) >= 24 and not _is_placeholder_text(arg, min_length=24)]
        if len(substantive_arguments) != len(arguments):
            errors.append("skeptical_review.json rejection_arguments must be substantive, not templated")
        elif not any(any(keyword in arg.lower() for keyword in SKEPTICAL_KEYWORDS) for arg in substantive_arguments):
            errors.append("skeptical_review.json must articulate an adversarial rejection risk")
    reviewer = data.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("skeptical_review.json missing reviewer")
    elif any(token in reviewer.lower() for token in ("template", "auto", "bot", "generated")):
        errors.append("skeptical_review.json reviewer must not look automated or templated")
    return errors


def _validate_human_approval(path: Path) -> list[str]:
    data = _load_json(path)
    errors: list[str] = []
    for key in ("approver", "approved_at", "scope", "justification"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"human_approval.json missing non-empty {key}")
    if data.get("approved") is not True:
        errors.append("human_approval.json must set approved=true")
    approver = data.get("approver")
    if isinstance(approver, str) and any(token in approver.lower() for token in ("auto", "bot", "system", "template", "generated")):
        errors.append("human_approval.json approver must identify a human, not automation")
    justification = data.get("justification")
    if isinstance(justification, str):
        if _is_placeholder_text(justification, min_length=40):
            errors.append("human_approval.json justification is content-free")
        if any(token in justification.lower() for token in ("auto-generated", "mechanically generated", "generated automatically")):
            errors.append("human_approval.json justification admits mechanical generation")
    return errors


def _collect_texts(bundle_root: Path, extensions: set[str] | None = None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    allowed = TEXT_EXTENSIONS if extensions is None else extensions
    for path in sorted(bundle_root.rglob("*")):
        if path.suffix.lower() not in allowed or not path.is_file():
            continue
        try:
            out.append((str(path.relative_to(bundle_root)), _read_text(path)))
        except OSError:
            continue
    return out


def _collect_tla_texts(bundle_root: Path) -> list[tuple[str, str]]:
    return _collect_texts(bundle_root, extensions={".tla", ".cfg"})


def _extract_state_variables(tla_text: str) -> list[str]:
    variables: list[str] = []
    for line in tla_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("VARIABLE"):
            continue
        _, names = stripped.split("VARIABLE", 1)
        for token in re.split(r"[\s,]+", names.strip()):
            if token:
                variables.append(token)
    return variables


def _extract_action_names(tla_text: str) -> list[str]:
    actions: list[str] = []
    for name, expr in re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$", tla_text, re.M):
        lower_name = name.lower()
        if lower_name in {"init", "next", "spec"} or "inv" in lower_name or "type" in lower_name:
            continue
        if "'" in expr or "\\/" in expr or "/\\" in expr:
            actions.append(name)
    return actions


def _check_tautological_invariants(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    invariant_header = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(.+)$", re.M)
    for rel_path, text in _collect_tla_texts(bundle_root):
        for name, expr in invariant_header.findall(text):
            if "inv" not in name.lower() and "invariant" not in name.lower():
                continue
            stripped = expr.strip()
            if any(pattern.search(stripped) for pattern in TAUTOLOGY_PATTERNS):
                findings.append(f"{rel_path}:{name} looks tautological")
    return findings


def _check_negative_tests(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    for rel_path, text in _collect_texts(bundle_root, extensions={".md", ".txt", ".tla", ".cfg"}):
        lowered = text.lower()
        if "negative test" in lowered and "malformed literal" in lowered:
            findings.append(f"{rel_path} describes a malformed-literal negative test")
        if "broken" in text and "literal" in lowered and "semantics" not in lowered:
            findings.append(f"{rel_path} looks like a toy broken-literal test")
    return findings


def _check_publication_claims(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    for rel_path, text in _collect_texts(bundle_root):
        lowered = text.lower()
        for phrase in BLOCKED_PUBLICATION_CLAIMS:
            if phrase in lowered:
                findings.append(f"{rel_path} contains blocked claim '{phrase}' before gate pass")
    return findings


def _check_stuttering_only_next(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    for rel_path, text in _collect_tla_texts(bundle_root):
        variables = _extract_state_variables(text)
        if not variables:
            continue
        next_match = re.search(r"^\s*Next\s*==\s*(.+?)(?:^\s*[A-Za-z_][A-Za-z0-9_]*\s*==|\Z)", text, re.M | re.S)
        if not next_match:
            continue
        next_body = next_match.group(1)
        stuttering_patterns = {
            var: re.compile(rf"{re.escape(var)}'\s*=\s*{re.escape(var)}(?![A-Za-z0-9_])")
            for var in variables
        }
        non_stuttering_patterns = {
            var: re.compile(rf"{re.escape(var)}'\s*=\s*(?!{re.escape(var)}(?![A-Za-z0-9_]))")
            for var in variables
        }
        if not all(pattern.search(next_body) for pattern in stuttering_patterns.values()):
            continue
        if any(pattern.search(next_body) for pattern in non_stuttering_patterns.values()):
            continue
        findings.append(f"{rel_path}:Next looks stuttering-only for all state variables")
    return findings


def _readme_text(bundle_root: Path) -> str:
    for rel_path, text in _collect_texts(bundle_root, extensions={".md", ".txt"}):
        if rel_path.lower().endswith("readme.md"):
            return text
    return ""


def _joined_explanatory_text(bundle_root: Path) -> str:
    parts: list[str] = []
    for _, text in _collect_texts(bundle_root, extensions={".md", ".txt"}):
        parts.append(text)
    return "\n".join(parts)


def _check_variable_action_mapping(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    readme = _joined_explanatory_text(bundle_root)
    if not readme.strip():
        return ["README explanation missing or unreadable"]
    lowered = readme.lower()
    variables: list[str] = []
    actions: list[str] = []
    for _, text in _collect_tla_texts(bundle_root):
        variables.extend(_extract_state_variables(text))
        actions.extend(_extract_action_names(text))
    missing_variables = [name for name in variables if name.lower() not in lowered]
    if variables and missing_variables:
        findings.append("README does not explain all state variables: " + ", ".join(sorted(set(missing_variables))))
    mentioned_actions = [name for name in actions if name.lower() in lowered]
    if actions and not mentioned_actions:
        findings.append("README does not map action names into outsider-facing prose")
    return findings


def _check_project_private_jargon(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    for rel_path, text in _collect_texts(bundle_root, extensions={".md", ".txt"}):
        lowered = text.lower()
        for term in PROJECT_PRIVATE_JARGON:
            if term not in lowered:
                continue
            if any(marker in lowered for marker in ("translated as", "means", "i.e.", "ordinary tla+", "outsider")):
                continue
            findings.append(f"{rel_path} uses project-private jargon without translation: '{term}'")
    return findings


def _check_visible_semantics_bounds_conflation(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    for rel_path, text in _collect_texts(bundle_root, extensions={".md", ".txt"}):
        lowered = " ".join(text.lower().split())
        if "semantics" not in lowered:
            continue
        if "tlc" in lowered and any(token in lowered for token in ("cap", "bound", "search depth")):
            if any(phrase in lowered for phrase in (
                "semantics of this model are the tlc cap",
                "semantics are the tlc cap",
                "meaning of the model",
                "cap is understood as the meaning",
            )):
                findings.append(f"{rel_path} conflates intrinsic semantics with TLC bounds")
    return findings


def _check_repository_fit_signal(bundle_root: Path) -> list[str]:
    readme = _joined_explanatory_text(bundle_root)
    if not readme.strip():
        return []
    lowered = readme.lower()
    if "repository fit" in lowered or "examples repository" in lowered or "tlaplus/examples" in lowered:
        return []
    if "publication-ready" in lowered or "public formal-methods contribution" in lowered:
        return ["README asserts public readiness without an explicit repository-fit argument"]
    return []


def _check_source_grounding(bundle_root: Path) -> list[str]:
    text = " ".join(_joined_explanatory_text(bundle_root).lower().split())
    findings: list[str] = []
    if not text:
        return ["No explanatory text available for source grounding"]
    if "what is modeled" not in text and "models" not in text and "modeled" not in text:
        findings.append("Explanatory text does not clearly state what is being modeled")
    if not any(marker in text for marker in ("comes from", "derived from", "based on", "source", "visible task", "task statement")):
        findings.append("Explanatory text does not say where the semantics come from")
    if not any(marker in text for marker in ("justified by", "tracks", "purpose", "chosen variable", "chosen action")):
        findings.append("Explanatory text does not justify the chosen variables/actions")
    return findings


def _check_repo_comparables_evidence(bundle_root: Path) -> list[str]:
    text = " ".join(_joined_explanatory_text(bundle_root).lower().split())
    if not text:
        return []
    mentions_repo_fit = any(marker in text for marker in ("repository fit", "tlaplus/examples", "examples repository", "belongs in"))
    mentions_comparable = any(marker in text for marker in COMPARABLE_EVIDENCE_MARKERS)
    mentions_example_name = any(
        token in text for token in (".tla", "counter-style example", "small bounded-state examples", "finite-state monitoring examples")
    )
    if mentions_repo_fit and not (mentions_comparable or mentions_example_name):
        return ["Repository-fit claim lacks comparable evidence"]
    return []


def _score_from_findings(findings: list[str]) -> dict[str, int]:
    lowered = "\n".join(findings).lower()
    tautology_count = sum("tautological" in item.lower() for item in findings)
    jargon_hit = "jargon" in lowered
    mapping_hit = "does not explain all state variables" in lowered or "does not map action names" in lowered
    bounds_hit = (
        "publication-ready" in lowered
        or "submittable upstream" in lowered
        or "malformed-literal" in lowered
        or "conflates intrinsic semantics with tlc bounds" in lowered
    )
    stuttering_hit = "stuttering-only" in lowered
    readme_missing = "readme explanation missing" in lowered
    repo_fit_hit = "repository-fit" in lowered or "repository fit" in lowered
    source_hit = "where the semantics come from" in lowered or "what is being modeled" in lowered or "justify the chosen variables/actions" in lowered
    comparables_hit = "comparable evidence" in lowered
    scores = {
        "formal_validity_score": 3,
        "external_admissibility_score": 3,
        "semantic_adequacy_score": 3,
        "source_grounding_score": 3,
        "outsider_comprehensibility_score": 3,
        "invariant_non_vacuity_score": 3,
        "semantics_vs_bounds_clarity_score": 3,
        "repository_fit_plausibility_score": 3,
        "repo_comparables_evidence_score": 3,
        "reviewer_rejection_risk_score": 0,
    }
    if tautology_count:
        scores["formal_validity_score"] -= 2
        scores["invariant_non_vacuity_score"] -= 3
        scores["reviewer_rejection_risk_score"] += 1
    if stuttering_hit:
        scores["formal_validity_score"] -= 1
        scores["semantic_adequacy_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if mapping_hit:
        scores["semantic_adequacy_score"] -= 2
        scores["source_grounding_score"] -= 1
        scores["outsider_comprehensibility_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if jargon_hit:
        scores["outsider_comprehensibility_score"] -= 2
        scores["external_admissibility_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 1
    if bounds_hit:
        scores["semantics_vs_bounds_clarity_score"] -= 2
        scores["repository_fit_plausibility_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if repo_fit_hit:
        scores["repository_fit_plausibility_score"] -= 2
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if source_hit:
        scores["source_grounding_score"] -= 2
        scores["semantic_adequacy_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if comparables_hit:
        scores["repo_comparables_evidence_score"] -= 2
        scores["repository_fit_plausibility_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if readme_missing:
        scores["semantic_adequacy_score"] -= 1
        scores["source_grounding_score"] -= 1
        scores["outsider_comprehensibility_score"] -= 1
        scores["external_admissibility_score"] -= 1
    for key in list(scores):
        if key == "reviewer_rejection_risk_score":
            scores[key] = max(0, min(3, scores[key]))
        else:
            scores[key] = max(0, min(3, scores[key]))
    return scores


def score_bundle(bundle_root: Path, *, require_artifacts: bool = True) -> dict[str, Any]:
    findings: list[str] = []
    report: dict[str, Any] = {
        "bundle_root": str(bundle_root),
        "required_artifacts": {},
    }
    if require_artifacts:
        validators = {
            "audience_translation.md": _validate_audience_translation,
            "semantics_boundary.md": _validate_semantics_boundary,
            "repo_fit_review.json": _validate_repo_fit,
            "skeptical_review.json": _validate_skeptical_review,
            "human_approval.json": _validate_human_approval,
        }
        for required_name in REQUIRED_FILES:
            artifact_path = bundle_root / required_name
            artifact_errors: list[str] = []
            if not artifact_path.exists():
                artifact_errors.append(f"missing required artifact {required_name}")
            else:
                try:
                    artifact_errors.extend(validators[required_name](artifact_path))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    artifact_errors.append(f"{required_name} invalid: {exc}")
            report["required_artifacts"][required_name] = {
                "present": artifact_path.exists(),
                "errors": artifact_errors,
            }
            findings.extend(artifact_errors)
    findings.extend(_check_tautological_invariants(bundle_root))
    findings.extend(_check_negative_tests(bundle_root))
    findings.extend(_check_publication_claims(bundle_root))
    findings.extend(_check_stuttering_only_next(bundle_root))
    findings.extend(_check_variable_action_mapping(bundle_root))
    findings.extend(_check_project_private_jargon(bundle_root))
    findings.extend(_check_visible_semantics_bounds_conflation(bundle_root))
    findings.extend(_check_repository_fit_signal(bundle_root))
    findings.extend(_check_source_grounding(bundle_root))
    findings.extend(_check_repo_comparables_evidence(bundle_root))
    scores = _score_from_findings(findings)
    report["errors"] = findings
    report["scores"] = scores
    report["ok"] = not findings
    return report


def validate_bundle(bundle_root: Path) -> dict[str, Any]:
    return score_bundle(bundle_root, require_artifacts=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--paths", nargs="*", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _self_test() -> dict[str, Any]:
    return {"ok": True, "required_files": list(REQUIRED_FILES)}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        print(_json_dumps(_self_test()))
        return 0
    repo_root = Path(args.repo_root).resolve()
    if not args.paths:
        print(_json_dumps({"ok": False, "errors": ["no formal paths provided"]}) if args.json else "no formal paths provided")
        return 2
    bundle_roots: list[Path] = []
    seen: set[str] = set()
    for rel_path in args.paths:
        bundle_root = _bundle_root_for_path(repo_root, rel_path)
        marker = str(bundle_root)
        if marker not in seen:
            seen.add(marker)
            bundle_roots.append(bundle_root)
    reports = [validate_bundle(bundle_root) for bundle_root in bundle_roots]
    ok = all(report["ok"] for report in reports)
    payload = {
        "ok": ok,
        "reports": reports,
        "formal_validity_score": min(report["scores"]["formal_validity_score"] for report in reports),
        "external_admissibility_score": min(report["scores"]["external_admissibility_score"] for report in reports),
    }
    if args.json:
        print(_json_dumps(payload))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
