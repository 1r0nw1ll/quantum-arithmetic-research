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
    "source_grounding.json",
    "repo_comparables.json",
)
OPTIONAL_REVIEW_FILES = (
    "human_approval.json",
    "human_override.json",
    "human_audit.json",
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
AUTHORITATIVE_SOURCE_TIERS = {"target_repo", "formalism", "canonical"}
NON_AUTHORITATIVE_SOURCE_TIERS = {"internal", "private", "generated", "summary", "self"}
LIMITING_EXCERPT_MARKERS = (
    "terminology only",
    "does not define semantics",
    "illustrative only",
    "not a semantic commitment",
    "for naming only",
    "example notation",
    "does not justify",
    "not intended as a full model",
)
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

# Score-axis classification for the Pass-7 intrinsic/completeness split.
# intrinsic = can be judged from the artifact itself (spec text, prose, proof body)
# completeness = depends on our local submission-bundle conventions
# aggregate = combined dimensions that reflect both
INTRINSIC_SCORE_KEYS = frozenset({
    "formal_validity_score",
    "semantic_adequacy_score",
    "outsider_comprehensibility_score",
    "invariant_non_vacuity_score",
    "semantics_vs_bounds_clarity_score",
    "repository_fit_plausibility_score",
})
BUNDLE_COMPLETENESS_SCORE_KEYS = frozenset({
    "source_grounding_score",
    "source_fidelity_score",
    "repo_comparables_evidence_score",
    "repo_comparable_support_score",
})
AGGREGATE_SCORE_KEYS = frozenset({
    "external_admissibility_score",
    "reviewer_rejection_risk_score",
})


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


def _normalized_words(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*", text.lower())
        if len(token) >= 4
    }


def _looks_self_referential_source(source_ref: str) -> bool:
    lowered = source_ref.lower()
    return any(
        marker in lowered for marker in (
            "readme.md",
            ".tla",
            "artifact_manifest",
            "qarm_proof_ledger",
            "audience_translation",
            "semantics_boundary",
            "skeptical_review",
            "repo_fit_review",
            "source_grounding",
            "repo_comparables",
        )
    )


def _artifact_elements(bundle_root: Path) -> tuple[list[str], list[str]]:
    variables: list[str] = []
    actions: list[str] = []
    for _, text in _collect_tla_texts(bundle_root):
        variables.extend(_extract_state_variables(text))
        actions.extend(_extract_action_names(text))
    return sorted(set(variables)), sorted(set(actions))


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
    target_repo = data.get("target_repo")
    if not isinstance(target_repo, str) or not target_repo.strip():
        errors.append("repo_fit_review.json missing non-empty target_repo")
    elif _is_placeholder_text(target_repo, min_length=6) and "/" not in target_repo:
        errors.append("repo_fit_review.json target_repo is content-free")
    for key in ("why_belongs", "maintainer_value"):
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


def _validate_source_grounding(path: Path, bundle_root: Path) -> list[str]:
    data = _load_json(path)
    errors: list[str] = []
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        return ["source_grounding.json must contain a non-empty entries list"]
    variables, actions = _artifact_elements(bundle_root)
    covered_variables: set[str] = set()
    covered_actions: set[str] = set()
    authoritative_semantics = 0
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"source_grounding.json entry {idx} must be an object")
            continue
        for field in ("claim", "artifact_element", "source_ref", "source_excerpt", "interpretation", "modeled_consequence"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"source_grounding.json entry {idx} missing non-empty {field}")
        if errors and any(f"entry {idx}" in err for err in errors):
            continue
        source_ref = str(entry["source_ref"]).strip()
        excerpt = str(entry["source_excerpt"]).strip()
        interpretation = str(entry["interpretation"]).strip()
        modeled_consequence = str(entry["modeled_consequence"]).strip()
        artifact_element = str(entry["artifact_element"]).strip()
        tier = str(entry.get("authority_tier", "")).strip().lower()
        if len(excerpt) < 24:
            errors.append(f"source_grounding.json entry {idx} source_excerpt is too short to support the claim")
        if _looks_self_referential_source(source_ref):
            errors.append(f"source_grounding.json entry {idx} is self-referential or cites generated bundle prose")
        if tier in NON_AUTHORITATIVE_SOURCE_TIERS:
            errors.append(f"source_grounding.json entry {idx} relies on non-authoritative source tier '{tier}'")
        excerpt_words = _normalized_words(excerpt)
        interpretation_words = _normalized_words(interpretation)
        if interpretation_words and excerpt_words:
            overlap = interpretation_words & excerpt_words
            if len(overlap) < max(2, min(len(interpretation_words), 4) // 2):
                errors.append(f"source_grounding.json entry {idx} interpretation appears to overreach the excerpt")
        lowered_excerpt = excerpt.lower()
        lowered_interpretation = interpretation.lower()
        lowered_consequence = modeled_consequence.lower()
        if any(marker in lowered_excerpt for marker in LIMITING_EXCERPT_MARKERS):
            if any(
                token in (lowered_interpretation + " " + lowered_consequence)
                for token in ("semantic", "justify", "prove", "ground", "repository fit", "maintainer")
            ):
                errors.append(f"source_grounding.json entry {idx} cherry-picks a limiting excerpt while overstating its consequence")
        if artifact_element.startswith("variable:"):
            covered_variables.add(artifact_element.split(":", 1)[1].strip())
        if artifact_element.startswith("action:"):
            covered_actions.add(artifact_element.split(":", 1)[1].strip())
        claim_text = str(entry["claim"]).lower()
        if "semantic" in claim_text and tier in AUTHORITATIVE_SOURCE_TIERS:
            authoritative_semantics += 1
    missing_variables = [name for name in variables if name not in covered_variables]
    missing_actions = [name for name in actions if name not in covered_actions]
    if missing_variables:
        errors.append("source_grounding.json does not cite all state variables: " + ", ".join(missing_variables))
    if missing_actions:
        errors.append("source_grounding.json does not cite all actions: " + ", ".join(missing_actions))
    if authoritative_semantics == 0:
        errors.append("source_grounding.json has no semantics claim grounded in an authoritative source tier")
    return errors


def _validate_repo_comparables(path: Path, bundle_root: Path) -> list[str]:
    data = _load_json(path)
    errors: list[str] = []
    target_repo = data.get("target_repo")
    if not isinstance(target_repo, str) or not target_repo.strip():
        errors.append("repo_comparables.json missing non-empty target_repo")
    candidate_scope = data.get("candidate_scope")
    if not isinstance(candidate_scope, str) or _is_placeholder_text(candidate_scope, min_length=18):
        errors.append("repo_comparables.json missing substantive candidate_scope")
    in_scope_rationale = data.get("in_scope_rationale")
    if not isinstance(in_scope_rationale, str) or _is_placeholder_text(in_scope_rationale, min_length=24):
        errors.append("repo_comparables.json missing substantive in_scope_rationale")
    comparables = data.get("comparables")
    if not isinstance(comparables, list) or len(comparables) < 2:
        return errors + ["repo_comparables.json must contain at least 2 comparable entries"]
    comparable_axes: set[str] = set()
    style_only_norms = True
    for idx, item in enumerate(comparables):
        if not isinstance(item, dict):
            errors.append(f"repo_comparables.json comparable {idx} must be an object")
            continue
        for field in ("artifact_name", "artifact_ref", "norm_supported"):
            value = item.get(field)
            if not isinstance(value, str) or _is_placeholder_text(value, min_length=12):
                errors.append(f"repo_comparables.json comparable {idx} missing substantive {field}")
        axes = item.get("similarity_axes")
        if not isinstance(axes, list) or not axes or not all(isinstance(axis, str) and axis.strip() for axis in axes):
            errors.append(f"repo_comparables.json comparable {idx} must include non-empty similarity_axes")
        else:
            comparable_axes.update(axis.strip().lower() for axis in axes)
        norm_supported = str(item.get("norm_supported", "")).lower()
        if any(token in norm_supported for token in ("semantic", "purpose", "scope", "audience", "behavior")):
            style_only_norms = False
        elif not any(token in norm_supported for token in ("style", "format", "readme", "layout", "structure")):
            style_only_norms = False
    if not comparable_axes.intersection({"structure", "semantics", "readme", "audience"}):
        errors.append("repo_comparables.json comparables do not support structural or semantic similarity")
    if "scope" not in " ".join(_normalized_words(str(in_scope_rationale).lower())) and "audience" not in str(in_scope_rationale).lower():
        errors.append("repo_comparables.json in_scope_rationale should explain scope or audience fit explicitly")
    joined_text = _joined_explanatory_text(bundle_root).lower()
    candidate_text = f"{candidate_scope} {in_scope_rationale} {joined_text}".lower()
    if style_only_norms and not comparable_axes.intersection({"semantics", "audience"}):
        errors.append("repo_comparables.json comparables support style only and do not justify semantic or audience fit")
    if any(token in candidate_text for token in ("internal", "research", "certification", "ledger", "calibration", "theorem")):
        if not comparable_axes.intersection({"semantics", "audience"}) or style_only_norms:
            errors.append("repo_comparables.json comparables do not justify the candidate's research/internal scope")
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


def _strip_tla_comments(text: str) -> str:
    """Remove TLA+ comments before token-level parsing.

    Block comments `(* ... *)` and end-of-line `\\* ...` both carry prose
    (including commas) that would otherwise be mis-parsed as variable or
    action names. Stripping them is a no-op for the outsider-facing text
    checks (which read the raw files) but cleans up structural extractors.
    """
    text = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\*.*$", " ", text, flags=re.M)
    return text


def _extract_state_variables(tla_text: str) -> list[str]:
    variables: list[str] = []
    stripped_text = _strip_tla_comments(tla_text)
    for line in stripped_text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("VARIABLE ") or stripped.startswith("VARIABLES ")):
            continue
        if stripped.startswith("VARIABLES "):
            names = stripped[len("VARIABLES "):]
        else:
            names = stripped[len("VARIABLE "):]
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
    # Pass-20 gap (3/4): vacuous-membership invariants. `TypeOK == counter \in Nat`
    # restates the variable's already-declared domain and adds no constraint.
    # We treat this as tautological at the same severity as the existing
    # `x = x` / `TRUE` patterns. Conservative: trigger only when the entire
    # invariant body is exactly one membership expression against a basic
    # TLA+ type (Nat / Int / BOOLEAN / Real / STRING). Multi-conjunct
    # invariants where each clause is also `\in Type` are also vacuous and
    # are caught by the same pattern when joined.
    vacuous_membership_single = re.compile(
        r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\\in\s*(?:Nat|Int|BOOLEAN|Real|STRING)\s*$"
    )
    vacuous_membership_per_clause = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*\s*\\in\s*(?:Nat|Int|BOOLEAN|Real|STRING)"
    )
    for rel_path, text in _collect_tla_texts(bundle_root):
        for name, expr in invariant_header.findall(text):
            lower_name = name.lower()
            # Only flag invariant-style definitions: those named TypeOK / *Inv* /
            # *Invariant*. Action and helper definitions are excluded.
            is_invariant_name = (
                "inv" in lower_name or "invariant" in lower_name
                or lower_name == "typeok" or lower_name.endswith("typeok")
                or lower_name.startswith("typeok")
            )
            if not is_invariant_name:
                continue
            stripped = expr.strip()
            if any(pattern.search(stripped) for pattern in TAUTOLOGY_PATTERNS):
                findings.append(f"{rel_path}:{name} looks tautological")
                continue
            # Single-clause membership. Distinct severity from
            # `TRUE`/`x=x`: well-formed but uninformative — revise tier.
            # The "vacuously" wording is matched separately in
            # _score_from_findings so the existing strict tautology
            # scoring does not over-penalize.
            if vacuous_membership_single.search(stripped):
                findings.append(
                    f"{rel_path}:{name} is vacuously satisfied "
                    f"(only restates a state variable's already-declared domain)"
                )
                continue
            # Conjunction of memberships only — split on `/\` and check each.
            clauses = [c.strip() for c in re.split(r"/\\", stripped) if c.strip()]
            if (len(clauses) >= 2
                    and all(vacuous_membership_per_clause.fullmatch(c) for c in clauses)):
                findings.append(
                    f"{rel_path}:{name} is vacuously satisfied "
                    f"(every conjunct restates a variable's already-declared domain)"
                )
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


def _check_variable_action_mapping(bundle_root: Path, *, tolerate_missing_readme: bool = False) -> list[str]:
    findings: list[str] = []
    readme = _joined_explanatory_text(bundle_root)
    if not readme.strip():
        return [] if tolerate_missing_readme else ["README explanation missing or unreadable"]
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


# Pass-20 gap (4/4): README/spec semantic misalignment.
#
# Pattern: README claims a complex named protocol (two-phase commit, Paxos,
# Raft, leader election, transaction processing, etc.) but the .tla file
# has only trivial counter-style state — one variable, one increment-style
# action, no protocol vocabulary.
#
# Conservative trigger (low false-positive risk on legitimate reduced models):
#   - README contains 2+ phrases from PROTOCOL_CLAIM_MARKERS
#   - .tla has ≤2 state variables AND ≤2 actions
#   - .tla source contains zero matches for any of the README's claimed
#     protocol vocabulary tokens (so a legitimate reduced model — one whose
#     spec at least references the protocol concepts — does not trip)
#
# Severity: this is reject-level (per the deception_regression fixture
# `readme_spec_misalignment` expected_outcome). The artifact misrepresents
# what it models.
PROTOCOL_CLAIM_MARKERS = (
    "two-phase commit",
    "two phase commit",
    "2pc",
    "paxos",
    "raft",
    "consensus protocol",
    "consensus algorithm",
    "leader election",
    "byzantine",
    "transaction processing",
    "transaction commit",
    "atomic commit",
    "prepare-phase",
    "prepare phase",
    "commit-phase",
    "commit phase",
    "coordinator and resource manager",
    "resource manager",
    "distributed commit",
)
# Vocabulary that, if present anywhere in the .tla file (variables,
# actions, comments), suggests the spec at least references the claimed
# protocol — legitimate reduced models score themselves out of the gate.
PROTOCOL_VOCABULARY_TOKENS = (
    # Distinctive protocol vocabulary — tokens generic to many spec
    # styles ("round", "phase", "step") are intentionally excluded so a
    # round-counter spec masquerading as 2PC does not score itself out
    # of the gate via incidental "round" presence.
    "coord", "manager", "prepare", "commit", "abort", "decided",
    "leader", "follower", "candidate", "consensus", "paxos", "raft",
    "byzantine", "txn", "transaction", "ballot", "quorum",
    "vote", "tally", "elect",
)


def _check_readme_spec_misalignment(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    readme = _joined_explanatory_text(bundle_root)
    if not readme.strip():
        return findings
    lowered_readme = readme.lower()
    matched_markers = [m for m in PROTOCOL_CLAIM_MARKERS if m in lowered_readme]
    if len(matched_markers) < 2:
        return findings
    for rel_path, raw_text in _collect_tla_texts(bundle_root):
        # Only score actual spec modules: must be a `.tla` file with at
        # least one state variable. Excludes TLC `.cfg` files and helper
        # `.tla` files (Utils, Digraph, MC*) that have zero state — they
        # are not the spec being judged.
        if not rel_path.lower().endswith(".tla"):
            continue
        text = _strip_tla_comments(raw_text)
        variables = _extract_state_variables(text)
        actions = _extract_action_names(text)
        if len(variables) == 0:
            continue
        if len(variables) > 2 or len(actions) > 2:
            continue
        lowered_spec = raw_text.lower()
        spec_protocol_hits = sum(
            1 for tok in PROTOCOL_VOCABULARY_TOKENS if tok in lowered_spec
        )
        if spec_protocol_hits == 0:
            findings.append(
                f"{rel_path}: README claims a {matched_markers[0]} protocol "
                f"({', '.join(matched_markers[:3])}) but the spec has only "
                f"{len(variables)} variable(s) / {len(actions)} action(s) and "
                f"contains zero protocol vocabulary tokens"
            )
    return findings


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
    fidelity_hit = "source_grounding.json" in lowered or "authoritative source tier" in lowered or "overreach the excerpt" in lowered
    comparable_support_hit = "repo_comparables.json" in lowered or "comparable entries" in lowered or "similarity" in lowered
    adversarial_hit = "adversarial check" in lowered
    scores = {
        "formal_validity_score": 3,
        "external_admissibility_score": 3,
        "semantic_adequacy_score": 3,
        "source_grounding_score": 3,
        "source_fidelity_score": 3,
        "outsider_comprehensibility_score": 3,
        "invariant_non_vacuity_score": 3,
        "semantics_vs_bounds_clarity_score": 3,
        "repository_fit_plausibility_score": 3,
        "repo_comparables_evidence_score": 3,
        "repo_comparable_support_score": 3,
        "reviewer_rejection_risk_score": 0,
    }
    vacuous_membership_count = sum("is vacuously satisfied" in item.lower() for item in findings)
    if tautology_count:
        scores["formal_validity_score"] -= 2
        scores["invariant_non_vacuity_score"] -= 3
        scores["reviewer_rejection_risk_score"] += 1
    elif vacuous_membership_count:
        # Pass-20 gap (3/4): well-formed-but-uninformative invariant.
        # Lower severity than `TRUE`/`x=x`: revise, not reject.
        # Penalty stays above 0 on invariant_non_vacuity_score so the
        # decision rule (reject when iv <= 0) does not fire.
        scores["invariant_non_vacuity_score"] -= 2
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
        scores["source_fidelity_score"] -= 1
        scores["semantic_adequacy_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if comparables_hit:
        scores["repo_comparables_evidence_score"] -= 2
        scores["repo_comparable_support_score"] -= 1
        scores["repository_fit_plausibility_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if fidelity_hit:
        scores["source_fidelity_score"] -= 2
        scores["source_grounding_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if comparable_support_hit:
        scores["repo_comparable_support_score"] -= 2
        scores["repo_comparables_evidence_score"] -= 1
        scores["repository_fit_plausibility_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if adversarial_hit:
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if readme_missing:
        scores["semantic_adequacy_score"] -= 1
        scores["source_grounding_score"] -= 1
        scores["source_fidelity_score"] -= 1
        scores["outsider_comprehensibility_score"] -= 1
        scores["external_admissibility_score"] -= 1
    # Pass-20 gap (4/4): README/spec misalignment finding signals the
    # spec misrepresents what it models — reject-tier. Zero out the
    # semantic-adequacy and external-admissibility scores; max out
    # rejection risk.
    misalignment_hit = (
        "zero protocol vocabulary tokens" in lowered
        or ("readme claims" in lowered and "but the spec has only" in lowered)
    )
    if misalignment_hit:
        scores["semantic_adequacy_score"] = 0
        scores["external_admissibility_score"] = 0
        scores["formal_validity_score"] -= 2
        scores["source_grounding_score"] -= 2
        scores["reviewer_rejection_risk_score"] += 3
    for key in list(scores):
        if key == "reviewer_rejection_risk_score":
            scores[key] = max(0, min(3, scores[key]))
        else:
            scores[key] = max(0, min(3, scores[key]))
    return scores


def _adversarial_evidence_findings(
    bundle_root: Path,
    source_grounding_errors: list[str],
    repo_comparables_errors: list[str],
) -> list[str]:
    findings: list[str] = []
    joined_text = _joined_explanatory_text(bundle_root).lower()
    if source_grounding_errors:
        findings.append("Adversarial check: claimed grounding does not robustly support the artifact")
    if repo_comparables_errors:
        findings.append("Adversarial check: repository-fit claim is overstated relative to the comparable set")
    if any(term in joined_text for term in PROJECT_PRIVATE_JARGON) and not any(marker in joined_text for marker in ("translated as", "means", "i.e.", "ordinary tla+", "outsider")):
        findings.append("Adversarial check: project-private theory appears to leak into public-facing prose without enough translation")
    return findings


def _gather_intrinsic_findings(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    findings.extend(_check_tautological_invariants(bundle_root))
    findings.extend(_check_negative_tests(bundle_root))
    findings.extend(_check_publication_claims(bundle_root))
    findings.extend(_check_stuttering_only_next(bundle_root))
    findings.extend(_check_variable_action_mapping(bundle_root, tolerate_missing_readme=True))
    findings.extend(_check_project_private_jargon(bundle_root))
    findings.extend(_check_visible_semantics_bounds_conflation(bundle_root))
    findings.extend(_check_repository_fit_signal(bundle_root))
    findings.extend(_check_readme_spec_misalignment(bundle_root))
    return findings


def _gather_completeness_findings(bundle_root: Path) -> list[str]:
    findings: list[str] = []
    validators = {
        "audience_translation.md": _validate_audience_translation,
        "semantics_boundary.md": _validate_semantics_boundary,
        "repo_fit_review.json": _validate_repo_fit,
        "skeptical_review.json": _validate_skeptical_review,
        "source_grounding.json": lambda path: _validate_source_grounding(path, bundle_root),
        "repo_comparables.json": lambda path: _validate_repo_comparables(path, bundle_root),
    }
    source_grounding_errors: list[str] = []
    repo_comparables_errors: list[str] = []
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
        findings.extend(artifact_errors)
        if required_name == "source_grounding.json":
            source_grounding_errors = artifact_errors
        if required_name == "repo_comparables.json":
            repo_comparables_errors = artifact_errors
    for optional_name in OPTIONAL_REVIEW_FILES:
        artifact_path = bundle_root / optional_name
        if artifact_path.exists():
            findings.extend(_validate_human_approval(artifact_path))
    findings.extend(_adversarial_evidence_findings(bundle_root, source_grounding_errors, repo_comparables_errors))
    findings.extend(_check_source_grounding(bundle_root))
    findings.extend(_check_repo_comparables_evidence(bundle_root))
    return findings


def score_intrinsic_legitimacy(bundle_root: Path) -> dict[str, Any]:
    """Pass-7 axis 1: judge the artifact from its own content.

    Does not require our submission-bundle files. Does not penalize upstream work
    for lacking Codex fixture phraseology. Produces only intrinsic + aggregate
    score dimensions.
    """
    findings = _gather_intrinsic_findings(bundle_root)
    raw_scores = _score_from_findings(findings)
    scores = {k: raw_scores[k] for k in raw_scores if k in INTRINSIC_SCORE_KEYS or k in AGGREGATE_SCORE_KEYS}
    return {
        "bundle_root": str(bundle_root),
        "findings": findings,
        "scores": scores,
        "ok": not findings,
    }


def score_submission_bundle_completeness(bundle_root: Path) -> dict[str, Any]:
    """Pass-7 axis 2: judge whether the artifact satisfies our local bundle format.

    This is the gate that still applies to our own outbound submission path.
    Upstream corpora will almost always score 'reject' here by design.
    """
    findings = _gather_completeness_findings(bundle_root)
    raw_scores = _score_from_findings(findings)
    scores = {k: raw_scores[k] for k in raw_scores if k in BUNDLE_COMPLETENESS_SCORE_KEYS or k in AGGREGATE_SCORE_KEYS}
    return {
        "bundle_root": str(bundle_root),
        "findings": findings,
        "scores": scores,
        "ok": not findings,
    }


def intrinsic_decision_from_scores(scores: dict[str, int]) -> str:
    """Decision under Pass-7 intrinsic-only axis. No bundle-completeness dimensions consulted."""
    fv = scores.get("formal_validity_score", 3)
    iv = scores.get("invariant_non_vacuity_score", 3)
    oc = scores.get("outsider_comprehensibility_score", 3)
    sa = scores.get("semantic_adequacy_score", 3)
    sb = scores.get("semantics_vs_bounds_clarity_score", 3)
    rf = scores.get("repository_fit_plausibility_score", 3)
    if fv <= 1 or iv <= 0:
        return "reject"
    if fv < 3 or iv < 3 or oc < 3 or sa < 3 or sb < 3 or rf < 3:
        return "revise"
    return "accept"


def bundle_completeness_decision_from_scores(scores: dict[str, int]) -> str:
    """Completeness-axis decision mirroring the evidence-layer portion of the
    original TLA reject criteria.

    Pre-Pass-7 behavior (tla_blind executor) rejected when
    external_admissibility hit 0 with any weak dimension, or when reviewer
    rejection risk saturated with weak evidence fidelity. Preserving that
    exact threshold here ensures deception fixtures that were reject under
    the combined monolithic scorer still reject under combined Pass-7.
    """
    sg = scores.get("source_grounding_score", 3)
    sf = scores.get("source_fidelity_score", 3)
    rce = scores.get("repo_comparables_evidence_score", 3)
    rcs = scores.get("repo_comparable_support_score", 3)
    ea = scores.get("external_admissibility_score", 3)
    rrr = scores.get("reviewer_rejection_risk_score", 0)
    if ea <= 0 and (sf <= 1 or rcs <= 1):
        return "reject"
    if rrr >= 3 and sf <= 1:
        return "reject"
    if sg <= 0 or sf <= 0:
        return "reject"
    if sg < 2 or sf < 2 or rce < 2 or rcs < 2:
        return "revise"
    return "accept"


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
            "source_grounding.json": lambda path: _validate_source_grounding(path, bundle_root),
            "repo_comparables.json": lambda path: _validate_repo_comparables(path, bundle_root),
        }
        source_grounding_errors: list[str] = []
        repo_comparables_errors: list[str] = []
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
            if required_name == "source_grounding.json":
                source_grounding_errors = artifact_errors
            if required_name == "repo_comparables.json":
                repo_comparables_errors = artifact_errors
        for optional_name in OPTIONAL_REVIEW_FILES:
            artifact_path = bundle_root / optional_name
            if artifact_path.exists():
                artifact_errors = _validate_human_approval(artifact_path)
                report["required_artifacts"][optional_name] = {
                    "present": True,
                    "errors": artifact_errors,
                }
                findings.extend(artifact_errors)
        findings.extend(_adversarial_evidence_findings(bundle_root, source_grounding_errors, repo_comparables_errors))
    findings.extend(_check_tautological_invariants(bundle_root))
    findings.extend(_check_negative_tests(bundle_root))
    findings.extend(_check_publication_claims(bundle_root))
    findings.extend(_check_stuttering_only_next(bundle_root))
    findings.extend(_check_variable_action_mapping(bundle_root))
    findings.extend(_check_project_private_jargon(bundle_root))
    findings.extend(_check_visible_semantics_bounds_conflation(bundle_root))
    findings.extend(_check_repository_fit_signal(bundle_root))
    findings.extend(_check_readme_spec_misalignment(bundle_root))
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
