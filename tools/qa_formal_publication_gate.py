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


def validate_bundle(bundle_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "bundle_root": str(bundle_root),
        "ok": True,
        "errors": [],
        "required_artifacts": {},
    }
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
        report["errors"].extend(artifact_errors)
    report["errors"].extend(_check_tautological_invariants(bundle_root))
    report["errors"].extend(_check_negative_tests(bundle_root))
    report["errors"].extend(_check_publication_claims(bundle_root))
    report["ok"] = not report["errors"]
    return report


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
        "formal_validity_score": 3 if ok else 0,
        "external_admissibility_score": 3 if ok else 0,
    }
    if args.json:
        print(_json_dumps(payload))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
