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


import re as _re


_INTRINSIC_SCORE_KEYS = frozenset({
    "formal_validity_score",
    "proof_correctness_score",
    "theorem_statement_fidelity_score",
})
_BUNDLE_COMPLETENESS_SCORE_KEYS = frozenset({
    "source_fidelity_score",
    "math_explanation_quality_score",
})
_AGGREGATE_SCORE_KEYS = frozenset({
    "external_admissibility_score",
    "reviewer_rejection_risk_score",
})


def _strip_lean_comments(text: str) -> str:
    """Remove Lean block comments (including /-- ... -/ doc comments) and
    end-of-line -- comments. This prevents sorry/admit tokens in docstrings
    or comment code-blocks from being misclassified as proof terms.
    """
    text = _re.sub(r"/-.*?-/", " ", text, flags=_re.DOTALL)
    text = _re.sub(r"--.*$", " ", text, flags=_re.M)
    return text


def _classify_sorry_contexts(lean_text: str) -> tuple[int, int]:
    """Return (deceptive_count, pedagogical_count).

    Deceptive sorry: a `theorem`/`lemma`/`example` whose proof term is `sorry`
    (or whose tactic block is `by sorry`). The artifact claims a theorem is
    proved while it is not.

    Pedagogical sorry: a structure-instance field assignment like
    `field := sorry` (common in MIL / textbook exercise stubs where the
    structure skeleton is complete but selected fields are reader exercises).

    Conservative: lines we can't classify as pedagogical are counted as
    deceptive, since proof-claim overstatement is the failure mode we
    actually care about. sorry/admit inside /- ... -/ block comments (including
    /-- ... -/ doc comments) and end-of-line -- comments is stripped before
    classification — those are docstring examples, not proof terms.
    """
    lean_text = _strip_lean_comments(lean_text)
    deceptive = 0
    pedagogical = 0
    # Structure-instance field assignments like "  field_name := sorry" at
    # indentation > 0 or after `where`/record syntax. Match lines of the form
    # "<ident> := sorry" NOT preceded by "theorem "/"lemma "/"example ".
    # First find all sorry occurrences (word-bounded), then classify.
    for m in _re.finditer(r"\bsorry\b|\badmit\b", lean_text):
        line_start = lean_text.rfind("\n", 0, m.start()) + 1
        line_end = lean_text.find("\n", m.start())
        if line_end == -1:
            line_end = len(lean_text)
        line = lean_text[line_start:line_end]
        # Walk back up to 20 preceding lines to find the nearest enclosing
        # `theorem`/`lemma`/`example`/`where`/`{` header.
        header_start = max(0, line_start - 20 * 120)
        preceding = lean_text[header_start:line_start]
        # Check if sorry is a structure-field assignment: pattern "<ident> := sorry"
        # on the same line, with the ident NOT being "theorem"/"lemma"/"example".
        field_pattern = _re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_']*)\s*:=\s*(?:by\s+)?sorry\b")
        field_match = field_pattern.match(line)
        if field_match:
            ident = field_match.group(1)
            if ident in {"theorem", "lemma", "example", "def"}:
                deceptive += 1
            else:
                # Confirm we're inside a structure/instance/where block by
                # checking that a `where` or `instance`/`structure` appears in
                # recent context.
                if _re.search(r"\b(where|instance|structure|class)\b", preceding):
                    pedagogical += 1
                else:
                    # free-standing `name := sorry` at top level — likely a def
                    # with a sorry body, deceptive.
                    deceptive += 1
            continue
        # `:= sorry` or `by sorry` as proof term of theorem/lemma/example/def
        if _re.search(r"\b(theorem|lemma|example|def)\b", preceding[-500:] if len(preceding) > 500 else preceding):
            deceptive += 1
        else:
            # sorry in a context we don't recognize (comment, string, macro);
            # don't count either way
            pass
    return deceptive, pedagogical


def _score_bundle(bundle_root: Path, *, lean_files: list[Path] | None = None) -> dict[str, Any]:
    """Combined Lean scorer. Kept for backward compatibility with existing callers.

    The internal Codex fixtures still expect this full combined scoring. The
    upstream-corpus benchmark should call _score_intrinsic_legitimacy or
    _score_submission_bundle_completeness for axis-separated results.
    """
    intrinsic = _score_intrinsic_legitimacy(bundle_root, lean_files=lean_files)
    completeness = _score_submission_bundle_completeness(bundle_root, lean_files=lean_files)
    findings = list(intrinsic["errors"]) + list(completeness["errors"])
    scores = {**intrinsic["scores"], **completeness["scores"]}
    # Re-reconcile aggregate scores from both axes (they each wrote their own
    # slice; prefer the more-penalized value for aggregate keys).
    scores["external_admissibility_score"] = min(
        intrinsic["scores"].get("external_admissibility_score", 3),
        completeness["scores"].get("external_admissibility_score", 3),
    )
    scores["reviewer_rejection_risk_score"] = max(
        intrinsic["scores"].get("reviewer_rejection_risk_score", 0),
        completeness["scores"].get("reviewer_rejection_risk_score", 0),
    )
    return {"ok": not findings, "errors": findings, "scores": scores}


def _read_lean_text(bundle_root: Path, lean_files: list[Path] | None) -> str:
    if lean_files is not None:
        return "\n".join(_read_text(p) for p in lean_files)
    return "\n".join(_read_text(p) for p in sorted(bundle_root.glob("*.lean")))


def _score_intrinsic_legitimacy(bundle_root: Path, *, lean_files: list[Path] | None = None) -> dict[str, Any]:
    """Pass-7 intrinsic axis for Lean artifacts.

    Judges from proof content only. No prose-phraseology requirements.
    Bundle-completeness dimensions (source_fidelity, math_explanation_quality)
    are not scored here.
    """
    findings: list[str] = []
    lean_text = _read_lean_text(bundle_root, lean_files)
    lowered_lean = lean_text.lower()

    scores = {
        "formal_validity_score": 3,
        "proof_correctness_score": 3,
        "theorem_statement_fidelity_score": 3,
        "external_admissibility_score": 3,
        "reviewer_rejection_risk_score": 0,
    }

    deceptive_sorry, pedagogical_sorry = _classify_sorry_contexts(lean_text)
    if deceptive_sorry > 0:
        findings.append(f"Lean proof contains {deceptive_sorry} deceptive sorry/admit (theorem/lemma/def proof term is sorry)")
        scores["formal_validity_score"] -= 2
        scores["proof_correctness_score"] -= 3
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 2
    if pedagogical_sorry > 0 and deceptive_sorry == 0:
        findings.append(f"Lean artifact contains {pedagogical_sorry} pedagogical sorry (structure-instance field stub); acceptable for textbook exercise corpora")
        # No score penalty. Finding is recorded for visibility only.

    # Removed the previous `group + nat` substring co-occurrence rule in
    # Pass 7: empirically it produced 19/43 false positives on MIL solutions
    # (fired whenever substrings "group" and "nat" appeared anywhere in the
    # pooled bundle text, e.g. a solutions/ directory that contains both
    # algebra chapters and number-theory chapters). The anti-pattern it was
    # trying to catch — a theorem typed over Nat whose proof imports Group
    # machinery — is already flagged by _classify_sorry_contexts (when the
    # proof is a sorry) and by the README-overclaim rule below when the prose
    # explicitly claims "wider abstraction" / "group-theoretic". A reliable
    # substance-level check would need a real Lean parser, not a substring
    # co-occurrence; deferring until that's warranted.

    # Prose overclaim detector stays intrinsic: the artifact's own README
    # making overreach claims about its scope is a content-level issue, not a
    # bundle-format one.
    prose = ""
    for rel in ("README.md", "source_grounding.json", "theorem_fit_review.json"):
        path = bundle_root / rel
        if path.exists():
            prose += "\n" + _read_text(path)
    lowered_prose = prose.lower()
    if any(token in lowered_prose for token in ("wider abstraction", "group-theoretic", "general group", "ambient algebraic identity")):
        findings.append("README overclaims source support through unnecessary abstraction")
        scores["theorem_statement_fidelity_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    # Pass-20 gap (1/4): broader scope-overclaim detector. Triggers only
    # when the README claims abstraction over algebraic-structure classes
    # (commutative monoids, rings, groups, etc.) AND the .lean source has
    # no typeclass-bracketed type variable (`[CommMonoid α]`, `[Group G]`,
    # …) — i.e. the theorem is over a concrete type but the prose
    # generalizes. Without the typeclass-presence guard this would
    # false-positive on legitimate mathlib-style theorems over abstract
    # structures.
    broad_overclaim_phrases = (
        "all commutative algebraic structures",
        "arbitrary commutative monoid",
        "commutative monoids and rings",
        "monoids and rings",
        "all algebraic structures",
        "for any algebraic structure",
        "generalizes to arbitrary",
    )
    has_typeclass_param = bool(_re.search(
        r"\[\s*(?:CommMonoid|CommGroup|CommRing|CommSemigroup|CommSemiring|"
        r"Monoid|Group|Ring|Field|Semigroup|Semiring|AddCommMonoid|AddCommGroup|"
        r"AddGroup|AddMonoid)\b",
        lean_text,
    ))
    if any(p in lowered_prose for p in broad_overclaim_phrases) and not has_typeclass_param:
        findings.append("README overclaims theorem scope to abstract algebraic structures the .lean source does not parameterize over")
        scores["theorem_statement_fidelity_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    # Pass-20 gap (2/4): vacuous-premise detector. A theorem that
    # quantifies over a known-empty type (Empty / PEmpty / False / Fin 0)
    # is vacuously satisfied. The proof is correct and the artifact is
    # well-formed, but the obligation is content-empty. Severity: revise
    # (per the fixture's expected_outcome — rejection is too harsh
    # because the proof is legitimate Lean).
    vacuous_premise_pattern = _re.compile(
        r"\b(theorem|lemma|example)\s+\w+\s*:\s*(?:[^:=]*?)"
        r"(?:∀|forall)\s+\w+(?:\s+\w+)*\s*:\s*"
        r"(Empty|PEmpty|False|Fin\s+0)\b",
        _re.MULTILINE,
    )
    if vacuous_premise_pattern.search(lean_text):
        findings.append("Theorem quantifies over an empty/uninhabited type — the obligation is vacuously satisfied")
        scores["theorem_statement_fidelity_score"] -= 1
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    for key in list(scores):
        scores[key] = max(0, min(3, scores[key]))
    return {"ok": not findings, "errors": findings, "scores": scores}


def _score_submission_bundle_completeness(bundle_root: Path, *, lean_files: list[Path] | None = None) -> dict[str, Any]:
    """Pass-7 completeness axis for Lean: Codex-specific README phraseology.

    Upstream Lean corpora are expected to fail most of these by design; that
    is not evidence against the upstream artifact's legitimacy, only against
    its fit for our local submission format.
    """
    findings: list[str] = []
    prose = ""
    for rel in ("README.md", "source_grounding.json", "theorem_fit_review.json"):
        path = bundle_root / rel
        if path.exists():
            prose += "\n" + _read_text(path)
    lowered_prose = prose.lower()

    scores = {
        "source_fidelity_score": 3,
        "math_explanation_quality_score": 3,
        "external_admissibility_score": 3,
        "reviewer_rejection_risk_score": 0,
    }

    if "theorem statement exactly matches" not in lowered_prose and "matches the task statement" not in lowered_prose:
        findings.append("README does not explicitly state theorem-statement fidelity")
        scores["math_explanation_quality_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "source excerpt" not in lowered_prose and '"source_excerpt"' not in lowered_prose:
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
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1
    if "example worth reviewing" not in lowered_prose and "maintainer value" not in lowered_prose:
        findings.append("Bundle does not justify why the proof is worth external review")
        scores["external_admissibility_score"] -= 1
        scores["reviewer_rejection_risk_score"] += 1

    for key in list(scores):
        scores[key] = max(0, min(3, scores[key]))
    return {"ok": not findings, "errors": findings, "scores": scores}


def _intrinsic_decision_from_scores(scores: dict[str, int]) -> str:
    fv = scores.get("formal_validity_score", 3)
    pc = scores.get("proof_correctness_score", 3)
    tf = scores.get("theorem_statement_fidelity_score", 3)
    if pc <= 0 or tf <= 0 or fv <= 0:
        return "reject"
    if fv < 3 or pc < 3 or tf < 3:
        return "revise"
    return "accept"


def _completeness_decision_from_scores(scores: dict[str, int]) -> str:
    sf = scores.get("source_fidelity_score", 3)
    me = scores.get("math_explanation_quality_score", 3)
    if sf <= 0:
        return "reject"
    if sf < 3 or me < 3:
        return "revise"
    return "accept"


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
