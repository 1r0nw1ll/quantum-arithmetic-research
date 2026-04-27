#!/usr/bin/env python3
# noqa: DECL-1 (shared eval infrastructure — not empirical QA code)
"""
Shared cross-domain core for the blind-eval suites.

This module is the result of Pass 9 (behavior-preserving extraction). It
collects the small set of utilities that were demonstrably duplicated
across the TLA+, Lean 4, and Upwork-style blind suites:

- decision-severity ordering and worst-of combination
- finding-bucket classification (the rule list grew incrementally as
  TLA, then Lean, then Upwork shipped — it is now a shared dictionary)
- hidden-label loading from `hidden_label/expected_*.json`
- bundle-presence detection given a per-domain required-files tuple

Domain-specific scoring stays out of here on purpose. Each domain still
owns its own:
- score axis names + axis semantics
- finding-emitting heuristics
- decision-from-scores helper (thresholds are domain-local)
- required bundle file list (Codex bundle for TLA; README + grounding +
  fit-review for Lean; not currently defined for Upwork)

Pass 9 acceptance criterion: behavior preserved. After migration, every
suite's outputs must be byte-identical to the pre-extraction baseline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# --- Decision-severity ordering ---------------------------------------------

ORDER: dict[str, int] = {"accept": 0, "revise": 1, "reject": 2}


def worst_of(a: str, b: str) -> str:
    """Return the worse of two decisions under ORDER (reject > revise > accept)."""
    return a if ORDER[a] >= ORDER[b] else b


def combined_decision(intrinsic: str, completeness: str | None) -> str:
    """Bundle-aware combination.

    If `completeness` is None (no bundle present, or domain has no completeness
    axis), the combined decision is just the intrinsic decision. Otherwise
    return the worse of the two.
    """
    if completeness is None:
        return intrinsic
    return worst_of(intrinsic, completeness)


# --- Finding-bucket classifier ----------------------------------------------
#
# Bucket rules are evaluated in order; the first matching bucket wins, which
# means more-specific buckets must come before more-general ones. The rules
# below were grown incrementally — TLA contributed the missing-artifact /
# explicit-evidence / outsider-translation patterns; Lean added pedagogical
# vs deceptive sorry; Upwork added the operational-deception patterns
# (overclaim, fake tests, requirement drop-out).

BUCKET_RULES: list[tuple[str, list[str]]] = [
    ("missing_required_artifact", [
        "missing required artifact",
    ]),
    ("substantive_issue", [
        "tautological",
        "stuttering-only",
        "deceptive sorry",
        "pedagogical sorry",
        "group-level claims for a natural-number",
        "scope creep",
        "publication-ready",
        "submittable upstream",
        "malformed-literal",
        "overclaims source support",
    ]),
    ("jargon_private_theory", [
        "jargon",
        "project-private",
    ]),
    ("missing_explicit_evidence", [
        "source_grounding",
        "source excerpt",
        "authoritative source tier",
        "overreach the excerpt",
        "grounding packet",
        "where the semantics come from",
        "what is being modeled",
        "justify the chosen variables/actions",
        "explicit excerpts",
        "adversarial check: claimed grounding",
        "no explanatory text available for source grounding",
    ]),
    ("weak_repo_fit_signal", [
        "repository-fit",
        "repository fit",
        "repo_comparables",
        "comparable evidence",
        "comparable entries",
        "similarity",
        "worth external review",
        "maintainer value",
    ]),
    ("weak_outsider_translation", [
        "audience",
        "outsider",
        "does not explain all state variables",
        "does not map action names",
        "conflates intrinsic semantics with tlc bounds",
        "readme explanation missing",
        "theorem-statement fidelity",
        "proof idea",
        "math_explanation",
        "does not explicitly state",
    ]),
]


def bucket_for_finding(finding: str) -> str:
    lowered = finding.lower()
    for bucket, patterns in BUCKET_RULES:
        for pat in patterns:
            if pat in lowered:
                return bucket
    return "other"


# --- Hidden-label loading ---------------------------------------------------

_HIDDEN_LABEL_FILENAMES = (
    "hidden_label/expected_scorecard.json",
    "hidden_label/expected_outcome.json",
)


def load_expected(case_dir: Path) -> tuple[str | None, str | None]:
    """Return (expected_decision, known_gap_tag) from the case's hidden_label/.

    Accepts both legacy `expected_scorecard.json` (review fixtures) and
    `expected_outcome.json` (repair/deception fixtures), and both the
    `decision` and legacy `final_decision` keys. Returns (None, None) if no
    label is present (e.g., generation cases).
    """
    for rel in _HIDDEN_LABEL_FILENAMES:
        p = case_dir / rel
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, None
            return d.get("decision") or d.get("final_decision"), d.get("known_gap")
    return None, None


# --- Bundle-presence detection ----------------------------------------------

def bundle_present(bundle_root: Path, required_files: Iterable[str]) -> bool:
    """True iff every file in `required_files` exists directly under `bundle_root`."""
    return all((bundle_root / name).exists() for name in required_files)


# --- Self-test --------------------------------------------------------------

def _self_test() -> dict[str, Any]:
    """Sanity check the shared utilities against representative inputs from
    each domain. Returns a JSON-serializable summary."""
    checks: list[tuple[str, bool, str]] = []

    # Decision ordering
    checks.append(("worst_of accept/revise -> revise", worst_of("accept", "revise") == "revise", "ordering"))
    checks.append(("worst_of revise/reject -> reject", worst_of("revise", "reject") == "reject", "ordering"))
    checks.append(("combined None completeness -> intrinsic", combined_decision("revise", None) == "revise", "ordering"))
    checks.append(("combined accept/reject -> reject", combined_decision("accept", "reject") == "reject", "ordering"))

    # Bucket classifier
    # Sample only findings the rule list was originally designed for (TLA +
    # Lean upstream paths). Upwork-specific findings land in "other" because
    # the bucket list is intentionally not extended in this extraction pass —
    # behavior is preserved exactly as it was in run_upstream_benchmark.py.
    samples = {
        "missing required artifact audience_translation.md": "missing_required_artifact",
        "Counter:Inv_S1 looks tautological": "substantive_issue",
        "Lean proof contains 1 deceptive sorry/admit (theorem proof term is sorry)": "substantive_issue",
        "Lean artifact contains 5 pedagogical sorry (structure-instance field stub)": "substantive_issue",
        "Adversarial check: claimed grounding does not robustly support the artifact": "missing_explicit_evidence",
        "README does not map action names into outsider-facing prose": "weak_outsider_translation",
        "Repository-fit claim lacks comparable evidence": "weak_repo_fit_signal",
        "weird unmatched finding here": "other",
    }
    for finding, expected_bucket in samples.items():
        actual = bucket_for_finding(finding)
        checks.append((f"bucket: {finding[:60]}...", actual == expected_bucket, f"got {actual}, want {expected_bucket}"))

    failures = [c for c in checks if not c[1]]
    return {
        "ok": not failures,
        "total_checks": len(checks),
        "failures": [{"check": c[0], "detail": c[2]} for c in failures],
    }


def main() -> int:
    result = _self_test()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
