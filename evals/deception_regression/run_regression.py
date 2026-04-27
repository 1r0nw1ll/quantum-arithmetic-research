#!/usr/bin/env python3
# noqa: DECL-1 (regression harness — not empirical QA code)
"""
Pass-7 deception regression suite.

Sweeps every TLA+ and Lean 4 fixture (existing blind-eval suites + new
fixtures under evals/deception_regression/fixtures/) through Pass-7 scoring.

Policy
------
For each fixture, inspect whether the submission bundle is actually present:

- TLA+: bundle present iff all 6 REQUIRED_FILES exist alongside the artifact
- Lean 4: bundle present iff README.md + source_grounding.json + theorem_fit_review.json all exist

If bundle present → decision = worst(intrinsic, completeness).
If bundle absent → decision = intrinsic (completeness axis not applicable).

This matches legacy behavior: the old TLA executor called
score_bundle(..., require_artifacts=False) for review/repair cases that
didn't ship a bundle, and True only for generation cases.

Verdict classification
----------------------
For each fixture with a hidden expected_decision:

- MATCH            — pass-7 decision == expected
- NEW_FALSE_ACCEPT — pass-7 < expected severity AND no `known_gap` tag
- NEW_FALSE_REJECT — pass-7 > expected severity
- KNOWN_GAP_TOLERATED — pass-7 < expected severity BUT fixture documents
                       a `known_gap` the pass-7 scorer is known to miss

Exit code: 0 iff zero NEW_FALSE_ACCEPT and zero NEW_FALSE_REJECT. Known gaps
do not fail the suite — they document what Pass 7 deliberately does not catch,
so that tightening proposals can be scoped from data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import qa_formal_publication_gate as tla_gate  # noqa: E402
from evals._blind_core import (  # noqa: E402
    ORDER,
    bundle_present,
    load_expected as _shared_load_expected,
    worst_of,
)


def _load_lean_scorer():
    path = REPO_ROOT / "evals" / "lean4_blind" / "execute_current_system.py"
    spec = importlib.util.spec_from_file_location("lean4_blind_executor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Per-domain required-bundle file lists. The bundle-presence check itself
# is now `bundle_present` from `evals._blind_core` (Pass 9 extraction);
# only the required-files tuples stay domain-local.
TLA_BUNDLE_FILES = (
    "audience_translation.md",
    "semantics_boundary.md",
    "repo_fit_review.json",
    "skeptical_review.json",
    "source_grounding.json",
    "repo_comparables.json",
)
LEAN_BUNDLE_FILES = ("README.md", "source_grounding.json", "theorem_fit_review.json")

# Backwards-compatible aliases for the helpers now living in _blind_core.
_worst = worst_of
_load_expected = _shared_load_expected


def _tla_bundle_present(bundle_root: Path) -> bool:
    return bundle_present(bundle_root, TLA_BUNDLE_FILES)


def _lean_bundle_present(bundle_root: Path) -> bool:
    return bundle_present(bundle_root, LEAN_BUNDLE_FILES)


def _score_tla_case(bundle_root: Path) -> dict[str, Any]:
    intr = tla_gate.score_intrinsic_legitimacy(bundle_root)
    i_dec = tla_gate.intrinsic_decision_from_scores(intr["scores"])
    if _tla_bundle_present(bundle_root):
        comp = tla_gate.score_submission_bundle_completeness(bundle_root)
        c_dec = tla_gate.bundle_completeness_decision_from_scores(comp["scores"])
        combined = _worst(i_dec, c_dec)
        return {
            "intrinsic_decision": i_dec,
            "completeness_decision": c_dec,
            "combined_decision": combined,
            "policy": "combined",
            "intrinsic_findings": intr["findings"],
            "completeness_findings": comp["findings"],
        }
    return {
        "intrinsic_decision": i_dec,
        "completeness_decision": None,
        "combined_decision": i_dec,
        "policy": "intrinsic_only",
        "intrinsic_findings": intr["findings"],
        "completeness_findings": [],
    }


def _score_lean_case(bundle_root: Path, lean_mod) -> dict[str, Any]:
    intr = lean_mod._score_intrinsic_legitimacy(bundle_root)
    i_dec = lean_mod._intrinsic_decision_from_scores(intr["scores"])
    if _lean_bundle_present(bundle_root):
        comp = lean_mod._score_submission_bundle_completeness(bundle_root)
        c_dec = lean_mod._completeness_decision_from_scores(comp["scores"])
        combined = _worst(i_dec, c_dec)
        return {
            "intrinsic_decision": i_dec,
            "completeness_decision": c_dec,
            "combined_decision": combined,
            "policy": "combined",
            "intrinsic_findings": intr["errors"],
            "completeness_findings": comp["errors"],
        }
    return {
        "intrinsic_decision": i_dec,
        "completeness_decision": None,
        "combined_decision": i_dec,
        "policy": "intrinsic_only",
        "intrinsic_findings": intr["errors"],
        "completeness_findings": [],
    }


def _classify(expected: str, combined: str, known_gap: str | None) -> str:
    if combined == expected:
        return "MATCH"
    if ORDER[combined] < ORDER[expected]:
        return "KNOWN_GAP_TOLERATED" if known_gap else "NEW_FALSE_ACCEPT"
    return "NEW_FALSE_REJECT"


def _load_upwork_scorer():
    path = REPO_ROOT / "evals" / "upwork_blind" / "execute_current_system.py"
    spec = importlib.util.spec_from_file_location("upwork_blind_executor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_expected_upwork(case_dir: Path) -> tuple[str | None, str | None]:
    """Upwork uses expected_outcome.json/decision OR expected_scorecard.json/decision."""
    return _load_expected(case_dir)


def _load_case_manifest(case_dir: Path) -> dict[str, Any]:
    """Read the Upwork case.json to pull task_spec keywords for requirement-coverage scoring."""
    p = case_dir / "case.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _score_upwork_case(bundle_root: Path, upwork_mod, task_spec: dict[str, Any] | None) -> dict[str, Any]:
    report = upwork_mod._score_bundle(bundle_root, task_spec=task_spec)
    decision = upwork_mod._decision_from_scores(report["scores"])
    return {
        "intrinsic_decision": decision,
        "completeness_decision": None,
        "combined_decision": decision,
        "policy": "upwork_monolithic",
        "intrinsic_findings": report["errors"],
        "completeness_findings": [],
    }


def _sweep() -> list[dict[str, Any]]:
    lean_mod = _load_lean_scorer()
    upwork_mod = _load_upwork_scorer()
    rows: list[dict[str, Any]] = []

    tla_roots = [
        (Path("evals/tla_blind/review_corpus"), "artifact", "tla_blind_review"),
        (Path("evals/tla_blind/repair_cases"), "input", "tla_blind_repair"),
        (Path("evals/tla_blind/deception_corpus"), "artifact", "tla_blind_deception"),
        (Path("evals/deception_regression/fixtures/tla"), "artifact", "pass7_deception_tla"),
    ]
    lean_roots = [
        (Path("evals/lean4_blind/review_corpus"), "artifact", "lean4_blind_review"),
        (Path("evals/lean4_blind/repair_cases"), "input", "lean4_blind_repair"),
        (Path("evals/deception_regression/fixtures/lean"), "artifact", "pass7_deception_lean"),
    ]
    upwork_roots = [
        (Path("evals/upwork_blind/review_corpus"), "artifact", "upwork_blind_review"),
        (Path("evals/upwork_blind/repair_cases"), "input", "upwork_blind_repair"),
        (Path("evals/upwork_blind/deception_corpus"), "artifact", "upwork_blind_deception"),
    ]
    for root, sub, suite in tla_roots:
        if not root.exists():
            continue
        for case in sorted(root.iterdir()):
            if not case.is_dir():
                continue
            bundle = case / sub
            if not bundle.exists():
                continue
            expected, known_gap = _load_expected(case)
            scored = _score_tla_case(bundle)
            verdict = _classify(expected, scored["combined_decision"], known_gap) if expected else "NO_LABEL"
            rows.append({
                "domain": "tla",
                "suite": suite,
                "case": case.name,
                "expected_decision": expected,
                "known_gap": known_gap,
                "verdict": verdict,
                **scored,
            })
    for root, sub, suite in lean_roots:
        if not root.exists():
            continue
        for case in sorted(root.iterdir()):
            if not case.is_dir():
                continue
            bundle = case / sub
            if not bundle.exists():
                continue
            expected, known_gap = _load_expected(case)
            scored = _score_lean_case(bundle, lean_mod)
            verdict = _classify(expected, scored["combined_decision"], known_gap) if expected else "NO_LABEL"
            rows.append({
                "domain": "lean4",
                "suite": suite,
                "case": case.name,
                "expected_decision": expected,
                "known_gap": known_gap,
                "verdict": verdict,
                **scored,
            })
    for root, sub, suite in upwork_roots:
        if not root.exists():
            continue
        for case in sorted(root.iterdir()):
            if not case.is_dir():
                continue
            bundle = case / sub
            if not bundle.exists():
                continue
            expected, known_gap = _load_expected_upwork(case)
            task_spec = _load_case_manifest(case)
            scored = _score_upwork_case(bundle, upwork_mod, task_spec)
            verdict = _classify(expected, scored["combined_decision"], known_gap) if expected else "NO_LABEL"
            rows.append({
                "domain": "upwork",
                "suite": suite,
                "case": case.name,
                "expected_decision": expected,
                "known_gap": known_gap,
                "verdict": verdict,
                **scored,
            })
    return rows


def _render_md(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# Pass-7 Deception Regression Suite", ""]
    lines.append("## Summary")
    lines.append(f"- Total fixtures: {summary['total']}")
    lines.append(f"- MATCH: {summary['match']}")
    lines.append(f"- NEW_FALSE_ACCEPT: {summary['new_false_accept']} (precision breakage)")
    lines.append(f"- NEW_FALSE_REJECT: {summary['new_false_reject']} (recall breakage)")
    lines.append(f"- KNOWN_GAP_TOLERATED: {summary['known_gap']} (documented blind spots)")
    lines.append(f"- NO_LABEL: {summary['no_label']}")
    lines.append(f"- **Exit code: {summary['exit_code']}** (0 = clean, non-zero = precision/recall regression)")
    lines.append("")
    lines.append("## Per-fixture verdict")
    by_verdict: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_verdict.setdefault(r["verdict"], []).append(r)
    for verdict in ("NEW_FALSE_ACCEPT", "NEW_FALSE_REJECT", "KNOWN_GAP_TOLERATED", "MATCH", "NO_LABEL"):
        bucket = by_verdict.get(verdict, [])
        if not bucket:
            continue
        lines.append(f"### {verdict} ({len(bucket)})")
        for r in bucket:
            tag = f" _(known_gap: {r['known_gap']})_" if r.get("known_gap") else ""
            lines.append(
                f"- [{r['domain']}/{r['suite']}] `{r['case']}`: "
                f"expected=`{r['expected_decision']}` combined=`{r['combined_decision']}` "
                f"(policy={r['policy']}, intrinsic={r['intrinsic_decision']}, "
                f"completeness={r['completeness_decision']}){tag}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _sweep()
    summary = {
        "total": len(rows),
        "match": sum(1 for r in rows if r["verdict"] == "MATCH"),
        "new_false_accept": sum(1 for r in rows if r["verdict"] == "NEW_FALSE_ACCEPT"),
        "new_false_reject": sum(1 for r in rows if r["verdict"] == "NEW_FALSE_REJECT"),
        "known_gap": sum(1 for r in rows if r["verdict"] == "KNOWN_GAP_TOLERATED"),
        "no_label": sum(1 for r in rows if r["verdict"] == "NO_LABEL"),
    }
    exit_code = 0 if (summary["new_false_accept"] == 0 and summary["new_false_reject"] == 0) else 1
    summary["exit_code"] = exit_code

    payload = {"summary": summary, "rows": rows}
    (RESULTS_ROOT / "deception_regression.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS_ROOT / "deception_regression.md").write_text(
        _render_md(rows, summary), encoding="utf-8"
    )
    print(json.dumps({"ok": exit_code == 0, **summary}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
