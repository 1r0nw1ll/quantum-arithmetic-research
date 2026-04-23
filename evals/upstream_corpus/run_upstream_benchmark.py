#!/usr/bin/env python3
# noqa: DECL-1 (benchmark utility — not empirical QA code)
"""
Pass (a) upstream-corpus benchmark.

Scores every upstream-approved artifact in the TLA+ (`tlaplus/Examples`) and
Lean 4 (`mathematics_in_lean` solutions) corpora with the as-is formal-
publication-gate harness. No evidence files are synthesized. The adapter
performs only file discovery and provenance tracking.

Report:
- acceptance rate on upstream approved corpus
- false reject rate on upstream approved corpus
- top rejection reasons
- finding-bucket breakdown:
    missing_required_artifact
    missing_explicit_evidence
    weak_outsider_translation
    weak_repo_fit_signal
    jargon_private_theory
    substantive_issue
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current"

TLA_ROOT = Path("/home/player2/upstream_corpora/tlaplus_examples")
LEAN_ROOT = Path("/home/player2/upstream_corpora/mathematics_in_lean")

sys.path.insert(0, str(REPO_ROOT / "tools"))
from qa_formal_publication_gate import score_bundle as score_tla_bundle  # noqa: E402


def _load_lean_scorer():
    path = REPO_ROOT / "evals" / "lean4_blind" / "execute_current_system.py"
    spec = importlib.util.spec_from_file_location("lean4_blind_executor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._score_bundle, mod._decision_from_scores


def _tla_decision_from_scores(scores: dict[str, int]) -> str:
    """Mirror of evals/tla_blind/execute_current_system.py::_decision_from_scores."""
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


def _git_head(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


# --- Finding classification -------------------------------------------------

BUCKET_RULES = [
    ("missing_required_artifact", [
        "missing required artifact",
    ]),
    ("substantive_issue", [
        "tautological",
        "stuttering-only",
        "sorry",
        "admit",
        "group-level claims for a natural-number",
        "publication-ready",
        "submittable upstream",
        "malformed-literal",
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


def _bucket_for_finding(finding: str) -> str:
    lowered = finding.lower()
    for bucket, patterns in BUCKET_RULES:
        for pat in patterns:
            if pat in lowered:
                return bucket
    return "other"


# --- Corpus discovery -------------------------------------------------------

def _discover_tla_cases() -> list[dict[str, Any]]:
    specs_root = TLA_ROOT / "specifications"
    rows = []
    for spec_dir in sorted(p for p in specs_root.iterdir() if p.is_dir()):
        tla_files = sorted(spec_dir.glob("*.tla"))
        if not tla_files:
            continue
        manifest_path = spec_dir / "manifest.json"
        tags: list[str] = []
        if manifest_path.exists():
            try:
                tags = json.loads(manifest_path.read_text())["tags"]
            except Exception:
                tags = []
        rows.append({
            "case_id": spec_dir.name,
            "domain": "tla",
            "bundle_root": str(spec_dir),
            "source_repo": "tlaplus/Examples",
            "source_path": str(spec_dir.relative_to(TLA_ROOT)),
            "tla_files": [p.name for p in tla_files],
            "has_readme": any((spec_dir / n).exists() for n in ("README", "README.md")),
            "has_manifest": manifest_path.exists(),
            "manifest_tags": tags,
            "expected_decision": "accept",
        })
    return rows


def _discover_lean_cases() -> list[dict[str, Any]]:
    rows = []
    for lean_path in sorted(LEAN_ROOT.rglob("Solutions_*.lean")):
        parent = lean_path.parent
        rows.append({
            "case_id": lean_path.stem,
            "domain": "lean4",
            "bundle_root": str(parent),
            "lean_file": lean_path.name,
            "source_repo": "leanprover-community/mathematics_in_lean",
            "source_path": str(lean_path.relative_to(LEAN_ROOT)),
            "has_readme": (parent / "README.md").exists(),
            "expected_decision": "accept",
        })
    return rows


# --- Scoring ---------------------------------------------------------------

def _score_tla_case(case: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(case["bundle_root"])
    report = score_tla_bundle(bundle_root, require_artifacts=True)
    decision = _tla_decision_from_scores(report["scores"])
    return {
        "case_id": case["case_id"],
        "domain": "tla",
        "source_repo": case["source_repo"],
        "source_path": case["source_path"],
        "expected_decision": case["expected_decision"],
        "decision": decision,
        "scores": report["scores"],
        "errors": report["errors"],
        "finding_buckets": [_bucket_for_finding(f) for f in report["errors"]],
    }


def _score_lean_case(case: dict[str, Any], scorer, decider) -> dict[str, Any]:
    bundle_root = Path(case["bundle_root"])
    report = scorer(bundle_root)
    decision = decider(report["scores"])
    return {
        "case_id": case["case_id"],
        "domain": "lean4",
        "source_repo": case["source_repo"],
        "source_path": case["source_path"],
        "expected_decision": case["expected_decision"],
        "decision": decision,
        "scores": report["scores"],
        "errors": report["errors"],
        "finding_buckets": [_bucket_for_finding(f) for f in report["errors"]],
    }


# --- Aggregation -----------------------------------------------------------

def _summarize_domain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    accept = sum(1 for r in rows if r["decision"] == "accept")
    revise = sum(1 for r in rows if r["decision"] == "revise")
    reject = sum(1 for r in rows if r["decision"] == "reject")
    false_rejects = [r for r in rows if r["decision"] == "reject" and r["expected_decision"] == "accept"]
    non_accept_predicts = [r for r in rows if r["decision"] != "accept"]

    bucket_counter: Counter[str] = Counter()
    cases_with_bucket: dict[str, set[str]] = {}
    for row in rows:
        seen_in_case: set[str] = set()
        for bucket in row["finding_buckets"]:
            bucket_counter[bucket] += 1
            seen_in_case.add(bucket)
        for bucket in seen_in_case:
            cases_with_bucket.setdefault(bucket, set()).add(row["case_id"])

    top_finding_strings = Counter()
    for row in rows:
        for f in row["errors"]:
            top_finding_strings[f] += 1

    # False reject decomposition: rejections driven purely by missing-artifact
    # vs rejections with substantive/other content findings mixed in.
    purely_missing = 0
    mixed = 0
    substantive_only = 0
    for row in false_rejects:
        buckets = set(row["finding_buckets"])
        if buckets == {"missing_required_artifact"}:
            purely_missing += 1
        elif "substantive_issue" in buckets and "missing_required_artifact" not in buckets:
            substantive_only += 1
        else:
            mixed += 1

    return {
        "total_cases": n,
        "decisions": {"accept": accept, "revise": revise, "reject": reject},
        "acceptance_rate": round(accept / n, 4) if n else 0.0,
        "false_reject_rate": round(len(false_rejects) / n, 4) if n else 0.0,
        "nonaccept_rate": round(len(non_accept_predicts) / n, 4) if n else 0.0,
        "finding_bucket_totals": dict(bucket_counter.most_common()),
        "cases_touched_by_bucket": {k: len(v) for k, v in cases_with_bucket.items()},
        "false_reject_decomposition": {
            "total_false_rejects": len(false_rejects),
            "purely_missing_required_artifact": purely_missing,
            "substantive_findings_only": substantive_only,
            "mixed_or_other": mixed,
        },
        "top_finding_strings": [
            {"finding": f, "count": c, "bucket": _bucket_for_finding(f)}
            for f, c in top_finding_strings.most_common(20)
        ],
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = ["# Upstream-Approved-Corpus Benchmark (Pass a, as-is)", ""]
    lines.append("## Provenance")
    for key, p in payload["provenance"].items():
        lines.append(f"- **{key}**: `{p['repo']}` @ `{p['sha']}`")
    lines.append("")
    for domain_key, summary in payload["domains"].items():
        lines.append(f"## {domain_key.upper()}")
        lines.append(f"- Total cases: {summary['total_cases']}")
        lines.append(f"- Decisions: {summary['decisions']}")
        lines.append(f"- **Acceptance rate on upstream approved corpus: {summary['acceptance_rate']:.2%}**")
        lines.append(f"- **False reject rate: {summary['false_reject_rate']:.2%}**")
        lines.append(f"- Non-accept rate (revise+reject): {summary['nonaccept_rate']:.2%}")
        lines.append("")
        lines.append("### Finding-bucket totals (count = total occurrences)")
        for bucket, count in summary["finding_bucket_totals"].items():
            touched = summary["cases_touched_by_bucket"].get(bucket, 0)
            lines.append(f"- `{bucket}`: {count} findings across {touched} cases")
        lines.append("")
        fr = summary["false_reject_decomposition"]
        lines.append("### False reject decomposition")
        lines.append(f"- Total false rejects: {fr['total_false_rejects']}")
        lines.append(f"- Driven purely by missing-required-artifact: {fr['purely_missing_required_artifact']}")
        lines.append(f"- Driven by substantive findings only (no missing-artifact): {fr['substantive_findings_only']}")
        lines.append(f"- Mixed / other: {fr['mixed_or_other']}")
        lines.append("")
        lines.append("### Top finding strings (verbatim, top 20)")
        for entry in summary["top_finding_strings"]:
            lines.append(f"- [{entry['bucket']}] `{entry['finding']}` — {entry['count']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # TLA+
    tla_cases = _discover_tla_cases()
    tla_rows = [_score_tla_case(c) for c in tla_cases]

    # Lean 4 (importlib-loaded to avoid copying ~80 lines of scoring logic)
    lean_scorer, lean_decider = _load_lean_scorer()
    lean_cases = _discover_lean_cases()
    lean_rows = [_score_lean_case(c, lean_scorer, lean_decider) for c in lean_cases]

    payload = {
        "provenance": {
            "tla": {"repo": "tlaplus/Examples", "sha": _git_head(TLA_ROOT)},
            "lean4": {"repo": "leanprover-community/mathematics_in_lean", "sha": _git_head(LEAN_ROOT)},
        },
        "inventory": {
            "tla_case_count": len(tla_cases),
            "lean4_case_count": len(lean_cases),
        },
        "domains": {
            "tla": _summarize_domain(tla_rows),
            "lean4": _summarize_domain(lean_rows),
        },
        "rows": {
            "tla": tla_rows,
            "lean4": lean_rows,
        },
        "cases": {
            "tla": tla_cases,
            "lean4": lean_cases,
        },
    }

    json_path = RESULTS_ROOT / "upstream_benchmark.json"
    md_path = RESULTS_ROOT / "upstream_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "tla_cases": len(tla_cases),
        "lean4_cases": len(lean_cases),
        "json_report": str(json_path.relative_to(REPO_ROOT)),
        "markdown_report": str(md_path.relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
