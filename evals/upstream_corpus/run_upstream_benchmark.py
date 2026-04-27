#!/usr/bin/env python3
# noqa: DECL-1 (benchmark utility — not empirical QA code)
"""
Pass-a + Pass-7 upstream-corpus benchmark.

Scores every upstream-approved artifact in the TLA+ (`tlaplus/Examples`) and
Lean 4 (`mathematics_in_lean` solutions) corpora. Each case is scored on both
axes:

- intrinsic legitimacy (does the artifact itself look legitimate?)
- submission-bundle completeness (does it satisfy our local bundle format?)

The adapter only performs file discovery and provenance tracking — it does
not synthesize evidence files. Missing local bundle files are reported as
completeness findings, not patched around.

Output:
- per-case: intrinsic decision + completeness decision + combined (worse of the two)
- per-domain: acceptance / false-reject rates under each axis
- delta: how many cases flip from reject (pass-a combined) to accept under
  intrinsic-only — i.e. how much of the pass-a 100% false-reject rate was
  bundle-dependence noise, vs deeper intrinsic overfit.
- finding-bucket breakdown per axis.
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
MATHLIB_ROOT = Path("/home/player2/upstream_corpora/mathlib4")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import qa_formal_publication_gate as tla_gate  # noqa: E402
from evals._blind_core import BUCKET_RULES, bucket_for_finding, combined_decision  # noqa: E402,F401


def _load_lean_scorer():
    path = REPO_ROOT / "evals" / "lean4_blind" / "execute_current_system.py"
    spec = importlib.util.spec_from_file_location("lean4_blind_executor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_head(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


# --- Finding classification -------------------------------------------------
#
# `BUCKET_RULES`, `bucket_for_finding`, and the worst-of-two combination
# helper now live in `evals/_blind_core` (Pass 9 extraction). The local
# aliases below preserve the original call sites without touching the rest
# of this file.

_bucket_for_finding = bucket_for_finding


def _combined_decision(intrinsic: str, completeness: str) -> str:
    """Combined submission decision: worse of the two axes.

    Used only for the 'outbound submission path' simulation — NOT for judging
    upstream artifacts, where only the intrinsic axis is meaningful.
    """
    return combined_decision(intrinsic, completeness)


# --- Corpus discovery -------------------------------------------------------

def _tla_leaf_spec_dirs(root: Path) -> list[Path]:
    """Return every directory under `root` that contains at least one .tla
    file directly (not in a subdir). This picks up both top-level spec dirs
    and nested leaves like SpecifyingSystems/HourClock, SDP_Verification/*."""
    seen: set[Path] = set()
    for tla_path in root.rglob("*.tla"):
        parent = tla_path.parent
        if parent not in seen:
            seen.add(parent)
    return sorted(seen)


def _discover_tla_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs_root = TLA_ROOT / "specifications"
    for spec_dir in _tla_leaf_spec_dirs(specs_root):
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
        rel = spec_dir.relative_to(TLA_ROOT)
        case_id = "/".join(rel.parts[1:]) if len(rel.parts) > 1 else spec_dir.name
        rows.append({
            "case_id": case_id,
            "domain": "tla",
            "bundle_root": str(spec_dir),
            "source_repo": "tlaplus/Examples",
            "source_path": str(rel),
            "tla_files": [p.name for p in tla_files],
            "has_readme": any((spec_dir / n).exists() for n in ("README", "README.md")),
            "has_manifest": manifest_path.exists(),
            "manifest_tags": tags,
            "expected_decision": "accept",
        })
    return rows


def _discover_lean_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # mathematics_in_lean: every completed Solutions_*.lean file
    for lean_path in sorted(LEAN_ROOT.rglob("Solutions_*.lean")):
        parent = lean_path.parent
        rows.append({
            "case_id": lean_path.stem,
            "domain": "lean4",
            "bundle_root": str(parent),
            "lean_file": str(lean_path),
            "source_repo": "leanprover-community/mathematics_in_lean",
            "source_path": str(lean_path.relative_to(LEAN_ROOT)),
            "has_readme": (parent / "README.md").exists(),
            "expected_decision": "accept",
        })
    # mathlib4 sample: Data/Nat basics — sparse-checkout-local subset
    if MATHLIB_ROOT.exists():
        nat_dir = MATHLIB_ROOT / "Mathlib" / "Data" / "Nat"
        mathlib_files = sorted(nat_dir.rglob("*.lean")) if nat_dir.exists() else []
        for lean_path in mathlib_files:
            rows.append({
                "case_id": f"mathlib/{lean_path.relative_to(MATHLIB_ROOT / 'Mathlib')}",
                "domain": "lean4",
                "bundle_root": str(lean_path.parent),
                "lean_file": str(lean_path),
                "source_repo": "leanprover-community/mathlib4",
                "source_path": str(lean_path.relative_to(MATHLIB_ROOT)),
                "has_readme": (lean_path.parent / "README.md").exists(),
                "expected_decision": "accept",
            })
    return rows


# --- Scoring ---------------------------------------------------------------

def _score_tla_case(case: dict[str, Any]) -> dict[str, Any]:
    bundle_root = Path(case["bundle_root"])
    intrinsic = tla_gate.score_intrinsic_legitimacy(bundle_root)
    completeness = tla_gate.score_submission_bundle_completeness(bundle_root)
    i_decision = tla_gate.intrinsic_decision_from_scores(intrinsic["scores"])
    c_decision = tla_gate.bundle_completeness_decision_from_scores(completeness["scores"])
    return {
        "case_id": case["case_id"],
        "domain": "tla",
        "source_repo": case["source_repo"],
        "source_path": case["source_path"],
        "expected_decision": case["expected_decision"],
        "intrinsic_decision": i_decision,
        "bundle_completeness_decision": c_decision,
        "combined_decision": _combined_decision(i_decision, c_decision),
        "intrinsic_scores": intrinsic["scores"],
        "intrinsic_findings": intrinsic["findings"],
        "intrinsic_finding_buckets": [_bucket_for_finding(f) for f in intrinsic["findings"]],
        "completeness_findings": completeness["findings"],
        "completeness_finding_buckets": [_bucket_for_finding(f) for f in completeness["findings"]],
    }


def _score_lean_case(case: dict[str, Any], lean_mod) -> dict[str, Any]:
    bundle_root = Path(case["bundle_root"])
    lean_file = Path(case["lean_file"])
    intrinsic = lean_mod._score_intrinsic_legitimacy(bundle_root, lean_files=[lean_file])
    completeness = lean_mod._score_submission_bundle_completeness(bundle_root, lean_files=[lean_file])
    i_decision = lean_mod._intrinsic_decision_from_scores(intrinsic["scores"])
    c_decision = lean_mod._completeness_decision_from_scores(completeness["scores"])
    return {
        "case_id": case["case_id"],
        "domain": "lean4",
        "source_repo": case["source_repo"],
        "source_path": case["source_path"],
        "expected_decision": case["expected_decision"],
        "intrinsic_decision": i_decision,
        "bundle_completeness_decision": c_decision,
        "combined_decision": _combined_decision(i_decision, c_decision),
        "intrinsic_scores": intrinsic["scores"],
        "intrinsic_findings": intrinsic["errors"],
        "intrinsic_finding_buckets": [_bucket_for_finding(f) for f in intrinsic["errors"]],
        "completeness_findings": completeness["errors"],
        "completeness_finding_buckets": [_bucket_for_finding(f) for f in completeness["errors"]],
    }


# --- Aggregation -----------------------------------------------------------

def _axis_summary(rows: list[dict[str, Any]], decision_key: str, findings_key: str, buckets_key: str) -> dict[str, Any]:
    n = len(rows)
    accept = sum(1 for r in rows if r[decision_key] == "accept")
    revise = sum(1 for r in rows if r[decision_key] == "revise")
    reject = sum(1 for r in rows if r[decision_key] == "reject")
    false_rejects = [r for r in rows if r[decision_key] == "reject" and r["expected_decision"] == "accept"]

    bucket_counter: Counter[str] = Counter()
    cases_with_bucket: dict[str, set[str]] = {}
    top_findings = Counter()
    for row in rows:
        seen = set()
        for f in row[findings_key]:
            top_findings[f] += 1
        for bucket in row[buckets_key]:
            bucket_counter[bucket] += 1
            seen.add(bucket)
        for bucket in seen:
            cases_with_bucket.setdefault(bucket, set()).add(row["case_id"])

    return {
        "total_cases": n,
        "decisions": {"accept": accept, "revise": revise, "reject": reject},
        "acceptance_rate": round(accept / n, 4) if n else 0.0,
        "false_reject_rate": round(len(false_rejects) / n, 4) if n else 0.0,
        "finding_bucket_totals": dict(bucket_counter.most_common()),
        "cases_touched_by_bucket": {k: len(v) for k, v in cases_with_bucket.items()},
        "top_finding_strings": [
            {"finding": f, "count": c, "bucket": _bucket_for_finding(f)}
            for f, c in top_findings.most_common(15)
        ],
    }


def _domain_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    intrinsic = _axis_summary(rows, "intrinsic_decision", "intrinsic_findings", "intrinsic_finding_buckets")
    completeness = _axis_summary(rows, "bundle_completeness_decision", "completeness_findings", "completeness_finding_buckets")
    combined = _axis_summary(rows, "combined_decision", "intrinsic_findings", "intrinsic_finding_buckets")

    # Pass-7 headline: how many cases flip from reject (combined) to accept
    # (intrinsic-only)? That measures how much of the pass-a 100% false-reject
    # rate was pure bundle-dependence.
    flipped = sum(
        1 for r in rows
        if r["combined_decision"] == "reject" and r["intrinsic_decision"] == "accept"
    )
    still_rejected_intrinsic = [
        r for r in rows
        if r["intrinsic_decision"] == "reject" and r["expected_decision"] == "accept"
    ]
    return {
        "total_cases": len(rows),
        "intrinsic": intrinsic,
        "bundle_completeness": completeness,
        "combined_submission_gate": combined,
        "pass7_delta": {
            "flipped_reject_to_accept_under_intrinsic_only": flipped,
            "still_reject_under_intrinsic_count": len(still_rejected_intrinsic),
            "still_reject_cases": [r["case_id"] for r in still_rejected_intrinsic],
            "interpretation": (
                f"{flipped}/{len(rows)} upstream-approved cases flip from reject (pass-a combined) "
                f"to accept under intrinsic-only scoring. The remaining "
                f"{len(still_rejected_intrinsic)} rejections are not bundle-dependence — "
                f"they are deeper intrinsic heuristic overfit."
            ),
        },
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = ["# Upstream-Approved-Corpus Benchmark (Pass 7 — intrinsic vs bundle-completeness)", ""]
    lines.append("## Provenance")
    for key, p in payload["provenance"].items():
        lines.append(f"- **{key}**: `{p['repo']}` @ `{p['sha']}`")
    lines.append("")
    lines.append("## Pass-7 Headlines")
    for domain_key, summary in payload["domains"].items():
        delta = summary["pass7_delta"]
        lines.append(f"- **{domain_key.upper()}**: "
                     f"pass-a combined accept rate **{summary['combined_submission_gate']['acceptance_rate']:.1%}**, "
                     f"pass-7 intrinsic-only accept rate **{summary['intrinsic']['acceptance_rate']:.1%}**. "
                     f"{delta['flipped_reject_to_accept_under_intrinsic_only']}/{summary['total_cases']} cases flipped "
                     f"reject → accept after separating bundle-completeness.")
    lines.append("")
    for domain_key, summary in payload["domains"].items():
        lines.append(f"## {domain_key.upper()}")
        lines.append(f"- Total cases: {summary['total_cases']}")
        lines.append("")
        lines.append("### Axis 1: intrinsic legitimacy (artifact-only, no bundle requirements)")
        i = summary["intrinsic"]
        lines.append(f"- Decisions: {i['decisions']}")
        lines.append(f"- **Acceptance rate: {i['acceptance_rate']:.2%}**")
        lines.append(f"- **False reject rate: {i['false_reject_rate']:.2%}**")
        lines.append("")
        lines.append("#### Finding-bucket totals (intrinsic axis)")
        for bucket, count in i["finding_bucket_totals"].items():
            touched = i["cases_touched_by_bucket"].get(bucket, 0)
            lines.append(f"- `{bucket}`: {count} findings across {touched} cases")
        lines.append("")
        lines.append("#### Top intrinsic finding strings")
        for entry in i["top_finding_strings"]:
            lines.append(f"- [{entry['bucket']}] `{entry['finding']}` — {entry['count']}")
        lines.append("")

        lines.append("### Axis 2: submission-bundle completeness (Codex bundle format)")
        c = summary["bundle_completeness"]
        lines.append(f"- Decisions: {c['decisions']}")
        lines.append(f"- Acceptance rate: {c['acceptance_rate']:.2%}")
        lines.append(f"- False reject rate: {c['false_reject_rate']:.2%}")
        lines.append(f"- (Expected to reject all upstream by design — these files lack our local bundle shape.)")
        lines.append("")

        lines.append("### Pass-a vs Pass-7 delta")
        d = summary["pass7_delta"]
        lines.append(f"- {d['interpretation']}")
        if d["still_reject_cases"]:
            lines.append("- Cases still rejected under intrinsic-only:")
            for cid in d["still_reject_cases"][:20]:
                lines.append(f"  - `{cid}`")
            if len(d["still_reject_cases"]) > 20:
                lines.append(f"  - ... (+{len(d['still_reject_cases']) - 20} more)")
        lines.append("")
        lines.append("### Combined (submission-gate simulation)")
        lines.append(f"- Decisions: {summary['combined_submission_gate']['decisions']}")
        lines.append(f"- Acceptance rate: {summary['combined_submission_gate']['acceptance_rate']:.2%}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    tla_cases = _discover_tla_cases()
    tla_rows = [_score_tla_case(c) for c in tla_cases]

    lean_mod = _load_lean_scorer()
    lean_cases = _discover_lean_cases()
    lean_rows = [_score_lean_case(c, lean_mod) for c in lean_cases]

    payload = {
        "provenance": {
            "tla": {"repo": "tlaplus/Examples", "sha": _git_head(TLA_ROOT)},
            "lean4_mil": {"repo": "leanprover-community/mathematics_in_lean", "sha": _git_head(LEAN_ROOT)},
            "lean4_mathlib": {"repo": "leanprover-community/mathlib4 (sparse: Data/Nat + select basics)", "sha": _git_head(MATHLIB_ROOT)},
        },
        "inventory": {
            "tla_case_count": len(tla_cases),
            "lean4_case_count": len(lean_cases),
        },
        "domains": {
            "tla": _domain_summary(tla_rows),
            "lean4": _domain_summary(lean_rows),
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
        "tla_intrinsic_accept_rate": payload["domains"]["tla"]["intrinsic"]["acceptance_rate"],
        "lean4_intrinsic_accept_rate": payload["domains"]["lean4"]["intrinsic"]["acceptance_rate"],
        "tla_combined_accept_rate": payload["domains"]["tla"]["combined_submission_gate"]["acceptance_rate"],
        "lean4_combined_accept_rate": payload["domains"]["lean4"]["combined_submission_gate"]["acceptance_rate"],
        "json_report": str(json_path.relative_to(REPO_ROOT)),
        "markdown_report": str(md_path.relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
