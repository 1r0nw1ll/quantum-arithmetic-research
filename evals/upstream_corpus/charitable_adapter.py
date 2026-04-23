#!/usr/bin/env python3
# noqa: DECL-1 (benchmark utility — not empirical QA code)
"""
Pass-7-b charitable adapter for upstream TLA+ specs.

Goal: measure how much of the Pass-7 `revise` load on upstream-approved
TLA+ specs is extraction debt (comments already in the .tla file that the
harness happens not to read) vs. real judgment debt (missing outsider
explanation that the upstream artifact simply does not contain).

Extraction rules (strict — Will's directive):

  Allowed:
    - (* block comments *) verbatim from .tla files
    - \\* end-of-line comments, preserved with the declaration line they
      annotate (so variable/action names travel with their comments)

  Forbidden:
    - inventing source_grounding content
    - inventing repo_comparables
    - paraphrasing weak evidence into stronger evidence
    - writing any prose that is not literally already in the .tla file

The adapter writes the extracted comments to a synthetic
`extracted_tla_comments.md` inside a tempdir copy of the spec, then runs
the unmodified Pass-7 intrinsic scorer against that tempdir. Upstream
repos are NOT modified.

Report: per-case delta under adapter-assisted intrinsic scoring.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current"
TLA_ROOT = Path("/home/player2/upstream_corpora/tlaplus_examples")

sys.path.insert(0, str(REPO_ROOT / "tools"))
import qa_formal_publication_gate as tla_gate  # noqa: E402


BLOCK_COMMENT = re.compile(r"\(\*(.*?)\*\)", re.DOTALL)
LINE_WITH_LINE_COMMENT = re.compile(r"^.*\\\*.*$", re.M)


def _strip_block_comment(body: str) -> str:
    """Clean a block-comment body. Drop full lines of only asterisks/whitespace,
    trim leading/trailing whitespace on each line. Keep prose verbatim."""
    lines = body.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Decoration-only lines like "*************" or "(*******)"
        if not stripped:
            continue
        if re.fullmatch(r"[\*\s]+", stripped):
            continue
        # Common leading decoration: lines beginning with leading spaces then asterisks
        # then prose. Strip leading asterisks+spaces.
        cleaned = re.sub(r"^\s*\*+\s?", "", line).rstrip()
        if not cleaned:
            continue
        out.append(cleaned)
    return "\n".join(out)


def extract_tla_comments(spec_dir: Path) -> str:
    """Return a synthetic prose blob of all comments in the spec's .tla files."""
    parts: list[str] = []
    for tla_path in sorted(spec_dir.glob("*.tla")):
        text = tla_path.read_text(encoding="utf-8", errors="replace")
        parts.append(f"## {tla_path.name} — block comments")
        for match in BLOCK_COMMENT.finditer(text):
            cleaned = _strip_block_comment(match.group(1))
            if cleaned:
                parts.append(cleaned)
                parts.append("")
        parts.append(f"## {tla_path.name} — declaration lines with inline comments")
        for match in LINE_WITH_LINE_COMMENT.finditer(text):
            line = match.group(0).rstrip()
            if line.strip():
                parts.append(line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _copy_to_tempdir(spec_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in spec_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)


def score_with_adapter(spec_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "bundle"
        _copy_to_tempdir(spec_dir, tmp)
        comments = extract_tla_comments(spec_dir)
        (tmp / "extracted_tla_comments.md").write_text(comments, encoding="utf-8")
        intrinsic = tla_gate.score_intrinsic_legitimacy(tmp)
        decision = tla_gate.intrinsic_decision_from_scores(intrinsic["scores"])
        return {
            "decision": decision,
            "scores": intrinsic["scores"],
            "findings": intrinsic["findings"],
            "extracted_bytes": len(comments),
        }


def score_without_adapter(spec_dir: Path) -> dict[str, Any]:
    intrinsic = tla_gate.score_intrinsic_legitimacy(spec_dir)
    decision = tla_gate.intrinsic_decision_from_scores(intrinsic["scores"])
    return {
        "decision": decision,
        "scores": intrinsic["scores"],
        "findings": intrinsic["findings"],
    }


def _discover_revise_cases() -> list[Path]:
    """Find all TLA upstream spec dirs whose baseline intrinsic decision is revise."""
    specs_root = TLA_ROOT / "specifications"
    out: list[Path] = []
    for spec_dir in sorted(p for p in specs_root.iterdir() if p.is_dir()):
        if not any(spec_dir.glob("*.tla")):
            continue
        baseline = score_without_adapter(spec_dir)
        if baseline["decision"] == "revise":
            out.append(spec_dir)
    return out


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    revise_cases = _discover_revise_cases()
    rows: list[dict[str, Any]] = []
    flipped_to_accept: list[str] = []
    still_revise: list[str] = []
    regressed: list[str] = []
    for spec_dir in revise_cases:
        baseline = score_without_adapter(spec_dir)
        adapted = score_with_adapter(spec_dir)
        baseline_decision = baseline["decision"]
        adapted_decision = adapted["decision"]
        row = {
            "case_id": spec_dir.name,
            "source_path": str(spec_dir.relative_to(TLA_ROOT)),
            "baseline_decision": baseline_decision,
            "adapted_decision": adapted_decision,
            "baseline_findings": baseline["findings"],
            "adapted_findings": adapted["findings"],
            "extracted_bytes": adapted["extracted_bytes"],
            "delta": "flipped_to_accept" if adapted_decision == "accept" and baseline_decision != "accept"
                    else "still_revise" if adapted_decision == "revise"
                    else "regressed" if adapted_decision == "reject"
                    else "other",
        }
        rows.append(row)
        if row["delta"] == "flipped_to_accept":
            flipped_to_accept.append(spec_dir.name)
        elif row["delta"] == "still_revise":
            still_revise.append(spec_dir.name)
        elif row["delta"] == "regressed":
            regressed.append(spec_dir.name)

    # Aggregate remaining revise reasons
    remaining_findings: Counter[str] = Counter()
    for row in rows:
        if row["adapted_decision"] == "revise":
            for f in row["adapted_findings"]:
                remaining_findings[f] += 1

    summary = {
        "provenance": {
            "tla": {"repo": "tlaplus/Examples", "sha": _git_head(TLA_ROOT)},
        },
        "total_revise_cases_at_baseline": len(revise_cases),
        "flipped_to_accept": len(flipped_to_accept),
        "still_revise": len(still_revise),
        "regressed": len(regressed),
        "flipped_cases": flipped_to_accept,
        "still_revise_cases": still_revise,
        "regressed_cases": regressed,
        "top_remaining_revise_reasons": [
            {"finding": f, "count": c} for f, c in remaining_findings.most_common(15)
        ],
    }

    payload = {"summary": summary, "rows": rows}
    (RESULTS_ROOT / "charitable_adapter.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Markdown report
    lines = ["# Pass-7-b Charitable Adapter Report", ""]
    lines.append("## Provenance")
    p = summary["provenance"]["tla"]
    lines.append(f"- **{p['repo']}** @ `{p['sha']}`")
    lines.append("")
    lines.append("## Measurement")
    lines.append(f"- Baseline Pass-7 intrinsic `revise` cases: **{summary['total_revise_cases_at_baseline']}**")
    lines.append(f"- Flipped `revise` → `accept` after charitable extraction: **{summary['flipped_to_accept']}**")
    lines.append(f"- Remaining `revise` after extraction: **{summary['still_revise']}**")
    lines.append(f"- Regressed (revise → reject): **{summary['regressed']}**")
    if summary["total_revise_cases_at_baseline"]:
        pct = summary["flipped_to_accept"] / summary["total_revise_cases_at_baseline"]
        lines.append(f"- Extraction-debt share: {pct:.1%} of the revise load was extraction debt (comments already in the .tla file)")
    lines.append("")
    lines.append("## Extraction rules (strict)")
    lines.append("- Only extracts `(* block comments *)` and `\\* line comments` already present in .tla files")
    lines.append("- Preserves declaration lines (VARIABLES, action definitions) that carry inline comments, so variable/action names travel with their explanations")
    lines.append("- Writes the extracted text to `extracted_tla_comments.md` inside a tempdir bundle copy")
    lines.append("- Does NOT modify upstream repos; does NOT synthesize evidence; does NOT paraphrase")
    lines.append("")
    if summary["flipped_cases"]:
        lines.append("## Cases flipped `revise` → `accept`")
        for c in summary["flipped_cases"]:
            lines.append(f"- `{c}`")
        lines.append("")
    if summary["still_revise_cases"]:
        lines.append("## Cases still `revise` after extraction")
        for c in summary["still_revise_cases"]:
            lines.append(f"- `{c}`")
        lines.append("")
    if summary["regressed_cases"]:
        lines.append("## REGRESSIONS — flipped revise → reject (should be empty)")
        for c in summary["regressed_cases"]:
            lines.append(f"- `{c}`")
        lines.append("")
    if summary["top_remaining_revise_reasons"]:
        lines.append("## Top remaining revise reasons (after extraction)")
        for item in summary["top_remaining_revise_reasons"]:
            lines.append(f"- `{item['finding']}` — {item['count']}")
        lines.append("")
    (RESULTS_ROOT / "charitable_adapter.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "ok": True,
        "total_revise_cases": summary["total_revise_cases_at_baseline"],
        "flipped_to_accept": summary["flipped_to_accept"],
        "still_revise": summary["still_revise"],
        "regressed": summary["regressed"],
        "json_report": str((RESULTS_ROOT / "charitable_adapter.json").relative_to(REPO_ROOT)),
        "markdown_report": str((RESULTS_ROOT / "charitable_adapter.md").relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
