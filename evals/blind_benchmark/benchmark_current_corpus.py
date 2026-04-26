#!/usr/bin/env python3
# noqa: DECL-1 (benchmark utility — not empirical QA code)
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "current"

DOMAINS = {
    "tla": {
        "label": "TLA+",
        "executor": REPO_ROOT / "evals" / "tla_blind" / "execute_current_system.py",
    },
    "lean4": {
        "label": "Lean 4",
        "executor": REPO_ROOT / "evals" / "lean4_blind" / "execute_current_system.py",
    },
    "upwork": {
        "label": "Upwork-style",
        "executor": REPO_ROOT / "evals" / "upwork_blind" / "execute_current_system.py",
    },
}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _run_executor(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{path} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    data = json.loads(proc.stdout)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not return a JSON object")
    return data


def _expected_decision(row: dict[str, Any]) -> str | None:
    return row.get("expected_decision")


def _error_classification(row: dict[str, Any]) -> str:
    errors = "\n".join(row.get("errors", [])).lower()
    predicted = row.get("decision")
    expected = row.get("expected_decision")
    if predicted == "accept" and expected != "accept":
        if "ground" in errors or "source" in errors:
            return "false_accept_source_fidelity"
        if "comparable" in errors or "repository-fit" in errors:
            return "false_accept_repo_fit"
        return "false_accept_general"
    if predicted == "reject" and expected != "reject":
        if "proof idea" in errors or "explanatory" in errors or "readme" in errors:
            return "false_reject_sparse_or_explanation"
        return "false_reject_general"
    if predicted == "revise" and expected == "accept":
        return "over_conservative_revision"
    if predicted == "accept" and expected == "revise":
        return "under_sensitive_revision"
    if predicted == "revise" and expected == "reject":
        return "under_rejecting_bad_artifact"
    if predicted == "reject" and expected == "accept":
        return "over_rejecting_good_artifact"
    return "other_misclassification"


def _confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = ("accept", "revise", "reject")
    matrix = {expected: {pred: 0 for pred in labels} for expected in labels}
    for row in rows:
        expected = row["expected_decision"]
        predicted = row["decision"]
        matrix[expected][predicted] += 1
    return matrix


def _score_distributions(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.get("scores", {}).keys())
    out: dict[str, dict[str, int]] = {}
    for key in sorted(keys):
        counter = Counter(int(row["scores"][key]) for row in rows if key in row.get("scores", {}))
        out[key] = {str(k): counter[k] for k in sorted(counter)}
    return out


def _class_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row[field]) for row in rows)
    return {label: counter.get(label, 0) for label in ("accept", "revise", "reject")}


def _status_from_rates(accuracy: float, false_accept_rate: float, false_reject_rate: float) -> str:
    if false_accept_rate > 0.15:
        return "over-trusting"
    if false_reject_rate > 0.20 and false_accept_rate <= 0.05:
        return "conservative"
    if accuracy >= 0.85 and false_accept_rate <= 0.10:
        return "balanced"
    return "mixed"


def _analyze_domain(payload: dict[str, Any], label: str) -> dict[str, Any]:
    labeled_rows = [row for row in payload["results"] if _expected_decision(row) in {"accept", "revise", "reject"}]
    correct = [row for row in labeled_rows if row["decision"] == row["expected_decision"]]
    wrong = [row for row in labeled_rows if row["decision"] != row["expected_decision"]]
    non_accept = [row for row in labeled_rows if row["expected_decision"] != "accept"]
    non_reject = [row for row in labeled_rows if row["expected_decision"] != "reject"]
    false_accepts = [row for row in labeled_rows if row["decision"] == "accept" and row["expected_decision"] != "accept"]
    false_rejects = [row for row in labeled_rows if row["decision"] == "reject" and row["expected_decision"] != "reject"]
    wrong_transitions = [
        {
            "case_id": row["case_id"],
            "layer": row["layer"],
            "expected": row["expected_decision"],
            "predicted": row["decision"],
            "classification": _error_classification(row),
            "errors": row.get("errors", []),
        }
        for row in wrong
    ]
    accuracy = (len(correct) / len(labeled_rows)) if labeled_rows else 0.0
    false_accept_rate = (len(false_accepts) / len(non_accept)) if non_accept else 0.0
    false_reject_rate = (len(false_rejects) / len(non_reject)) if non_reject else 0.0
    return {
        "label": label,
        "total_labeled_fixtures": len(labeled_rows),
        "expected_label_counts": _class_counts(labeled_rows, "expected_decision"),
        "predicted_label_counts": _class_counts(labeled_rows, "decision"),
        "confusion_matrix": _confusion_matrix(labeled_rows),
        "overall_decision_accuracy": round(accuracy, 4),
        "false_accept_rate": round(false_accept_rate, 4),
        "false_reject_rate": round(false_reject_rate, 4),
        "score_distributions": _score_distributions(labeled_rows),
        "false_accepts": [
            {
                "case_id": row["case_id"],
                "layer": row["layer"],
                "expected": row["expected_decision"],
                "predicted": row["decision"],
            }
            for row in false_accepts
        ],
        "false_rejects": [
            {
                "case_id": row["case_id"],
                "layer": row["layer"],
                "expected": row["expected_decision"],
                "predicted": row["decision"],
            }
            for row in false_rejects
        ],
        "wrong_transitions": wrong_transitions,
        "status_call": _status_from_rates(accuracy, false_accept_rate, false_reject_rate),
    }


def _cross_domain_summary(domain_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = sum(report["total_labeled_fixtures"] for report in domain_reports.values())
    correct = sum(
        report["total_labeled_fixtures"] - len(report["wrong_transitions"])
        for report in domain_reports.values()
    )
    wrong = [item for report in domain_reports.values() for item in report["wrong_transitions"]]
    false_accepts = [item for report in domain_reports.values() for item in report["false_accepts"]]
    false_rejects = [item for report in domain_reports.values() for item in report["false_rejects"]]
    pattern_counts = Counter(item["classification"] for item in wrong)
    return {
        "total_labeled_fixtures": total,
        "overall_accuracy": round((correct / total) if total else 0.0, 4),
        "false_accept_count": len(false_accepts),
        "false_reject_count": len(false_rejects),
        "top_recurring_failure_patterns": [
            {"classification": name, "count": count}
            for name, count in pattern_counts.most_common()
        ],
        "recommended_next_fixes": (
            [
                "No corpus-level misclassifications on the current labeled suites. The smallest next step is to enlarge the deception and borderline corpora rather than retuning thresholds."
            ]
            if not wrong else
            [
                "Tighten shared source-fidelity heuristics before adjusting domain-local thresholds.",
                "Protect sparse-but-faithful artifacts from explanation-only penalties.",
            ]
        ),
    }


def _markdown_matrix(matrix: dict[str, dict[str, int]]) -> list[str]:
    cols = ["accept", "revise", "reject"]
    lines = ["| expected \\ predicted | accept | revise | reject |", "|---|---:|---:|---:|"]
    for expected in cols:
        row = matrix[expected]
        lines.append(f"| {expected} | {row['accept']} | {row['revise']} | {row['reject']} |")
    return lines


def _render_report(payload: dict[str, Any]) -> str:
    lines = ["# Blind Corpus Benchmark Sweep", ""]
    cross = payload["cross_domain_summary"]
    lines.extend([
        "## Cross-Domain Summary",
        f"- Total labeled fixtures: {cross['total_labeled_fixtures']}",
        f"- Overall accuracy: {cross['overall_accuracy']:.2%}",
        f"- False accept count: {cross['false_accept_count']}",
        f"- False reject count: {cross['false_reject_count']}",
        "",
    ])
    if cross["top_recurring_failure_patterns"]:
        lines.append("### Recurring Failure Patterns")
        for item in cross["top_recurring_failure_patterns"]:
            lines.append(f"- `{item['classification']}`: {item['count']}")
        lines.append("")
    lines.append("### Recommended Next Fixes")
    for item in cross["recommended_next_fixes"]:
        lines.append(f"- {item}")
    lines.append("")
    for key, report in payload["domains"].items():
        lines.extend([
            f"## {report['label']}",
            f"- Labeled fixtures: {report['total_labeled_fixtures']}",
            f"- Overall decision accuracy: {report['overall_decision_accuracy']:.2%}",
            f"- False accept rate: {report['false_accept_rate']:.2%}",
            f"- False reject rate: {report['false_reject_rate']:.2%}",
            f"- Status call: {report['status_call']}",
            "",
            "### Label Counts",
            f"- Expected: {report['expected_label_counts']}",
            f"- Predicted: {report['predicted_label_counts']}",
            "",
            "### Confusion Matrix",
            *_markdown_matrix(report["confusion_matrix"]),
            "",
            "### False Accepts",
        ])
        if report["false_accepts"]:
            for item in report["false_accepts"]:
                lines.append(f"- `{item['case_id']}` ({item['layer']}): expected `{item['expected']}`, predicted `{item['predicted']}`")
        else:
            lines.append("- none")
        lines.extend(["", "### False Rejects"])
        if report["false_rejects"]:
            for item in report["false_rejects"]:
                lines.append(f"- `{item['case_id']}` ({item['layer']}): expected `{item['expected']}`, predicted `{item['predicted']}`")
        else:
            lines.append("- none")
        lines.extend(["", "### Wrong Transitions"])
        if report["wrong_transitions"]:
            for item in report["wrong_transitions"]:
                lines.append(
                    f"- `{item['case_id']}` ({item['layer']}): expected `{item['expected']}`, predicted `{item['predicted']}`, classified as `{item['classification']}`"
                )
        else:
            lines.append("- none")
        lines.extend(["", "### Score Distributions"])
        for score_name, dist in report["score_distributions"].items():
            lines.append(f"- `{score_name}`: {dist}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    domain_payloads = {
        key: _run_executor(spec["executor"])
        for key, spec in DOMAINS.items()
    }
    domain_reports = {
        key: _analyze_domain(payload, DOMAINS[key]["label"])
        for key, payload in domain_payloads.items()
    }
    combined = {
        "domains": domain_reports,
        "cross_domain_summary": _cross_domain_summary(domain_reports),
    }
    json_path = RESULTS_ROOT / "blind_corpus_benchmark.json"
    md_path = RESULTS_ROOT / "blind_corpus_benchmark.md"
    json_path.write_text(_json_dumps(combined) + "\n", encoding="utf-8")
    md_path.write_text(_render_report(combined), encoding="utf-8")
    print(_json_dumps({"ok": True, "json_report": str(json_path.relative_to(REPO_ROOT)), "markdown_report": str(md_path.relative_to(REPO_ROOT))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
