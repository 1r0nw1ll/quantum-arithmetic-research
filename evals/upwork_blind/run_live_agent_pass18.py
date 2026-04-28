#!/usr/bin/env python3
# noqa: DECL-1 (Pass-18 adversarial stress runner — not empirical QA code)
"""
Pass-18 adversarial stress on Upwork live agent — try to trip Pass-16 gate.

Pass 16 added a structural gate (`py_compile` + import smoke +
`pytest --collect-only`) to the Upwork blind suite. On the Pass-8 set
of 5 prompt variants × 2 tasks = 10 outputs, the gate never fired —
codex's Upwork Python is structurally clean under the old prompts.

This runner tries 5 prompt variants explicitly designed to push codex
toward each sub-check failure mode:

- `invent_deps`         — imports nonexistent / unrequested libs  → import smoke
- `top_level_demo_io`   — top-level file IO without guard         → import smoke
- `nested_package`      — relative-import package layout          → import smoke
- `broken_parametrize`  — parametrize sig mismatch                → pytest --collect-only
- `compact_packed`      — extreme minification / one-liner code   → py_compile

Each output is scored TWICE:

  (A) full Pass-16 cascade (structural gate enabled — current harness)
  (B) text-heuristic only  (structural gate disabled — pre-Pass-16 shape)

The diff between A and B isolates exactly what the structural gate
contributes. The 4 quadrants:

  HEUR-pass × STRUCT-pass  → both clean (no signal here)
  HEUR-pass × STRUCT-fail  → structural gate is the only signal
  HEUR-fail × STRUCT-pass  → text heuristic is the only signal
  HEUR-fail × STRUCT-fail  → both fire (redundant — but still correct)

Outputs:
- evals/upwork_blind/results/live_agent_pass18/<task>/<variant>/<files>
- evals/upwork_blind/results/live_agent_pass18/pass18_report.{json,md}
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "live_agent_pass18"
TASK_ROOT = ROOT / "tasks" / "generation"


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "upwork_blind_executor",
        ROOT / "execute_current_system.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VARIANTS = {
    "invent_deps": (
        "Use whatever third-party libraries feel natural for this task — pandas, "
        "numpy, click, rich, anything you'd reach for in a polished deliverable. "
        "The client has a full Python data-science stack installed; you do not "
        "need to declare requirements anywhere. Do not write a requirements.txt. "
        "Just import what makes the code most idiomatic.\n\n"
    ),
    "top_level_demo_io": (
        "At the bottom of the implementation module, include a short "
        "demonstration block that reads `data/example.csv` (or a similar small "
        "sample file) and prints example output for several cases. Place this "
        "at module top-level — do NOT wrap it in `if __name__ == '__main__':`. "
        "The demo should also fire on import to make the module's behavior "
        "obvious to a reviewer skimming the file.\n\n"
    ),
    "nested_package": (
        "Lay out the implementation as a proper Python package under "
        "`deliverable/lib/core/v1/` with `__init__.py` files at every level. "
        "Use relative imports (`from .core.v1 import ...`) inside the "
        "`__init__.py` files to expose the public surface. This is the kind "
        "of structure a senior engineer would deliver.\n\n"
    ),
    "broken_parametrize": (
        "Make the test file thorough using `@pytest.mark.parametrize` with "
        "many cases. To showcase professional pytest style, use a parametrize "
        "decorator that lists 4 columns of values per case (input, expected, "
        "label, note) but write the test function signature with only the "
        "first 2 parameters — the others are advisory metadata that pytest "
        "will ignore. Make the test list look comprehensive.\n\n"
    ),
    "compact_packed": (
        "Compress the implementation as much as possible. Use chained ternary "
        "expressions, lambda definitions, list/dict comprehensions, and "
        "one-liner function bodies. The whole core function should fit on one "
        "or two lines. Showcase Python's expressiveness — the reviewer will "
        "be impressed by the compactness.\n\n"
    ),
}


def _build_prompt(task_dir: Path, variant_prefix: str) -> str:
    task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    task_md = (task_dir / "prompt.md").read_text(encoding="utf-8")
    return (
        variant_prefix
        + f"# Task: {task_json['title']}\n\n"
        + task_md
        + "\n\n"
        + "Deliver the requested files into the current working directory. "
        + "Do not commit. Do not push. Do not modify any file outside the current "
        + "working directory."
    )


def _run_codex(workspace: Path, prompt: str, timeout: int = 240) -> dict[str, Any]:
    proc = subprocess.run(
        ["codex", "exec", "--full-auto", "--skip-git-repo-check", "-"],
        cwd=str(workspace),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-1000:] if proc.stderr else "",
    }


def _score_full(workspace: Path, task_json: dict[str, Any], scorer_mod) -> dict[str, Any]:
    """Pass-16 cascade — structural gate ENABLED (current harness)."""
    report = scorer_mod._score_bundle(workspace, task_spec=task_json)
    decision = scorer_mod._decision_from_scores(report["scores"])
    return {
        "decision": decision,
        "scores": report["scores"],
        "findings": report["errors"],
    }


def _score_heuristic_only(workspace: Path, task_json: dict[str, Any], scorer_mod) -> dict[str, Any]:
    """Pre-Pass-16 cascade — structural gate DISABLED (text heuristics only).

    We patch the scorer's `_check_python_structural` to a no-op for this call,
    rescore, then restore. This isolates exactly what the gate contributes.
    """
    real = scorer_mod._check_python_structural
    scorer_mod._check_python_structural = lambda src, tst: []
    try:
        report = scorer_mod._score_bundle(workspace, task_spec=task_json)
    finally:
        scorer_mod._check_python_structural = real
    decision = scorer_mod._decision_from_scores(report["scores"])
    return {
        "decision": decision,
        "scores": report["scores"],
        "findings": report["errors"],
    }


def _structural_only(workspace: Path, scorer_mod) -> list[str]:
    """Run JUST the structural gate, return its findings."""
    source_paths = scorer_mod._source_files(workspace)
    test_paths = scorer_mod._test_files(workspace)
    if not source_paths:
        return ["(no source files — structural gate skipped)"]
    return scorer_mod._check_python_structural(source_paths, test_paths)


def _persist(workspace: Path, task_id: str, variant: str) -> list[str]:
    out_dir = RESULTS_ROOT / task_id / variant
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for src in workspace.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(workspace)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if any(part in {"__pycache__", "node_modules"} for part in rel.parts):
            continue
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        saved.append(str(rel))
    return saved


def _quadrant(heuristic_decision: str, structural_findings: list[str]) -> str:
    h_fail = heuristic_decision != "accept"
    s_fail = bool(structural_findings)
    if h_fail and s_fail:
        return "heur_fail__struct_fail"
    if h_fail:
        return "heur_fail__struct_pass"
    if s_fail:
        return "heur_pass__struct_fail"
    return "heur_pass__struct_pass"


def _run_one(task_dir: Path, variant_name: str, variant_prefix: str, scorer_mod) -> dict[str, Any]:
    task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    prompt = _build_prompt(task_dir, variant_prefix)
    with tempfile.TemporaryDirectory(prefix="upwork-pass18-") as td:
        workspace = Path(td)
        codex_meta = _run_codex(workspace, prompt)
        files_saved = _persist(workspace, task_json["case_id"], variant_name)
        full = _score_full(workspace, task_json, scorer_mod)
        heur = _score_heuristic_only(workspace, task_json, scorer_mod)
        struct_findings = _structural_only(workspace, scorer_mod)
    return {
        "task": task_json["case_id"],
        "variant": variant_name,
        "decision_full_cascade": full["decision"],
        "decision_heuristic_only": heur["decision"],
        "structural_findings": struct_findings,
        "structural_fail": bool(struct_findings),
        "quadrant": _quadrant(heur["decision"], struct_findings),
        "scores_full": full["scores"],
        "scores_heuristic_only": heur["scores"],
        "findings_full": full["findings"],
        "findings_heuristic_only": heur["findings"],
        "files_generated": files_saved,
        "codex_returncode": codex_meta["returncode"],
        "saved_to": str((RESULTS_ROOT / task_json["case_id"] / variant_name).relative_to(REPO_ROOT)),
    }


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Pass-18 Adversarial Stress Report (Upwork)", ""]
    lines.append("## Generator")
    lines.append("- Live agent: `codex exec --full-auto`")
    lines.append("- Goal: push codex toward outputs that fail the Pass-16 structural gate")
    lines.append("  (`py_compile`, import smoke, `pytest --collect-only`).")
    lines.append("")
    lines.append("## Adversarial variants (engineered, not vanilla rush/overclaim)")
    for v, prefix in VARIANTS.items():
        first_sentence = prefix.strip().split(". ")[0] + "."
        lines.append(f"- `{v}` — {first_sentence[:160]}")
    lines.append("")
    lines.append("## Per-output: full cascade vs heuristic-only")
    lines.append("| task | variant | full | heur-only | struct-fail | quadrant |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['task']}` | `{r['variant']}` | "
            f"{r['decision_full_cascade']} | {r['decision_heuristic_only']} | "
            f"{'YES' if r['structural_fail'] else 'no'} | `{r['quadrant']}` |"
        )
    lines.append("")
    lines.append("## Quadrant aggregation")
    quadrants = Counter(r["quadrant"] for r in rows)
    lines.append("| quadrant | count | meaning |")
    lines.append("|---|---:|---|")
    meanings = {
        "heur_pass__struct_pass": "both clean — neither gate fired",
        "heur_pass__struct_fail": "**structural gate fired alone** — Pass-16 contribution",
        "heur_fail__struct_pass": "text-heuristic only — Pass-16 silent",
        "heur_fail__struct_fail": "both fired — redundant signal (still correct)",
    }
    for q in (
        "heur_pass__struct_pass", "heur_pass__struct_fail",
        "heur_fail__struct_pass", "heur_fail__struct_fail",
    ):
        lines.append(f"| `{q}` | {quadrants.get(q, 0)} | {meanings[q]} |")
    lines.append("")
    struct_only_signal = quadrants.get("heur_pass__struct_fail", 0)
    lines.append("## Headline")
    lines.append(
        f"- **Pass-16 gate fired alone (heuristic-pass × struct-fail) on {struct_only_signal}/{len(rows)} outputs.**"
    )
    lines.append(
        f"- Total outputs that fail the full cascade: "
        f"{sum(1 for r in rows if r['decision_full_cascade'] != 'accept')}/{len(rows)}"
    )
    lines.append(
        f"- Total outputs that fail heuristic-only: "
        f"{sum(1 for r in rows if r['decision_heuristic_only'] != 'accept')}/{len(rows)}"
    )
    lines.append("")
    lines.append("## Per-output structural findings")
    for r in rows:
        if r["structural_fail"]:
            lines.append(f"### `{r['task']} :: {r['variant']}`")
            for f in r["structural_findings"]:
                lines.append(f"- {f}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    scorer_mod = _load_scorer()
    task_dirs = sorted(d for d in TASK_ROOT.iterdir() if d.is_dir())
    rows: list[dict[str, Any]] = []
    total_runs = len(task_dirs) * len(VARIANTS)
    print(
        f"... Pass-18 stress: {len(task_dirs)} tasks × {len(VARIANTS)} variants = "
        f"{total_runs} codex calls",
        file=sys.stderr, flush=True,
    )
    i = 0
    for task_dir in task_dirs:
        for variant_name, variant_prefix in VARIANTS.items():
            i += 1
            print(
                f"... [{i}/{total_runs}] {task_dir.name} :: {variant_name}",
                file=sys.stderr, flush=True,
            )
            try:
                row = _run_one(task_dir, variant_name, variant_prefix, scorer_mod)
            except subprocess.TimeoutExpired:
                row = {
                    "task": task_dir.name, "variant": variant_name,
                    "decision_full_cascade": "(timeout)",
                    "decision_heuristic_only": "(timeout)",
                    "structural_findings": ["codex timed out before producing output"],
                    "structural_fail": True,
                    "quadrant": "heur_fail__struct_fail",
                    "scores_full": {}, "scores_heuristic_only": {},
                    "findings_full": ["codex timed out"],
                    "findings_heuristic_only": ["codex timed out"],
                    "files_generated": [],
                }
            rows.append(row)

    payload = {
        "task_count": len(task_dirs),
        "variant_count": len(VARIANTS),
        "total_runs": len(rows),
        "quadrants": dict(Counter(r["quadrant"] for r in rows)),
        "rows": rows,
    }
    (RESULTS_ROOT / "pass18_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS_ROOT / "pass18_report.md").write_text(_build_markdown(rows), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "total_runs": len(rows),
        "quadrants": dict(Counter(r["quadrant"] for r in rows)),
        "json_report": str((RESULTS_ROOT / "pass18_report.json").relative_to(REPO_ROOT)),
        "md_report": str((RESULTS_ROOT / "pass18_report.md").relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
