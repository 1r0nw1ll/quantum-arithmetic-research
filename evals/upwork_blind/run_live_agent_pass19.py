#!/usr/bin/env python3
# noqa: DECL-1 (Pass-19 weaker-model stress runner — not empirical QA code)
"""
Pass-19: rerun Pass-18 adversarial Upwork stress on a weaker model.

Pass 18 ran 5 engineered adversarial variants × 2 Upwork tasks against
codex (`codex exec --full-auto`). The Pass-16 structural gate fired
alone on 0/10 outputs — codex's craft is robust enough that the gate
has no measured operational contribution on this task class.

The synthesis (Pass 17) flagged this as the cleanest remaining
empirical gap: the gate may matter for weaker models. Pass 19 tests
that hypothesis by running the same 5 variants × 2 tasks against
`opencode/gpt-5-nano` — the smallest gpt-5 variant, plausibly less
robust against adversarial framing than codex.

Same design as Pass 18:
- Same task set (`bugfix_factorial_zero`, `script_csv_domain_count`)
- Same 5 adversarial variants (`invent_deps`, `top_level_demo_io`,
  `nested_package`, `broken_parametrize`, `compact_packed`)
- Each output scored TWICE — full cascade vs heuristic-only — so the
  gate's contribution is isolated by quadrant
- 4-quadrant breakdown: heur-pass × struct-pass / heur-pass × struct-fail
  / heur-fail × struct-pass / heur-fail × struct-fail

Outputs:
- evals/upwork_blind/results/live_agent_pass19/<task>/<variant>/<files>
- evals/upwork_blind/results/live_agent_pass19/pass19_report.{json,md}
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
RESULTS_ROOT = ROOT / "results" / "live_agent_pass19"
TASK_ROOT = ROOT / "tasks" / "generation"

# Weaker model. opencode/gpt-5-nano is the smallest gpt-5 variant
# available in the local opencode model menu. If it resists exactly
# like codex, that's a stronger empirical claim that the gate is
# silent across the family — not just on the flagship.
MODEL = "opencode/gpt-5-nano"


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "upwork_blind_executor",
        ROOT / "execute_current_system.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Identical to Pass 18 — same engineered prompts, so the only changing
# variable across passes is the model.
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


def _run_opencode(workspace: Path, prompt: str, timeout: int = 300) -> dict[str, Any]:
    """Invoke opencode in headless mode against the weaker model.

    `opencode run` writes files via its tool layer to the cwd. Same shape
    as `codex exec` for our purposes — we drop into a tempdir, run, then
    persist the workspace.
    """
    proc = subprocess.run(
        ["opencode", "run", "-m", MODEL, "--format", "default", prompt],
        cwd=str(workspace),
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
    """Pre-Pass-16 cascade — structural gate DISABLED (text heuristics only)."""
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
    with tempfile.TemporaryDirectory(prefix="upwork-pass19-") as td:
        workspace = Path(td)
        opencode_meta = _run_opencode(workspace, prompt)
        files_saved = _persist(workspace, task_json["case_id"], variant_name)
        full = _score_full(workspace, task_json, scorer_mod)
        heur = _score_heuristic_only(workspace, task_json, scorer_mod)
        struct_findings = _structural_only(workspace, scorer_mod)
    return {
        "task": task_json["case_id"],
        "variant": variant_name,
        "model": MODEL,
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
        "opencode_returncode": opencode_meta["returncode"],
        "saved_to": str((RESULTS_ROOT / task_json["case_id"] / variant_name).relative_to(REPO_ROOT)),
    }


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Pass-19 Weaker-Model Stress Report (Upwork)", ""]
    lines.append("## Generator")
    lines.append(f"- Live agent: `opencode run -m {MODEL}`")
    lines.append("- Goal: rerun Pass-18 adversarial variants on a weaker model")
    lines.append("  and measure whether the Pass-16 structural gate fires when")
    lines.append("  craft drops. Pass 18 baseline (codex): gate fired alone on 0/10.")
    lines.append("")
    lines.append("## Adversarial variants (identical to Pass 18)")
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
    lines.append("| quadrant | Pass 19 (gpt-5-nano) | Pass 18 (codex) | meaning |")
    lines.append("|---|---:|---:|---|")
    pass18_quadrants = {
        "heur_pass__struct_pass": 7,
        "heur_pass__struct_fail": 0,
        "heur_fail__struct_pass": 3,
        "heur_fail__struct_fail": 0,
    }
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
        lines.append(
            f"| `{q}` | {quadrants.get(q, 0)} | {pass18_quadrants[q]} | {meanings[q]} |"
        )
    lines.append("")
    struct_only_signal = quadrants.get("heur_pass__struct_fail", 0)
    lines.append("## Headline")
    lines.append(
        f"- **Pass-16 gate fired alone (heuristic-pass × struct-fail) on "
        f"{struct_only_signal}/{len(rows)} outputs** (Pass 18 / codex: 0/10)."
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
    any_findings = False
    for r in rows:
        if r["structural_fail"]:
            any_findings = True
            lines.append(f"### `{r['task']} :: {r['variant']}`")
            for f in r["structural_findings"]:
                lines.append(f"- {f}")
            lines.append("")
    if not any_findings:
        lines.append("*(none — gate silent across all outputs, same as Pass 18)*")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    scorer_mod = _load_scorer()
    task_dirs = sorted(d for d in TASK_ROOT.iterdir() if d.is_dir())
    rows: list[dict[str, Any]] = []
    total_runs = len(task_dirs) * len(VARIANTS)
    print(
        f"... Pass-19 stress: {len(task_dirs)} tasks × {len(VARIANTS)} variants = "
        f"{total_runs} {MODEL} calls",
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
                    "model": MODEL,
                    "decision_full_cascade": "(timeout)",
                    "decision_heuristic_only": "(timeout)",
                    "structural_findings": ["opencode timed out before producing output"],
                    "structural_fail": True,
                    "quadrant": "heur_fail__struct_fail",
                    "scores_full": {}, "scores_heuristic_only": {},
                    "findings_full": ["opencode timed out"],
                    "findings_heuristic_only": ["opencode timed out"],
                    "files_generated": [],
                }
            rows.append(row)

    payload = {
        "model": MODEL,
        "task_count": len(task_dirs),
        "variant_count": len(VARIANTS),
        "total_runs": len(rows),
        "quadrants": dict(Counter(r["quadrant"] for r in rows)),
        "rows": rows,
    }
    (RESULTS_ROOT / "pass19_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RESULTS_ROOT / "pass19_report.md").write_text(_build_markdown(rows), encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "model": MODEL,
        "total_runs": len(rows),
        "quadrants": dict(Counter(r["quadrant"] for r in rows)),
        "json_report": str((RESULTS_ROOT / "pass19_report.json").relative_to(REPO_ROOT)),
        "md_report": str((RESULTS_ROOT / "pass19_report.md").relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
