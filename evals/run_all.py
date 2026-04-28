#!/usr/bin/env python3
# noqa: DECL-1 (master runner — not empirical QA code)
"""
Pass-11 master runner for the v1 blind-eval harness.

Runs all evaluation paths and emits a single summary report at
`evals/results/v1_baseline_report.md`.

Default scope (~5-10 seconds):
- _blind_core self-test
- cross-domain blind benchmark (TLA + Lean + Upwork + SWE-Bench)
- deception regression (precision guard)
- upstream-corpus benchmark (TLA tlaplus/Examples + Lean MIL/mathlib)
- charitable adapter (TLA revise → accept under honest comment extraction)

Opt-in (~10-15 minutes, requires `codex` on PATH):
- Upwork live-agent stress (5 prompt variants × 2 generation tasks)
  via --with-live-agent

Exit code:
  0 — every suite within its acceptance bound
  1 — at least one suite reports a real regression (false_accept/
       false_reject; <100% labeled accuracy; >0 charitable-adapter
       regressions; non-zero new bucket count in deception regression)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _run(cmd: list[str], parse_json: bool = True) -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = round(time.time() - t0, 2)
    out: dict[str, Any] = {
        "cmd": " ".join(cmd),
        "exit": proc.returncode,
        "elapsed_s": elapsed,
        "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
    }
    if parse_json:
        try:
            out["payload"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            out["payload"] = None
            out["stdout_tail"] = proc.stdout[-500:]
    else:
        out["stdout_tail"] = proc.stdout[-1000:]
    return out


def run_blind_core_selftest() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "_blind_core" / "__init__.py")])


def run_blind_benchmark() -> dict[str, Any]:
    out = _run([sys.executable, str(ROOT / "blind_benchmark" / "benchmark_current_corpus.py")])
    md = (ROOT / "blind_benchmark" / "results" / "current" / "blind_corpus_benchmark.md")
    if md.exists():
        head = md.read_text(encoding="utf-8").splitlines()[:8]
        out["headline"] = "\n".join(head)
    return out


def run_deception_regression() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "deception_regression" / "run_regression.py")])


def run_upstream_benchmark() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "upstream_corpus" / "run_upstream_benchmark.py")])


def run_charitable_adapter() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "upstream_corpus" / "charitable_adapter.py")])


def run_swe_bench_calibration() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "swe_bench_blind" / "run_calibration.py")])


def run_live_agent_upwork() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "upwork_blind" / "run_live_agent.py")])


def run_live_agent_swe_bench() -> dict[str, Any]:
    return _run([sys.executable, str(ROOT / "swe_bench_blind" / "run_live_agent.py")])


def _verdict(name: str, result: dict[str, Any]) -> tuple[str, str]:
    """Return (status_tag, one_line_summary). status_tag ∈ {OK, FAIL, ERROR, SKIP}."""
    if result.get("exit") != 0:
        return ("FAIL" if name in {"deception_regression"} else "ERROR",
                f"exit={result['exit']} (stderr_tail={result['stderr_tail'][:200]!r})")
    payload = result.get("payload")
    if not payload:
        return ("ERROR", "non-JSON output from suite")
    if name == "_blind_core":
        return ("OK" if payload.get("ok") else "FAIL",
                f"{payload.get('total_checks', '?')} self-test checks; failures={len(payload.get('failures', []))}")
    if name == "blind_benchmark":
        # blind_benchmark prints {ok, json_report, markdown_report} only;
        # parse the json report for the actual metrics.
        json_path = REPO_ROOT / payload.get("json_report", "")
        if json_path.exists():
            d = json.loads(json_path.read_text(encoding="utf-8"))
            cs = d.get("cross_domain_summary", {})
            n = cs.get("total_labeled_fixtures", 0)
            acc = cs.get("overall_accuracy", 0)
            fa = cs.get("false_accept_count", 0)
            fr = cs.get("false_reject_count", 0)
            status = "OK" if (acc == 1.0 and fa == 0 and fr == 0) else "FAIL"
            return (status, f"{n} labeled, accuracy={acc:.2%}, false_accept={fa}, false_reject={fr}")
    if name == "deception_regression":
        nfa = payload.get("new_false_accept", 0)
        nfr = payload.get("new_false_reject", 0)
        kg = payload.get("known_gap", 0)
        total = payload.get("total", 0)
        match = payload.get("match", 0)
        status = "OK" if (nfa == 0 and nfr == 0) else "FAIL"
        return (status, f"{total} fixtures, MATCH={match}, NEW_FALSE_ACCEPT={nfa}, NEW_FALSE_REJECT={nfr}, KNOWN_GAP={kg}")
    if name == "upstream_benchmark":
        tla = payload.get("tla_intrinsic_accept_rate", 0)
        lean = payload.get("lean4_intrinsic_accept_rate", 0)
        return ("OK", f"TLA intrinsic accept={tla:.1%} ({payload.get('tla_cases',0)} cases); Lean intrinsic accept={lean:.1%} ({payload.get('lean4_cases',0)} cases)")
    if name == "charitable_adapter":
        flipped = payload.get("flipped_to_accept", 0)
        still = payload.get("still_revise", 0)
        regressed = payload.get("regressed", 0)
        total = payload.get("total_revise_cases", 0)
        status = "OK" if regressed == 0 else "FAIL"
        return (status, f"{total} revise baseline; {flipped} flipped→accept; {still} still revise; {regressed} regressed")
    if name == "swe_bench_calibration":
        match = payload.get("designed_truth_match", 0)
        labeled = payload.get("designed_truth_labeled", 0)
        tp = payload.get("executed_truth_true_positives", 0)
        fa = payload.get("executed_truth_false_accepts", 0)
        fr = payload.get("executed_truth_false_revise_or_reject", 0)
        testable = payload.get("executed_truth_testable", 0)
        status = "OK" if (fa == 0 and fr == 0) else "FAIL"
        return (status, f"designed truth {match}/{labeled}; executed truth {tp}/{testable} TP, {fa} FA, {fr} FR")
    if name == "live_agent_upwork":
        total = payload.get("total_runs", 0)
        return ("OK", f"{total} live-agent runs (codex); see live_agent_report.md")
    if name == "live_agent_swe_bench":
        total = payload.get("total_runs", 0)
        return ("OK", f"{total} live-agent runs (codex); see live_agent_report.md")
    return ("ERROR", "unknown suite")


def render_report(rows: list[dict[str, Any]], elapsed_total: float) -> str:
    lines = ["# v1 Blind-Eval Harness — Baseline Report", ""]
    lines.append(f"_Total wall-clock: {elapsed_total:.1f}s_")
    lines.append("")
    lines.append("## Summary")
    lines.append("| suite | status | summary | elapsed |")
    lines.append("|---|---|---|---:|")
    overall_ok = True
    for row in rows:
        if row["status"] in {"FAIL", "ERROR"}:
            overall_ok = False
        lines.append(f"| `{row['name']}` | **{row['status']}** | {row['summary']} | {row['elapsed_s']}s |")
    lines.append("")
    lines.append(f"**Overall:** {'OK — all suites within acceptance bounds' if overall_ok else 'FAIL — at least one suite regressed'}")
    lines.append("")
    lines.append("## Suite details")
    for row in rows:
        lines.append(f"### {row['name']}")
        lines.append(f"- Command: `{row['cmd']}`")
        lines.append(f"- Exit: {row['exit']}, elapsed: {row['elapsed_s']}s")
        lines.append(f"- Status: **{row['status']}** — {row['summary']}")
        if row.get("headline"):
            lines.append("```")
            lines.append(row["headline"])
            lines.append("```")
        if row.get("stderr_tail") and row["status"] != "OK":
            lines.append("- stderr tail:")
            lines.append("```")
            lines.append(row["stderr_tail"][-400:])
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-live-agent",
        action="store_true",
        help="Also run the Upwork live-agent stress (~10-15 min, needs codex on PATH).",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    suites: list[tuple[str, callable]] = [
        ("_blind_core", run_blind_core_selftest),
        ("blind_benchmark", run_blind_benchmark),
        ("deception_regression", run_deception_regression),
        ("upstream_benchmark", run_upstream_benchmark),
        ("charitable_adapter", run_charitable_adapter),
        ("swe_bench_calibration", run_swe_bench_calibration),
    ]
    if args.with_live_agent:
        suites.append(("live_agent_upwork", run_live_agent_upwork))
        suites.append(("live_agent_swe_bench", run_live_agent_swe_bench))

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for name, fn in suites:
        print(f"... running {name}", file=sys.stderr, flush=True)
        result = fn()
        status, summary = _verdict(name, result)
        rows.append({"name": name, **result, "status": status, "summary": summary})
        print(f"    [{status}] {summary}", file=sys.stderr, flush=True)
    elapsed_total = time.time() - t0

    md_path = RESULTS / "v1_baseline_report.md"
    json_path = RESULTS / "v1_baseline_report.json"
    md_path.write_text(render_report(rows, elapsed_total), encoding="utf-8")
    json_path.write_text(
        json.dumps({"rows": rows, "elapsed_total_s": elapsed_total}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Exit non-zero if any suite FAILed (ERRORed counts as fail too)
    overall_ok = all(r["status"] == "OK" for r in rows)
    print(json.dumps({
        "ok": overall_ok,
        "elapsed_s": round(elapsed_total, 1),
        "report": str(md_path.relative_to(REPO_ROOT)),
        "json": str(json_path.relative_to(REPO_ROOT)),
        "summary": [{"suite": r["name"], "status": r["status"], "summary": r["summary"]} for r in rows],
    }, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
