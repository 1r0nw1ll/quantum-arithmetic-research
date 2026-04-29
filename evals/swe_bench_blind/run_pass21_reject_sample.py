#!/usr/bin/env python3
# noqa: DECL-1 (Pass-21 reject-sample truth runner — not empirical QA code)
"""
Pass 21: SWE-Bench reject-sample FAIL_TO_PASS execution.

The Pass-V1.3 expansion ran FAIL_TO_PASS on the 6 cascade survivors
(heuristic-accept ∧ apply-check passes). 6/6 actually fixed their bugs,
giving us strong survivor truth. But the synthesis flagged this as
*not* full-confusion truth: patches the cascade rejected were never
executed, so we have precision on survivors but no recall measurement
on the underlying corpus.

This runner closes that gap. For every V1.3 codex output that the
cascade rejected (heuristic decision is `reject` or `revise`) and that
actually saved a `patch.diff`, run the full FAIL_TO_PASS pipeline:

  1. Pull the official SWE-Bench docker image if not present
  2. Apply test_patch (gold tests for FAIL_TO_PASS)
  3. git apply --check the codex patch (structural validity)
  4. git apply the codex patch
  5. Run pytest on the FAIL_TO_PASS test names

Outputs the 4-cell confusion matrix. The interesting cell is
**FALSE REJECT**: cascade said reject/revise but the patch actually
fixes the bug. That's a recall miss — currently unmeasured.

Outputs:
  evals/swe_bench_blind/results/pass21_reject_sample/pass21_report.{json,md}
"""
from __future__ import annotations

import json
import importlib.util
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "pass21_reject_sample"
SAMPLE_PATH = Path("/home/player2/upstream_corpora/swe_bench_verified/sample_v2_expansion.json")
EXPANSION_RESULTS = ROOT / "results" / "expansion_v2"

# Reuse the docker runner helpers from Pass V1.3.
_spec = importlib.util.spec_from_file_location(
    "expansion_docker", ROOT / "run_expansion_docker.py"
)
_expansion_docker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_expansion_docker)


def _filter_codex_to_production(patch_text: str) -> str:
    """Drop test-file hunks from a codex patch (same fix as Pass V1.3 docker runner).

    Codex frequently adds its own tests that conflict with the gold
    test_patch. We apply only production hunks.
    """
    out_lines: list[str] = []
    keep = True
    for line in patch_text.splitlines(keepends=True):
        if line.startswith("diff --git"):
            # New file diff header. Decide whether to keep based on path.
            paths = re.findall(r" a/(\S+) b/(\S+)", line)
            if paths:
                a_path, b_path = paths[0]
                is_test = (
                    "/test_" in f"/{a_path}" or "/test_" in f"/{b_path}"
                    or a_path.startswith("test_") or b_path.startswith("test_")
                    or "/tests/" in a_path or "/tests/" in b_path
                    or a_path.startswith("tests/") or b_path.startswith("tests/")
                )
                keep = not is_test
            else:
                keep = True
        if keep:
            out_lines.append(line)
    return "".join(out_lines)


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    sample_by_inst = {r["instance_id"]: r for r in sample}

    expansion = json.loads((EXPANSION_RESULTS / "expansion_v2_report.json").read_text(encoding="utf-8"))
    rows = expansion["rows"]

    # Reject-sample: heuristic decision is not accept AND not timeout AND patch.diff exists.
    candidates: list[dict[str, Any]] = []
    for r in rows:
        if r.get("decision") == "accept":
            continue
        if r.get("decision") == "(timeout)":
            continue
        patch_path = REPO_ROOT / r["saved_to"] / "patch.diff"
        if not patch_path.exists():
            continue
        candidates.append(r)

    print(
        f"... Pass 21 reject-sample: {len(candidates)} heuristic-non-accept "
        f"codex outputs with saved patch.diff",
        file=sys.stderr, flush=True,
    )

    docker_results: list[dict[str, Any]] = []
    for i, r in enumerate(sorted(candidates, key=lambda x: (x["instance_id"], x["variant"]))):
        inst_id = r["instance_id"]
        variant = r["variant"]
        rec = sample_by_inst[inst_id]
        image = _expansion_docker._instance_to_image(inst_id)
        print(
            f"... [{i+1}/{len(candidates)}] {inst_id} :: {variant} (heuristic={r['decision']}) image={image}",
            file=sys.stderr, flush=True,
        )
        pull = _expansion_docker._pull_image(image)
        if not pull["ok"]:
            docker_results.append({
                "instance_id": inst_id, "variant": variant,
                "heuristic_decision": r["decision"],
                "image_pull_ok": False,
                "outcome": "image_pull_failed",
                "error": pull.get("stderr_tail", "")[:300],
            })
            continue

        patch_path = REPO_ROOT / r["saved_to"] / "patch.diff"
        patch_text = patch_path.read_text(encoding="utf-8")
        production_only = _filter_codex_to_production(patch_text)
        fail_to_pass = (
            json.loads(rec["FAIL_TO_PASS"]) if isinstance(rec["FAIL_TO_PASS"], str)
            else rec["FAIL_TO_PASS"]
        )

        try:
            res = _expansion_docker._docker_run_test(
                image, rec["test_patch"], production_only, fail_to_pass
            )
        except subprocess.TimeoutExpired:
            res = {"exit": -1, "stdout_tail": ["TIMEOUT"], "stderr_tail": []}

        # Categorize the outcome:
        #   - apply_check_failed: the patch literally won't apply
        #   - applied_but_test_fails: the patch applies but tests don't pass
        #   - actually_passes: the patch applies AND tests pass — FALSE REJECT
        tail_text = "\n".join(res["stdout_tail"])
        if "APPLY_CHECK_FAIL" in tail_text:
            outcome = "apply_check_failed"
            classification = "true_reject_structural"
        elif res["exit"] == 0:
            outcome = "actually_passes"
            classification = "false_reject_recall_miss"
        else:
            outcome = "applied_but_test_fails"
            classification = "true_reject_semantic"

        print(
            f"    {variant}: outcome={outcome} classification={classification} exit={res['exit']}",
            file=sys.stderr, flush=True,
        )
        docker_results.append({
            "instance_id": inst_id,
            "variant": variant,
            "heuristic_decision": r["decision"],
            "image_pull_ok": True,
            "outcome": outcome,
            "classification": classification,
            "exit": res["exit"],
            "stdout_tail": res["stdout_tail"][-6:],
            "patch_path": str(patch_path.relative_to(REPO_ROOT)),
        })

    # Aggregate
    classifications = Counter(d["classification"] for d in docker_results if "classification" in d)
    pull_failed = sum(1 for d in docker_results if d.get("outcome") == "image_pull_failed")

    summary = {
        "total_reject_sample": len(candidates),
        "tested": len(docker_results) - pull_failed,
        "image_pull_failures": pull_failed,
        "classifications": dict(classifications),
        # Compose with the V1.3 survivor numbers for full confusion matrix
        # on the testable subset (post-cascade survivor truth + reject-sample
        # truth):
        "v1_3_cascade_survivors_actually_pass": 6,  # known from Pass V1.3
        "v1_3_cascade_survivors_total": 6,
        "false_reject_count": classifications.get("false_reject_recall_miss", 0),
        "true_reject_structural": classifications.get("true_reject_structural", 0),
        "true_reject_semantic": classifications.get("true_reject_semantic", 0),
    }

    payload = {"summary": summary, "results": docker_results}
    (RESULTS_ROOT / "pass21_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Markdown
    lines = ["# Pass 21 — SWE-Bench Reject-Sample FAIL_TO_PASS Report", ""]
    lines.append("## Goal")
    lines.append(
        "Convert V1.3's survivor-only truth into a full confusion-matrix truth "
        "on the testable subset by running FAIL_TO_PASS against codex outputs "
        "the cascade *rejected* (heuristic non-accept). The interesting cell "
        "is `false_reject_recall_miss`: cascade said reject/revise but the "
        "patch actually fixes the bug."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append(f"- Reject-sample size: **{summary['total_reject_sample']}** "
                 "(heuristic non-accept ∧ patch.diff saved)")
    lines.append(f"- Tested: **{summary['tested']}** "
                 f"(image-pull failures: {summary['image_pull_failures']})")
    lines.append("")
    lines.append("### Classifications on the reject sample")
    lines.append(f"- True reject (structural — won't apply): **{summary['true_reject_structural']}**")
    lines.append(f"- True reject (semantic — applies but tests fail): **{summary['true_reject_semantic']}**")
    lines.append(f"- **False reject (would have actually passed): {summary['false_reject_count']}**")
    lines.append("")
    lines.append("## Full confusion matrix on the testable subset")
    fr = summary["false_reject_count"]
    tp = summary["v1_3_cascade_survivors_actually_pass"]
    tr_struct = summary["true_reject_structural"]
    tr_sem = summary["true_reject_semantic"]
    fa = 0  # V1.3 already showed 0 false accepts on cascade survivors
    lines.append("| | actually_fixes | does_not_fix |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **cascade=accept** (V1.3 survivors) | {tp} (TP) | {fa} (FA) |")
    lines.append(f"| **cascade=reject/revise** (this pass) | {fr} (FR) | {tr_struct + tr_sem} (TN) |")
    lines.append("")
    if (tp + fr) > 0:
        recall = tp / (tp + fr)
        lines.append(f"- Precision (TP / (TP + FA)) = **{tp}/{tp + fa} = {tp/(tp+fa)*100:.1f}%**" if (tp + fa) else "")
        lines.append(f"- Recall (TP / (TP + FR)) = **{tp}/{tp + fr} = {recall*100:.1f}%**")
    lines.append("")
    lines.append("## Per-output detail")
    lines.append("| instance_id | variant | heuristic | outcome | classification |")
    lines.append("|---|---|---|---|---|")
    for d in docker_results:
        cls = d.get("classification", "—")
        outcome = d.get("outcome", "—")
        lines.append(
            f"| `{d['instance_id']}` | `{d['variant']}` | "
            f"{d.get('heuristic_decision', '—')} | {outcome} | {cls} |"
        )
    lines.append("")
    (RESULTS_ROOT / "pass21_report.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "ok": True,
        **summary,
        "json_report": str((RESULTS_ROOT / "pass21_report.json").relative_to(REPO_ROOT)),
        "md_report": str((RESULTS_ROOT / "pass21_report.md").relative_to(REPO_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
