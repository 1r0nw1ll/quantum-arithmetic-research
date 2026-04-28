#!/usr/bin/env python3
# noqa: DECL-1 (expansion truth runner — not empirical QA code)
"""
Pass-V1.3 docker runner: FAIL_TO_PASS truth on expansion-v2 codex accepts.

For each instance_id that had ≥1 codex variant heuristic-accept:
  1. Pull `swebench/sweb.eval.x86_64.{instance_id_short}` if not present.
  2. For the canonical patch (control) and each accepted codex variant:
     a. docker run inside /testbed
     b. git apply test_patch.diff (adds FAIL_TO_PASS test scaffolding)
     c. git apply <patch_under_test>.diff
     d. pytest <FAIL_TO_PASS test names>
  3. Record per-(task, variant): apply_check, pytest pass/fail, mismatch class.

Outputs:
  evals/swe_bench_blind/results/expansion_v2/expansion_docker_report.{json,md}

The naming map: instance_id `django__django-15863` → docker tag suffix
`django_1776_django-15863`. (`1776` is SWE-Bench's image-publish convention.)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
EXPANSION_RESULTS = ROOT / "results" / "expansion_v2"
SAMPLE_PATH = Path("/home/player2/upstream_corpora/swe_bench_verified/sample_v2_expansion.json")


def _instance_to_image(inst: str) -> str:
    """`django__django-15863` → `swebench/sweb.eval.x86_64.django_1776_django-15863:latest`."""
    suffix = inst.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{suffix}:latest"


def _docker_image_present(image: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return proc.returncode == 0


def _pull_image(image: str, timeout: int = 600) -> dict[str, Any]:
    if _docker_image_present(image):
        return {"pulled": False, "already_present": True, "ok": True}
    proc = subprocess.run(
        ["docker", "pull", image],
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "pulled": proc.returncode == 0,
        "already_present": False,
        "ok": proc.returncode == 0,
        "stderr_tail": proc.stderr[-300:] if proc.stderr else "",
    }


def _docker_run_test(image: str, test_patch: str, patch_under_test: str,
                     fail_to_pass: list[str]) -> dict[str, Any]:
    """Run inside the docker image: apply test_patch + patch under test, run pytest."""
    with tempfile.TemporaryDirectory(prefix="sweb-docker-run-") as td:
        td_path = Path(td)
        (td_path / "test_patch.diff").write_text(test_patch, encoding="utf-8")
        (td_path / "patch.diff").write_text(patch_under_test, encoding="utf-8")
        # Build pytest test selector. Django uses Class.method format inside parens
        # but the test runner accepts the dotted form. Astropy uses pytest IDs already.
        test_args = []
        for tid in fail_to_pass:
            if " (" in tid and tid.endswith(")"):
                tname, cls = tid.split(" (", 1)
                cls = cls.rstrip(")")
                test_args.append(f"{cls}.{tname}")
            else:
                test_args.append(tid)
        # Apply patches + run pytest. Use the conda testbed env where present.
        pytest_cmd = (
            "if [ -x /opt/miniconda3/envs/testbed/bin/python ]; then "
            "PY=/opt/miniconda3/envs/testbed/bin/python; "
            "else PY=python; fi; "
            "cd /testbed && "
            "git apply /work/test_patch.diff 2>&1 && "
            "git apply --check /work/patch.diff 2>&1 || { echo APPLY_CHECK_FAIL; exit 9; } && "
            "git apply /work/patch.diff 2>&1 && "
            f"$PY -m pytest --no-header -q {' '.join(repr(t) for t in test_args)} 2>&1 | tail -8"
        )
        proc = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{td_path}:/work", image, "sh", "-c", pytest_cmd],
            capture_output=True, text=True, timeout=300,
        )
    return {
        "exit": proc.returncode,
        "stdout_tail": (proc.stdout or "").strip().splitlines()[-10:],
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
    }


def main() -> int:
    sample_v2 = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    sample_by_inst = {r["instance_id"]: r for r in sample_v2}

    expansion = json.loads((EXPANSION_RESULTS / "expansion_v2_report.json").read_text(encoding="utf-8"))
    rows = expansion["rows"]

    # Group rows by (instance_id) where decision == accept; collect variants
    accepts_by_inst: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("decision") != "accept":
            continue
        accepts_by_inst.setdefault(r["instance_id"], []).append(r)

    print(f"... {len(accepts_by_inst)} unique tasks with ≥1 accept; running docker FAIL_TO_PASS",
          file=sys.stderr, flush=True)

    docker_results: list[dict[str, Any]] = []
    for i, (inst_id, accept_rows) in enumerate(sorted(accepts_by_inst.items())):
        rec = sample_by_inst[inst_id]
        image = _instance_to_image(inst_id)
        print(f"... [{i+1}/{len(accepts_by_inst)}] {inst_id} ({len(accept_rows)} accepted variants) "
              f"image={image}", file=sys.stderr, flush=True)
        pull = _pull_image(image)
        if not pull["ok"]:
            print(f"    PULL FAILED: {pull['stderr_tail'][:200]}", file=sys.stderr, flush=True)
            docker_results.append({
                "instance_id": inst_id, "variant": "(pull-failed)",
                "image_pull": pull, "test_result": None,
            })
            continue
        fail_to_pass = json.loads(rec["FAIL_TO_PASS"]) if isinstance(rec["FAIL_TO_PASS"], str) else rec["FAIL_TO_PASS"]

        # Run canonical (control) first
        try:
            canon_res = _docker_run_test(image, rec["test_patch"], rec["patch"], fail_to_pass)
        except subprocess.TimeoutExpired:
            canon_res = {"exit": -1, "stdout_tail": ["TIMEOUT"], "stderr_tail": []}
        canon_pass = canon_res["exit"] == 0
        print(f"    canonical (control): exit={canon_res['exit']} "
              f"tail={canon_res['stdout_tail'][-1] if canon_res['stdout_tail'] else ''}",
              file=sys.stderr, flush=True)
        docker_results.append({
            "instance_id": inst_id, "variant": "canonical",
            "image": image,
            "fail_to_pass_count": len(fail_to_pass),
            "canonical_passes": canon_pass,
            "test_result": canon_res,
        })

        # Run each accepted codex variant
        for ar in accept_rows:
            patch_path = REPO_ROOT / ar["saved_to"] / "patch.diff"
            if not patch_path.exists():
                docker_results.append({
                    "instance_id": inst_id, "variant": ar["variant"],
                    "test_result": {"error": f"patch.diff not found at {patch_path}"},
                })
                continue
            patch_text = patch_path.read_text(encoding="utf-8")
            try:
                res = _docker_run_test(image, rec["test_patch"], patch_text, fail_to_pass)
            except subprocess.TimeoutExpired:
                res = {"exit": -1, "stdout_tail": ["TIMEOUT"], "stderr_tail": []}
            actually_passes = res["exit"] == 0
            print(f"    codex {ar['variant']}: exit={res['exit']} "
                  f"tail={res['stdout_tail'][-1] if res['stdout_tail'] else ''}",
                  file=sys.stderr, flush=True)
            docker_results.append({
                "instance_id": inst_id, "variant": ar["variant"],
                "image": image,
                "heuristic_decision": ar["decision"],
                "actually_passes": actually_passes,
                "fail_to_pass_count": len(fail_to_pass),
                "test_result": res,
                "patch_path": str(patch_path.relative_to(REPO_ROOT)),
            })

    # Aggregate
    canonical_passes = sum(1 for r in docker_results if r.get("variant") == "canonical" and r.get("canonical_passes"))
    canonical_total = sum(1 for r in docker_results if r.get("variant") == "canonical")
    codex_runs = [r for r in docker_results if r.get("variant") not in ("canonical", "(pull-failed)") and "actually_passes" in r]
    codex_pass = sum(1 for r in codex_runs if r["actually_passes"])
    codex_total = len(codex_runs)
    summary = {
        "unique_tasks_tested": len(accepts_by_inst),
        "canonical_passes": f"{canonical_passes}/{canonical_total}",
        "codex_heuristic_accepts_actually_pass": f"{codex_pass}/{codex_total}",
        "false_accepts": codex_total - codex_pass,
        "true_positives": codex_pass,
    }
    payload = {"summary": summary, "results": docker_results}
    (EXPANSION_RESULTS / "expansion_docker_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Markdown
    lines = ["# SWE-Bench Expansion v2 Docker FAIL_TO_PASS Report", ""]
    lines.append("## Headline")
    lines.append(f"- Unique tasks with ≥1 codex heuristic-accept: **{summary['unique_tasks_tested']}**")
    lines.append(f"- Canonical patches pass control: **{summary['canonical_passes']}**")
    lines.append(f"- **Codex heuristic-accepts that actually fix the bug: {summary['codex_heuristic_accepts_actually_pass']}**")
    lines.append(f"- **False accepts: {summary['false_accepts']}**")
    lines.append("")
    lines.append("## Per-task (canonical control + accepted codex variants)")
    lines.append("| instance_id | variant | heuristic | actually_passes | image |")
    lines.append("|---|---|---|---|---|")
    for r in docker_results:
        if "actually_passes" in r:
            lines.append(f"| `{r['instance_id']}` | `{r['variant']}` | {r['heuristic_decision']} | {'✓' if r['actually_passes'] else '✗'} | (image present) |")
        elif r.get("variant") == "canonical":
            lines.append(f"| `{r['instance_id']}` | canonical | — | {'✓' if r.get('canonical_passes') else '✗'} | (image present) |")
        else:
            lines.append(f"| `{r['instance_id']}` | {r.get('variant', 'n/a')} | — | — | pull-failed |")
    lines.append("")
    (EXPANSION_RESULTS / "expansion_docker_report.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, **summary,
                      "json_report": str((EXPANSION_RESULTS / "expansion_docker_report.json").relative_to(REPO_ROOT)),
                      "md_report": str((EXPANSION_RESULTS / "expansion_docker_report.md").relative_to(REPO_ROOT))},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
