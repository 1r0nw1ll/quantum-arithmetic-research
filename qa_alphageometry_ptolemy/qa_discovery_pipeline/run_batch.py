#!/usr/bin/env python3
"""
qa_discovery_pipeline/run_batch.py

Reads a QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1 JSON, executes each queued run
via qa_conjecture_prove/run_episode.py, emits per-run pipeline run records,
and bundles everything into a QA_DISCOVERY_BUNDLE_SCHEMA.v1 artifact with
a computed this_bundle_hash.

Usage:
  python3 qa_discovery_pipeline/run_batch.py \
    --plan qa_discovery_pipeline/fixtures/plan_valid.json \
    --out_dir qa_discovery_pipeline/out \
    [--toolchain_id lean4.12.0] \
    [--episode-timeout-s 30] \
    [--prev_bundle_hash 0000...] \
    [--created_utc 2026-02-11T00:00:00Z]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_canonical(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_canonical(obj) + "\n")


# ---------------------------------------------------------------------------
# Minimal plan validation
# ---------------------------------------------------------------------------

def _validate_plan(plan: Dict) -> None:
    assert plan.get("schema_id") == "QA_DISCOVERY_BATCH_PLAN_SCHEMA.v1", "bad schema_id"
    assert isinstance(plan.get("run_queue"), list) and len(plan["run_queue"]) > 0, "empty run_queue"
    det = plan.get("determinism", {})
    assert det.get("canonical_json") is True, "canonical_json must be true"
    assert plan.get("merkle_parent"), "missing merkle_parent"


# ---------------------------------------------------------------------------
# Episode runner (subprocess)
# ---------------------------------------------------------------------------

def _run_episode(root: str, episode_path: str, out_dir: str,
                 k: int, toolchain_id: str, timeout_s: float | None) -> Tuple[int, str, str, bool]:
    harness = os.path.join(root, "qa_conjecture_prove", "run_episode.py")
    if not os.path.exists(harness):
        return (127, "", f"missing harness: {harness}", False)
    cmd = [sys.executable, harness,
           "--episode", episode_path,
           "--out_dir", out_dir,
           "--k", str(k),
           "--toolchain_id", toolchain_id]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return p.returncode, p.stdout.strip(), p.stderr.strip(), False
    except subprocess.TimeoutExpired as exc:
        # With text=True, these should be str (or None), but keep it defensive.
        out = exc.stdout or ""
        err = exc.stderr or ""
        if not isinstance(out, str):
            out = out.decode("utf-8", errors="replace")
        if not isinstance(err, str):
            err = err.decode("utf-8", errors="replace")
        return 124, out.strip(), err.strip(), True
    except Exception as exc:
        return 125, "", f"exception: {type(exc).__name__}: {exc}", False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Execute QA_DISCOVERY_BATCH_PLAN and emit run records + bundle.")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--toolchain_id", default="lean4.12.0")
    ap.add_argument(
        "--episode-timeout-s",
        type=float,
        default=30,
        help="Per-episode subprocess timeout (seconds); 0 disables the timeout.",
    )
    ap.add_argument("--prev_bundle_hash", default="0" * 64)
    ap.add_argument("--created_utc", default=None)
    ap.add_argument("--os_id", default="linux")
    args = ap.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_dir)  # qa_alphageometry_ptolemy/

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)
    _validate_plan(plan)

    created = args.created_utc or _now_utc()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    toolchain = {"python": py_ver, "os": args.os_id, "toolchain_id": args.toolchain_id}

    agent_id = plan["agent_id"]
    pipeline_id = plan["pipeline_id"]
    plan_id = plan["plan_id"]
    merkle_parent = plan["merkle_parent"]

    os.makedirs(args.out_dir, exist_ok=True)
    timeout_s = None if float(args.episode_timeout_s) <= 0 else float(args.episode_timeout_s)

    run_refs: List[Dict[str, str]] = []
    artifact_refs: List[Dict[str, str]] = []
    fail_counts: Dict[str, int] = {}
    runs_total = 0
    runs_success = 0

    for item in plan["run_queue"]:
        run_id = str(item["run_id"])
        episode_ref = item["episode_ref"]
        k = int(item["k"])
        runs_total += 1

        per_run_out = os.path.join(args.out_dir, f"out_{run_id}")
        os.makedirs(per_run_out, exist_ok=True)

        ep_path = str(episode_ref["path_or_hash"])
        if not os.path.isabs(ep_path):
            ep_path = os.path.join(root, ep_path)

        steps: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, str]] = []
        episode_ok = True

        # Step 0: load
        if os.path.exists(ep_path):
            steps.append({"step_index": 0, "op": "LOAD_EPISODE",
                          "result": {"status": "SUCCESS", "witness_hash": _sha256("LOAD|" + ep_path)}})
        else:
            steps.append({"step_index": 0, "op": "LOAD_EPISODE",
                          "result": {"status": "FAIL", "fail_type": "EPISODE_NOT_FOUND",
                                     "invariant_diff": {"path": ep_path}}})
            episode_ok = False

        # Step 1: validate + run
        if episode_ok:
            rc, out, err, timed_out = _run_episode(
                root,
                ep_path,
                per_run_out,
                k,
                args.toolchain_id,
                timeout_s=timeout_s,
            )
            if rc == 0:
                steps.append({"step_index": 1, "op": "VALIDATE_EPISODE",
                              "result": {"status": "SUCCESS", "witness_hash": _sha256("OK|" + run_id)}})
            else:
                fail_type = "PIPELINE_TIMEOUT" if timed_out else "PIPELINE_SUBPROCESS_FAIL"
                steps.append({"step_index": 1, "op": "VALIDATE_EPISODE",
                              "result": {"status": "FAIL", "fail_type": fail_type,
                                         "invariant_diff": {
                                             "rc": rc,
                                             "timeout_s": timeout_s,
                                             "stdout": out[:200],
                                             "stderr": err[:200],
                                         }}})
                episode_ok = False

        # Step 2: frontier artifact
        frontier_path = os.path.join(per_run_out, "frontier_snapshot.json")
        if episode_ok and os.path.exists(frontier_path):
            steps.append({"step_index": 2, "op": "EMIT_FRONTIER",
                          "result": {"status": "SUCCESS", "witness_hash": _sha256("FR|" + run_id)}})
            artifacts.append({"family": "QA_CONJECTURE_PROVE_CONTROL_LOOP.v1",
                              "schema_id": "QA_FRONTIER_SNAPSHOT_SCHEMA.v1",
                              "path_or_hash": os.path.relpath(frontier_path, root)})
        elif episode_ok:
            steps.append({"step_index": 2, "op": "EMIT_FRONTIER",
                          "result": {"status": "FAIL", "fail_type": "ARTIFACT_MISSING",
                                     "invariant_diff": {"expected": frontier_path}}})
            episode_ok = False

        # Step 3: receipt artifact
        receipt_path = os.path.join(per_run_out, "bounded_return_receipt.json")
        if episode_ok and os.path.exists(receipt_path):
            steps.append({"step_index": 3, "op": "EMIT_RETURN_RECEIPT",
                          "result": {"status": "SUCCESS", "witness_hash": _sha256("RR|" + run_id)}})
            artifacts.append({"family": "QA_CONJECTURE_PROVE_CONTROL_LOOP.v1",
                              "schema_id": "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1",
                              "path_or_hash": os.path.relpath(receipt_path, root)})
        elif episode_ok:
            steps.append({"step_index": 3, "op": "EMIT_RETURN_RECEIPT",
                          "result": {"status": "FAIL", "fail_type": "ARTIFACT_MISSING",
                                     "invariant_diff": {"expected": receipt_path}}})
            episode_ok = False

        # Build run record
        status = "SUCCESS" if episode_ok else "FAIL"
        run_obj: Dict[str, Any] = {
            "schema_id": "QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1",
            "run_id": run_id,
            "created_utc": created,
            "agent_id": agent_id,
            "pipeline_id": pipeline_id,
            "toolchain": toolchain,
            "inputs": {
                "episode_ref": episode_ref,
                "k": k,
                "determinism": {
                    "canonical_json": True,
                    "frontier_sort": "(-priority,state_hash)",
                    "bfs_tie_breaker": "(generator,step_index,output_hash)",
                },
            },
            "execution": {"steps": steps},
            "outputs": {"artifacts": artifacts},
            "result": {"status": status},
            "merkle_parent": merkle_parent,
        }
        if status == "FAIL":
            # Lift last FAIL step's fail_type
            for s in reversed(steps):
                if s["result"]["status"] == "FAIL":
                    run_obj["result"]["fail_type"] = s["result"]["fail_type"]
                    run_obj["result"]["invariant_diff"] = s["result"]["invariant_diff"]
                    break
            else:
                run_obj["result"]["fail_type"] = "PIPELINE_FAILED"
                run_obj["result"]["invariant_diff"] = {"reason": "unknown"}
            fail_counts[run_obj["result"]["fail_type"]] = fail_counts.get(run_obj["result"]["fail_type"], 0) + 1
        else:
            runs_success += 1

        run_path = os.path.join(args.out_dir, f"run_{run_id}.json")
        _write_canonical(run_path, run_obj)
        run_refs.append({"family": "QA_DISCOVERY_PIPELINE.v1",
                         "path_or_hash": os.path.relpath(run_path, root)})
        artifact_refs.extend(artifacts)

    # Bundle
    bundle_id = _sha256(plan_id + "|bundle|" + created)
    # Ensure artifact_refs is non-empty (schema requires minItems: 1)
    if not artifact_refs:
        artifact_refs = [{"family": "QA_DISCOVERY_PIPELINE.v1",
                          "schema_id": "QA_DISCOVERY_PIPELINE_RUN_SCHEMA.v1",
                          "path_or_hash": run_refs[0]["path_or_hash"]}]

    bundle: Dict[str, Any] = {
        "schema_id": "QA_DISCOVERY_BUNDLE_SCHEMA.v1",
        "bundle_id": bundle_id,
        "created_utc": created,
        "agent_id": agent_id,
        "pipeline_id": pipeline_id,
        "plan_ref": {"family": "QA_DISCOVERY_PIPELINE.v1",
                     "path_or_hash": os.path.relpath(os.path.abspath(args.plan), root)},
        "run_refs": run_refs,
        "artifact_refs": artifact_refs,
        "summary": {"runs_total": runs_total, "runs_success": runs_success, "fail_counts": fail_counts},
        "hash_chain": {"prev_bundle_hash": args.prev_bundle_hash, "this_bundle_hash": ""},
    }
    # Compute self-hash
    bundle["hash_chain"]["this_bundle_hash"] = _sha256(_canonical(bundle))

    bundle_path = os.path.join(args.out_dir, f"bundle_{bundle_id[:16]}.json")
    _write_canonical(bundle_path, bundle)

    print("VALID")
    for rr in run_refs:
        print(f"WROTE {rr['path_or_hash']}")
    print(f"WROTE {os.path.relpath(bundle_path, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
