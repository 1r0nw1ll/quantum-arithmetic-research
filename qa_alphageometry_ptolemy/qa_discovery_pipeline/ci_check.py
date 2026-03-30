#!/usr/bin/env python3
"""
qa_discovery_pipeline/ci_check.py

CI mini-check for QA_DISCOVERY_PIPELINE.v1 outputs.

Validates:
  1) Each run record (run_*.json) against:
       qa_discovery_pipeline/qa_discovery_pipeline_validator.py run <file> --ci
  2) Each conjecture-prove artifact referenced in each run record against:
       qa_conjecture_prove/qa_conjecture_prove_validator.py frontier|receipt <file> --ci
  3) Each bundle (bundle_*.json) against:
       qa_discovery_pipeline/qa_discovery_pipeline_validator.py bundle <file> --ci
  4) Bundle summary.fail_counts consistency with observed run statuses

Batch policy:
  - default: FAIL batch if ANY run has result.status == "FAIL"
  - --allow_fail: allow FAIL runs, but REQUIRE typed failure + a valid receipt artifact

Usage:
  python3 qa_discovery_pipeline/ci_check.py --out_dir /tmp/disc_test
  python3 qa_discovery_pipeline/ci_check.py --out_dir /tmp/disc_test --allow_fail

Exit codes:
  0 PASS
  1 FAIL (validation or policy failure)
  2 FAIL (no runs found / missing validators)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _repo_root() -> str:
    this_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(this_dir, ".."))


def _run_cmd(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    merged = out + ("\n" + err if err else "")
    return p.returncode, merged.strip()


def _resolve(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(root, p)


def _validate_pipeline(root: str, kind: str, path: str) -> Tuple[bool, str]:
    v = os.path.join(root, "qa_discovery_pipeline", "qa_discovery_pipeline_validator.py")
    if not os.path.exists(v):
        return False, f"missing discovery pipeline validator: {v}"
    rc, out = _run_cmd([sys.executable, v, kind, path, "--ci"])
    return rc == 0, out


def _validate_cp(root: str, kind: str, path: str) -> Tuple[bool, str]:
    v = os.path.join(root, "qa_conjecture_prove", "qa_conjecture_prove_validator.py")
    if not os.path.exists(v):
        return False, f"missing conjecture-prove validator: {v}"
    rc, out = _run_cmd([sys.executable, v, kind, path, "--ci"])
    return rc == 0, out


def main() -> int:
    ap = argparse.ArgumentParser(description="CI-check discovery pipeline batch outputs.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--allow_fail", action="store_true",
                    help="Allow FAIL runs if typed + receipt present.")
    args = ap.parse_args()

    root = _repo_root()
    out_dir = os.path.abspath(args.out_dir)

    run_paths = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
    if not run_paths:
        print(f"[FAIL] no run_*.json found in {out_dir}")
        return 2

    failures: List[str] = []
    validated_cp = 0
    observed_fail_counts: Dict[str, int] = {}
    observed_total = 0
    observed_success = 0

    for rp in run_paths:
        observed_total += 1

        # 1) Validate run record schema
        ok, msg = _validate_pipeline(root, "run", rp)
        if not ok:
            failures.append(f"[RUN_SCHEMA_FAIL] {os.path.basename(rp)} :: {msg}")
            continue
        print(msg)

        run_obj = _read_json(rp)
        run_status = run_obj.get("result", {}).get("status")
        run_fail_type = run_obj.get("result", {}).get("fail_type")
        run_diff = run_obj.get("result", {}).get("invariant_diff")

        if run_status == "SUCCESS":
            observed_success += 1
        elif run_status == "FAIL" and run_fail_type:
            observed_fail_counts[run_fail_type] = observed_fail_counts.get(run_fail_type, 0) + 1

        artifacts = run_obj.get("outputs", {}).get("artifacts", [])
        has_frontier = any(a.get("schema_id") == "QA_FRONTIER_SNAPSHOT_SCHEMA.v1" for a in artifacts)
        has_receipt = any(a.get("schema_id") == "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1" for a in artifacts)

        # 2) Validate referenced conjecture-prove artifacts
        for a in artifacts:
            schema_id = a.get("schema_id")
            p = a.get("path_or_hash")
            if not p:
                failures.append(f"[ARTIFACT_REF_MISSING] {os.path.basename(rp)} :: missing path_or_hash")
                continue

            abs_p = _resolve(root, p)
            if not os.path.exists(abs_p):
                failures.append(f"[ARTIFACT_MISSING] {os.path.basename(rp)} :: {p}")
                continue

            if schema_id == "QA_FRONTIER_SNAPSHOT_SCHEMA.v1":
                ok2, msg2 = _validate_cp(root, "frontier", abs_p)
                if not ok2:
                    failures.append(f"[FRONTIER_INVALID] {os.path.basename(rp)} :: {msg2}")
                else:
                    print(msg2)
                    validated_cp += 1

            elif schema_id == "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1":
                ok2, msg2 = _validate_cp(root, "receipt", abs_p)
                if not ok2:
                    failures.append(f"[RECEIPT_INVALID] {os.path.basename(rp)} :: {msg2}")
                else:
                    print(msg2)
                    validated_cp += 1

        # 3) Batch policy enforcement
        if run_status == "FAIL":
            if not args.allow_fail:
                failures.append(f"[RUN_FAIL] {os.path.basename(rp)} :: status=FAIL fail_type={run_fail_type}")
            else:
                if not run_fail_type or not isinstance(run_diff, dict):
                    failures.append(f"[RUN_FAIL_UNTYPED] {os.path.basename(rp)} :: missing fail_type/invariant_diff")
                if not has_receipt:
                    failures.append(f"[RUN_FAIL_NO_RECEIPT] {os.path.basename(rp)} :: FAIL run must reference receipt")

        if run_status == "SUCCESS":
            if not has_frontier:
                failures.append(f"[SUCCESS_MISSING_FRONTIER] {os.path.basename(rp)} :: expected frontier artifact")
            if not has_receipt:
                failures.append(f"[SUCCESS_MISSING_RECEIPT] {os.path.basename(rp)} :: expected receipt artifact")

    # 4) Validate bundle(s)
    bundle_paths = sorted(glob.glob(os.path.join(out_dir, "bundle_*.json")))
    for bp in bundle_paths:
        ok, msg = _validate_pipeline(root, "bundle", bp)
        if not ok:
            failures.append(f"[BUNDLE_SCHEMA_FAIL] {os.path.basename(bp)} :: {msg}")
            continue
        print(msg)

        # Verify summary consistency
        bundle_obj = _read_json(bp)
        summ = bundle_obj.get("summary", {})
        b_total = summ.get("runs_total", -1)
        b_success = summ.get("runs_success", -1)
        b_fails = summ.get("fail_counts", {})

        if b_total != observed_total:
            failures.append(
                f"[BUNDLE_SUMMARY_MISMATCH] {os.path.basename(bp)} :: "
                f"runs_total={b_total} but observed {observed_total}"
            )
        if b_success != observed_success:
            failures.append(
                f"[BUNDLE_SUMMARY_MISMATCH] {os.path.basename(bp)} :: "
                f"runs_success={b_success} but observed {observed_success}"
            )
        if b_fails != observed_fail_counts:
            failures.append(
                f"[BUNDLE_SUMMARY_MISMATCH] {os.path.basename(bp)} :: "
                f"fail_counts={json.dumps(b_fails, sort_keys=True)} but observed "
                f"{json.dumps(observed_fail_counts, sort_keys=True)}"
            )

    if failures:
        print("FAIL")
        for f in failures:
            print(f)
        return 1

    print("PASS")
    print(f"validated_run_records={len(run_paths)} validated_cp_artifacts={validated_cp} validated_bundles={len(bundle_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
