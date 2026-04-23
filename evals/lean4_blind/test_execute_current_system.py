# noqa: DECL-1 (eval harness test — not empirical QA code)
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXECUTOR = ROOT / "execute_current_system.py"


def main() -> int:
    proc = subprocess.run(
        [sys.executable, str(EXECUTOR)],
        capture_output=True,
        text=True,
        cwd=str(ROOT.parents[1]),
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        return 1
    payload = json.loads(proc.stdout)
    if payload.get("mismatches"):
        print(f"FAIL: unexpected mismatches: {payload['mismatches']}")
        return 1
    generation_rows = [row for row in payload["results"] if row["layer"] == "generation"]
    if len(generation_rows) < 2:
        print("FAIL: expected Lean generation rows")
        return 1
    if not any(row["layer"] == "review" and row["case_id"] == "polished_bad_group_proof" and row["decision"] == "reject" for row in payload["results"]):
        print("FAIL: polished bad proof case was not rejected")
        return 1
    if not any(row["layer"] == "review" and row["case_id"] == "sparse_legit_even_double" and row["decision"] == "accept" for row in payload["results"]):
        print("FAIL: sparse legitimate proof case was not accepted")
        return 1
    if not any(row["layer"] == "repair" and row["case_id"] == "revise_induction_explanation" and row["decision"] == "revise" for row in payload["results"]):
        print("FAIL: revise induction explanation case was not revised")
        return 1
    print("PASS: lean4 execute_current_system starter corpus matched expectations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

