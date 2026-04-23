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
    generation_rows = [row for row in payload["results"] if row["layer"] == "generation"]
    if len(generation_rows) < 2:
        print("FAIL: expected generation rows")
        return 1
    for row in generation_rows:
        if "result_bundle" not in row:
            print(f"FAIL: missing result_bundle for {row['case_id']}")
            return 1
        if row["scores"]["formal_validity_score"] is None or row["scores"]["external_admissibility_score"] is None:
            print(f"FAIL: missing top-line scores for {row['case_id']}")
            return 1
    print("PASS: execute_current_system generation path produced scored result bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
