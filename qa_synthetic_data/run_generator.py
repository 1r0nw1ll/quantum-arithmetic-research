"""
Generate QA_SYNTHETIC_DATASET_v1.jsonl

Usage:
    python run_generator.py

Output: data/QA_SYNTHETIC_DATASET_v1.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tasks import generate_all_tasks
from verify import verify_row


def main():
    print("Generating QA synthetic tasks (mod-9)...")
    rows = generate_all_tasks()

    # Verify all rows
    n_fail = 0
    for row in rows:
        ok = verify_row(row)
        row["verifier_outcome"] = ok
        if not ok:
            n_fail += 1
            print(f"  FAIL: {row['task_type']} input={row['input']} answer={row['answer']}")

    out_path = os.path.join(os.path.dirname(__file__), "data", "QA_SYNTHETIC_DATASET_v1.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(',', ':')) + "\n")

    # Summary
    by_type = {}
    by_diff = {}
    for row in rows:
        by_type[row["task_type"]] = by_type.get(row["task_type"], 0) + 1
        by_diff[row["difficulty"]] = by_diff.get(row["difficulty"], 0) + 1

    print(f"\nTotal tasks: {len(rows)}  |  Failures: {n_fail}")
    print("By type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {v}")
    print("By difficulty:")
    for k, v in sorted(by_diff.items()):
        print(f"  {k}: {v}")
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
