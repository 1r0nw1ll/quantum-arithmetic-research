"""
Generate QA synthetic reasoning datasets with orbit-family-based splits.

Usage:
    python run_generator.py [--modulus 9] [--modulus 24]
    python run_generator.py  # runs both

Outputs per modulus:
    data/QA_SYNTHETIC_mod{N}_all.jsonl         — full dataset
    data/QA_SYNTHETIC_mod{N}_train.jsonl       — cosmos orbits 0..N_train-1
    data/QA_SYNTHETIC_mod{N}_dev.jsonl         — cosmos orbits N_train..N_dev-1
    data/QA_SYNTHETIC_mod{N}_test.jsonl        — held-out orbit families

Split logic:
    - singularity + all satellite states → train (easy scaffold)
    - cosmos orbits: sorted by orbit index, 70/15/15 split
    - OOD test includes all satellite states of types not seen in train
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from tasks import generate_all_tasks
from verify import verify_row
from core import compute_all_orbits, classify_orbit


def split_tasks(rows: list, modulus: int) -> dict:
    """Split rows into train/dev/test by orbit family."""
    # Identify distinct cosmos orbit families by their canonical first state
    all_orbits = compute_all_orbits(modulus)
    max_len = max(len(o) for o in all_orbits.values())

    # Map each cosmos orbit to a family index (sorted for determinism)
    cosmos_roots = sorted(set(
        tuple(orbit[0]) for orbit in all_orbits.values()
        if len(orbit) == max_len
    ))
    n = len(cosmos_roots)
    n_train = max(1, int(n * 0.70))
    n_dev   = max(1, int(n * 0.15))
    # rest go to test

    train_roots = set(cosmos_roots[:n_train])
    dev_roots   = set(cosmos_roots[n_train:n_train + n_dev])
    test_roots  = set(cosmos_roots[n_train + n_dev:])

    def get_orbit_root(b, e):
        orbit = all_orbits.get((b, e))
        if orbit is None:
            return None
        return tuple(orbit[0])

    train_rows, dev_rows, test_rows = [], [], []

    for row in rows:
        inp = row["input"]
        b, e = inp["b"], inp["e"]
        root = get_orbit_root(b, e)
        orbit = all_orbits.get((b, e), [])
        cls = classify_orbit(len(orbit), max_len)

        if cls in ("singularity", "satellite"):
            # Easy scaffold — all in train
            train_rows.append(row)
        elif root in train_roots:
            train_rows.append(row)
        elif root in dev_roots:
            dev_rows.append(row)
        else:
            test_rows.append(row)

    return {"train": train_rows, "dev": dev_rows, "test": test_rows}


def run_one_modulus(modulus: int, data_dir: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Generating mod-{modulus} tasks...")
    rows = generate_all_tasks(modulus)

    # Verify
    n_fail = 0
    for row in rows:
        ok = verify_row(row)
        row["verifier_outcome"] = ok
        if not ok:
            n_fail += 1
            print(f"  FAIL: {row['task_type']} input={row['input']} answer={row['answer']}")

    # Split
    splits = split_tasks(rows, modulus)

    # Write files
    base = f"QA_SYNTHETIC_mod{modulus}"
    paths = {}
    for split_name, split_rows in [("all", rows)] + list(splits.items()):
        path = os.path.join(data_dir, f"{base}_{split_name}.jsonl")
        with open(path, "w") as f:
            for row in split_rows:
                f.write(json.dumps(row, separators=(',', ':')) + "\n")
        paths[split_name] = path

    # Summary
    by_type = {}
    by_diff = {}
    for row in rows:
        by_type[row["task_type"]] = by_type.get(row["task_type"], 0) + 1
        by_diff[row["difficulty"]] = by_diff.get(row["difficulty"], 0) + 1

    print(f"mod-{modulus}: {len(rows)} tasks | {n_fail} failures")
    print(f"  By type:       " + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    print(f"  By difficulty: " + "  ".join(f"{k}={v}" for k, v in sorted(by_diff.items())))
    print(f"  Splits:        train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}")
    print(f"  Written to:    {data_dir}/{base}_*.jsonl")

    return {
        "modulus": modulus,
        "total": len(rows),
        "failures": n_fail,
        "by_type": by_type,
        "by_diff": by_diff,
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, action="append",
                        help="Modulus to generate (may be repeated). Default: 9 and 24.")
    args = parser.parse_args()

    moduli = args.modulus if args.modulus else [9, 24]

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    summaries = []
    for m in moduli:
        summaries.append(run_one_modulus(m, data_dir))

    print(f"\n{'='*60}")
    print("Done.")
    for s in summaries:
        print(f"  mod-{s['modulus']}: {s['total']} tasks, {s['failures']} failures, "
              f"splits {s['split_sizes']}")


if __name__ == "__main__":
    main()
