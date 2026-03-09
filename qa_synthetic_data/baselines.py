"""
QA-ORBIT trivial baselines.

Usage:
    python baselines.py data/QA_SYNTHETIC_mod9_test.jsonl [--train data/QA_SYNTHETIC_mod9_train.jsonl]

Baselines:
  1. MajorityBaseline  — predicts most common answer per task type (from train split)
  2. RandomBaseline    — uniform random valid answer per task type
  3. SymbolicBaseline  — exact computation (100% ceiling)
"""

import json
import sys
import os
import argparse
import random

sys.path.insert(0, os.path.dirname(__file__))
from verify import verify_row

TASK_TYPES = ["invariant_pred", "orbit_class", "reachability", "shortest_witness"]


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Majority baseline ──────────────────────────────────────────────────────────

def build_majority(train_rows):
    """Compute most common answer per task_type from training data."""
    counts = {t: {} for t in TASK_TYPES}
    for row in train_rows:
        tt = row["task_type"]
        ans = json.dumps(row["answer"], sort_keys=True)
        counts[tt][ans] = counts[tt].get(ans, 0) + 1
    majority = {}
    for tt, c in counts.items():
        if c:
            best = max(c, key=c.get)
            majority[tt] = json.loads(best)
        else:
            majority[tt] = None
    return majority


def predict_majority(row, majority):
    return majority.get(row["task_type"])


# ── Random baseline ────────────────────────────────────────────────────────────

def predict_random(row):
    tt   = row["task_type"]
    inp  = row["input"]
    m    = inp["modulus"]
    if tt == "invariant_pred":
        return random.randint(0, m - 1)
    elif tt == "orbit_class":
        return random.choice(["cosmos", "satellite", "singularity"])
    elif tt == "reachability":
        return random.random() < 0.5
    elif tt == "shortest_witness":
        return random.randint(0, m - 1)
    return None


# ── Symbolic baseline ──────────────────────────────────────────────────────────

def predict_symbolic(row):
    """Re-run exact computation. Should be 100% correct."""
    from core import qa_norm, compute_orbit, compute_all_orbits, classify_orbit
    inp  = row["input"]
    tt   = row["task_type"]
    m    = inp["modulus"]
    b, e = inp["b"], inp["e"]

    if tt == "invariant_pred":
        return qa_norm(b, e, m)
    elif tt == "orbit_class":
        orbit = compute_orbit(b, e, m)
        all_o = compute_all_orbits(m)
        max_l = max(len(o) for o in all_o.values())
        return classify_orbit(len(orbit), max_l)
    elif tt == "reachability":
        start  = (b, e)
        target = (inp["b_target"], inp["e_target"])
        orbit  = compute_orbit(*start, m)
        return target in orbit
    elif tt == "shortest_witness":
        start  = (b, e)
        target = (inp["b_target"], inp["e_target"])
        orbit  = compute_orbit(*start, m)
        idx    = {s: i for i, s in enumerate(orbit)}
        if target not in idx:
            return -1
        L = len(orbit)
        return (idx[target] - idx[start]) % L
    return None


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(rows, predict_fn, name):
    correct   = {t: 0 for t in TASK_TYPES}
    total     = {t: 0 for t in TASK_TYPES}
    diff_corr = {"easy": 0, "medium": 0, "hard": 0}
    diff_tot  = {"easy": 0, "medium": 0, "hard": 0}

    for row in rows:
        tt   = row["task_type"]
        pred = predict_fn(row)
        diff = row.get("difficulty", "easy")
        hit  = (pred == row["answer"])
        total[tt]      += 1
        diff_tot[diff] += 1
        if hit:
            correct[tt]      += 1
            diff_corr[diff]  += 1

    n = len(rows)
    overall = sum(correct.values()) / n if n else 0.0

    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    print(f"  {'Task type':<22} {'correct':>7} {'total':>6} {'acc':>7}")
    print(f"  {'─'*48}")
    for tt in TASK_TYPES:
        t = total[tt]
        c = correct[tt]
        acc = c / t if t else 0.0
        print(f"  {tt:<22} {c:>7} {t:>6} {acc:>7.3f}")
    print(f"  {'─'*48}")
    print(f"  {'OVERALL':<22} {sum(correct.values()):>7} {n:>6} {overall:>7.3f}")

    print(f"\n  By difficulty:")
    for d in ["easy", "medium", "hard"]:
        t = diff_tot[d]
        c = diff_corr[d]
        acc = c / t if t else 0.0
        print(f"    {d:<10} {c:>5}/{t:<5} {acc:.3f}")

    return overall


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help="JSONL test (or eval) split")
    parser.add_argument("--train", default=None,
                        help="JSONL train split (for majority baseline). "
                             "If omitted, majority is computed from test file itself.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    test_rows  = load_jsonl(args.test_file)
    train_rows = load_jsonl(args.train) if args.train else test_rows

    print(f"Test file:  {args.test_file}  ({len(test_rows)} rows)")
    print(f"Train file: {args.train or '(same as test)'}  ({len(train_rows)} rows)")

    majority = build_majority(train_rows)

    evaluate(test_rows, lambda r: predict_majority(r, majority), "MajorityBaseline")
    evaluate(test_rows, predict_random,   "RandomBaseline")
    evaluate(test_rows, predict_symbolic, "SymbolicBaseline (ceiling)")


if __name__ == "__main__":
    main()
