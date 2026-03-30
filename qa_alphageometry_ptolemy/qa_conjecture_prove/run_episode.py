#!/usr/bin/env python3
"""
qa_conjecture_prove/run_episode.py

Tiny harness: loads an episode JSON, re-validates it, then emits:
  (1) a frontier snapshot JSON
  (2) a bounded return-in-k receipt JSON

Canonical JSON (sorted keys, minimal separators). Deterministic.

Usage:
  python run_episode.py --episode <path> --out_dir <dir> [--k 2] [--toolchain_id lean4.12.0]
  python run_episode.py --self-test
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import deque
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Import the validator (sibling module)
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_this_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from qa_conjecture_prove.qa_conjecture_prove_validator import validate_episode


# ---------------------------------------------------------------------------
# Graph extraction from episode
# ---------------------------------------------------------------------------

def _episode_edges(ep: Dict) -> List[Tuple[str, str, Dict]]:
    edges = []
    for s in sorted(ep["steps"], key=lambda x: x["step_index"]):
        edges.append((str(s["input_hash"]), str(s["output_hash"]), s))
    return edges


def _compute_frontier(ep: Dict) -> Tuple[List[str], List[Dict]]:
    steps = sorted(ep["steps"], key=lambda x: x["step_index"])
    inputs = [str(s["input_hash"]) for s in steps]
    outputs = [str(s["output_hash"]) for s in steps]

    all_states = sorted(set(inputs + outputs))
    sinks = set(outputs) - set(inputs)
    sinks.add(str(steps[-1]["output_hash"]))

    last_seen = {}
    for s in steps:
        last_seen[str(s["output_hash"])] = int(s["step_index"])

    denom = max(1, len(steps) - 1)
    tmp = []
    for st in sinks:
        recency = last_seen.get(st, -1)
        priority = 0.5 if recency < 0 else float(recency) / float(denom)
        tmp.append((priority, st))
    tmp.sort(key=lambda t: (-t[0], t[1]))

    frontier = [{"state_hash": st, "layer": "formal", "priority": round(p, 6)} for p, st in tmp]
    return all_states, frontier


def _find_return(ep: Dict, start: str, target: str, k: int) -> Tuple[bool, List[Dict], Dict]:
    edges = _episode_edges(ep)
    adj: Dict[str, List[Tuple[str, Dict]]] = {}
    for u, v, step in edges:
        adj.setdefault(u, []).append((v, step))
    for u in adj:
        adj[u].sort(key=lambda t: (
            str(t[1].get("action", {}).get("generator", "")),
            int(t[1].get("step_index", 0)),
            str(t[0]),
        ))

    q: deque = deque()
    q.append((start, []))
    visited: Dict[str, int] = {start: 0}
    expanded = 0

    while q:
        state, path = q.popleft()
        depth = len(path)
        expanded += 1

        if state == target and depth > 0:
            return True, path, {"visited_nodes": len(visited), "expanded": expanded}
        if depth >= k:
            continue

        for v, step in adj.get(state, []):
            next_depth = depth + 1
            # Accept a bounded return immediately, even when target==start
            # has depth 0 in the visited map.
            if next_depth <= k and v == target:
                receipt_step = {
                    "step_index": int(step["step_index"]),
                    "trace_ref": {"family": str(step["trace_ref"]["family"]), "trace_id": str(step["trace_ref"]["trace_id"])},
                    "input_hash": str(step["input_hash"]),
                    "output_hash": str(step["output_hash"]),
                }
                return True, path + [receipt_step], {"visited_nodes": len(visited), "expanded": expanded}
            if v in visited and visited[v] <= next_depth:
                continue
            visited[v] = next_depth
            receipt_step = {
                "step_index": int(step["step_index"]),
                "trace_ref": {"family": str(step["trace_ref"]["family"]), "trace_id": str(step["trace_ref"]["trace_id"])},
                "input_hash": str(step["input_hash"]),
                "output_hash": str(step["output_hash"]),
            }
            q.append((v, path + [receipt_step]))

    return False, [], {"visited_nodes": len(visited), "expanded": expanded}


def _self_test() -> bool:
    print("[self-test] bounded return semantics")

    episode = {
        "steps": [
            {
                "step_index": 0,
                "trace_ref": {"family": "QA_SELFTEST", "trace_id": "t0"},
                "input_hash": "S",
                "output_hash": "A",
                "action": {"generator": "sigma"},
            },
            {
                "step_index": 1,
                "trace_ref": {"family": "QA_SELFTEST", "trace_id": "t1"},
                "input_hash": "A",
                "output_hash": "S",
                "action": {"generator": "mu"},
            },
        ]
    }

    ok = True

    found_k1, path_k1, _ = _find_return(episode, "S", "S", 1)
    if found_k1:
        print("  [FAIL] expected NO_RETURN for k=1")
        ok = False
    else:
        print("  [PASS] k=1 -> NO_RETURN_WITHIN_K")

    found_k2, path_k2, _ = _find_return(episode, "S", "S", 2)
    if not found_k2:
        print("  [FAIL] expected RETURN_FOUND for k=2")
        ok = False
    elif len(path_k2) != 2:
        print(f"  [FAIL] expected path length 2 for k=2, got {len(path_k2)}")
        ok = False
    else:
        print("  [PASS] k=2 -> RETURN_FOUND (path length 2)")

    return ok


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def _emit_frontier(ep: Dict, out_dir: str) -> str:
    visited, frontier = _compute_frontier(ep)
    now = _now_utc()

    snap = {
        "schema_id": "QA_FRONTIER_SNAPSHOT_SCHEMA.v1",
        "snapshot_id": _sha256(ep["episode_id"] + "|frontier|" + now),
        "created_utc": now,
        "agent_id": ep["agent_id"],
        "generator_set_id": ep["generator_set_id"],
        "frontier": frontier,
        "visited": visited,
        "score_model": {"novelty_weight": 1.0, "reuse_weight": 0.5, "obstruction_diversity_weight": 0.75},
        "hash_chain": {"prev_snapshot_hash": "0" * 64, "this_snapshot_hash": ""},
        "invariant_diff": {"frontier_size": len(frontier), "visited_size": len(visited)},
    }

    # Compute self-hash with empty this_snapshot_hash
    snap["hash_chain"]["this_snapshot_hash"] = _sha256(_canonical(snap))

    path = os.path.join(out_dir, "frontier_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_canonical(snap) + "\n")
    return path


def _emit_receipt(ep: Dict, out_dir: str, k: int, toolchain_id: str) -> str:
    start = str(ep["initial_state"]["state_hash"])
    target = start
    now = _now_utc()

    found, path_items, stats = _find_return(ep, start, target, k)

    receipt = {
        "schema_id": "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1",
        "receipt_id": _sha256(ep["episode_id"] + "|return|" + str(k) + "|" + now),
        "created_utc": now,
        "agent_id": ep["agent_id"],
        "generator_set_id": ep["generator_set_id"],
        "start_state": {"layer": ep["initial_state"]["layer"], "state_hash": start},
        "return_target_state": {"layer": ep["initial_state"]["layer"], "state_hash": target},
        "k": k,
        "search": {
            "algorithm": "BFS",
            "budget": {"max_nodes": 10000, "max_seconds": 10},
            "determinism": {"root_ordering": "lex", "tie_breaker": "stable", "toolchain_id": toolchain_id},
        },
        "result": {},
        "merkle_parent": ep.get("merkle_parent", "0" * 64),
        "invariant_diff": {},
    }

    if found:
        receipt["result"] = {"status": "RETURN_FOUND", "path": path_items}
        receipt["invariant_diff"] = {"k": k, "path_length": len(path_items)}
    else:
        receipt["result"] = {
            "status": "NO_RETURN_WITHIN_K",
            "fail_type": "NO_RETURN_WITHIN_K",
            "invariant_diff": {"k": k, "visited_nodes": stats.get("visited_nodes", 0), "frontier_exhausted": True},
        }
        receipt["invariant_diff"] = {"k": k, "result": "NO_RETURN_WITHIN_K"}

    path = os.path.join(out_dir, "bounded_return_receipt.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_canonical(receipt) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return 0 if _self_test() else 1

    ap = argparse.ArgumentParser(description="Replay + validate episode; emit frontier snapshot + bounded return receipt.")
    ap.add_argument("--episode", required=True, help="Path to episode JSON")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--k", type=int, default=2, help="Bound for return-in-k receipt (default: 2)")
    ap.add_argument("--toolchain_id", default="lean4.12.0")
    args = ap.parse_args()

    with open(args.episode, "r", encoding="utf-8") as f:
        ep = json.load(f)

    result = validate_episode(ep)
    if not result.ok:
        sys.stderr.write(f"FAIL: {result.fail_type} {json.dumps(result.invariant_diff)}\n")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    frontier_path = _emit_frontier(ep, args.out_dir)
    receipt_path = _emit_receipt(ep, args.out_dir, k=args.k, toolchain_id=args.toolchain_id)

    print(f"VALID\nWROTE {frontier_path}\nWROTE {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
