#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import jsonschema


POLICY = "GREEDY_MIN_ENERGY_TIEBREAK_MOVE_KEY"


def fail(
    fail_type: str,
    invariant_diff: Dict[str, Any],
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "fail_type": fail_type,
        "invariant_diff": invariant_diff,
        "details": details or {},
    }


def validate_cert(cert: Dict[str, Any]) -> Dict[str, Any]:
    """
    QA-style validator entrypoint.

    Returns:
      Success: {"ok": True, "value": {"gates": [...]} }
      Failure: {"ok": False, "fail_type": "...", "invariant_diff": {...}, "details": {..., "gates": [...]} }
    """
    schema = _load_schema()
    gates: List[Dict[str, Any]] = []

    g1 = gate_1_schema_and_domain(cert, schema)
    gates.append(g1)
    if not g1["ok"]:
        return _wrap_fail(g1["fail_type"], g1["invariant_diff"], g1["details"], gates)

    g2 = gate_2_recompute_moves(cert)
    gates.append(g2)
    if not g2["ok"]:
        return _wrap_fail(g2["fail_type"], g2["invariant_diff"], g2["details"], gates)

    g3 = gate_3_recompute_energy(cert)
    gates.append(g3)
    if not g3["ok"]:
        return _wrap_fail(g3["fail_type"], g3["invariant_diff"], g3["details"], gates)

    g4 = gate_4_verify_attempt_log(cert)
    gates.append(g4)
    if not g4["ok"]:
        return _wrap_fail(g4["fail_type"], g4["invariant_diff"], g4["details"], gates)

    g5 = gate_5_verify_policy_and_recoverability(cert)
    gates.append(g5)
    if not g5["ok"]:
        return _wrap_fail(g5["fail_type"], g5["invariant_diff"], g5["details"], gates)

    return {"ok": True, "value": {"gates": gates}}


def _wrap_fail(
    fail_type: str,
    invariant_diff: Dict[str, Any],
    details: Dict[str, Any],
    gates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "ok": False,
        "fail_type": fail_type,
        "invariant_diff": invariant_diff,
        "details": {**details, "gates": gates},
    }


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)


def _load_schema() -> Dict[str, Any]:
    with open(_here("schema.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Move:
    name: str
    i: int

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "i": self.i}


Bits = Tuple[int, ...]


def parse_bits(arr: List[Any]) -> Bits:
    return tuple(int(x) for x in arr)


def bits_to_list(bits: Bits) -> List[int]:
    return [int(x) for x in bits]


def parse_move(d: Dict[str, Any]) -> Move:
    return Move(name=str(d["name"]), i=int(d["i"]))


def _move_key(m: Move) -> Tuple[str, int]:
    return (m.name, int(m.i))


def allowed_move_keys_from_generator_set(generator_set: List[Dict[str, Any]]) -> set[Tuple[str, int]]:
    keys: set[Tuple[str, int]] = set()
    for g in generator_set:
        keys.add((str(g["name"]), int(g["i"])))
    return keys


def ensure_moves_subset_of_generator_set(
    steps: List[Dict[str, Any]],
    allowed_keys: set[Tuple[str, int]],
) -> Optional[Dict[str, Any]]:
    allowed_sorted = sorted(list(allowed_keys))
    allowed_json = [[name, i] for (name, i) in allowed_sorted]
    for step_index, step in enumerate(steps):
        chosen = parse_move(step["chosen_move"])
        if _move_key(chosen) not in allowed_keys:
            return fail(
                "MOVE_NOT_IN_GENERATOR_SET",
                {
                    "step_index": step_index,
                    "where": "chosen_move",
                    "move": chosen.as_dict(),
                    "allowed": allowed_json,
                },
                {"why": "chosen_move must be drawn from declared generator_set"},
            )

        for attempt_index, entry in enumerate(step["attempted_moves"]):
            mv = parse_move(entry["move"])
            if _move_key(mv) not in allowed_keys:
                return fail(
                    "MOVE_NOT_IN_GENERATOR_SET",
                    {
                        "step_index": step_index,
                        "attempt_index": attempt_index,
                        "where": "attempted_moves",
                        "move": mv.as_dict(),
                        "allowed": allowed_json,
                    },
                    {"why": "attempted_moves must be drawn from declared generator_set"},
                )
    return None


def validate_bits_domain(bits: Bits, n: int) -> bool:
    return len(bits) == n and all(b in (0, 1) for b in bits)


def apply_move(bits: Bits, mv: Move, n: int) -> Dict[str, Any]:
    if not validate_bits_domain(bits, n):
        return {
            "ok": False,
            "fail_type": "DOMAIN_INVALID",
            "invariant_diff": {"expected_len": n, "got_len": len(bits)},
            "details": {"why": "state bits must be length N with entries in {0,1}"},
        }

    if mv.i < 0 or mv.i >= n:
        return {
            "ok": False,
            "fail_type": "MOVE_PRECONDITION_FAILED",
            "invariant_diff": {"move": mv.as_dict(), "required": {"i_range": [0, n - 1]}},
            "details": {"why": "move index i out of range for N"},
        }

    b_list = list(bits)
    if mv.name == "flip":
        b_list[mv.i] = 1 - b_list[mv.i]
        return {"ok": True, "value": bits_to_list(tuple(b_list))}

    if mv.name == "flip_if_one":
        if b_list[mv.i] != 1:
            return {
                "ok": False,
                "fail_type": "MOVE_PRECONDITION_FAILED",
                "invariant_diff": {"move": mv.as_dict(), "required": {"bit_i": 1}, "got": {"bit_i": b_list[mv.i]}},
                "details": {"why": "flip_if_one requires bit i == 1"},
            }
        b_list[mv.i] = 0
        return {"ok": True, "value": bits_to_list(tuple(b_list))}

    return {
        "ok": False,
        "fail_type": "MOVE_PRECONDITION_FAILED",
        "invariant_diff": {"move": mv.as_dict(), "required": {"name_in": ["flip", "flip_if_one"]}},
        "details": {"why": "unknown move name"},
    }


def hamming(a: Bits, b: Bits) -> int:
    return int(sum(1 for x, y in zip(a, b) if x != y))


def energy_hamming_to_nearest_memory(state: Bits, memories: List[Bits]) -> int:
    return min(hamming(state, m) for m in memories)


def move_sort_key(m: Move) -> str:
    return f"{m.name}:{m.i}"


def compile_generator_moves(generator_set: List[Dict[str, Any]]) -> List[Move]:
    seen: set[Tuple[str, int]] = set()
    moves: List[Move] = []
    for g in generator_set:
        mv = Move(name=str(g["name"]), i=int(g["i"]))
        key = _move_key(mv)
        if key in seen:
            continue
        seen.add(key)
        moves.append(mv)
    return moves


def min_steps_within_k(start: Bits, goal: Bits, moves: List[Move], n: int, k: int) -> Optional[int]:
    if k < 0:
        return None
    if start == goal:
        return 0

    q: Deque[Tuple[Bits, int]] = deque()
    q.append((start, 0))
    visited = {start}

    while q:
        s, dist = q.popleft()
        if dist == k:
            continue
        for mv in moves:
            res = apply_move(s, mv, n)
            if not res["ok"]:
                continue
            s2 = parse_bits(res["value"])
            if s2 in visited:
                continue
            if s2 == goal:
                return dist + 1
            visited.add(s2)
            q.append((s2, dist + 1))

    return None


def gate_1_schema_and_domain(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        jsonschema.validate(instance=cert, schema=schema)
    except jsonschema.ValidationError as e:
        return fail(
            "SCHEMA_INVALID",
            {"jsonschema_path": list(e.absolute_schema_path), "instance_path": list(e.absolute_path)},
            {"error": str(e)},
        )

    n = int(cert["N"])
    obj = cert["objective"]
    if obj["type"] != "HAMMING_TO_NEAREST_MEMORY":
        return fail("DOMAIN_INVALID", {"expected_objective": "HAMMING_TO_NEAREST_MEMORY", "got": obj["type"]})

    memories_raw = obj["memories"]
    memories = [parse_bits(m) for m in memories_raw]
    for idx, m in enumerate(memories):
        if not validate_bits_domain(m, n):
            return fail(
                "DOMAIN_INVALID",
                {"where": "objective.memories", "memory_index": idx, "expected_len": n, "got_len": len(m)},
                {"why": "each memory must be a bitvector of length N"},
            )

    start = parse_bits(cert["run"]["start_state"])
    if not validate_bits_domain(start, n):
        return fail(
            "DOMAIN_INVALID",
            {"where": "run.start_state", "expected_len": n, "got_len": len(start)},
            {"why": "start_state must be a bitvector of length N"},
        )

    for step_index, step in enumerate(cert["run"]["steps"]):
        sb = parse_bits(step["state_before"])
        sa = parse_bits(step["state_after"])
        if not validate_bits_domain(sb, n) or not validate_bits_domain(sa, n):
            return fail(
                "DOMAIN_INVALID",
                {"where": "run.steps", "step_index": step_index, "expected_len": n},
                {"why": "state_before/state_after must be bitvectors of length N"},
            )

    for gen_index, g in enumerate(cert["generator_set"]):
        i = int(g["i"])
        if i < 0 or i >= n:
            return fail(
                "DOMAIN_INVALID",
                {"where": "generator_set", "generator_index": gen_index, "i": i, "required_i_range": [0, n - 1]},
                {"why": "generator i must satisfy 0 <= i < N"},
            )

    return {"ok": True, "value": {"gate": 1}}


def gate_2_recompute_moves(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    run = cert["run"]
    if run["policy"] != POLICY:
        return fail(
            "POLICY_MISMATCH",
            {"expected": POLICY, "got": run["policy"]},
            {"why": "run.policy not supported by this validator"},
        )

    start = parse_bits(run["start_state"])
    prev = start
    for step_index, step in enumerate(run["steps"]):
        sb = parse_bits(step["state_before"])
        if sb != prev:
            return fail(
                "STATE_CHAIN_MISMATCH",
                {
                    "step_index": step_index,
                    "expected_state_before": bits_to_list(prev),
                    "got_state_before": bits_to_list(sb),
                },
                {"why": "step.state_before does not match previous state_after (or start_state for first step)"},
            )

        chosen = parse_move(step["chosen_move"])
        got = apply_move(sb, chosen, n)
        if not got["ok"]:
            return fail(
                "CHOSEN_MOVE_ILLEGAL",
                {"step_index": step_index, "chosen_move": chosen.as_dict(), "failure": got},
                {"why": "chosen move failed when recomputed"},
            )

        sa = parse_bits(step["state_after"])
        expected_after = parse_bits(got["value"])
        if sa != expected_after:
            return fail(
                "STATE_AFTER_MISMATCH",
                {
                    "step_index": step_index,
                    "chosen_move": chosen.as_dict(),
                    "expected_state_after": bits_to_list(expected_after),
                    "got_state_after": bits_to_list(sa),
                },
                {"why": "step.state_after differs from recomputation"},
            )
        prev = sa

    return {"ok": True, "value": {"gate": 2, "steps_checked": len(run["steps"])}}


def gate_3_recompute_energy(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    obj = cert["objective"]
    memories = [parse_bits(m) for m in obj["memories"]]
    for m in memories:
        if not validate_bits_domain(m, n):
            return fail("DOMAIN_INVALID", {"where": "objective.memories", "expected_len": n})

    steps = cert["run"]["steps"]
    for step_index, step in enumerate(steps):
        sb = parse_bits(step["state_before"])
        sa = parse_bits(step["state_after"])
        eb = energy_hamming_to_nearest_memory(sb, memories)
        ea = energy_hamming_to_nearest_memory(sa, memories)

        if int(step["energy_before"]) != eb:
            return fail(
                "ENERGY_BEFORE_MISMATCH",
                {"step_index": step_index, "expected": eb, "got": int(step["energy_before"])},
                {"state_before": bits_to_list(sb)},
            )
        if int(step["energy_after"]) != ea:
            return fail(
                "ENERGY_AFTER_MISMATCH",
                {"step_index": step_index, "expected": ea, "got": int(step["energy_after"])},
                {"state_after": bits_to_list(sa)},
            )

    return {"ok": True, "value": {"gate": 3, "steps_checked": len(steps)}}


def gate_4_verify_attempt_log(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    steps = cert["run"]["steps"]

    allowed_keys = allowed_move_keys_from_generator_set(cert["generator_set"])
    subset_fail = ensure_moves_subset_of_generator_set(steps, allowed_keys)
    if subset_fail is not None:
        return subset_fail

    for step_index, step in enumerate(steps):
        sb = parse_bits(step["state_before"])
        attempted = step["attempted_moves"]
        if len(attempted) == 0:
            return fail("EMPTY_ATTEMPT_LOG", {"step_index": step_index})

        for attempt_index, entry in enumerate(attempted):
            mv = parse_move(entry["move"])
            recomputed = apply_move(sb, mv, n)
            logged = entry["result"]

            if bool(logged["ok"]) != bool(recomputed["ok"]):
                return fail(
                    "ATTEMPT_OK_MISMATCH",
                    {"step_index": step_index, "attempt_index": attempt_index, "move": mv.as_dict()},
                    {"recomputed": recomputed, "logged": logged},
                )

            if recomputed["ok"]:
                if logged.get("value") != recomputed["value"]:
                    return fail(
                        "ATTEMPT_VALUE_MISMATCH",
                        {
                            "step_index": step_index,
                            "attempt_index": attempt_index,
                            "move": mv.as_dict(),
                            "expected_value": recomputed["value"],
                            "got_value": logged.get("value"),
                        },
                    )
            else:
                if logged.get("fail_type") != recomputed["fail_type"]:
                    return fail(
                        "ATTEMPT_FAILTYPE_MISMATCH",
                        {
                            "step_index": step_index,
                            "attempt_index": attempt_index,
                            "move": mv.as_dict(),
                            "expected_fail_type": recomputed["fail_type"],
                            "got_fail_type": logged.get("fail_type"),
                        },
                    )
                for k, v in recomputed.get("invariant_diff", {}).items():
                    if logged.get("invariant_diff", {}).get(k) != v:
                        return fail(
                            "ATTEMPT_INVARIANT_DIFF_MISMATCH",
                            {
                                "step_index": step_index,
                                "attempt_index": attempt_index,
                                "move": mv.as_dict(),
                                "missing_or_mismatched_key": k,
                                "expected_value": v,
                                "got_value": logged.get("invariant_diff", {}).get(k),
                            },
                        )

        chosen = parse_move(step["chosen_move"])
        if not any(parse_move(x["move"]) == chosen for x in attempted):
            return fail(
                "CHOSEN_NOT_IN_ATTEMPTS",
                {"step_index": step_index, "chosen_move": chosen.as_dict()},
                {"why": "chosen_move must be included in attempted_moves for auditability"},
            )

    return {"ok": True, "value": {"gate": 4, "steps_checked": len(steps)}}


def gate_5_verify_policy_and_recoverability(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    obj = cert["objective"]
    memories = [parse_bits(m) for m in obj["memories"]]
    run = cert["run"]
    steps = run["steps"]

    # Part A: greedy policy (argmin energy_after over attempted legal moves)
    for step_index, step in enumerate(steps):
        chosen = parse_move(step["chosen_move"])
        sb = parse_bits(step["state_before"])

        legal: List[Tuple[int, str, Move, Bits]] = []
        for entry in step["attempted_moves"]:
            mv = parse_move(entry["move"])
            res = entry["result"]
            if res["ok"]:
                s2 = parse_bits(res["value"])
                if not validate_bits_domain(s2, n):
                    return fail(
                        "DOMAIN_INVALID",
                        {"where": "attempted_moves.result.value", "step_index": step_index, "move": mv.as_dict()},
                    )
                e2 = energy_hamming_to_nearest_memory(s2, memories)
                legal.append((e2, move_sort_key(mv), mv, s2))

        if len(legal) == 0:
            return fail(
                "NO_LEGAL_MOVES",
                {"step_index": step_index, "state_before": bits_to_list(sb)},
                {"why": "policy requires at least one legal attempted move"},
            )

        legal.sort(key=lambda x: (x[0], x[1]))
        best_e, _, best_mv, _ = legal[0]
        if chosen != best_mv:
            return fail(
                "POLICY_VIOLATION",
                {
                    "step_index": step_index,
                    "expected_best_move": best_mv.as_dict(),
                    "got_chosen_move": chosen.as_dict(),
                    "expected_best_energy_after": best_e,
                },
                {"why": "chosen_move is not greedy argmin over attempted legal moves (tie-break by move key)"},
            )

    # Part B: optional probes (BFS using declared generator_set)
    probes = run.get("return_in_k_probes")
    if probes is None:
        return {"ok": True, "value": {"gate": 5, "steps_checked": len(steps), "probes_checked": 0}}

    compiled_moves = compile_generator_moves(cert["generator_set"])

    t_to_state_after: Dict[int, Bits] = {}
    for st in steps:
        t_val = int(st["t"])
        if t_val in t_to_state_after:
            return fail(
                "DUPLICATE_STEP_T",
                {"t": t_val},
                {"why": "run.steps contains duplicate t values; probe start_from_step_t ambiguous"},
            )
        t_to_state_after[t_val] = parse_bits(st["state_after"])

    checked = 0
    for p in probes:
        probe_id = p["probe_id"]
        k = int(p["k"])
        goal = parse_bits(p["goal_state"])
        expected = p["expected"]

        if not validate_bits_domain(goal, n):
            return fail("DOMAIN_INVALID", {"where": "return_in_k_probes.goal_state", "probe_id": probe_id, "expected_len": n})

        if "start_state" in p:
            start = parse_bits(p["start_state"])
        else:
            t_ref = int(p["start_from_step_t"])
            if t_ref not in t_to_state_after:
                return fail(
                    "PROBE_BAD_STEP_REF",
                    {"probe_id": probe_id, "start_from_step_t": t_ref, "available_t": sorted(t_to_state_after.keys())},
                    {"why": "probe references a step t that does not exist in run.steps"},
                )
            start = t_to_state_after[t_ref]

        if not validate_bits_domain(start, n):
            return fail("DOMAIN_INVALID", {"where": "return_in_k_probes.start", "probe_id": probe_id, "expected_len": n})

        min_steps = min_steps_within_k(start, goal, compiled_moves, n, k)
        reachable = min_steps is not None

        if bool(expected["reachable"]) != reachable:
            if bool(expected["reachable"]) and not reachable:
                return fail(
                    "NON_RECOVERABLE",
                    {
                        "probe_id": probe_id,
                        "start": bits_to_list(start),
                        "goal": bits_to_list(goal),
                        "k": k,
                        "expected_reachable": True,
                        "got_reachable": False,
                    },
                    {"why": "goal not reachable within k under declared generator_set"},
                )
            return fail(
                "PROBE_REACHABILITY_MISMATCH",
                {
                    "probe_id": probe_id,
                    "start": bits_to_list(start),
                    "goal": bits_to_list(goal),
                    "k": k,
                    "expected_reachable": bool(expected["reachable"]),
                    "got_reachable": reachable,
                    "got_min_steps": min_steps,
                },
                {"why": "BFS reachability does not match expected.reachable"},
            )

        if "min_steps" in expected:
            if not reachable:
                return fail(
                    "PROBE_MIN_STEPS_BUT_UNREACHABLE",
                    {"probe_id": probe_id, "expected_min_steps": int(expected["min_steps"])},
                    {"why": "expected.min_steps provided but goal is unreachable"},
                )
            if int(expected["min_steps"]) != int(min_steps):
                return fail(
                    "PROBE_MIN_STEPS_MISMATCH",
                    {
                        "probe_id": probe_id,
                        "start": bits_to_list(start),
                        "goal": bits_to_list(goal),
                        "k": k,
                        "expected_min_steps": int(expected["min_steps"]),
                        "got_min_steps": int(min_steps),
                    },
                    {"why": "BFS minimum steps does not match expected.min_steps"},
                )

        checked += 1

    return {"ok": True, "value": {"gate": 5, "steps_checked": len(steps), "probes_checked": checked}}


def _main() -> int:
    ap = argparse.ArgumentParser(description="Validate QA_HOPFIELD_REACHABILITY_RUN_CERT.v1 certificates.")
    ap.add_argument("cert_json", nargs="?", help="Path to certificate JSON")
    ap.add_argument("--demo", action="store_true", help="Validate the PASS fixture and exit")
    ap.add_argument("--self-test", action="store_true", help="Validate PASS plus all FAIL_* fixtures")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.demo:
        cert_path = _here("fixtures", "PASS_N8_TWO_MEMORIES.json")
    elif args.cert_json:
        cert_path = args.cert_json
    else:
        ap.error("Provide cert_json or --demo")

    with open(cert_path, "r", encoding="utf-8") as f:
        cert = json.load(f)

    result = validate_cert(cert)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 2


def _self_test() -> int:
    fixtures_dir = _here("fixtures")

    pass_path = os.path.join(fixtures_dir, "PASS_N8_TWO_MEMORIES.json")
    with open(pass_path, "r", encoding="utf-8") as f:
        cert = json.load(f)
    res = validate_cert(cert)
    if not res.get("ok"):
        print(f"[FAIL] PASS fixture failed: {pass_path}", file=sys.stderr)
        return 2
    print(f"[PASS] PASS fixture: {pass_path}")

    fail_fixtures = sorted(fn for fn in os.listdir(fixtures_dir) if fn.startswith("FAIL_") and fn.endswith(".json"))
    accepted_fail_fixtures: List[Tuple[str, str]] = []
    pat = re.compile(r"^FAIL_([A-Z0-9_]+?)(?=__|\.json)(?:__.+)?\.json$")
    for fn in fail_fixtures:
        m = pat.match(fn)
        if not m:
            print(f"[WARN] Skipping nonconforming FAIL fixture name: {fn}", file=sys.stderr)
            continue
        accepted_fail_fixtures.append((fn, m.group(1)))

    if len(fail_fixtures) > 0 and len(accepted_fail_fixtures) == 0:
        print("[WARN] All FAIL_* fixtures were skipped due to naming; only PASS was validated", file=sys.stderr)
    elif len(fail_fixtures) == 0:
        print("[WARN] No FAIL_* fixtures found; only PASS was validated", file=sys.stderr)

    for fn, expected_fail_type in accepted_fail_fixtures:
        full_path = os.path.join(fixtures_dir, fn)
        with open(full_path, "r", encoding="utf-8") as f:
            cert = json.load(f)
        res = validate_cert(cert)
        if res.get("ok"):
            print(f"[FAIL] Expected failure but got ok=true: {full_path}", file=sys.stderr)
            return 2
        got_fail_type = res.get("fail_type")
        if got_fail_type != expected_fail_type:
            print(
                f"[FAIL] Wrong fail_type for {full_path}: expected={expected_fail_type} got={got_fail_type}",
                file=sys.stderr,
            )
            return 2
        print(f"[PASS] FAIL fixture ({expected_fail_type}): {full_path}")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

