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


POLICY = "GREEDY_MIN_ENERGY_TIEBREAK_MOVE_NAME"


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

    g1 = gate_1_schema(cert, schema)
    gates.append(g1)
    if not g1["ok"]:
        return _wrap_fail("SCHEMA_INVALID", g1["invariant_diff"], g1["details"], gates)

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


# -----------------------
# Helpers: state & energy
# -----------------------


@dataclass(frozen=True)
class State:
    b: int
    e: int

    def as_dict(self) -> Dict[str, int]:
        return {"b": self.b, "e": self.e}


@dataclass(frozen=True)
class Move:
    name: str
    k: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.name == "lambda_k":
            d["k"] = int(self.k) if self.k is not None else None
        return d


def parse_state(d: Dict[str, Any]) -> State:
    return State(int(d["b"]), int(d["e"]))


def parse_move(d: Dict[str, Any]) -> Move:
    name = d["name"]
    k = d.get("k")
    return Move(name=name, k=int(k) if k is not None else None)


def energy_l2_to_target(s: State, target: State) -> int:
    db = s.b - target.b
    de = s.e - target.e
    return int(db * db + de * de)


def is_in_natural_domain(s: State) -> bool:
    return s.b >= 1 and s.e >= 1


def is_in_caps(s: State, n: int) -> bool:
    return s.b <= n and s.e <= n


def apply_move(s: State, move: Move, n: int) -> Dict[str, Any]:
    """
    Returns QA-style result:
      ok: True  -> {"ok": True, "value": {"b":..,"e":..}}
      ok: False -> {"ok": False, "fail_type": ..., "invariant_diff": ..., "details": ...}
    """
    if not is_in_natural_domain(s):
        return {
            "ok": False,
            "fail_type": "NOT_IN_NATURAL_DOMAIN",
            "invariant_diff": {"domain": "NATURAL", "b": s.b, "e": s.e, "min_b": 1, "min_e": 1},
            "details": {"why": "state_before not in natural domain"},
        }
    if not is_in_caps(s, n):
        return {
            "ok": False,
            "fail_type": "OUT_OF_BOUNDS",
            "invariant_diff": {"constraint": "CAPS", "b": s.b, "e": s.e, "N": n},
            "details": {"why": "state_before exceeds Caps(N,N)"},
        }

    if move.name == "sigma":
        s2 = State(s.b, s.e + 1)
    elif move.name == "mu":
        s2 = State(s.e, s.b)
    elif move.name == "lambda_k":
        if move.k is None or move.k < 2:
            return {
                "ok": False,
                "fail_type": "MOVE_PRECONDITION_FAILED",
                "invariant_diff": {"move": "lambda_k", "required": {"k_min": 2}, "got": {"k": move.k}},
                "details": {"why": "lambda_k requires integer k >= 2"},
            }
        k = int(move.k)
        s2 = State(s.b * k, s.e * k)
    elif move.name == "nu":
        if (s.b % 2 != 0) or (s.e % 2 != 0):
            return {
                "ok": False,
                "fail_type": "MOVE_PRECONDITION_FAILED",
                "invariant_diff": {
                    "move": "nu",
                    "required": {"b_even": True, "e_even": True},
                    "got": {"b": s.b, "e": s.e},
                },
                "details": {"why": "nu requires both b and e even"},
            }
        s2 = State(s.b // 2, s.e // 2)
    else:
        return {
            "ok": False,
            "fail_type": "MOVE_PRECONDITION_FAILED",
            "invariant_diff": {"move": move.name, "required": {"name_in": ["sigma", "mu", "lambda_k", "nu"]}},
            "details": {"why": "unknown move name"},
        }

    if not is_in_natural_domain(s2):
        return {
            "ok": False,
            "fail_type": "NOT_IN_NATURAL_DOMAIN",
            "invariant_diff": {"domain": "NATURAL", "b": s2.b, "e": s2.e, "min_b": 1, "min_e": 1},
            "details": {"why": "state_after not in natural domain"},
        }
    if not is_in_caps(s2, n):
        return {
            "ok": False,
            "fail_type": "OUT_OF_BOUNDS",
            "invariant_diff": {"constraint": "CAPS", "b": s2.b, "e": s2.e, "N": n},
            "details": {"why": "state_after exceeds Caps(N,N)"},
        }

    return {"ok": True, "value": s2.as_dict()}


def compile_generator_moves(generator_set: List[Dict[str, Any]]) -> List[Move]:
    """
    Compile declared generator_set into concrete Move instances.
    Preserves order, but deduplicates identical moves.
    """
    seen: set[tuple[str, Optional[int]]] = set()
    moves: List[Move] = []
    for g in generator_set:
        mv = Move(name=g["name"], k=int(g["k"]) if g["name"] == "lambda_k" else None)
        key = (mv.name, mv.k)
        if key in seen:
            continue
        seen.add(key)
        moves.append(mv)
    return moves


def min_steps_within_k(start: State, goal: State, moves: List[Move], n: int, k: int) -> Optional[int]:
    """
    BFS over Caps(N,N) using the declared moves.
    Returns minimum steps if reachable within k, else None.
    """
    if k < 0:
        return None
    if start == goal:
        return 0

    q: Deque[Tuple[State, int]] = deque()
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
            s2 = parse_state(res["value"])
            if s2 in visited:
                continue
            if s2 == goal:
                return dist + 1
            visited.add(s2)
            q.append((s2, dist + 1))

    return None


def _move_key(m: Move) -> tuple[str, Optional[int]]:
    return (m.name, m.k if m.name == "lambda_k" else None)


def allowed_move_keys_from_generator_set(generator_set: List[Dict[str, Any]]) -> set[tuple[str, Optional[int]]]:
    keys: set[tuple[str, Optional[int]]] = set()
    for g in generator_set:
        name = g["name"]
        k = int(g["k"]) if name == "lambda_k" else None
        keys.add((name, k))
    return keys


def ensure_moves_subset_of_generator_set(
    steps: List[Dict[str, Any]],
    allowed_keys: set[tuple[str, Optional[int]]],
) -> Optional[Dict[str, Any]]:
    """
    Returns a FAIL dict if any attempted/chosen move is not in generator_set; else None.
    """
    allowed_sorted = sorted(list(allowed_keys))
    allowed_json = [[name, k] for (name, k) in allowed_sorted]
    for i, step in enumerate(steps):
        chosen = parse_move(step["chosen_move"])
        if _move_key(chosen) not in allowed_keys:
            return fail(
                "MOVE_NOT_IN_GENERATOR_SET",
                {
                    "step_index": i,
                    "where": "chosen_move",
                    "move": chosen.as_dict(),
                    "allowed": allowed_json,
                },
                {"why": "chosen_move must be drawn from declared generator_set"},
            )

        for j, entry in enumerate(step["attempted_moves"]):
            mv = parse_move(entry["move"])
            if _move_key(mv) not in allowed_keys:
                return fail(
                    "MOVE_NOT_IN_GENERATOR_SET",
                    {
                        "step_index": i,
                        "attempt_index": j,
                        "where": "attempted_moves",
                        "move": mv.as_dict(),
                        "allowed": allowed_json,
                    },
                    {"why": "attempted_moves must be drawn from declared generator_set"},
                )

    return None


# -------------
# Gate 1: schema
# -------------


def gate_1_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    try:
        jsonschema.validate(instance=cert, schema=schema)
        return {"ok": True, "value": {"gate": 1}}
    except jsonschema.ValidationError as e:
        return {
            "ok": False,
            "fail_type": "SCHEMA_INVALID",
            "invariant_diff": {"jsonschema_path": list(e.absolute_schema_path), "instance_path": list(e.absolute_path)},
            "details": {"error": str(e)},
        }


# -------------------------
# Gate 2: recompute moves
# -------------------------


def gate_2_recompute_moves(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    run = cert["run"]

    if run["policy"] != POLICY:
        return {
            "ok": False,
            "fail_type": "POLICY_MISMATCH",
            "invariant_diff": {"expected": POLICY, "got": run["policy"]},
            "details": {"why": "run.policy not supported by this validator"},
        }

    start = parse_state(run["start_state"])
    steps = run["steps"]

    prev = start
    for i, step in enumerate(steps):
        sb = parse_state(step["state_before"])
        if sb != prev:
            return {
                "ok": False,
                "fail_type": "STATE_CHAIN_MISMATCH",
                "invariant_diff": {
                    "step_index": i,
                    "expected_state_before": prev.as_dict(),
                    "got_state_before": sb.as_dict(),
                },
                "details": {
                    "why": "step.state_before does not match previous state_after (or start_state for first step)"
                },
            }

        chosen = parse_move(step["chosen_move"])
        got = apply_move(sb, chosen, n)
        if not got["ok"]:
            return {
                "ok": False,
                "fail_type": "CHOSEN_MOVE_ILLEGAL",
                "invariant_diff": {"step_index": i, "chosen_move": chosen.as_dict(), "failure": got},
                "details": {"why": "chosen move failed when recomputed"},
            }

        sa = parse_state(step["state_after"])
        expected_after = parse_state(got["value"])
        if sa != expected_after:
            return {
                "ok": False,
                "fail_type": "STATE_AFTER_MISMATCH",
                "invariant_diff": {
                    "step_index": i,
                    "chosen_move": chosen.as_dict(),
                    "expected_state_after": expected_after.as_dict(),
                    "got_state_after": sa.as_dict(),
                },
                "details": {"why": "step.state_after differs from recomputation"},
            }

        prev = sa

    return {"ok": True, "value": {"gate": 2, "steps_checked": len(steps)}}


# --------------------------
# Gate 3: recompute energy
# --------------------------


def gate_3_recompute_energy(cert: Dict[str, Any]) -> Dict[str, Any]:
    obj = cert["objective"]
    if obj["type"] != "L2_TO_TARGET":
        return {
            "ok": False,
            "fail_type": "OBJECTIVE_UNSUPPORTED",
            "invariant_diff": {"expected": "L2_TO_TARGET", "got": obj["type"]},
            "details": {"why": "only L2_TO_TARGET supported"},
        }
    target = parse_state(obj["target_state"])

    steps = cert["run"]["steps"]
    for i, step in enumerate(steps):
        sb = parse_state(step["state_before"])
        sa = parse_state(step["state_after"])

        eb = energy_l2_to_target(sb, target)
        ea = energy_l2_to_target(sa, target)

        if int(step["energy_before"]) != eb:
            return {
                "ok": False,
                "fail_type": "ENERGY_BEFORE_MISMATCH",
                "invariant_diff": {"step_index": i, "expected": eb, "got": int(step["energy_before"])},
                "details": {"why": "energy_before incorrect", "state_before": sb.as_dict()},
            }
        if int(step["energy_after"]) != ea:
            return {
                "ok": False,
                "fail_type": "ENERGY_AFTER_MISMATCH",
                "invariant_diff": {"step_index": i, "expected": ea, "got": int(step["energy_after"])},
                "details": {"why": "energy_after incorrect", "state_after": sa.as_dict()},
            }

    return {"ok": True, "value": {"gate": 3, "steps_checked": len(steps)}}


# ------------------------------------
# Gate 4: verify attempted move logging
# ------------------------------------


def gate_4_verify_attempt_log(cert: Dict[str, Any]) -> Dict[str, Any]:
    n = int(cert["N"])
    steps = cert["run"]["steps"]

    allowed_keys = allowed_move_keys_from_generator_set(cert["generator_set"])
    subset_fail = ensure_moves_subset_of_generator_set(steps, allowed_keys)
    if subset_fail is not None:
        return subset_fail

    for i, step in enumerate(steps):
        sb = parse_state(step["state_before"])
        attempted = step["attempted_moves"]

        if len(attempted) == 0:
            return {
                "ok": False,
                "fail_type": "EMPTY_ATTEMPT_LOG",
                "invariant_diff": {"step_index": i},
                "details": {"why": "attempted_moves must be non-empty"},
            }

        for j, entry in enumerate(attempted):
            move = parse_move(entry["move"])
            recomputed = apply_move(sb, move, n)
            logged = entry["result"]

            if bool(logged["ok"]) != bool(recomputed["ok"]):
                return {
                    "ok": False,
                    "fail_type": "ATTEMPT_OK_MISMATCH",
                    "invariant_diff": {"step_index": i, "attempt_index": j, "move": move.as_dict()},
                    "details": {"why": "attempted move ok flag differs from recomputation", "recomputed": recomputed, "logged": logged},
                }

            if recomputed["ok"]:
                if logged.get("value") != recomputed["value"]:
                    return {
                        "ok": False,
                        "fail_type": "ATTEMPT_VALUE_MISMATCH",
                        "invariant_diff": {
                            "step_index": i,
                            "attempt_index": j,
                            "move": move.as_dict(),
                            "expected_value": recomputed["value"],
                            "got_value": logged.get("value"),
                        },
                        "details": {"why": "attempted move value differs from recomputation"},
                    }
            else:
                if logged.get("fail_type") != recomputed["fail_type"]:
                    return {
                        "ok": False,
                        "fail_type": "ATTEMPT_FAILTYPE_MISMATCH",
                        "invariant_diff": {
                            "step_index": i,
                            "attempt_index": j,
                            "move": move.as_dict(),
                            "expected_fail_type": recomputed["fail_type"],
                            "got_fail_type": logged.get("fail_type"),
                        },
                        "details": {"why": "attempted move fail_type differs from recomputation"},
                    }

                for k, v in recomputed.get("invariant_diff", {}).items():
                    if logged.get("invariant_diff", {}).get(k) != v:
                        return {
                            "ok": False,
                            "fail_type": "ATTEMPT_INVARIANT_DIFF_MISMATCH",
                            "invariant_diff": {
                                "step_index": i,
                                "attempt_index": j,
                                "move": move.as_dict(),
                                "missing_or_mismatched_key": k,
                                "expected_value": v,
                                "got_value": logged.get("invariant_diff", {}).get(k),
                            },
                            "details": {"why": "attempted move invariant_diff missing required recomputed keys/values"},
                        }

        chosen = parse_move(step["chosen_move"])
        if not any(parse_move(x["move"]) == chosen for x in attempted):
            return {
                "ok": False,
                "fail_type": "CHOSEN_NOT_IN_ATTEMPTS",
                "invariant_diff": {"step_index": i, "chosen_move": chosen.as_dict()},
                "details": {"why": "chosen_move must be included in attempted_moves for auditability"},
            }

    return {"ok": True, "value": {"gate": 4, "steps_checked": len(steps)}}


# ---------------------------------------------------------
# Gate 5: greedy policy + optional return-in-k recoverability
# ---------------------------------------------------------


def gate_5_verify_policy_and_recoverability(cert: Dict[str, Any]) -> Dict[str, Any]:
    obj = cert["objective"]
    target = parse_state(obj["target_state"])
    run = cert["run"]
    steps = run["steps"]
    n = int(cert["N"])

    # Part A: greedy policy (argmin over attempted legal moves)
    for i, step in enumerate(steps):
        chosen = parse_move(step["chosen_move"])

        legal: List[Tuple[int, str, Move, State]] = []
        for entry in step["attempted_moves"]:
            mv = parse_move(entry["move"])
            res = entry["result"]
            if res["ok"]:
                s2 = parse_state(res["value"])
                e2 = energy_l2_to_target(s2, target)
                mv_key = move_sort_key(mv)
                legal.append((e2, mv_key, mv, s2))

        if len(legal) == 0:
            return {
                "ok": False,
                "fail_type": "NO_LEGAL_MOVES",
                "invariant_diff": {"step_index": i, "state_before": step["state_before"]},
                "details": {"why": "policy requires at least one legal attempted move"},
            }

        legal.sort(key=lambda x: (x[0], x[1]))
        best_e, _, best_mv, _ = legal[0]

        if chosen != best_mv:
            return {
                "ok": False,
                "fail_type": "POLICY_VIOLATION",
                "invariant_diff": {
                    "step_index": i,
                    "expected_best_move": best_mv.as_dict(),
                    "got_chosen_move": chosen.as_dict(),
                    "expected_best_energy_after": best_e,
                },
                "details": {"why": "chosen_move is not greedy argmin over attempted legal moves (tie-break by move name)"},
            }

    # Part B: optional probes (BFS using declared generator_set)
    probes = run.get("return_in_k_probes")
    if probes is None:
        return {"ok": True, "value": {"gate": 5, "steps_checked": len(steps), "probes_checked": 0}}

    compiled_moves = compile_generator_moves(cert["generator_set"])

    t_to_state_after: Dict[int, State] = {}
    for st in steps:
        t_val = int(st["t"])
        if t_val in t_to_state_after:
            return {
                "ok": False,
                "fail_type": "DUPLICATE_STEP_T",
                "invariant_diff": {"t": t_val},
                "details": {"why": "run.steps contains duplicate t values; probe start_from_step_t ambiguous"},
            }
        t_to_state_after[t_val] = parse_state(st["state_after"])

    checked = 0
    for p in probes:
        probe_id = p["probe_id"]
        k = int(p["k"])
        goal = parse_state(p["goal_state"])
        expected = p["expected"]

        if "start_state" in p:
            start = parse_state(p["start_state"])
        else:
            t_ref = int(p["start_from_step_t"])
            if t_ref not in t_to_state_after:
                return {
                    "ok": False,
                    "fail_type": "PROBE_BAD_STEP_REF",
                    "invariant_diff": {
                        "probe_id": probe_id,
                        "start_from_step_t": t_ref,
                        "available_t": sorted(t_to_state_after.keys()),
                    },
                    "details": {"why": "probe references a step t that does not exist in run.steps"},
                }
            start = t_to_state_after[t_ref]

        min_steps = min_steps_within_k(start, goal, compiled_moves, n, k)
        reachable = min_steps is not None

        if bool(expected["reachable"]) != reachable:
            return {
                "ok": False,
                "fail_type": "PROBE_REACHABILITY_MISMATCH",
                "invariant_diff": {
                    "probe_id": probe_id,
                    "start": start.as_dict(),
                    "goal": goal.as_dict(),
                    "k": k,
                    "expected_reachable": bool(expected["reachable"]),
                    "got_reachable": reachable,
                    "got_min_steps": min_steps,
                },
                "details": {"why": "BFS reachability does not match expected.reachable"},
            }

        if "min_steps" in expected:
            if not reachable:
                return {
                    "ok": False,
                    "fail_type": "PROBE_MIN_STEPS_BUT_UNREACHABLE",
                    "invariant_diff": {"probe_id": probe_id, "expected_min_steps": int(expected["min_steps"])},
                    "details": {"why": "expected.min_steps provided but goal is unreachable"},
                }
            if int(expected["min_steps"]) != int(min_steps):
                return {
                    "ok": False,
                    "fail_type": "PROBE_MIN_STEPS_MISMATCH",
                    "invariant_diff": {
                        "probe_id": probe_id,
                        "start": start.as_dict(),
                        "goal": goal.as_dict(),
                        "k": k,
                        "expected_min_steps": int(expected["min_steps"]),
                        "got_min_steps": int(min_steps),
                    },
                    "details": {"why": "BFS minimum steps does not match expected.min_steps"},
                }

        checked += 1

    return {"ok": True, "value": {"gate": 5, "steps_checked": len(steps), "probes_checked": checked}}


def move_sort_key(m: Move) -> str:
    if m.name != "lambda_k":
        return m.name
    return f"{m.name}:{int(m.k) if m.k is not None else -1}"


def _main() -> int:
    ap = argparse.ArgumentParser(description="Validate QA_REACHABILITY_DESCENT_RUN_CERT.v1 certificates.")
    ap.add_argument("cert_json", nargs="?", help="Path to certificate JSON")
    ap.add_argument("--demo", action="store_true", help="Validate the PASS fixture and exit")
    ap.add_argument("--self-test", action="store_true", help="Validate PASS plus all FAIL_* fixtures")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if args.demo:
        cert_path = _here("fixtures", "PASS_N6_v1.json")
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

    pass_path = os.path.join(fixtures_dir, "PASS_N6_v1.json")
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
        expected_fail_type = m.group(1)
        accepted_fail_fixtures.append((fn, expected_fail_type))

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
