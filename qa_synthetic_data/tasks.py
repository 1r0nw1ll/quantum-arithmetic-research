"""
QA synthetic task generators.
Four task families:
  1. reachability      — is (b*,e*) reachable from (b,e)?
  2. shortest_witness  — minimum steps from (b,e) to (b*,e*)
  3. invariant_pred    — compute f(b,e) = (b²+be-e²) mod modulus
  4. orbit_class       — classify orbit as cosmos/satellite/singularity
"""

import hashlib
import json
from core import (
    qa_step, qa_norm, compute_orbit, all_states,
    compute_all_orbits, classify_orbit,
)


def _canonical_hash(input_dict: dict) -> str:
    payload = json.dumps(input_dict, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(b"QA_SYNTHETIC\x00" + payload.encode()).hexdigest()


def _difficulty_steps(steps: int) -> str:
    if steps <= 2:
        return "easy"
    elif steps <= 5:
        return "medium"
    else:
        return "hard"


def _difficulty_orbit(orbit_class: str) -> str:
    if orbit_class == "singularity":
        return "easy"
    elif orbit_class == "satellite":
        return "medium"
    else:
        return "hard"


def _make_row(task_type, input_dict, answer, witness, difficulty):
    return {
        "task_type": task_type,
        "input": input_dict,
        "answer": answer,
        "witness": witness,
        "difficulty": difficulty,
        "canonical_hash": _canonical_hash({"task_type": task_type, **input_dict}),
    }


def generate_reachability_tasks(orbits: dict, modulus: int, max_orbit_length: int) -> list:
    """Generate reachability tasks with balanced True/False ratio.

    For each state (b,e), generates:
      - L True tasks (all within-orbit reachable targets)
      - L False tasks (unreachable targets from outside the orbit, cycling
        through available external states to match the True count)

    This produces an approximately 50/50 True/False ratio per orbit,
    replacing the previous 1-False-per-state design which yielded ~92% True.
    """
    rows = []
    external_cache: dict = {}

    for (b, e), orbit in orbits.items():
        orbit_set = {s: i for i, s in enumerate(orbit)}
        L = len(orbit)
        start_idx = orbit_set[(b, e)]

        # True tasks — all within-orbit pairs
        for (bt, et), idx_t in orbit_set.items():
            input_dict = {"b": b, "e": e, "b_target": bt, "e_target": et, "modulus": modulus}
            steps = (idx_t - start_idx) % L
            rows.append(_make_row("reachability", input_dict, True, steps,
                                  _difficulty_steps(steps)))

        # False tasks — L unreachable targets (cycle through external states)
        orbit_id = id(orbit)
        if orbit_id not in external_cache:
            external_cache[orbit_id] = [s for s in all_states(modulus)
                                        if s not in orbit_set]
        external = external_cache[orbit_id]
        if external:
            for k in range(L):
                other = external[k % len(external)]
                input_dict = {"b": b, "e": e,
                              "b_target": other[0], "e_target": other[1],
                              "modulus": modulus}
                rows.append(_make_row("reachability", input_dict, False, None, "easy"))

    return rows


def generate_shortest_witness_tasks(orbits: dict, modulus: int, max_orbit_length: int) -> list:
    rows = []
    for (b, e), orbit in orbits.items():
        orbit_set = {s: i for i, s in enumerate(orbit)}
        L = len(orbit)
        start_idx = orbit_set[(b, e)]
        for (bt, et), idx_t in orbit_set.items():
            steps = (idx_t - start_idx) % L
            witness_states = [list(orbit[(start_idx + k) % L]) for k in range(steps + 1)]
            input_dict = {"b": b, "e": e, "b_target": bt, "e_target": et, "modulus": modulus}
            rows.append(_make_row(
                "shortest_witness", input_dict,
                steps, witness_states,
                _difficulty_steps(steps)
            ))
    return rows


def generate_invariant_tasks(modulus: int) -> list:
    rows = []
    for (b, e) in all_states(modulus):
        val = qa_norm(b, e, modulus)
        input_dict = {"b": b, "e": e, "modulus": modulus}
        rows.append(_make_row("invariant_pred", input_dict, val, None, "easy"))
    return rows


def generate_orbit_class_tasks(orbits: dict, modulus: int, max_orbit_length: int) -> list:
    rows = []
    for (b, e), orbit in orbits.items():
        L = len(orbit)
        cls = classify_orbit(L, max_orbit_length)
        input_dict = {"b": b, "e": e, "modulus": modulus}
        rows.append(_make_row("orbit_class", input_dict, cls, L, _difficulty_orbit(cls)))
    return rows


def generate_all_tasks(modulus: int) -> list:
    all_orbit_map = compute_all_orbits(modulus)
    max_orbit_length = max(len(o) for o in all_orbit_map.values())

    rows = []
    rows += generate_invariant_tasks(modulus)
    rows += generate_orbit_class_tasks(all_orbit_map, modulus, max_orbit_length)
    rows += generate_reachability_tasks(all_orbit_map, modulus, max_orbit_length)
    rows += generate_shortest_witness_tasks(all_orbit_map, modulus, max_orbit_length)
    return rows
