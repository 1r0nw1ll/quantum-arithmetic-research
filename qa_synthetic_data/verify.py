"""
Deterministic verifier for QA synthetic tasks.
Re-runs computation independently and checks the stored answer.
"""

from core import qa_step, qa_norm, compute_orbit, all_states, compute_all_orbits, classify_orbit, MODULUS


def verify_row(row: dict) -> bool:
    task_type = row["task_type"]
    inp = row["input"]
    answer = row["answer"]

    if task_type == "invariant_pred":
        expected = qa_norm(inp["b"], inp["e"])
        return answer == expected

    elif task_type == "orbit_class":
        orbit = compute_orbit(inp["b"], inp["e"])
        all_orbits = compute_all_orbits()
        max_len = max(len(o) for o in all_orbits.values())
        expected = classify_orbit(len(orbit), max_len)
        return answer == expected

    elif task_type == "reachability":
        start = (inp["b"], inp["e"])
        target = (inp["b_target"], inp["e_target"])
        orbit = compute_orbit(*start)
        reachable = target in orbit
        return answer == reachable

    elif task_type == "shortest_witness":
        start = (inp["b"], inp["e"])
        target = (inp["b_target"], inp["e_target"])
        orbit = compute_orbit(*start)
        orbit_idx = {s: i for i, s in enumerate(orbit)}
        if target not in orbit_idx:
            return answer == -1
        L = len(orbit)
        start_idx = orbit_idx[start]
        target_idx = orbit_idx[target]
        expected_steps = (target_idx - start_idx) % L
        return answer == expected_steps

    return False
