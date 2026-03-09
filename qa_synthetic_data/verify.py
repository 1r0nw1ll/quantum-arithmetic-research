"""
Deterministic verifier for QA synthetic tasks.
Re-runs computation independently and checks the stored answer.
"""

from core import qa_step, qa_norm, compute_orbit, compute_all_orbits, classify_orbit


def verify_row(row: dict) -> bool:
    task_type = row["task_type"]
    inp = row["input"]
    answer = row["answer"]
    modulus = inp["modulus"]

    if task_type == "invariant_pred":
        return answer == qa_norm(inp["b"], inp["e"], modulus)

    elif task_type == "orbit_class":
        orbit = compute_orbit(inp["b"], inp["e"], modulus)
        all_orbits = compute_all_orbits(modulus)
        max_len = max(len(o) for o in all_orbits.values())
        return answer == classify_orbit(len(orbit), max_len)

    elif task_type == "reachability":
        start = (inp["b"], inp["e"])
        target = (inp["b_target"], inp["e_target"])
        orbit = compute_orbit(*start, modulus)
        return answer == (target in orbit)

    elif task_type == "shortest_witness":
        start = (inp["b"], inp["e"])
        target = (inp["b_target"], inp["e_target"])
        orbit = compute_orbit(*start, modulus)
        orbit_idx = {s: i for i, s in enumerate(orbit)}
        if target not in orbit_idx:
            return answer == -1
        L = len(orbit)
        steps = (orbit_idx[target] - orbit_idx[start]) % L
        return answer == steps

    return False
