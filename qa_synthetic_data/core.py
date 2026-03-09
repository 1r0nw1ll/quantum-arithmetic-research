"""
QA modular arithmetic core.
States: (b, e) where b, e in {0, ..., modulus-1}.
Update: d = (b + e) % modulus, a = (b + 2*e) % modulus.
Supports any modulus >= 2.
"""


def qa_step(b: int, e: int, modulus: int) -> tuple:
    return (b + e) % modulus, (b + 2 * e) % modulus


def qa_norm(b: int, e: int, modulus: int) -> int:
    """f(b,e) = b^2 + b*e - e^2 mod modulus"""
    return (b * b + b * e - e * e) % modulus


def compute_orbit(b: int, e: int, modulus: int) -> list:
    orbit = [(b, e)]
    cur = qa_step(b, e, modulus)
    while cur != (b, e):
        orbit.append(cur)
        cur = qa_step(*cur, modulus)
    return orbit


def all_states(modulus: int) -> list:
    return [(b, e) for b in range(modulus) for e in range(modulus)]


def compute_all_orbits(modulus: int) -> dict:
    """Returns {(b,e): orbit_list} for all states."""
    visited = {}
    result = {}
    for state in all_states(modulus):
        if state in visited:
            result[state] = result[visited[state]]
            continue
        orbit = compute_orbit(*state, modulus)
        for s in orbit:
            visited[s] = orbit[0]
        result[orbit[0]] = orbit
        for s in orbit[1:]:
            result[s] = orbit
    return result


def classify_orbit(orbit_length: int, max_orbit_length: int) -> str:
    if orbit_length == 1:
        return "singularity"
    elif orbit_length == max_orbit_length:
        return "cosmos"
    else:
        return "satellite"
