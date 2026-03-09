"""
QA mod-9 arithmetic core.
States: (b, e) where b, e in {0, ..., 8}.
Update: d = (b + e) % 9, a = (b + 2*e) % 9.
"""

MODULUS = 9


def qa_step(b: int, e: int) -> tuple:
    return (b + e) % MODULUS, (b + 2 * e) % MODULUS


def qa_norm(b: int, e: int) -> int:
    """f(b,e) = b^2 + b*e - e^2 mod 9"""
    return (b * b + b * e - e * e) % MODULUS


def compute_orbit(b: int, e: int) -> list:
    orbit = [(b, e)]
    cur = qa_step(b, e)
    while cur != (b, e):
        orbit.append(cur)
        cur = qa_step(*cur)
    return orbit


def all_states() -> list:
    return [(b, e) for b in range(MODULUS) for e in range(MODULUS)]


def compute_all_orbits() -> dict:
    """Returns {(b,e): orbit_list} for all states."""
    visited = {}
    result = {}
    for state in all_states():
        if state in visited:
            result[state] = result[visited[state]]
            continue
        orbit = compute_orbit(*state)
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
