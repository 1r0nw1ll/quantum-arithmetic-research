QA_COMPLIANCE = "library_module — demo entry point for qa_observer"
"""Self-contained demo of QA orbit dynamics and coherence measurement.

Run:
    python -m qa_observer.demo
"""

import numpy as np

# Use canonical orbit_family when available; standalone fallback otherwise
try:
    from qa_orbit_rules import orbit_family, orbit_period  # noqa: ORBIT-6
    _qa_mod = None  # use canonical
except ImportError:
    orbit_family = None  # populated below via _classify_orbit

def qa_mod(x: int, m: int) -> int:
    """A1-compliant: result in {1,...,m}."""
    return ((int(x) - 1) % m) + 1


def qa_step(b: int, e: int, m: int = 24):
    """One QA step: (b, e) -> (e, d) where d = b + e (mod m, A1)."""
    d = qa_mod(b + e, m)  # noqa: A2-1
    return e, d


def _orbit_period(b: int, e: int, m: int = 24) -> int:
    """Compute orbit period of (b, e) under the T-operator."""
    cb, ce = qa_step(b, e, m)
    steps = 1
    while (cb, ce) != (b, e):
        cb, ce = qa_step(cb, ce, m)
        steps += 1
        if steps > m * m:
            break
    return steps


def _classify_orbit(b: int, e: int, m: int = 24) -> str:
    """Classify orbit: Singularity (period 1), Satellite (period 8), Cosmos (period 24)."""
    period = _orbit_period(b, e, m)
    if period == 1:
        return "Singularity"
    elif period <= 8:
        return "Satellite"
    else:
        return "Cosmos"


# Wire up: canonical if available, fallback otherwise
if orbit_family is None:
    orbit_family = _classify_orbit
    orbit_period = _orbit_period


def demo_orbits(m: int = 24):
    """Show the three orbit types with example trajectories."""
    print(f"=== QA Orbit Structure (mod {m}) ===\n")

    examples = [
        (24, 24, "Singularity"),
        (8, 8, "Satellite"),
        (1, 1, "Cosmos"),
    ]
    counts = {}

    for b, e, label in examples:
        period = _orbit_period(b, e, m)
        path = [(b, e)]
        cb, ce = b, e
        for _ in range(min(7, period - 1)):
            cb, ce = qa_step(cb, ce, m)
            path.append((cb, ce))
        trail = " -> ".join(f"({bb},{ee})" for bb, ee in path)
        if period > len(path):
            trail += f" -> ... ({period} total)"
        print(f"  {label:12s}  start=({b},{e})  period={period:2d}  {trail}")

    # Count all pairs
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            fam = _classify_orbit(b, e, m)
            counts[fam] = counts.get(fam, 0) + 1

    print(f"\n  All {m*m} pairs: "
          f"{counts.get('Cosmos', 0)} Cosmos, "
          f"{counts.get('Satellite', 0)} Satellite, "
          f"{counts.get('Singularity', 0)} Singularity")


def demo_qci():
    """Demonstrate QCI (QA Coherence Index) on synthetic signal vs noise."""
    print("\n=== QA Coherence Index (QCI) ===\n")

    m = 24
    np.random.seed(42)

    # Structured signal: Fibonacci-seeded orbit
    structured = []
    b, e = 1, 1
    for _ in range(200):
        structured.append(b)
        b, e = qa_step(b, e, m)

    # Random noise: uniform draws from {1,...,24}
    noise = [int(x) for x in np.random.randint(1, m + 1, size=200)]

    # Compute T-operator match rate (core of QCI)
    def t_match_rate(seq, window=40):
        matches = []
        for t in range(len(seq) - 2):
            pred = qa_mod(seq[t] + seq[t + 1], m)
            matches.append(1 if pred == seq[t + 2] else 0)
        # Rolling mean
        rates = []
        for i in range(len(matches) - window + 1):
            rates.append(sum(matches[i:i + window]) / window)
        return rates

    s_rates = t_match_rate(structured)
    n_rates = t_match_rate(noise)

    s_mean = np.mean(s_rates) if s_rates else 0
    n_mean = np.mean(n_rates) if n_rates else 0
    baseline = 1.0 / m

    print(f"  T-operator prediction accuracy (window=40):")
    print(f"    Structured orbit:  {s_mean:.3f}  (perfect = 1.000)")
    print(f"    Random noise:      {n_mean:.3f}  (chance = {baseline:.3f})")
    print(f"    Separation:        {s_mean - n_mean:.3f}")
    print(f"\n  QCI distinguishes structured dynamics from noise.")


def demo_chromogeometry():
    """Show the chromogeometric identity C^2 + F^2 = G^2 for QA directions."""
    print("\n=== Chromogeometry (Wildberger Theorem 6) ===\n")

    directions = [(2, 1), (3, 2), (5, 3), (8, 5), (13, 8)]
    print(f"  {'(d,e)':>8s}  {'C=2de':>6s}  {'F=d*d-e*e':>10s}  {'G=d*d+e*e':>10s}  C*C+F*F=G*G")
    for d, e in directions:
        C = 2 * d * e
        F = d * d - e * e
        G = d * d + e * e
        check = C * C + F * F == G * G
        print(f"  {str((d,e)):>8s}  {C:6d}  {F:10d}  {G:10d}  {check}")

    print("\n  Green^2 + Red^2 = Blue^2 for ALL integer direction vectors.")
    print("  QA restricts to Fibonacci directions: the arithmetic of geometry.")


def main():
    print("Quantum Arithmetic (QA) Observer Demo")
    print("=" * 42)
    demo_orbits()
    demo_qci()
    demo_chromogeometry()
    print("\n" + "=" * 42)
    print("Learn more: https://github.com/1r0nw1ll/quantum-arithmetic-research")


if __name__ == "__main__":
    main()
