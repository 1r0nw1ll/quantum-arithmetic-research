"""Verify Item 1 of Bateson Learning Levels sketch: Level-II operators exist on S_9.

An operator phi: S_9 -> S_9 is Level II iff it is NOT orbit-preserving under the
QA dynamic T(b,e) = (e, qa_mod(b+e, 9)). Level I operators preserve orbit
membership; Level II operators cross orbit boundaries.

We verify Level II existence by:
  1. Enumerating S_9 (81 points)
  2. Computing orbit decomposition under T
  3. Testing candidate endomorphisms
  4. Reporting which cross orbit boundaries (= Level II) and which preserve them (= Level I)
"""

QA_COMPLIANCE = "theory_verification — enumerates finite S_9 state space, tests integer endomorphisms against canonical orbit classification; no observer, no empirical data, no floats"

from qa_arithmetic import qa_step
from qa_orbit_rules import orbit_family  # noqa: ORBIT-5 — canonical source


def enumerate_orbits(m: int = 9):
    """Decompose S_m into T-orbits. Returns list of orbit tuples."""
    seen = set()
    orbits = []
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            orbit = []
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                orbit.append(cur)
                cur = qa_step(cur[0], cur[1], m)
            orbits.append(tuple(orbit))
    return orbits


def orbit_index_map(orbits):
    """Map each point to its orbit index."""
    idx = {}
    for i, orbit in enumerate(orbits):
        for pt in orbit:
            idx[pt] = i
    return idx


def count_orbit_crossings(phi, orbits, idx):
    """Count points (b,e) where phi(b,e) lies in a different orbit."""
    n = 0
    cross = []
    for orbit in orbits:
        for pt in orbit:
            image = phi(*pt)
            if idx.get(image) != idx[pt]:
                n += 1
                cross.append((pt, image, idx[pt], idx.get(image)))
    return n, cross


def family_crossings(phi):
    """Count points where phi changes the orbit FAMILY (cosmos/satellite/singularity)."""
    changes = []
    for b in range(1, 10):
        for e in range(1, 10):
            src_fam = orbit_family(int(b), int(e), 9)
            bi, ei = phi(b, e)
            dst_fam = orbit_family(int(bi), int(ei), 9)
            if src_fam != dst_fam:
                changes.append(((b, e), src_fam, (bi, ei), dst_fam))
    return changes


# Candidate endomorphisms of S_9

def phi_constant(b, e):
    return (3, 3)


def phi_scalar_2(b, e):
    # (b,e) -> (2b mod 9, 2e mod 9), adjusted to {1,...,9}
    return (((2 * b - 1) % 9) + 1, ((2 * e - 1) % 9) + 1)


def phi_scalar_3(b, e):
    # (b,e) -> (3b mod 9, 3e mod 9)
    return (((3 * b - 1) % 9) + 1, ((3 * e - 1) % 9) + 1)


def phi_swap(b, e):
    return (e, b)


def phi_male_to_female(b, e):
    # Double e, swap b<->e: (b,e) -> (2e, b) with mod
    return (((2 * e - 1) % 9) + 1, b)


def phi_reduce_mod3_lift(b, e):
    # Reduce mod 3, lift to {3,6,9} satellite grid
    return (3 * (((b - 1) % 3) + 1), 3 * (((e - 1) % 3) + 1))


def main():
    m = 9
    orbits = enumerate_orbits(m)
    idx = orbit_index_map(orbits)

    print(f"=== S_{m} orbit decomposition ===")
    print(f"Total points: {sum(len(o) for o in orbits)}")
    print(f"Number of orbits: {len(orbits)}")
    for i, o in enumerate(orbits):
        fam = orbit_family(int(o[0][0]), int(o[0][1]), m)
        print(f"  Orbit {i}: length={len(o)}, family={fam}, rep={o[0]}")
    print()

    candidates = [
        ("phi_constant (b,e)->(3,3)", phi_constant),
        ("phi_scalar_2 (b,e)->(2b,2e) mod 9", phi_scalar_2),
        ("phi_scalar_3 (b,e)->(3b,3e) mod 9", phi_scalar_3),
        ("phi_swap (b,e)->(e,b)", phi_swap),
        ("phi_male_to_female (b,e)->(2e,b)", phi_male_to_female),
        ("phi_reduce_mod3_lift", phi_reduce_mod3_lift),
    ]

    print("=== Level-I / Level-II classification of candidate operators ===")
    for name, phi in candidates:
        n_cross, crossings = count_orbit_crossings(phi, orbits, idx)
        fam_changes = family_crossings(phi)
        level = "LEVEL I (orbit-preserving)" if n_cross == 0 else f"LEVEL II ({n_cross}/81 orbit crossings)"
        n_fam = len(fam_changes)
        fam_str = f"{n_fam}/81 family changes" if n_fam > 0 else "preserves families"
        print(f"  {name}")
        print(f"    -> {level}, {fam_str}")
        if fam_changes and n_fam <= 5:
            for ch in fam_changes:
                print(f"       example: {ch}")
        elif fam_changes:
            print(f"       first 3 examples: {fam_changes[:3]}")
        print()

    print("=== Conclusion ===")
    any_level_2 = any(
        count_orbit_crossings(phi, orbits, idx)[0] > 0
        for _, phi in candidates
    )
    print(f"Level-II operators exist on S_9: {any_level_2}")
    any_family_cross = any(
        len(family_crossings(phi)) > 0
        for _, phi in candidates
    )
    print(f"Family-crossing operators exist on S_9: {any_family_cross}")


if __name__ == "__main__":
    main()
