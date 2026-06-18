# PRIMARY-SOURCE-EXEMPT: reason=pure-algebra cert validator; primary sources in mapping_protocol_ref.json: Iverson (1993) Pythagorean Arithmetic Vols I-III, Wall (1960) doi:10.1080/00029890.1960.11989541, Dale (2026) Five Families paper

QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic on {1..9}^2; "
    "sigma: (b,e) -> (dr(2e), b) is a permutation of {1..9}^2; "
    "orbit classes Cosmos/Satellite/Singularity are closed under sigma; "
    "Theorem NT: 'octave', 'digital root' are observer projections; no float QA state"
)

"""
Cert [402]: QA Octave Orbit Permutation
The digital-root octave map σ: (b,e) → (dr(2e), b) is a permutation of {1..9}²
with cycle type (1, 4², 12⁶) and order 12. It maps each QA orbit class to itself.

Claims:
  C1  σ is a bijection on {1..9}² (81 → 81)
  C2  Orbit preservation: σ maps Cosmos→Cosmos, Satellite→Satellite, Singularity fixed
  C3  Cycle type: 1 fixed point (Singularity) + 2 four-cycles (Satellite) + 6 twelve-cycles (Cosmos)
  C4  Order: σ^12 = identity on all 81 pairs
  C5  Satellite 4-cycles identified: {(3,3),(6,3),(6,6),(3,6)} and {(6,9),(9,6),(3,9),(9,3)}

The cycle type (1, 4², 12⁶) mirrors the orbit sizes (1, 8, 72):
  Singularity (1 pair) = 1 fixed point of σ
  Satellite (8 pairs) = 2 × 4-cycles of σ
  Cosmos (72 pairs) = 6 × 12-cycles of σ
"""

import json
import hashlib
import sys
from collections import Counter

# Family table from cert [398] (9x9 digital-root classification)
TABLE = {
    (1,1):'F',(1,2):'F',(1,3):'L',(1,4):'P',(1,5):'F',(1,6):'P',(1,7):'L',(1,8):'F',(1,9):'F',
    (2,1):'L',(2,2):'L',(2,3):'F',(2,4):'L',(2,5):'P',(2,6):'P',(2,7):'L',(2,8):'F',(2,9):'L',
    (3,1):'P',(3,2):'P',(3,3):'T',(3,4):'L',(3,5):'F',(3,6):'T',(3,7):'F',(3,8):'L',(3,9):'T',
    (4,1):'F',(4,2):'P',(4,3):'F',(4,4):'P',(4,5):'P',(4,6):'L',(4,7):'L',(4,8):'P',(4,9):'P',
    (5,1):'P',(5,2):'L',(5,3):'L',(5,4):'P',(5,5):'P',(5,6):'F',(5,7):'P',(5,8):'F',(5,9):'P',
    (6,1):'L',(6,2):'F',(6,3):'T',(6,4):'F',(6,5):'L',(6,6):'T',(6,7):'P',(6,8):'P',(6,9):'T',
    (7,1):'F',(7,2):'L',(7,3):'P',(7,4):'P',(7,5):'L',(7,6):'F',(7,7):'L',(7,8):'L',(7,9):'L',
    (8,1):'F',(8,2):'L',(8,3):'P',(8,4):'F',(8,5):'P',(8,6):'L',(8,7):'F',(8,8):'F',(8,9):'F',
    (9,1):'F',(9,2):'L',(9,3):'T',(9,4):'P',(9,5):'P',(9,6):'T',(9,7):'L',(9,8):'F',(9,9):'N',
}
COSMOS = frozenset(p for p, f in TABLE.items() if f in ('F','L','P'))
SATELLITE = frozenset(p for p, f in TABLE.items() if f == 'T')
SINGULARITY = frozenset(p for p, f in TABLE.items() if f == 'N')

ALL_PAIRS = [(b, e) for b in range(1, 10) for e in range(1, 10)]

def dr(n):
    return ((n - 1) % 9) + 1

def sigma(b, e):
    """Octave map: (b,e) -> (dr(2e), b)"""
    return (dr(2 * e), b)

def sigma_k(b, e, k):
    for _ in range(k):
        b, e = sigma(b, e)
    return (b, e)


def self_test():
    results = {"ok": True, "checks": 5, "failures": [], "detail": {}}

    # C1: σ is a bijection
    images = set(sigma(b, e) for b, e in ALL_PAIRS)
    c1_pass = len(images) == 81
    if not c1_pass:
        results["ok"] = False
        results["failures"].append(f"C1: σ not bijective: {len(images)} images ≠ 81")
    results["detail"]["C1"] = {
        "domain_size": 81,
        "image_size": len(images),
        "bijective": c1_pass,
        "pass": c1_pass,
    }

    # C2: Orbit preservation
    cosmos_preserved = all(sigma(b, e) in COSMOS for b, e in COSMOS)
    sat_preserved = all(sigma(b, e) in SATELLITE for b, e in SATELLITE)
    sing_fixed = all(sigma(b, e) == (b, e) for b, e in SINGULARITY)
    c2_pass = cosmos_preserved and sat_preserved and sing_fixed
    if not c2_pass:
        results["ok"] = False
        if not cosmos_preserved:
            results["failures"].append("C2: Cosmos not preserved under σ")
        if not sat_preserved:
            results["failures"].append("C2: Satellite not preserved under σ")
        if not sing_fixed:
            results["failures"].append("C2: Singularity not fixed by σ")
    results["detail"]["C2"] = {
        "cosmos_preserved": cosmos_preserved,
        "satellite_preserved": sat_preserved,
        "singularity_fixed": sing_fixed,
        "pass": c2_pass,
    }

    # C3: Cycle type (1, 4², 12⁶)
    visited = set()
    cycles = []
    for start in ALL_PAIRS:
        if start in visited:
            continue
        cycle = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            cur = sigma(*cur)
        cycles.append(tuple(cycle))
    cycle_lengths = Counter(len(c) for c in cycles)
    expected_cycle_type = {1: 1, 4: 2, 12: 6}
    c3_pass = dict(cycle_lengths) == expected_cycle_type
    if not c3_pass:
        results["ok"] = False
        results["failures"].append(f"C3: cycle type {dict(cycle_lengths)} ≠ expected {expected_cycle_type}")
    # Orbit class of each cycle
    cycle_orbit_classes = []
    for c in cycles:
        fam_set = frozenset(TABLE[p] for p in c)
        fam_label = 'Singularity' if fam_set <= {'N'} else ('Satellite' if fam_set <= {'T'} else 'Cosmos')
        cycle_orbit_classes.append({"length": len(c), "orbit_class": fam_label})
    results["detail"]["C3"] = {
        "cycle_type": dict(cycle_lengths),
        "expected": expected_cycle_type,
        "orbit_classes": sorted(cycle_orbit_classes, key=lambda x: x["length"]),
        "pass": c3_pass,
    }

    # C4: σ^12 = identity
    c4_pass = all(sigma_k(b, e, 12) == (b, e) for b, e in ALL_PAIRS)
    order_check = {1: False, 2: False, 3: False, 4: False, 6: False, 12: True}
    for k in [1, 2, 3, 4, 6]:
        order_check[k] = all(sigma_k(b, e, k) == (b, e) for b, e in ALL_PAIRS)
    if not c4_pass:
        results["ok"] = False
        results["failures"].append("C4: σ^12 ≠ identity")
    results["detail"]["C4"] = {
        "sigma_pow_12_is_identity": c4_pass,
        "order_checks": order_check,
        "order": 12 if c4_pass and not any(order_check[k] for k in [1,2,3,4,6]) else "divides 12",
        "pass": c4_pass,
    }

    # C5: Satellite 4-cycles identified
    sat_cycles = [c for c in cycles if len(c) == 4]
    sat_cycle_sets = [frozenset(c) for c in sat_cycles]
    expected_cycle_A = frozenset([(3,3),(6,3),(6,6),(3,6)])
    expected_cycle_B = frozenset([(6,9),(9,6),(3,9),(9,3)])
    c5_pass = (expected_cycle_A in sat_cycle_sets and expected_cycle_B in sat_cycle_sets)
    if not c5_pass:
        results["ok"] = False
        results["failures"].append(f"C5: Satellite 4-cycles {[set(c) for c in sat_cycles]} ≠ expected")
    results["detail"]["C5"] = {
        "satellite_4_cycles": [sorted(c) for c in sat_cycles],
        "expected_A": sorted(expected_cycle_A),
        "expected_B": sorted(expected_cycle_B),
        "all_satellite_pairs_in_cycles": all(p in expected_cycle_A | expected_cycle_B for p in SATELLITE),
        "pass": c5_pass,
    }

    return results


if __name__ == "__main__":
    result = self_test()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    sha = hashlib.sha256(output.encode()).hexdigest()
    print(f"SHA-256: {sha}", file=sys.stderr)
    if not result["ok"]:
        sys.exit(1)
