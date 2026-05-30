#!/usr/bin/env python3
"""
FST/Briddell domain sweep: QA orbit classification of Field Structure Theory integers.

For each FST loop count / structural integer, enumerate all Pythagorean triples
directly (trial divisor pairs — all FST values are small), classify each triple's
(d, e) Fermat direction by QA orbit (cosmos / satellite / singularity).

QA_COMPLIANCE = observer_projection_experiment
# noqa: T2-D-1 — inputs are structural integers from FST primary source,
#                 not random distributions; Pythagorean triple is the observer projection
"""

import sys
import math
import json
from collections import Counter, defaultdict
from typing import Iterator

sys.path.insert(0, "/Users/player3/signal_experiments")
from qa_orbit_rules import orbit_family


def a1(v: int, m: int = 24) -> int:
    """A1-compliant mod reduction: result in {1,...,m}, never 0."""
    r = v % m
    return r if r else m


def all_divisor_pairs(F: int) -> Iterator[tuple[int, int]]:
    """Yield (b, a) with b*a=F, 1 ≤ b ≤ a, via trial division up to sqrt(F)."""
    for b in range(1, math.isqrt(F) + 1):
        if F % b == 0:
            a = F // b
            yield b, a


def pythagorean_triples_from_F(F: int) -> list[dict]:
    """
    Enumerate all Pythagorean triples where F = d²-e² (i.e. F is one leg).

    For each same-parity divisor pair (b, a) with b*a=F, b < a:
      d = (a+b)/2,  e = (a-b)/2
      C = d²+e²  (hypotenuse),  G = 2*d*e  (other leg)
      prim = gcd(d,e) == 1
    """
    triples = []
    for b, a in all_divisor_pairs(F):
        if b == a:
            continue  # degenerate (F is a perfect square, b=a=sqrt(F), e=0)
        if (a + b) % 2 != 0:
            continue  # different parity → non-integer d,e
        d = (a + b) // 2
        e = (a - b) // 2
        if e == 0:
            continue  # degenerate
        C = d * d + e * e
        G = 2 * d * e
        is_prim = math.gcd(d, e) == 1
        triples.append({
            "b_div": b, "a_div": a,
            "d": d, "e": e,
            "C": C, "G": G,
            "is_primitive": is_prim,
            "d_mod24": d % 24,
            "e_mod24": e % 24,
            "b_qa": a1(d),
            "e_qa": a1(e),
            "orbit": orbit_family(a1(d), a1(e), 24),
        })
    return triples


# ── FST integer catalog ────────────────────────────────────────────────────────
# Source: qa_fst/qa_fst_completion_paper.tex (Briddell/Dale, April 2026 v2)

FST_CATALOG = [
    # STF hierarchy: 3^k
    (3,    "STF-1",        "3^1"),
    (9,    "STF-2",        "3^2"),
    (27,   "STF-3",        "3^3"),
    (81,   "STF-4",        "3^4"),
    (243,  "STF-5",        "3^5 / pion-range"),
    (729,  "STF-6",        "3^6 / up-quark-cluster"),
    (2187, "lambda",       "3^7 / lambda loop count"),
    (6561, "STF-8",        "3^8 (next hypothetical)"),

    # Proton and sub-clusters
    (1836, "proton",       "2^2*3^3*17; loops between iter-6 and iter-7"),
    (378,  "top-cluster",  "243+81+27+27; top sub-triangle of proton"),
    (351,  "lambda-diff",  "2187-1836 = 3^3*13; actual lambda→proton gap"),
    (729,  "up-quark",     "3^6; each of two bottom sub-triangles"),
    (1458, "two-729",      "2*729; bottom two sub-triangles combined"),

    # Extended: doublings and other combinations
    (4374, "2x-lambda",    "2*2187"),
    (3672, "2x-proton",    "2*1836"),
    (17,   "17-factor",    "prime in 1836=4*27*17; why 17?"),
    (51,   "3x17",         "3*17; product of two factors of 1836"),
    (204,  "12x17",        "12*17; another combo"),
    (1836*2187, "proton-times-lambda", "1836*2187; product"),
]

# Deduplicate by F
_seen: dict[int, list[str]] = defaultdict(list)
for F, label, note in FST_CATALOG:
    _seen[F].append(label)
CATALOG = [(F, labels[0], note) for F, labels, note in
           [(F, _seen[F], next(n for _F,_,n in FST_CATALOG if _F==F)) for F in dict.fromkeys(f for f,_,_ in FST_CATALOG)]]


# ── Run sweep ──────────────────────────────────────────────────────────────────

print("=" * 80)
print("FST / Briddell — QA Pythagorean Orbit Sweep")
print("=" * 80)
print(f"{'F':>8}  {'label':18s}  {'n_tri':>5}  all-orbits                    prim-orbits")
print("-" * 80)

all_orbits = Counter()
prim_orbits_global = Counter()
results = []

for F, label, note in CATALOG:
    triples = pythagorean_triples_from_F(F)
    oc = Counter(t["orbit"] for t in triples)
    pc = Counter(t["orbit"] for t in triples if t["is_primitive"])
    for o, n in oc.items():
        all_orbits[o] += n
    for o, n in pc.items():
        prim_orbits_global[o] += n

    oa = "  ".join(f"{o}:{n}" for o, n in sorted(oc.items())) or "none"
    pa = "  ".join(f"{o}:{n}" for o, n in sorted(pc.items())) or "none"
    print(f"{F:>8}  {label:18s}  {len(triples):>5}  [{oa}]  prim=[{pa}]")

    results.append({
        "F": F, "label": label, "note": note,
        "triple_count": len(triples),
        "orbit_counts": dict(oc),
        "prim_orbit_counts": dict(pc),
        "triples": triples,
    })

print("=" * 80)
total = sum(all_orbits.values())
ptotal = sum(prim_orbits_global.values())
print(f"\nGlobal orbit distribution — ALL triples ({total} total):")
for orb in ("cosmos", "satellite", "singularity"):
    n = all_orbits.get(orb, 0)
    print(f"  {orb:12s}: {n:4d} / {total:4d} = {100*n/max(1,total):5.1f}%")

print(f"\nGlobal orbit distribution — PRIMITIVE triples ({ptotal} total):")
for orb in ("cosmos", "satellite", "singularity"):
    n = prim_orbits_global.get(orb, 0)
    print(f"  {orb:12s}: {n:4d} / {ptotal:4d} = {100*n/max(1,ptotal):5.1f}%")

# ── Non-cosmos triples detail ─────────────────────────────────────────────────
print("\n── Satellite / Singularity triples (non-Cosmos) ────────────────────────")
found_any = False
for entry in results:
    for t in entry["triples"]:
        if t["orbit"] != "cosmos":
            found_any = True
            print(
                f"  F={entry['F']:>8} ({entry['label']:18s})  "
                f"div=({t['b_div']},{t['a_div']})  "
                f"d%24={t['d_mod24']:2d}  e%24={t['e_mod24']:2d}  "
                f"orbit={t['orbit']:12s}  prim={t['is_primitive']}"
            )
if not found_any:
    print("  (all triples are Cosmos)")

# ── Powers-of-3 table ─────────────────────────────────────────────────────────
print("\n── STF hierarchy (powers of 3) orbit table ─────────────────────────────")
for entry in results:
    if entry["F"] in {3, 9, 27, 81, 243, 729, 2187, 6561}:
        oc = entry["orbit_counts"]
        pc = entry["prim_orbit_counts"]
        print(f"  3^k={entry['F']:>6}  all={oc}  prim={pc}")
        # Show divisor pairs for small values
        if entry["F"] <= 729:
            for t in entry["triples"]:
                print(
                    f"         div=({t['b_div']:>4},{t['a_div']:>4})  "
                    f"d={t['d']:>4} e={t['e']:>4}  "
                    f"d%24={t['d_mod24']:2d} e%24={t['e_mod24']:2d}  "
                    f"orbit={t['orbit']}  prim={t['is_primitive']}"
                )

# ── Proton detail ─────────────────────────────────────────────────────────────
print("\n── Proton (1836) — full orbit decomposition ──────────────────────────────")
proton = next(e for e in results if e["F"] == 1836)
for t in proton["triples"]:
    print(
        f"  div=({t['b_div']:>5},{t['a_div']:>5})  "
        f"d={t['d']:>5} e={t['e']:>5}  "
        f"d%24={t['d_mod24']:2d} e%24={t['e_mod24']:2d}  "
        f"orbit={t['orbit']:12s}  prim={t['is_primitive']}"
    )

# ── Save ──────────────────────────────────────────────────────────────────────
import os
os.makedirs("/Users/player3/signal_experiments/results/geometry", exist_ok=True)
out = "/Users/player3/signal_experiments/results/geometry/fst_orbit_sweep_2026-05-30.json"
with open(out, "w") as f:
    json.dump({
        "catalog_size": len(CATALOG),
        "results": results,
        "global_orbit_tally": dict(all_orbits),
        "global_prim_orbit_tally": dict(prim_orbits_global),
    }, f, indent=2, default=str)
print(f"\nSaved: {out}")
