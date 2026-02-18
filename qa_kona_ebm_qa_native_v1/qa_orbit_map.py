"""
qa_orbit_map.py

Enumerates and classifies all 81 QA state-manifold states (b,e) for mod 9.

QA step: (b,e) -> (e, d) where d = (b+e) % 9; if d==0 then d=9.

Orbit classification:
  - SINGULARITY: orbit length 1  (1 state:  (9,9))
  - SATELLITE:   orbit length 8  (8 states)
  - COSMOS:      orbit length 24 (72 states in 3 distinct 24-cycles)

Canonical state ordering: sorted([(b,e) for b in range(1,10) for e in range(1,10)])
Orbit IDs: enumerate distinct orbits in order of first appearance in canonical sorted
           state list. Cosmos sub-orbits get IDs 0,1,2; Satellite gets 3; Singularity 4.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Dict, List, Tuple


def qa_step(b: int, e: int, mod: int = 9) -> Tuple[int, int]:
    d = (b + e) % mod
    if d == 0:
        d = mod
    return (e, d)


def get_orbit(b: int, e: int, mod: int = 9) -> List[Tuple[int, int]]:
    orbit = []
    state = (b, e)
    seen = set()
    while state not in seen:
        seen.add(state)
        orbit.append(state)
        state = qa_step(state[0], state[1], mod)
    return orbit


def _orbit_label(length: int) -> str:
    if length == 1:
        return "SINGULARITY"
    if length == 8:
        return "SATELLITE"
    if length == 24:
        return "COSMOS"
    raise ValueError(f"Unexpected orbit length {length}")


def build_orbit_map(mod: int = 9):
    states = sorted([(b, e) for b in range(1, mod + 1) for e in range(1, mod + 1)])

    orbit_members = {}
    state_to_rep = {}
    for b, e in states:
        orbit = get_orbit(b, e, mod)
        rep = min(orbit)
        state_to_rep[(b, e)] = rep
        if rep not in orbit_members:
            orbit_members[rep] = orbit

    first_appearance = {}
    for idx, (b, e) in enumerate(states):
        rep = state_to_rep[(b, e)]
        if rep not in first_appearance:
            first_appearance[rep] = idx

    reps_by_appearance = sorted(first_appearance.keys(), key=lambda r: first_appearance[r])

    rep_to_id = {}
    cosmos_counter = 0
    for rep in reps_by_appearance:
        length = len(orbit_members[rep])
        lbl = _orbit_label(length)
        if lbl == "COSMOS":
            rep_to_id[rep] = cosmos_counter
            cosmos_counter += 1
        elif lbl == "SATELLITE":
            rep_to_id[rep] = 3
        elif lbl == "SINGULARITY":
            rep_to_id[rep] = 4

    orbit_labels = []
    orbit_ids = []
    for b, e in states:
        rep = state_to_rep[(b, e)]
        length = len(orbit_members[rep])
        orbit_labels.append(_orbit_label(length))
        orbit_ids.append(rep_to_id[rep])

    return states, orbit_labels, orbit_ids


def orbit_map_hash(mod: int = 9) -> str:
    states, orbit_labels, orbit_ids = build_orbit_map(mod)
    payload = json.dumps(
        {
            "states": [[b, e] for b, e in states],
            "orbit_labels": orbit_labels,
            "orbit_ids": orbit_ids,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    states, orbit_labels, orbit_ids = build_orbit_map()
    print(f"{'idx':>4}  {'(b,e)':>6}  {'orbit_label':>12}  {'orbit_id':>9}")
    print("-" * 42)
    for i, ((b, e), lbl, oid) in enumerate(zip(states, orbit_labels, orbit_ids)):
        print(f"{i:>4}  ({b},{e})  {lbl:>12}  {oid:>9}")

    label_counts = Counter(orbit_labels)
    print()
    print("Orbit summary:")
    for lbl in ["COSMOS", "SATELLITE", "SINGULARITY"]:
        print(f"  {lbl}: {label_counts[lbl]} states")
    print(f"  Total: {len(states)} states")
    print()
    print(f"orbit_map_hash: {orbit_map_hash()}")
