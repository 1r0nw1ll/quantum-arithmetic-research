#!/usr/bin/env python3
# scripts/branching_score.py
from __future__ import annotations
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple, Set

def branching_score(problem: dict) -> Dict[str, int]:
    """
    Rule-batch discriminativity scorer.

    Predicts how many rules will fire and how many facts will be generated,
    matching the actual beam search architecture (1 successor per rule).
    """
    givens = problem.get("givens", [])

    # Collect relations
    parallel_adj: Dict[int, Set[int]] = defaultdict(set)
    perp_by_hub: Dict[int, Set[int]] = defaultdict(set)
    perp_pairs: List[Tuple[int,int]] = []
    oncircle: Dict[int, Set[int]] = defaultdict(set)  # circle_id -> points
    online: Dict[int, Set[int]] = defaultdict(set)    # line_id -> points
    coincident_adj: Dict[int, Set[int]] = defaultdict(set)
    concentric_adj: Dict[int, Set[int]] = defaultdict(set)
    equality_adj: Dict[int, Set[int]] = defaultdict(set)

    for f in givens:
        if not isinstance(f, dict) or len(f) != 1:
            continue
        k = next(iter(f.keys()))
        v = f[k]
        if not isinstance(v, list):
            continue

        if k == "Parallel" and len(v) == 2:
            a, b = v
            parallel_adj[a].add(b)
            parallel_adj[b].add(a)

        elif k == "Perpendicular" and len(v) == 2:
            a, b = v
            perp_pairs.append((a, b))
            perp_pairs.append((b, a))
            perp_by_hub[a].add(b)

        elif k == "OnCircle" and len(v) == 2:
            p, c = v
            oncircle[c].add(p)

        elif k == "OnLine" and len(v) == 2:
            p, line = v
            online[line].add(p)

        elif k == "CoincidentLines" and len(v) == 2:
            a, b = v
            coincident_adj[a].add(b)
            coincident_adj[b].add(a)

        elif k == "ConcentricCircles" and len(v) == 2:
            a, b = v
            concentric_adj[a].add(b)
            concentric_adj[b].add(a)

        elif k == "EqualLength" and len(v) == 2:
            a, b = v
            equality_adj[a].add(b)
            equality_adj[b].add(a)

    # Rule-batch predictions
    # 1) PerpendicularToParallel: shared-perp hubs
    shared_perp_pairs = sum(comb(len(spokes), 2) for spokes in perp_by_hub.values() if len(spokes) >= 2)

    # 2) ParallelTransitivity: length-2 paths
    parallel_len2 = sum(comb(len(nbrs), 2) for nbrs in parallel_adj.values() if len(nbrs) >= 2)

    # 3) ParallelPerpendicular: perp + parallel propagation
    perp_parallel_prop = sum(len(parallel_adj.get(a, set())) for hub, a in perp_pairs)

    # 4) OnCircleToConcyclic: C(n,4) quartets per circle
    oncircle_quads = sum(comb(len(points), 4) for points in oncircle.values() if len(points) >= 4)

    # 5) OnLineToCollinear: C(n,3) triplets per line
    online_triplets = sum(comb(len(points), 3) for points in online.values() if len(points) >= 3)

    # 6) CoincidentLineTransitivity: length-2 paths
    coincident_len2 = sum(comb(len(nbrs), 2) for nbrs in coincident_adj.values() if len(nbrs) >= 2)

    # 7) ConcentricTransitivity: length-2 paths
    concentric_len2 = sum(comb(len(nbrs), 2) for nbrs in concentric_adj.values() if len(nbrs) >= 2)

    # 8) EqualityTransitivity: length-2 paths
    equality_len2 = sum(comb(len(nbrs), 2) for nbrs in equality_adj.values() if len(nbrs) >= 2)

    # Fact volume score (total new facts predicted)
    fact_volume_score = (
        shared_perp_pairs + parallel_len2 + perp_parallel_prop +
        oncircle_quads + online_triplets + coincident_len2 +
        concentric_len2 + equality_len2
    )

    # Rule surface score (number of distinct rule families triggered)
    components = [
        shared_perp_pairs, parallel_len2, perp_parallel_prop,
        oncircle_quads, online_triplets, coincident_len2,
        concentric_len2, equality_len2
    ]
    rule_surface_score = sum(1 for c in components if c > 0)

    return {
        "fact_volume_score": fact_volume_score,
        "rule_surface_score": rule_surface_score,
        "shared_perp_pairs": shared_perp_pairs,
        "parallel_len2": parallel_len2,
        "perp_parallel_prop": perp_parallel_prop,
        "oncircle_quads": oncircle_quads,
        "online_triplets": online_triplets,
        "coincident_len2": coincident_len2,
        "concentric_len2": concentric_len2,
        "equality_len2": equality_len2,
    }

def main(paths: List[str]) -> None:
    for p in paths:
        path = Path(p)
        obj = json.loads(path.read_text())
        s = branching_score(obj)

        print(f"\n{path.name}:")
        print(f"  Rule surface score: {s['rule_surface_score']} (distinct rule families)")
        print(f"  Fact volume score:  {s['fact_volume_score']} (total new facts predicted)")
        print(f"  Breakdown:")
        print(f"    - Shared-perp pairs:  {s['shared_perp_pairs']}")
        print(f"    - Parallel len-2:     {s['parallel_len2']}")
        print(f"    - Perp*Parallel prop: {s['perp_parallel_prop']}")
        print(f"    - OnCircle quads:     {s['oncircle_quads']}")
        print(f"    - OnLine triplets:    {s['online_triplets']}")
        print(f"    - Coincident len-2:   {s['coincident_len2']}")
        print(f"    - Concentric len-2:   {s['concentric_len2']}")
        print(f"    - Equality len-2:     {s['equality_len2']}")

        # Discriminativity assessment
        is_discriminative = s['rule_surface_score'] >= 4 and s['fact_volume_score'] >= 25
        status = "✅ PASS" if is_discriminative else "❌ FAIL"
        print(f"  Discriminative: {status} (need rule_surface≥4 AND fact_volume≥25)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/branching_score.py <json> [<json>...]")
        raise SystemExit(2)
    main(sys.argv[1:])
