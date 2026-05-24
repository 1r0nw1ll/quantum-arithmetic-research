"""
Deterministic benchmark workload fixture builder.

Five query modes, each designed to test a distinct regime:

  full_structured      All QA predicates active. QA wins H1/H2.
  orbit_only           Only orbit_val filter. QA rich buckets degenerate to orbit index.
                       Expected: H1 fails (waste(qa) ≈ waste(table)).
  random_attribute     b_mod / e_mod predicates, uncorrelated with QA law.
                       QA cannot bucket by these → H1 fails.
  range_only           b/e range predicates. Table column-range scan is the natural fit.
                       QA orbit pre-filter gives no advantage over b/e ranges.
                       Expected: H1 fails; table may win.
  mixed_heterogeneous  50/50 split of full_structured and random_attribute queries.
                       Partial support expected; report honestly.

Each query dict contains the full set of predicate fields (unused fields set to
sentinel _UNCONSTRAINED / None / False so _passes_filter handles them uniformly).

Sentinel value for "no constraint on this numeric predicate":
  _UNCONSTRAINED = 10**15  — safely above max possible I or area for N ≤ 1000.
"""
from __future__ import annotations
import random
import math
from typing import Any

from qa_backend import QAPacket

_UNCONSTRAINED: int = 10 ** 15

# Modes where H1 should fail (QA pre-filter advantage shrinks/disappears)
FALSIFIER_MODES = {"orbit_only", "random_attribute", "range_only"}

VALID_MODES = {
    "full_structured",
    "orbit_only",
    "random_attribute",
    "range_only",
    "mixed_heterogeneous",
}


def _base_query(qi: int, seeds: list[tuple[int, int]], k: int,
                orbit_mod: int, orbit_val: int) -> dict[str, Any]:
    """Skeleton with all predicate fields at their "unconstrained" sentinels."""
    return {
        "query_id": f"q{qi:04d}",
        "seeds": seeds,
        "k": k,
        "orbit_mod": orbit_mod,
        "orbit_val": orbit_val,
        # QA-structured predicates — default unconstrained
        "i_gap_max": _UNCONSTRAINED,
        "area_max": _UNCONSTRAINED,
        "shape_sig": None,
        "parity": None,
        "require_primitive": False,
        "axis_check": False,
        "pyth_check": False,
        "broad_expand": False,
        # Non-QA predicates — default absent
        "b_mod_n": None,    # b % b_mod_n == b_mod_val
        "b_mod_val": None,
        "e_mod_n": None,
        "e_mod_val": None,
        "b_lo": None,       # b_lo <= b <= b_hi
        "b_hi": None,
        "e_lo": None,       # e_lo <= e <= e_hi
        "e_hi": None,
    }


# ── Mode builders ──────────────────────────────────────────────────────────────

def _build_full_structured(
    N: int, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tightness_choices = [1.0, 1.5, 3.0, 8.0, 32.0, 256.0]
    queries = []
    for qi in range(n_queries):
        n_seeds = rng.randint(1, 3)
        seed_pkts = [QAPacket(rng.randint(1, N), rng.randint(1, N))
                     for _ in range(n_seeds)]
        ref = seed_pkts[0]
        k = rng.choice([2, 3, 4])
        orbit_mod = rng.choice([9, 24])
        orbit_val = ref.orbit_9 if orbit_mod == 9 else ref.orbit_24
        tight = rng.choice(tightness_choices)
        i_gap_max = max(ref.I, int(ref.I * tight) + 1)
        area_max = max(ref.area, int(ref.area * tight) + 1)
        shape_sig = ref.shape_sig if rng.random() < 0.35 else None
        parity = ref.parity if rng.random() < 0.4 else None
        require_primitive = (rng.random() < 0.25) and ref.primitive
        axis_check = rng.random() < 0.5
        pyth_check = rng.random() < 0.5
        broad_expand = (k >= 3
                        and i_gap_max <= max(1, ref.I + 5)
                        and area_max <= max(1, ref.area + ref.area // 2))
        q = _base_query(qi, [(p.b, p.e) for p in seed_pkts], k, orbit_mod, orbit_val)
        q.update({
            "i_gap_max": i_gap_max,
            "area_max": area_max,
            "shape_sig": shape_sig,
            "parity": parity,
            "require_primitive": require_primitive,
            "axis_check": axis_check,
            "pyth_check": pyth_check,
            "broad_expand": broad_expand,
        })
        queries.append(q)
    return queries


def _build_orbit_only(
    N: int, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    """Only orbit_val constraint.  No i_gap, area, shape_sig, parity, etc.
    QA rich buckets degenerate to the same orbit index table uses → H1 should fail."""
    rng = random.Random(seed)
    queries = []
    for qi in range(n_queries):
        n_seeds = rng.randint(1, 3)
        seed_pkts = [QAPacket(rng.randint(1, N), rng.randint(1, N))
                     for _ in range(n_seeds)]
        ref = seed_pkts[0]
        k = rng.choice([2, 3, 4])
        orbit_mod = rng.choice([9, 24])
        orbit_val = ref.orbit_9 if orbit_mod == 9 else ref.orbit_24
        # All other predicates left at unconstrained sentinels
        q = _base_query(qi, [(p.b, p.e) for p in seed_pkts], k, orbit_mod, orbit_val)
        queries.append(q)
    return queries


def _build_random_attribute(
    N: int, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    """Modular predicates on raw b, e — uncorrelated with QA orbit/i_gap/area structure.
    QA cannot pre-filter these using invariant buckets → H1 should fail."""
    rng = random.Random(seed)
    # Moduli not correlated with 9 or 24
    non_qa_moduli = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    queries = []
    for qi in range(n_queries):
        n_seeds = rng.randint(1, 3)
        seed_pkts = [QAPacket(rng.randint(1, N), rng.randint(1, N))
                     for _ in range(n_seeds)]
        ref = seed_pkts[0]
        k = rng.choice([2, 3, 4])
        orbit_mod = rng.choice([9, 24])
        orbit_val = ref.orbit_9 if orbit_mod == 9 else ref.orbit_24
        b_mod_n = rng.choice(non_qa_moduli)
        b_mod_val = ref.b % b_mod_n   # seed always passes
        # 60% of queries also add e_mod constraint
        e_mod_n = rng.choice(non_qa_moduli) if rng.random() < 0.6 else None
        e_mod_val = (ref.e % e_mod_n) if e_mod_n is not None else None
        q = _base_query(qi, [(p.b, p.e) for p in seed_pkts], k, orbit_mod, orbit_val)
        q.update({
            "b_mod_n": b_mod_n,
            "b_mod_val": b_mod_val,
            "e_mod_n": e_mod_n,
            "e_mod_val": e_mod_val,
        })
        queries.append(q)
    return queries


def _build_range_only(
    N: int, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    """b/e range predicates. QA orbit-based pre-filter provides no structural advantage
    for range lookups.  Table column-range index is the natural fit; both degrade similarly
    here, but the key point is QA wins nothing extra beyond orbit."""
    rng = random.Random(seed)
    queries = []
    for qi in range(n_queries):
        n_seeds = rng.randint(1, 3)
        seed_pkts = [QAPacket(rng.randint(1, N), rng.randint(1, N))
                     for _ in range(n_seeds)]
        ref = seed_pkts[0]
        k = rng.choice([2, 3, 4])
        orbit_mod = rng.choice([9, 24])
        orbit_val = ref.orbit_9 if orbit_mod == 9 else ref.orbit_24
        # Tight b/e range: window of width ~10-30% of N, centred on ref
        half_w_b = rng.randint(max(1, N // 20), max(2, N // 8))
        b_lo = max(1, ref.b - half_w_b)
        b_hi = min(N, ref.b + half_w_b)
        half_w_e = rng.randint(max(1, N // 20), max(2, N // 8))
        e_lo = max(1, ref.e - half_w_e)
        e_hi = min(N, ref.e + half_w_e)
        q = _base_query(qi, [(p.b, p.e) for p in seed_pkts], k, orbit_mod, orbit_val)
        q.update({
            "b_lo": b_lo, "b_hi": b_hi,
            "e_lo": e_lo, "e_hi": e_hi,
        })
        queries.append(q)
    return queries


def _build_mixed_heterogeneous(
    N: int, n_queries: int, seed: int
) -> list[dict[str, Any]]:
    """50% full_structured + 50% random_attribute.  Report honestly."""
    rng = random.Random(seed)
    half = n_queries // 2
    structured = _build_full_structured(N, half, rng.randint(0, 10**9))
    random_attr = _build_random_attribute(N, n_queries - half, rng.randint(0, 10**9))
    # Re-number and shuffle
    all_q = structured + random_attr
    rng.shuffle(all_q)
    for i, q in enumerate(all_q):
        q["query_id"] = f"q{i:04d}"
    return all_q


# ── Public API ─────────────────────────────────────────────────────────────────

def build_workload(
    N: int = 250,
    n_queries: int = 250,
    seed: int = 42,
    query_mode: str = "full_structured",
) -> list[dict[str, Any]]:
    if query_mode not in VALID_MODES:
        raise ValueError(f"Unknown query_mode {query_mode!r}. "
                         f"Valid: {sorted(VALID_MODES)}")
    builders = {
        "full_structured": _build_full_structured,
        "orbit_only": _build_orbit_only,
        "random_attribute": _build_random_attribute,
        "range_only": _build_range_only,
        "mixed_heterogeneous": _build_mixed_heterogeneous,
    }
    return builders[query_mode](N, n_queries, seed)
