#!/usr/bin/env python3
"""
generate_fixtures.py  —  compute and write all QA_ENERGY_CERT.v1.1 fixtures.

Run from repo root:
    python3 qa_energy_cert_v1_1/generate_fixtures.py

Writes to qa_energy_cert_v1_1/fixtures/*.json
"""
from __future__ import annotations

import json
import os
import sys

# Make validator importable without install
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from validator import (
    DOMAIN_CAPS_TR,
    StateTR,
    _all_states,
    _apply_generator,
    _bfs_from,
    _build_adj,
    _compute_family_interaction_stats,
    _compute_family_power_stats,
    _compute_power_stats,
    _in_domain,
    _reverse_bfs_from,
    _scc_kosaraju,
    _state_to_obj,
)


def _out(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    print(f"  wrote {os.path.basename(path)}")


def _energy_map_list(dist: dict, domain: str) -> list:
    return [{"state": _state_to_obj(s), "energy": e}
            for s, e in sorted(dist.items(),
                                key=lambda x: list(_state_to_obj(x[0]).values()))]


def _return_map_list(rdist: dict, domain: str) -> list:
    return [{"state": _state_to_obj(s), "return_energy": e}
            for s, e in sorted(rdist.items(),
                                key=lambda x: list(_state_to_obj(x[0]).values()))]


def build_cert(
    cert_id: str,
    domain: str,
    N: int,
    ref: object,
    gens: list,
    return_in_k_tests_spec: list,
    power_tests: list | None = None,
    interaction_horizon: int | None = None,
    # Optional overrides for FAIL fixtures:
    override_family_interaction_stats: list | None = None,
) -> dict:
    adj = _build_adj(N, gens, domain)
    dist = _bfs_from(ref, adj)
    rdist = _reverse_bfs_from(ref, adj)

    nodes = _all_states(N, domain)
    comps = _scc_kosaraju(nodes, adj)
    scc_count = len(comps)
    max_scc_size = max(len(c) for c in comps) if comps else 0

    power = _compute_power_stats(dist, adj)
    fam_power = _compute_family_power_stats(gens, dist, adj)
    fam_interact = _compute_family_interaction_stats(gens, dist, adj)

    rik_tests = []
    for state_obj, k, expect in return_in_k_tests_spec:
        rik_tests.append({
            "state": _state_to_obj(state_obj),
            "k": k,
            "expect_can_return": expect,
        })

    cert: dict = {
        "schema_version": "QA_ENERGY_CERT.v1.1",
        "cert_id": cert_id,
        "created_utc": "2026-02-26T00:00:00Z",
        "domain": domain,
        "N": N,
        "reference_state": _state_to_obj(ref),
        "generator_set": gens,
        "energy_map": _energy_map_list(dist, domain),
        "return_energy_map": _return_map_list(rdist, domain),
        "return_in_k_tests": rik_tests,
        "expected_summary": {
            "reachable_count": len(dist),
            "max_energy": max(dist.values()) if dist else 0,
            "scc_count": scc_count,
            "max_scc_size": max_scc_size,
            "power_stats": power,
            "family_power_stats": fam_power,
            "family_interaction_stats": (
                override_family_interaction_stats
                if override_family_interaction_stats is not None
                else fam_interact
            ),
        },
    }

    if power_tests is not None:
        cert["power_tests"] = power_tests
    if interaction_horizon is not None:
        cert["interaction_horizon"] = interaction_horizon

    return cert


def main() -> None:
    fx = os.path.join(_here, "fixtures")
    os.makedirs(fx, exist_ok=True)

    # ── PASS_FEAR: N=2, ref=(0,2), generators: fear_up+fear_down+fear_lock ───
    fear_gens = [
        {"name": "fear_up",   "family_tag": "fear"},
        {"name": "fear_down", "family_tag": "fear"},
        {"name": "fear_lock", "family_tag": "fear"},
    ]
    ref_02 = StateTR(T=0, R=2)
    adj_fear = _build_adj(2, fear_gens, DOMAIN_CAPS_TR)
    dist_fear = _bfs_from(ref_02, adj_fear)
    rdist_fear = _reverse_bfs_from(ref_02, adj_fear)

    pass_fear = build_cert(
        cert_id="energy_caps_tr_N2_fear_only",
        domain=DOMAIN_CAPS_TR,
        N=2,
        ref=ref_02,
        gens=fear_gens,
        return_in_k_tests_spec=[
            (ref_02, 0, True),
            (StateTR(T=2, R=0), 100, False),   # fear-only: can't increase R
            (StateTR(T=1, R=2), 3, True),
        ],
        power_tests=[
            {"name": "fear_up",
             "delta_constraints": {"min_delta_gte": 0, "mean_sign": "positive"}},
            {"name": "fear_down",
             "delta_constraints": {"max_delta_lte": 0, "mean_sign": "negative"}},
            {"name": "fear_lock",
             "delta_constraints": {"min_delta_gte": 0, "mean_sign": "positive"}},
        ],
        interaction_horizon=2,
    )
    _out(os.path.join(fx, "PASS_FEAR.json"), pass_fear)

    # ── PASS_LOVE: N=2, ref=(2,0), generators: love_soothe+love_support+love_reframe
    love_gens = [
        {"name": "love_soothe",  "family_tag": "love"},
        {"name": "love_support", "family_tag": "love"},
        {"name": "love_reframe", "family_tag": "love"},
    ]
    ref_20 = StateTR(T=2, R=0)
    pass_love = build_cert(
        cert_id="energy_caps_tr_N2_love_only",
        domain=DOMAIN_CAPS_TR,
        N=2,
        ref=ref_20,
        gens=love_gens,
        return_in_k_tests_spec=[
            (ref_20, 0, True),
            (StateTR(T=0, R=2), 100, False),  # love-only DAG: no path back to (2,0)
            (StateTR(T=1, R=1), 100, False),  # no return path (love can't increase T)
        ],
        power_tests=[
            {"name": "love_soothe",
             "delta_constraints": {"mean_sign": "nonnegative"}},
            {"name": "love_support",
             "delta_constraints": {"mean_sign": "nonnegative"}},
            {"name": "love_reframe",
             "delta_constraints": {"mean_sign": "nonnegative"}},
        ],
        interaction_horizon=2,
    )
    _out(os.path.join(fx, "PASS_LOVE.json"), pass_love)

    # ── PASS_MIXED: N=2, ref=(0,2), all 6 generators ─────────────────────────
    mixed_gens = [
        {"name": "fear_up",      "family_tag": "fear"},
        {"name": "fear_down",    "family_tag": "fear"},
        {"name": "fear_lock",    "family_tag": "fear"},
        {"name": "love_soothe",  "family_tag": "love"},
        {"name": "love_support", "family_tag": "love"},
        {"name": "love_reframe", "family_tag": "love"},
    ]
    pass_mixed = build_cert(
        cert_id="energy_caps_tr_N2_mixed",
        domain=DOMAIN_CAPS_TR,
        N=2,
        ref=ref_02,
        gens=mixed_gens,
        return_in_k_tests_spec=[
            (ref_02, 0, True),
            (StateTR(T=2, R=0), 3, True),   # via love_soothe×2, should return
            (StateTR(T=1, R=1), 2, True),   # love_soothe(1,1)=(0,2)
        ],
        interaction_horizon=2,
    )
    _out(os.path.join(fx, "PASS_MIXED.json"), pass_mixed)

    # ── FAIL_POWER: PASS_FEAR but with impossible power_tests constraint ──────
    import copy
    fail_power = copy.deepcopy(pass_fear)
    fail_power["cert_id"] = "energy_caps_tr_N2_fear_FAIL_power"
    # fear_up always has positive mean, but we assert it must be negative → fail
    fail_power["power_tests"] = [
        {"name": "fear_up",
         "delta_constraints": {"mean_sign": "negative"}}  # impossible: mean is +1
    ]
    _out(os.path.join(fx, "FAIL_POWER.json"), fail_power)

    # ── FAIL_INTERACTION: PASS_MIXED but one family_interaction_stats entry is wrong
    fail_interact = copy.deepcopy(pass_mixed)
    fail_interact["cert_id"] = "energy_caps_tr_N2_mixed_FAIL_interaction"
    # Corrupt one entry in family_interaction_stats
    fis = fail_interact["expected_summary"]["family_interaction_stats"]
    if fis:
        # Flip the sign of mean_delta_num in the first entry
        fis[0] = dict(fis[0])
        fis[0]["mean_delta_num"] = -fis[0]["mean_delta_num"] - 999
    _out(os.path.join(fx, "FAIL_INTERACTION.json"), fail_interact)

    # ── FAIL_HORIZON: interaction_horizon=3 → DOMAIN_INVALID ─────────────────
    fail_horizon = copy.deepcopy(pass_fear)
    fail_horizon["cert_id"] = "energy_caps_tr_N2_fear_FAIL_horizon"
    fail_horizon["interaction_horizon"] = 3
    _out(os.path.join(fx, "FAIL_HORIZON.json"), fail_horizon)

    print("Done.")


if __name__ == "__main__":
    main()
