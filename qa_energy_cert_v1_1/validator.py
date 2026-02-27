#!/usr/bin/env python3
"""
validator.py  —  QA_ENERGY_CERT.v1.1

Generator-relative energy cert.
Energy(s) = minimal BFS path length from reference_state under generator_set.

Supported domains
-----------------
CAPS_BE  : states (b, e)  where b,e ∈ {1..N}
           generators: sigma, mu, lambda_k, nu
CAPS_TR  : states (T, R)  where T,R ∈ {0..N}
           T = Threat level  (0=min), R = Regulation level (0=min)
           generators: fear_up, fear_down, fear_lock,
                       love_soothe, love_support, love_reframe
           optional family_tag ∈ {fear, love} per generator

Gates
-----
Gate 1 : JSON schema + domain validity + interaction_horizon lock (must be 2)
Gate 2 : Recompute energy_map  (BFS from reference_state)
Gate 3 : Recompute return_energy_map (reverse BFS to reference_state)
Gate 4 : Energy monotonicity  (1-Lipschitz + tight predecessor + ref=0)
Gate 5 : return_in_k_tests
Gate 6 : SCC recompute + power_stats + family_power_stats +
         family_interaction_stats + optional power_tests

Fail types (exhaustive)
-----------------------
SCHEMA_INVALID, DOMAIN_INVALID, RECOMPUTE_MISMATCH,
ENERGY_MONOTONICITY_VIOLATION, RETURN_IN_K_MISMATCH,
SCC_MISMATCH, POWER_STATS_MISMATCH, POWER_TESTS_VIOLATION
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── domain constants ──────────────────────────────────────────────────────────

DOMAIN_CAPS_BE = "CAPS_BE"
DOMAIN_CAPS_TR = "CAPS_TR"
SUPPORTED_DOMAINS = {DOMAIN_CAPS_BE, DOMAIN_CAPS_TR}

CAPS_BE_GEN_NAMES = {"sigma", "mu", "lambda_k", "nu"}
CAPS_TR_GEN_NAMES = {"fear_up", "fear_down", "fear_lock",
                     "love_soothe", "love_support", "love_reframe"}
VALID_FAMILY_TAGS = {"fear", "love"}

ALLOWED_FAIL_TYPES = [
    "SCHEMA_INVALID",
    "DOMAIN_INVALID",
    "RECOMPUTE_MISMATCH",
    "ENERGY_MONOTONICITY_VIOLATION",
    "RETURN_IN_K_MISMATCH",
    "SCC_MISMATCH",
    "POWER_STATS_MISMATCH",
    "POWER_TESTS_VIOLATION",
    "EPISODE_SAMPLES_MISMATCH",
]

# ── result envelope ───────────────────────────────────────────────────────────

def _ok(value: Any) -> Dict[str, Any]:
    return {"ok": True, "value": value}


def _fail(fail_type: str, invariant_diff: Dict[str, Any],
          details: Dict[str, Any]) -> Dict[str, Any]:
    assert fail_type in ALLOWED_FAIL_TYPES, f"Unknown fail_type: {fail_type}"
    return {"ok": False, "fail_type": fail_type,
            "invariant_diff": invariant_diff, "details": details}


# ── JSON / hash helpers ───────────────────────────────────────────────────────

def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_hex(obj: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(obj)).hexdigest()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _schema_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")


def _validate_schema(obj: Dict[str, Any]) -> None:
    import jsonschema
    schema = _load_json(_schema_path())
    jsonschema.validate(instance=obj, schema=schema)


# ── state types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StateBE:
    b: int
    e: int


@dataclass(frozen=True)
class StateTR:
    T: int
    R: int


State = Any  # Union[StateBE, StateTR]


def _parse_state(o: Dict[str, Any], domain: str) -> State:
    if domain == DOMAIN_CAPS_BE:
        return StateBE(b=int(o["b"]), e=int(o["e"]))
    return StateTR(T=int(o["T"]), R=int(o["R"]))


def _state_to_obj(s: State) -> Dict[str, int]:
    if isinstance(s, StateBE):
        return {"b": s.b, "e": s.e}
    return {"T": s.T, "R": s.R}


def _in_domain(s: State, N: int, domain: str) -> bool:
    if domain == DOMAIN_CAPS_BE:
        return 1 <= s.b <= N and 1 <= s.e <= N
    return 0 <= s.T <= N and 0 <= s.R <= N


def _all_states(N: int, domain: str) -> List[State]:
    if domain == DOMAIN_CAPS_BE:
        return [StateBE(b, e) for b in range(1, N + 1) for e in range(1, N + 1)]
    return [StateTR(T, R) for T in range(N + 1) for R in range(N + 1)]


# ── generator application ────────────────────────────────────────────────────

def _apply_generator(s: State, gen: Dict[str, Any], N: int,
                     domain: str) -> Optional[State]:
    name = gen["name"]
    if domain == DOMAIN_CAPS_BE:
        if name == "sigma":
            t = StateBE(s.b, s.e + 1)
        elif name == "mu":
            t = StateBE(s.e, s.b)
        elif name == "lambda_k":
            k = int(gen.get("k", 0))
            if k < 2:
                return None
            t = StateBE(s.b * k, s.e * k)
        elif name == "nu":
            if s.b % 2 == 0 and s.e % 2 == 0:
                t = StateBE(s.b // 2, s.e // 2)
            else:
                return None
        else:
            return None
        return t if _in_domain(t, N, domain) else None
    else:  # CAPS_TR
        if name == "fear_up":
            t = StateTR(s.T + 1, s.R)
        elif name == "fear_down":
            t = StateTR(s.T - 1, s.R)
        elif name == "fear_lock":
            t = StateTR(s.T, s.R - 1)
        elif name == "love_soothe":
            t = StateTR(s.T - 1, s.R + 1)
        elif name == "love_support":
            t = StateTR(s.T, s.R + 1)
        elif name == "love_reframe":
            t = StateTR(s.T - 1, s.R)
        else:
            return None
        return t if _in_domain(t, N, domain) else None


# ── graph construction ────────────────────────────────────────────────────────

def _build_adj(N: int, gens: List[Dict[str, Any]],
               domain: str) -> Dict[State, List[Tuple[str, State]]]:
    adj: Dict[State, List[Tuple[str, State]]] = {}
    for s in _all_states(N, domain):
        outs: List[Tuple[str, State]] = []
        for g in gens:
            t = _apply_generator(s, g, N, domain)
            if t is not None:
                outs.append((g["name"], t))
        adj[s] = outs
    return adj


# ── BFS ───────────────────────────────────────────────────────────────────────

def _bfs_from(ref: State,
              adj: Dict[State, List[Tuple[str, State]]]) -> Dict[State, int]:
    dist: Dict[State, int] = {ref: 0}
    q: deque = deque([ref])
    while q:
        s = q.popleft()
        for _, t in adj[s]:
            if t not in dist:
                dist[t] = dist[s] + 1
                q.append(t)
    return dist


def _reverse_bfs_from(ref: State,
                      adj: Dict[State, List[Tuple[str, State]]]) -> Dict[State, int]:
    rev: Dict[State, List[State]] = defaultdict(list)
    for s, outs in adj.items():
        for _, t in outs:
            rev[t].append(s)
    dist: Dict[State, int] = {ref: 0}
    q: deque = deque([ref])
    while q:
        cur = q.popleft()
        for prev in rev[cur]:
            if prev not in dist:
                dist[prev] = dist[cur] + 1
                q.append(prev)
    return dist


# ── SCC (Kosaraju, iterative) ─────────────────────────────────────────────────

def _scc_kosaraju(nodes: List[State],
                  adj: Dict[State, List[Tuple[str, State]]]) -> List[List[State]]:
    rev: Dict[State, List[State]] = defaultdict(list)
    for s, outs in adj.items():
        for _, t in outs:
            rev[t].append(s)

    seen: set = set()
    order: List[State] = []

    for start in nodes:
        if start in seen:
            continue
        seen.add(start)
        stack = [(start, iter(adj.get(start, [])))]
        while stack:
            node, it = stack[-1]
            try:
                _, w = next(it)
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(adj.get(w, []))))
            except StopIteration:
                order.append(node)
                stack.pop()

    seen2: set = set()
    comps: List[List[State]] = []

    for start in reversed(order):
        if start in seen2:
            continue
        seen2.add(start)
        comp: List[State] = [start]
        stack2 = [start]
        while stack2:
            node = stack2.pop()
            for w in rev[node]:
                if w not in seen2:
                    seen2.add(w)
                    comp.append(w)
                    stack2.append(w)
        comps.append(comp)

    return comps


# ── monotonicity checks ────────────────────────────────────────────────────────

def _check_monotonicity(
    dist: Dict[State, int],
    adj: Dict[State, List[Tuple[str, State]]],
    ref: State,
) -> Optional[Dict[str, Any]]:
    if dist.get(ref) != 0:
        return {"kind": "REF_NOT_ZERO", "ref_energy": dist.get(ref)}

    rev: Dict[State, List[State]] = defaultdict(list)
    for s, outs in adj.items():
        for _, t in outs:
            rev[t].append(s)

    for s, es in dist.items():
        if s == ref:
            continue
        has_tight = any(p in dist and dist[p] == es - 1 for p in rev[s])
        if not has_tight:
            return {"kind": "NO_TIGHT_PREDECESSOR",
                    "state": _state_to_obj(s), "energy": es}

    for s, outs in adj.items():
        if s not in dist:
            continue
        es = dist[s]
        for _, t in outs:
            if t not in dist:
                continue
            if dist[t] > es + 1:
                return {"kind": "EDGE_LIPSCHITZ_VIOLATION",
                        "from": _state_to_obj(s), "to": _state_to_obj(t),
                        "E_from": es, "E_to": dist[t]}
    return None


# ── power stats ───────────────────────────────────────────────────────────────

def _compute_power_stats(
    dist: Dict[State, int],
    adj: Dict[State, List[Tuple[str, State]]],
) -> List[Dict[str, Any]]:
    deltas: Dict[str, List[int]] = defaultdict(list)
    for s, outs in adj.items():
        if s not in dist:
            continue
        for gname, t in outs:
            if t not in dist:
                continue
            deltas[gname].append(dist[t] - dist[s])
    result = []
    for name in sorted(deltas):
        xs = deltas[name]
        if not xs:
            continue
        result.append({
            "name": name,
            "min_delta": min(xs),
            "max_delta": max(xs),
            "mean_delta_num": sum(xs),
            "mean_delta_den": len(xs),
        })
    return result


def _compute_family_power_stats(
    gens: List[Dict[str, Any]],
    dist: Dict[State, int],
    adj: Dict[State, List[Tuple[str, State]]],
) -> List[Dict[str, Any]]:
    tag_map: Dict[str, str] = {g["name"]: g["family_tag"]
                                for g in gens if "family_tag" in g}
    if not tag_map:
        return []
    deltas: Dict[str, List[int]] = defaultdict(list)
    for s, outs in adj.items():
        if s not in dist:
            continue
        for gname, t in outs:
            if gname not in tag_map or t not in dist:
                continue
            deltas[tag_map[gname]].append(dist[t] - dist[s])
    result = []
    for tag in sorted(deltas):
        xs = deltas[tag]
        if not xs:
            continue
        result.append({
            "family_tag": tag,
            "min_delta": min(xs),
            "max_delta": max(xs),
            "mean_delta_num": sum(xs),
            "mean_delta_den": len(xs),
        })
    return result


def _compute_family_interaction_stats(
    gens: List[Dict[str, Any]],
    dist: Dict[State, int],
    adj: Dict[State, List[Tuple[str, State]]],
) -> List[Dict[str, Any]]:
    tag_map: Dict[str, str] = {g["name"]: g["family_tag"]
                                for g in gens if "family_tag" in g}
    if not tag_map:
        return []
    # 2-step paths: s -g1-> t -g2-> u; ΔE₂ = E(u) - E(s)
    deltas: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for s, outs1 in adj.items():
        if s not in dist:
            continue
        es = dist[s]
        for g1name, t in outs1:
            f1 = tag_map.get(g1name)
            if f1 is None or t not in dist:
                continue
            for g2name, u in adj[t]:
                f2 = tag_map.get(g2name)
                if f2 is None or u not in dist:
                    continue
                deltas[(f1, f2)].append(dist[u] - es)
    all_families = sorted({f for k in deltas for f in k})
    result = []
    for f1 in all_families:
        for f2 in all_families:
            xs = deltas.get((f1, f2), [])
            if not xs:
                continue
            result.append({
                "from_family": f1,
                "to_family": f2,
                "min_delta": min(xs),
                "max_delta": max(xs),
                "mean_delta_num": sum(xs),
                "mean_delta_den": len(xs),
            })
    return result


# ── episode helpers ───────────────────────────────────────────────────────────

def _gen_delta_bounds(power_stats: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Map generator name → (min_delta, max_delta) from certified power_stats."""
    return {p["name"]: (int(p["min_delta"]), int(p["max_delta"])) for p in power_stats}


def _pair_delta_bounds(
    family_interaction_stats: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Map (from_family, to_family) → (min_delta, max_delta) from certified interaction stats."""
    return {
        (p["from_family"], p["to_family"]): (int(p["min_delta"]), int(p["max_delta"]))
        for p in family_interaction_stats
    }


# ── map list helpers ──────────────────────────────────────────────────────────

def _energy_list_to_dict(rows: List[Dict], key: str,
                         domain: str) -> Dict[State, int]:
    return {_parse_state(r["state"], domain): int(r[key]) for r in rows}


# ── main validator ────────────────────────────────────────────────────────────

def validate_cert(cert: Dict[str, Any]) -> Dict[str, Any]:
    # ── Gate 1: schema + domain + interaction_horizon ─────────────────────────
    try:
        _validate_schema(cert)
    except Exception as exc:
        return _fail("SCHEMA_INVALID",
                     {"schema_version": cert.get("schema_version")},
                     {"error": str(exc)})

    domain = cert.get("domain", "")
    if domain not in SUPPORTED_DOMAINS:
        return _fail("DOMAIN_INVALID",
                     {"domain": {"expected_one_of": sorted(SUPPORTED_DOMAINS),
                                 "got": domain}},
                     {"where": "root"})

    N = cert["N"]
    ref_obj = cert["reference_state"]
    try:
        ref = _parse_state(ref_obj, domain)
    except (KeyError, TypeError) as exc:
        return _fail("DOMAIN_INVALID",
                     {"reference_state": {"error": str(exc)}},
                     {"where": "reference_state"})

    if not _in_domain(ref, N, domain):
        return _fail("DOMAIN_INVALID",
                     {"reference_state": {"got": _state_to_obj(ref),
                                          "N": N, "domain": domain}},
                     {"why": "reference_state out of domain"})

    ih = cert.get("interaction_horizon", 2)
    if ih != 2:
        return _fail("DOMAIN_INVALID",
                     {"interaction_horizon": {"expected": 2, "got": ih}},
                     {"why": "v1.1 hard-locks interaction_horizon to 2"})

    gens = cert["generator_set"]
    valid_names = CAPS_BE_GEN_NAMES if domain == DOMAIN_CAPS_BE else CAPS_TR_GEN_NAMES
    for g in gens:
        if g["name"] not in valid_names:
            return _fail("DOMAIN_INVALID",
                         {"generator": {"name": g["name"],
                                        "valid_for_domain": domain,
                                        "valid_names": sorted(valid_names)}},
                         {"where": "generator_set"})
        if "family_tag" in g and g["family_tag"] not in VALID_FAMILY_TAGS:
            return _fail("DOMAIN_INVALID",
                         {"family_tag": {"got": g["family_tag"],
                                         "valid": sorted(VALID_FAMILY_TAGS)}},
                         {"where": f"generator {g['name']}"})

    # ── build graph ───────────────────────────────────────────────────────────
    adj = _build_adj(N, gens, domain)

    # ── Gate 2: recompute energy_map ──────────────────────────────────────────
    dist = _bfs_from(ref, adj)
    cert_energy = _energy_list_to_dict(cert["energy_map"], "energy", domain)

    if set(cert_energy.keys()) != set(dist.keys()):
        return _fail("RECOMPUTE_MISMATCH",
                     {"reachable_set": {"expected_count": len(dist),
                                        "got_count": len(cert_energy)}},
                     {"expected": sorted([_state_to_obj(s) for s in dist],
                                         key=lambda x: list(x.values())),
                      "got": sorted([_state_to_obj(s) for s in cert_energy],
                                    key=lambda x: list(x.values()))})

    for s, es in dist.items():
        if cert_energy.get(s) != es:
            return _fail("RECOMPUTE_MISMATCH",
                         {"energy": {"state": _state_to_obj(s),
                                     "expected": es,
                                     "got": cert_energy.get(s)}},
                         {"where": "energy_map"})

    # ── Gate 3: recompute return_energy_map ───────────────────────────────────
    rdist = _reverse_bfs_from(ref, adj)
    cert_return = _energy_list_to_dict(cert["return_energy_map"],
                                       "return_energy", domain)

    if set(cert_return.keys()) != set(rdist.keys()):
        return _fail("RECOMPUTE_MISMATCH",
                     {"return_reachable_set": {"expected_count": len(rdist),
                                               "got_count": len(cert_return)}},
                     {"where": "return_energy_map"})

    for s, rs in rdist.items():
        if cert_return.get(s) != rs:
            return _fail("RECOMPUTE_MISMATCH",
                         {"return_energy": {"state": _state_to_obj(s),
                                            "expected": rs,
                                            "got": cert_return.get(s)}},
                         {"where": "return_energy_map"})

    # ── Gate 4: energy monotonicity ───────────────────────────────────────────
    mono_fail = _check_monotonicity(dist, adj, ref)
    if mono_fail is not None:
        return _fail("ENERGY_MONOTONICITY_VIOLATION",
                     {"monotonicity": mono_fail},
                     {"where": "gate_4"})

    # ── Gate 5: return_in_k_tests ─────────────────────────────────────────────
    for test in cert["return_in_k_tests"]:
        s = _parse_state(test["state"], domain)
        k = int(test["k"])
        expect = bool(test["expect_can_return"])
        can = (s in rdist) and (rdist[s] <= k)
        if can != expect:
            return _fail("RETURN_IN_K_MISMATCH",
                         {"return_in_k": {"state": _state_to_obj(s), "k": k,
                                           "expected": expect, "got": can,
                                           "return_energy": rdist.get(s)}},
                         {"where": "gate_5"})

    # ── Gate 6: SCC + power stats + family stats + power_tests ───────────────
    nodes = _all_states(N, domain)
    comps = _scc_kosaraju(nodes, adj)
    scc_count = len(comps)
    max_scc_size = max(len(c) for c in comps) if comps else 0

    power = _compute_power_stats(dist, adj)
    fam_power = _compute_family_power_stats(gens, dist, adj)
    fam_interact = _compute_family_interaction_stats(gens, dist, adj)

    exp = cert["expected_summary"]

    if exp["reachable_count"] != len(dist):
        return _fail("RECOMPUTE_MISMATCH",
                     {"reachable_count": {"expected": len(dist),
                                          "got": exp["reachable_count"]}},
                     {"where": "expected_summary.reachable_count"})

    if exp["max_energy"] != (max(dist.values()) if dist else 0):
        return _fail("RECOMPUTE_MISMATCH",
                     {"max_energy": {"expected": max(dist.values()) if dist else 0,
                                     "got": exp["max_energy"]}},
                     {"where": "expected_summary.max_energy"})

    if exp["scc_count"] != scc_count or exp["max_scc_size"] != max_scc_size:
        return _fail("SCC_MISMATCH",
                     {"scc": {"expected": {"scc_count": scc_count,
                                           "max_scc_size": max_scc_size},
                               "got": {"scc_count": exp["scc_count"],
                                       "max_scc_size": exp["max_scc_size"]}}},
                     {"where": "gate_6"})

    # compare power_stats as name-keyed dict
    exp_ps = {p["name"]: p for p in exp["power_stats"]}
    got_ps = {p["name"]: p for p in power}
    if exp_ps != got_ps:
        return _fail("POWER_STATS_MISMATCH",
                     {"power_stats": {"expected": exp_ps, "got": got_ps}},
                     {"where": "gate_6.power_stats"})

    # compare family_power_stats
    exp_fps = {p["family_tag"]: p for p in exp["family_power_stats"]}
    got_fps = {p["family_tag"]: p for p in fam_power}
    if exp_fps != got_fps:
        return _fail("POWER_STATS_MISMATCH",
                     {"family_power_stats": {"expected": exp_fps, "got": got_fps}},
                     {"where": "gate_6.family_power_stats"})

    # compare family_interaction_stats
    exp_fis = {(p["from_family"], p["to_family"]): p
               for p in exp["family_interaction_stats"]}
    got_fis = {(p["from_family"], p["to_family"]): p
               for p in fam_interact}
    if exp_fis != got_fis:
        return _fail("POWER_STATS_MISMATCH",
                     {"family_interaction_stats": {"expected": exp_fis,
                                                    "got": got_fis}},
                     {"where": "gate_6.family_interaction_stats"})

    # optional power_tests
    for pt in cert.get("power_tests", []):
        gname = pt["name"]
        dc = pt["delta_constraints"]
        row = got_ps.get(gname)
        if row is None:
            # generator not in reachable edges — skip
            continue
        if "min_delta_gte" in dc and row["min_delta"] < dc["min_delta_gte"]:
            return _fail("POWER_TESTS_VIOLATION",
                         {"power_test": {"generator": gname,
                                          "constraint": "min_delta_gte",
                                          "required": dc["min_delta_gte"],
                                          "got": row["min_delta"]}},
                         {"where": "gate_6.power_tests"})
        if "max_delta_lte" in dc and row["max_delta"] > dc["max_delta_lte"]:
            return _fail("POWER_TESTS_VIOLATION",
                         {"power_test": {"generator": gname,
                                          "constraint": "max_delta_lte",
                                          "required": dc["max_delta_lte"],
                                          "got": row["max_delta"]}},
                         {"where": "gate_6.power_tests"})
        if "mean_sign" in dc:
            num = row["mean_delta_num"]
            sign = dc["mean_sign"]
            ok_sign = (
                (sign == "positive" and num > 0) or
                (sign == "negative" and num < 0) or
                (sign == "zero" and num == 0) or
                (sign == "nonnegative" and num >= 0) or
                (sign == "nonpositive" and num <= 0)
            )
            if not ok_sign:
                return _fail("POWER_TESTS_VIOLATION",
                             {"power_test": {"generator": gname,
                                              "constraint": "mean_sign",
                                              "required": sign,
                                              "got_mean_num": num}},
                             {"where": "gate_6.power_tests"})

    # ── Gate 7 (optional): episode_samples consistency ────────────────────────
    episodes = cert.get("episode_samples")
    episode_summary: Optional[Dict] = None
    if episodes is not None:
        name_to_family: Dict[str, str] = {
            g["name"]: g["family_tag"] for g in gens if "family_tag" in g
        }
        gen_bounds = _gen_delta_bounds(power)
        pair_bounds = _pair_delta_bounds(fam_interact)

        # Accumulators for episode_summary (output-only, non-cert)
        ep_checked_steps = 0
        ep_checked_pairs = 0
        ep_skipped_pairs = 0
        ep_dE_values: List[int] = []
        ep_dE2_values: List[int] = []
        fam_step: Dict[str, List[int]] = defaultdict(list)
        fam_pair: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        episode_labels: List[Dict] = []

        for ep in episodes:
            ep_id = ep.get("episode_id", "<missing>")
            steps = ep["steps"]
            states_raw = ep["states"]

            if len(states_raw) != len(steps) + 1:
                return _fail(
                    "EPISODE_SAMPLES_MISMATCH",
                    {"episode": {"episode_id": ep_id, "error": "length_mismatch",
                                  "len_states": len(states_raw),
                                  "len_steps": len(steps)}},
                    {"where": "episode_samples"},
                )

            try:
                seq = [_parse_state(s, domain) for s in states_raw]
            except Exception as exc:
                return _fail(
                    "EPISODE_SAMPLES_MISMATCH",
                    {"episode": {"episode_id": ep_id, "error": "state_parse_error"}},
                    {"exception": str(exc)},
                )

            for idx, s in enumerate(seq):
                if not _in_domain(s, N, domain):
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "state_out_of_domain",
                                      "index": idx, "state": states_raw[idx]}},
                        {"where": "episode_samples"},
                    )
                if s not in dist:
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "state_unreachable",
                                      "index": idx, "state": states_raw[idx]}},
                        {"where": "episode_samples"},
                    )

            # Episode-local accumulators for labels (safe after reachability confirmed)
            E_seq: List[int] = [dist[s] for s in seq]
            dE2_seq: List[int] = []
            pos = neg = zero = 0
            pos2 = neg2 = zero2 = 0
            startE = E_seq[0]
            endE   = E_seq[-1]
            minE   = min(E_seq)
            maxE   = max(E_seq)

            # Per-step ΔE bounds (by generator name)
            for i, gname in enumerate(steps):
                if gname not in gen_bounds:
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "unknown_generator",
                                      "step": i, "name": gname}},
                        {"where": "episode_samples"},
                    )
                dE = dist[seq[i + 1]] - dist[seq[i]]
                mn, mx = gen_bounds[gname]
                if dE < mn or dE > mx:
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "deltaE_out_of_bounds",
                                      "step": i, "generator": gname,
                                      "deltaE": dE, "bounds": [mn, mx]}},
                        {"where": "episode_samples"},
                    )
                ep_checked_steps += 1
                ep_dE_values.append(dE)
                if dE > 0:
                    pos += 1
                elif dE < 0:
                    neg += 1
                else:
                    zero += 1
                ftag = name_to_family.get(gname)
                if ftag is not None:
                    fam_step[ftag].append(dE)

            # 2-step ΔE₂ bounds (by tagged family pair)
            for i in range(len(steps) - 1):
                f1 = name_to_family.get(steps[i])
                f2 = name_to_family.get(steps[i + 1])
                if f1 is None or f2 is None:
                    ep_skipped_pairs += 1
                    continue  # untagged generator — skip interaction check
                pair = (f1, f2)
                if pair not in pair_bounds:
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "missing_family_pair_bounds",
                                      "pair": list(pair)}},
                        {"where": "episode_samples"},
                    )
                dE2 = dist[seq[i + 2]] - dist[seq[i]]
                mn2, mx2 = pair_bounds[pair]
                if dE2 < mn2 or dE2 > mx2:
                    return _fail(
                        "EPISODE_SAMPLES_MISMATCH",
                        {"episode": {"episode_id": ep_id, "error": "deltaE2_out_of_bounds",
                                      "step_pair": i, "pair": list(pair),
                                      "deltaE2": dE2, "bounds": [mn2, mx2]}},
                        {"where": "episode_samples"},
                    )
                ep_checked_pairs += 1
                ep_dE2_values.append(dE2)
                fam_pair[(f1, f2)].append(dE2)
                dE2_seq.append(dE2)
                if dE2 > 0:
                    pos2 += 1
                elif dE2 < 0:
                    neg2 += 1
                else:
                    zero2 += 1

            # ── episode_labels: stable primary_label + tags + measures ──────────
            at_ref_end = (seq[-1] == ref)
            if at_ref_end:
                primary_label = "RETURN_TO_REF"
            elif neg == 0 and pos > 0:
                primary_label = "MONOTONE_ESCALATION"
            elif pos == 0 and neg > 0:
                primary_label = "MONOTONE_RECOVERY"
            elif pos > 0 and neg > 0:
                primary_label = "OSCILLATORY"
            else:
                primary_label = "STASIS"

            tags: List[str] = []

            netE = endE - startE
            if netE > 0:
                tags.append("NET_POS")
            elif netE < 0:
                tags.append("NET_NEG")
            else:
                tags.append("NET_ZERO")

            if endE == maxE and maxE > startE:
                tags.append("PEAK_AT_END")
            elif maxE > endE and maxE > startE:
                tags.append("PEAK_BEFORE_END")
            else:
                tags.append("NO_PEAK")

            amp = maxE - minE
            if amp == 0:
                tags.append("AMP_ZERO")
            elif amp == 1:
                tags.append("AMP_ONE")
            else:
                tags.append("AMP_GE_2")

            if len(steps) >= 2:
                if pos2 > neg2:
                    tags.append("PAIR_POS")
                elif neg2 > pos2:
                    tags.append("PAIR_NEG")
                else:
                    tags.append("PAIR_TIE")

            fear_steps = sum(1 for g in steps if name_to_family.get(g) == "fear")
            love_steps = sum(1 for g in steps if name_to_family.get(g) == "love")
            labeled = fear_steps + love_steps
            if labeled == 0:
                tags.append("FAMILY_UNLABELED")
            elif fear_steps > love_steps:
                tags.append("FAMILY_FEAR_DOMINANT")
            elif love_steps > fear_steps:
                tags.append("FAMILY_LOVE_DOMINANT")
            else:
                tags.append("FAMILY_BALANCED")

            episode_labels.append({
                "episode_id": ep_id,
                "primary_label": primary_label,
                "tags": tags,
                "measures": {
                    "startE": startE, "endE": endE, "netE": netE,
                    "minE": minE, "maxE": maxE, "amp": amp,
                    "pos": pos, "neg": neg, "zero": zero,
                },
            })

        def _fam_step_entry(tag: str, vals: List[int]) -> Dict:
            return {
                "family_tag": tag,
                "count": len(vals),
                "min_delta": min(vals),
                "max_delta": max(vals),
                "mean_delta_num": sum(vals),
                "mean_delta_den": len(vals),
            }

        def _fam_pair_entry(f1: str, f2: str, vals: List[int]) -> Dict:
            return {
                "from_family": f1,
                "to_family": f2,
                "count": len(vals),
                "min_delta2": min(vals),
                "max_delta2": max(vals),
                "mean_delta2_num": sum(vals),
                "mean_delta2_den": len(vals),
            }

        episode_summary = {
            "episode_count": len(episodes),
            "checked_steps": ep_checked_steps,
            "checked_pairs": ep_checked_pairs,
            "skipped_unlabeled_pairs": ep_skipped_pairs,
            "dE_min": min(ep_dE_values) if ep_dE_values else None,
            "dE_max": max(ep_dE_values) if ep_dE_values else None,
            "dE2_min": min(ep_dE2_values) if ep_dE2_values else None,
            "dE2_max": max(ep_dE2_values) if ep_dE2_values else None,
            "observed_family_step_stats": [
                _fam_step_entry(tag, fam_step[tag])
                for tag in sorted(fam_step)
            ],
            "observed_family_pair_stats": [
                _fam_pair_entry(f1, f2, fam_pair[(f1, f2)])
                for (f1, f2) in sorted(fam_pair)
            ],
            "episode_labels": sorted(
                episode_labels,
                key=lambda x: str(x.get("episode_id", "")),
            ),
        }

    cert_sha = _sha256_hex(cert)
    result: Dict = {
        "reachable_count": len(dist),
        "max_energy": max(dist.values()) if dist else 0,
        "scc_count": scc_count,
        "max_scc_size": max_scc_size,
        "power_stats": power,
        "family_power_stats": fam_power,
        "family_interaction_stats": fam_interact,
        "cert_sha256": cert_sha,
    }
    if episode_summary is not None:
        result["episode_summary"] = episode_summary
    return _ok(result)


# ── self-test ─────────────────────────────────────────────────────────────────

_FIXTURES = [
    # (filename, should_pass, expected_fail_type)
    ("PASS_FEAR.json",        True,  None),
    ("PASS_LOVE.json",        True,  None),
    ("PASS_MIXED.json",       True,  None),
    ("FAIL_POWER.json",       False, "POWER_TESTS_VIOLATION"),
    ("FAIL_INTERACTION.json", False, "POWER_STATS_MISMATCH"),
    ("FAIL_HORIZON.json",     False, "DOMAIN_INVALID"),
    ("FAIL_EPISODE.json",     False, "EPISODE_SAMPLES_MISMATCH"),
]


def self_test(as_json: bool) -> int:
    base = os.path.dirname(os.path.abspath(__file__))
    fx = os.path.join(base, "fixtures")
    ok = True
    details = []
    for name, should_pass, expected_fail_type in _FIXTURES:
        path = os.path.join(fx, name)
        if not os.path.exists(path):
            details.append({"fixture": name, "ok": None,
                             "expected_ok": should_pass,
                             "fail_type": None, "note": "MISSING"})
            ok = False
            continue
        obj = _load_json(path)
        r = validate_cert(obj)
        passed = bool(r.get("ok"))
        if should_pass != passed:
            ok = False
        actual_fail_type = None if passed else r.get("fail_type")
        if (not should_pass) and expected_fail_type and \
                actual_fail_type != expected_fail_type:
            ok = False
        details.append({"fixture": name, "ok": passed,
                         "expected_ok": should_pass,
                         "fail_type": actual_fail_type})
    if as_json:
        print(json.dumps({"ok": ok, "fixtures": details},
                         indent=2, sort_keys=True))
    else:
        print("=== QA_ENERGY_CERT.v1.1 SELF-TEST ===")
        for d in details:
            if d.get("note") == "MISSING":
                print(f"  {d['fixture']}: MISSING (FAIL)")
                continue
            match = d["ok"] == d["expected_ok"]
            print(f"  {d['fixture']}: {'PASS' if match else 'FAIL'}"
                  + (f"  (fail_type={d['fail_type']})" if not d["ok"] else ""))
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="QA_ENERGY_CERT.v1.1 validator")
    ap.add_argument("file", nargs="?", help="Certificate JSON to validate")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(as_json=args.json)

    if not args.file:
        ap.print_help()
        return 2

    obj = _load_json(args.file)
    r = validate_cert(obj)
    if args.json:
        print(json.dumps(r, indent=2, sort_keys=True))
    else:
        if r.get("ok"):
            v = r["value"]
            print(f"PASS  reachable={v['reachable_count']}"
                  f"  max_E={v['max_energy']}"
                  f"  scc={v['scc_count']}/{v['max_scc_size']}")
        else:
            print(f"FAIL  {r['fail_type']}: {r['invariant_diff']}")
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
