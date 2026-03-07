#!/usr/bin/env python3
"""QA E8 Alignment Audit Cert v1 — family [100]

Full-population audit of QA orbit state alignment with the E8 root system
under the fixed canonical embedding (b,e,d,a,b,e,d,a)/norm.

Pre-registered decision rule (applied before inspecting results):
  STRUCTURAL if ALL of:
    median_max_cosine > 0.85
    mean_within_orbit_std_12 < 0.05
    mean_orbit_persistence_12 > 0.5
    gap_vs_random > 0.10
  INCIDENTAL otherwise.

Honest result from the mod-9 full population (81 states):
  median_max_cosine = 0.9113  (passes 0.85 threshold)
  mean_within_orbit_std_12 = 0.0426  (passes < 0.05)
  mean_orbit_persistence_12 = 0.833  (passes > 0.5)
  gap_vs_random = -0.019  (FAILS > 0.10 — random baseline exceeds QA)
  → VERDICT: INCIDENTAL (projection artifact of the (v,v)/norm embedding)

Gates:
  A — Enumerate all orbit states from modulus; verify total_states, orbit_count.
  B — Compute 8D embeddings, cosine to all 240 E8 roots; verify median/mean/std
      and 12-cycle statistics.
  C — Apply pre-registered decision rule; verify claimed verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema

TOL  = 1e-6   # coarser tolerance — floating-point median across 81 values
EPS  = 1e-12

FAIL_SCHEMA        = "SCHEMA_INVALID"
FAIL_STATE_COUNT   = "STATE_COUNT_MISMATCH"
FAIL_STATS_MISMATCH = "STATS_MISMATCH"
FAIL_VERDICT       = "VERDICT_MISMATCH"

# ── Pre-registered decision rule (frozen) ────────────────────────────────────
THRESHOLD_MEDIAN  = 0.85
THRESHOLD_STD     = 0.05
THRESHOLD_PERSIST = 0.50
THRESHOLD_GAP     = 0.10


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


# ── E8 roots ─────────────────────────────────────────────────────────────────

def _build_e8_normalized() -> List[Tuple[float, ...]]:
    roots = []
    for i, j in itertools.combinations(range(8), 2):
        for si, sj in itertools.product([-1, 1], repeat=2):
            r = [0.0] * 8; r[i] = si; r[j] = sj
            roots.append(tuple(r))
    for signs in itertools.product([-1, 1], repeat=8):
        if signs.count(-1) % 2 == 0:
            roots.append(tuple(s * 0.5 for s in signs))
    assert len(roots) == 240
    def _vn(v: tuple) -> float:
        return math.sqrt(sum(x * x for x in v))
    return [tuple(x / _vn(r) for x in r) for r in roots]


_E8N: List[Tuple[float, ...]] = _build_e8_normalized()


# ── Canonical embedding ───────────────────────────────────────────────────────

def _embed(b: int, e: int, d: int, a: int) -> Tuple[float, ...]:
    """Fixed map: (b,e,d,a,b,e,d,a) / ||·||."""
    v = (float(b), float(e), float(d), float(a),
         float(b), float(e), float(d), float(a))
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _max_cosine(v8: Tuple[float, ...]) -> float:
    return max(abs(sum(v8[k] * r[k] for k in range(8))) for r in _E8N)


def _argmax_root(v8: Tuple[float, ...]) -> int:
    best_i, best_v = 0, -1.0
    for i, r in enumerate(_E8N):
        val = abs(sum(v8[k] * r[k] for k in range(8)))
        if val > best_v:
            best_v, best_i = val, i
    return best_i


# ── QA orbit enumeration ─────────────────────────────────────────────────────

def _qa_step(b: int, e: int, mod: int) -> Tuple[int, int]:
    d = (b + e) % mod or mod
    a = (b + 2 * e) % mod or mod
    return d, a


def _enumerate_orbits(mod: int) -> List[List[Tuple[int, int, int, int]]]:
    seen: set = set()
    orbits: List[List[Tuple[int, int, int, int]]] = []
    for b0 in range(1, mod + 1):
        for e0 in range(1, mod + 1):
            if (b0, e0) in seen:
                continue
            orbit: List[Tuple[int, int, int, int]] = []
            b, e = b0, e0
            vis: set = set()
            while (b, e) not in vis:
                vis.add((b, e))
                d, a = _qa_step(b, e, mod)
                orbit.append((b, e, d, a))
                b, e = d, a
            orbits.append(orbit)
            for s in orbit:
                seen.add(s[:2])
    return orbits


# ── Compute full audit statistics ─────────────────────────────────────────────

def _compute_audit(mod: int, rng_seed: int, n_random: int) -> Dict[str, Any]:
    orbits = _enumerate_orbits(mod)
    total_states = sum(len(o) for o in orbits)
    orbit_count  = len(orbits)

    all_mcs: List[float] = []
    len12_stds: List[float] = []
    len12_persists: List[float] = []

    for orb in orbits:
        L   = len(orb)
        mcs = [_max_cosine(_embed(*s)) for s in orb]
        all_mcs.extend(mcs)
        if L == 12:
            len12_stds.append(statistics.stdev(mcs))
            roots = [_argmax_root(_embed(*s)) for s in orb]
            persist = sum(1 for i in range(L) if roots[i] == roots[(i + 1) % L]) / L
            len12_persists.append(persist)

    median_mc = statistics.median(all_mcs)
    mean_mc   = sum(all_mcs) / len(all_mcs)
    std_mc    = statistics.stdev(all_mcs)

    mean_std_12     = sum(len12_stds) / len(len12_stds) if len12_stds else 0.0
    mean_persist_12 = sum(len12_persists) / len(len12_persists) if len12_persists else 0.0

    rng = random.Random(rng_seed)
    rand_mcs = [
        _max_cosine(_embed(*[rng.uniform(1, mod) for _ in range(4)]))
        for _ in range(n_random)
    ]
    rand_median = statistics.median(rand_mcs)
    gap         = median_mc - rand_median

    verdict = "STRUCTURAL" if (
        median_mc > THRESHOLD_MEDIAN
        and mean_std_12 < THRESHOLD_STD
        and mean_persist_12 > THRESHOLD_PERSIST
        and gap > THRESHOLD_GAP
    ) else "INCIDENTAL"

    return {
        "total_states":              total_states,
        "orbit_count":               orbit_count,
        "median_max_cosine":         median_mc,
        "mean_max_cosine":           mean_mc,
        "std_max_cosine":            std_mc,
        "mean_within_orbit_std_12":  mean_std_12,
        "mean_orbit_persistence_12": mean_persist_12,
        "random_baseline_median":    rand_median,
        "gap_vs_random":             gap,
        "verdict":                   verdict,
    }


# ── Gates ─────────────────────────────────────────────────────────────────────

def gate_schema(cert: Dict[str, Any], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        jsonschema.Draft202012Validator(schema).validate(cert)
    except jsonschema.ValidationError as exc:
        return {"ok": False, "fail_type": FAIL_SCHEMA,
                "invariant_diff": {}, "details": {"error": str(exc)}, "witnesses": []}
    return None


def gate_a_state_count(cert: Dict[str, Any], audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for field in ("total_states", "orbit_count"):
        claimed = int(cert["claimed"][field])
        actual  = int(audit[field])
        if claimed != actual:
            return {"ok": False, "fail_type": FAIL_STATE_COUNT,
                    "invariant_diff": {f"claimed.{field}": {"claimed": claimed, "recomputed": actual}},
                    "details": {}, "witnesses": []}
    return None


def gate_b_stats(cert: Dict[str, Any], audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fields = [
        "median_max_cosine", "mean_max_cosine", "std_max_cosine",
        "mean_within_orbit_std_12", "mean_orbit_persistence_12",
        "random_baseline_median", "gap_vs_random",
    ]
    diffs = {}
    for f in fields:
        claimed = float(cert["claimed"][f])
        actual  = float(audit[f])
        if not _close(claimed, actual):
            diffs[f"claimed.{f}"] = {"claimed": claimed, "recomputed": actual}
    if diffs:
        return {"ok": False, "fail_type": FAIL_STATS_MISMATCH,
                "invariant_diff": diffs, "details": {}, "witnesses": []}
    return None


def gate_c_verdict(cert: Dict[str, Any], audit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    claimed = cert["claimed"]["verdict"]
    actual  = audit["verdict"]
    if claimed != actual:
        return {"ok": False, "fail_type": FAIL_VERDICT,
                "invariant_diff": {"claimed.verdict": {"claimed": claimed, "recomputed": actual,
                                                        "rule": "STRUCTURAL iff gap>0.10 AND median>0.85 AND std12<0.05 AND persist>0.50"}},
                "details": {}, "witnesses": []}
    return None


# ── Top-level ─────────────────────────────────────────────────────────────────

def validate_cert(cert: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    err = gate_schema(cert, schema)
    if err is not None:
        return err

    p = cert["audit_params"]
    audit = _compute_audit(
        mod=int(p["modulus"]),
        rng_seed=int(p["random_seed"]),
        n_random=int(p["random_baseline_n"]),
    )

    err = gate_a_state_count(cert, audit)
    if err is not None:
        return err

    err = gate_b_stats(cert, audit)
    if err is not None:
        return err

    err = gate_c_verdict(cert, audit)
    if err is not None:
        return err

    return {
        "ok": True,
        "family": 100,
        "cert_sha256": sha256_hex(canonical_json_bytes(cert)),
        "recomputed": audit,
        "witnesses": [],
    }


# ── Self-test ──────────────────────────────────────────────────────────────────

def run_self_test() -> int:
    base   = Path(__file__).resolve().parent
    schema = load_json(base / "schema.json")

    fixtures: List[Tuple[str, bool, Optional[str]]] = [
        ("pass_mod9_incidental.json", True,  None),
        ("fail_wrong_verdict.json",   False, FAIL_VERDICT),
        ("fail_stats_mismatch.json",  False, FAIL_STATS_MISMATCH),
        ("fail_schema.json",          False, FAIL_SCHEMA),
    ]

    failed: List[str] = []
    results: List[Dict[str, Any]] = []

    for name, expected_ok, expected_fail in fixtures:
        cert    = load_json(base / "fixtures" / name)
        out     = validate_cert(cert, schema)
        got_ok  = bool(out.get("ok"))
        got_fail = out.get("fail_type")
        ok_match   = got_ok == expected_ok
        fail_match = expected_fail is None or got_fail == expected_fail
        if not ok_match or not fail_match:
            failed.append(name)
        results.append({
            "fixture": name, "expected_ok": expected_ok,
            "expected_fail_type": expected_fail,
            "got_ok": got_ok, "got_fail_type": got_fail,
        })

    payload = {"ok": len(failed) == 0, "failed_fixtures": failed, "fixtures": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA E8 Alignment Audit Cert v1")
    parser.add_argument("--schema", default="schema.json")
    parser.add_argument("--cert")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test())

    if not args.cert:
        parser.error("provide --cert or --self-test")

    schema = load_json(Path(args.schema))
    cert   = load_json(Path(args.cert))
    print(json.dumps(validate_cert(cert, schema), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
