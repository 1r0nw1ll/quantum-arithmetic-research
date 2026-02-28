#!/usr/bin/env python3
"""
QA_GENERATOR_FAILURE_UNIFICATION_CERT.v1 — Validator

5 gates:

  Gate 1 — Schema + carrier cross-check
             Every failure_tag in generator_tagging must be in the failure_algebra_ref
             carrier (extracted from the referenced [76] fixture file).

  Gate 2 — Digest recompute
             canonical_sha256 and schema_sha256 must be correct.

  Gate 3 — T1 finite image
             Recompute F(G) = {τ(g)} and verify image_sha256.
             Verify F(G) ⊆ referenced [76] carrier.

  Gate 4 — T2 SCC + T3 path propagation
             Recompute SCC count / max_scc_size from the generator-induced graph
             on the domain (CAPS_TR or CAPS_BE).
             Spot-check each sampled_path's expected_join via the referenced [76]
             join_table.

  Gate 5 — T4 energy monotonicity
             Recompute ok_only_state_count, non_ok_required_state_count,
             min_energy_ok, min_energy_non_ok from BFS.
             Verify monotone_holds = (min_energy_non_ok > min_energy_ok).

Usage:
  python validator.py --self-test
  python validator.py --demo
  python validator.py <cert.json>
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── paths ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
_SCHEMA = _HERE / "schema.json"
_FIXTURES = _HERE / "fixtures"

# ── failure algebra fixture lookup ─────────────────────────────────────────────

# The validator needs to load the referenced [76] cert to extract carrier + join table.
# We keep a small in-memory cache of known [76] cert files by canonical_sha256.
_FA_CACHE: Dict[str, dict] = {}


def _load_fa_ref(sha256_hex: str) -> Optional[dict]:
    """Load a [76] cert from _FA_CACHE or by scanning known fixture locations."""
    if sha256_hex in _FA_CACHE:
        return _FA_CACHE[sha256_hex]
    # Scan standard locations
    candidates: List[Path] = []
    for root in [
        _HERE.parent / "qa_failure_algebra_structure_cert_v1" / "fixtures",
        _HERE.parent / "qa_alphageometry_ptolemy" / "qa_failure_algebra_structure_cert_v1" / "fixtures",
    ]:
        if root.exists():
            candidates.extend(root.glob("*.json"))
    for path in candidates:
        try:
            with open(path) as f:
                obj = json.load(f)
            h = _sha256_hex(_canonical_compact(obj))
            _FA_CACHE[h] = obj
            if h == sha256_hex:
                return obj
        except Exception:
            continue
    return None


# ── energy cert fixture lookup ─────────────────────────────────────────────────

_EC_CACHE: Dict[str, dict] = {}


def _load_ec_ref(sha256_hex: str) -> Optional[dict]:
    if sha256_hex in _EC_CACHE:
        return _EC_CACHE[sha256_hex]
    candidates: List[Path] = []
    for root in [
        _HERE.parent / "qa_energy_cert_v1_1" / "fixtures",
        _HERE.parent / "qa_alphageometry_ptolemy" / "qa_energy_cert_v1_1" / "fixtures",
    ]:
        if root.exists():
            candidates.extend(root.glob("*.json"))
    for path in candidates:
        try:
            with open(path) as f:
                obj = json.load(f)
            h = _sha256_hex(_canonical_compact(obj))
            _EC_CACHE[h] = obj
            if h == sha256_hex:
                return obj
        except Exception:
            continue
    return None


# ── hash utilities ─────────────────────────────────────────────────────────────

def _canonical_compact(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _compute_canonical_sha256(cert: dict) -> str:
    copy = json.loads(_canonical_compact(cert))
    copy["digests"]["canonical_sha256"] = "0" * 64
    return _sha256_hex(_canonical_compact(copy))


# ── domain graph ───────────────────────────────────────────────────────────────

CAPS_TR_GENS = {"fear_up", "fear_down", "fear_lock", "love_soothe", "love_support", "love_reframe"}


def _apply_caps_tr(state: Tuple[int, int], name: str, N: int) -> Optional[Tuple[int, int]]:
    T, R = state
    if name == "fear_up":      nT, nR = T + 1, R
    elif name == "fear_down":  nT, nR = T - 1, R
    elif name == "fear_lock":  nT, nR = T, R - 1
    elif name == "love_soothe": nT, nR = T - 1, R + 1
    elif name == "love_support": nT, nR = T, R + 1
    elif name == "love_reframe": nT, nR = T - 1, R
    else:
        return None
    if 0 <= nT <= N and 0 <= nR <= N:
        return (nT, nR)
    return None


def _build_graph(gen_names: List[str], N: int, domain: str) -> Dict[Tuple, List[Tuple[str, Tuple]]]:
    if domain != "CAPS_TR":
        raise ValueError(f"Unsupported domain: {domain}")
    all_states = [(T, R) for T in range(N + 1) for R in range(N + 1)]
    adj: Dict[Tuple, List] = {s: [] for s in all_states}
    for s in all_states:
        for name in gen_names:
            t = _apply_caps_tr(s, name, N)
            if t is not None:
                adj[s].append((name, t))
    return adj


def _kosaraju_scc(adj: Dict) -> Tuple[int, int]:
    """Returns (scc_count, max_scc_size)."""
    states = list(adj.keys())
    visited: Set = set()
    order: List = []

    def dfs1(start):
        stack = [(start, iter(adj[start]))]
        visited.add(start)
        while stack:
            v, it = stack[-1]
            try:
                _, u = next(it)
                if u not in visited:
                    visited.add(u)
                    stack.append((u, iter(adj[u])))
            except StopIteration:
                order.append(v)
                stack.pop()

    for s in states:
        if s not in visited:
            dfs1(s)

    radj: Dict = {s: [] for s in states}
    for s in states:
        for _, t in adj[s]:
            radj[t].append((None, s))

    comp: Dict = {}

    def dfs2(start, label):
        stack = [start]
        comp[start] = label
        while stack:
            v = stack.pop()
            for _, u in radj[v]:
                if u not in comp:
                    comp[u] = label
                    stack.append(u)

    scc_id = 0
    for s in reversed(order):
        if s not in comp:
            dfs2(s, scc_id)
            scc_id += 1

    sizes = defaultdict(int)
    for c in comp.values():
        sizes[c] += 1
    return scc_id, max(sizes.values()) if sizes else 0


def _bfs_energy(ref_state: Tuple, adj: Dict) -> Dict[Tuple, int]:
    energy: Dict[Tuple, int] = {ref_state: 0}
    q = deque([ref_state])
    while q:
        s = q.popleft()
        for _, t in adj[s]:
            if t not in energy:
                energy[t] = energy[s] + 1
                q.append(t)
    return energy


# ── Gate 4 helpers ─────────────────────────────────────────────────────────────

def _build_join_lookup(fa_cert: dict) -> Dict[Tuple[str, str], str]:
    lk: Dict[Tuple[str, str], str] = {}
    for row in fa_cert.get("join_table", []):
        a, b, j = row["a"], row["b"], row["join"]
        lk[(a, b)] = j
        lk[(b, a)] = j
    return lk


def _join_path(steps: List[str], tau: Dict[str, str], join_lk: Dict) -> str:
    result = "OK"
    for g in steps:
        tag = tau.get(g)
        if tag is None:
            return f"UNKNOWN_GENERATOR:{g}"
        key = (result, tag)
        result = join_lk.get(key) or join_lk.get((tag, result))
        if result is None:
            return f"JOIN_MISSING:{key}"
    return result


# ── Main validator ─────────────────────────────────────────────────────────────

def validate(cert: dict, cert_path: Optional[str] = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    # ── Gate 1: schema + carrier cross-check ──────────────────────────────────
    required_top = {"cert_id", "cert_version", "created_utc", "domain", "N",
                    "generator_tagging", "failure_algebra_ref", "energy_cert_ref",
                    "theorems", "digests"}
    missing = required_top - set(cert.keys())
    if missing:
        errors.append(f"Gate 1 FAIL: missing top-level keys: {sorted(missing)}")
        return False, errors

    if cert.get("cert_version") != "QA_GENERATOR_FAILURE_UNIFICATION_CERT.v1":
        errors.append(f"Gate 1 FAIL: wrong cert_version: {cert.get('cert_version')!r}")

    fa_ref = cert["failure_algebra_ref"]
    fa_sha = fa_ref.get("canonical_sha256", "")
    fa_cert = _load_fa_ref(fa_sha)
    if fa_cert is None:
        errors.append(f"Gate 1 FAIL: cannot load failure_algebra_ref (sha256={fa_sha[:12]}...); "
                      f"ensure qa_failure_algebra_structure_cert_v1/fixtures/ is present")
        return False, errors

    carrier = set(fa_cert.get("carrier", []))
    if not carrier:
        errors.append("Gate 1 FAIL: referenced [76] cert has empty carrier")
        return False, errors

    tau: Dict[str, str] = {}
    for entry in cert.get("generator_tagging", []):
        name = entry.get("name", "")
        tag = entry.get("failure_tag", "")
        if tag not in carrier:
            errors.append(f"Gate 1 FAIL: generator '{name}' has failure_tag='{tag}' not in [76] carrier {sorted(carrier)}")
        tau[name] = tag

    if errors:
        return False, errors

    # ── Gate 2: digest recompute ───────────────────────────────────────────────
    expected_canonical = _compute_canonical_sha256(cert)
    stored_canonical = cert["digests"].get("canonical_sha256", "")
    if expected_canonical != stored_canonical:
        errors.append(f"Gate 2 FAIL: canonical_sha256 mismatch\n"
                      f"  stored:   {stored_canonical}\n"
                      f"  computed: {expected_canonical}")

    schema_bytes = _SCHEMA.read_bytes() if _SCHEMA.exists() else b""
    expected_schema = _sha256_hex(schema_bytes)
    stored_schema = cert["digests"].get("schema_sha256", "")
    if schema_bytes and expected_schema != stored_schema:
        errors.append(f"Gate 2 FAIL: schema_sha256 mismatch\n"
                      f"  stored:   {stored_schema}\n"
                      f"  computed: {expected_schema}")

    if errors:
        return False, errors

    # ── Gate 3: T1 finite image ────────────────────────────────────────────────
    computed_image = sorted(set(tau.values()))
    t1 = cert["theorems"]["T1_finite_image"]
    stored_image = sorted(t1.get("image_elements", []))
    if computed_image != stored_image:
        errors.append(f"Gate 3 FAIL: T1 image_elements mismatch\n"
                      f"  cert:     {stored_image}\n"
                      f"  computed: {computed_image}")

    image_hash = _sha256_hex(_canonical_compact(computed_image))
    stored_image_hash = t1.get("image_sha256", "")
    if image_hash != stored_image_hash:
        errors.append(f"Gate 3 FAIL: T1 image_sha256 mismatch\n"
                      f"  stored:   {stored_image_hash}\n"
                      f"  computed: {image_hash}")

    # Verify F(G) ⊆ carrier
    image_set = set(computed_image)
    if not image_set.issubset(carrier):
        errors.append(f"Gate 3 FAIL: F(G) not ⊆ carrier: {image_set - carrier} not in carrier")

    if errors:
        return False, errors

    # ── Gate 4: T2 SCC + T3 path propagation ──────────────────────────────────
    domain = cert.get("domain", "")
    N = cert.get("N", 0)
    gen_names = [e["name"] for e in cert.get("generator_tagging", [])]

    try:
        adj = _build_graph(gen_names, N, domain)
        scc_count, max_scc = _kosaraju_scc(adj)
    except ValueError as e:
        errors.append(f"Gate 4 FAIL: cannot build graph: {e}")
        return False, errors

    t2 = cert["theorems"]["T2_scc_bounds"]
    if scc_count != t2.get("scc_count"):
        errors.append(f"Gate 4 FAIL: T2 scc_count: cert={t2.get('scc_count')} computed={scc_count}")
    if max_scc != t2.get("max_scc_size"):
        errors.append(f"Gate 4 FAIL: T2 max_scc_size: cert={t2.get('max_scc_size')} computed={max_scc}")

    scc_hash_input = json.dumps({"max_scc_size": max_scc, "scc_count": scc_count},
                                sort_keys=True, separators=(",", ":"))
    computed_scc_hash = _sha256_hex(scc_hash_input)
    if computed_scc_hash != t2.get("scc_hash", ""):
        errors.append(f"Gate 4 FAIL: T2 scc_hash mismatch\n"
                      f"  stored:   {t2.get('scc_hash')}\n"
                      f"  computed: {computed_scc_hash}")

    # T3 path propagation
    join_lk = _build_join_lookup(fa_cert)
    t3 = cert["theorems"]["T3_path_propagation"]
    for i, path_spec in enumerate(t3.get("sampled_paths", [])):
        steps = path_spec.get("steps", [])
        expected = path_spec.get("expected_join", "")
        # validate generator names exist in tau
        unknown = [s for s in steps if s not in tau]
        if unknown:
            errors.append(f"Gate 4 FAIL: T3 path[{i}] unknown generators: {unknown}")
            continue
        computed_join = _join_path(steps, tau, join_lk)
        if computed_join != expected:
            errors.append(f"Gate 4 FAIL: T3 path[{i}] {steps}\n"
                          f"  cert expected_join: {expected}\n"
                          f"  computed:           {computed_join}")

    if errors:
        return False, errors

    # ── Gate 5: T4 energy monotonicity ────────────────────────────────────────
    ec_sha = cert["energy_cert_ref"].get("canonical_sha256", "")
    ec_cert = _load_ec_ref(ec_sha)
    if ec_cert is None:
        errors.append(f"Gate 5 WARN: cannot load energy_cert_ref (sha256={ec_sha[:12]}...); "
                      f"skipping T4 cross-check")
    else:
        # Extract reference state from [80] cert
        ref_raw = ec_cert.get("reference_state", {})
        if domain == "CAPS_TR":
            ref_state = (ref_raw.get("T", 0), ref_raw.get("R", 0))
        else:
            errors.append(f"Gate 5 FAIL: unsupported domain for T4: {domain}")
            return False, errors

        ok_gens = {name for name, tag in tau.items() if tag == fa_cert.get("unit", "OK")}
        ok_adj = {s: [(g, t) for g, t in outs if g in ok_gens]
                  for s, outs in adj.items()}
        all_energy = _bfs_energy(ref_state, adj)
        ok_energy = _bfs_energy(ref_state, ok_adj)

        ok_count = len(ok_energy)
        non_ok_states = {s: e for s, e in all_energy.items() if s not in ok_energy}
        non_ok_count = len(non_ok_states)
        min_e_ok = min(ok_energy.values()) if ok_energy else 0
        min_e_non_ok = min(non_ok_states.values()) if non_ok_states else 0
        monotone = (min_e_non_ok > min_e_ok) if non_ok_states else True

        t4 = cert["theorems"]["T4_energy_monotonicity"]
        if ok_count != t4.get("ok_only_state_count"):
            errors.append(f"Gate 5 FAIL: T4 ok_only_state_count: cert={t4.get('ok_only_state_count')} computed={ok_count}")
        if non_ok_count != t4.get("non_ok_required_state_count"):
            errors.append(f"Gate 5 FAIL: T4 non_ok_required_state_count: cert={t4.get('non_ok_required_state_count')} computed={non_ok_count}")
        if min_e_ok != t4.get("min_energy_ok"):
            errors.append(f"Gate 5 FAIL: T4 min_energy_ok: cert={t4.get('min_energy_ok')} computed={min_e_ok}")
        if min_e_non_ok != t4.get("min_energy_non_ok"):
            errors.append(f"Gate 5 FAIL: T4 min_energy_non_ok: cert={t4.get('min_energy_non_ok')} computed={min_e_non_ok}")
        if monotone != t4.get("monotone_holds"):
            errors.append(f"Gate 5 FAIL: T4 monotone_holds: cert={t4.get('monotone_holds')} computed={monotone}")

    return (len(errors) == 0), errors


# ── Self-test ──────────────────────────────────────────────────────────────────

def self_test():
    fixtures = [
        ("valid_caps_tr_fear_love.json", True),
        ("invalid_tag_not_in_carrier.json", False),
    ]
    passed = 0
    for fname, expect_pass in fixtures:
        path = _FIXTURES / fname
        if not path.exists():
            print(f"  SKIP {fname} (not found)")
            continue
        with open(path) as f:
            cert = json.load(f)
        ok, errs = validate(cert, str(path))
        status = "PASS" if ok else "FAIL"
        expected = "PASS" if expect_pass else "FAIL"
        mark = "✓" if (ok == expect_pass) else "✗"
        print(f"  [{status}] {mark} {fname} (expected {expected})")
        if ok != expect_pass:
            for e in errs:
                print(f"       {e}")
        else:
            passed += 1
    print(f"\n  {passed}/{len(fixtures)} self-test(s) passed")
    return passed == len(fixtures)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QA_GENERATOR_FAILURE_UNIFICATION_CERT.v1 validator")
    parser.add_argument("cert", nargs="?", help="Path to cert JSON")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.self_test or args.demo:
        ok = self_test()
        sys.exit(0 if ok else 1)

    if not args.cert:
        parser.print_help()
        sys.exit(1)

    with open(args.cert) as f:
        cert = json.load(f)
    ok, errs = validate(cert, args.cert)
    if ok:
        print(f"PASS: {args.cert}")
    else:
        print(f"FAIL: {args.cert}")
        for e in errs:
            print(f"  {e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
