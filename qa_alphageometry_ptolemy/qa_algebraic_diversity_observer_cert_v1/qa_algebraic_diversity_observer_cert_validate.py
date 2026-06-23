# Primary source: Thornton M.A. (2026) arXiv:2604.03634; Thornton M.A. (2026) arXiv:2604.19983; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8
"""QA Algebraic Diversity Observer Cert [501] validator.

CLAIM (narrow): The G-function G(b,e)=(b+e)^2+e^2 over QA mod-9 orbits defines
algebraically diverse observer channels in the sense of Thornton's Algebraic
Diversity framework (arXiv:2604.03634, arXiv:2604.19983):

  (AD_1) G is injective on each Cosmos orbit: 24 pairwise-distinct G-values per
         orbit. Integer form of "Z/24Z is the minimal matched group for Cosmos
         G-signals" (Thornton Replacement Theorem, arXiv:2604.03634).

  (AD_2) G is injective on the Satellite orbit: 8 pairwise-distinct G-values.
         Integer form of "Z/8Z is the minimal matched group for Satellite
         G-signals."

  (AD_3) No proper divisor k|24, k<24 yields G-period-k repetition in any Cosmos
         orbit; no proper divisor k|8, k<8 in the Satellite. Z/24Z and Z/8Z are
         the MINIMAL matched groups (Thornton blind matching, arXiv:2604.19983).

  (AD_4) The Satellite G-set {45,90,117,153,180,225,261,306} is disjoint from all
         Cosmos G-sets. Satellite and Cosmos are algebraically diverse,
         non-overlapping observer channels.

Checks: AD_1/AD_2/AD_3/AD_4/SRC/F.
Schema: QA_ALGEBRAIC_DIVERSITY_CERT.v1
"""

from __future__ import annotations
import json, os, sys

_SCHEMA = "QA_ALGEBRAIC_DIVERSITY_CERT.v1"
_M = 9
_COSMOS_PERIOD = 24
_SATELLITE_PERIOD = 8
_EXPECTED_SAT_G_SET = frozenset({45, 90, 117, 153, 180, 225, 261, 306})


# ── QA arithmetic (A1-compliant) ──────────────────────────────────────────────

def qa_step(b: int, e: int, m: int = _M) -> tuple[int, int]:
    """A1: sigma(b,e) = (e, ((b+e-1) % m) + 1)."""
    return e, ((b + e - 1) % m) + 1


def qa_period(b: int, e: int, m: int = _M) -> int:
    b0, e0 = b, e
    for k in range(1, m * m * m + 2):
        b, e = qa_step(b, e, m)
        if b == b0 and e == e0:
            return k
    raise RuntimeError(f"period not found for ({b0},{e0}) mod {m}")


def qa_G(b: int, e: int) -> int:
    """G(b,e) = (b+e)^2 + e^2  (raw d=b+e; A2-compliant, no **2)."""
    d = b + e
    return d * d + e * e


def get_orbit(b0: int, e0: int, m: int = _M) -> list[tuple[int, int]]:
    orbit = [(b0, e0)]
    b, e = qa_step(b0, e0, m)
    while (b, e) != (b0, e0):
        orbit.append((b, e))
        b, e = qa_step(b, e, m)
    return orbit


def cosmos_orbits(m: int = _M) -> list[list[tuple[int, int]]]:
    """Three Cosmos orbits in descending G-sum order."""
    cosmos = {(b, e) for b in range(1, m+1) for e in range(1, m+1)
              if qa_period(b, e, m) == _COSMOS_PERIOD}
    raw: list[list[tuple[int, int]]] = []
    remaining = set(cosmos)
    while remaining:
        start = min(remaining)
        o = get_orbit(*start, m)
        raw.append(o)
        for p in o: remaining.discard(p)
    raw.sort(key=lambda o: -sum(qa_G(b, e) for b, e in o))
    return raw


def satellite_orbit(m: int = _M) -> list[tuple[int, int]]:
    sat = {(b, e) for b in range(1, m+1) for e in range(1, m+1)
           if qa_period(b, e, m) == _SATELLITE_PERIOD}
    start = min(sat)
    return get_orbit(*start, m)


# ── Cert checks ───────────────────────────────────────────────────────────────

def _check_fixture(data: dict) -> list[str]:
    errs: list[str] = []

    # SRC
    if data.get("schema_version") != _SCHEMA:
        errs.append(f"SRC: schema_version != {_SCHEMA!r}")

    # F
    fl = data.get("fail_ledger")
    if fl is not None and not isinstance(fl, list):
        errs.append("F: fail_ledger must be a list")

    m = data.get("modulus", _M)
    c_orbits = cosmos_orbits(m)
    s_orbit = satellite_orbit(m)

    # AD_1: G injective on each Cosmos orbit
    for k, o in enumerate(c_orbits, 1):
        g_vals = [qa_G(b, e) for b, e in o]
        if len(set(g_vals)) != len(g_vals):
            errs.append(f"AD_1: G not injective on Cosmos O_{k}: duplicate G-values")
        decl_key = f"cosmos_g_injective_o{k}"
        if decl_key in data and data[decl_key] is not True:
            errs.append(f"AD_1: declared {decl_key}={data[decl_key]!r} but computed True")

    # AD_2: G injective on Satellite orbit
    g_sat = [qa_G(b, e) for b, e in s_orbit]
    if len(set(g_sat)) != len(g_sat):
        errs.append("AD_2: G not injective on Satellite orbit: duplicate G-values")
    if "satellite_g_injective" in data and data["satellite_g_injective"] is not True:
        errs.append(f"AD_2: declared satellite_g_injective={data['satellite_g_injective']!r} but computed True")

    # AD_3: no sub-period in Cosmos or Satellite
    cosmos_proper_divs = [k for k in range(1, _COSMOS_PERIOD) if _COSMOS_PERIOD % k == 0]
    sat_proper_divs = [k for k in range(1, _SATELLITE_PERIOD) if _SATELLITE_PERIOD % k == 0]

    for ki, o in enumerate(c_orbits, 1):
        g_vals = [qa_G(b, e) for b, e in o]
        n = len(g_vals)
        for k in cosmos_proper_divs:
            if all(g_vals[j] == g_vals[(j + k) % n] for j in range(n)):
                errs.append(f"AD_3: Cosmos O_{ki} has G-period {k} (divisor of 24) — Z/{k}Z suffices, Z/24Z NOT minimal")

    g_s_list = [qa_G(b, e) for b, e in s_orbit]
    ns = len(g_s_list)
    for k in sat_proper_divs:
        if all(g_s_list[j] == g_s_list[(j + k) % ns] for j in range(ns)):
            errs.append(f"AD_3: Satellite has G-period {k} (divisor of 8) — Z/{k}Z suffices, Z/8Z NOT minimal")

    # AD_4: Satellite G-set disjoint from all Cosmos G-sets
    sat_g_set = frozenset(qa_G(b, e) for b, e in s_orbit)
    decl_sat = data.get("satellite_g_set")
    if decl_sat is not None and frozenset(decl_sat) != sat_g_set:
        errs.append(f"AD_4: declared satellite_g_set {sorted(decl_sat)} != computed {sorted(sat_g_set)}")
    for ki, o in enumerate(c_orbits, 1):
        cosmos_g_set = frozenset(qa_G(b, e) for b, e in o)
        overlap = sat_g_set & cosmos_g_set
        if overlap:
            errs.append(f"AD_4: Satellite G-set overlaps Cosmos O_{ki}: {sorted(overlap)}")

    return errs


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test() -> None:
    base = os.path.dirname(__file__)
    fix_dir = os.path.join(base, "fixtures")
    results: dict[str, object] = {"ok": True, "details": []}

    for fname in sorted(os.listdir(fix_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(fix_dir, fname)
        with open(path) as f:
            data = json.load(f)
        errs = _check_fixture(data)
        expect_pass = fname.startswith("pass_")
        expect_fail = fname.startswith("fail_")
        passed = len(errs) == 0

        if expect_pass and not passed:
            results["ok"] = False  # type: ignore[assignment]
            results["details"].append({"file": fname, "verdict": "UNEXPECTED_FAIL", "errors": errs})  # type: ignore[union-attr]
        elif expect_fail and passed:
            results["ok"] = False  # type: ignore[assignment]
            results["details"].append({"file": fname, "verdict": "UNEXPECTED_PASS"})  # type: ignore[union-attr]
        else:
            verdict = "PASS" if expect_pass else "FAIL_AS_EXPECTED"
            results["details"].append({"file": fname, "verdict": verdict})  # type: ignore[union-attr]

    print(json.dumps(results))


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        print("Usage: python3 qa_algebraic_diversity_observer_cert_validate.py --self-test")
