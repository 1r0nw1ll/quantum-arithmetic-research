# Primary source: Wall D.D. (1960) doi:10.2307/2309169; Wildberger N.J. (2005) ISBN 978-0-9757492-0-8; Pudelko M.T. (2025) arXiv:2510.24882
QA_COMPLIANCE = (
    "cert_validator -- pure integer arithmetic (Cosmos chamber: "
    "G(b,e)=(b+e)*(b+e)+e*e with raw d=b+e (A2); sigma(b,e)=(e,((b+e-1)%m)+1) (A1); "
    "G-sums 3429/3321/3213 arithmetic progression d=-12m; G-min at (k,1) G=(k+1)^2+1; "
    "Fibonacci prefix F(5)..F(13) in O1 via integer recurrence); "
    "Theorem NT: 'G-value' is an integer; no float state, no continuous observer in QA layer"
)
"""QA Cosmos Chamber Cert [500] validator.

CLAIM (narrow): The 72 Cosmos pairs of the QA mod-9 Fibonacci shift decompose into
exactly 3 sub-orbits of length 24 that are distinguished by their G-arithmetic
signature:

  (1) Three orbits, each individually closed under QA negation.
  (2) G-sums form an arithmetic progression: d = -12*m = -108.
  (3) Orbit O_k (k=1,2,3 descending G-sum) has unique G-minimum at (k,1),
      G(k,1) = (k+1)^2 + 1.
  (4) First 5 G-values of O_1 are F(5),F(7),F(9),F(11),F(13) = 5,13,34,89,233.

Checks: CCH_1/CCH_2/CCH_3/CCH_4/CCH_5/SRC/F.
Schema: QA_COSMOS_CHAMBER_CERT.v1
"""

from __future__ import annotations
import json, os, sys

_SCHEMA = "QA_COSMOS_CHAMBER_CERT.v1"
_M = 9
_G_SUM_DIFF = 108   # = 12 * m
_EXPECTED_FIB_PREFIX = (5, 13, 34, 89, 233)   # F(5), F(7), F(9), F(11), F(13)


# ── QA arithmetic (A1-compliant) ──────────────────────────────────────────────

def qa_step(b: int, e: int, m: int = _M) -> tuple[int, int]:
    """A1: sigma(b,e) = (e, ((b+e-1) % m) + 1)."""
    return e, ((b + e - 1) % m) + 1


def qa_neg(b: int, m: int = _M) -> int:
    """QA additive inverse in {1,...,m}."""
    r = b % m
    return m - r if r != 0 else m


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


# ── Orbit enumeration ─────────────────────────────────────────────────────────

def get_orbit(b0: int, e0: int, m: int = _M) -> list[tuple[int, int]]:
    orbit = [(b0, e0)]
    b, e = qa_step(b0, e0, m)
    while (b, e) != (b0, e0):
        orbit.append((b, e))
        b, e = qa_step(b, e, m)
    return orbit


def cosmos_orbits(m: int = _M) -> list[list[tuple[int, int]]]:
    """Return the 3 Cosmos sub-orbits in descending G-sum order."""
    cosmos = {(b, e) for b in range(1, m+1) for e in range(1, m+1)
              if qa_period(b, e, m) == 24}
    raw: list[list[tuple[int, int]]] = []
    remaining = set(cosmos)
    while remaining:
        start = min(remaining)
        o = get_orbit(*start, m)
        raw.append(o)
        for p in o:
            remaining.discard(p)
    raw.sort(key=lambda o: -sum(qa_G(b, e) for b, e in o))
    return raw


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
    orbits = cosmos_orbits(m)

    # CCH_1: exactly 3 orbits of length 24
    if len(orbits) != 3 or any(len(o) != 24 for o in orbits):
        errs.append(f"CCH_1: expected 3×24, got {[len(o) for o in orbits]}")

    # CCH_2: each orbit individually negation-closed
    for k, o in enumerate(orbits, 1):
        o_set = set(o)
        for b, e in o:
            nb, ne = qa_neg(b, m), qa_neg(e, m)
            if (nb, ne) not in o_set:
                errs.append(f"CCH_2: O_{k} not negation-closed: neg({b},{e})=({nb},{ne})")
                break

    # CCH_3: G-sum arithmetic progression d = -12*m
    g_sums = tuple(sum(qa_G(b, e) for b, e in o) for o in orbits)
    decl_sums = data.get("g_sums")
    if decl_sums is not None and tuple(decl_sums) != g_sums:
        errs.append(f"CCH_3: declared g_sums {tuple(decl_sums)} != computed {g_sums}")
    diff12m = 12 * m
    if len(g_sums) >= 3:
        if g_sums[0] - g_sums[1] != diff12m:
            errs.append(f"CCH_3: O1-O2 diff={g_sums[0]-g_sums[1]}, expected {diff12m}")
        if g_sums[1] - g_sums[2] != diff12m:
            errs.append(f"CCH_3: O2-O3 diff={g_sums[1]-g_sums[2]}, expected {diff12m}")

    # CCH_4: G-minimum at canonical pair (k,1), G(k,1) = (k+1)^2 + 1
    decl_pairs = data.get("g_min_pairs")
    for k, o in enumerate(orbits, 1):
        g_vals = [(qa_G(b, e), b, e) for b, e in o]
        g_min, b_min, e_min = min(g_vals)
        expected_g = (k + 1) * (k + 1) + 1
        if decl_pairs is not None:
            dp = tuple(decl_pairs[k-1])
            if dp != (k, 1):
                errs.append(f"CCH_4: O_{k} declared g_min_pair {dp} != ({k},1)")
        if (b_min, e_min) != (k, 1) or g_min != expected_g:
            errs.append(f"CCH_4: O_{k} g_min at ({b_min},{e_min})={g_min}, expected ({k},1)={expected_g}")

    # CCH_5: first 5 G-values from (1,1) in O1 = F(5),F(7),F(9),F(11),F(13)
    if len(orbits) >= 1:
        o1 = orbits[0]
        try:
            idx = o1.index((1, 1))
        except ValueError:
            errs.append("CCH_5: (1,1) not in O1")
            idx = None
        if idx is not None:
            g_prefix = tuple(qa_G(o1[(idx + j) % 24][0], o1[(idx + j) % 24][1]) for j in range(5))
            decl_prefix = data.get("fib_prefix")
            if decl_prefix is not None and tuple(decl_prefix) != _EXPECTED_FIB_PREFIX:
                errs.append(f"CCH_5: declared fib_prefix {tuple(decl_prefix)} != {_EXPECTED_FIB_PREFIX}")
            if g_prefix != _EXPECTED_FIB_PREFIX:
                errs.append(f"CCH_5: computed prefix {g_prefix} != {_EXPECTED_FIB_PREFIX}")

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
        print("Usage: python3 qa_cosmos_chamber_cert_validate.py --self-test")
