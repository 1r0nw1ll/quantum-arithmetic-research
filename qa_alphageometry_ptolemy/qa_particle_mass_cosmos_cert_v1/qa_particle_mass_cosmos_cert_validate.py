# Primary source: CODATA 2018 (Mohr, Newell, Taylor, Tiesinga) via NIST Physics Constants.
# DOI: 10.1103/RevModPhys.93.025010. Particle masses in electron-mass units.
# Particle Data Group (2022). Review of Particle Physics. PTEP 2022, 083C01.
# DOI: 10.1093/ptep/ptac097. FST values: Briddell, D. (2024). Field Structure Theory.
QA_COMPLIANCE = (
    "cert_validator — integer arithmetic on particle masses in electron-mass units; "
    "orbit class by divisibility structure of (d,e) pairs from F=d²-e² and C=2de "
    "decompositions; no float feedback into QA layer"
)

import json
import os
import sys
import argparse
from collections import Counter
from math import gcd

SCHEMA_VERSION = "QA_PARTICLE_MASS_COSMOS_CERT.v1"
FAMILY_ID = 290
SLUG = "qa_particle_mass_cosmos_cert_v1"

# Particle masses in electron-mass units (integer-rounded CODATA 2018 values).
PARTICLES = {
    "electron":      1,
    "muon":          207,
    "tau":           3477,
    "pion_plus":     273,
    "pion_zero":     264,
    "kaon_plus":     966,
    "kaon_zero":     974,
    "eta":           1072,
    "rho":           1517,
    "omega_meson":   1532,
    "phi_meson":     1995,
    "proton":        1836,
    "neutron":       1839,
    "lambda_b":      2183,
    "sigma_plus":    2328,
    "sigma_zero":    2334,
    "sigma_minus":   2343,
    "delta":         2411,
    "omega_baryon":  3273,
    "deuteron":      3671,
    "triton":        5497,
    "alpha":         7294,
}

# FST constellation values (Briddell 2024 Field Structure Theory).
FST_VALUES = {
    "STF_1":        3,
    "STF_2":        9,
    "STF_3":        27,
    "STF_4":        81,
    "STF_5":        243,
    "STF_6":        729,
    "STF_7_lambda": 2187,
    "STF_8":        6561,
    "proton_fst":   1836,
    "structor":     6,
    "top_cluster":  378,
    "lambda_diff":  351,
    "x2_729":       1458,
    "x2_lambda":    4374,
    "x2_proton":    3672,
    "factor_17":    17,
}

ALL_VALUES = {**PARTICLES, **FST_VALUES}


def _factor_complete(n: int) -> dict:
    factors = {}
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def _all_divisors(factors: dict) -> list:
    divs = [1]
    for p, e in factors.items():
        divs = [d * p**k for d in divs for k in range(e + 1)]
    return sorted(set(divs))


def _orbit_class(d: int, e: int, m: int = 24) -> str:
    """Divisor-shortcut orbit classifier (exact for m=24, gcd(m,5)=1).
    Singularity: b==m AND e==m. Satellite: (m//3)|b AND (m//3)|e. Cosmos: else.
    Matches canonical orbit_family period computation for m=24.
    """
    def a1(v):
        r = v % m
        return r if r else m
    bq, eq = a1(d), a1(e)
    sat_div = m // 3
    if bq == m and eq == m:
        return "singularity"
    if sat_div > 0 and bq % sat_div == 0 and eq % sat_div == 0:
        return "satellite"
    return "cosmos"


def f_triples(F: int) -> list:
    """All (d,e,is_primitive) with d>e>0 and d²-e²=F."""
    if F % 4 == 2:
        return []
    divs = _all_divisors(_factor_complete(F))
    out = []
    for b in divs:
        if b * b > F:
            break
        a = F // b
        if a * b != F or (a + b) % 2 != 0:
            continue
        d, e = (a + b) // 2, (a - b) // 2
        if e == 0:
            continue
        out.append((d, e, gcd(b, a) == 1))
    return out


def c_triples(C: int) -> list:
    """All (d,e,is_primitive) with d>e>0 and 2de=C."""
    if C % 2 != 0:
        return []
    half = C // 2
    divs = _all_divisors(_factor_complete(half))
    out = []
    for e in divs:
        if e * e >= half:
            break
        d = half // e
        if d * e != half:
            continue
        out.append((d, e, gcd(d, e) == 1))
    return out


def classify_value(mass: int) -> dict:
    ft = f_triples(mass)
    ct = c_triples(mass)
    f_orbits = Counter(_orbit_class(d, e) for d, e, _ in ft)
    c_orbits = Counter(_orbit_class(d, e) for d, e, _ in ct)
    return {
        "mass": mass,
        "f_triples": len(ft),
        "c_triples": len(ct),
        "f_orbits": dict(f_orbits),
        "c_orbits": dict(c_orbits),
        "f_all_cosmos": all(_orbit_class(d, e) == "cosmos" for d, e, _ in ft),
        "c_all_cosmos": all(_orbit_class(d, e) == "cosmos" for d, e, _ in ct),
        "mod4": mass % 4,
        "f_impossible": mass % 4 == 2,
    }


def _gate_pmc1_cosmos_monopoly() -> tuple:
    violations = []
    for label, mass in ALL_VALUES.items():
        r = classify_value(mass)
        if not r["f_all_cosmos"]:
            violations.append(f"{label}(m={mass}) F-orbits={r['f_orbits']}")
        if not r["c_all_cosmos"]:
            violations.append(f"{label}(m={mass}) C-orbits={r['c_orbits']}")
    if violations:
        return False, "COSMOS_MONOPOLY_VIOLATED: " + "; ".join(violations)
    return True, f"PMC_1 PASS: all {len(ALL_VALUES)} values Cosmos-only in F and C"


def _gate_pmc2_mod4_coverage() -> tuple:
    mod4_2 = {k: v for k, v in ALL_VALUES.items() if v % 4 == 2}
    failures = []
    for label, mass in mod4_2.items():
        if f_triples(mass):
            failures.append(f"{label}(m={mass}) has F-triples")
        if not c_triples(mass):
            failures.append(f"{label}(m={mass}) has no C-triples")
    if failures:
        return False, "MOD4_COVERAGE_FAILED: " + "; ".join(failures)
    return True, f"PMC_2 PASS: all {len(mod4_2)} ≡2 mod 4 values: 0 F-triples, ≥1 C-triple"


def _gate_pmc3_partition_completeness() -> tuple:
    empty = [
        f"{label}(m={mass})"
        for label, mass in ALL_VALUES.items()
        if mass > 1 and not f_triples(mass) and not c_triples(mass)
    ]
    if empty:
        return False, "PARTITION_INCOMPLETE: " + ", ".join(empty)
    return True, "PMC_3 PASS: every value >1 has ≥1 triple in F or C"


def _gate_pmc4_proton_c_primary() -> tuple:
    ft = f_triples(1836)
    ct = c_triples(1836)
    f_prim = [(d, e) for d, e, p in ft if p]
    c_prim = [(d, e) for d, e, p in ct if p]
    if f_prim:
        return False, f"PROTON_F_PRIMITIVE_UNEXPECTED: {f_prim}"
    if not c_prim:
        return False, "PROTON_C_PRIMARY_MISSING"
    return True, f"PMC_4 PASS: proton F-prim=0, C-prim={len(c_prim)}"


def _gate_pmc5_singularity_blockade() -> tuple:
    for label, mass in ALL_VALUES.items():
        for d, e, _ in f_triples(mass) + c_triples(mass):
            orb = _orbit_class(d, e)
            if orb in ("singularity", "satellite"):
                return False, f"BLOCKADE_BROKEN: {label}(m={mass}) (d={d},e={e}) → {orb}"
    return True, "PMC_5 PASS: singularity and satellite blockade holds across all values"


GATES = [
    ("PMC_1", _gate_pmc1_cosmos_monopoly),
    ("PMC_2", _gate_pmc2_mod4_coverage),
    ("PMC_3", _gate_pmc3_partition_completeness),
    ("PMC_4", _gate_pmc4_proton_c_primary),
    ("PMC_5", _gate_pmc5_singularity_blockade),
]


def _gate_src(fixture: dict) -> tuple:
    src = fixture.get("primary_source", "")
    if not src or len(src) < 10:
        return False, "MISSING_PRIMARY_SOURCE"
    return True, "SRC PASS"


def _gate_f(fixture: dict) -> tuple:
    kind = fixture.get("fixture_kind")
    mass = fixture.get("mass")
    coord = fixture.get("coord")
    expected_class = fixture.get("expected_class")
    expected_fail_type = fixture.get("expected_fail_type")

    if kind not in ("pass", "fail"):
        return False, f"INVALID_FIXTURE_KIND: {kind}"

    if kind == "fail":
        if expected_fail_type == "MISSING_FIELD":
            if mass is None or coord is None or expected_class is None:
                return True, "F PASS (MISSING_FIELD confirmed)"
            return False, "MISSING_FIELD fixture has all fields present"
        if expected_fail_type == "WRONG_CLASS":
            if mass is None:
                return False, "WRONG_CLASS fixture missing mass"
            triples = f_triples(mass) if coord == "F" else c_triples(mass)
            actual = set(_orbit_class(d, e) for d, e, _ in triples)
            if expected_class not in actual:
                return True, f"F PASS (WRONG_CLASS confirmed: actual={actual})"
            return False, f"WRONG_CLASS fixture: {expected_class} IS in actual={actual}"
        if expected_fail_type == "IMPOSSIBLE_COORD":
            if mass is None:
                return False, "IMPOSSIBLE_COORD fixture missing mass"
            if coord == "F" and mass % 4 == 2:
                return True, "F PASS (IMPOSSIBLE_COORD confirmed)"
            return False, f"IMPOSSIBLE_COORD fixture not actually impossible: m={mass} coord={coord}"
        return False, f"UNKNOWN_FAIL_TYPE: {expected_fail_type}"

    # pass fixture
    if mass is None:
        return False, "MISSING_FIELD: mass"
    if coord not in ("F", "C"):
        return False, f"INVALID_COORD: {coord}"
    if expected_class != "cosmos":
        return False, f"EXPECTED_CLASS must be cosmos, got {expected_class}"
    triples = f_triples(mass) if coord == "F" else c_triples(mass)
    if not triples:
        return False, f"NO_TRIPLES: mass={mass} coord={coord}"
    actual = set(_orbit_class(d, e) for d, e, _ in triples)
    if actual != {"cosmos"}:
        return False, f"WRONG_ORBIT: mass={mass} coord={coord} actual={actual}"
    return True, f"F PASS: mass={mass} coord={coord} cosmos ({len(triples)} triples)"


def validate_fixture(path: str) -> dict:
    with open(path) as f:
        fixture = json.load(f)
    sv = fixture.get("schema_version", "")
    if sv != SCHEMA_VERSION:
        return {"ok": False, "path": path, "error": f"SCHEMA_MISMATCH: {sv!r}"}
    src_ok, src_msg = _gate_src(fixture)
    if not src_ok:
        return {"ok": False, "path": path, "error": src_msg}
    f_ok, f_msg = _gate_f(fixture)
    return {"ok": f_ok, "path": path, "error": None if f_ok else f_msg}


def run_self_test(base_dir: str) -> dict:
    errors = []
    for gate_name, gate_fn in GATES:
        ok, msg = gate_fn()
        if not ok:
            errors.append(f"GATE {gate_name} FAILED: {msg}")
    fixture_dir = os.path.join(base_dir, "fixtures")
    pass_count = fail_count = 0
    if os.path.isdir(fixture_dir):
        for fname in sorted(os.listdir(fixture_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(fixture_dir, fname)
            result = validate_fixture(fpath)
            if fname.startswith("pass_"):
                pass_count += 1
                if not result["ok"]:
                    errors.append(f"GATE F FAILED (pass fixture {fname}): FAIL (expected PASS): {result['error']}")
            elif fname.startswith("fail_"):
                fail_count += 1
                if not result["ok"]:
                    errors.append(f"GATE F FAILED (fail fixture {fname}): FAIL (expected PASS): {result['error']}")
    return {
        "ok": len(errors) == 0,
        "family_id": FAMILY_ID,
        "slug": SLUG,
        "schema_version": SCHEMA_VERSION,
        "pass_fixtures": pass_count,
        "fail_fixtures": fail_count,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("fixture", nargs="?")
    args = parser.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    if args.self_test:
        result = run_self_test(base)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
    if args.fixture:
        result = validate_fixture(args.fixture)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)
    print(json.dumps({"error": "provide --self-test or a fixture path"}))
    sys.exit(1)
