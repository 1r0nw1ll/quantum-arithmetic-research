"""
QA I Ching Trigram Orbit Cert [285]

Primary sources:
  Iverson, B. (n.d.). 'Eight Keynotes.' Sympathetic Vibratory Physics articles.
    www.svpvril.com/svpweb39.html. Accessed 2026-05-30.
  Wilhelm, R. (trans. Baynes, C.F.) (1950). The I Ching or Book of Changes.
    Princeton University Press. Bollingen Series XIX. ISBN 0-691-09750-X.

Encoding: 3-bit integer, LSB=bottom line, solid=1, broken=0 (Fuxi/Earlier Heaven):
  0=Kun(Earth), 1=Zhen(Thunder), 2=Kan(Water), 3=Dui(Lake),
  4=Gen(Mountain), 5=Li(Fire), 6=Xun(Wind), 7=Qian(Heaven)

Claim: Kun=0 is A1-excluded; Dui=3 and Xun=6 have Satellite access;
no code has Singularity access (max=7<9); remaining 5 codes are Cosmos-only.
"""

QA_COMPLIANCE = (
    "cert_validator — integer arithmetic on trigram_code in {0,...,7}; "
    "orbit class by divisibility by 3 and 9; no float feedback into QA layer"
)

import json
import os
import sys
from typing import Optional

SCHEMA_VERSION = "QA_ICHING_TRIGRAM_ORBIT_CERT.v1"
FAMILY_ID = 285
SLUG = "qa_iching_trigram_orbit_cert_v1"

# 8 I Ching trigrams: (code, name, symbol, element)
TRIGRAMS = [
    (0, "Kun",  "☷", "Earth"),
    (1, "Zhen", "☳", "Thunder"),
    (2, "Kan",  "☵", "Water"),
    (3, "Dui",  "☱", "Lake"),
    (4, "Gen",  "☶", "Mountain"),
    (5, "Li",   "☲", "Fire"),
    (6, "Xun",  "☴", "Wind"),
    (7, "Qian", "☰", "Heaven"),
]

# Expected orbit class for each code 0..7
EXPECTED_CLASS = {
    0: "a1_excluded",   # Kun=0 excluded by A1 axiom
    1: "coprime_to_3",  # Zhen  — Cosmos only
    2: "coprime_to_3",  # Kan   — Cosmos only
    3: "mul_3_not_9",   # Dui   — Satellite access (3|3, 9∄3)
    4: "coprime_to_3",  # Gen   — Cosmos only
    5: "coprime_to_3",  # Li    — Cosmos only
    6: "mul_3_not_9",   # Xun   — Satellite access (3|6, 9∄6)
    7: "coprime_to_3",  # Qian  — Cosmos only
}


def _orbit_class(code: int) -> str:
    """Return QA orbit access class for trigram code in {0,...,7}."""
    if code < 0 or code > 7:
        raise ValueError(f"trigram_code must be in {{0,...,7}}; got {code}")
    if code == 0:
        return "a1_excluded"
    if code % 9 == 0:
        return "mul_9"
    if code % 3 == 0:
        return "mul_3_not_9"
    return "coprime_to_3"


REQUIRED_FIELDS = {"schema_version", "fixture_kind", "primary_source",
                   "trigram_code", "trigram_name", "expected_class"}


def _check_pass_fixture(data: dict) -> Optional[str]:
    """Validate a pass fixture. Returns None on PASS or an error string on FAIL."""
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if data["schema_version"] != SCHEMA_VERSION:
        return f"WRONG_SCHEMA: {data['schema_version']}"
    if data["fixture_kind"] != "pass":
        return f"WRONG_KIND: {data['fixture_kind']}"
    code = data["trigram_code"]
    if not isinstance(code, int) or code < 0 or code > 7:
        return f"OUT_OF_RANGE: trigram_code={code}"
    actual = _orbit_class(code)
    declared = data["expected_class"]
    if actual != declared:
        return f"WRONG_CLASS: code={code} actual={actual} declared={declared}"
    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    """Check that a fail fixture correctly declares and triggers its expected_fail_type."""
    if "expected_fail_type" not in data:
        return "FAIL_FIXTURE_MISSING_expected_fail_type"
    fail_type = data["expected_fail_type"]
    if fail_type not in ("MISSING_FIELD", "WRONG_CLASS", "OUT_OF_RANGE"):
        return f"UNKNOWN_expected_fail_type: {fail_type}"

    # Simulate what the pass-checker would do — the fail must trigger
    if fail_type == "MISSING_FIELD":
        missing = REQUIRED_FIELDS - set(data.keys())
        if not missing:
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected MISSING_FIELD but all fields present"
        return None  # correctly fires

    if fail_type == "WRONG_CLASS":
        # Needs required fields and a wrong expected_class
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            return f"FAIL_FIXTURE_WRONG_FAIL: expected WRONG_CLASS but missing {sorted(missing)}"
        code = data.get("trigram_code")
        if not isinstance(code, int) or code < 0 or code > 7:
            return f"FAIL_FIXTURE_WRONG_FAIL: expected WRONG_CLASS but code out of range: {code}"
        actual = _orbit_class(code)
        if actual == data.get("expected_class"):
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected_class happens to be correct"
        return None

    if fail_type == "OUT_OF_RANGE":
        code = data.get("trigram_code")
        if isinstance(code, int) and 0 <= code <= 7:
            return f"FAIL_FIXTURE_DID_NOT_FAIL: code={code} is in range"
        return None

    return None


def validate_fixture(path: str) -> tuple[bool, str]:
    """Validate a single fixture file. Returns (ok, message)."""
    with open(path) as f:
        data = json.load(f)

    kind = data.get("fixture_kind")
    if kind == "pass":
        err = _check_pass_fixture(data)
        if err:
            return False, f"FAIL (expected PASS): {err}"
        return True, "PASS"
    elif kind == "fail":
        err = _check_fail_fixture(data)
        if err:
            return False, f"FAIL (fail-fixture check): {err}"
        return True, "PASS (expected FAIL)"
    else:
        return False, f"UNKNOWN fixture_kind: {kind}"


def _gate_check(label: str, condition: bool, detail: str = "") -> Optional[str]:
    if not condition:
        return f"GATE {label} FAILED: {detail}"
    return None


def self_test() -> dict:
    """Run all gates and fixture checks. Returns JSON-serialisable result dict."""
    errors = []

    # --- Gates ---
    # KOA_1: Kun (code=0) is a1_excluded
    cls0 = _orbit_class(0)
    err = _gate_check("KOA_1", cls0 == "a1_excluded",
                      f"Kun code=0 returned {cls0}, want a1_excluded")
    if err:
        errors.append(err)

    # KOA_2: Dui=3 and Xun=6 have Satellite access (mul_3_not_9)
    for code, name in [(3, "Dui"), (6, "Xun")]:
        cls = _orbit_class(code)
        err = _gate_check("KOA_2", cls == "mul_3_not_9",
                          f"{name} code={code} returned {cls}, want mul_3_not_9")
        if err:
            errors.append(err)

    # KOA_3: No code in {0,...,7} has Singularity access (mul_9)
    # Structural: max code=7 < 9, so 9 cannot divide any code. Verify.
    mul9_codes = [c for c in range(8) if c > 0 and c % 9 == 0]
    err = _gate_check("KOA_3", len(mul9_codes) == 0,
                      f"Unexpected mul_9 codes: {mul9_codes}")
    if err:
        errors.append(err)

    # KOA_4: Exhaustive classification matches EXPECTED_CLASS for all 8 codes
    for code in range(8):
        actual = _orbit_class(code)
        expected = EXPECTED_CLASS[code]
        err = _gate_check("KOA_4", actual == expected,
                          f"code={code} actual={actual} expected={expected}")
        if err:
            errors.append(err)

    # SRC gate: mapping_protocol_ref.json present
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src_path):
        errors.append("GATE SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src_path) as f:
            ref = json.load(f)
        if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("GATE SRC FAILED: wrong protocol_version in mapping_protocol_ref.json")

    # F gate: all fail fixtures fire their expected_fail_type
    fixture_dir = os.path.join(here, "fixtures")
    pass_fixtures = []
    fail_fixtures = []
    if os.path.isdir(fixture_dir):
        for fname in sorted(os.listdir(fixture_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(fixture_dir, fname)
            with open(fpath) as f:
                data = json.load(f)
            kind = data.get("fixture_kind")
            ok, msg = validate_fixture(fpath)
            if kind == "pass":
                pass_fixtures.append(fname)
                if not ok:
                    errors.append(f"GATE F FAILED (pass fixture {fname}): {msg}")
            elif kind == "fail":
                fail_fixtures.append(fname)
                if not ok:
                    errors.append(f"GATE F FAILED (fail fixture {fname}): {msg}")
    else:
        errors.append("GATE F FAILED: fixtures/ directory missing")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "family_id": FAMILY_ID,
        "slug": SLUG,
        "schema_version": SCHEMA_VERSION,
        "pass_fixtures": len(pass_fixtures),
        "fail_fixtures": len(fail_fixtures),
        "errors": errors,
    }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        result = self_test()
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["ok"] else 1)

    # Validate a single fixture file
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <fixture.json>  |  --self-test", file=sys.stderr)
        sys.exit(1)
    ok, msg = validate_fixture(sys.argv[1])
    print(msg)
    sys.exit(0 if ok else 1)
