"""
QA I Ching Hexagram Orbit Cert [286]

Primary sources:
  Iverson, B. (n.d.). 'Eight Keynotes.' Sympathetic Vibratory Physics articles.
    www.svpvril.com/svpweb39.html. Accessed 2026-05-30.
  Wilhelm, R. (trans. Baynes, C.F.) (1950). The I Ching or Book of Changes.
    Princeton University Press. Bollingen Series XIX. ISBN 0-691-09750-X.

Encoding: hexagram_code = lower + 8 * upper, where lower and upper are the
3-bit trigram codes (LSB=bottom line, solid=1, broken=0). Codes range 0..63.

Key algebraic fact: 8 ≡ -1 mod 9, so
  hexagram_code mod 9 ≡ (lower - upper) mod 9.

Theorem:
  (a) code = 0 (Kun-Kun, all-broken): A1-excluded.
  (b) code divisible by 9 iff lower = upper (doubled trigram, 7 hexagrams).
  (c) code divisible by 3 but not 9 iff lower ≡ upper mod 3, lower ≠ upper
      (same-class pair, 14 hexagrams) → Satellite access.
  (d) code coprime to 3 iff lower ≢ upper mod 3
      (mixed-class pair, 42 hexagrams) → Cosmos-only.

Singularity hexagrams are exactly the 7 doubled-trigram pure hexagrams
(code 9=☳☳, 18=☵☵, 27=☱☱, 36=☶☶, 45=☲☲, 54=☴☴, 63=☰☰).
"""

QA_COMPLIANCE = (
    "cert_validator — integer arithmetic on hexagram_code in {0,...,63}; "
    "orbit class by divisibility by 3 and 9 via (lower - upper) mod 9; "
    "no float feedback into QA layer"
)

import json
import os
import sys
from typing import Optional

SCHEMA_VERSION = "QA_ICHING_HEXAGRAM_ORBIT_CERT.v1"
FAMILY_ID = 286
SLUG = "qa_iching_hexagram_orbit_cert_v1"

TRIGRAM_NAMES = [
    "Kun", "Zhen", "Kan", "Dui", "Gen", "Li", "Xun", "Qian"
]
TRIGRAM_SYMBOLS = ["☷", "☳", "☵", "☱", "☶", "☲", "☴", "☰"]

# Expected orbit class for each hexagram code 0..63
def _expected_class(code: int) -> str:
    if code == 0:
        return "a1_excluded"
    if code % 9 == 0:
        return "mul_9"
    if code % 3 == 0:
        return "mul_3_not_9"
    return "coprime_to_3"


def _decode(code: int) -> tuple[int, int]:
    """Return (lower, upper) trigram codes for hexagram code."""
    upper = code // 8
    lower = code % 8
    return lower, upper


def _orbit_class(code: int) -> str:
    """Return orbit access class for hexagram code in {0,...,63}."""
    if code < 0 or code > 63:
        raise ValueError(f"hexagram_code must be in {{0,...,63}}; got {code}")
    return _expected_class(code)


REQUIRED_FIELDS = {
    "schema_version", "fixture_kind", "primary_source",
    "hexagram_code", "lower_trigram_code", "upper_trigram_code",
    "expected_class",
}


def _check_pass_fixture(data: dict) -> Optional[str]:
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if data["schema_version"] != SCHEMA_VERSION:
        return f"WRONG_SCHEMA: {data['schema_version']}"
    if data["fixture_kind"] != "pass":
        return f"WRONG_KIND: {data['fixture_kind']}"

    code = data["hexagram_code"]
    if not isinstance(code, int) or code < 0 or code > 63:
        return f"OUT_OF_RANGE: hexagram_code={code}"

    lower = data["lower_trigram_code"]
    upper = data["upper_trigram_code"]
    if not isinstance(lower, int) or lower < 0 or lower > 7:
        return f"OUT_OF_RANGE: lower_trigram_code={lower}"
    if not isinstance(upper, int) or upper < 0 or upper > 7:
        return f"OUT_OF_RANGE: upper_trigram_code={upper}"

    # Verify code = lower + 8*upper
    expected_code = lower + 8 * upper
    if code != expected_code:
        return f"CODE_MISMATCH: declared code={code} but lower={lower}+8*upper={upper} gives {expected_code}"

    actual = _orbit_class(code)
    declared = data["expected_class"]
    if actual != declared:
        return f"WRONG_CLASS: code={code} actual={actual} declared={declared}"
    return None


def _check_fail_fixture(data: dict) -> Optional[str]:
    if "expected_fail_type" not in data:
        return "FAIL_FIXTURE_MISSING_expected_fail_type"
    fail_type = data["expected_fail_type"]
    if fail_type not in ("MISSING_FIELD", "WRONG_CLASS", "OUT_OF_RANGE", "CODE_MISMATCH"):
        return f"UNKNOWN_expected_fail_type: {fail_type}"

    if fail_type == "MISSING_FIELD":
        missing = REQUIRED_FIELDS - set(data.keys())
        if not missing:
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected MISSING_FIELD but all fields present"
        return None

    if fail_type == "WRONG_CLASS":
        missing = REQUIRED_FIELDS - set(data.keys())
        if missing:
            return f"FAIL_FIXTURE_WRONG_FAIL: expected WRONG_CLASS but missing {sorted(missing)}"
        code = data.get("hexagram_code")
        if not isinstance(code, int) or code < 0 or code > 63:
            return f"FAIL_FIXTURE_WRONG_FAIL: expected WRONG_CLASS but code out of range: {code}"
        actual = _orbit_class(code)
        if actual == data.get("expected_class"):
            return "FAIL_FIXTURE_DID_NOT_FAIL: expected_class happens to be correct"
        return None

    if fail_type == "OUT_OF_RANGE":
        code = data.get("hexagram_code")
        if isinstance(code, int) and 0 <= code <= 63:
            return f"FAIL_FIXTURE_DID_NOT_FAIL: code={code} is in range"
        return None

    if fail_type == "CODE_MISMATCH":
        code = data.get("hexagram_code")
        lower = data.get("lower_trigram_code")
        upper = data.get("upper_trigram_code")
        if isinstance(lower, int) and isinstance(upper, int) and isinstance(code, int):
            if code == lower + 8 * upper:
                return "FAIL_FIXTURE_DID_NOT_FAIL: code matches lower+8*upper"
        return None

    return None


def validate_fixture(path: str) -> tuple[bool, str]:
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


def self_test() -> dict:
    errors = []

    # KOH_1: Kun-Kun (code=0) is a1_excluded
    cls0 = _orbit_class(0)
    if cls0 != "a1_excluded":
        errors.append(f"GATE KOH_1 FAILED: code=0 returned {cls0}, want a1_excluded")

    # KOH_2: Exactly 7 doubled-trigram codes have Singularity access
    sing_codes = [c for c in range(1, 64) if c % 9 == 0]
    if len(sing_codes) != 7:
        errors.append(f"GATE KOH_2 FAILED: expected 7 Singularity codes, got {len(sing_codes)}")
    for code in sing_codes:
        lower, upper = _decode(code)
        if lower != upper:
            errors.append(f"GATE KOH_2 FAILED: Singularity code={code} has lower={lower}≠upper={upper}")

    # KOH_3: Exactly 14 Satellite-access codes
    sat_codes = [c for c in range(1, 64) if c % 3 == 0 and c % 9 != 0]
    if len(sat_codes) != 14:
        errors.append(f"GATE KOH_3 FAILED: expected 14 Satellite codes, got {len(sat_codes)}")
    for code in sat_codes:
        lower, upper = _decode(code)
        if lower % 3 != upper % 3:
            errors.append(f"GATE KOH_3 FAILED: Satellite code={code} has lower%3={lower%3}≠upper%3={upper%3}")
        if lower == upper:
            errors.append(f"GATE KOH_3 FAILED: Satellite code={code} has lower=upper={lower} (should be Sing)")

    # KOH_4: Exactly 42 Cosmos-only codes
    cos_codes = [c for c in range(1, 64) if c % 3 != 0]
    if len(cos_codes) != 42:
        errors.append(f"GATE KOH_4 FAILED: expected 42 Cosmos codes, got {len(cos_codes)}")

    # KOH_5: Algebraic theorem — code mod 9 = (lower - upper) mod 9
    for upper in range(8):
        for lower in range(8):
            code = lower + 8 * upper
            lhs = code % 9
            rhs = (lower - upper) % 9
            if lhs != rhs:
                errors.append(f"GATE KOH_5 FAILED: code={code} mod9={lhs} but (lower-upper)%9={rhs}")

    # KOH_6: Partition exhaustive (1+7+14+42=64)
    total = 1 + len(sing_codes) + len(sat_codes) + len(cos_codes)
    if total != 64:
        errors.append(f"GATE KOH_6 FAILED: partition sums to {total}, want 64")

    # SRC gate
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src_path):
        errors.append("GATE SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src_path) as f:
            ref = json.load(f)
        if ref.get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
            errors.append("GATE SRC FAILED: wrong protocol_version")

    # F gate: all fixtures valid
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
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <fixture.json>  |  --self-test", file=sys.stderr)
        sys.exit(1)
    ok, msg = validate_fixture(sys.argv[1])
    print(msg)
    sys.exit(0 if ok else 1)
