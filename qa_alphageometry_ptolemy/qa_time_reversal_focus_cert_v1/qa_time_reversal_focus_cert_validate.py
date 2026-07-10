"""
QA Time-Reversal Focusing Cert [522]

Primary sources:
  (Fink, 1992) "Time reversal of ultrasonic fields." IEEE Trans. UFFC
    39(5):555-566. DOI 10.1109/58.156174
  (Prada & Fink, 1994) "Eigenmodes of the time reversal operator." Wave Motion
    20:151-163. DOI 10.1016/0165-2125(94)90039-6
  (Yariv, 1978) IEEE J. Quantum Electron. 14(9):650-660. DOI 10.1109/JQE.1978.1069870
  (Soffer, 1986) Opt. Lett. 11(2):118-120. DOI 10.1364/OL.11.000118

CLAIM: cert [518]'s distortion-correction operator -- qa_neg, the standard
involution (adjugate) = phase conjugation = TIME REVERSAL -- run IN REVERSE
FOCUSES energy back onto a source through a scattering medium (Fink's time-reversal
mirror), rather than correcting distortion at a receiver. A source at x0 propagates
(per-element phase G(i,x0)) and scatters (medium H_i); the array records
r_i = qa_add(G(i,x0), H_i, s); TIME-REVERSE each element (r_i* = qa_neg(r_i)) and
re-emit toward x -> field_i(x) = qa_add(G(i,x), H_i, r_i*). Because the medium term
appears on record and re-emit it CANCELS: field_i(x) = qa_add(G(i,x)-G(i,x0)-s). At
x=x0 every element carries the same phase qa_neg(s) -> coherent focal peak, THROUGH
the medium. Elsewhere the phases scatter. SAME-MEDIUM SPECIFICITY (the [518]
fingerprint, for focusing): re-emit through a different screen H' and the field at
the source is no longer constant -> the focus is destroyed.

Checks (deterministic, integer-only, pure stdlib):
  FOCUS_CONSTANT    re-emit through matched medium: field at the source is constant
                    across all elements (= qa_neg(s)) -> maximally coherent focus
  MEDIUM_CANCEL     that constant equals qa_neg(s) exactly, independent of H
  OFF_SOURCE_SCATTER field at an off-source location is NOT constant (scatters)
  SPECIFICITY       re-emit through a mismatched medium H': field at the source is
                    NOT constant -> focus only through the matched medium
  A1_RANGE          every phase in {1,...,m}
  SRC / F           mapping ref present; fixtures behave as declared

Builds on certs [518] (four-wave-mixing conjugator), [519]; companion to the
phase-conjugation cluster [520],[521].
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

QA_COMPLIANCE = (
    "cert_validator -- integer phase arithmetic on {1,...,m} (identity=m, never 0); "
    "the coherent-sum focus magnitude is a float observer-layer readout used only in "
    "the reference impl, never QA state (Theorem NT). Geometry/medium synthesis is "
    "observer-layer input."
)

SCHEMA_VERSION = "QA_TIME_REVERSAL_FOCUS_CERT.v1"
FAMILY_ID = 522
SLUG = "qa_time_reversal_focus_cert_v1"

MECH_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
               "m", "source_phase", "prop_to_source", "prop_to_off",
               "medium", "medium_alt"}
EMP_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
              "peak_focus", "focal_gain", "specificity_gap", "source_recovered"}


# ---------------------------------------------------------------------------
# QA phase algebra + time-reversal focusing (pure stdlib)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(m: int, *xs: int) -> int:
    return qa_mod(sum(int(x) for x in xs), m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-int(a), m)


def record(prop_src: List[int], medium: List[int], s: int, m: int) -> List[int]:
    """Array records the source through propagation + scattering medium."""
    return [qa_add(m, g, h, s) for g, h in zip(prop_src, medium)]


def reemit(prop: List[int], medium: List[int], tr: List[int], m: int) -> List[int]:
    """Re-emit the time-reversed record through a medium toward a location."""
    return [qa_add(m, g, h, t) for g, h, t in zip(prop, medium, tr)]


def _in_range(seq, m):
    return all(isinstance(v, int) and 1 <= v <= m for v in seq)


# ---------------------------------------------------------------------------
# Fixture checks
# ---------------------------------------------------------------------------
def _check_mechanism(data) -> Optional[str]:
    missing = MECH_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    m = data["m"]
    if not (isinstance(m, int) and m >= 2):
        return f"OUT_OF_RANGE: m={m}"
    s = data["source_phase"]
    g_src, g_off = data["prop_to_source"], data["prop_to_off"]
    H, Hp = data["medium"], data["medium_alt"]
    n = len(g_src)
    if not (len(g_off) == len(H) == len(Hp) == n and n >= 2):
        return "OUT_OF_RANGE: array length mismatch or n<2"
    for seq in (g_src, g_off, H, Hp):
        if not _in_range(seq, m):
            return f"OUT_OF_RANGE: value not in 1..{m}"
    if not (isinstance(s, int) and 1 <= s <= m):
        return f"OUT_OF_RANGE: source_phase={s}"

    tr = [qa_neg(r, m) for r in record(g_src, H, s, m)]

    # FOCUS_CONSTANT + MEDIUM_CANCEL: field at the source through matched medium is
    # constant across elements and equals qa_neg(s), independent of the medium H.
    field_src = reemit(g_src, H, tr, m)
    if len(set(field_src)) != 1:
        return "WRONG_MECHANISM: field at source is not constant (no coherent focus)"
    if field_src[0] != qa_neg(s, m):
        return f"WRONG_MECHANISM: focal phase {field_src[0]} != qa_neg(s)={qa_neg(s, m)}"

    # OFF_SOURCE_SCATTER: an off-source location must genuinely differ geometrically
    # (else it is not a distinct point); its field must NOT be constant.
    if all((qa_add(m, go, qa_neg(gs, m)) == qa_add(m, g_off[0], qa_neg(g_src[0], m)))
           for go, gs in zip(g_off, g_src)):
        return "OUT_OF_RANGE: prop_to_off differs from prop_to_source by a constant (not a distinct location)"
    field_off = reemit(g_off, H, tr, m)
    if len(set(field_off)) == 1:
        return "WRONG_MECHANISM: field at off-source location is constant (would focus everywhere)"

    # SPECIFICITY: the mismatched medium must genuinely differ (not H+const), and
    # re-emitting through it must destroy the focus (field at source not constant).
    if all((qa_add(m, hp, qa_neg(h, m)) == qa_add(m, Hp[0], qa_neg(H[0], m)))
           for hp, h in zip(Hp, H)):
        return "OUT_OF_RANGE: medium_alt differs from medium by a constant (not a distinct medium)"
    field_mis = reemit(g_src, Hp, tr, m)
    if len(set(field_mis)) == 1:
        return "WRONG_MECHANISM: mismatched medium still focuses (no same-medium specificity)"

    return None


def _check_empirical(data) -> Optional[str]:
    missing = EMP_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if data["source_recovered"] is not True:
        return "WRONG_EMPIRICAL: source_recovered is not True"
    if not (data["peak_focus"] >= 0.99):
        return f"WRONG_EMPIRICAL: peak_focus {data['peak_focus']} < 0.99"
    if not (data["focal_gain"] > 1.0):
        return f"WRONG_EMPIRICAL: focal_gain {data['focal_gain']} <= 1.0 (no focus)"
    if not (data["specificity_gap"] > 0.3):
        return (f"WRONG_EMPIRICAL: specificity_gap {data['specificity_gap']} <= 0.3 "
                "(no same-medium specificity)")
    return None


def _check_pass_fixture(data) -> Optional[str]:
    if data.get("schema_version") != SCHEMA_VERSION:
        return f"WRONG_SCHEMA: {data.get('schema_version')}"
    if data.get("fixture_kind") != "pass":
        return f"WRONG_KIND: {data.get('fixture_kind')}"
    kind = data.get("kind")
    if kind == "mechanism":
        return _check_mechanism(data)
    if kind == "empirical":
        return _check_empirical(data)
    return f"UNKNOWN_kind: {kind}"


def _check_fail_fixture(data) -> Optional[str]:
    ft = data.get("expected_fail_type")
    if ft not in ("MISSING_FIELD", "WRONG_MECHANISM", "WRONG_EMPIRICAL", "OUT_OF_RANGE"):
        return f"UNKNOWN_expected_fail_type: {ft}"
    kind = data.get("kind")
    req = MECH_FIELDS if kind == "mechanism" else EMP_FIELDS
    if ft == "MISSING_FIELD":
        return None if (req - set(data.keys())) else "FAIL_FIXTURE_DID_NOT_FAIL: all present"
    if req - set(data.keys()):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {ft} but a required field missing"
    err = _check_pass_fixture({**data, "fixture_kind": "pass"})
    if err is None:
        return f"FAIL_FIXTURE_DID_NOT_FAIL: expected {ft} but consistent"
    if not err.startswith(ft):
        return f"FAIL_FIXTURE_WRONG_FAIL: expected {ft} but got {err}"
    return None


def validate_fixture(path: str):
    with open(path) as f:
        data = json.load(f)
    kind = data.get("fixture_kind")
    if kind == "pass":
        err = _check_pass_fixture(data)
        return (False, f"FAIL (expected PASS): {err}") if err else (True, "PASS")
    if kind == "fail":
        err = _check_fail_fixture(data)
        return (False, f"FAIL (fail-fixture): {err}") if err else (True, "PASS (expected FAIL)")
    return False, f"UNKNOWN fixture_kind: {kind}"


# ---------------------------------------------------------------------------
# Deterministic self-test
# ---------------------------------------------------------------------------
def self_test() -> dict:
    errors: List[str] = []
    m = 24
    s = 5
    # 8-element array; propagation phases to the source and to a distinct off-source
    # location (relative geometry genuinely differs across elements), and a
    # scattering medium plus a distinct alternative medium.
    g_src = [3, 5, 8, 12, 15, 19, 22, 24]
    g_off = [7, 8, 8, 13, 11, 2, 20, 6]     # not g_src + const
    H = [11, 4, 23, 6, 17, 2, 9, 14]
    Hp = [5, 18, 7, 20, 3, 21, 1, 16]        # not H + const

    tr = [qa_neg(r, m) for r in record(g_src, H, s, m)]

    # FOCUS_CONSTANT + MEDIUM_CANCEL
    field_src = reemit(g_src, H, tr, m)
    if len(set(field_src)) != 1:
        errors.append("FOCUS_CONSTANT FAILED: field at source not constant")
    if field_src and field_src[0] != qa_neg(s, m):
        errors.append("MEDIUM_CANCEL FAILED: focal phase != qa_neg(s)")

    # OFF_SOURCE_SCATTER
    field_off = reemit(g_off, H, tr, m)
    if len(set(field_off)) == 1:
        errors.append("OFF_SOURCE_SCATTER FAILED: off-source field is constant")

    # SPECIFICITY
    field_mis = reemit(g_src, Hp, tr, m)
    if len(set(field_mis)) == 1:
        errors.append("SPECIFICITY FAILED: mismatched medium still focuses")

    # medium-independence: focus holds for a SECOND arbitrary medium too
    H2 = [2, 2, 19, 5, 5, 13, 8, 21]
    tr2 = [qa_neg(r, m) for r in record(g_src, H2, s, m)]
    if len(set(reemit(g_src, H2, tr2, m))) != 1:
        errors.append("MEDIUM_CANCEL FAILED: focus not medium-independent")

    # A1_RANGE
    if not _in_range([qa_add(m, a, b) for a in range(1, m + 1) for b in (1, 7, m)], m):
        errors.append("A1_RANGE FAILED")

    # SRC
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "mapping_protocol_ref.json")
    if not os.path.exists(src):
        errors.append("SRC FAILED: mapping_protocol_ref.json missing")
    else:
        with open(src) as f:
            if json.load(f).get("protocol_version") != "QA_MAPPING_PROTOCOL_REF.v1":
                errors.append("SRC FAILED: wrong protocol_version")

    # F
    fixture_dir = os.path.join(here, "fixtures")
    pass_fixtures, fail_fixtures = [], []
    if os.path.isdir(fixture_dir):
        for fname in sorted(os.listdir(fixture_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(fixture_dir, fname)
            with open(fpath) as f:
                fx = json.load(f)
            ok, msg = validate_fixture(fpath)
            if fx.get("fixture_kind") == "pass":
                pass_fixtures.append(fname)
                if not ok:
                    errors.append(f"F FAILED (pass {fname}): {msg}")
            elif fx.get("fixture_kind") == "fail":
                fail_fixtures.append(fname)
                if not ok:
                    errors.append(f"F FAILED (fail {fname}): {msg}")
    else:
        errors.append("F FAILED: fixtures/ directory missing")

    return {
        "ok": len(errors) == 0,
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
        print(f"Usage: {sys.argv[0]} <fixture.json> | --self-test", file=sys.stderr)
        sys.exit(1)
    ok, msg = validate_fixture(sys.argv[1])
    print(msg)
    sys.exit(0 if ok else 1)
