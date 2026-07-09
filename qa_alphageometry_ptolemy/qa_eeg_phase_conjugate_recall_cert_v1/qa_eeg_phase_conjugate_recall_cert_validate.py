"""
QA Phase-Conjugate EEG Brain-State Recall Cert [520]

Primary sources:
  (Shoeb, 2009) "Application of Machine Learning to Epileptic Seizure Onset
    Detection and Treatment." MIT PhD thesis. (CHB-MIT Scalp EEG Database)
  (Goldberger, 2000) "PhysioBank, PhysioToolkit, and PhysioNet." Circulation
    101(23):e215-e220. DOI 10.1161/01.CIR.101.23.e215
  (Soffer, 1986) Opt. Lett. 11(2):118-120 DOI 10.1364/OL.11.000118
  (Owechko, 1987) IEEE J. Quantum Electron. 25(3):619-634

CLAIM: the cert [519] phase-conjugate associative memory performs artifact-robust
EEG brain-state recall. A 10s multi-channel window -> topographic QA phase vector
(per-channel z-scored log-power -> phase in {1,...,m}); stored brain-states are
recalled from corrupted/partial probes. KEY MECHANISM: under a global
reference-shift artifact (probe -> qa_add(probe, phi)), NAIVE nearest-overlap
classification collapses toward chance, while PHASE-LOCKED classification (scan the
global compensation phase maximising overlap, read the class in that compensated
frame -- the [518] phase-conjugate mirror self-locking to the medium) stays
correct. Certified here deterministically on synthetic separable patterns; the
real 7-patient CHB-MIT empirical record (reference impl
qa_eeg_phase_conjugate_recall.py) is recorded with provenance -- it cannot be
recomputed in CI (24GB EEG data is not in the git tree).

Checks (deterministic, integer-only, pure stdlib):
  OVERLAP_MATCH       phase-conjugate overlap == exact match count
  PLC_MECHANISM       phase-locked classification recovers the true class for all phi
  NAIVE_FAILS         naive classification returns the WRONG class under a global
                      shift that maps one class onto another (the problem is real)
  DISTORTION_ARTIFACT the artifact is exactly qa_add(probe, phi) (a modular shift)
  EMPIRICAL_WITNESS   the recorded CHB-MIT witness is internally consistent
                      (phase_locked > naive under artifact; recall > chance)
  A1_RANGE            every state in {1,...,m}
  SRC / F             mapping ref present; fixtures behave as declared

Builds on certs [518], [519].
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

QA_COMPLIANCE = (
    "cert_validator -- integer phase arithmetic on {1,...,m} (identity=m, never 0); "
    "overlap/vote scores are float observer-layer quantities, never QA state "
    "(Theorem NT). EEG band-power/log/z-score are observer projections that occur "
    "only in the reference impl, not in this validator."
)

SCHEMA_VERSION = "QA_EEG_PHASE_CONJUGATE_RECALL_CERT.v1"
FAMILY_ID = 520
SLUG = "qa_eeg_phase_conjugate_recall_cert_v1"

MECH_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
               "m", "class_prototypes", "prototype_labels", "probe",
               "probe_label", "global_phi",
               "expected_naive_class", "expected_phase_locked_class"}
EMP_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
              "patient", "recall_clean", "chance",
              "phase_locked_phi6", "naive_phi6"}


# ---------------------------------------------------------------------------
# QA phase algebra + classifiers (pure stdlib)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(a: int, b: int, m: int) -> int:
    return qa_mod(a + b, m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-a, m)


def overlap(x: List[int], pat: List[int], m: int) -> int:
    return sum(1 for xi, pi in zip(x, pat) if qa_add(xi, qa_neg(pi, m), m) == m)


def naive_class(probe, protos, labels, m):
    C = [overlap(probe, p, m) for p in protos]
    return labels[C.index(max(C))]


def phase_locked_class(probe, protos, labels, m):
    """Global-shift-robust: scan compensation phase psi; take (psi, prototype)
    with maximal overlap; read the class in that compensated frame."""
    best_k, best = 0, -1
    for psi in range(1, m + 1):
        shifted = [qa_add(v, psi, m) for v in probe]
        C = [overlap(shifted, p, m) for p in protos]
        k = C.index(max(C))
        if C[k] > best:
            best, best_k = C[k], k
    return labels[best_k]


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
    protos, labels = data["class_prototypes"], data["prototype_labels"]
    probe = data["probe"]
    if not (isinstance(m, int) and m >= 2):
        return f"OUT_OF_RANGE: m={m}"
    # Structural validation (no silent truncation on malformed fixtures).
    if len(labels) != len(protos):
        return f"OUT_OF_RANGE: len(labels)={len(labels)} != len(protos)={len(protos)}"
    if not protos:
        return "OUT_OF_RANGE: empty class_prototypes"
    N = len(probe)
    if any(len(p) != N for p in protos):
        return "OUT_OF_RANGE: prototype length != probe length"
    for seq in protos + [probe]:
        if not _in_range(seq, m):
            return f"OUT_OF_RANGE: value not in 1..{m}"
    nc = naive_class(probe, protos, labels, m)
    pc = phase_locked_class(probe, protos, labels, m)
    if nc != data["expected_naive_class"]:
        return f"WRONG_MECHANISM: naive_class={nc} != expected {data['expected_naive_class']}"
    if pc != data["expected_phase_locked_class"]:
        return f"WRONG_MECHANISM: phase_locked_class={pc} != expected {data['expected_phase_locked_class']}"
    # The certified property: phase-locked recovers the TRUE class.
    if data["probe_label"] is not None and pc != data["probe_label"]:
        return f"WRONG_MECHANISM: phase_locked did not recover true class {data['probe_label']}"
    # Prove the artifact is what fools naive: REMOVING the declared global shift
    # must let naive classify correctly (uses global_phi, and demonstrates that
    # the phi shift -- not the pattern -- is the cause of any naive failure).
    phi = data["global_phi"]
    if phi is not None and data["probe_label"] is not None:
        compensated = [qa_add(v, qa_neg(phi, m), m) for v in probe]
        if naive_class(compensated, protos, labels, m) != data["probe_label"]:
            return ("WRONG_MECHANISM: removing global_phi does not restore the correct "
                    "naive class — the fixture does not demonstrate the artifact shift")
    return None


def _check_empirical(data) -> Optional[str]:
    missing = EMP_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    # Internal consistency of the recorded real-data witness.
    if not (data["recall_clean"] > data["chance"]):
        return f"WRONG_EMPIRICAL: recall_clean {data['recall_clean']} <= chance {data['chance']}"
    if not (data["phase_locked_phi6"] > data["naive_phi6"]):
        return (f"WRONG_EMPIRICAL: phase_locked_phi6 {data['phase_locked_phi6']} "
                f"<= naive_phi6 {data['naive_phi6']} (artifact robustness must hold)")
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
        return None if (req - set(data.keys())) else "FAIL_FIXTURE_DID_NOT_FAIL: all fields present"
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
    errors: list[str] = []
    m = 24
    N = 8
    # Two brain-state prototypes that differ in SPATIAL PATTERN, not by a uniform
    # offset (as real seizure vs baseline topographies do). Because they are NOT
    # global shifts of each other, phase-lock can compensate a global artifact and
    # still tell the classes apart; naive classification is fooled by the shift.
    protoA = [2, 2, 2, 2, 20, 20, 20, 20]     # low-anterior / high-posterior
    protoB = [20, 20, 20, 20, 2, 2, 2, 2]     # the opposite topography
    protos, labels = [protoA, protoB], [0, 1]

    # OVERLAP_MATCH
    for a in (protoA, protoB):
        for b in (protoA, protoB):
            if overlap(a, b, m) != sum(1 for x, y in zip(a, b) if x == y):
                errors.append("OVERLAP_MATCH FAILED")

    # PLC_MECHANISM + NAIVE_FAILS for ALL modular shifts phi. Probe = corrupted
    # protoA, shifted by phi. Phase-lock must recover the true class (0) for every
    # phi; naive must be fooled by at least one phi (the artifact is real).
    base = [2, 2, 2, 3, 20, 20, 19, 20]        # protoA with 2 sites off
    # phi in {1,...,m}; the additive identity is m, so phi=m is the no-shift case
    # and {1,...,m} covers all m distinct modular shifts (A1-clean).
    naive_wrong_seen = False
    for phi in range(1, m + 1):
        probe = [qa_add(v, phi, m) for v in base]
        if phase_locked_class(probe, protos, labels, m) != 0:
            errors.append(f"PLC_MECHANISM FAILED: phi={phi} phase-lock did not recover class 0")
        if naive_class(probe, protos, labels, m) != 0:
            naive_wrong_seen = True   # naive is fooled at some phi (expected)
    if not naive_wrong_seen:
        errors.append("NAIVE_FAILS FAILED: naive never fooled — the artifact is not challenging the classifier")

    # DISTORTION_ARTIFACT: the artifact is exactly a modular shift, involution-free
    for phi in range(1, m + 1):
        x = [7, 3, 20, 1]
        shifted = [qa_add(v, phi, m) for v in x]
        undo = [qa_add(v, qa_neg(phi, m), m) for v in shifted]
        if undo != x:
            errors.append(f"DISTORTION_ARTIFACT FAILED: shift by phi={phi} not invertible")
            break

    # A1_RANGE
    if not _in_range([qa_add(a, b, m) for a in range(1, m + 1) for b in (1, 7, m)], m):
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
