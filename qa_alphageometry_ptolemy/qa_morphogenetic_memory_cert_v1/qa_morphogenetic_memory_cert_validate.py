"""
QA Phase-Conjugate Morphogenetic Memory Cert [521]

Primary sources:
  (Levin, 2021) "Technological Approach to Mind Everywhere." Front. Syst.
    Neurosci. 15:768201. DOI 10.3389/fnsys.2021.768201
  (Pezzulo & Levin, 2015) "Re-membering the body: top-down control of
    regeneration." Integr. Biol. 7:1487-1517. DOI 10.1039/c5ib00221d
  (Hopfield, 1982) PNAS 79(8):2554-2558. DOI 10.1073/pnas.79.8.2554
  (Soffer, 1986) Opt. Lett. 11(2):118-120. DOI 10.1364/OL.11.000118

CLAIM: Levin's bioelectric target-morphology attractor -- tissue navigating back
to the correct body plan from damaged states -- is the cert [519] phase-conjugate
associative memory (on cert [518]'s conjugator). Body plans are 2D QA phase
fields; a DAMAGED (amputated) field regrows to the correct target by
content-addressable recall. KEY MECHANISM: a body plan regenerates to its OWN
stored plan (no chimera); and under a systemic bioelectric shift (probe ->
qa_add(probe, phi)) naive regeneration is fooled while PHASE-LOCKED recall (the
[518] mirror self-locking to the perturbing medium) restores the correct
morphology. Certified deterministically on synthetic spatially-different body
plans; the reference-impl empirical record is recorded with provenance.

Checks (deterministic, integer-only, pure stdlib):
  REGEN                 an amputated body plan regenerates to the true plan
  WHICH_PLAN            regeneration selects the true plan, not another
  NO_CHIMERA            the regenerated field is an EXACT stored plan
  SYSTEMIC_PHASE_LOCK   phase-locked recall regenerates through a systemic shift
  NAIVE_SYSTEMIC_FAILS  naive regeneration is fooled by a systemic shift (control)
  OVERLAP_MATCH         phase-conjugate overlap == exact match count
  A1_RANGE              every state in {1,...,m}
  SRC / F               mapping ref present; fixtures behave as declared

Builds on certs [518], [519]; companion to cert [520].
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

QA_COMPLIANCE = (
    "cert_validator -- integer phase arithmetic on {1,...,m} (identity=m, never 0); "
    "overlap scores are float observer-layer quantities, never QA state (Theorem "
    "NT). Body-plan field synthesis / damage occur only in the reference impl."
)

SCHEMA_VERSION = "QA_MORPHOGENETIC_MEMORY_CERT.v1"
FAMILY_ID = 521
SLUG = "qa_morphogenetic_memory_cert_v1"

MECH_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
               "m", "body_plans", "plan_labels", "damaged_probe",
               "true_plan", "global_phi", "expected_naive_plan",
               "expected_regen_plan"}
EMP_FIELDS = {"schema_version", "fixture_kind", "primary_source", "kind",
              "regen_fidelity", "no_chimera_rate",
              "phase_locked_systemic", "naive_systemic"}


# ---------------------------------------------------------------------------
# QA phase algebra + regeneration (pure stdlib)
# ---------------------------------------------------------------------------
def qa_mod(x: int, m: int) -> int:
    return ((int(x) - 1) % m) + 1


def qa_add(a: int, b: int, m: int) -> int:
    return qa_mod(a + b, m)


def qa_neg(a: int, m: int) -> int:
    return qa_mod(-a, m)


def overlap(x: List[int], plan: List[int], m: int) -> int:
    return sum(1 for xi, pi in zip(x, plan) if qa_add(xi, qa_neg(pi, m), m) == m)


def regen_plan(probe, plans, labels, m):
    """Content-addressable regeneration: the stored plan best matching the probe."""
    C = [overlap(probe, p, m) for p in plans]
    return labels[C.index(max(C))]


def regen_plan_phase_locked(probe, plans, labels, m):
    """Systemic-shift-robust regeneration: scan the global compensation phase psi;
    take the (psi, plan) with maximal overlap; read the plan in that frame."""
    best_k, best = 0, -1
    for psi in range(1, m + 1):
        shifted = [qa_add(v, psi, m) for v in probe]
        C = [overlap(shifted, p, m) for p in plans]
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
    plans, labels = data["body_plans"], data["plan_labels"]
    probe = data["damaged_probe"]
    if not (isinstance(m, int) and m >= 2):
        return f"OUT_OF_RANGE: m={m}"
    if len(labels) != len(plans):
        return f"OUT_OF_RANGE: len(labels)={len(labels)} != len(plans)={len(plans)}"
    if not plans:
        return "OUT_OF_RANGE: empty body_plans"
    Np = len(probe)
    if any(len(p) != Np for p in plans):
        return "OUT_OF_RANGE: plan length != probe length"
    for seq in plans + [probe]:
        if not _in_range(seq, m):
            return f"OUT_OF_RANGE: value not in 1..{m}"
    nc = regen_plan(probe, plans, labels, m)
    pc = regen_plan_phase_locked(probe, plans, labels, m)
    # Pin the naive behaviour: the fixture must declare what naive regeneration
    # returns, so a fixture claiming "systemic shift fools naive" actually proves
    # it (naive != true_plan) rather than being trusted.
    if nc != data["expected_naive_plan"]:
        return f"WRONG_MECHANISM: naive plan={nc} != expected {data['expected_naive_plan']}"
    # phase-locked regeneration recovers the TRUE plan (robust to any systemic shift)
    if pc != data["expected_regen_plan"]:
        return f"WRONG_MECHANISM: phase_locked plan={pc} != expected {data['expected_regen_plan']}"
    if data["true_plan"] is not None and pc != data["true_plan"]:
        return f"WRONG_MECHANISM: phase-lock did not regenerate the true plan {data['true_plan']}"
    # NAIVE_SYSTEMIC_FAILS: under a real systemic shift the fixture must show naive
    # genuinely fooled (naive != true_plan) — otherwise it does not demonstrate the
    # control the cert claims.
    phi0 = data["global_phi"]
    if (phi0 is not None and qa_mod(phi0, m) != m and data["true_plan"] is not None
            and data["expected_naive_plan"] == data["true_plan"]):
        return ("WRONG_MECHANISM: global_phi is a systemic shift but naive is not fooled "
                "(expected_naive_plan == true_plan) — fixture does not demonstrate "
                "NAIVE_SYSTEMIC_FAILS")
    # Prove the systemic shift is what fools naive: removing it must restore
    # correct naive regeneration (uses global_phi, demonstrates the perturbation).
    phi = data["global_phi"]
    if phi is not None and data["true_plan"] is not None:
        compensated = [qa_add(v, qa_neg(phi, m), m) for v in probe]
        if regen_plan(compensated, plans, labels, m) != data["true_plan"]:
            return ("WRONG_MECHANISM: removing global_phi does not restore correct naive "
                    "regeneration — the fixture does not demonstrate a systemic shift")
    return None


def _check_empirical(data) -> Optional[str]:
    missing = EMP_FIELDS - set(data.keys())
    if missing:
        return f"MISSING_FIELD: {sorted(missing)}"
    if not (data["no_chimera_rate"] >= 0.99):
        return f"WRONG_EMPIRICAL: no_chimera_rate {data['no_chimera_rate']} < 0.99"
    if not (data["phase_locked_systemic"] > data["naive_systemic"]):
        return (f"WRONG_EMPIRICAL: phase_locked_systemic {data['phase_locked_systemic']} "
                f"<= naive_systemic {data['naive_systemic']} (systemic robustness must hold)")
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
    errors: list[str] = []
    m = 24
    # Two body plans that differ in SPATIAL PATTERN (not a global shift of each
    # other), like anterior-posterior vs its mirror. A systemic shift then fools
    # naive regeneration but not phase-lock.
    planA = [2, 2, 2, 2, 20, 20, 20, 20]
    planB = [20, 20, 20, 20, 2, 2, 2, 2]
    plans, labels = [planA, planB], [0, 1]

    # OVERLAP_MATCH
    for a in (planA, planB):
        for b in (planA, planB):
            if overlap(a, b, m) != sum(1 for x, y in zip(a, b) if x == y):
                errors.append("OVERLAP_MATCH FAILED")

    # REGEN + WHICH_PLAN + NO_CHIMERA: amputate a boundary region of planA
    # (keeping enough distinguishing cells that regeneration is unambiguous even
    # under a systemic shift — amputating a WHOLE distinguishing half would make
    # planA/planB genuinely ambiguous, an honest failure mode outside this witness)
    amputated = [2, 2, 2, 7, 7, 20, 20, 20]     # boundary cells damaged
    if regen_plan(amputated, plans, labels, m) != 0:
        errors.append("REGEN FAILED: amputated planA did not regenerate plan 0")
    # No chimera: the regenerated field is exactly a stored plan (planA)
    C = [overlap(amputated, p, m) for p in plans]
    if plans[C.index(max(C))] != planA:
        errors.append("NO_CHIMERA FAILED: regenerated field is not an exact stored plan")

    # SYSTEMIC_PHASE_LOCK + NAIVE_SYSTEMIC_FAILS across ALL modular shifts
    naive_fooled = False
    for phi in range(1, m + 1):
        probe = [qa_add(v, phi, m) for v in amputated]
        if regen_plan_phase_locked(probe, plans, labels, m) != 0:
            errors.append(f"SYSTEMIC_PHASE_LOCK FAILED: phi={phi} did not regenerate plan 0")
        if regen_plan(probe, plans, labels, m) != 0:
            naive_fooled = True
    if not naive_fooled:
        errors.append("NAIVE_SYSTEMIC_FAILS FAILED: naive never fooled by a systemic shift")

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
