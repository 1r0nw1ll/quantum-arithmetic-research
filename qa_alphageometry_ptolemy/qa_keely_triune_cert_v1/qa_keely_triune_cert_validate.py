#!/usr/bin/env python3
QA_COMPLIANCE = "observer=cert_validator, state_alphabet=orbit_triune_mapping"
"""QA Keely Triune Cert family [153] — certifies the mapping between
Keely's Triune (three vibratory modes) and QA's three orbit types.

KEELY'S TRIUNE (svpwiki.com, Dale Pond SVP):
  ENHARMONIC — terrestrial, matter, entropy, radiation (1st subdivision)
  DOMINANT   — neutral center, life, dynamic balance (4th subdivision)
  HARMONIC   — celestial, spirit, syntropy, concentration (7th subdivision)

QA THREE ORBITS (Iverson):
  SINGULARITY — 1-cycle fixed point (9,9); neutral, static
  SATELLITE   — 8-cycle bounded orbit; material, symmetric
  COSMOS      — 24-cycle expansive orbit; dynamic, linear

CERTIFIED MAPPING:
  DOMINANT   ↔ SINGULARITY (neutral center = fixed point)
  ENHARMONIC ↔ SATELLITE   (bounded/material = 8-cycle)
  HARMONIC   ↔ COSMOS      (expansive/celestial = 24-cycle)

STRUCTURAL PROPERTIES:
  - Three orbits partition the full state space (no overlap, complete coverage)
  - Orbit periods: 1, 8, 24; LCM(1,8,24)=24 (cosmos contains both)
  - {3,6,9} mod 9 = triune numbers (Tesla "3-6-9"); these are exactly
    the singularity-class residues (multiples of 3)
  - Three Brinton laws map: Assimilation=cosmos(expansion), Individualization=satellite(contraction), Dominant=singularity(balance)

Checks: KT_1 (schema), KT_MAP (triune↔orbit mapping declared correctly),
KT_PART (three orbits partition state space), KT_PERIOD (periods 1,8,24),
KT_369 ({3,6,9}=singularity residues), KT_LCM (LCM(1,8,24)=24),
KT_W (>=3 witnesses), KT_F (mod-9 fundamental present).
"""

import json
import os
import sys
from math import gcd


SCHEMA = "QA_KEELY_TRIUNE_CERT.v1"

VALID_TRIUNE = frozenset(["ENHARMONIC", "DOMINANT", "HARMONIC"])
VALID_ORBITS = frozenset(["SINGULARITY", "SATELLITE", "COSMOS"])
CANONICAL_MAP = {
    "DOMINANT": "SINGULARITY",
    "ENHARMONIC": "SATELLITE",
    "HARMONIC": "COSMOS",
}
ORBIT_PERIODS = {"SINGULARITY": 1, "SATELLITE": 8, "COSMOS": 24}
SINGULARITY_RESIDUES = frozenset({0, 3, 6})  # mod 9


def lcm(a, b):
    return a * b // gcd(a, b)


def validate(cert, *, collect_errors=True):
    errors = []
    warnings = []

    def err(chk, msg):
        errors.append({"check_id": chk, "message": msg})

    if cert.get("schema_version") != SCHEMA:
        err("KT_1", f"schema_version must be {SCHEMA}")

    # KT_MAP — triune-orbit mapping
    mapping = cert.get("triune_orbit_mapping", {})
    for triune, orbit in CANONICAL_MAP.items():
        decl = mapping.get(triune)
        if decl is not None and decl != orbit:
            err("KT_MAP", f"{triune} should map to {orbit}, declared {decl}")

    # KT_PART — three orbits partition state space
    partition = cert.get("orbit_partition", {})
    if partition:
        counts = partition.get("orbit_counts", {})
        total = sum(counts.values()) if counts else 0
        modulus = partition.get("modulus", 0)
        if modulus > 0:
            expected_total = modulus * modulus  # state space = modulus²
            if total != expected_total:
                err("KT_PART", f"orbit counts sum to {total}, expected {expected_total} for mod-{modulus}")
        # Check all three orbits represented
        for orb in VALID_ORBITS:
            if orb not in counts or counts[orb] <= 0:
                err("KT_PART", f"orbit {orb} missing or zero in partition")

    # KT_PERIOD — orbit periods
    periods = cert.get("orbit_periods", {})
    for orb, expected_p in ORBIT_PERIODS.items():
        decl_p = periods.get(orb)
        if decl_p is not None and decl_p != expected_p:
            err("KT_PERIOD", f"{orb} period declared {decl_p}, expected {expected_p}")

    # KT_369 — {3,6,9≡0} = singularity residues
    decl_369 = cert.get("singularity_residues_mod9")
    if decl_369 is not None:
        if frozenset(decl_369) != SINGULARITY_RESIDUES:
            err("KT_369", f"singularity residues declared {sorted(decl_369)}, expected {sorted(SINGULARITY_RESIDUES)}")

    # KT_LCM — LCM(1,8,24) = 24
    decl_lcm = cert.get("orbit_lcm")
    computed_lcm = lcm(lcm(1, 8), 24)
    if decl_lcm is not None and decl_lcm != computed_lcm:
        err("KT_LCM", f"LCM declared {decl_lcm}, computed {computed_lcm}")

    # KT_W — witness triune-orbit examples
    witnesses = cert.get("witnesses", [])
    if len(witnesses) < 3:
        err("KT_W", f"need >=3 witnesses, got {len(witnesses)}")

    # KT_F — mod-9 fundamental
    has_mod9 = cert.get("modulus") == 9 or (partition and partition.get("modulus") == 9)
    if not has_mod9:
        warnings.append("KT_F: modulus 9 not explicitly declared")

    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    fix_dir = os.path.join(here, "fixtures")
    expected = {
        "kt_pass_triune_mapping.json": True,
        "kt_pass_brinton_laws.json": True,
    }
    results = []
    for fname, should_pass in expected.items():
        path = os.path.join(fix_dir, fname)
        with open(path) as f:
            cert = json.load(f)
        res = validate(cert)
        ok = res["ok"] == should_pass
        results.append({
            "fixture": fname,
            "expected_pass": should_pass,
            "actual_pass": res["ok"],
            "ok": ok,
            "errors": res["errors"],
            "warnings": res["warnings"],
        })
    return results


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        results = self_test()
        all_ok = all(r["ok"] for r in results)
        print(json.dumps({"ok": all_ok, "results": results}, indent=2))
    elif len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            cert = json.load(f)
        print(json.dumps(validate(cert), indent=2))
    else:
        print("Usage: python qa_keely_triune_cert_validate.py [--self-test | <fixture.json>]")
