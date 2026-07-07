#!/usr/bin/env python3
"""
qa_keely_aggregation_cert_validate.py

Validator for QA_KEELY_AGGREGATION_CERT.v1  [family 187]

Certifies: Keely's 5 Aggregation/Disintegration Laws (Category 4 of Vibes
5-category framework) mapped to QA state composition and decomposition.

Laws: 3 (Corporeal Oscillations), 12 (Oscillating Atomic Substances),
      28 (Chemical Dissociation), 34 (Atomic Dissociation),
      35 (Atomolic Synthesis of Chemical Elements)

Core mapping: non-isolated states modify each other through coupling
(Law 3); orbit density determines effective pitch (Law 12); discord
causes dissociation = orbit separation (Laws 28, 34); pitch selection
determines orbit membership deterministically (Law 35).

Checks:
  KAG_1       — schema_version matches
  KAG_LAWS    — all 5 law numbers present
  KAG_COUPLE  — isolated/coupled witness orbits genuinely reclassified
                from (b,e,modulus) via classify_orbit, not just trusted
                as declared strings (Law 3) -- previously listed in the
                docstring but not implemented at all
  KAG_DENSITY — orbit density genuinely recomputed as count/period ratio
                (not just a bare cosmos_count>satellite_count
                inequality), plus 72+8+1==modulus**2 (Law 12)
  KAG_DISSOC  — concordant/discordant pair orbits genuinely reclassified
                from (b,e,modulus) and cross-orbit-ness recomputed
                (Laws 28, 34) -- previously listed in the docstring but
                not implemented at all
  KAG_SYNTH   — d, a, f_value, orbit genuinely recomputed from the
                declared (b,e,modulus) and compared, not just a trusted
                boolean flag (Law 35)
  KAG_W       — at least 3 witnesses
  KAG_F       — fail_ledger well-formed

Primary source: Pond, D. (svpwiki.com), Keely's 40 Laws of Vibratory
Physics ("Law of Corporeal Oscillations", "Law of Oscillating Atomic
Substances", "Law of Chemical Dissociation", "Law of Atomic
Dissociation", "Law of Atomolic Synthesis of Chemical Elements"). QA
orbit/determinism structure per Iverson (1991).
"""

QA_COMPLIANCE = "cert_validator — validates Keely aggregation law mappings; no float state"

import json
import sys
from pathlib import Path

SCHEMA_VERSION = "QA_KEELY_AGGREGATION_CERT.v1"
REQUIRED_LAWS = frozenset([3, 12, 28, 34, 35])
ORBIT_PERIOD = {"SINGULARITY": 1, "SATELLITE": 8, "COSMOS": 24}


def qa_mod(x, m):
    """A1-compliant: result in {1,...,m}, never 0."""
    return ((int(x) - 1) % m) + 1


def classify_orbit(b, e, m):
    b_m, e_m = qa_mod(b, m), qa_mod(e, m)
    if b_m == m and e_m == m:
        return "SINGULARITY"
    if b_m % 3 == 0 and e_m % 3 == 0:
        return "SATELLITE"
    return "COSMOS"


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors = []
    warnings = []

    # KAG_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(f"KAG_1: schema_version mismatch: got {sv!r}, expected {SCHEMA_VERSION!r}")

    # KAG_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("KAG_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("KAG_F: fail_ledger must be a list")

    if cert.get("result") == "FAIL":
        return errors, warnings

    # KAG_LAWS: all 5 laws present
    laws = cert.get("laws", {})
    declared_nums = set()
    if isinstance(laws, dict):
        declared_nums = {int(k) for k in laws.keys()}
    elif isinstance(laws, list):
        declared_nums = {entry.get("law_number") for entry in laws if isinstance(entry, dict)}
    missing = REQUIRED_LAWS - declared_nums
    if missing:
        errors.append(f"KAG_LAWS: missing law numbers: {sorted(missing)}")

    # KAG_DENSITY: orbit density genuinely recomputed as count/period
    # ratio, plus the modulus**2 = sum-of-counts invariant (Law 12).
    density = cert.get("orbit_density")
    if density:
        modulus = density.get("modulus")
        cosmos_count = density.get("cosmos_count")
        satellite_count = density.get("satellite_count")
        singularity_count = density.get("singularity_count")
        counts = [cosmos_count, satellite_count, singularity_count]
        if modulus is not None and all(c is not None for c in counts):
            total = cosmos_count + satellite_count + singularity_count
            if total != modulus * modulus:
                errors.append(f"KAG_DENSITY: cosmos+satellite+singularity={total}, "
                              f"expected modulus**2={modulus*modulus}")
            cosmos_ratio = cosmos_count / ORBIT_PERIOD["COSMOS"]
            satellite_ratio = satellite_count / ORBIT_PERIOD["SATELLITE"]
            singularity_ratio = singularity_count / ORBIT_PERIOD["SINGULARITY"]
            if not (cosmos_ratio > satellite_ratio):
                errors.append(f"KAG_DENSITY: cosmos density ratio {cosmos_ratio} "
                              f"does not exceed satellite ratio {satellite_ratio} "
                              f"(Law 12 requires pitch/density to vary directly)")
            if satellite_ratio != singularity_ratio:
                errors.append(f"KAG_DENSITY: satellite ratio {satellite_ratio} != "
                              f"singularity ratio {singularity_ratio} (both should be 1 "
                              f"state/step per the standard 81-state partition)")
    else:
        warnings.append("KAG_DENSITY: orbit_density block not declared")

    # KAG_COUPLE: isolated/coupled witness orbits genuinely reclassified
    # from (b,e,modulus), not trusted as declared strings (Law 3).
    modulus_default = density.get("modulus", 9) if density else 9
    witnesses = cert.get("witnesses", [])
    for idx, w in enumerate(witnesses):
        if 3 not in w.get("law_refs", []):
            continue
        iso = w.get("isolated_state")
        coup = w.get("coupled_state")
        if iso is not None:
            b, e = iso.get("b"), iso.get("e")
            m = iso.get("modulus", modulus_default)
            declared = iso.get("orbit")
            if b is not None and e is not None and declared is not None:
                actual = classify_orbit(b, e, m)
                if actual != declared:
                    errors.append(f"KAG_COUPLE: witness[{idx}].isolated_state ({b},{e}) "
                                  f"is genuinely {actual}, not declared {declared}")
        if coup is not None:
            b, e = coup.get("b"), coup.get("e")
            m = coup.get("modulus", modulus_default)
            declared = coup.get("orbit")
            if b is not None and e is not None and declared is not None:
                actual = classify_orbit(b, e, m)
                if actual != declared:
                    errors.append(f"KAG_COUPLE: witness[{idx}].coupled_state ({b},{e}) "
                                  f"is genuinely {actual}, not declared {declared}")

    # KAG_DISSOC: concordant/discordant pair orbits genuinely
    # reclassified and cross-orbit-ness recomputed (Laws 28, 34).
    for idx, w in enumerate(witnesses):
        if not ({28, 34} & set(w.get("law_refs", []))):
            continue
        conc = w.get("concordant_pair")
        disc = w.get("discordant_pair")
        if conc is not None:
            b1, e1, b2, e2 = conc.get("b1"), conc.get("e1"), conc.get("b2"), conc.get("e2")
            m = conc.get("modulus", modulus_default)
            o1, o2 = classify_orbit(b1, e1, m), classify_orbit(b2, e2, m)
            declared_orbit = conc.get("orbit")
            if declared_orbit is not None and (o1 != declared_orbit or o2 != declared_orbit):
                errors.append(f"KAG_DISSOC: witness[{idx}].concordant_pair genuinely "
                              f"({o1},{o2}), not both declared {declared_orbit}")
            if conc.get("concordant") is True and o1 != o2:
                errors.append(f"KAG_DISSOC: witness[{idx}].concordant_pair declared "
                              f"concordant=true but orbits differ ({o1} vs {o2})")
        if disc is not None:
            b1, e1, b2, e2 = disc.get("b1"), disc.get("e1"), disc.get("b2"), disc.get("e2")
            m = disc.get("modulus", modulus_default)
            o1, o2 = classify_orbit(b1, e1, m), classify_orbit(b2, e2, m)
            declared_orbits = disc.get("orbits")
            if declared_orbits is not None and [o1, o2] != list(declared_orbits):
                errors.append(f"KAG_DISSOC: witness[{idx}].discordant_pair genuinely "
                              f"{[o1, o2]}, not declared {declared_orbits}")
            if disc.get("dissociated") is True and o1 == o2:
                errors.append(f"KAG_DISSOC: witness[{idx}].discordant_pair declared "
                              f"dissociated=true but orbits are both {o1} (same orbit, "
                              f"not genuinely cross-orbit)")

    # KAG_SYNTH: d, a, f_value, orbit genuinely recomputed from the
    # declared (b,e,modulus), not just a trusted boolean flag (Law 35).
    synth = cert.get("deterministic_synthesis")
    if synth:
        if synth.get("fully_determined") is not True:
            errors.append("KAG_SYNTH: deterministic_synthesis.fully_determined must be true")
    for idx, w in enumerate(witnesses):
        if 35 not in w.get("law_refs", []):
            continue
        inp = w.get("input_pair")
        derived = w.get("derived_tuple")
        m = w.get("modulus", modulus_default)
        if inp is not None and derived is not None:
            b, e = inp.get("b"), inp.get("e")
            d_exp, a_exp = qa_mod(b + e, m), qa_mod(b + 2 * e, m)
            if derived.get("d") != d_exp or derived.get("a") != a_exp:
                errors.append(f"KAG_SYNTH: witness[{idx}] declared derived_tuple "
                              f"d={derived.get('d')},a={derived.get('a')}, expected "
                              f"d={d_exp},a={a_exp} for (b={b},e={e},m={m})")
            f_exp = b * b + b * e - e * e
            if w.get("f_value") is not None and w.get("f_value") != f_exp:
                errors.append(f"KAG_SYNTH: witness[{idx}] declared f_value="
                              f"{w.get('f_value')}, expected {f_exp}")
            orbit_exp = classify_orbit(b, e, m)
            if w.get("orbit") is not None and w.get("orbit") != orbit_exp:
                errors.append(f"KAG_SYNTH: witness[{idx}] declared orbit="
                              f"{w.get('orbit')}, expected {orbit_exp}")

    # KAG_W: witnesses
    if len(witnesses) < 3:
        errors.append(f"KAG_W: need >= 3 witnesses, got {len(witnesses)}")

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("kag_pass_composition.json", True),
        ("kag_fail_bad_density.json", True),
    ]
    results = []
    all_ok = True

    for fname, should_pass in expected:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errs, warns = validate(fpath)
            passed = len(errs) == 0
        except Exception as ex:
            results.append({"fixture": fname, "ok": False, "error": str(ex)})
            all_ok = False
            continue

        if should_pass and not passed:
            results.append({"fixture": fname, "ok": False,
                            "error": f"expected PASS but got errors: {errs}"})
            all_ok = False
        elif not should_pass and passed:
            results.append({"fixture": fname, "ok": False,
                            "error": "expected FAIL but got PASS"})
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})

    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Keely Aggregation Cert [187] validator")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or list(
        (Path(__file__).parent / "fixtures").glob("*.json"))

    total_errors = 0
    for path in paths:
        path = Path(path)
        print(f"Validating {path.name}...")
        try:
            errs, warns = validate(path)
        except Exception as ex:
            print(f"  ERROR: {ex}")
            total_errors += 1
            continue
        for w in warns:
            print(f"  WARN: {w}")
        for e in errs:
            print(f"  FAIL: {e}")
        if not errs:
            print("  PASS")
        else:
            total_errors += len(errs)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll fixtures validated.")
        sys.exit(0)


if __name__ == "__main__":
    main()
