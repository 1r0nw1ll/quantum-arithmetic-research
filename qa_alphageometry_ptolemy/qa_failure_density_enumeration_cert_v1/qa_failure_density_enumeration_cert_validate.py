#!/usr/bin/env python3
"""qa_failure_density_enumeration_cert_validate.py

Validator for QA_FAILURE_DENSITY_ENUMERATION_CERT.v1  [family 263]

First sharp-claim cert derived from the Kochenderfer 2026 'Algorithms for
Validation' bridge (Kochenderfer, 2026; docs/specs/QA_KOCHENDERFER_BRIDGE.md).

Anchors cert [194] qa_cognition_space_morphospace_cert_v1 (Sole, 2026;
arxiv:2601.12837 via Dale, 2026) ratios |reachable_set| / |S_9| in
{1/81, 8/81, 72/81} and recasts them in Kochenderfer Ch. 7 vocabulary as
exact failure-density enumeration:
    p_fail = E_{tau ~ p(.)}[1{tau not in psi}] = integral 1{tau not in psi} p(tau) d tau
specializes to
    p_fail = |{s in S_m : s not in psi}| / |S_m|
on the finite QA orbit graph.

The cert also runs a head-to-head against Kochenderfer's direct-sampling
estimator (Algorithm 7.1) at N in {100, 1000, 10000} and verifies:
- the seeded sampling estimate falls inside a |error| <= 4 * sigma envelope
  (sigma = sqrt(p (1 - p) / N) per Kochenderfer eq. 7.3);
- the QA enumeration error is identically zero.

Source attribution: (Kochenderfer, 2026) Algorithms for Validation, MIT
Press CC-BY-NC-ND, val.pdf 441pp, anchored at
docs/theory/kochenderfer_validation_excerpts.md#val-7-1-direct-estimation-pfail
and docs/specs/QA_KOCHENDERFER_BRIDGE.md.

Checks:
    FDE_1        — schema_version matches
    FDE_RATIO    — exact_enumeration declared ratios reproduce bit-exact
    FDE_SAMPLING — each (class, N, seed) test case |p_hat - p_true| <= 4 sigma
    FDE_STDERR   — theoretical sigma reproduced for each (p, N); enum error == 0
    FDE_UTIL     — tools/qa_kg/orbit_failure_enumeration.py importable + 4 funcs
    FDE_SRC      — source_attribution cites Kochenderfer + cert [194]
    FDE_WIT      — at least 3 orbit witnesses (one per class)
    FDE_F        — fail_ledger well-formed

CLAIM SCOPE: this cert does not prove that QA enumeration is novel as
probability theory; it proves that an existing QA reachability cert
([194]) can be re-expressed in Kochenderfer's validation vocabulary
and gains an exact finite-state estimator with zero sampling variance,
while the corresponding direct estimator exhibits the expected Bernoulli
sampling error.
"""

QA_COMPLIANCE = "cert_validator — verifies exact integer enumeration over finite QA mod-9 orbit graph reproduces cert [194] ratios; sampling helpers are observer projections at output boundary; no float in QA state path"

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

# Make the repo root importable so tools/qa_kg/orbit_failure_enumeration.py
# is reachable regardless of CWD when meta_validator runs us.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.qa_kg.orbit_failure_enumeration import (  # noqa: E402
    ORBIT_CLASSES,
    enumerate_orbit_class_counts,
    exact_success_failure_probability,
    direct_sampling_estimate,
    theoretical_standard_error,
)

SCHEMA_VERSION = "QA_FAILURE_DENSITY_ENUMERATION_CERT.v1"

# 4-sigma envelope for the sampling gate. Per Kochenderfer eq. 7.3 a
# Bernoulli estimator with N samples has standard error sqrt(p(1-p)/N);
# 4 sigma covers ~99.99% of seeded outcomes for any p in [0, 1].
K_SIGMA = 4

# Expected exact ratios on S_9 (from cert [194]).
EXPECTED_S9_RATIOS = {
    "singularity": Fraction(1, 81),
    "satellite":   Fraction(8, 81),
    "cosmos":      Fraction(72, 81),
}


def _canonical_fraction(numerator, denominator) -> Fraction:
    return Fraction(int(numerator), int(denominator))


def validate(path):
    with open(path) as f:
        cert = json.load(f)

    errors: list[str] = []
    warnings: list[str] = []

    # FDE_1: schema version
    sv = cert.get("schema_version")
    if sv != SCHEMA_VERSION:
        errors.append(
            f"FDE_1: schema_version mismatch: got {sv!r}, expected "
            f"{SCHEMA_VERSION!r}"
        )

    # FDE_F: fail_ledger
    fl = cert.get("fail_ledger")
    if fl is None:
        warnings.append("FDE_F: fail_ledger missing")
    elif not isinstance(fl, list):
        errors.append("FDE_F: fail_ledger must be a list")

    # Documented FAIL fixtures short-circuit the gate sweep — they only
    # need to be well-formed (schema + fail_ledger). Mirrors the [194]
    # csm_fail_wrong_agency.json convention.
    if cert.get("result") == "FAIL":
        return errors, warnings

    # FDE_SRC: source_attribution must mention Kochenderfer AND cert [194].
    src = str(cert.get("source_attribution", ""))
    if "Kochenderfer" not in src:
        errors.append(
            "FDE_SRC: source_attribution must cite Kochenderfer 2026 "
            "(Algorithms for Validation)"
        )
    if "194" not in src:
        errors.append(
            "FDE_SRC: source_attribution must cite cert [194] "
            "qa_cognition_space_morphospace_cert_v1 as the anchor"
        )

    # FDE_UTIL: utility module exposes the four named functions.
    for fn in (
        "enumerate_orbit_class_counts",
        "exact_success_failure_probability",
        "direct_sampling_estimate",
        "theoretical_standard_error",
    ):
        if fn not in globals():
            errors.append(f"FDE_UTIL: utility function {fn!r} not importable")

    # FDE_RATIO: exact_enumeration declared ratios reproduce bit-exact.
    enum_decl = cert.get("exact_enumeration")
    if not isinstance(enum_decl, dict):
        errors.append("FDE_RATIO: exact_enumeration section missing")
    else:
        modulus = int(enum_decl.get("modulus", 0))
        if modulus != 9:
            errors.append(
                f"FDE_RATIO: this cert version anchors mod-9 only "
                f"(per cert [194]); got modulus={modulus}"
            )
        else:
            counts = enumerate_orbit_class_counts(9)
            declared_counts = enum_decl.get("counts", {})
            for cls in ORBIT_CLASSES:
                if int(declared_counts.get(cls, -1)) != counts[cls]:
                    errors.append(
                        f"FDE_RATIO: declared {cls} count="
                        f"{declared_counts.get(cls)}, computed {counts[cls]}"
                    )
            declared_total = int(declared_counts.get("total", -1))
            if declared_total != counts["total"]:
                errors.append(
                    f"FDE_RATIO: declared total={declared_total}, "
                    f"computed {counts['total']}"
                )
            for cls in ORBIT_CLASSES:
                expected = EXPECTED_S9_RATIOS[cls]
                p = exact_success_failure_probability(9, cls)
                if p["p_success"] != expected:
                    errors.append(
                        f"FDE_RATIO: utility produced p_success={p['p_success']} "
                        f"for {cls}, expected {expected}"
                    )
            ratios = enum_decl.get("ratios", {})
            for cls in ORBIT_CLASSES:
                cell = ratios.get(cls, {})
                num = int(cell.get("numerator", -1))
                den = int(cell.get("denominator", -1))
                if Fraction(num, den) != EXPECTED_S9_RATIOS[cls]:
                    errors.append(
                        f"FDE_RATIO: declared {cls} ratio={num}/{den}, "
                        f"expected {EXPECTED_S9_RATIOS[cls].numerator}/"
                        f"{EXPECTED_S9_RATIOS[cls].denominator}"
                    )

    # FDE_SAMPLING + FDE_STDERR: each declared (class, N, seed) test case
    # must (a) reproduce bit-exact under fixed seed, (b) fall inside the
    # 4-sigma envelope, (c) match the theoretical sigma from Kochenderfer
    # eq. 7.3.
    sampling_cases = cert.get("sampling_test_cases", [])
    if not isinstance(sampling_cases, list) or len(sampling_cases) < 3:
        errors.append(
            "FDE_SAMPLING: need >= 3 sampling_test_cases (one per N); got "
            f"{len(sampling_cases) if isinstance(sampling_cases, list) else 'malformed'}"
        )
    else:
        for i, case in enumerate(sampling_cases):
            cls = case.get("target_class")
            if cls not in ORBIT_CLASSES:
                errors.append(
                    f"FDE_SAMPLING[{i}]: target_class={cls!r} not in {ORBIT_CLASSES}"
                )
                continue
            n = int(case.get("n_samples", 0))
            seed = case.get("seed")
            if seed is None:
                errors.append(
                    f"FDE_SAMPLING[{i}]: seed required for reproducibility "
                    f"(cert [263] explicitly forbids unseeded sampling)"
                )
                continue
            seed = int(seed)
            est = direct_sampling_estimate(9, cls, n, seed)
            exact = exact_success_failure_probability(9, cls)
            p_true = float(exact["p_success"])
            sigma = theoretical_standard_error(p_true, n)
            err = abs(est["p_hat"] - p_true)

            declared_p_hat = case.get("expected_p_hat")
            if declared_p_hat is not None:
                if abs(float(declared_p_hat) - est["p_hat"]) > 1e-12:
                    errors.append(
                        f"FDE_SAMPLING[{i}]: declared expected_p_hat="
                        f"{declared_p_hat} != utility-recomputed "
                        f"p_hat={est['p_hat']} (seed={seed} not reproducible?)"
                    )
            declared_n_hit = case.get("expected_n_in_target")
            if declared_n_hit is not None:
                if int(declared_n_hit) != int(est["n_in_target"]):
                    errors.append(
                        f"FDE_SAMPLING[{i}]: declared expected_n_in_target="
                        f"{declared_n_hit} != utility {est['n_in_target']}"
                    )
            if err > K_SIGMA * sigma + 1e-12:
                errors.append(
                    f"FDE_SAMPLING[{i}]: cls={cls} N={n} seed={seed} "
                    f"|p_hat - p_true|={err:.6f} exceeds {K_SIGMA}*sigma="
                    f"{K_SIGMA * sigma:.6f}"
                )
            declared_sigma = case.get("theoretical_sigma")
            if declared_sigma is not None:
                if abs(float(declared_sigma) - sigma) > 1e-9:
                    errors.append(
                        f"FDE_STDERR[{i}]: declared theoretical_sigma="
                        f"{declared_sigma} != recomputed {sigma:.9f}"
                    )

    enum_err = cert.get("enumeration_error")
    if enum_err is not None and float(enum_err) != 0.0:
        errors.append(
            f"FDE_STDERR: enumeration_error must be 0 (exact); got {enum_err}"
        )

    # FDE_WIT: at least 3 witnesses, one per class.
    witnesses = cert.get("witnesses", [])
    if not isinstance(witnesses, list) or len(witnesses) < 3:
        errors.append(
            f"FDE_WIT: need >= 3 witnesses (one per class), got "
            f"{len(witnesses) if isinstance(witnesses, list) else 'malformed'}"
        )
    else:
        seen_classes = {w.get("orbit_class") for w in witnesses}
        for cls in ORBIT_CLASSES:
            if cls not in seen_classes:
                errors.append(
                    f"FDE_WIT: missing witness for class {cls!r}"
                )

    return errors, warnings


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    expected = [
        ("pass_mod9_cognition_morphospace.json", True),
        ("pass_sampling_comparison.json", True),
        ("fail_bad_ratio.json", True),
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
            results.append({
                "fixture": fname, "ok": False,
                "error": f"expected PASS but got errors: {errs}",
            })
            all_ok = False
        elif not should_pass and passed:
            results.append({
                "fixture": fname, "ok": False,
                "error": "expected FAIL but got PASS",
            })
            all_ok = False
        else:
            results.append({"fixture": fname, "ok": True, "errors": errs})
    return {"ok": all_ok, "results": results}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="QA Failure Density Enumeration Cert [263] validator")
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
