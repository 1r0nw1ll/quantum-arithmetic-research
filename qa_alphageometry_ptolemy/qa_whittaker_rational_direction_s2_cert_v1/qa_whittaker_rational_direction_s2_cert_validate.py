#!/usr/bin/env python3
"""QA Whittaker Rational Direction S2 Cert validator.
# RT1_OBSERVER_FILE: Whittaker rational direction — classical trig as verification for RT claims

Candidate family ID: [273], unregistered until hostile review.

Layer 2 of the Whittaker -> QA development ladder. This validator builds an
exact rational direction set on S^2 from Layer 1 QA-rational ratios.

Claim scope: exact finite rational geometry only. This cert does not prove
Whittaker 1903, Maxwell/EM, scalar-potential physics, density,
equidistribution, convergence, geodesy, or ellipsoid physics.
"""

QA_COMPLIANCE = "cert_validator - exact rational S2 geometry; integer + fractions.Fraction construction; no **2; floats observer-side reporting only"

import argparse
import json
import math
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

SCHEMA_VERSION = "QA_WHITTAKER_RATIONAL_DIRECTION_S2_CERT.v1"
CERT_SLUG = "qa_whittaker_rational_direction_s2_cert_v1"
CANDIDATE_FAMILY_ID = 273
ALLOWED_M = {3, 5, 9}
CHART = "inverse_stereographic_excluding_south_pole"


def _err(errors, code, msg):
    errors.append(f"{code}: {msg}")


def _frac_from_json(value):
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, list) and len(value) == 2:
        return Fraction(value[0], value[1])
    raise ValueError(f"cannot parse Fraction from {value!r}")


def _frac_to_json(value):
    return f"{value.numerator}/{value.denominator}"


def _packet_to_json(packet):
    x, y, z, den = packet
    return [x, y, z, den]


def _canonical_packet_from_fracs(x, y, z):
    den = 1
    for value in (x, y, z):
        den = den * value.denominator // gcd(den, value.denominator)
    nums = [
        x.numerator * (den // x.denominator),
        y.numerator * (den // y.denominator),
        z.numerator * (den // z.denominator),
    ]
    common = abs(den)
    for num in nums:
        common = gcd(common, abs(num))
    if common > 1:
        nums = [num // common for num in nums]
        den = den // common
    if den < 0:
        nums = [-num for num in nums]
        den = -den
    return (nums[0], nums[1], nums[2], den)


def enumerate_seed_ratios(m):
    """Return seeds, unique ratio provenance, and raw ratio count."""
    seeds = []
    ratio_channels = {}
    raw_ratio_count = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if gcd(b, e) != 1:
                continue
            d = b + e
            a = b + 2 * e
            C = 2 * d * e
            F = a * b
            G = d * d + e * e
            seeds.append((b, e, d, a, C, F, G))
            for channel, num in (("C", C), ("F", F)):
                ratio = Fraction(num, G)
                ratio_channels.setdefault(ratio, set()).add(channel)
                raw_ratio_count += 1
    return seeds, ratio_channels, raw_ratio_count


def s2_point_from_ratios(r, s):
    """Rational inverse stereographic chart packet and Fraction direction."""
    one = Fraction(1, 1)
    denom = one + r * r + s * s
    x = 2 * r / denom
    y = 2 * s / denom
    z = (one - r * r - s * s) / denom
    packet = _canonical_packet_from_fracs(x, y, z)
    return packet, (x, y, z)


def build_model(m):
    seeds, ratio_channels, raw_ratio_count = enumerate_seed_ratios(m)
    ratios = sorted(ratio_channels)
    point_by_direction = {}
    for r in ratios:
        for s in ratios:
            packet, direction = s2_point_from_ratios(r, s)
            point_by_direction.setdefault(direction, packet)

    points = [point_by_direction[key] for key in sorted(point_by_direction)]
    raw_pair_count = len(ratios) * len(ratios)
    duplicate_count = raw_pair_count - len(points)
    r_channel_collision_count = sum(
        1 for channels in ratio_channels.values()
        if "C" in channels and "F" in channels
    )

    z_pos = sum(1 for _, _, z, _ in points if z > 0)
    z_zero = sum(1 for _, _, z, _ in points if z == 0)
    z_neg = sum(1 for _, _, z, _ in points if z < 0)
    x_zero = sum(1 for x, _, _, _ in points if x == 0)
    y_zero = sum(1 for _, y, _, _ in points if y == 0)

    n_max = max(point[3] for point in points) if points else 0
    n_max_sq = n_max * n_max
    n_max_fourth = n_max_sq * n_max_sq

    true_min = None
    witness = None
    pair_count = 0
    for i in range(len(points)):
        xi, yi, zi, den_i = points[i]
        for j in range(i + 1, len(points)):
            xj, yj, zj, den_j = points[j]
            cx = yi * zj - zi * yj
            cy = zi * xj - xi * zj
            cz = xi * yj - yi * xj
            cross_norm_sq_num = cx * cx + cy * cy + cz * cz
            pair_count += 1
            if cross_norm_sq_num < 1:
                raise ValueError(
                    f"duplicate/parallel directions after dedupe at pair {i},{j}"
                )
            norm_den = den_i * den_i * den_j * den_j
            sin_sq = Fraction(cross_norm_sq_num, norm_den)
            pair_denominator_bound = Fraction(1, norm_den)
            finite_set_bound = Fraction(1, n_max_fourth)
            if sin_sq < pair_denominator_bound:
                raise ValueError(
                    f"pair denominator bound failed at pair {i},{j}: "
                    f"{sin_sq} < {pair_denominator_bound}"
                )
            if sin_sq < finite_set_bound:
                raise ValueError(
                    f"finite N_max denominator bound failed at pair {i},{j}: "
                    f"{sin_sq} < {finite_set_bound}"
                )
            dot_num = xi * xj + yi * yj + zi * zj
            dot_den = den_i * den_j
            if true_min is None or sin_sq < true_min:
                true_min = sin_sq
                witness = {
                    "i": i,
                    "j": j,
                    "p": points[i],
                    "q": points[j],
                    "dot_num": dot_num,
                    "dot_den": dot_den,
                    "cross_norm_sq_num": cross_norm_sq_num,
                    "normalized_sin_sq": sin_sq,
                }

    approx_angle = None
    if true_min is not None:
        approx_angle = math.asin(math.sqrt(float(true_min)))

    return {
        "m": m,
        "seed_count": len(seeds),
        "raw_ratio_count": raw_ratio_count,
        "unique_R_count": len(ratios),
        "raw_pair_count": raw_pair_count,
        "unique_S2_direction_count": len(points),
        "duplicate_count": duplicate_count,
        "R_channel_provenance_collision_count": r_channel_collision_count,
        "z_sign_counts": {"positive": z_pos, "zero": z_zero, "negative": z_neg},
        "coordinate_plane_counts": {"x0": x_zero, "y0": y_zero, "z0": z_zero},
        "pair_count": pair_count,
        "N_max": n_max,
        "N_max_fourth_power": n_max_fourth,
        "true_min_normalized_sin_sq": true_min,
        "observer_angle_for_true_min_sin_sq_witness": approx_angle,
        "separation_witness": witness,
        "points": points,
        "ratio_channels": ratio_channels,
    }


def _check_source(data, errors):
    src = data.get("source_attribution", "")
    if not (
        isinstance(src, str)
        and "Whittaker" in src
        and "1903" in src
        and "10.1007/BF01444290" in src
    ):
        _err(
            errors,
            "W3D_SRC",
            "source_attribution must mention Whittaker, 1903, and DOI 10.1007/BF01444290",
        )


def _check_non_claims(data, errors):
    non_claims = data.get("non_claims", [])
    if not isinstance(non_claims, list):
        _err(errors, "W3D_DECL", "non_claims must be a list")
        return
    required = [
        "Whittaker 1903 theorem",
        "Maxwell",
        "electromagnetism",
        "scalar-potential physics",
        "density",
        "equidistribution",
        "geodesy",
        "ellipsoid physics",
    ]
    blob = " | ".join(str(item) for item in non_claims)
    for term in required:
        if term not in blob:
            _err(errors, "W3D_DECL", f"non_claims missing {term!r}")


def _check_chart(data, errors):
    chart = data.get("chart", {})
    if not isinstance(chart, dict):
        _err(errors, "W3D_3", "chart must be object")
        return
    if chart.get("name") != CHART:
        _err(errors, "W3D_3", f"chart.name must be {CHART!r}")
    if chart.get("parameter_source") != "R_m x R_m":
        _err(errors, "W3D_3", "chart.parameter_source must be 'R_m x R_m'")

    overclaims = []
    for key in (
        "claims_full_sphere",
        "claims_antipodal_closure",
        "claims_sign_reflection_closure",
        "claims_density",
        "claims_equidistribution",
    ):
        if chart.get(key) is not False:
            overclaims.append(key)
    if overclaims:
        _err(errors, "W3D_3", f"chart overclaims or omits false flags: {overclaims}")


def _check_counts(data, model, errors):
    declared = data.get("expected_counts", {})
    if not isinstance(declared, dict):
        _err(errors, "W3D_2", "expected_counts must be object")
        return
    fields = [
        "seed_count",
        "raw_ratio_count",
        "unique_R_count",
        "raw_pair_count",
        "unique_S2_direction_count",
        "duplicate_count",
        "R_channel_provenance_collision_count",
    ]
    for field in fields:
        if declared.get(field) != model[field]:
            _err(
                errors,
                "W3D_2",
                f"{field} mismatch: declared={declared.get(field)!r}, actual={model[field]!r}",
            )


def _check_optional_reporting(data, model, errors):
    expected_z = data.get("z_sign_counts")
    if expected_z is not None and expected_z != model["z_sign_counts"]:
        _err(errors, "W3D_2", f"z_sign_counts mismatch: declared={expected_z!r}, actual={model['z_sign_counts']!r}")
    expected_planes = data.get("coordinate_plane_counts")
    if expected_planes is not None and expected_planes != model["coordinate_plane_counts"]:
        _err(errors, "W3D_2", f"coordinate_plane_counts mismatch: declared={expected_planes!r}, actual={model['coordinate_plane_counts']!r}")


def _check_sphere_identity(data, model, errors):
    for x, y, z, den in model["points"]:
        lhs = x * x + y * y + z * z
        rhs = den * den
        if lhs != rhs:
            _err(errors, "W3D_1", f"sphere identity failed for {(x, y, z, den)}")
            return

    witness = data.get("sphere_identity_witness")
    if witness is None:
        _err(errors, "W3D_WIT", "sphere_identity_witness required")
        return
    if not (isinstance(witness, list) and len(witness) == 4):
        _err(errors, "W3D_WIT", "sphere_identity_witness must be [x,y,z,den]")
        return
    if not all(isinstance(value, int) for value in witness):
        _err(errors, "W3D_WIT", "sphere_identity_witness values must be integers")
        return
    x, y, z, den = witness
    if x * x + y * y + z * z != den * den:
        _err(errors, "W3D_1", f"declared sphere_identity_witness fails: {witness!r}")


def _check_channel_provenance(data, model, errors):
    provenance = data.get("R_channel_provenance")
    if not isinstance(provenance, dict):
        _err(errors, "W3D_2", "R_channel_provenance metadata must be object")
        return
    if provenance.get("mode") != "pooled_R_with_labeled_C_F_channels":
        _err(errors, "W3D_2", "R_channel_provenance.mode must preserve pooled geometry plus labeled C/F channels")
    if provenance.get("collision_count") != model["R_channel_provenance_collision_count"]:
        _err(errors, "W3D_2", "R_channel_provenance.collision_count mismatch")


def _check_separation(data, model, errors):
    sep = data.get("expected_separation", {})
    if not isinstance(sep, dict):
        _err(errors, "W3D_4", "expected_separation must be object")
        return

    if sep.get("pair_count") != model["pair_count"]:
        _err(errors, "W3D_4", f"pair_count mismatch: declared={sep.get('pair_count')!r}, actual={model['pair_count']!r}")
    if sep.get("N_max") != model["N_max"]:
        _err(errors, "W3D_4", f"N_max mismatch: declared={sep.get('N_max')!r}, actual={model['N_max']!r}")
    if sep.get("N_max_fourth_power") != model["N_max_fourth_power"]:
        _err(errors, "W3D_4", "N_max_fourth_power mismatch")

    declared_min = sep.get("true_min_normalized_sin_sq")
    if declared_min is None:
        _err(errors, "W3D_4", "expected_separation.true_min_normalized_sin_sq required")
    else:
        try:
            declared_frac = _frac_from_json(declared_min)
            if declared_frac != model["true_min_normalized_sin_sq"]:
                _err(
                    errors,
                    "W3D_4",
                    f"true_min_normalized_sin_sq mismatch: declared={declared_frac}, actual={model['true_min_normalized_sin_sq']}",
                )
        except Exception as exc:
            _err(errors, "W3D_4", f"bad true_min_normalized_sin_sq: {exc}")

    claimed_bound = sep.get("claimed_min_normalized_sin_sq_bound")
    if claimed_bound is not None:
        try:
            bound = _frac_from_json(claimed_bound)
            if model["true_min_normalized_sin_sq"] < bound:
                _err(
                    errors,
                    "W3D_4",
                    f"claimed_min_normalized_sin_sq_bound overclaims: declared={bound}, actual={model['true_min_normalized_sin_sq']}",
                )
        except Exception as exc:
            _err(errors, "W3D_4", f"bad claimed_min_normalized_sin_sq_bound: {exc}")

    theorem_bound = Fraction(1, model["N_max_fourth_power"])
    if model["true_min_normalized_sin_sq"] < theorem_bound:
        _err(errors, "W3D_4", "finite denominator theorem bound failed")

    witness = model["separation_witness"]
    if witness is None:
        _err(errors, "W3D_4", "missing computed separation witness")
        return
    if witness["cross_norm_sq_num"] < 1:
        _err(errors, "W3D_4", "computed cross_norm_sq_num < 1")

    declared_witness = sep.get("witness_pair")
    if declared_witness is not None:
        if not isinstance(declared_witness, dict):
            _err(errors, "W3D_4", "witness_pair must be object")
        else:
            if declared_witness.get("p") != _packet_to_json(witness["p"]):
                _err(errors, "W3D_4", "witness_pair.p mismatch")
            if declared_witness.get("q") != _packet_to_json(witness["q"]):
                _err(errors, "W3D_4", "witness_pair.q mismatch")
            if declared_witness.get("cross_norm_sq_num") != witness["cross_norm_sq_num"]:
                _err(errors, "W3D_4", "witness_pair.cross_norm_sq_num mismatch")


def _check_noncanonical_duplicate(data, errors):
    dup = data.get("noncanonical_duplicate_direction")
    if dup is None:
        return
    if not isinstance(dup, dict):
        _err(errors, "W3D_2", "noncanonical_duplicate_direction must be object")
        return
    p = dup.get("p")
    q = dup.get("q")
    if not (
        isinstance(p, list)
        and isinstance(q, list)
        and len(p) == 4
        and len(q) == 4
        and all(isinstance(value, int) for value in p + q)
    ):
        _err(errors, "W3D_2", "noncanonical duplicate packets must be integer [x,y,z,den]")
        return
    px, py, pz, pd = p
    qx, qy, qz, qd = q
    same = (
        Fraction(px, pd) == Fraction(qx, qd)
        and Fraction(py, pd) == Fraction(qy, qd)
        and Fraction(pz, pd) == Fraction(qz, qd)
    )
    if same and p != q:
        _err(errors, "W3D_2", "noncanonical duplicate direction must be canonicalized/deduplicated")


def validate(path):
    errors = []
    warnings = []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"W3D_1: failed to read/parse {path}: {exc}"], warnings

    if data.get("schema_version") != SCHEMA_VERSION:
        _err(errors, "W3D_1", f"schema_version must be {SCHEMA_VERSION!r}")
        return errors, warnings
    if data.get("cert_slug") != CERT_SLUG:
        _err(errors, "W3D_DECL", f"cert_slug must be {CERT_SLUG!r}")
    if data.get("candidate_family_id") != CANDIDATE_FAMILY_ID:
        _err(errors, "W3D_DECL", f"candidate_family_id must be {CANDIDATE_FAMILY_ID}")

    m = data.get("m")
    if not isinstance(m, int) or m not in ALLOWED_M:
        _err(errors, "W3D_DECL", f"m must be one of {sorted(ALLOWED_M)}")
        return errors, warnings

    _check_source(data, errors)
    _check_non_claims(data, errors)
    _check_chart(data, errors)

    try:
        model = build_model(m)
    except Exception as exc:
        _err(errors, "W3D_BUILD", f"model construction failed: {exc}")
        return errors, warnings

    _check_counts(data, model, errors)
    _check_optional_reporting(data, model, errors)
    _check_sphere_identity(data, model, errors)
    _check_channel_provenance(data, model, errors)
    _check_separation(data, model, errors)
    _check_noncanonical_duplicate(data, errors)

    fl = data.get("fail_ledger")
    if fl is not None:
        if not isinstance(fl, dict):
            _err(errors, "W3D_F", "fail_ledger must be object")
        else:
            if not isinstance(fl.get("expected_failure_codes"), list):
                _err(errors, "W3D_F", "fail_ledger.expected_failure_codes must be list")
            if not isinstance(fl.get("rationale"), str):
                _err(errors, "W3D_F", "fail_ledger.rationale must be string")

    return errors, warnings


_FIXTURES_EXPECTED = [
    ("pass_s2_m3_exact_sphere.json", True),
    ("pass_s2_m5_duplicate_accounting.json", True),
    ("pass_s2_m9_separation_theorem.json", True),
    ("fail_s2_bad_sphere_identity.json", False),
    ("fail_s2_wrong_duplicate_count.json", False),
    ("fail_s2_overclaimed_full_sphere.json", False),
    ("fail_s2_false_separation_bound.json", False),
    ("fail_s2_noncanonical_duplicate_direction.json", False),
]


def _self_test():
    fixtures_dir = Path(__file__).parent / "fixtures"
    results = []
    all_ok = True
    for fname, should_pass in _FIXTURES_EXPECTED:
        fpath = fixtures_dir / fname
        if not fpath.exists():
            results.append({"fixture": fname, "ok": False, "error": "file not found"})
            all_ok = False
            continue
        try:
            errors, warnings = validate(fpath)
            passed = not errors
        except Exception as exc:
            results.append({"fixture": fname, "ok": False, "error": str(exc)})
            all_ok = False
            continue
        if should_pass and not passed:
            results.append({
                "fixture": fname,
                "ok": False,
                "error": f"expected PASS but got {errors}",
                "warnings": warnings,
            })
            all_ok = False
        elif not should_pass and passed:
            results.append({
                "fixture": fname,
                "ok": False,
                "error": "expected FAIL but got PASS",
                "warnings": warnings,
            })
            all_ok = False
        else:
            results.append({
                "fixture": fname,
                "ok": True,
                "errors_seen": errors if not should_pass else [],
            })
    return {"ok": all_ok, "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="QA Whittaker Rational Direction S2 Cert validator"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.self_test:
        result = _self_test()
        print(json.dumps(result, sort_keys=True, indent=2))
        sys.exit(0 if result["ok"] else 1)

    paths = args.paths or sorted((Path(__file__).parent / "fixtures").glob("*.json"))
    total_errors = 0
    for path in paths:
        path = Path(path)
        errors, warnings = validate(path)
        print(f"Validating {path.name}...")
        for warning in warnings:
            print(f"  WARN: {warning}")
        for error in errors:
            print(f"  FAIL: {error}")
        if errors:
            total_errors += len(errors)
        else:
            print("  PASS")
    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    print("\nAll fixtures validated.")


if __name__ == "__main__":
    main()
