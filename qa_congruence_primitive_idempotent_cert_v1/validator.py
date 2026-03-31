#!/usr/bin/env python3
"""
QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1

Backend-relative primitivity for central idempotents in Z(Q[PSL2(Z/72Z)]).

Usage:
  - Validate a certificate:
      python qa_congruence_primitive_idempotent_cert_v1/validator.py <cert.json>
  - Emit examples into examples/:
      python qa_congruence_primitive_idempotent_cert_v1/validator.py --emit_examples [--ensure_backend]
  - Demo:
      python qa_congruence_primitive_idempotent_cert_v1/validator.py --demo [--ensure_backend]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
FAMILY_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = FAMILY_ROOT / "examples"


def _read_json(path: Path) -> object:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _load_backend_validator():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import qa_congruence_center_algebra_cert_v1.validator as backend_val  # type: ignore

    return backend_val


def _load_engine():
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import qa_psl2_mod72_center_algebra as eng  # type: ignore

    return eng


def _parse_sparse_vec(num_orbits: int, sparse: object) -> Tuple[List[Fraction], Optional[dict]]:
    if not isinstance(sparse, list):
        return [], {"reason": "sparse vector must be a list"}
    out = [Fraction(0, 1) for _ in range(num_orbits)]
    seen = set()
    for triple in sparse:
        if not (isinstance(triple, list) and len(triple) == 3):
            return [], {"reason": "each sparse entry must be [orbit_index,num,den]"}
        idx = int(triple[0])
        num = int(triple[1])
        den = int(triple[2])
        if den == 0:
            return [], {"reason": "denominator must be nonzero", "orbit_index": idx}
        if den < 0:
            num, den = -num, -den
        if not (0 <= idx < num_orbits):
            return [], {"reason": "orbit_index out of range", "orbit_index": idx}
        if idx in seen:
            return [], {"reason": "duplicate orbit_index in sparse vector", "orbit_index": idx}
        seen.add(idx)
        frac = Fraction(num, den)
        if frac.numerator != num or frac.denominator != den:
            return [], {
                "reason": "rational must be normalized (gcd=1, den>0)",
                "orbit_index": idx,
                "given": [num, den],
                "normalized": [frac.numerator, frac.denominator],
            }
        out[idx] = frac
    return out, None


def _sparse_from_vec(vec: Sequence[Fraction]) -> List[List[int]]:
    out: List[List[int]] = []
    for i, x in enumerate(vec):
        if x:
            out.append([i, x.numerator, x.denominator])
    return out


def _is_zero(vec: Sequence[Fraction]) -> bool:
    return all(x == 0 for x in vec)


def _vec_eq(a: Sequence[Fraction], b: Sequence[Fraction]) -> bool:
    return len(a) == len(b) and all(x == y for (x, y) in zip(a, b))


def _vec_sub(a: Sequence[Fraction], b: Sequence[Fraction]) -> List[Fraction]:
    return [x - y for (x, y) in zip(a, b)]


def _primitivity_fingerprint(dep_sha: str, e_sparse: object, scope: object) -> str:
    payload = {"dependency_sha256": dep_sha, "e": e_sparse, "scope": scope}
    return _sha256_json(payload)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    fail_type: Optional[str] = None
    message: str = ""
    witness: Optional[dict] = None


def _ok(message: str = "") -> ValidationResult:
    return ValidationResult(ok=True, message=message)


def _fail(fail_type: str, message: str, witness: Optional[dict] = None) -> ValidationResult:
    return ValidationResult(ok=False, fail_type=fail_type, message=message, witness=witness)


def validate_cert(path: Path) -> ValidationResult:
    cert = _read_json(path)
    if not isinstance(cert, dict):
        return _fail("SCHEMA_INVALID", "certificate must be a JSON object")
    if cert.get("schema_id") != "QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1":
        return _fail("SCHEMA_INVALID", f"schema_id mismatch: {cert.get('schema_id')!r}")

    status = cert.get("status")
    if status not in ("pass_exact", "fail_witness"):
        return _fail("SCHEMA_INVALID", f"unsupported status: {status!r}")

    fail_type = cert.get("fail_type") if status == "fail_witness" else None

    modulus_n = int(cert.get("modulus_n", -1))
    if modulus_n != 72:
        return _fail("SCHEMA_INVALID", f"modulus_n must be 72, got {modulus_n}")

    basis = str(cert.get("basis", ""))
    if basis != "psl2_mod72_orbit_basis":
        return _fail("SCHEMA_INVALID", f"unsupported basis: {basis!r}")

    dep = cert.get("dependency")
    if not isinstance(dep, dict):
        return _fail("SCHEMA_INVALID", "dependency must be an object")

    dep_family = dep.get("family")
    dep_profile_req = dep.get("profile_required")
    dep_path_str = dep.get("cert_path")
    dep_sha = dep.get("cert_sha256")
    if dep_family != "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1":
        return _fail("SCHEMA_INVALID", "dependency.family must be QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1")
    if dep_profile_req != "full_backend":
        return _fail("SCHEMA_INVALID", "dependency.profile_required must be full_backend")
    if not isinstance(dep_path_str, str) or not isinstance(dep_sha, str):
        return _fail("SCHEMA_INVALID", "dependency.cert_path and dependency.cert_sha256 must be strings")

    dep_path = _resolve_repo_path(dep_path_str)
    if not dep_path.exists():
        return _fail("SCHEMA_INVALID", f"dependency cert not found: {dep_path}")

    actual_dep_sha = _sha256_file(dep_path)
    if actual_dep_sha != dep_sha:
        if status == "fail_witness" and fail_type == "BACKEND_CERT_HASH_MISMATCH":
            w = cert.get("witness")
            if isinstance(w, dict) and w.get("expected") == dep_sha and w.get("actual") == actual_dep_sha:
                return _ok("verified BACKEND_CERT_HASH_MISMATCH witness")
        return _fail("BACKEND_CERT_HASH_MISMATCH", "dependency cert hash mismatch", {"expected": dep_sha, "actual": actual_dep_sha})

    backend_val = _load_backend_validator()
    backend_res = backend_val.validate_cert(dep_path)
    if not backend_res.ok:
        return _fail("BACKEND_CERT_INVALID", "dependency backend cert did not validate", {"backend_fail_type": backend_res.fail_type})

    backend_cert = _read_json(dep_path)
    backend_profile = str(backend_cert.get("profile", ""))
    if backend_profile != "full_backend":
        if status == "fail_witness" and fail_type == "DEPENDENCY_PROFILE_MISMATCH":
            w = cert.get("witness")
            if isinstance(w, dict) and w.get("actual_profile") == backend_profile and w.get("required_profile") == "full_backend":
                return _ok("verified DEPENDENCY_PROFILE_MISMATCH witness")
        return _fail("DEPENDENCY_PROFILE_MISMATCH", "dependency backend cert must have profile=full_backend", {"actual_profile": backend_profile})

    cache_files = backend_cert.get("cache_files")
    if not isinstance(cache_files, dict):
        return _fail("BACKEND_CERT_INVALID", "dependency backend cert missing cache_files")
    needed = ["sl2_mod8", "sl2_mod9", "psl2_mod72_orbits"]
    for k in needed:
        if k not in cache_files:
            return _fail("BACKEND_CERT_INVALID", f"dependency cache_files missing: {k}")

    cache_paths = {k: _resolve_repo_path(str(cache_files[k])) for k in needed}
    for k, p in cache_paths.items():
        if not p.exists():
            return _fail("BACKEND_CERT_INVALID", f"backend cache file missing: {k} -> {p}")

    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(cache_paths["sl2_mod8"]))
    cache9 = eng.FactorCenterCache.from_json(_read_json(cache_paths["sl2_mod9"]))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(cache_paths["psl2_mod72_orbits"]))
    if orbit_cache.num_orbits != 375:
        return _fail("BACKEND_CERT_INVALID", f"unexpected orbit count: {orbit_cache.num_orbits}")

    e_sparse = cert.get("e")
    e, err = _parse_sparse_vec(orbit_cache.num_orbits, e_sparse)
    if err is not None:
        return _fail("WELL_FORMEDNESS_FAILURE", "invalid e sparse vector", err)

    def mul(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)

    ee = mul(e, e)
    if ee != e:
        # Provide witness coordinate.
        for k in range(orbit_cache.num_orbits):
            if ee[k] != e[k]:
                return _fail(
                    "E_NOT_IDEMPOTENT",
                    "e^2 != e",
                    {"mismatch_orbit_index": k, "expected_at": [e[k].numerator, e[k].denominator], "got_at": [ee[k].numerator, ee[k].denominator]},
                )
        return _fail("E_NOT_IDEMPOTENT", "e^2 != e")

    if status == "fail_witness":
        w = cert.get("witness")
        if not isinstance(w, dict):
            return _fail("SCHEMA_INVALID", "fail_witness requires witness object")
        a_sparse = w.get("a")
        b_sparse = w.get("b")
        a, err = _parse_sparse_vec(orbit_cache.num_orbits, a_sparse)
        if err is not None:
            return _fail("WELL_FORMEDNESS_FAILURE", "invalid witness a sparse vector", err)
        b, err = _parse_sparse_vec(orbit_cache.num_orbits, b_sparse)
        if err is not None:
            return _fail("WELL_FORMEDNESS_FAILURE", "invalid witness b sparse vector", err)

        if _is_zero(a) or _is_zero(b):
            return _fail("REFINEMENT_WITNESS_INVALID", "a and b must be nonzero")

        if _vec_eq(_vec_sub(e, a), b) is False or _vec_eq(_vec_sub(e, b), a) is False:
            return _fail("REFINEMENT_WITNESS_INVALID", "a+b must equal e")

        aa = mul(a, a)
        bb = mul(b, b)
        if aa != a:
            return _fail("REFINEMENT_WITNESS_INVALID", "a is not idempotent")
        if bb != b:
            return _fail("REFINEMENT_WITNESS_INVALID", "b is not idempotent")

        ab = mul(a, b)
        ba = mul(b, a)
        if any(x != 0 for x in ab) or any(x != 0 for x in ba):
            return _fail("REFINEMENT_WITNESS_INVALID", "a and b are not orthogonal (ab!=0 or ba!=0)")

        return _ok("verified refinement witness (non-primitive)")

    # pass_exact: certify primitivity relative to a finite scope for a.
    scope = cert.get("primitivity_scope")
    if not isinstance(scope, dict):
        return _fail("SCHEMA_INVALID", "pass_exact requires primitivity_scope object")
    if scope.get("kind") != "finite_a_search":
        return _fail("SCHEMA_INVALID", "unsupported primitivity_scope.kind (expected finite_a_search)")

    allowed_idxs = scope.get("allowed_orbit_indices")
    if not (isinstance(allowed_idxs, list) and all(isinstance(x, int) for x in allowed_idxs)):
        return _fail("WELL_FORMEDNESS_FAILURE", "allowed_orbit_indices must be a list of ints")
    allowed_idxs = list(dict.fromkeys(int(x) for x in allowed_idxs))
    if any((x < 0 or x >= orbit_cache.num_orbits) for x in allowed_idxs):
        return _fail("WELL_FORMEDNESS_FAILURE", "allowed_orbit_indices contains out-of-range index")

    max_support = int(scope.get("max_support", -1))
    if max_support <= 0:
        return _fail("WELL_FORMEDNESS_FAILURE", "max_support must be > 0")

    allowed_coeffs_raw = scope.get("allowed_coeffs")
    if not (isinstance(allowed_coeffs_raw, list) and all(isinstance(x, list) and len(x) == 2 for x in allowed_coeffs_raw)):
        return _fail("WELL_FORMEDNESS_FAILURE", "allowed_coeffs must be a list of [num,den]")
    allowed_coeffs: List[Fraction] = []
    for pair in allowed_coeffs_raw:
        num = int(pair[0])
        den = int(pair[1])
        if den == 0:
            return _fail("WELL_FORMEDNESS_FAILURE", "allowed_coeffs contains zero denominator")
        if den < 0:
            num, den = -num, -den
        frac = Fraction(num, den)
        if frac.numerator != num or frac.denominator != den:
            return _fail("WELL_FORMEDNESS_FAILURE", "allowed_coeff must be normalized", {"given": [num, den], "normalized": [frac.numerator, frac.denominator]})
        allowed_coeffs.append(frac)
    # Remove duplicates while preserving order.
    seen = set()
    allowed_coeffs = [c for c in allowed_coeffs if not (c in seen or seen.add(c))]
    # For refinement, zero coefficient is pointless; ignore if present.
    allowed_coeffs = [c for c in allowed_coeffs if c != 0]
    if not allowed_coeffs:
        return _fail("WELL_FORMEDNESS_FAILURE", "allowed_coeffs must include a nonzero coefficient")

    # Exhaustive enumeration: choose support subset S (1..max_support), assign coeffs from allowed_coeffs.
    # Lex order: by support tuple, then coeff tuple.
    limit_checked = 0
    for s in range(1, min(max_support, len(allowed_idxs)) + 1):
        for support in itertools.combinations(allowed_idxs, s):
            for coeff_tuple in itertools.product(allowed_coeffs, repeat=s):
                a = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
                for idx, coeff in zip(support, coeff_tuple):
                    a[idx] = coeff
                limit_checked += 1

                if _is_zero(a) or _vec_eq(a, e):
                    continue
                aa = mul(a, a)
                if aa != a:
                    continue
                # Require a <= e in the idempotent sense: a e = a.
                ae = mul(a, e)
                if ae != a:
                    continue
                # Found a nontrivial refinement witness.
                b = _vec_sub(e, a)
                if _is_zero(b):
                    continue
                bb = mul(b, b)
                if bb != b:
                    continue
                ab = mul(a, b)
                if any(x != 0 for x in ab):
                    continue
                return _fail(
                    "PRIMITIVITY_VIOLATION",
                    "found a nontrivial refinement within the declared scope",
                    {"a": _sparse_from_vec(a), "b": _sparse_from_vec(b), "checked": limit_checked},
                )

    expected_fp = cert.get("primitivity_fingerprint_sha256")
    if not isinstance(expected_fp, str):
        return _fail("SCHEMA_INVALID", "primitivity_fingerprint_sha256 is required for pass_exact")
    actual_fp = _primitivity_fingerprint(dep_sha=dep_sha, e_sparse=e_sparse, scope=scope)
    if actual_fp != expected_fp:
        return _fail(
            "PRIMITIVITY_FINGERPRINT_MISMATCH",
            "primitivity fingerprint mismatch",
            {"expected": expected_fp, "actual": actual_fp},
        )

    return _ok(f"primitive in declared scope (checked {limit_checked} candidates)")


def _ensure_backend_pass_cert(ensure_backend: bool) -> Path:
    backend_examples = REPO_ROOT / "qa_congruence_center_algebra_cert_v1" / "examples"
    backend_pass = backend_examples / "PASS_center_algebra_full_build.json"
    if backend_pass.exists():
        return backend_pass
    if not ensure_backend:
        raise SystemExit("missing backend PASS_center_algebra_full_build.json (run backend validator --emit_examples or pass --ensure_backend)")
    backend_val = _load_backend_validator()
    eng = _load_engine()
    eng.build()
    backend_val.emit_examples()
    if not backend_pass.exists():
        raise RuntimeError("failed to ensure backend PASS_center_algebra_full_build.json")
    return backend_pass


def _subgroup_average_N9_sparse(backend_pass: Path) -> Tuple[int, List[List[int]]]:
    """
    Build the central idempotent u_N9 = (1/|N9|) Σ_{g in N9} g, where
    N9 corresponds to the normal subgroup with first CRT factor in {±I8}.

    Returns:
      (unit_orbit_idx, sparse vector for u_N9).
    """
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))

    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    id8 = cache8.identity_class
    neg_id8 = cache8.neg_identity_class
    want = {id8, neg_id8}

    orbits_in_N9 = []
    for o in orbit_cache.orbits:
        if {int(o.i8), int(o.neg_i8)} == want:
            orbits_in_N9.append(int(o.orbit_index))
    orbits_in_N9.sort()

    # N9 is isomorphic to SL2(Z/9Z), order 648.
    coeff = Fraction(1, 648)
    sparse = [[idx, coeff.numerator, coeff.denominator] for idx in orbits_in_N9]
    return unit_idx, sparse


def _emit_fail_refinable_identity(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    unit_idx, a_sparse = _subgroup_average_N9_sparse(backend_pass)
    # e = 1 (identity in group algebra)
    e_sparse = [[unit_idx, 1, 1]]
    # b = 1 - a
    # store b explicitly as sparse: start with 1 at unit and subtract 1/648 at each N9 orbit.
    a_dict = {idx: Fraction(num, den) for (idx, num, den) in [(t[0], t[1], t[2]) for t in a_sparse]}
    b_dict: Dict[int, Fraction] = {unit_idx: Fraction(1, 1)}
    for idx, coeff in a_dict.items():
        b_dict[idx] = b_dict.get(idx, Fraction(0, 1)) - coeff
    b_sparse = [[idx, v.numerator, v.denominator] for idx, v in sorted(b_dict.items()) if v]

    return {
        "schema_id": "QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1",
        "status": "fail_witness",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "e": e_sparse,
        "witness": {"a": a_sparse, "b": b_sparse},
    }


def _emit_pass_scope_demo(backend_pass: Path) -> dict:
    """
    PASS example: certify u_N9 is primitive relative to a tiny finite refinement scope
    (searches for very small-support sub-idempotents).
    """
    dep_sha = _sha256_file(backend_pass)
    unit_idx, e_sparse = _subgroup_average_N9_sparse(backend_pass)

    # Restrict candidate 'a' to a small set of orbit indices (smallest N9 orbits) and coeff 1/648.
    allowed_orbits = [t[0] for t in e_sparse[:6]]  # six smallest orbit indices in N9
    scope = {
        "kind": "finite_a_search",
        "allowed_orbit_indices": allowed_orbits,
        "max_support": 2,
        "allowed_coeffs": [[1, 648]],
    }
    fp = _primitivity_fingerprint(dep_sha=dep_sha, e_sparse=e_sparse, scope=scope)
    return {
        "schema_id": "QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1",
        "status": "pass_exact",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "e": e_sparse,
        "primitivity_scope": scope,
        "primitivity_fingerprint_sha256": fp,
    }


def _emit_fail_dependency_unit_law_only() -> dict:
    dep_path = REPO_ROOT / "qa_congruence_center_algebra_cert_v1" / "examples" / "PASS_unit_law.json"
    dep_sha = _sha256_file(dep_path)
    # e = 1 at unit orbit 0 is not correct; this cert should fail on dependency before parsing meaningfully.
    return {
        "schema_id": "QA_CONGRUENCE_PRIMITIVE_IDEMPOTENT_CERT.v1",
        "status": "fail_witness",
        "fail_type": "DEPENDENCY_PROFILE_MISMATCH",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(dep_path.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "e": [],
        "witness": {"actual_profile": "unit_law_only", "required_profile": "full_backend"},
    }


def emit_examples(ensure_backend: bool) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    backend_pass = _ensure_backend_pass_cert(ensure_backend=ensure_backend)

    _write_json(EXAMPLES_DIR / "PASS_primitive_in_scope_uN9.json", _emit_pass_scope_demo(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_refinable_identity_via_uN9.json", _emit_fail_refinable_identity(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_dependency_unit_law_only.json", _emit_fail_dependency_unit_law_only())
    print(f"Wrote examples to: {EXAMPLES_DIR}")


def demo(ensure_backend: bool) -> int:
    emit_examples(ensure_backend=ensure_backend)
    paths = [
        EXAMPLES_DIR / "PASS_primitive_in_scope_uN9.json",
        EXAMPLES_DIR / "FAIL_refinable_identity_via_uN9.json",
        EXAMPLES_DIR / "FAIL_dependency_unit_law_only.json",
    ]
    ok = True
    for p in paths:
        res = validate_cert(p)
        if res.ok:
            print(f"✓ OK: {p} ({res.message})")
        else:
            ok = False
            print(f"✗ FAIL: {p} ({res.fail_type}) {res.message}")
            if res.witness:
                print(json.dumps(res.witness, indent=2, sort_keys=True))
    return 0 if ok else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cert", nargs="?", default=None)
    parser.add_argument("--emit_examples", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--ensure_backend", action="store_true")
    args = parser.parse_args(argv)

    if args.emit_examples:
        emit_examples(ensure_backend=args.ensure_backend)
        return 0
    if args.demo:
        return demo(ensure_backend=args.ensure_backend)
    if not args.cert:
        raise SystemExit("usage: validator.py <cert.json> | --emit_examples | --demo")

    res = validate_cert(Path(args.cert))
    if res.ok:
        print(f"✓ OK: {args.cert} ({res.message})")
        return 0
    print(f"✗ FAIL: {args.cert} ({res.fail_type}) {res.message}")
    if res.witness:
        print(json.dumps(res.witness, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
