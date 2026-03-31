#!/usr/bin/env python3
"""
QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1

Validates packet decompositions of central idempotents in Z(Q[PSL2(Z/72Z)]),
strictly relative to a certified backend:
  QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1 with profile=full_backend.

Usage:
  - Validate a certificate:
      python qa_congruence_character_packet_cert_v1/validator.py <cert.json>
  - Emit examples into examples/ (requires backend cert to exist; see --ensure_backend):
      python qa_congruence_character_packet_cert_v1/validator.py --emit_examples [--ensure_backend]
  - Demo:
      python qa_congruence_character_packet_cert_v1/validator.py --demo [--ensure_backend]
"""

from __future__ import annotations

import argparse
import hashlib
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
        # Require normalized form in input: Fraction will reduce; ensure it matches.
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


def _packet_fingerprint(dep_sha256: str, basis: str, packet_sparse: object, target: object = None) -> str:
    payload = {"dependency_sha256": dep_sha256, "basis": basis, "packet": packet_sparse, "target": target}
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

    if cert.get("schema_id") != "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1":
        return _fail("SCHEMA_INVALID", f"schema_id mismatch: {cert.get('schema_id')!r}")

    status = cert.get("status")
    if status not in ("pass_exact", "fail_witness"):
        return _fail("SCHEMA_INVALID", f"unsupported status: {status!r}")

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
        if status == "fail_witness" and cert.get("fail_type") == "BACKEND_CERT_HASH_MISMATCH":
            w = cert.get("witness")
            if isinstance(w, dict) and w.get("expected") == dep_sha and w.get("actual") == actual_dep_sha:
                return _ok("verified BACKEND_CERT_HASH_MISMATCH witness")
        return _fail(
            "BACKEND_CERT_HASH_MISMATCH",
            "dependency cert hash mismatch",
            {"expected": dep_sha, "actual": actual_dep_sha, "dependency_path": dep_path_str},
        )

    backend_val = _load_backend_validator()
    backend_res = backend_val.validate_cert(dep_path)
    if not backend_res.ok:
        return _fail("BACKEND_CERT_INVALID", "dependency backend cert did not validate", {"backend_fail_type": backend_res.fail_type})

    backend_cert = _read_json(dep_path)
    backend_profile = str(backend_cert.get("profile", ""))
    if backend_profile != "full_backend":
        if status == "fail_witness" and cert.get("fail_type") == "DEPENDENCY_PROFILE_MISMATCH":
            w = cert.get("witness")
            if isinstance(w, dict) and w.get("actual_profile") == backend_profile and w.get("required_profile") == "full_backend":
                return _ok("verified DEPENDENCY_PROFILE_MISMATCH witness")
        return _fail(
            "DEPENDENCY_PROFILE_MISMATCH",
            "dependency backend cert must have profile=full_backend",
            {"actual_profile": backend_profile, "required_profile": "full_backend"},
        )

    if backend_cert.get("schema_id") != "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1" or backend_cert.get("status") != "pass_exact":
        return _fail("BACKEND_CERT_INVALID", "dependency backend cert must be pass_exact QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1")

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

    # Locate identity orbit index (unit).
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]
    unit = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    unit[unit_idx] = Fraction(1, 1)

    packet = cert.get("packet")
    if not isinstance(packet, list):
        return _fail("SCHEMA_INVALID", "packet must be a list")

    elems: List[List[Fraction]] = []
    sparse_packet: List[List[List[int]]] = []
    for e_idx, eobj in enumerate(packet):
        if not isinstance(eobj, dict):
            return _fail("PACKET_WELL_FORMEDNESS_FAILURE", "packet entries must be objects", {"packet_index": e_idx})
        sparse = eobj.get("sparse")
        vec, err = _parse_sparse_vec(orbit_cache.num_orbits, sparse)
        if err is not None:
            err["packet_index"] = e_idx
            return _fail("PACKET_WELL_FORMEDNESS_FAILURE", "invalid sparse vector", err)
        elems.append(vec)
        sparse_packet.append(_sparse_from_vec(vec))

    # Optional target resolution: z = Σ λ_i e_i
    target = cert.get("target_element")
    target_decomp = cert.get("target_decomposition")
    target_vec: Optional[List[Fraction]] = None
    if target is not None:
        target_vec, err = _parse_sparse_vec(orbit_cache.num_orbits, target)
        if err is not None:
            return _fail("PACKET_WELL_FORMEDNESS_FAILURE", "invalid target_element sparse vector", err)
        if not (isinstance(target_decomp, list) and len(target_decomp) == len(elems)):
            return _fail("SCHEMA_INVALID", "target_decomposition must be a list of length len(packet)")
        lambdas: List[Fraction] = []
        for i, lam in enumerate(target_decomp):
            if not (isinstance(lam, list) and len(lam) == 2):
                return _fail("SCHEMA_INVALID", "each target_decomposition entry must be [num,den]", {"index": i})
            num = int(lam[0])
            den = int(lam[1])
            if den == 0:
                return _fail("SCHEMA_INVALID", "target lambda denominator must be nonzero", {"index": i})
            if den < 0:
                num, den = -num, -den
            frac = Fraction(num, den)
            if frac.numerator != num or frac.denominator != den:
                return _fail("PACKET_WELL_FORMEDNESS_FAILURE", "target lambda must be normalized", {"index": i, "given": [num, den], "normalized": [frac.numerator, frac.denominator]})
            lambdas.append(frac)

    def mul(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)

    # Fail-witness handling: verify the claimed failure with the stored witness.
    if status == "fail_witness":
        fail_type = cert.get("fail_type")
        w = cert.get("witness")
        if not isinstance(fail_type, str) or not isinstance(w, dict):
            return _fail("SCHEMA_INVALID", "fail_witness requires fail_type (str) and witness (object)")

        if fail_type == "IDEMPOTENCE_FAILURE":
            i = int(w.get("element_index", -1))
            k = int(w.get("mismatch_orbit_index", -1))
            if not (0 <= i < len(elems)) or not (0 <= k < orbit_cache.num_orbits):
                return _fail("SCHEMA_INVALID", "bad witness indices for IDEMPOTENCE_FAILURE", w)
            sq = mul(elems[i], elems[i])
            if sq == elems[i]:
                return _fail("FALSE_FAIL_WITNESS", "witness claims non-idempotent but element is idempotent", w)
            exp = Fraction(int(w["expected_at"][0]), int(w["expected_at"][1]))
            got = Fraction(int(w["got_at"][0]), int(w["got_at"][1]))
            if elems[i][k] != exp or sq[k] != got or exp == got:
                return _fail("FALSE_FAIL_WITNESS", "witness does not match recomputation", w)
            return _ok("verified IDEMPOTENCE_FAILURE witness")

        if fail_type == "ORTHOGONALITY_FAILURE":
            i = int(w.get("i", -1))
            j = int(w.get("j", -1))
            k = int(w.get("mismatch_orbit_index", -1))
            if not (0 <= i < len(elems)) or not (0 <= j < len(elems)) or i == j or not (0 <= k < orbit_cache.num_orbits):
                return _fail("SCHEMA_INVALID", "bad witness indices for ORTHOGONALITY_FAILURE", w)
            prod = mul(elems[i], elems[j])
            if all(x == 0 for x in prod):
                return _fail("FALSE_FAIL_WITNESS", "witness claims non-orthogonal but product is zero", w)
            got = Fraction(int(w["got_at"][0]), int(w["got_at"][1]))
            if prod[k] != got or got == 0:
                return _fail("FALSE_FAIL_WITNESS", "witness does not match recomputation at mismatch index", w)
            return _ok("verified ORTHOGONALITY_FAILURE witness")

        if fail_type == "SUM_TO_IDENTITY_FAILURE":
            k = int(w.get("mismatch_orbit_index", -1))
            if not (0 <= k < orbit_cache.num_orbits):
                return _fail("SCHEMA_INVALID", "bad mismatch_orbit_index for SUM_TO_IDENTITY_FAILURE", w)
            s = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
            for e in elems:
                for idx in range(orbit_cache.num_orbits):
                    s[idx] += e[idx]
            if s == unit:
                return _fail("FALSE_FAIL_WITNESS", "witness claims sum!=1 but sum equals identity", w)
            exp = Fraction(int(w["expected_at"][0]), int(w["expected_at"][1]))
            got = Fraction(int(w["got_at"][0]), int(w["got_at"][1]))
            if unit[k] != exp or s[k] != got or exp == got:
                return _fail("FALSE_FAIL_WITNESS", "witness does not match recomputation", w)
            return _ok("verified SUM_TO_IDENTITY_FAILURE witness")

        if fail_type == "DEPENDENCY_PROFILE_MISMATCH":
            # already handled above
            return _fail("SCHEMA_INVALID", "DEPENDENCY_PROFILE_MISMATCH should have been returned earlier")

        if fail_type == "BACKEND_CERT_HASH_MISMATCH":
            return _fail("SCHEMA_INVALID", "BACKEND_CERT_HASH_MISMATCH should have been returned earlier")

        return _fail("SCHEMA_INVALID", f"unknown fail_type: {fail_type}")

    # pass_exact: run gates.
    # Gate 3: idempotence
    for i, e in enumerate(elems):
        sq = mul(e, e)
        if sq != e:
            # Provide a deterministic witness (first mismatch orbit).
            for k in range(orbit_cache.num_orbits):
                if sq[k] != e[k]:
                    return _fail(
                        "IDEMPOTENCE_FAILURE",
                        "packet element is not idempotent",
                        {
                            "element_index": i,
                            "mismatch_orbit_index": k,
                            "expected_at": [e[k].numerator, e[k].denominator],
                            "got_at": [sq[k].numerator, sq[k].denominator],
                        },
                    )
            return _fail("IDEMPOTENCE_FAILURE", "packet element is not idempotent", {"element_index": i})

    # Gate 4: orthogonality
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            prod = mul(elems[i], elems[j])
            if any(x != 0 for x in prod):
                for k in range(orbit_cache.num_orbits):
                    if prod[k] != 0:
                        return _fail(
                            "ORTHOGONALITY_FAILURE",
                            "packet elements are not orthogonal",
                            {"i": i, "j": j, "mismatch_orbit_index": k, "got_at": [prod[k].numerator, prod[k].denominator]},
                        )
                return _fail("ORTHOGONALITY_FAILURE", "packet elements are not orthogonal", {"i": i, "j": j})

    # Gate 5: sum-to-identity
    s = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    for e in elems:
        for k in range(orbit_cache.num_orbits):
            s[k] += e[k]
    if s != unit:
        for k in range(orbit_cache.num_orbits):
            if s[k] != unit[k]:
                return _fail(
                    "SUM_TO_IDENTITY_FAILURE",
                    "packet does not sum to identity",
                    {
                        "mismatch_orbit_index": k,
                        "expected_at": [unit[k].numerator, unit[k].denominator],
                        "got_at": [s[k].numerator, s[k].denominator],
                    },
                )
        return _fail("SUM_TO_IDENTITY_FAILURE", "packet does not sum to identity")

    # Gate 6: optional target resolution
    if target_vec is not None:
        lambdas = [Fraction(int(lam[0]), int(lam[1])) for lam in target_decomp]  # type: ignore[arg-type]
        rhs = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
        for lam, e in zip(lambdas, elems):
            if lam == 0:
                continue
            for k in range(orbit_cache.num_orbits):
                rhs[k] += lam * e[k]
        if rhs != target_vec:
            for k in range(orbit_cache.num_orbits):
                if rhs[k] != target_vec[k]:
                    return _fail(
                        "TARGET_RESOLUTION_FAILURE",
                        "target element resolution failed",
                        {
                            "mismatch_orbit_index": k,
                            "expected_at": [target_vec[k].numerator, target_vec[k].denominator],
                            "got_at": [rhs[k].numerator, rhs[k].denominator],
                        },
                    )
            return _fail("TARGET_RESOLUTION_FAILURE", "target element resolution failed")

    # Gate 7: fingerprint
    expected_fp = cert.get("packet_fingerprint_sha256")
    if not isinstance(expected_fp, str):
        return _fail("SCHEMA_INVALID", "packet_fingerprint_sha256 is required for pass_exact")
    actual_fp = _packet_fingerprint(dep_sha256=dep_sha, basis=basis, packet_sparse=sparse_packet, target=target)
    if actual_fp != expected_fp:
        return _fail(
            "PACKET_FINGERPRINT_MISMATCH",
            "packet fingerprint mismatch",
            {"expected": expected_fp, "actual": actual_fp},
        )

    return _ok("all gates passed")


def _ensure_backend_examples() -> Path:
    backend_examples = REPO_ROOT / "qa_congruence_center_algebra_cert_v1" / "examples"
    backend_pass = backend_examples / "PASS_center_algebra_full_build.json"
    if backend_pass.exists():
        return backend_pass

    backend_val = _load_backend_validator()
    eng = _load_engine()
    eng.build()
    backend_val.emit_examples()
    if not backend_pass.exists():
        raise RuntimeError("failed to ensure backend PASS_center_algebra_full_build.json")
    return backend_pass


def _emit_pass_identity_singleton(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    packet = [{"name": "1", "sparse": [[0, 0, 1]]}]  # placeholder; unit is located via backend, not by index here
    # Represent identity as a sparse vector with orbit index taken from backend at emit time.
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]
    packet = [{"name": "1", "sparse": [[unit_idx, 1, 1]]}]
    fp = _packet_fingerprint(dep_sha256=dep_sha, basis="psl2_mod72_orbit_basis", packet_sparse=[packet[0]["sparse"]])
    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
        "status": "pass_exact",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "packet": packet,
        "packet_fingerprint_sha256": fp,
    }


def _emit_pass_two_packet_partition(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    packet = [
        {"name": "0", "sparse": []},
        {"name": "1", "sparse": [[unit_idx, 1, 1]]},
    ]
    fp = _packet_fingerprint(dep_sha256=dep_sha, basis="psl2_mod72_orbit_basis", packet_sparse=[p["sparse"] for p in packet])
    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
        "status": "pass_exact",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "packet": packet,
        "packet_fingerprint_sha256": fp,
    }


def _emit_fail_dependency_unit_law_only() -> dict:
    dep_path = REPO_ROOT / "qa_congruence_center_algebra_cert_v1" / "examples" / "PASS_unit_law.json"
    dep_sha = _sha256_file(dep_path)
    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
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
        "packet": [{"name": "1", "sparse": []}],
        "witness": {"actual_profile": "unit_law_only", "required_profile": "full_backend"},
    }


def _emit_fail_non_idempotent(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    # Pick a deterministic non-unit orbit index and certify its non-idempotence.
    cand = [o.orbit_index for o in orbit_cache.orbits if o.orbit_index != unit_idx]
    cand.sort()
    chosen = None
    witness = None
    for idx in cand[:50]:
        e = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
        e[idx] = Fraction(1, 1)
        sq = eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, e, e)
        if sq != e:
            chosen = idx
            for k in range(orbit_cache.num_orbits):
                if sq[k] != e[k]:
                    witness = {
                        "element_index": 0,
                        "mismatch_orbit_index": k,
                        "expected_at": [e[k].numerator, e[k].denominator],
                        "got_at": [sq[k].numerator, sq[k].denominator],
                    }
                    break
            break
    if chosen is None or witness is None:
        raise RuntimeError("could not find deterministic non-idempotent basis element (unexpected)")

    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
        "status": "fail_witness",
        "fail_type": "IDEMPOTENCE_FAILURE",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "packet": [{"name": "K_orbit", "sparse": [[chosen, 1, 1]]}],
        "witness": witness,
    }


def _emit_fail_non_orthogonal(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    # e1 = 1, e2 = 1 => idempotence holds individually, orthogonality fails.
    e = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    e[unit_idx] = Fraction(1, 1)
    prod = eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, e, e)
    if prod[unit_idx] == 0:
        raise RuntimeError("unexpected: 1*1 had zero at unit coordinate")

    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
        "status": "fail_witness",
        "fail_type": "ORTHOGONALITY_FAILURE",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "packet": [
            {"name": "1a", "sparse": [[unit_idx, 1, 1]]},
            {"name": "1b", "sparse": [[unit_idx, 1, 1]]},
        ],
        "witness": {"i": 0, "j": 1, "mismatch_orbit_index": unit_idx, "got_at": [prod[unit_idx].numerator, prod[unit_idx].denominator]},
    }


def _emit_fail_incomplete_sum(backend_pass: Path) -> dict:
    dep_sha = _sha256_file(backend_pass)
    backend_cert = _read_json(backend_pass)
    cache_files = backend_cert["cache_files"]
    eng = _load_engine()
    cache8 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod8"])))
    cache9 = eng.FactorCenterCache.from_json(_read_json(_resolve_repo_path(cache_files["sl2_mod9"])))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(_resolve_repo_path(cache_files["psl2_mod72_orbits"])))
    id_pair = (cache8.identity_class, cache9.identity_class)
    unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    # packet [0] sums to 0, not identity.
    unit_at = Fraction(1, 1)
    got_at = Fraction(0, 1)
    return {
        "schema_id": "QA_CONGRUENCE_CHARACTER_PACKET_CERT.v1",
        "status": "fail_witness",
        "fail_type": "SUM_TO_IDENTITY_FAILURE",
        "modulus_n": 72,
        "basis": "psl2_mod72_orbit_basis",
        "dependency": {
            "family": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
            "profile_required": "full_backend",
            "cert_path": str(backend_pass.relative_to(REPO_ROOT)),
            "cert_sha256": dep_sha,
        },
        "packet": [{"name": "0", "sparse": []}],
        "witness": {
            "mismatch_orbit_index": unit_idx,
            "expected_at": [unit_at.numerator, unit_at.denominator],
            "got_at": [got_at.numerator, got_at.denominator],
        },
    }


def emit_examples(ensure_backend: bool) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    backend_pass = _ensure_backend_examples() if ensure_backend else (REPO_ROOT / "qa_congruence_center_algebra_cert_v1" / "examples" / "PASS_center_algebra_full_build.json")
    if not backend_pass.exists():
        raise SystemExit("missing backend PASS_center_algebra_full_build.json; run qa_congruence_center_algebra_cert_v1/validator.py --emit_examples --build_caches or pass --ensure_backend")

    _write_json(EXAMPLES_DIR / "PASS_identity_singleton_packet.json", _emit_pass_identity_singleton(backend_pass))
    _write_json(EXAMPLES_DIR / "PASS_two_packet_partition.json", _emit_pass_two_packet_partition(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_non_idempotent_packet.json", _emit_fail_non_idempotent(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_non_orthogonal_packet.json", _emit_fail_non_orthogonal(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_incomplete_sum.json", _emit_fail_incomplete_sum(backend_pass))
    _write_json(EXAMPLES_DIR / "FAIL_dependency_unit_law_only.json", _emit_fail_dependency_unit_law_only())
    print(f"Wrote examples to: {EXAMPLES_DIR}")


def demo(ensure_backend: bool) -> int:
    emit_examples(ensure_backend=ensure_backend)
    paths = [
        EXAMPLES_DIR / "PASS_identity_singleton_packet.json",
        EXAMPLES_DIR / "PASS_two_packet_partition.json",
        EXAMPLES_DIR / "FAIL_non_idempotent_packet.json",
        EXAMPLES_DIR / "FAIL_non_orthogonal_packet.json",
        EXAMPLES_DIR / "FAIL_incomplete_sum.json",
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
    parser.add_argument("--ensure_backend", action="store_true", help="If backend PASS cert is missing, build caches and emit backend examples.")
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

