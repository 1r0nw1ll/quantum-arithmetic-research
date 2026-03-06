#!/usr/bin/env python3
"""
QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1

Validates the internal CRT center-algebra backend implemented in:
  qa_psl2_mod72_center_algebra.py

Usage:
  - Validate a certificate:
      python qa_congruence_center_algebra_cert_v1/validator.py <cert.json>
  - Emit example certificates (writes into examples/):
      python qa_congruence_center_algebra_cert_v1/validator.py --emit_examples
  - Demo (validates examples/):
      python qa_congruence_center_algebra_cert_v1/validator.py --demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def _frac_to_key(x: Fraction) -> Tuple[int, int]:
    return (x.numerator, x.denominator)


def _make_sparse_vec(vec: Sequence[Fraction]) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for i, x in enumerate(vec):
        if x:
            out.append((i, x.numerator, x.denominator))
    return out


def _vec_from_sparse(n: int, sparse: Sequence[Sequence[int]]) -> List[Fraction]:
    v = [Fraction(0, 1) for _ in range(n)]
    for triple in sparse:
        if len(triple) != 3:
            raise ValueError("sparse triple must be [idx,num,den]")
        idx, num, den = (int(triple[0]), int(triple[1]), int(triple[2]))
        if not (0 <= idx < n):
            raise ValueError("sparse idx out of range")
        v[idx] = Fraction(num, den)
    return v


def _battery_orbit_indices(num_orbits: int, unit_idx: int, seed: int, count: int) -> List[int]:
    if count <= 0:
        raise ValueError("battery count must be >0")
    x = seed % num_orbits
    out: List[int] = []
    # LCG; deterministic across runs.
    while len(out) < count:
        x = (1103515245 * x + 12345) % num_orbits
        if x == unit_idx:
            continue
        if x not in out:
            out.append(x)
    return out


def _small_orbit_indices(orbit_cache, unit_idx: int, count: int, offset: int = 0) -> List[int]:
    """
    Deterministic, fast-to-multiply orbit indices: pick the smallest conjugacy classes (by size_psl),
    excluding the unit orbit.
    """
    pairs = [(int(o.size_psl), int(o.orbit_index)) for o in orbit_cache.orbits if int(o.orbit_index) != unit_idx]
    pairs.sort()
    if count <= 0:
        raise ValueError("count must be >0")
    if offset < 0:
        raise ValueError("offset must be >=0")
    if offset + count > len(pairs):
        raise ValueError("offset+count exceeds available orbits")
    return [pairs[offset + i][1] for i in range(count)]


def _fingerprint_center_algebra(
    mul_fn,
    num_orbits: int,
    unit_idx: int,
    basis_indices: Sequence[int],
) -> str:
    """
    Deterministic multiplication fingerprint:
    - choose sparse orbit vectors from a battery;
    - hash all pairwise products in fixed order.
    """
    vecs: List[List[Fraction]] = []
    # sparse basis vectors with small exact coefficients (keep this intentionally small/fast)
    coeffs = [Fraction(1, 1), Fraction(2, 1), Fraction(-1, 3)]
    for t, idx in enumerate(list(basis_indices)[: min(len(basis_indices), len(coeffs))]):
        v = [Fraction(0, 1) for _ in range(num_orbits)]
        v[idx] = coeffs[t]
        vecs.append(v)
    # no combinations here: keep products cheap and stable

    h = hashlib.sha256()
    h.update(f"num_orbits={num_orbits};unit_idx={unit_idx};basis={list(basis_indices)};".encode("utf-8"))
    for i, a in enumerate(vecs):
        # commutativity lets us hash only i<=j products deterministically
        for j, b in enumerate(vecs):
            if j < i:
                continue
            ab = mul_fn(a, b)
            h.update(f"prod({i},{j})=".encode("utf-8"))
            for k in range(num_orbits):
                x = ab[k]
                if x:
                    h.update(f"{k}:{x.numerator}/{x.denominator},".encode("utf-8"))
            h.update(b";")
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    fail_type: Optional[str] = None
    message: str = ""
    witness: Optional[dict] = None


def _fail(fail_type: str, message: str, witness: Optional[dict] = None) -> ValidationResult:
    return ValidationResult(ok=False, fail_type=fail_type, message=message, witness=witness)


def _ok(message: str = "") -> ValidationResult:
    return ValidationResult(ok=True, message=message)


def _load_engine():
    # Import from repo root.
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import qa_psl2_mod72_center_algebra as eng  # type: ignore

    return eng


def _resolve_repo_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def validate_cert(path: Path) -> ValidationResult:
    cert = _read_json(path)
    if not isinstance(cert, dict):
        return _fail("SCHEMA_INVALID", "certificate must be a JSON object")

    schema_id = cert.get("schema_id")
    if schema_id != "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1":
        return _fail("SCHEMA_INVALID", f"schema_id mismatch: {schema_id!r}")

    status = cert.get("status")
    if status not in ("pass_exact", "fail_witness"):
        return _fail("SCHEMA_INVALID", f"unsupported status: {status!r}")

    profile = str(cert.get("profile", "full_backend"))
    if profile not in ("full_backend", "unit_law_only"):
        return _fail("SCHEMA_INVALID", f"unsupported profile: {profile!r}")

    modulus_n = int(cert.get("modulus_n", -1))
    if modulus_n != 72:
        return _fail("SCHEMA_INVALID", f"modulus_n must be 72, got {modulus_n}")

    engine_path = _resolve_repo_path(str(cert.get("engine_source_path", "qa_psl2_mod72_center_algebra.py")))
    if not engine_path.exists():
        return _fail("SCHEMA_INVALID", f"engine_source_path not found: {engine_path}")
    engine_sha = _sha256_file(engine_path)

    expected_engine_sha = cert.get("engine_source_sha256")
    if status == "pass_exact" and expected_engine_sha != engine_sha:
        return _fail(
            "ENGINE_HASH_MISMATCH",
            "engine source hash mismatch",
            {"expected": expected_engine_sha, "actual": engine_sha},
        )

    cache_files = cert.get("cache_files")
    if not isinstance(cache_files, dict):
        return _fail("SCHEMA_INVALID", "cache_files must be an object")

    needed_keys = ["sl2_mod8", "sl2_mod9", "psl2_mod72_orbits"]
    for k in needed_keys:
        if k not in cache_files:
            return _fail("SCHEMA_INVALID", f"cache_files missing key: {k}")

    cache_paths = {k: _resolve_repo_path(str(cache_files[k])) for k in needed_keys}
    for k, p in cache_paths.items():
        if not p.exists():
            return _fail("SCHEMA_INVALID", f"cache file not found: {k} -> {p}")

    actual_cache_sha = {k: _sha256_file(p) for k, p in cache_paths.items()}

    expected_cache_sha = cert.get("cache_sha256")
    if status == "pass_exact":
        if not isinstance(expected_cache_sha, dict):
            return _fail("SCHEMA_INVALID", "cache_sha256 must be an object for pass_exact")
        for k in needed_keys:
            if expected_cache_sha.get(k) != actual_cache_sha[k]:
                return _fail(
                    "CACHE_HASH_MISMATCH",
                    f"cache sha256 mismatch for {k}",
                    {"cache": k, "expected": expected_cache_sha.get(k), "actual": actual_cache_sha[k]},
                )
    else:
        # fail_witness: validate the claimed mismatch witness types below
        pass

    eng = _load_engine()

    # Load factor and orbit caches as dataclasses.
    cache8 = eng.FactorCenterCache.from_json(_read_json(cache_paths["sl2_mod8"]))
    cache9 = eng.FactorCenterCache.from_json(_read_json(cache_paths["sl2_mod9"]))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(cache_paths["psl2_mod72_orbits"]))

    # A few always-on sanity checks for both statuses.
    if cache8.group_order != 384 or cache8.num_classes != 30:
        return _fail(
            "FACTOR_PARTITION_FAILURE",
            "factor cache for mod8 has unexpected order/classes",
            {"order": cache8.group_order, "num_classes": cache8.num_classes},
        )
    if cache9.group_order != 648 or cache9.num_classes != 25:
        return _fail(
            "FACTOR_PARTITION_FAILURE",
            "factor cache for mod9 has unexpected order/classes",
            {"order": cache9.group_order, "num_classes": cache9.num_classes},
        )
    if orbit_cache.num_orbits != 375:
        return _fail("ORBIT_COVERAGE_FAILURE", "orbit cache has unexpected num_orbits", {"num_orbits": orbit_cache.num_orbits})

    # Handle fail_witness certificates early.
    if status == "fail_witness":
        fail_type = cert.get("fail_type")
        witness = cert.get("witness")
        if not isinstance(fail_type, str) or not isinstance(witness, dict):
            return _fail("SCHEMA_INVALID", "fail_witness requires fail_type (str) and witness (object)")

        if fail_type == "CACHE_HASH_MISMATCH":
            cache_key = witness.get("cache")
            expected = witness.get("expected")
            actual = witness.get("actual")
            if cache_key not in needed_keys:
                return _fail("SCHEMA_INVALID", "witness.cache must be one of cache_files keys", witness=witness)
            if actual != actual_cache_sha[cache_key]:
                return _fail("FALSE_FAIL_WITNESS", "witness.actual does not match recomputed cache hash", witness=witness)
            if expected == actual:
                return _fail("FALSE_FAIL_WITNESS", "witness claims mismatch but expected==actual", witness=witness)
            return _ok(f"verified CACHE_HASH_MISMATCH for {cache_key}")

        if fail_type == "UNIT_LAW_FAILURE":
            claimed_unit = int(witness.get("claimed_unit_orbit_index", -1))
            if not (0 <= claimed_unit < orbit_cache.num_orbits):
                return _fail("SCHEMA_INVALID", "claimed_unit_orbit_index out of range", witness=witness)
            a_sparse = witness.get("a_sparse")
            if not isinstance(a_sparse, list):
                return _fail("SCHEMA_INVALID", "witness.a_sparse must be a list of [idx,num,den] triples", witness=witness)
            a = _vec_from_sparse(orbit_cache.num_orbits, a_sparse)
            e = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
            e[claimed_unit] = Fraction(1, 1)

            def mul_fn(x, y):
                return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, x, y)

            ea = mul_fn(e, a)
            ae = mul_fn(a, e)
            if ea == a and ae == a:
                return _fail("FALSE_FAIL_WITNESS", "witness claims unit law failure but unit law holds for provided a", witness=witness)

            mismatch_idx = int(witness.get("mismatch_index", -1))
            if not (0 <= mismatch_idx < orbit_cache.num_orbits):
                # If not provided, accept by locating our own mismatch.
                for i in range(orbit_cache.num_orbits):
                    if ea[i] != a[i] or ae[i] != a[i]:
                        mismatch_idx = i
                        break
            if mismatch_idx < 0:
                return _fail("FALSE_FAIL_WITNESS", "could not locate mismatch index (unexpected)", witness=witness)

            expected_at = witness.get("expected_at")
            left_at = witness.get("left_at")
            right_at = witness.get("right_at")
            if not (isinstance(expected_at, list) and isinstance(left_at, list) and isinstance(right_at, list)):
                return _fail(
                    "SCHEMA_INVALID",
                    "witness must include expected_at/left_at/right_at as [num,den] lists",
                    witness=witness,
                )
            exp_f = Fraction(int(expected_at[0]), int(expected_at[1]))
            left_f = Fraction(int(left_at[0]), int(left_at[1]))
            right_f = Fraction(int(right_at[0]), int(right_at[1]))

            if a[mismatch_idx] != exp_f:
                return _fail("FALSE_FAIL_WITNESS", "witness.expected_at does not match a at mismatch_index", witness=witness)
            if ea[mismatch_idx] != left_f:
                return _fail("FALSE_FAIL_WITNESS", "witness.left_at does not match e*a at mismatch_index", witness=witness)
            if ae[mismatch_idx] != right_f:
                return _fail("FALSE_FAIL_WITNESS", "witness.right_at does not match a*e at mismatch_index", witness=witness)

            if left_f == exp_f and right_f == exp_f:
                return _fail("FALSE_FAIL_WITNESS", "witness does not demonstrate a unit law failure at mismatch_index", witness=witness)
            return _ok("verified UNIT_LAW_FAILURE witness for claimed unit index")

        return _fail("SCHEMA_INVALID", f"unknown fail_type: {fail_type}", witness=witness)

    # pass_exact: run gates.
    # --- Gate 1/2 (full_backend only): recompute factor partitions and verify cache consistency ---
    if profile == "full_backend":
        for (n, cache) in [(8, cache8), (9, cache9)]:
            elements = eng.enumerate_sl2_mod_n(n)
            if len(elements) != cache.group_order:
                return _fail("FACTOR_PARTITION_FAILURE", f"recomputed element count mismatch for mod{n}")
            classes, elem_to_class = eng.conjugacy_classes_sl2_mod_n(elements, n)
            if len(classes) != cache.num_classes:
                return _fail("FACTOR_PARTITION_FAILURE", f"recomputed class count mismatch for mod{n}")
            if sum(len(c) for c in classes) != cache.group_order:
                return _fail("FACTOR_PARTITION_FAILURE", f"class partition size mismatch for mod{n}")
            reps = [c[0] for c in classes]
            reps_hash = _sha256_json([list(r) for r in reps])
            if reps_hash != cache.reps_hash:
                return _fail(
                    "FACTOR_PARTITION_FAILURE",
                    f"reps_hash mismatch for mod{n}",
                    {"expected": cache.reps_hash, "actual": reps_hash},
                )
            if reps != cache.class_reps:
                return _fail("FACTOR_PARTITION_FAILURE", f"class representatives mismatch for mod{n}")
            # Identity and -Identity singleton classes.
            if cache.class_reps[cache.identity_class] != eng.identity(n):
                return _fail("FACTOR_PARTITION_FAILURE", f"identity_class representative mismatch for mod{n}")
            if cache.class_reps[cache.neg_identity_class] != eng.neg_mod(eng.identity(n), n):
                return _fail("FACTOR_PARTITION_FAILURE", f"neg_identity_class representative mismatch for mod{n}")
            if cache.class_sizes[cache.identity_class] != 1 or cache.class_sizes[cache.neg_identity_class] != 1:
                return _fail("FACTOR_PARTITION_FAILURE", f"identity/-identity classes must be singletons for mod{n}")

            # Verify inv_class and neg_class maps on reps match recomputation.
            for i, rep in enumerate(cache.class_reps):
                inv_rep = eng.inv_mod(rep, n)
                neg_rep = eng.neg_mod(rep, n)
                if cache.inv_class[i] != elem_to_class[inv_rep]:
                    return _fail("FACTOR_PARTITION_FAILURE", f"inv_class mismatch at i={i} for mod{n}")
                if cache.neg_class[i] != elem_to_class[neg_rep]:
                    return _fail("FACTOR_PARTITION_FAILURE", f"neg_class mismatch at i={i} for mod{n}")

    # --- Gate 2 (full_backend only): factor structure constants sanity (unit + associativity battery) ---
    def check_factor_mult(cache, salt: int) -> Optional[ValidationResult]:
        r = cache.num_classes
        # Shape checks and integrality/nonnegativity.
        if len(cache.mult) != r:
            return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "mult outer dimension mismatch")
        for i in range(r):
            if len(cache.mult[i]) != r:
                return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "mult middle dimension mismatch", {"i": i})
            for j in range(r):
                row = cache.mult[i][j]
                if len(row) != r:
                    return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "mult inner dimension mismatch", {"i": i, "j": j})
                for k, m in enumerate(row):
                    if not isinstance(m, int) or m < 0:
                        return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "mult entry must be a nonnegative int", {"i": i, "j": j, "k": k, "m": m})

        idc = cache.identity_class
        # Unit law in class-sum basis.
        for j in range(r):
            row = cache.mult[idc][j]
            for k in range(r):
                want = 1 if k == j else 0
                if row[k] != want:
                    return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "left unit law failure", {"j": j, "k": k, "got": row[k], "want": want})
        for i in range(r):
            row = cache.mult[i][idc]
            for k in range(r):
                want = 1 if k == i else 0
                if row[k] != want:
                    return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "right unit law failure", {"i": i, "k": k, "got": row[k], "want": want})

        # Associativity battery on basis triples (i,j,k).
        triples = []
        x = salt % r
        while len(triples) < 32:
            x = (1664525 * x + 1013904223) % (2**32)
            i = x % r
            x = (1664525 * x + 1013904223) % (2**32)
            j = x % r
            x = (1664525 * x + 1013904223) % (2**32)
            k = x % r
            triples.append((i, j, k))

        for (i, j, k) in triples:
            # (K_i K_j) K_k
            left = [0] * r
            for t in range(r):
                mijt = cache.mult[i][j][t]
                if mijt:
                    for u in range(r):
                        left[u] += mijt * cache.mult[t][k][u]
            # K_i (K_j K_k)
            right = [0] * r
            for t in range(r):
                mjkt = cache.mult[j][k][t]
                if mjkt:
                    for u in range(r):
                        right[u] += mjkt * cache.mult[i][t][u]
            if left != right:
                return _fail("FACTOR_STRUCTURE_CONSTANTS_FAILURE", "associativity battery failure", {"i": i, "j": j, "k": k})
        return None

    if profile == "full_backend":
        bad = check_factor_mult(cache8, salt=8)
        if bad:
            return bad
        bad = check_factor_mult(cache9, salt=9)
        if bad:
            return bad

    # --- Gate 3: orbit quotient correctness ---
    if orbit_cache.r8 != cache8.num_classes or orbit_cache.r9 != cache9.num_classes:
        return _fail("ORBIT_COVERAGE_FAILURE", "orbit cache factor dimensions mismatch")
    if len(orbit_cache.pair_to_orbit) != orbit_cache.r8 * orbit_cache.r9:
        return _fail("ORBIT_COVERAGE_FAILURE", "pair_to_orbit length mismatch")
    seen_pairs = set()
    for o in orbit_cache.orbits:
        a = (o.i8, o.i9)
        b = (o.neg_i8, o.neg_i9)
        if a == b:
            return _fail("ORBIT_COVERAGE_FAILURE", "fixed sign-orbit found (unexpected for N=72)", {"pair": a})
        if orbit_cache.pair_to_orbit[o.i8 * orbit_cache.r9 + o.i9] != o.orbit_index:
            return _fail("ORBIT_COVERAGE_FAILURE", "pair_to_orbit mismatch for orbit rep", {"orbit_index": o.orbit_index})
        if orbit_cache.pair_to_orbit[o.neg_i8 * orbit_cache.r9 + o.neg_i9] != o.orbit_index:
            return _fail("ORBIT_COVERAGE_FAILURE", "pair_to_orbit mismatch for orbit neg rep", {"orbit_index": o.orbit_index})
        seen_pairs.add(a)
        seen_pairs.add(b)

        want_size = cache8.class_sizes[o.i8] * cache9.class_sizes[o.i9]
        if o.size_psl != want_size:
            return _fail("ORBIT_COVERAGE_FAILURE", "orbit size_psl mismatch", {"orbit_index": o.orbit_index, "got": o.size_psl, "want": want_size})

        rep72 = eng.psl_normal_form_mod_n(eng.crt_combine_mat_mod72(cache8.class_reps[o.i8], cache9.class_reps[o.i9]), 72)
        if tuple(rep72) != tuple(o.rep72):
            return _fail("ORBIT_REPRESENTATIVE_FAILURE", "orbit representative mismatch", {"orbit_index": o.orbit_index})

    if len(seen_pairs) != cache8.num_classes * cache9.num_classes:
        return _fail("ORBIT_COVERAGE_FAILURE", "orbit coverage mismatch", {"seen": len(seen_pairs), "want": cache8.num_classes * cache9.num_classes})

    # Identity orbit must exist and yield rep72 = identity(72) in PSL normal form.
    id_pair = (cache8.identity_class, cache9.identity_class)
    id_orbit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]
    if tuple(orbit_cache.orbits[id_orbit_idx].rep72) != tuple(eng.identity(72)):
        return _fail("ORBIT_REPRESENTATIVE_FAILURE", "identity orbit representative mismatch", {"id_orbit_idx": id_orbit_idx})

    # --- Gate 4: orbit center algebra checks (unit always; commutativity/associativity only for full_backend) ---
    def mul_fn(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
        return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)

    e = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    e[id_orbit_idx] = Fraction(1, 1)

    # deterministic basis vectors for checks: smallest conjugacy classes first (fast).
    battery_seed = int(cert.get("battery_seed", 0))
    battery_count = int(cert.get("battery_count", 4))
    # battery_seed selects an offset into the sorted-small-orbit list; keep it small by default.
    offset = int(cert.get("battery_small_orbit_offset", battery_seed))
    idxs = _small_orbit_indices(orbit_cache, unit_idx=id_orbit_idx, count=max(3, battery_count), offset=offset)
    vecs: List[List[Fraction]] = []
    for t, idx in enumerate(idxs[:battery_count]):
        v = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
        v[idx] = Fraction(t + 1, 1)
        vecs.append(v)

    # Unit law.
    for t, a in enumerate(vecs):
        if mul_fn(e, a) != a or mul_fn(a, e) != a:
            return _fail("UNIT_LAW_FAILURE", "unit law failure on deterministic battery", {"case": t})

    if profile == "full_backend":
        # Commutativity (small deterministic subset).
        comm_pairs = [(0, 1), (1, 2), (0, min(3, len(vecs) - 1))]
        for (i, j) in comm_pairs:
            if i == j or i >= len(vecs) or j >= len(vecs):
                continue
            if mul_fn(vecs[i], vecs[j]) != mul_fn(vecs[j], vecs[i]):
                return _fail("COMMUTATIVITY_FAILURE", "commutativity failure on deterministic subset", {"i": i, "j": j})

        # Associativity (small deterministic subset).
        assoc_triples = [(0, 1, 2), (1, 2, 3), (2, 3, 0)]
        for (i, j, k) in assoc_triples:
            if i >= len(vecs) or j >= len(vecs) or k >= len(vecs):
                continue
            left = mul_fn(mul_fn(vecs[i], vecs[j]), vecs[k])
            right = mul_fn(vecs[i], mul_fn(vecs[j], vecs[k]))
            if left != right:
                return _fail("ASSOCIATIVITY_FAILURE", "associativity failure on deterministic subset", {"i": i, "j": j, "k": k})

    # --- Gate 5: fingerprint ---
    if profile == "full_backend":
        expected_fp = cert.get("algebra_fingerprint_sha256")
        if not isinstance(expected_fp, str):
            return _fail("SCHEMA_INVALID", "algebra_fingerprint_sha256 must be present for pass_exact (full_backend)")
        fp_basis = idxs[:3]
        actual_fp = _fingerprint_center_algebra(
            mul_fn=mul_fn,
            num_orbits=orbit_cache.num_orbits,
            unit_idx=id_orbit_idx,
            basis_indices=fp_basis,
        )
        if expected_fp != actual_fp:
            return _fail("FINGERPRINT_MISMATCH", "algebra fingerprint mismatch", {"expected": expected_fp, "actual": actual_fp})

    return _ok("all gates passed")


def _emit_pass_cert(profile: str) -> dict:
    eng = _load_engine()
    engine_path = REPO_ROOT / "qa_psl2_mod72_center_algebra.py"

    cache8_path = REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod8_center_cache.json"
    cache9_path = REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod9_center_cache.json"
    orbit_path = REPO_ROOT / "qa_center_algebra_cache" / "psl2_mod72_orbit_cache.json"

    cache8 = eng.FactorCenterCache.from_json(_read_json(cache8_path))
    cache9 = eng.FactorCenterCache.from_json(_read_json(cache9_path))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(orbit_path))

    id_pair = (cache8.identity_class, cache9.identity_class)
    id_orbit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]

    def mul_fn(a, b):
        return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)

    battery_seed = 0
    battery_count = 4
    idxs = _small_orbit_indices(orbit_cache, unit_idx=id_orbit_idx, count=3, offset=0)
    fp = _fingerprint_center_algebra(
        mul_fn=mul_fn,
        num_orbits=orbit_cache.num_orbits,
        unit_idx=id_orbit_idx,
        basis_indices=idxs,
    )

    out = {
        "schema_id": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
        "status": "pass_exact",
        "profile": profile,
        "modulus_n": 72,
        "engine_source_path": "qa_psl2_mod72_center_algebra.py",
        "engine_source_sha256": _sha256_file(engine_path),
        "cache_files": {
            "sl2_mod8": "qa_center_algebra_cache/sl2_mod8_center_cache.json",
            "sl2_mod9": "qa_center_algebra_cache/sl2_mod9_center_cache.json",
            "psl2_mod72_orbits": "qa_center_algebra_cache/psl2_mod72_orbit_cache.json",
        },
        "cache_sha256": {
            "sl2_mod8": _sha256_file(cache8_path),
            "sl2_mod9": _sha256_file(cache9_path),
            "psl2_mod72_orbits": _sha256_file(orbit_path),
        },
        "battery_seed": battery_seed,
        "battery_count": battery_count,
        "battery_small_orbit_offset": 0,
    }
    if profile == "full_backend":
        out["algebra_fingerprint_sha256"] = fp
    return out


def _emit_fail_bad_factor_hash() -> dict:
    base = _emit_pass_cert(profile="full_backend")
    # Corrupt one expected cache hash but include witness with the real actual hash.
    cache_key = "sl2_mod8"
    actual = base["cache_sha256"][cache_key]
    expected = "0" * 64
    base["status"] = "fail_witness"
    base["fail_type"] = "CACHE_HASH_MISMATCH"
    base["witness"] = {"cache": cache_key, "expected": expected, "actual": actual}
    # Replace declared expected hashes with the corrupted one to match the witness narrative.
    base["cache_sha256"][cache_key] = expected
    # Fingerprint not required for fail_witness.
    base.pop("algebra_fingerprint_sha256", None)
    return base


def _emit_fail_bad_unit_witness() -> dict:
    eng = _load_engine()

    cache8_path = REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod8_center_cache.json"
    cache9_path = REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod9_center_cache.json"
    orbit_path = REPO_ROOT / "qa_center_algebra_cache" / "psl2_mod72_orbit_cache.json"

    cache8 = eng.FactorCenterCache.from_json(_read_json(cache8_path))
    cache9 = eng.FactorCenterCache.from_json(_read_json(cache9_path))
    orbit_cache = eng.PSL2Mod72OrbitCache.from_json(_read_json(orbit_path))

    id_pair = (cache8.identity_class, cache9.identity_class)
    true_unit_idx = orbit_cache.pair_to_orbit[id_pair[0] * orbit_cache.r9 + id_pair[1]]
    # Pick a wrong claimed unit orbit index deterministically.
    claimed_unit_idx = (true_unit_idx + 1) % orbit_cache.num_orbits
    if claimed_unit_idx == true_unit_idx:
        claimed_unit_idx = (true_unit_idx + 2) % orbit_cache.num_orbits

    def mul_fn(a, b):
        return eng.mul_psl2_mod72_center_orbits(cache8, cache9, orbit_cache, a, b)

    # Choose a sparse vector a for which the wrong unit fails (small orbit for speed).
    idxs = _small_orbit_indices(orbit_cache, unit_idx=true_unit_idx, count=1, offset=0)
    a = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    a[idxs[0]] = Fraction(1, 1)

    e_wrong = [Fraction(0, 1) for _ in range(orbit_cache.num_orbits)]
    e_wrong[claimed_unit_idx] = Fraction(1, 1)
    ea = mul_fn(e_wrong, a)
    ae = mul_fn(a, e_wrong)
    if ea == a and ae == a:
        # Extremely unlikely; if it happens, pick another a.
        a[idxs[0]] = Fraction(2, 1)
        ea = mul_fn(e_wrong, a)
        ae = mul_fn(a, e_wrong)
        if ea == a and ae == a:
            raise RuntimeError("could not construct a wrong-unit witness (unexpected)")

    mismatch_idx = None
    for i in range(orbit_cache.num_orbits):
        if ea[i] != a[i] or ae[i] != a[i]:
            mismatch_idx = i
            break
    assert mismatch_idx is not None

    return {
        "schema_id": "QA_CONGRUENCE_CENTER_ALGEBRA_CERT.v1",
        "status": "fail_witness",
        "fail_type": "UNIT_LAW_FAILURE",
        "modulus_n": 72,
        "engine_source_path": "qa_psl2_mod72_center_algebra.py",
        "engine_source_sha256": _sha256_file(REPO_ROOT / "qa_psl2_mod72_center_algebra.py"),
        "cache_files": {
            "sl2_mod8": "qa_center_algebra_cache/sl2_mod8_center_cache.json",
            "sl2_mod9": "qa_center_algebra_cache/sl2_mod9_center_cache.json",
            "psl2_mod72_orbits": "qa_center_algebra_cache/psl2_mod72_orbit_cache.json",
        },
        "cache_sha256": {
            "sl2_mod8": _sha256_file(cache8_path),
            "sl2_mod9": _sha256_file(cache9_path),
            "psl2_mod72_orbits": _sha256_file(orbit_path),
        },
        "witness": {
            "claimed_unit_orbit_index": claimed_unit_idx,
            "a_sparse": _make_sparse_vec(a),
            "mismatch_index": mismatch_idx,
            "expected_at": [a[mismatch_idx].numerator, a[mismatch_idx].denominator],
            "left_at": [ea[mismatch_idx].numerator, ea[mismatch_idx].denominator],
            "right_at": [ae[mismatch_idx].numerator, ae[mismatch_idx].denominator],
        },
    }


def emit_examples() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    full_cert = _emit_pass_cert(profile="full_backend")
    unit_cert = _emit_pass_cert(profile="unit_law_only")
    _write_json(EXAMPLES_DIR / "PASS_center_algebra_full_build.json", full_cert)
    _write_json(EXAMPLES_DIR / "PASS_unit_law.json", unit_cert)
    # derive fail certs without recomputing fingerprints
    bad_hash = dict(full_cert)
    bad_hash["cache_sha256"] = dict(full_cert["cache_sha256"])
    bad_hash["status"] = "fail_witness"
    bad_hash["fail_type"] = "CACHE_HASH_MISMATCH"
    cache_key = "sl2_mod8"
    actual = full_cert["cache_sha256"][cache_key]
    expected = "0" * 64
    bad_hash["witness"] = {"cache": cache_key, "expected": expected, "actual": actual}
    bad_hash["cache_sha256"][cache_key] = expected
    bad_hash.pop("algebra_fingerprint_sha256", None)
    _write_json(EXAMPLES_DIR / "FAIL_bad_factor_hash.json", bad_hash)
    _write_json(EXAMPLES_DIR / "FAIL_bad_unit_witness.json", _emit_fail_bad_unit_witness())
    print(f"Wrote examples to: {EXAMPLES_DIR}")


def demo() -> int:
    emit_examples()
    paths = [
        EXAMPLES_DIR / "PASS_center_algebra_full_build.json",
        EXAMPLES_DIR / "PASS_unit_law.json",
        EXAMPLES_DIR / "FAIL_bad_factor_hash.json",
        EXAMPLES_DIR / "FAIL_bad_unit_witness.json",
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
    parser.add_argument(
        "--require_caches",
        action="store_true",
        help="Do not build caches; require cache files already exist.",
    )
    parser.add_argument(
        "--build_caches",
        action="store_true",
        help="Explicitly build caches before emitting examples/demo.",
    )
    args = parser.parse_args(argv)

    if args.emit_examples:
        if args.build_caches or not args.require_caches:
            _load_engine().build()
        else:
            # Ensure caches exist.
            for p in [
                REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod8_center_cache.json",
                REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod9_center_cache.json",
                REPO_ROOT / "qa_center_algebra_cache" / "psl2_mod72_orbit_cache.json",
            ]:
                if not p.exists():
                    raise SystemExit(f"missing required cache file: {p}")
        emit_examples()
        return 0
    if args.demo:
        if args.build_caches or not args.require_caches:
            _load_engine().build()
        else:
            for p in [
                REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod8_center_cache.json",
                REPO_ROOT / "qa_center_algebra_cache" / "sl2_mod9_center_cache.json",
                REPO_ROOT / "qa_center_algebra_cache" / "psl2_mod72_orbit_cache.json",
            ]:
                if not p.exists():
                    raise SystemExit(f"missing required cache file: {p}")
        return demo()
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
