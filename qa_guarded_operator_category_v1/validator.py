#!/usr/bin/env python3
"""
validator.py

QA_GUARDED_OPERATOR_CATEGORY_CERT.v1 validator (Machine tract).

Gates:
  1) Schema anchor validity
  2) Generator semantics + rho anchor coherence
  3) Composition recompute integrity (word -> matrix -> action)
  4) Determinant + finite mod-n subgroup checks
  5) Failure-obstruction completeness for lambda_k and nu

CLI:
  python validator.py <cert.json>
  python validator.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass
class Diff:
    gate: int
    fail_type: str
    path: str
    reason: str


@dataclass
class GateResult:
    gate_id: int
    status: GateStatus
    message: str
    diffs: List[Diff] = field(default_factory=list)


def _pass(gate_id: int, message: str) -> GateResult:
    return GateResult(gate_id, GateStatus.PASS, message)


def _fail(gate_id: int, fail_type: str, path: str, reason: str) -> GateResult:
    return GateResult(
        gate_id,
        GateStatus.FAIL,
        f"{fail_type} @ {path} -- {reason}",
        [Diff(gate=gate_id, fail_type=fail_type, path=path, reason=reason)],
    )


Matrix2 = Tuple[Tuple[int, int], Tuple[int, int]]


def _to_matrix2(path: str, value: Any) -> Tuple[Optional[Matrix2], Optional[GateResult]]:
    if not isinstance(value, list) or len(value) != 2:
        return None, _fail(0, "SCHEMA_TYPE_MISMATCH", path, "expected 2x2 array")
    if not all(isinstance(row, list) and len(row) == 2 for row in value):
        return None, _fail(0, "SCHEMA_TYPE_MISMATCH", path, "expected 2x2 array")
    flat = [value[0][0], value[0][1], value[1][0], value[1][1]]
    if not all(isinstance(x, int) for x in flat):
        return None, _fail(0, "SCHEMA_TYPE_MISMATCH", path, "matrix entries must be integers")
    return ((flat[0], flat[1]), (flat[2], flat[3])), None


def _mat_mul(a: Matrix2, b: Matrix2) -> Matrix2:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def _mat_det(m: Matrix2) -> int:
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def _mat_vec_mul(m: Matrix2, v: Tuple[int, int]) -> Tuple[int, int]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    )


def _reduce_mod_n(m: Matrix2, n: int) -> Matrix2:
    return (
        (m[0][0] % n, m[0][1] % n),
        (m[1][0] % n, m[1][1] % n),
    )


def _all_gl2_mod_n(n: int) -> List[Matrix2]:
    mats: List[Matrix2] = []
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    det = (a * d - b * c) % n
                    if math.gcd(det, n) == 1:
                        mats.append(((a, b), (c, d)))
    return mats


def _det_pm_one_subset_mod_n(n: int) -> List[Matrix2]:
    out: List[Matrix2] = []
    target = {1 % n, (-1) % n}
    for m in _all_gl2_mod_n(n):
        det = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) % n
        if det in target:
            out.append(m)
    return out


def _generated_subgroup_mod_n(ms: Matrix2, mm: Matrix2, n: int) -> List[Matrix2]:
    gens = [_reduce_mod_n(ms, n), _reduce_mod_n(mm, n)]
    identity = ((1 % n, 0), (0, 1 % n))
    seen = {identity}
    queue = [identity]

    while queue:
        cur = queue.pop(0)
        for g in gens:
            nxt = _reduce_mod_n(_mat_mul(cur, g), n)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return sorted(seen)


def _eval_word_matrix(word: Sequence[str], ms: Matrix2, mm: Matrix2) -> Matrix2:
    acc: Matrix2 = ((1, 0), (0, 1))
    for token in word:
        if token == "sigma":
            acc = _mat_mul(ms, acc)
        elif token == "mu":
            acc = _mat_mul(mm, acc)
        else:
            raise ValueError(f"unknown token: {token}")
    return acc


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_semantics_registry(base_dir: str) -> Dict[str, Any]:
    reg_path = os.path.join(base_dir, "generator_semantics_registry.json")
    return _load_json(reg_path)


def _gate1_schema(cert: Any) -> GateResult:
    if not isinstance(cert, dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", ".", "cert must be a JSON object")

    if cert.get("schema_id") != "QA_GUARDED_OPERATOR_CATEGORY_CERT.v1":
        return _fail(
            1,
            "SCHEMA_ID_MISMATCH",
            "schema_id",
            f"expected 'QA_GUARDED_OPERATOR_CATEGORY_CERT.v1', got {cert.get('schema_id')!r}",
        )

    required = ["cert_id", "created_utc", "subject", "rho", "claims", "failure_obstructions"]
    for k in required:
        if k not in cert:
            return _fail(1, "SCHEMA_REQUIRED_FIELD_MISSING", k, f"required field '{k}' missing")

    if not isinstance(cert["cert_id"], str) or len(cert["cert_id"]) < 6:
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "cert_id", "cert_id must be string len>=6")

    if not isinstance(cert["subject"], dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "subject", "subject must be object")
    if not isinstance(cert["rho"], dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "rho", "rho must be object")
    if not isinstance(cert["claims"], dict):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "claims", "claims must be object")
    if not isinstance(cert["failure_obstructions"], list):
        return _fail(1, "SCHEMA_TYPE_MISMATCH", "failure_obstructions", "must be array")

    return _pass(1, "schema anchor valid")


def _gate2_anchors(cert: Dict[str, Any], base_dir: str) -> GateResult:
    subject = cert["subject"]
    for k in ("generator_set", "state_domain", "matrix_convention", "generator_semantics_ref"):
        if k not in subject:
            return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"subject.{k}", "required")

    gset = subject["generator_set"]
    if not isinstance(gset, list) or len(gset) < 2:
        return _fail(2, "SCHEMA_TYPE_MISMATCH", "subject.generator_set", "must be list with >=2 items")
    if not all(isinstance(x, str) for x in gset):
        return _fail(2, "SCHEMA_TYPE_MISMATCH", "subject.generator_set", "items must be strings")
    if len(set(gset)) != len(gset):
        return _fail(2, "GENERATOR_SET_INVALID", "subject.generator_set", "duplicates not allowed")

    allowed = {"sigma", "mu", "lambda_k", "nu"}
    bad = [x for x in gset if x not in allowed]
    if bad:
        return _fail(2, "GENERATOR_SET_INVALID", "subject.generator_set", f"unknown generators: {bad}")
    if "sigma" not in gset or "mu" not in gset:
        return _fail(2, "GENERATOR_SET_INVALID", "subject.generator_set", "sigma and mu required")

    if subject["state_domain"] not in ("Z2", "N2"):
        return _fail(2, "SCHEMA_TYPE_MISMATCH", "subject.state_domain", "must be Z2 or N2")
    if subject["matrix_convention"] != "column_vector":
        return _fail(2, "MATRIX_CONVENTION_INVALID", "subject.matrix_convention", "must be column_vector")

    registry = _load_semantics_registry(base_dir)
    ref = subject["generator_semantics_ref"]
    if ref not in registry.get("definitions", {}):
        return _fail(2, "SEMANTICS_REF_UNRESOLVED", "subject.generator_semantics_ref", f"unknown ref {ref!r}")

    rho = cert["rho"]
    for k in ("sigma", "mu"):
        if k not in rho:
            return _fail(2, "SCHEMA_REQUIRED_FIELD_MISSING", f"rho.{k}", "required")

    ms, err = _to_matrix2("rho.sigma", rho["sigma"])
    if err:
        err.gate_id = 2
        err.diffs[0].gate = 2
        return err
    mm, err = _to_matrix2("rho.mu", rho["mu"])
    if err:
        err.gate_id = 2
        err.diffs[0].gate = 2
        return err

    expected_sigma = ((1, 0), (1, 1))
    expected_mu = ((0, 1), (1, 0))
    if ms != expected_sigma:
        return _fail(2, "RHO_ANCHOR_MISMATCH", "rho.sigma", f"expected {expected_sigma}, got {ms}")
    if mm != expected_mu:
        return _fail(2, "RHO_ANCHOR_MISMATCH", "rho.mu", f"expected {expected_mu}, got {mm}")

    # Semantics consistency: chosen semantics must define sigma as shear map.
    sigma_map = registry["definitions"][ref].get("sigma")
    if sigma_map != "(b,e)->(b,e+b)":
        return _fail(
            2,
            "SEMANTICS_MISMATCH",
            "subject.generator_semantics_ref",
            "this cert family requires sigma shear semantics (b,e)->(b,e+b)",
        )

    return _pass(2, f"anchors coherent; semantics_ref={ref}")


def _gate3_composition(cert: Dict[str, Any]) -> GateResult:
    claims = cert["claims"]
    if "composition_samples" not in claims:
        return _fail(3, "SCHEMA_REQUIRED_FIELD_MISSING", "claims.composition_samples", "required")
    samples = claims["composition_samples"]
    if not isinstance(samples, list) or len(samples) < 1:
        return _fail(3, "SCHEMA_TYPE_MISMATCH", "claims.composition_samples", "must be non-empty array")

    ms = ((1, 0), (1, 1))
    mm = ((0, 1), (1, 0))

    for i, sample in enumerate(samples):
        p = f"claims.composition_samples[{i}]"
        if not isinstance(sample, dict):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", p, "sample must be object")
        for k in ("word", "expected_matrix", "seed", "expected_state"):
            if k not in sample:
                return _fail(3, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{k}", "required")

        word = sample["word"]
        if not isinstance(word, list) or not all(t in ("sigma", "mu") for t in word):
            return _fail(3, "WORD_INVALID", f"{p}.word", "word tokens must be sigma|mu")

        expected_matrix, err = _to_matrix2(f"{p}.expected_matrix", sample["expected_matrix"])
        if err:
            err.gate_id = 3
            err.diffs[0].gate = 3
            return err

        seed = sample["seed"]
        target = sample["expected_state"]
        if not (isinstance(seed, list) and len(seed) == 2 and all(isinstance(x, int) for x in seed)):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", f"{p}.seed", "seed must be int[2]")
        if not (isinstance(target, list) and len(target) == 2 and all(isinstance(x, int) for x in target)):
            return _fail(3, "SCHEMA_TYPE_MISMATCH", f"{p}.expected_state", "expected_state must be int[2]")

        recomputed = _eval_word_matrix(word, ms, mm)
        if recomputed != expected_matrix:
            return _fail(
                3,
                "COMPOSITION_MATRIX_MISMATCH",
                f"{p}.expected_matrix",
                f"expected {expected_matrix}, recomputed {recomputed}",
            )

        out = _mat_vec_mul(recomputed, (seed[0], seed[1]))
        if out != (target[0], target[1]):
            return _fail(
                3,
                "COMPOSITION_STATE_MISMATCH",
                f"{p}.expected_state",
                f"expected {target}, recomputed {list(out)}",
            )

    return _pass(3, f"composition recompute OK ({len(samples)} samples)")


def _gate4_det_and_modn(cert: Dict[str, Any]) -> GateResult:
    claims = cert["claims"]
    if "determinants" not in claims:
        return _fail(4, "SCHEMA_REQUIRED_FIELD_MISSING", "claims.determinants", "required")
    dets = claims["determinants"]
    if not isinstance(dets, dict):
        return _fail(4, "SCHEMA_TYPE_MISMATCH", "claims.determinants", "must be object")

    ms = ((1, 0), (1, 1))
    mm = ((0, 1), (1, 0))
    if dets.get("sigma") != _mat_det(ms):
        return _fail(4, "DET_ASSERTION_MISMATCH", "claims.determinants.sigma", "incorrect det claim")
    if dets.get("mu") != _mat_det(mm):
        return _fail(4, "DET_ASSERTION_MISMATCH", "claims.determinants.mu", "incorrect det claim")

    checks = claims.get("mod_n_checks")
    if not isinstance(checks, list) or len(checks) < 1:
        return _fail(4, "SCHEMA_TYPE_MISMATCH", "claims.mod_n_checks", "must be non-empty array")

    for i, check in enumerate(checks):
        p = f"claims.mod_n_checks[{i}]"
        if not isinstance(check, dict):
            return _fail(4, "SCHEMA_TYPE_MISMATCH", p, "must be object")
        for k in ("n", "expected_generated_size", "expected_det_pm_one_size"):
            if k not in check:
                return _fail(4, "SCHEMA_REQUIRED_FIELD_MISSING", f"{p}.{k}", "required")
        n = check["n"]
        if not isinstance(n, int) or n < 2:
            return _fail(4, "SCHEMA_TYPE_MISMATCH", f"{p}.n", "n must be integer >=2")

        gen_size = len(_generated_subgroup_mod_n(ms, mm, n))
        det_pm_size = len(_det_pm_one_subset_mod_n(n))

        if check["expected_generated_size"] != gen_size:
            return _fail(
                4,
                "MODN_GENERATED_SIZE_MISMATCH",
                f"{p}.expected_generated_size",
                f"expected {check['expected_generated_size']}, recomputed {gen_size}",
            )
        if check["expected_det_pm_one_size"] != det_pm_size:
            return _fail(
                4,
                "MODN_DET_PM_SIZE_MISMATCH",
                f"{p}.expected_det_pm_one_size",
                f"expected {check['expected_det_pm_one_size']}, recomputed {det_pm_size}",
            )
        if gen_size != det_pm_size:
            return _fail(4, "MODN_IMAGE_MISMATCH", p, f"generated {gen_size} != det_pm_one {det_pm_size}")

    return _pass(4, f"det + mod-n checks OK ({len(checks)} checks)")


def _find_obstruction(obstructions: Iterable[Dict[str, Any]], obstruction_id: str) -> Optional[Dict[str, Any]]:
    for ob in obstructions:
        if isinstance(ob, dict) and ob.get("obstruction_id") == obstruction_id:
            return ob
    return None


def _gate5_obstructions(cert: Dict[str, Any]) -> GateResult:
    gset = cert["subject"]["generator_set"]
    guards = cert["subject"].get("guards", {})
    obs = cert["failure_obstructions"]

    if not isinstance(obs, list):
        return _fail(5, "SCHEMA_TYPE_MISMATCH", "failure_obstructions", "must be array")

    if "lambda_k" in gset:
        ob = _find_obstruction(obs, "lambda_k_det_not_pm1")
        if ob is None:
            return _fail(5, "OBSTRUCTION_MISSING", "failure_obstructions", "missing lambda_k obstruction")
        witness = ob.get("witness")
        if not isinstance(witness, dict):
            return _fail(5, "OBSTRUCTION_INVALID", "failure_obstructions.lambda_k_det_not_pm1.witness", "must be object")
        k = witness.get("k")
        if not isinstance(k, int) or abs(k) == 1:
            return _fail(5, "OBSTRUCTION_INVALID", "failure_obstructions.lambda_k_det_not_pm1.witness.k", "must be int with |k|!=1")
        mat = witness.get("matrix")
        det = witness.get("det")
        expected_mat = [[k, 0], [0, k]]
        expected_det = k * k
        if mat != expected_mat or det != expected_det:
            return _fail(
                5,
                "LAMBDA_K_OBSTRUCTION_MISMATCH",
                "failure_obstructions.lambda_k_det_not_pm1.witness",
                f"expected matrix={expected_mat}, det={expected_det}",
            )
        if det in (1, -1):
            return _fail(5, "LAMBDA_K_OBSTRUCTION_MISMATCH", "failure_obstructions.lambda_k_det_not_pm1.witness.det", "det must not be +/-1")

    if "nu" in gset:
        if not isinstance(guards, dict) or "nu" not in guards:
            return _fail(5, "NU_GUARD_MISSING", "subject.guards.nu", "nu requires guard {'type':'parity','requires':'both_even'}")
        nu_guard = guards.get("nu")
        if (
            not isinstance(nu_guard, dict)
            or nu_guard.get("type") != "parity"
            or nu_guard.get("requires") != "both_even"
        ):
            return _fail(
                5,
                "NU_GUARD_INVALID",
                "subject.guards.nu",
                "nu guard must be {'type':'parity','requires':'both_even'}",
            )
        ob = _find_obstruction(obs, "nu_not_integer_total_linear")
        if ob is None:
            return _fail(5, "OBSTRUCTION_MISSING", "failure_obstructions", "missing nu obstruction")
        eqs = ob.get("equations")
        if not isinstance(eqs, list):
            return _fail(5, "OBSTRUCTION_INVALID", "failure_obstructions.nu_not_integer_total_linear.equations", "must be array")
        required_eqs = {"N*(2,0)^T=(1,0)^T", "N*(0,2)^T=(0,1)^T"}
        if not required_eqs.issubset(set(str(x) for x in eqs)):
            return _fail(
                5,
                "NU_OBSTRUCTION_MISMATCH",
                "failure_obstructions.nu_not_integer_total_linear.equations",
                "missing required linear constraints",
            )

    return _pass(5, "obstruction completeness OK")


def validate_certificate(cert: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    gates = [
        _gate1_schema(cert),
    ]
    if gates[-1].status == GateStatus.FAIL:
        return _to_result(gates)

    gates.append(_gate2_anchors(cert, base_dir))
    if gates[-1].status == GateStatus.FAIL:
        return _to_result(gates)

    gates.append(_gate3_composition(cert))
    if gates[-1].status == GateStatus.FAIL:
        return _to_result(gates)

    gates.append(_gate4_det_and_modn(cert))
    if gates[-1].status == GateStatus.FAIL:
        return _to_result(gates)

    gates.append(_gate5_obstructions(cert))
    return _to_result(gates)


def _to_result(gates: List[GateResult]) -> Dict[str, Any]:
    ok = all(g.status == GateStatus.PASS for g in gates)
    diffs: List[Dict[str, Any]] = []
    for g in gates:
        for d in g.diffs:
            diffs.append({"gate": d.gate, "fail_type": d.fail_type, "path": d.path, "reason": d.reason})

    return {
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
        "gates": [
            {
                "gate_id": g.gate_id,
                "status": g.status.value,
                "message": g.message,
            }
            for g in gates
        ],
        "invariant_diff": diffs,
    }


def run_self_test(base_dir: str) -> int:
    fixtures = [
        ("valid_min.json", True),
        ("invalid_nu_unguarded.json", False),
        ("invalid_nu_guard_malformed.json", False),
        ("invalid_lambda_det_claim.json", False),
    ]

    all_ok = True
    for name, expected_ok in fixtures:
        p = os.path.join(base_dir, "fixtures", name)
        if not os.path.exists(p):
            print(f"[FAIL] self-test {name}: fixture missing")
            all_ok = False
            continue
        try:
            cert = _load_json(p)
        except Exception as exc:
            print(f"[FAIL] self-test {name}: parse error: {exc}")
            all_ok = False
            continue

        result = validate_certificate(cert, base_dir)
        actual_ok = bool(result["ok"])
        if actual_ok == expected_ok:
            print(f"[PASS] self-test {name}: got {actual_ok} (expected {expected_ok})")
        else:
            print(f"[FAIL] self-test {name}: got {actual_ok} (expected {expected_ok})")
            print(json.dumps(result, indent=2, sort_keys=True))
            all_ok = False

    if all_ok:
        print("[PASS] all self-tests")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA_GUARDED_OPERATOR_CATEGORY_CERT.v1 certificates")
    parser.add_argument("file", nargs="?", help="certificate JSON file")
    parser.add_argument("--self-test", action="store_true", help="run validator self-tests")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.self_test:
        return run_self_test(base_dir)

    if not args.file:
        parser.error("provide a certificate JSON file or use --self-test")

    cert = _load_json(args.file)
    result = validate_certificate(cert, base_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
