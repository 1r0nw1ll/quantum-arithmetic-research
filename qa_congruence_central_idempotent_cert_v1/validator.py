#!/usr/bin/env python3
"""
Validator for QA_CONGRUENCE_CENTRAL_IDEMPOTENT_CERT.v1

v1 supports:
  - pass_exact for the sparse central elements 0 and 1 (identity)
  - fail_witness using the identity-coordinate of (e^2 - e) in the class-average basis

Rationale: exact disproof of naive bucket indicators is high-ROI and does not
require building the full class-multiplication table.

Usage:
  python qa_congruence_central_idempotent_cert_v1/validator.py --demo
  python qa_congruence_central_idempotent_cert_v1/validator.py <cert.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure repo root importability.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qa_congruence_sector_label_cert_v1.validator import (  # noqa: E402
    expected_psl2_order,
)
from qa_resonance_spectral_experiment import (  # noqa: E402
    digital_root_9,
    mod_psl_normal_form,
    mul_mod_n,
    inv_mod_n,
    conj_mod_n,
    reduce_mod_n,
    S,
    T,
    T_INV,
)


SCHEMA_ID = "QA_CONGRUENCE_CENTRAL_IDEMPOTENT_CERT.v1"

Label = Tuple[int, int]
ModMat = Tuple[int, int, int, int]


def label_from_int(x: int) -> Label:
    return (x % 24, digital_root_9(x))


def label_trace2_mod72(x: ModMat, n: int) -> Label:
    a, b, c, d = x
    tr = (a + d) % n
    t2 = (tr * tr) % n
    return label_from_int(t2)


def label_disc_mod72(x: ModMat, n: int) -> Label:
    a, b, c, d = x
    tr = (a + d) % n
    t2 = (tr * tr) % n
    disc = (t2 - 4) % n
    return label_from_int(disc)


LABELS: Dict[str, Callable[[ModMat, int], Label]] = {
    "trace2_mod72": label_trace2_mod72,
    "disc_mod72": label_disc_mod72,
}


def bfs_generate_psl2_mod_n(n: int) -> List[ModMat]:
    s = reduce_mod_n(S, n)
    t = reduce_mod_n(T, n)
    u = reduce_mod_n(T_INV, n)
    gens = (s, t, u)

    ident = mod_psl_normal_form((1 % n, 0, 0, 1 % n), n=n)
    seen = {ident}
    q = deque([ident])
    while q:
        cur = q.popleft()
        for g in gens:
            nxt = mul_mod_n(cur, g, n=n)
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return sorted(seen)


def conjugacy_classes(elements: List[ModMat], n: int) -> Tuple[List[List[ModMat]], Dict[ModMat, int]]:
    """
    Deterministic conjugacy class computation by BFS under conjugation by generators.
    """
    s = reduce_mod_n(S, n)
    t = reduce_mod_n(T, n)
    u = reduce_mod_n(T_INV, n)
    conj_gens = (s, t, u)

    unassigned = set(elements)
    cls_list: List[List[ModMat]] = []
    elem_to_cls: Dict[ModMat, int] = {}

    while unassigned:
        start = min(unassigned)
        q = deque([start])
        cur_cls = []
        unassigned.remove(start)
        while q:
            x = q.popleft()
            cur_cls.append(x)
            for g in conj_gens:
                y = conj_mod_n(g, x, n=n)
                if y in unassigned:
                    unassigned.remove(y)
                    q.append(y)
        cur_cls.sort()
        idx = len(cls_list)
        for x in cur_cls:
            elem_to_cls[x] = idx
        cls_list.append(cur_cls)

    # Deterministic class ordering: sort by smallest element (already sorted in each class).
    cls_list.sort(key=lambda c: c[0])
    # Rebuild elem_to_cls for new ordering.
    elem_to_cls = {}
    for idx, c in enumerate(cls_list):
        for x in c:
            elem_to_cls[x] = idx
    return cls_list, elem_to_cls


def class_order_hash(classes: List[List[ModMat]]) -> str:
    reps = [c[0] for c in classes]
    payload = json.dumps(reps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_fraction(obj: Dict[str, Any]) -> Fraction:
    return Fraction(int(obj["num"]), int(obj["den"]))


def build_alpha_vector(
    *,
    classes: List[List[ModMat]],
    n: int,
    construction_mode: str,
    label_name: Optional[str],
    bucket: Optional[Label],
    coefficients_sparse: Optional[List[Dict[str, Any]]],
) -> List[Fraction]:
    r = len(classes)
    alpha = [Fraction(0, 1) for _ in range(r)]

    if construction_mode in ("bucket_indicator", "bucket_average"):
        if not label_name or label_name not in LABELS:
            raise SystemExit("✗ COEFFICIENT_VECTOR_INVALID: unknown or missing label_name")
        if bucket is None:
            raise SystemExit("✗ COEFFICIENT_VECTOR_INVALID: missing bucket")
        label_fn = LABELS[label_name]
        chosen = []
        for i, c in enumerate(classes):
            lab = label_fn(c[0], n)
            if lab == bucket:
                chosen.append(i)
        if not chosen:
            raise SystemExit("✗ COEFFICIENT_VECTOR_INVALID: empty bucket (no classes selected)")
        if construction_mode == "bucket_indicator":
            for i in chosen:
                alpha[i] = Fraction(1, 1)
        else:
            m = Fraction(1, len(chosen))
            for i in chosen:
                alpha[i] = m
        return alpha

    if construction_mode == "coefficients_sparse":
        if coefficients_sparse is None:
            raise SystemExit("✗ COEFFICIENT_VECTOR_INVALID: missing coefficients_sparse")
        for entry in coefficients_sparse:
            idx = int(entry["class_index"])
            if not (0 <= idx < r):
                raise SystemExit("✗ COEFFICIENT_VECTOR_INVALID: class_index out of range")
            alpha[idx] = parse_fraction(entry)
        return alpha

    raise SystemExit(f"✗ COEFFICIENT_VECTOR_INVALID: unknown construction_mode {construction_mode!r}")


def identity_coordinate_diff(alpha: List[Fraction], class_sizes: List[int], inv_of: List[int]) -> Fraction:
    """
    For e = Σ α_i A_i with A_i class-average basis, compute the coefficient of the identity element
    in (e^2 - e).

    Fact: coeff_1(A_i A_j) = 1/|C_i| if C_j = C_i^{-1}, else 0.
    Therefore:
      coeff_1(e^2) = Σ_i α_i α_{inv(i)} / |C_i|
      coeff_1(e)   = α_identity_class = α_0
    """
    s = Fraction(0, 1)
    for i, ai in enumerate(alpha):
        if ai == 0:
            continue
        aj = alpha[inv_of[i]]
        if aj == 0:
            continue
        s += ai * aj / class_sizes[i]
    return s - alpha[0]


def verify(cert: Dict[str, Any], cert_path: Path) -> None:
    if cert.get("schema_id") != SCHEMA_ID:
        raise SystemExit(f"✗ SCHEMA_INVALID: schema_id must be {SCHEMA_ID}")

    n = int(cert["modulus_n"])
    if n != 72:
        raise SystemExit("✗ SCHEMA_INVALID: v1 fixes modulus_n=72")

    elements = bfs_generate_psl2_mod_n(n)
    exp = expected_psl2_order(n)
    if len(elements) != exp:
        raise SystemExit(f"✗ GROUP_ENUMERATION_INCOMPLETE: got={len(elements)} expected={exp}")

    classes, elem_to_cls = conjugacy_classes(elements, n)
    r = len(classes)
    sizes = [len(c) for c in classes]
    order_hash = class_order_hash(classes)

    # Precompute inverse-class map.
    inv_of = [0] * r
    for i, c in enumerate(classes):
        rep = c[0]
        inv_rep = inv_mod_n(rep, n=n)
        inv_of[i] = elem_to_cls[inv_rep]

    construction_mode = cert["construction_mode"]
    label_name = None
    bucket = None
    coefficients_sparse = None
    if construction_mode in ("bucket_indicator", "bucket_average"):
        label_name = cert["label_name"]
        bucket = (int(cert["bucket_spec"]["label_mod24"]), int(cert["bucket_spec"]["label_dr9"]))
    elif construction_mode == "coefficients_sparse":
        coefficients_sparse = cert["coefficients_sparse"]
    else:
        raise SystemExit("✗ SCHEMA_INVALID: unsupported construction_mode in v1")

    alpha = build_alpha_vector(
        classes=classes,
        n=n,
        construction_mode=construction_mode,
        label_name=label_name,
        bucket=bucket,
        coefficients_sparse=coefficients_sparse,
    )

    status = cert["status"]
    if status == "pass_exact":
        # v1 supports exact PASS only for 0 and 1, represented sparsely.
        support = [(i, a) for i, a in enumerate(alpha) if a != 0]
        if not support:
            print(f"✓ OK: {cert_path} verifies 0 is idempotent in Z(Q[G])")
            return
        if len(support) == 1 and support[0][0] == 0 and support[0][1] == 1:
            print(f"✓ OK: {cert_path} verifies 1 is idempotent in Z(Q[G])")
            return
        raise SystemExit("✗ IDEMPOTENCE_VIOLATION: v1 pass_exact supports only 0 and 1 fixtures")

    if status == "fail_witness":
        # Verify the identity-coordinate witness.
        diff0 = identity_coordinate_diff(alpha, sizes, inv_of)
        w = cert["witness"]
        cls_idx = int(w["class_index"])
        if cls_idx != 0:
            raise SystemExit("✗ FALSE_FAIL_WITNESS: v1 witness must target identity class_index=0")
        claimed = parse_fraction(w["diff_coeff"])
        if diff0 != claimed:
            raise SystemExit(f"✗ FALSE_FAIL_WITNESS: diff mismatch got={diff0} claimed={claimed}")
        if diff0 == 0:
            raise SystemExit("✗ FALSE_FAIL_WITNESS: claimed nonzero but recomputed zero")
        print(f"✓ OK: {cert_path} verifies idempotence FAIL witness (identity coordinate diff={diff0})")
        return

    raise SystemExit(f"✗ SCHEMA_INVALID: unsupported status {status!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cert", nargs="?", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        base = Path(__file__).resolve().parent / "examples"
        for p in sorted(base.glob("*.json")):
            with p.open() as f:
                verify(json.load(f), p)
        return 0

    if not args.cert:
        raise SystemExit("usage: validator.py <cert.json> OR --demo")

    cert_path = Path(args.cert)
    with cert_path.open() as f:
        verify(json.load(f), cert_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

