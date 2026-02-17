#!/usr/bin/env python3
"""
qa_rt_corpus_generator.py

Certified Scene Corpus Generator for QA_ARTEXPLORER_SCENE_ADAPTER families.

Generates N random non-degenerate 3D triangles, builds complete cert JSON
(ART_PARSE_SCENE -> RT_COMPUTE -> RT_VALIDATE_LAW_EQUATION),
validates each via the appropriate family validator, and writes a
CORPUS_MANIFEST.json with aggregate stats.

Supports two substrates:
  - float64 (default): family [45] v1, IEEE-754 doubles, relative tolerance 1e-9
  - exact: family [50] v2, integer coords, unreduced rational pairs, tolerance = 0

Usage:
    python qa_rt_corpus_generator.py --count 50 --output-dir qa_rt_corpus/
    python qa_rt_corpus_generator.py --count 10000 --output-dir qa_rt_corpus/burn_in --seed 42
    python qa_rt_corpus_generator.py --count 20 --laws RT_LAW_04,RT_LAW_05
    python qa_rt_corpus_generator.py --count 500 --near-degenerate-rate 0.5
    python qa_rt_corpus_generator.py --count 100 --substrate exact --no-write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import random


# ---------------------------------------------------------------------------
# Deterministic helpers (must match validator exactly)
# ---------------------------------------------------------------------------

def _canonical_json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _step_hash(move_id: str, inputs: Dict, outputs: Dict) -> str:
    payload = {"inputs": inputs, "move_id": move_id, "outputs": outputs}
    return _sha256_hex(_canonical_json_compact(payload))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [a[i] - b[i] for i in range(3)]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _quadrance(a: List[float], b: List[float]) -> float:
    d = _vec_sub(b, a)
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _spread(v1: List[float], v2: List[float]) -> Optional[float]:
    q1 = _dot(v1, v1)
    q2 = _dot(v2, v2)
    if q1 == 0.0 or q2 == 0.0:
        return None
    d = _dot(v1, v2)
    return 1.0 - (d * d) / (q1 * q2)


def _cross_mag_sq(a: List[float], b: List[float], c: List[float]) -> float:
    ab = _vec_sub(b, a)
    ac = _vec_sub(c, a)
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return cx * cx + cy * cy + cz * cz


def _compute_rt(a: List[float], b: List[float], c: List[float]):
    """Returns (Q, s) or raises on degenerate."""
    Q1 = _quadrance(b, c)
    Q2 = _quadrance(c, a)
    Q3 = _quadrance(a, b)

    ab = _vec_sub(b, a)
    ac = _vec_sub(c, a)
    bc = _vec_sub(c, b)
    ba = _vec_sub(a, b)
    ca = _vec_sub(a, c)
    cb = _vec_sub(b, c)

    sA = _spread(ab, ac)
    sB = _spread(bc, ba)
    sC = _spread(ca, cb)
    if sA is None or sB is None or sC is None:
        raise ValueError("Degenerate triangle")

    return [Q1, Q2, Q3], [sA, sB, sC]


# ---------------------------------------------------------------------------
# RT Law equations
# ---------------------------------------------------------------------------

def _cross_law(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    Q1, Q2, Q3 = Q
    s3 = s[2]
    lhs = (Q1 + Q2 - Q3) ** 2
    rhs = 4.0 * Q1 * Q2 * (1.0 - s3)
    return lhs, rhs, abs(lhs - rhs)


def _triple_spread(Q: List[float], s: List[float]) -> Tuple[float, float, float]:
    s1, s2, s3 = s
    lhs = (s1 + s2 + s3) ** 2
    rhs = 2.0 * (s1 ** 2 + s2 ** 2 + s3 ** 2) + 4.0 * s1 * s2 * s3
    return lhs, rhs, abs(lhs - rhs)


_LAW_FNS = {
    "RT_LAW_04": _cross_law,
    "RT_LAW_05": _triple_spread,
}


# ---------------------------------------------------------------------------
# Triangle generators
# ---------------------------------------------------------------------------

def _gen_scalene(rng: random.Random, scale: float = 10.0) -> List[List[float]]:
    """Random scalene triangle with coords in [-scale, scale]."""
    while True:
        pts = [[rng.uniform(-scale, scale) for _ in range(3)] for _ in range(3)]
        if _cross_mag_sq(pts[0], pts[1], pts[2]) > 1e-6 * scale * scale:
            return pts


def _gen_right(rng: random.Random, scale: float = 10.0) -> List[List[float]]:
    """Right triangle at origin, right angle at A."""
    a = [0.0, 0.0, 0.0]
    length1 = rng.uniform(scale * 0.1, scale)
    length2 = rng.uniform(scale * 0.1, scale)
    b = [length1, 0.0, 0.0]
    c = [0.0, length2, 0.0]
    return [a, b, c]


def _gen_equilateral(rng: random.Random, scale: float = 10.0) -> List[List[float]]:
    """Equilateral triangle in XY plane."""
    s = rng.uniform(scale * 0.1, scale)
    a = [0.0, 0.0, 0.0]
    b = [s, 0.0, 0.0]
    c = [s / 2.0, s * math.sqrt(3) / 2.0, 0.0]
    return [a, b, c]


def _gen_near_degenerate(rng: random.Random, scale: float = 10.0) -> List[List[float]]:
    """Nearly collinear triangle (small cross product)."""
    while True:
        a = [0.0, 0.0, 0.0]
        b = [rng.uniform(scale * 0.5, scale), 0.0, 0.0]
        c = [rng.uniform(scale * 0.2, scale * 0.8), rng.uniform(scale * 0.0001, scale * 0.005), 0.0]
        if _cross_mag_sq(a, b, c) > 1e-12 * scale * scale:
            return [a, b, c]


def _build_generators(near_degenerate_rate: Optional[float] = None):
    """Build generator list with optional near-degenerate rate override."""
    if near_degenerate_rate is not None:
        nd = max(0.0, min(1.0, near_degenerate_rate))
        remaining = 1.0 - nd
        return [
            ("scalene", _gen_scalene, remaining * 0.6 / 0.9),
            ("right", _gen_right, remaining * 0.2 / 0.9),
            ("equilateral", _gen_equilateral, remaining * 0.1 / 0.9),
            ("near_degenerate", _gen_near_degenerate, nd),
        ]
    return [
        ("scalene", _gen_scalene, 0.60),
        ("right", _gen_right, 0.20),
        ("equilateral", _gen_equilateral, 0.10),
        ("near_degenerate", _gen_near_degenerate, 0.10),
    ]


def _pick_generator(rng: random.Random, generators):
    r = rng.random()
    cum = 0.0
    for name, fn, weight in generators:
        cum += weight
        if r < cum:
            return name, fn
    return generators[0][0], generators[0][1]


# ---------------------------------------------------------------------------
# Cert builder
# ---------------------------------------------------------------------------

def build_cert(
    index: int,
    triangle_type: str,
    verts: List[List[float]],
    laws: List[str],
    timestamp: str,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Build a cert and return (cert, law_residuals).

    law_residuals maps law_id -> residual for this cert.
    """
    a, b, c = verts
    Q, s = _compute_rt(a, b, c)

    # Use full precision — validator recomputes from vertex coords and
    # compares Q/s exactly, so rounding would cause mismatches.

    cert_id = f"qa_rt_corpus_{index:06d}"
    object_id = f"tri_{index:06d}"

    scene_raw = {
        "version": "1.0",
        "timestamp": timestamp,
        "instances": [{
            "id": f"tri-{index:06d}",
            "type": "Triangle",
            "parameters": {"triangle_type": triangle_type},
            "transform": {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
            },
            "appearance": {"opacity": 1.0, "visible": True},
            "metadata": {"label": f"corpus_{triangle_type}_{index:06d}"},
        }],
        "metadata": {"instanceCount": 1},
    }
    scene_raw_sha = _sha256_hex(_canonical_json_compact(scene_raw))

    parsed_obj = {
        "object_id": object_id,
        "object_type": "TRIANGLE",
        "label": f"Corpus {triangle_type} triangle #{index}",
        "vertices": [
            {"id": "A", "coord": list(a)},
            {"id": "B", "coord": list(b)},
            {"id": "C", "coord": list(c)},
        ],
        "faces": [["A", "B", "C"]],
    }

    steps = []
    step_num = 1

    # Step 1: ART_PARSE_SCENE
    s1_inputs = {"scene_raw_sha256": scene_raw_sha}
    s1_outputs = {"parsed_object_ids": [object_id]}
    steps.append({
        "step_id": f"s{step_num}",
        "move_id": "ART_PARSE_SCENE",
        "inputs": s1_inputs,
        "outputs": s1_outputs,
        "step_hash_sha256": _step_hash("ART_PARSE_SCENE", s1_inputs, s1_outputs),
    })
    step_num += 1

    # Step 2: RT_COMPUTE_TRIANGLE_INVARIANTS
    s2_inputs = {"triangle_object_id": object_id}
    s2_outputs = {"Q": Q, "s": s}
    steps.append({
        "step_id": f"s{step_num}",
        "move_id": "RT_COMPUTE_TRIANGLE_INVARIANTS",
        "inputs": s2_inputs,
        "outputs": s2_outputs,
        "step_hash_sha256": _step_hash("RT_COMPUTE_TRIANGLE_INVARIANTS", s2_inputs, s2_outputs),
    })
    step_num += 1

    # Law verification steps
    law_residuals: Dict[str, float] = {}
    for law_id in laws:
        if law_id not in _LAW_FNS:
            continue
        lhs, rhs, residual = _LAW_FNS[law_id](Q, s)
        law_residuals[law_id] = residual

        law_inputs = {"law_id": law_id, "Q": Q, "s": s, "triangle_object_id": object_id}
        law_outputs = {"verified": True, "residual": residual, "lhs": lhs, "rhs": rhs}
        steps.append({
            "step_id": f"s{step_num}",
            "move_id": "RT_VALIDATE_LAW_EQUATION",
            "inputs": law_inputs,
            "outputs": law_outputs,
            "step_hash_sha256": _step_hash("RT_VALIDATE_LAW_EQUATION", law_inputs, law_outputs),
        })
        step_num += 1

    cert = {
        "schema_version": "v1",
        "cert_type": "QA_ARTEXPLORER_SCENE_ADAPTER.v1",
        "cert_id": cert_id,
        "created_utc": timestamp,
        "source_semantics": {
            "upstream": {
                "name": "ARTexplorer (Algebraic Rational Trigonometry Explorer)",
                "repo_url": "https://github.com/arossti/ARTexplorer",
                "app_url": "https://arossti.github.io/ARTexplorer/",
            },
            "export": {
                "format": "artexplorer_scene_json",
                "exported_by": "qa_rt_corpus_generator.py",
            },
        },
        "base_algebra": {
            "name": "Q",
            "properties": {"integral_domain": True, "field": True, "no_zero_divisors": True},
        },
        "scene": {
            "coordinate_system": "XYZ",
            "scene_raw": scene_raw,
            "scene_raw_sha256": scene_raw_sha,
        },
        "derivation": {
            "parsed_objects": [parsed_obj],
            "steps": steps,
        },
        "result": {
            "rt_invariants": {
                "triangles": [{"triangle_object_id": object_id, "Q": Q, "s": s}],
            },
            "invariant_diff": {
                "steps": [
                    {"step_id": "s1", "adds": [f"parsed_objects[{object_id}]"]},
                    {"step_id": "s2", "adds": [
                        f"rt_invariants.triangles[{object_id}].Q",
                        f"rt_invariants.triangles[{object_id}].s",
                    ]},
                ] + [
                    {"step_id": f"s{3 + i}", "adds": [f"law_verification[{law}].verified"]}
                    for i, law in enumerate(laws) if law in _LAW_FNS
                ],
                "provenance": {
                    "upstream_tool": "qa_rt_corpus_generator.py",
                    "upstream_format": "synthetic",
                    "triangle_type": triangle_type,
                },
            },
        },
        "determinism_contract": {
            "canonical_json": True,
            "stable_sorting": True,
            "no_rng": True,
            "invariant_diff_defined": True,
        },
    }
    return cert, law_residuals


# ---------------------------------------------------------------------------
# Exact-mode geometry (integer coordinates, unreduced rational pairs)
# ---------------------------------------------------------------------------

def _vec_sub_int(a: List[int], b: List[int]) -> List[int]:
    return [a[i] - b[i] for i in range(3)]


def _dot_int(a: List[int], b: List[int]) -> int:
    return sum(a[i] * b[i] for i in range(3))


def _quadrance_int(a: List[int], b: List[int]) -> int:
    d = _vec_sub_int(b[:3], a[:3])
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def _spread_pair(v1: List[int], v2: List[int]) -> Optional[Dict[str, int]]:
    q1 = _dot_int(v1, v1)
    q2 = _dot_int(v2, v2)
    if q1 == 0 or q2 == 0:
        return None
    d = q1 * q2
    dot_val = _dot_int(v1, v2)
    n = d - dot_val * dot_val
    return {"n": n, "d": d}


def _cross_mag_sq_int(a: List[int], b: List[int], c: List[int]) -> int:
    ab = _vec_sub_int(b[:3], a[:3])
    ac = _vec_sub_int(c[:3], a[:3])
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return cx * cx + cy * cy + cz * cz


def _compute_rt_exact(a: List[int], b: List[int], c: List[int]):
    """Returns (Q, s) where Q=[int,int,int] and s=[{n,d},{n,d},{n,d}]."""
    Q1 = _quadrance_int(b, c)
    Q2 = _quadrance_int(c, a)
    Q3 = _quadrance_int(a, b)

    ab = _vec_sub_int(b[:3], a[:3])
    ac = _vec_sub_int(c[:3], a[:3])
    bc = _vec_sub_int(c[:3], b[:3])
    ba = _vec_sub_int(a[:3], b[:3])
    ca = _vec_sub_int(a[:3], c[:3])
    cb = _vec_sub_int(b[:3], c[:3])

    sA = _spread_pair(ab, ac)
    sB = _spread_pair(bc, ba)
    sC = _spread_pair(ca, cb)
    if sA is None or sB is None or sC is None:
        raise ValueError("Degenerate triangle (zero quadrance)")
    return [Q1, Q2, Q3], [sA, sB, sC]


def _exact_cross_law_check(Q, s):
    """Cross law integer check. Returns True if satisfied."""
    Q1, Q2, Q3 = Q
    s3n, s3d = s[2]["n"], s[2]["d"]
    t = Q1 + Q2 - Q3
    lhs = t * t * s3d
    rhs = 4 * Q1 * Q2 * (s3d - s3n)
    return lhs, rhs, lhs == rhs


def _exact_triple_spread_check(s):
    """Triple spread integer check. Returns True if satisfied."""
    n1, d1 = s[0]["n"], s[0]["d"]
    n2, d2 = s[1]["n"], s[1]["d"]
    n3, d3 = s[2]["n"], s[2]["d"]
    cd = d1 * d2 * d3
    sum_num = n1 * d2 * d3 + n2 * d1 * d3 + n3 * d1 * d2
    lhs = sum_num * sum_num
    s1sq = n1 * n1 * d2 * d2 * d3 * d3
    s2sq = n2 * n2 * d1 * d1 * d3 * d3
    s3sq = n3 * n3 * d1 * d1 * d2 * d2
    prod = n1 * n2 * n3 * d1 * d2 * d3
    rhs = 2 * (s1sq + s2sq + s3sq) + 4 * prod
    return lhs, rhs, lhs == rhs


_EXACT_LAW_FNS = {
    "RT_LAW_04": _exact_cross_law_check,
    "RT_LAW_05": _exact_triple_spread_check,
}


# Exact-mode triangle generators (integer coords)

def _gen_scalene_int(rng: random.Random, scale: float = 10.0) -> List[List[int]]:
    """Random scalene triangle with integer coords in [-scale, scale]."""
    s = int(scale)
    if s < 2:
        s = 2
    while True:
        pts = [[rng.randint(-s, s) for _ in range(3)] for _ in range(3)]
        if _cross_mag_sq_int(pts[0], pts[1], pts[2]) > 0:
            return pts


def _gen_right_int(rng: random.Random, scale: float = 10.0) -> List[List[int]]:
    """Right triangle at origin with integer coords."""
    s = max(1, int(scale))
    while True:
        l1 = rng.randint(1, s)
        l2 = rng.randint(1, s)
        a = [0, 0, 0]
        b = [l1, 0, 0]
        c = [0, l2, 0]
        if _cross_mag_sq_int(a, b, c) > 0:
            return [a, b, c]


def _gen_near_degenerate_int(rng: random.Random, scale: float = 10.0) -> List[List[int]]:
    """Nearly collinear triangle with integer coords (small cross product)."""
    s = max(2, int(scale))
    while True:
        a = [0, 0, 0]
        b = [rng.randint(s // 2, s), 0, 0]
        c = [rng.randint(1, s - 1), rng.choice([1, -1]), 0]
        if _cross_mag_sq_int(a, b, c) > 0:
            return [a, b, c]


def _build_generators_exact(near_degenerate_rate: Optional[float] = None):
    if near_degenerate_rate is not None:
        nd = max(0.0, min(1.0, near_degenerate_rate))
        remaining = 1.0 - nd
        return [
            ("scalene_int", _gen_scalene_int, remaining * 0.7),
            ("right_int", _gen_right_int, remaining * 0.3),
            ("near_degenerate_int", _gen_near_degenerate_int, nd),
        ]
    return [
        ("scalene_int", _gen_scalene_int, 0.60),
        ("right_int", _gen_right_int, 0.30),
        ("near_degenerate_int", _gen_near_degenerate_int, 0.10),
    ]


def build_cert_exact(
    index: int,
    triangle_type: str,
    verts: List[List[int]],
    laws: List[str],
    timestamp: str,
) -> Dict[str, Any]:
    """Build a v2 exact-mode cert."""
    a, b, c = verts
    Q, s = _compute_rt_exact(a, b, c)

    cert_id = f"qa_rt_corpus_exact_{index:06d}"
    object_id = f"tri_{index:06d}"

    scene_raw = {
        "version": "1.0",
        "timestamp": timestamp,
        "instances": [{
            "id": f"tri-{index:06d}",
            "type": "Triangle",
            "parameters": {"triangle_type": triangle_type},
            "transform": {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "scale": {"x": 1, "y": 1, "z": 1},
            },
            "appearance": {"opacity": 1.0, "visible": True},
            "metadata": {"label": f"corpus_exact_{triangle_type}_{index:06d}"},
        }],
        "metadata": {"instanceCount": 1},
    }
    scene_raw_sha = _sha256_hex(_canonical_json_compact(scene_raw))

    parsed_obj = {
        "object_id": object_id,
        "object_type": "TRIANGLE",
        "label": f"Corpus exact {triangle_type} triangle #{index}",
        "vertices": [
            {"id": "A", "coord": list(a)},
            {"id": "B", "coord": list(b)},
            {"id": "C", "coord": list(c)},
        ],
        "faces": [["A", "B", "C"]],
    }

    steps = []
    step_num = 1

    # Step 1: ART_PARSE_SCENE
    s1_inputs = {"scene_raw_sha256": scene_raw_sha}
    s1_outputs = {"parsed_object_ids": [object_id]}
    steps.append({
        "step_id": f"s{step_num}",
        "move_id": "ART_PARSE_SCENE",
        "inputs": s1_inputs,
        "outputs": s1_outputs,
        "step_hash_sha256": _step_hash("ART_PARSE_SCENE", s1_inputs, s1_outputs),
    })
    step_num += 1

    # Step 2: RT_COMPUTE_TRIANGLE_INVARIANTS
    s2_inputs = {"triangle_object_id": object_id}
    s2_outputs = {"Q": Q, "s": s}
    steps.append({
        "step_id": f"s{step_num}",
        "move_id": "RT_COMPUTE_TRIANGLE_INVARIANTS",
        "inputs": s2_inputs,
        "outputs": s2_outputs,
        "step_hash_sha256": _step_hash("RT_COMPUTE_TRIANGLE_INVARIANTS", s2_inputs, s2_outputs),
    })
    step_num += 1

    # Law verification steps (exact)
    for law_id in laws:
        if law_id not in _EXACT_LAW_FNS:
            continue
        law_fn = _EXACT_LAW_FNS[law_id]
        if law_id == "RT_LAW_05":
            lhs, rhs, ok = law_fn(s)
        else:
            lhs, rhs, ok = law_fn(Q, s)

        law_inputs = {"law_id": law_id, "Q": Q, "s": s, "triangle_object_id": object_id}
        law_outputs = {"verified": ok, "lhs": lhs, "rhs": rhs}
        steps.append({
            "step_id": f"s{step_num}",
            "move_id": "RT_VALIDATE_LAW_EQUATION",
            "inputs": law_inputs,
            "outputs": law_outputs,
            "step_hash_sha256": _step_hash("RT_VALIDATE_LAW_EQUATION", law_inputs, law_outputs),
        })
        step_num += 1

    cert = {
        "schema_version": "v2",
        "cert_type": "QA_ARTEXPLORER_SCENE_ADAPTER.v2",
        "cert_id": cert_id,
        "created_utc": timestamp,
        "compute_substrate": "qa_rational_pair_noreduce",
        "source_semantics": {
            "upstream": {
                "name": "ARTexplorer (Algebraic Rational Trigonometry Explorer)",
                "repo_url": "https://github.com/arossti/ARTexplorer",
                "app_url": "https://arossti.github.io/ARTexplorer/",
            },
            "export": {
                "format": "artexplorer_scene_json",
                "exported_by": "qa_rt_corpus_generator.py (exact mode)",
            },
        },
        "base_algebra": {
            "name": "Q",
            "properties": {"integral_domain": True, "field": True, "no_zero_divisors": True},
        },
        "scene": {
            "coordinate_system": "XYZ",
            "scene_raw": scene_raw,
            "scene_raw_sha256": scene_raw_sha,
        },
        "derivation": {
            "parsed_objects": [parsed_obj],
            "steps": steps,
        },
        "result": {
            "rt_invariants": {
                "triangles": [{"triangle_object_id": object_id, "Q": Q, "s": s}],
            },
            "invariant_diff": {
                "steps": [
                    {"step_id": "s1", "adds": [f"parsed_objects[{object_id}]"]},
                    {"step_id": "s2", "adds": [
                        f"rt_invariants.triangles[{object_id}].Q",
                        f"rt_invariants.triangles[{object_id}].s",
                    ]},
                ] + [
                    {"step_id": f"s{3 + i}", "adds": [f"law_verification[{law}].verified"]}
                    for i, law in enumerate(laws) if law in _EXACT_LAW_FNS
                ],
                "provenance": {
                    "upstream_tool": "qa_rt_corpus_generator.py (exact mode)",
                    "upstream_format": "synthetic (integer coordinates)",
                    "triangle_type": triangle_type,
                    "compute_substrate": "qa_rational_pair_noreduce",
                },
            },
        },
        "determinism_contract": {
            "canonical_json": True,
            "stable_sorting": True,
            "no_rng": True,
            "invariant_diff_defined": True,
        },
    }
    return cert


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Certified RT scene corpus generator")
    ap.add_argument("--count", type=int, default=50, help="Number of certs to generate")
    ap.add_argument("--output-dir", default="qa_rt_corpus/", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--laws", default="RT_LAW_04,RT_LAW_05",
                    help="Comma-separated law IDs to verify")
    ap.add_argument("--near-degenerate-rate", type=float, default=None,
                    help="Override near-degenerate triangle fraction (0.0-1.0)")
    ap.add_argument("--coord-scale", type=float, default=10.0,
                    help="Coordinate magnitude scale (default 10.0)")
    ap.add_argument("--no-write", action="store_true",
                    help="Skip writing individual cert files (stats-only mode)")
    ap.add_argument("--substrate", default="float64", choices=["float64", "exact"],
                    help="Compute substrate: float64 (v1) or exact (v2 rational pairs)")
    args = ap.parse_args(argv)

    laws = [l.strip() for l in args.laws.split(",") if l.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    use_exact = args.substrate == "exact"

    if use_exact:
        generators = _build_generators_exact(args.near_degenerate_rate)
    else:
        generators = _build_generators(args.near_degenerate_rate)

    rng = random.Random(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Import validator directly for speed (avoids subprocess per cert)
    if use_exact:
        validator_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "qa_artexplorer_scene_adapter_v2")
    else:
        validator_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "qa_artexplorer_scene_adapter_v1")
    sys.path.insert(0, validator_dir)
    from validator import validate_cert, GateStatus as VS
    sys.path.pop(0)

    passed = 0
    failed = 0
    type_counts: Dict[str, int] = {}
    type_fail_counts: Dict[str, int] = defaultdict(int)
    q_min = float("inf")
    q_max = float("-inf")
    s_min = float("inf")
    s_max = float("-inf")

    # Residual tracking: law_id -> list of (residual, cert_index, triangle_type)
    residual_log: Dict[str, List[Tuple[float, int, str]]] = defaultdict(list)

    # Worst-case tracking: law_id -> (worst_residual, worst_index, worst_type)
    worst_case: Dict[str, Tuple[float, int, str]] = {}

    progress_interval = max(1, args.count // 20)

    coord_scale = args.coord_scale

    for i in range(args.count):
        tri_type, gen_fn = _pick_generator(rng, generators)
        verts = gen_fn(rng, scale=coord_scale)

        if use_exact:
            cert = build_cert_exact(i, tri_type, verts, laws, timestamp)
            law_residuals = {}  # exact mode: no residuals, just pass/fail
        else:
            cert, law_residuals = build_cert(i, tri_type, verts, laws, timestamp)

        if not args.no_write:
            out_path = os.path.join(args.output_dir, f"cert_{i:06d}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cert, f, indent=2, sort_keys=False)

        # Validate directly (no subprocess)
        res = validate_cert(cert)
        ok = all(r.status == VS.PASS for r in res)
        if ok:
            passed += 1
        else:
            failed += 1
            type_fail_counts[tri_type] += 1
            fail_gates = [r.gate for r in res if r.status == VS.FAIL]
            if failed <= 10:  # Only print first 10 failures
                print(f"FAIL #{i}: type={tri_type} gates={fail_gates}")

        type_counts[tri_type] = type_counts.get(tri_type, 0) + 1

        # Track residuals
        for law_id, residual in law_residuals.items():
            residual_log[law_id].append((residual, i, tri_type))
            if law_id not in worst_case or residual > worst_case[law_id][0]:
                worst_case[law_id] = (residual, i, tri_type)

        # Track Q/s ranges
        tri = cert["result"]["rt_invariants"]["triangles"][0]
        for q in tri["Q"]:
            q_min = min(q_min, q)
            q_max = max(q_max, q)
        if not use_exact:
            for sv in tri["s"]:
                s_min = min(s_min, sv)
                s_max = max(s_max, sv)
        else:
            for sv in tri["s"]:
                if isinstance(sv, dict) and sv["d"] != 0:
                    val = sv["n"] / sv["d"]
                    s_min = min(s_min, val)
                    s_max = max(s_max, val)

        # Progress
        if (i + 1) % progress_interval == 0:
            pct = 100 * (i + 1) / args.count
            print(f"  [{pct:5.1f}%] {i+1}/{args.count}  passed={passed} failed={failed}",
                  file=sys.stderr)

    # Compute residual statistics
    residual_stats = {}
    for law_id, entries in residual_log.items():
        residuals = [r for r, _, _ in entries]
        residuals.sort()
        n = len(residuals)
        mean = sum(residuals) / n if n else 0
        median = residuals[n // 2] if n else 0
        p95 = residuals[int(n * 0.95)] if n else 0
        p99 = residuals[int(n * 0.99)] if n else 0
        worst_r, worst_i, worst_t = worst_case[law_id]

        # Per-type breakdown
        by_type: Dict[str, Dict[str, Any]] = {}
        type_residuals: Dict[str, List[float]] = defaultdict(list)
        for r, _, t in entries:
            type_residuals[t].append(r)
        for t, rs in type_residuals.items():
            rs.sort()
            nt = len(rs)
            by_type[t] = {
                "count": nt,
                "max": rs[-1],
                "mean": sum(rs) / nt,
                "p95": rs[int(nt * 0.95)] if nt > 1 else rs[-1],
            }

        residual_stats[law_id] = {
            "count": n,
            "min": residuals[0] if n else 0,
            "max": residuals[-1] if n else 0,
            "mean": mean,
            "median": median,
            "p95": p95,
            "p99": p99,
            "worst_cert_index": worst_i,
            "worst_cert_type": worst_t,
            "by_triangle_type": by_type,
        }

    # Write manifest
    manifest = {
        "generated_utc": timestamp,
        "seed": args.seed,
        "count": args.count,
        "laws_verified": laws,
        "passed": passed,
        "failed": failed,
        "failure_rate": failed / args.count if args.count else 0,
        "triangle_type_distribution": type_counts,
        "triangle_type_failures": dict(type_fail_counts) if type_fail_counts else {},
        "q_range": [q_min, q_max],
        "s_range": [s_min, s_max],
        "residual_stats": residual_stats,
        "coord_scale": coord_scale,
        "near_degenerate_rate": args.near_degenerate_rate,
        "compute_substrate": "exact" if use_exact else "float64",
        "validator_rel_tolerance": 0 if use_exact else 1e-9,
    }
    manifest_path = os.path.join(args.output_dir, "CORPUS_MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Summary
    print(f"\nCorpus: {passed} passed, {failed} failed out of {args.count}")
    if type_fail_counts:
        print(f"Failures by type: {dict(type_fail_counts)}")
    for law_id, stats in residual_stats.items():
        print(f"{law_id}: max={stats['max']:.2e}  mean={stats['mean']:.2e}  "
              f"p95={stats['p95']:.2e}  p99={stats['p99']:.2e}  "
              f"worst=cert_{stats['worst_cert_index']:06d} ({stats['worst_cert_type']})")
    print(f"Manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
