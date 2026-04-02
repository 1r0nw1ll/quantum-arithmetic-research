#!/usr/bin/env python3
QA_COMPLIANCE = "observer=legacy_script, state_alphabet=mod24"
"""
qa_entity_encoder.py — Encode entities into QA tuples with overrides

Entry point:
  python qa_entity_encoder.py --in artifacts/knowledge/qa_entities.json \
                              --overrides qa_entity_overrides.yaml \
                              --out artifacts/knowledge/qa_entity_encodings.json

Notes
- Deterministic hash-based mapping from entity name → (b,e), then d=(b+e)%24, a=(b+2e)%24.
- Supports manual overrides for key terms via qa_entity_overrides.yaml.
- Computes simple invariants and E8 alignment using QA Lab utilities if available.

Usage
  - Minimal: python qa_entity_encoder.py
  - With overrides: python qa_entity_encoder.py --overrides qa_entity_overrides.yaml
  - Prints a short summary upon completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None


# Try to use QA Lab utilities for E8 roots and alignment
def _import_fastpath():
    try:
        # namespace package-style import
        import qa_lab.qa_fastpath as fp  # type: ignore
        return fp
    except Exception:
        pass
    try:
        # fallback: add qa_lab to sys.path and import module
        sys.path.append(os.path.abspath("qa_lab"))
        import qa_fastpath as fp  # type: ignore
        return fp
    except Exception:
        return None


def _import_e8_simple():
    try:
        import qa_lab.qa_e8_alignment as e8  # type: ignore
        return e8
    except Exception:
        try:
            sys.path.append(os.path.abspath("qa_lab"))
            import qa_e8_alignment as e8  # type: ignore
            return e8
        except Exception:
            return None


def hash_to_qa_tuple(name: str, modulus: int = 24) -> Tuple[int, int, int, int]:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    b = int.from_bytes(h[0:4], "big") % modulus
    e = int.from_bytes(h[4:8], "big") % modulus
    d = (b + e) % modulus
    a = (b + 2 * e) % modulus
    return int(b), int(e), int(d), int(a)


def apply_overrides(name: str, b: int, e: int, overrides: Dict[str, dict] | None) -> Tuple[int, int]:
    if not overrides:
        return b, e
    ov = overrides.get(name)
    if not ov:
        return b, e
    b2 = ov.get("b", b)
    e2 = ov.get("e", e)
    # allow explicit tuple override if provided
    if "tuple" in ov and isinstance(ov["tuple"], (list, tuple)) and len(ov["tuple"]) == 4:
        tb, te, td, ta = ov["tuple"]
        return int(tb) % 24, int(te) % 24
    return int(b2) % 24, int(e2) % 24


def triangle_residual(b: int, e: int, d: int, a: int) -> float:
    # As in qa_fastpath.triangle_gate residual
    c = 2.0 * e * d
    f = b * a
    g = e * e + d * d
    res = abs(c * c + f * f - g * g)
    return float(res)


def compute_e8_alignment_for_tuple(fp, e8_module, b: int, e: int, d: int, a: int) -> float:
    # Prefer qa_fastpath build + e8_scores_auto with real roots
    try:
        if fp is not None:
            roots_info = fp.get_e8_roots()
            if roots_info is not None:
                roots, _unit = roots_info
                import numpy as np  # local import
                vec = fp.build_e8_vectors(np.array([b], dtype=float),
                                          np.array([e], dtype=float),
                                          np.array([d], dtype=float),
                                          np.array([a], dtype=float))
                scores = fp.e8_scores_auto(vec, roots)
                return float(scores[0])
    except Exception:
        pass
    # Fallback: simple 4D ideal-root cosine in e8_module
    if e8_module is not None:
        try:
            return float(e8_module.e8_alignment_single(float(b), float(e), float(d), float(a)))
        except Exception:
            pass
    # Last resort
    return 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Encode entities into QA tuples with optional overrides")
    parser.add_argument("--in", dest="in_path", default="artifacts/knowledge/qa_entities.json",
                        help="Input entities JSON path (from extractor)")
    parser.add_argument("--overrides", dest="overrides_path", default="qa_entity_overrides.yaml",
                        help="YAML file with manual overrides for key entities")
    parser.add_argument("--out", dest="out_path", default="artifacts/knowledge/qa_entity_encodings.json",
                        help="Output JSON path for encodings")
    args = parser.parse_args(argv)

    if not os.path.exists(args.in_path):
        print(f"[ERROR] Entities JSON not found: {args.in_path}", file=sys.stderr)
        return 2

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entities = data.get("entities", [])

    overrides: Dict[str, dict] | None = None
    if yaml is not None and os.path.exists(args.overrides_path):
        with open(args.overrides_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        overrides = y.get("overrides", {}) if isinstance(y, dict) else {}
    else:
        overrides = {}

    fp = _import_fastpath()
    e8_module = _import_e8_simple()

    encodings: List[Dict] = []
    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        b, e, d, a = hash_to_qa_tuple(name)
        b, e = apply_overrides(name, b, e, overrides)
        d = (b + e) % 24
        a = (b + 2 * e) % 24
        # simple deterministic loss from triangle residual, normalized
        tri_res = triangle_residual(b, e, d, a)
        scale = 1.0 + (b*b + e*e + d*d + a*a)
        loss = float(tri_res / scale)
        e8 = compute_e8_alignment_for_tuple(fp, e8_module, b, e, d, a)
        k = 0.1
        hi = float(e8 * math.exp(-k * loss))

        encodings.append({
            "name": name,
            "slug": ent.get("slug"),
            "section": ent.get("section"),
            "definition": ent.get("definition", ""),
            **({"symbol": ent.get("symbol")} if ent.get("symbol") else {}),
            "b": int(b), "e": int(e), "d": int(d), "a": int(a),
            "loss": loss,
            "e8_alignment": e8,
            "hi": hi,
        })

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": os.path.abspath(args.in_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "count": len(encodings),
            "encodings": encodings,
        }, f, indent=2, ensure_ascii=False)

    print(f"[qa_entity_encoder] Encoded {len(encodings)} entities → {args.out_path}")
    return 0


if __name__ == "__main__":  # --- Main ---
    raise SystemExit(main())

