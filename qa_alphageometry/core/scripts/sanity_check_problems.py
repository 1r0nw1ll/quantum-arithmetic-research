#!/usr/bin/env python3
# scripts/sanity_check_problems.py
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set

# Based on working fixtures p01..p10
FACT_ARITY = {
    "Parallel": 2,
    "Perpendicular": 2,
    "CoincidentLines": 2,
    "EqualLength": 2,
    "OnLine": 2,
    "OnCircle": 2,
    "ConcentricCircles": 2,
    "Collinear": 3,
    "Concyclic": 4,
}

REQUIRED_KEYS = {"id", "description", "givens", "goals", "difficulty"}

def _err(errors: List[str], path: Path, msg: str) -> None:
    errors.append(f"{path.name}: {msg}")

def validate_fact(errors: List[str], path: Path, where: str, idx: int, fact_obj: Any) -> Tuple[str, Tuple[int, ...]] | None:
    if not isinstance(fact_obj, dict) or len(fact_obj) != 1:
        _err(errors, path, f"{where}[{idx}] must be a single-key dict, got: {fact_obj!r}")
        return None
    fact_type = next(iter(fact_obj.keys()))
    payload = fact_obj[fact_type]

    if fact_type not in FACT_ARITY:
        _err(errors, path, f"{where}[{idx}] unknown fact type '{fact_type}'")
        return None

    if not isinstance(payload, list):
        _err(errors, path, f"{where}[{idx}] '{fact_type}' payload must be a list, got: {type(payload).__name__}")
        return None

    arity = FACT_ARITY[fact_type]
    if len(payload) != arity:
        _err(errors, path, f"{where}[{idx}] '{fact_type}' arity={arity}, got {len(payload)}: {payload}")
        return None

    if not all(isinstance(x, int) for x in payload):
        _err(errors, path, f"{where}[{idx}] '{fact_type}' payload must be ints, got: {payload}")
        return None

    # trivial invalids
    if arity == 2 and payload[0] == payload[1]:
        _err(errors, path, f"{where}[{idx}] '{fact_type}' has identical ids: {payload}")
        return None
    if any(x < 0 for x in payload):
        _err(errors, path, f"{where}[{idx}] '{fact_type}' has negative id: {payload}")
        return None

    return fact_type, tuple(payload)

def validate_problem(path: Path) -> List[str]:
    errors: List[str] = []
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        return [f"{path.name}: JSON parse error: {e}"]

    missing = REQUIRED_KEYS - set(obj.keys()) if isinstance(obj, dict) else REQUIRED_KEYS
    if missing:
        return [f"{path.name}: missing required keys: {sorted(missing)}"]

    if not isinstance(obj["givens"], list) or not isinstance(obj["goals"], list):
        return [f"{path.name}: 'givens' and 'goals' must be lists"]

    seen: Set[Tuple[str, Tuple[int, ...]]] = set()
    for where in ("givens", "goals"):
        for i, f in enumerate(obj[where]):
            parsed = validate_fact(errors, path, where, i, f)
            if parsed:
                if parsed in seen:
                    _err(errors, path, f"duplicate fact in {where}[{i}]: {parsed[0]}{list(parsed[1])}")
                seen.add(parsed)

    if not isinstance(obj["id"], str) or not obj["id"]:
        _err(errors, path, "'id' must be a non-empty string")
    if not isinstance(obj["description"], str):
        _err(errors, path, "'description' must be a string")
    if not isinstance(obj["difficulty"], int):
        _err(errors, path, "'difficulty' must be an int")

    return errors

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/sanity_check_problems.py <file_or_dir> [<file_or_dir>...]")
        return 2

    all_paths: List[Path] = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            all_paths.extend(sorted(p.rglob("*.json")))
        else:
            all_paths.append(p)

    all_errors: List[str] = []
    for p in all_paths:
        all_errors.extend(validate_problem(p))

    if all_errors:
        print("❌ Sanity check FAILED:\n")
        print("\n".join(all_errors))
        return 1

    print(f"✅ Sanity check passed for {len(all_paths)} JSON file(s).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
