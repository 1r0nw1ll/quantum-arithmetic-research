#!/usr/bin/env python3
"""QA Orbit Theorem Map validator.

Validates exported orbit theorem-map JSON by recomputing QA orbit families and
multibase features from integer arithmetic, then replaying declared leaf paths.
This intentionally does not import the workbench that generated the map.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from qa_orbit_rules import orbit_family  # noqa: E402

QA_COMPLIANCE = "cert_validator — orbit_theorem_map; integer QA orbit classification; no float state; orbit_family from qa_orbit_rules"

SCHEMA_VERSION = "QA_ORBIT_THEOREM_MAP_CERT.v1"
CERT_SLUG = "qa_orbit_theorem_map_cert_v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valuation(n: int, p: int) -> int:
    if n == 0:
        return 99
    n = abs(n)
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def prime_basis_features(name: str, n: int, primes: list[int]) -> dict[str, str]:
    feats = {
        f"{name}.sign": "neg" if n < 0 else "zero" if n == 0 else "pos",
        f"{name}.abs_bucket": "0" if n == 0 else "1" if abs(n) == 1 else "many",
    }
    for p in primes:
        pp = p * p
        feats[f"{name}.r{p}"] = str(n % p)
        feats[f"{name}.r{pp}"] = str(n % pp)
        feats[f"{name}.v{p}"] = str(valuation(n, p))
        feats[f"{name}.unit{p}"] = "0" if n % p == 0 else "1"
    return feats


def qa_tuple_features(b: int, e: int, primes: list[int], modulus: int) -> dict[str, str]:
    d = b + e
    a = b + 2 * e
    c = 2 * d * e
    f = a * b
    g = d * d + e * e
    h = c + f
    i = abs(c - f)
    values = {"b": b, "e": e, "d": d, "a": a, "C": c, "F": f, "G": g, "H": h, "I": i}
    gcd_be = gcd(abs(b), abs(e))
    gcd_v = {p: valuation(gcd_be, p) for p in primes}
    modulus_v = {p: valuation(modulus, p) for p in primes}
    line3 = {p: (e - 3 * b) % p == 0 for p in primes}
    feats: dict[str, str] = {
        "qa.gcd_be": str(gcd_be),
        "qa.parity_de": f"{d % 2}{e % 2}",
        "qa.CF_order": "C_gt_F" if c > f else "C_eq_F" if c == f else "F_gt_C",
    }
    for key, value in values.items():
        feats.update(prime_basis_features(key, value, primes))
    for p in primes:
        gp = gcd_v[p]
        mp = modulus_v[p]
        feats[f"m.v{p}"] = str(mp)
        feats[f"qa.v{p}_gcd_be"] = str(gp)
        feats[f"qa.v{p}_G_minus_2gcd"] = str(valuation(g, p) - 2 * gp)
        feats[f"qa.r{p}_e_minus_3b"] = str((e - 3 * b) % p)
        feats[f"qa.r{p}_e_minus_2b"] = str((e - 2 * b) % p)
        feats[f"qa.r{p}_e_plus_b"] = str((e + b) % p)
        feats[f"qa.r{p}_b_zero"] = "1" if b % p == 0 else "0"
        feats[f"qa.r{p}_e_zero"] = "1" if e % p == 0 else "0"
        feats[f"qa.r{p}_be_both_zero"] = "1" if b % p == 0 and e % p == 0 else "0"
        feats[f"qa.r{p}_be_line_3"] = "1" if line3[p] else "0"
        feats[f"qa.v{p}_gcd_minus_m"] = str(gp - mp)
        feats[f"qa.v{p}_gcd_eq_m"] = "1" if gp == mp else "0"
        feats[f"qa.v{p}_gcd_ge_m"] = "1" if gp >= mp else "0"
        feats[f"qa.v{p}_gcd_ge_m_minus1"] = "1" if gp >= max(mp - 1, 0) else "0"
        feats[f"qa.v{p}_gcd_level_m"] = (
            "above" if gp > mp else "at" if gp == mp else "one_below" if gp == mp - 1 else "below"
        )
    for scale_p in primes:
        if modulus_v[scale_p] == 0:
            continue
        for line_p in primes:
            if line_p == scale_p or modulus_v[line_p] == 0:
                continue
            suffix = f"v{scale_p}_gcd_ge_m_and_r{line_p}_line_3"
            feats[f"qa.{suffix}"] = "1" if gcd_v[scale_p] >= modulus_v[scale_p] and line3[line_p] else "0"
            suffix = f"v{scale_p}_gcd_eq_m_and_r{line_p}_line_3"
            feats[f"qa.{suffix}"] = "1" if gcd_v[scale_p] == modulus_v[scale_p] and line3[line_p] else "0"
    return feats


def qa_step(b: int, e: int, m: int) -> tuple[int, int]:
    return e, ((b + e - 1) % m) + 1


def orbit_rows_for_moduli(moduli: list[int], primes: list[int]) -> list[dict[str, Any]]:
    rows = []
    idx = 0
    for m in moduli:
        for b in range(1, m + 1):
            for e in range(1, m + 1):
                values = {"m": m, "b": b, "e": e}
                feats: dict[str, str] = {}
                for key, value in values.items():
                    feats.update(prime_basis_features(key, value, primes))
                feats.update(qa_tuple_features(b, e, primes, m))
                rows.append({"index": idx, "values": values, "label": orbit_family(b, e, m), "features": feats})
                idx += 1
    return rows


def split_rows(rows: list[dict[str, Any]], split_within_moduli: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not split_within_moduli:
        return rows, rows
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["values"]["m"], row["label"])].append(row)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda r: (r["values"]["b"], r["values"]["e"]))
        if len(bucket) == 1:
            train_rows.append(bucket[0])
            continue
        for idx, row in enumerate(bucket):
            (test_rows if idx % 3 == 0 else train_rows).append(row)
    return train_rows, test_rows


def modulus_regime(m: int, primes: list[int]) -> str:
    parts = []
    for p in primes:
        v = valuation(m, p)
        if v:
            parts.append(f"{p}^{v}")
    return "*".join(parts) if parts else "unit"


def path_matches(row: dict[str, Any], path: list[dict[str, str]]) -> bool:
    feats = row["features"]
    for step in path:
        actual = feats.get(step["feature"])
        expected = step["value"]
        if step["branch"] == "yes" and actual != expected:
            return False
        if step["branch"] == "no" and actual == expected:
            return False
    return True


def predict_from_leaf_paths(node: dict[str, Any], row: dict[str, Any]) -> tuple[str | None, list[str]]:
    hits = []
    for idx, leaf in enumerate(node.get("leaf_paths", [])):
        if path_matches(row, leaf.get("path", [])):
            hits.append((idx, leaf.get("predict")))
    if len(hits) != 1:
        return None, [f"TM_PATH: row {row['values']} matched {len(hits)} leaves in {node.get('id')}"]
    return hits[0][1], []


def evaluate_node(node: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    errors = 0
    problems: list[str] = []
    for row in rows:
        pred, pred_errors = predict_from_leaf_paths(node, row)
        problems.extend(pred_errors)
        actual = row["label"]
        confusion[actual][pred or "(none)"] += 1
        if pred != actual:
            errors += 1
    total = len(rows)
    accuracy_num = total - errors
    accuracy = f"{accuracy_num}/{total}" if total and errors else "1/1" if total else "0/1"
    return {
        "rows": total,
        "errors": errors,
        "accuracy": accuracy,
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
    }, problems


def validate_root_predicate(node: dict[str, Any]) -> list[str]:
    if node.get("kind") not in {"exact_decision_tree", "exact_regime_tree"}:
        return []
    root = node.get("root_predicate")
    if root is None and node.get("id") == "orbit.global_tree":
        return []
    leaf_paths = node.get("leaf_paths", [])
    if not leaf_paths or not leaf_paths[0].get("path"):
        return [f"TM_SCHEMA: node {node.get('id')} has no leaf path root"]
    observed = {
        "feature": leaf_paths[0]["path"][0]["feature"],
        "value": leaf_paths[0]["path"][0]["value"],
    }
    if root is not None and root != observed:
        return [f"TM_ROOT: node {node.get('id')} root_predicate {root} != first path root {observed}"]
    return []


def compare_eval(node_id: str, field: str, declared: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    if declared != recomputed:
        return [f"TM_EVAL: {node_id}.{field} declared {declared} != recomputed {recomputed}"]
    return []


def validate_theorem_map(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("kind") != "qa_orbit_theorem_map":
        errors.append("TM_SCHEMA: kind must be qa_orbit_theorem_map")
    config = payload.get("config", {})
    primes = config.get("primes")
    train_moduli = config.get("train_moduli")
    test_moduli = config.get("test_moduli")
    if not isinstance(primes, list) or not isinstance(train_moduli, list) or not isinstance(test_moduli, list):
        errors.append("TM_SCHEMA: config requires primes, train_moduli, and test_moduli lists")
        return False, errors
    all_moduli = sorted(set(train_moduli) | set(test_moduli))
    rows = orbit_rows_for_moduli(all_moduli, primes)
    train_rows, test_rows = split_rows(rows, bool(config.get("split_within_moduli")))
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        errors.append("TM_SCHEMA: nodes must be a list")
        return False, errors
    by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
    global_node = by_id.get("orbit.global_tree")
    if global_node:
        errors.extend(validate_root_predicate(global_node))
        train_eval, problems = evaluate_node(global_node, train_rows)
        errors.extend(problems)
        errors.extend(compare_eval("orbit.global_tree", "train_eval", global_node.get("train_eval"), train_eval))
        test_eval, problems = evaluate_node(global_node, test_rows)
        errors.extend(problems)
        errors.extend(compare_eval("orbit.global_tree", "test_eval", global_node.get("test_eval"), test_eval))
    else:
        errors.append("TM_SCHEMA: missing orbit.global_tree node")
    for node in nodes:
        if not isinstance(node, dict) or node.get("kind") != "exact_regime_tree":
            continue
        regime = node.get("regime")
        errors.extend(validate_root_predicate(node))
        regime_train = [r for r in train_rows if modulus_regime(r["values"]["m"], primes) == regime]
        regime_test = [r for r in test_rows if modulus_regime(r["values"]["m"], primes) == regime]
        train_eval, problems = evaluate_node(node, regime_train)
        errors.extend(problems)
        errors.extend(compare_eval(str(node.get("id")), "train_eval", node.get("train_eval"), train_eval))
        test_eval, problems = evaluate_node(node, regime_test)
        errors.extend(problems)
        errors.extend(compare_eval(str(node.get("id")), "test_eval", node.get("test_eval"), test_eval))
    cover = by_id.get("orbit.per_regime_cover")
    if cover:
        total_rows = total_errors = 0
        for node in nodes:
            if isinstance(node, dict) and node.get("kind") == "exact_regime_tree":
                eval_obj = node.get("test_eval", {})
                total_rows += int(eval_obj.get("rows", 0))
                total_errors += int(eval_obj.get("errors", 0))
        overall = {
            "rows": total_rows,
            "errors": total_errors,
            "accuracy": "1/1" if total_rows and total_errors == 0 else f"{total_rows - total_errors}/{total_rows}" if total_rows else "0/1",
        }
        if cover.get("overall_test") != overall:
            errors.append(f"TM_COVER: declared overall {cover.get('overall_test')} != recomputed {overall}")
    return len(errors) == 0, errors


def run_self_test(cert_dir: Path) -> tuple[bool, dict[str, Any]]:
    fixture_dir = cert_dir / "fixtures"
    pass_fixtures = sorted(fixture_dir.glob("pass_*.json"))
    fail_fixtures = sorted(fixture_dir.glob("fail_*.json"))
    errors: list[str] = []
    for path in pass_fixtures:
        ok, fixture_errors = validate_theorem_map(load_json(path))
        if not ok:
            errors.append(f"SELF_PASS: {path.name} failed: {fixture_errors}")
    for path in fail_fixtures:
        fixture = load_json(path)
        expected = fixture.get("expected_fail_type")
        ok, fixture_errors = validate_theorem_map(fixture)
        if ok:
            errors.append(f"SELF_FAIL: {path.name} unexpectedly passed")
        elif expected and not any(str(err).startswith(str(expected)) for err in fixture_errors):
            errors.append(f"SELF_FAIL: {path.name} did not trigger {expected}: {fixture_errors}")
    payload = {
        "ok": len(errors) == 0,
        "slug": CERT_SLUG,
        "schema_version": SCHEMA_VERSION,
        "pass_fixtures": len(pass_fixtures),
        "fail_fixtures": len(fail_fixtures),
        "errors": errors,
    }
    return payload["ok"], payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA orbit theorem-map JSON")
    parser.add_argument("theorem_map", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    cert_dir = Path(__file__).resolve().parent
    if args.self_test:
        ok, payload = run_self_test(cert_dir)
        print(canonical_json(payload))
        return 0 if ok else 1
    if not args.theorem_map:
        parser.error("provide theorem-map JSON or --self-test")
    ok, errors = validate_theorem_map(load_json(args.theorem_map))
    print(canonical_json({"ok": ok, "errors": errors}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
