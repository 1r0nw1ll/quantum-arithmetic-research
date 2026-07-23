#!/usr/bin/env python3
"""Stage 5 interpretable rule-family mining for QA arithmetic targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Iterator


DOMAIN = "QA_QUANTUM_ARITHMETIC_RULE_FAMILY_MINING_STAGE5.v1"


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def factor_count(n: int) -> int:
    count = 0
    temp = n
    while temp % 2 == 0 and temp > 1:
        count += 1
        temp //= 2
    divisor = 3
    while divisor <= math.isqrt(temp):
        while temp % divisor == 0:
            count += 1
            temp //= divisor
        divisor += 2
    if temp > 1:
        count += 1
    return count


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    limit = math.isqrt(n)
    while divisor <= limit:
        if n % divisor == 0:
            return False
        divisor += 2
    return True


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    temp = n
    while temp % 2 == 0 and temp > 1:
        factors.append(2)
        temp //= 2
    divisor = 3
    while divisor <= math.isqrt(temp):
        while temp % divisor == 0:
            factors.append(divisor)
            temp //= divisor
        divisor += 2
    if temp > 1:
        factors.append(temp)
    return factors


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def largest_prime_factor(n: int) -> int:
    factors = prime_factors(n)
    return max(factors) if factors else 1


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = e + d
    B = b * b
    E = e * e
    D = d * d
    A = a * a
    X = e * d
    C = e * d * 2
    F = a * b
    G = D + E
    J = d * b
    K = d * a
    W = d * (e + a)
    return {
        "b": b,
        "e": e,
        "d": d,
        "a": a,
        "B": B,
        "E": E,
        "D": D,
        "A": A,
        "X": X,
        "C": C,
        "F": F,
        "G": G,
        "J": J,
        "K": K,
        "W": W,
        "Y": A - D,
        "Z": E + K,
    }


def label_for(row: dict[str, int], target: str) -> int:
    X = row["X"]
    F = row["F"]
    W = row["W"]
    G = row["G"]
    if target == "X_semiprime":
        return int(is_semiprime(X))
    if target == "F_semiprime":
        return int(is_semiprime(F))
    if target == "W_semiprime":
        return int(is_semiprime(W))
    if target == "G_prime":
        return int(is_prime(G))
    if target == "squarefree_X":
        return int(is_squarefree(X))
    if target == "X_omega_3":
        return int(factor_count(X) == 3)
    if target == "X_lpf_ge_sqrt":
        return int(largest_prime_factor(X) >= math.isqrt(X))
    raise ValueError(f"unknown target: {target}")


def square_pairs(start: int, end: int) -> Iterator[tuple[int, int]]:
    for b in range(start, end + 1):
        for e in range(start, end + 1):
            yield b, e


def band_pairs(b_start: int, b_end: int, e_start: int, e_end: int) -> Iterator[tuple[int, int]]:
    for b in range(b_start, b_end + 1):
        for e in range(e_start, e_end + 1):
            yield b, e


def random_pairs(limit: int, count: int, seed: int) -> Iterator[tuple[int, int]]:
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < count:
        pair = (rng.randint(1, limit), rng.randint(1, limit))
        if pair not in seen:
            seen.add(pair)
            yield pair


def build_windows(args: argparse.Namespace) -> list[dict[str, object]]:
    return [
        {"name": "square_101_300", "pairs": lambda: square_pairs(101, 300), "total": 200 * 200},
        {"name": "square_3001_10000", "pairs": lambda: square_pairs(3001, 10000), "total": 7000 * 7000},
        {"name": "band_b101_1000_e1_100", "pairs": lambda: band_pairs(101, 1000, 1, 100), "total": 900 * 100},
        {"name": "band_b1_100_e101_1000", "pairs": lambda: band_pairs(1, 100, 101, 1000), "total": 100 * 900},
        {
            "name": "random_sparse_1e6",
            "pairs": lambda: random_pairs(1_000_000, args.random_count, args.seed + 1000),
            "total": args.random_count,
        },
    ]


def sample_pairs(
    pairs: Iterable[tuple[int, int]],
    total_pairs: int | None,
    cap: int,
    seed: int,
) -> tuple[list[dict[str, int]], bool]:
    if total_pairs is not None and total_pairs <= cap:
        return [qa_values(b, e) for b, e in pairs], False
    if total_pairs is None:
        rows = []
        for index, (b, e) in enumerate(pairs):
            if index >= cap:
                break
            rows.append(qa_values(b, e))
        return rows, True
    rng = random.Random(seed)
    selected_indices = set(rng.sample(range(total_pairs), cap))
    rows = []
    for index, (b, e) in enumerate(pairs):
        if index in selected_indices:
            rows.append(qa_values(b, e))
            if len(rows) == cap:
                break
    return rows, True


def mask_for(rows: list[dict[str, int]], predicate: Callable[[dict[str, int]], bool]) -> int:
    mask = 0
    for index, row in enumerate(rows):
        if predicate(row):
            mask |= 1 << index
    return mask


def label_mask(rows: list[dict[str, int]], target: str) -> int:
    mask = 0
    for index, row in enumerate(rows):
        if label_for(row, target):
            mask |= 1 << index
    return mask


def metrics_from_mask(rule_mask: int, labels: int, total: int) -> dict[str, float | int]:
    tp = (rule_mask & labels).bit_count()
    support = rule_mask.bit_count()
    positives = labels.bit_count()
    fp = support - tp
    fn = positives - tp
    tn = total - tp - fp - fn
    precision = tp / support if support else 0.0
    recall = tp / positives if positives else 0.0
    base_rate = positives / total if total else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
    }


def residue_predicates(fields: list[str], moduli: list[int]) -> list[dict[str, object]]:
    predicates = []
    for field in fields:
        for modulus in moduli:
            for residue in range(modulus):
                predicates.append(
                    {
                        "family": "single_residue",
                        "rule": f"{field} mod {modulus} == {residue}",
                        "func": lambda row, field=field, modulus=modulus, residue=residue: row[field] % modulus == residue,
                    }
                )
    return predicates


def gcd_predicates() -> list[dict[str, object]]:
    pairs = [("X", "W"), ("X", "F"), ("F", "W"), ("b", "e"), ("d", "a"), ("C", "F")]
    predicates = []
    for left, right in pairs:
        predicates.append(
            {
                "family": "gcd_predicate",
                "rule": f"gcd({left},{right}) == 1",
                "func": lambda row, left=left, right=right: math.gcd(row[left], row[right]) == 1,
            }
        )
        predicates.append(
            {
                "family": "gcd_predicate",
                "rule": f"gcd({left},{right}) > 1",
                "func": lambda row, left=left, right=right: math.gcd(row[left], row[right]) > 1,
            }
        )
    return predicates


def parity_predicates() -> list[dict[str, object]]:
    predicates = []
    for b_parity in [0, 1]:
        for e_parity in [0, 1]:
            predicates.append(
                {
                    "family": "parity_class",
                    "rule": f"b parity {b_parity} AND e parity {e_parity}",
                    "func": lambda row, b_parity=b_parity, e_parity=e_parity: (
                        row["b"] % 2 == b_parity and row["e"] % 2 == e_parity
                    ),
                }
            )
    return predicates


def qa_identity_predicates() -> list[dict[str, object]]:
    return [
        {"family": "qa_identity", "rule": "X divides W", "func": lambda row: row["W"] % row["X"] == 0},
        {"family": "qa_identity", "rule": "F divides W", "func": lambda row: row["F"] != 0 and row["W"] % row["F"] == 0},
        {"family": "qa_identity", "rule": "G is odd", "func": lambda row: row["G"] % 2 == 1},
        {"family": "qa_identity", "rule": "W > 2*D", "func": lambda row: row["W"] > 2 * row["D"]},
        {"family": "qa_identity", "rule": "X <= D", "func": lambda row: row["X"] <= row["D"]},
        {"family": "qa_identity", "rule": "F <= W", "func": lambda row: row["F"] <= row["W"]},
        {"family": "qa_identity", "rule": "C < F", "func": lambda row: row["C"] < row["F"]},
        {"family": "qa_identity", "rule": "A-D is square", "func": lambda row: int(math.isqrt(row["Y"])) * int(math.isqrt(row["Y"])) == row["Y"]},
    ]


def reduced_pair(left: int, right: int) -> tuple[int, int]:
    if left == 0 and right == 0:
        return (0, 0)
    divisor = math.gcd(abs(left), abs(right))
    return (left // divisor, right // divisor)


def slope_key(row: dict[str, int]) -> tuple[int, int]:
    return reduced_pair(row["e"], row["b"])


def origin_ray_key(row: dict[str, int]) -> tuple[int, int]:
    return reduced_pair(row["e"] - 2, row["b"] - 1)


def frequent_value_predicates(
    rows: list[dict[str, int]],
    labels: int,
    family: str,
    key_fn: Callable[[dict[str, int]], object],
    label_fn: Callable[[object], str],
    min_support: int,
    top_values: int,
) -> list[dict[str, object]]:
    buckets: dict[object, int] = {}
    for index, row in enumerate(rows):
        key = key_fn(row)
        buckets[key] = buckets.get(key, 0) | (1 << index)
    scored = []
    for key, mask in buckets.items():
        metric = metrics_from_mask(mask, labels, len(rows))
        if int(metric["support"]) >= min_support:
            scored.append((float(metric["lift"]), float(metric["precision"]), int(metric["support"]), key))
    predicates = []
    for _, _, _, key in sorted(scored, reverse=True)[:top_values]:
        predicates.append(
            {
                "family": family,
                "rule": label_fn(key),
                "func": lambda row, key=key, key_fn=key_fn: key_fn(row) == key,
            }
        )
    return predicates


def distance_band(row: dict[str, int], band_size: int) -> int:
    delta_b = row["b"] - 1
    delta_e = row["e"] - 2
    return ((delta_b * delta_b) + (delta_e * delta_e)) // band_size


def orbit_key(row: dict[str, int], modulus: int) -> tuple[int, int, int, int]:
    return (row["b"] % modulus, row["e"] % modulus, row["d"] % modulus, row["a"] % modulus)


def build_base_predicates(
    train_rows: list[dict[str, int]],
    train_labels: int,
    fields: list[str],
    moduli: list[int],
    min_support: int,
    top_values: int,
    distance_band_size: int,
) -> list[dict[str, object]]:
    predicates = []
    predicates.extend(residue_predicates(fields, moduli))
    predicates.extend(gcd_predicates())
    predicates.extend(parity_predicates())
    predicates.extend(qa_identity_predicates())
    predicates.extend(
        frequent_value_predicates(
            train_rows,
            train_labels,
            "slope_ray",
            slope_key,
            lambda key: f"reduced e:b == {key[0]}:{key[1]}",
            min_support,
            top_values,
        )
    )
    predicates.extend(
        frequent_value_predicates(
            train_rows,
            train_labels,
            "origin_ray",
            origin_ray_key,
            lambda key: f"reduced (e-2):(b-1) == {key[0]}:{key[1]}",
            min_support,
            top_values,
        )
    )
    predicates.extend(
        frequent_value_predicates(
            train_rows,
            train_labels,
            "distance_band",
            lambda row: distance_band(row, distance_band_size),
            lambda key: f"distance_band_{distance_band_size} == {key}",
            min_support,
            top_values,
        )
    )
    for modulus in [m for m in moduli if m <= 11]:
        predicates.extend(
            frequent_value_predicates(
                train_rows,
                train_labels,
                "modular_orbit_class",
                lambda row, modulus=modulus: (modulus, orbit_key(row, modulus)),
                lambda key: f"(b,e,d,a) mod {key[0]} == {key[1]}",
                min_support,
                top_values,
            )
        )
    return predicates


def materialize_masks(rows: list[dict[str, int]], predicates: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for predicate in predicates:
        out.append(
            {
                "family": predicate["family"],
                "rule": predicate["rule"],
                "mask": mask_for(rows, predicate["func"]),
                "func": predicate["func"],
            }
        )
    return out


def compatible(rule_parts: list[dict[str, object]]) -> bool:
    seen = set()
    for part in rule_parts:
        label = str(part["rule"])
        if label in seen:
            return False
        seen.add(label)
    return True


def combine_family_name(parts: list[dict[str, object]]) -> str:
    families = sorted({str(part["family"]) for part in parts})
    return "+".join(families)


def evaluate_selected_on_test(
    selected: list[dict[str, object]],
    test_rows: list[dict[str, int]],
    test_labels: int,
) -> list[dict[str, object]]:
    by_rule = {str(item["rule"]): item for item in selected}
    needed = []
    for item in selected:
        for part in item["parts"]:
            needed.append(part)
    unique_parts = {str(part["rule"]): part for part in needed}
    test_parts = materialize_masks(test_rows, list(unique_parts.values()))
    test_masks = {str(part["rule"]): int(part["mask"]) for part in test_parts}
    rows = []
    for rule, item in by_rule.items():
        mask = (1 << len(test_rows)) - 1
        for part in item["parts"]:
            mask &= test_masks[str(part["rule"])]
        metrics = metrics_from_mask(mask, test_labels, len(test_rows))
        row = {
            "rule": rule,
            "family": item["family"],
            "rule_size": item["rule_size"],
            "train_precision": item["train"]["precision"],
            "train_recall": item["train"]["recall"],
            "train_f1": item["train"]["f1"],
            "train_lift": item["train"]["lift"],
            "train_support": item["train"]["support"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1": metrics["f1"],
            "test_lift": metrics["lift"],
            "test_support": metrics["support"],
            "test_true_positive": metrics["true_positive"],
        }
        rows.append(row)
    return rows


def top_candidates(
    parts: list[dict[str, object]],
    labels: int,
    total: int,
    min_support: int,
    top_k: int,
    max_single_for_conjunctions: int,
    max_triples: int,
) -> list[dict[str, object]]:
    scored: list[dict[str, object]] = []
    for part in parts:
        metrics = metrics_from_mask(int(part["mask"]), labels, total)
        if int(metrics["support"]) >= min_support:
            scored.append(
                {
                    "family": part["family"],
                    "rule": part["rule"],
                    "rule_size": 1,
                    "parts": [part],
                    "mask": int(part["mask"]),
                    "train": metrics,
                }
            )
    ranked_single = sorted(
        scored,
        key=lambda item: (float(item["train"]["lift"]), float(item["train"]["precision"]), int(item["train"]["support"])),
        reverse=True,
    )
    seed_parts = [item["parts"][0] for item in ranked_single[:max_single_for_conjunctions]]
    for left, right in combinations(seed_parts, 2):
        if not compatible([left, right]):
            continue
        mask = int(left["mask"]) & int(right["mask"])
        metrics = metrics_from_mask(mask, labels, total)
        if int(metrics["support"]) >= min_support:
            parts_pair = [left, right]
            scored.append(
                {
                    "family": combine_family_name(parts_pair),
                    "rule": " AND ".join(str(part["rule"]) for part in parts_pair),
                    "rule_size": 2,
                    "parts": parts_pair,
                    "mask": mask,
                    "train": metrics,
                }
            )
    triples_seen = 0
    triple_seed = seed_parts[: min(36, len(seed_parts))]
    for first, second, third in combinations(triple_seed, 3):
        if triples_seen >= max_triples:
            break
        parts_three = [first, second, third]
        if not compatible(parts_three):
            continue
        mask = int(first["mask"]) & int(second["mask"]) & int(third["mask"])
        metrics = metrics_from_mask(mask, labels, total)
        if int(metrics["support"]) >= min_support:
            triples_seen += 1
            scored.append(
                {
                    "family": combine_family_name(parts_three),
                    "rule": " AND ".join(str(part["rule"]) for part in parts_three),
                    "rule_size": 3,
                    "parts": parts_three,
                    "mask": mask,
                    "train": metrics,
                }
            )
    return sorted(
        scored,
        key=lambda item: (
            float(item["train"]["lift"]),
            float(item["train"]["precision"]),
            float(item["train"]["recall"]),
            -int(item["rule_size"]),
        ),
        reverse=True,
    )[:top_k]


def null_max_lift(labels: int, total: int, support: int, iterations: int, seed: int) -> float:
    if support <= 0 or total <= 0:
        return 0.0
    positives = labels.bit_count()
    if positives <= 0:
        return 0.0
    rng = random.Random(seed)
    max_lift = 0.0
    base_rate = positives / total
    indices = list(range(total))
    for _ in range(iterations):
        selected = rng.sample(indices, min(support, total))
        hits = 0
        for index in selected:
            if (labels >> index) & 1:
                hits += 1
        precision = hits / support
        max_lift = max(max_lift, precision / base_rate if base_rate else 0.0)
    return max_lift


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [piece.strip() for piece in args.fields.split(",") if piece.strip()]
    moduli = [int(piece.strip()) for piece in args.moduli.split(",") if piece.strip()]
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    wanted_windows = {piece.strip() for piece in args.windows.split(",") if piece.strip()}
    windows = [window for window in build_windows(args) if window["name"] in wanted_windows]

    train_rows = [qa_values(b, e) for b, e in square_pairs(1, 100)]
    all_rule_rows = []
    ledger_rows = []
    target_summaries = {}
    for target_index, target in enumerate(targets):
        train_labels = label_mask(train_rows, target)
        predicates = build_base_predicates(
            train_rows,
            train_labels,
            fields,
            moduli,
            args.min_support,
            args.top_values,
            args.distance_band_size,
        )
        train_parts = materialize_masks(train_rows, predicates)
        selected = top_candidates(
            train_parts,
            train_labels,
            len(train_rows),
            args.min_support,
            args.top_rules,
            args.max_single_for_conjunctions,
            args.max_triples,
        )
        target_summaries[target] = {
            "train_positive_rows": train_labels.bit_count(),
            "candidate_predicates": len(predicates),
            "selected_rules": len(selected),
        }
        for window_index, window in enumerate(windows):
            test_rows, sampled = sample_pairs(
                window["pairs"](),
                window["total"],
                args.sample_cap,
                args.seed + window_index,
            )
            test_labels = label_mask(test_rows, target)
            tested_rules = evaluate_selected_on_test(selected, test_rows, test_labels)
            for rank, rule_row in enumerate(
                sorted(tested_rules, key=lambda row: (float(row["test_lift"]), float(row["test_f1"])), reverse=True),
                start=1,
            ):
                rule_hash = domain_sha256(
                    f"{DOMAIN}.rule",
                    canonical_json(
                        {
                            "target": target,
                            "train_window": "square_1_100",
                            "test_window": window["name"],
                            "rule": rule_row["rule"],
                            "family": rule_row["family"],
                        }
                    ),
                )
                row = {
                    "target": target,
                    "train_window": "square_1_100",
                    "test_window": window["name"],
                    "sampled": sampled,
                    "rank": rank,
                    "rule": rule_row["rule"],
                    "family": rule_row["family"],
                    "rule_size": rule_row["rule_size"],
                    "train_precision": rule_row["train_precision"],
                    "train_recall": rule_row["train_recall"],
                    "train_f1": rule_row["train_f1"],
                    "train_lift": rule_row["train_lift"],
                    "train_support": rule_row["train_support"],
                    "test_precision": rule_row["test_precision"],
                    "test_recall": rule_row["test_recall"],
                    "test_f1": rule_row["test_f1"],
                    "test_lift": rule_row["test_lift"],
                    "test_support": rule_row["test_support"],
                    "test_true_positive": rule_row["test_true_positive"],
                    "hash": rule_hash,
                }
                all_rule_rows.append(row)
            best = max(tested_rules, key=lambda row: (float(row["test_lift"]), float(row["test_f1"])), default=None)
            if best is None:
                observed_lift = 0.0
                support = 0
                verdict = "NO_RULES"
            else:
                observed_lift = float(best["test_lift"])
                support = int(best["test_support"])
                verdict = "INTERPRETABLE_SIGNAL"
            null_lift = null_max_lift(
                test_labels,
                len(test_rows),
                support,
                args.null_iterations,
                args.seed + 100000 + target_index * 1000 + window_index,
            )
            if best is None or int(best["test_true_positive"]) < args.min_test_hits:
                verdict = "LOW_SUPPORT"
            elif observed_lift <= null_lift:
                verdict = "NULL_COMPETITIVE"
            ledger_hash = domain_sha256(
                f"{DOMAIN}.ledger",
                canonical_json(
                    {
                        "target": target,
                        "train_window": "square_1_100",
                        "test_window": window["name"],
                        "feature_set": "interpretable_rule_families",
                        "model": "best_out_of_window_symbolic_rule",
                        "observed_lift": observed_lift,
                        "null_max_lift": null_lift,
                        "verdict": verdict,
                    }
                ),
            )
            ledger_rows.append(
                {
                    "target": target,
                    "train_window": "square_1_100",
                    "test_window": window["name"],
                    "feature_set": "interpretable_rule_families",
                    "model": "best_out_of_window_symbolic_rule",
                    "observed_lift": observed_lift,
                    "null_max_lift": null_lift,
                    "verdict": verdict,
                    "hash": ledger_hash,
                }
            )

    rules_csv = out_dir / args.rules_csv
    ledger_csv = out_dir / args.ledger_csv
    write_csv(rules_csv, all_rule_rows)
    write_csv(ledger_csv, ledger_rows)
    payload = {
        "stage_id": "qa_quantum_arithmetic_rule_family_mining_stage5",
        "hypothesis": (
            "Readable QA coordinate rule families should show which residue, gcd, orbit, slope, distance, and "
            "identity-derived predicates persist out-of-window, distinct from opaque Hebbian score enrichment."
        ),
        "parameters": {
            "fields": fields,
            "moduli": moduli,
            "targets": targets,
            "windows": [window["name"] for window in windows],
            "sample_cap": args.sample_cap,
            "min_support": args.min_support,
            "min_test_hits": args.min_test_hits,
            "top_values": args.top_values,
            "top_rules": args.top_rules,
            "max_single_for_conjunctions": args.max_single_for_conjunctions,
            "max_triples": args.max_triples,
            "distance_band_size": args.distance_band_size,
            "null_iterations": args.null_iterations,
            "seed": args.seed,
        },
        "train": {"window": "square_1_100", "rows": len(train_rows), "targets": target_summaries},
        "artifacts": {"rules_csv": str(rules_csv), "ledger_csv": str(ledger_csv)},
        "ledger": ledger_rows,
        "honest_interpretation": (
            "Rules are selected on the train window but ranked by external test lift for inspection. Ledger verdicts "
            "compare the best external rule against same-density random positive controls. This is interpretable "
            "pattern mining, not a proof of a target law."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    summary_path = out_dir / args.summary_json
    summary_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            fields="b,e,d,a,X,F,G,W",
            moduli="2,3,4,5",
            targets="X_semiprime,F_semiprime",
            windows="square_101_300",
            sample_cap=300,
            min_support=4,
            min_test_hits=1,
            top_values=8,
            top_rules=12,
            max_single_for_conjunctions=18,
            max_triples=200,
            distance_band_size=25,
            null_iterations=2,
            random_count=300,
            seed=71,
            rules_csv="qa_quantum_arithmetic_rule_family_stage5_rules.csv",
            ledger_csv="qa_quantum_arithmetic_result_ledger_stage6.csv",
            summary_json="qa_quantum_arithmetic_rule_family_stage5.json",
        )
        payload = run(args)
        ok = (
            len(payload["ledger"]) == 2
            and Path(payload["artifacts"]["rules_csv"]).exists()
            and Path(payload["artifacts"]["ledger_csv"]).exists()
        )
        return {"ok": ok, "ledger_rows": len(payload["ledger"])}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5 interpretable QA arithmetic rule-family miner.")
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--fields", default="b,e,d,a,X,F,G,W")
    parser.add_argument("--moduli", default="2,3,4,5,7,8,9,11,13")
    parser.add_argument("--targets", default="X_semiprime,F_semiprime,W_semiprime,X_omega_3,squarefree_X")
    parser.add_argument(
        "--windows",
        default="square_101_300,square_3001_10000,band_b101_1000_e1_100,band_b1_100_e101_1000,random_sparse_1e6",
    )
    parser.add_argument("--sample-cap", type=int, default=30000)
    parser.add_argument("--min-support", type=int, default=12)
    parser.add_argument("--min-test-hits", type=int, default=5)
    parser.add_argument("--top-values", type=int, default=16)
    parser.add_argument("--top-rules", type=int, default=40)
    parser.add_argument("--max-single-for-conjunctions", type=int, default=60)
    parser.add_argument("--max-triples", type=int, default=2000)
    parser.add_argument("--distance-band-size", type=int, default=250)
    parser.add_argument("--null-iterations", type=int, default=25)
    parser.add_argument("--random-count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--rules-csv", default="qa_quantum_arithmetic_rule_family_stage5_rules.csv")
    parser.add_argument("--ledger-csv", default="qa_quantum_arithmetic_result_ledger_stage6.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_rule_family_stage5.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    verdicts: dict[str, int] = {}
    for row in payload["ledger"]:
        verdict = str(row["verdict"])
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    print(f"[qa_quantum_arithmetic_rule_family_stage5] wrote {payload['artifacts']['rules_csv']}")
    print(f"[qa_quantum_arithmetic_rule_family_stage5] wrote {payload['artifacts']['ledger_csv']}")
    print(f"[qa_quantum_arithmetic_rule_family_stage5] ledger_rows={len(payload['ledger'])}")
    print(f"[qa_quantum_arithmetic_rule_family_stage5] verdicts={canonical_json(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
