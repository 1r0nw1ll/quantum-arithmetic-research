#!/usr/bin/env python3
"""Stage 31 orbit-dynamic transition targets with leak controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_orbit_rules import orbit_family, orbit_period, qa_step


DOMAIN = "QA_QUANTUM_ARITHMETIC_ORBIT_DYNAMIC_TRANSITION_STAGE31.v1"
MODULI = (9, 24)
ORBIT_FAMILY_CODE = {"cosmos": 0, "satellite": 1, "singularity": 2}
DEFAULT_TARGETS = (
    "orbit9_satellite_control,"
    "orbit24_satellite_control,"
    "orbit9_24_family_disagree_control,"
    "period9_eq_period24_control,"
    "t1_X_omega_increases,"
    "t1_F_omega_increases,"
    "t1_X_squarefree_flip,"
    "t1_F_squarefree_flip,"
    "t1_X_semiprime_flip,"
    "t1_F_semiprime_flip,"
    "t1_gcd_X_current_next_gt1,"
    "t1_gcd_F_current_next_gt1,"
    "t2_X_omega_same,"
    "t2_F_squarefree_same"
)

ORBIT_DERIVED_TARGETS = {
    "orbit9_satellite_control",
    "orbit24_satellite_control",
    "orbit9_24_family_disagree_control",
    "period9_eq_period24_control",
}


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def qa_mod_coord(value: int, modulus: int) -> int:
    return ((value - 1) % modulus) + 1


def prime_factors(n: int) -> list[int]:
    factors: list[int] = []
    remaining = n
    while remaining > 1 and remaining % 2 == 0:
        factors.append(2)
        remaining //= 2
    p = 3
    while p * p <= remaining:
        while remaining % p == 0:
            factors.append(p)
            remaining //= p
        p += 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def factor_count(n: int) -> int:
    return len(prime_factors(n))


def is_squarefree(n: int) -> bool:
    factors = prime_factors(n)
    return len(factors) == len(set(factors))


def is_semiprime(n: int) -> bool:
    return factor_count(n) == 2


def cap(n: int, high: int = 8) -> int:
    return min(high, n)


def qa_values(b: int, e: int) -> dict[str, int]:
    d = b + e
    a = b + 2 * e
    D = d * d
    X = e * d
    F = a * b
    G = D + e * e
    return {"b": b, "e": e, "d": d, "a": a, "D": D, "X": X, "F": F, "G": G}


def t_integer_state(b: int, e: int, steps: int) -> tuple[int, int]:
    cur_b = b
    cur_e = e
    for _ in range(steps):
        cur_b, cur_e = cur_e, cur_b + cur_e
    return cur_b, cur_e


def orbit_family_for(row: dict[str, int], modulus: int) -> str:
    b = qa_mod_coord(row["b"], modulus)
    e = qa_mod_coord(row["e"], modulus)
    return orbit_family(int(b), int(e), modulus)


def orbit_family_code(row: dict[str, int], modulus: int) -> int:
    return ORBIT_FAMILY_CODE[orbit_family_for(row, modulus)]


def orbit_index_map(m: int) -> dict[tuple[int, int], int]:
    seen: set[tuple[int, int]] = set()
    out: dict[tuple[int, int], int] = {}
    index = 0
    for b in range(1, m + 1):
        for e in range(1, m + 1):
            if (b, e) in seen:
                continue
            cur = (b, e)
            while cur not in seen:
                seen.add(cur)
                out[cur] = index
                cur = qa_step(cur[0], cur[1], m)
            index += 1
    return out


ORBIT_INDEX = {9: orbit_index_map(9), 24: orbit_index_map(24)}


def orbit_id(row: dict[str, int], modulus: int) -> int:
    return ORBIT_INDEX[modulus][(qa_mod_coord(row["b"], modulus), qa_mod_coord(row["e"], modulus))]


def dynamic_values(row: dict[str, int], steps: int) -> dict[str, int]:
    b, e = t_integer_state(row["b"], row["e"], steps)
    return qa_values(b, e)


def label_for(row: dict[str, int], target: str) -> int:
    t1 = dynamic_values(row, 1)
    if target == "orbit9_satellite_control":
        return int(orbit_family_for(row, 9) == "satellite")
    if target == "orbit24_satellite_control":
        return int(orbit_family_for(row, 24) == "satellite")
    if target == "orbit9_24_family_disagree_control":
        return int(orbit_family_for(row, 9) != orbit_family_for(row, 24))
    if target == "period9_eq_period24_control":
        p9 = orbit_period(qa_mod_coord(row["b"], 9), qa_mod_coord(row["e"], 9), 9)
        p24 = orbit_period(qa_mod_coord(row["b"], 24), qa_mod_coord(row["e"], 24), 24)
        return int(p9 == p24)
    if target == "t1_X_omega_increases":
        return int(factor_count(t1["X"]) > factor_count(row["X"]))
    if target == "t1_F_omega_increases":
        return int(factor_count(t1["F"]) > factor_count(row["F"]))
    if target == "t1_X_squarefree_flip":
        return int(is_squarefree(t1["X"]) != is_squarefree(row["X"]))
    if target == "t1_F_squarefree_flip":
        return int(is_squarefree(t1["F"]) != is_squarefree(row["F"]))
    if target == "t1_X_semiprime_flip":
        return int(is_semiprime(t1["X"]) != is_semiprime(row["X"]))
    if target == "t1_F_semiprime_flip":
        return int(is_semiprime(t1["F"]) != is_semiprime(row["F"]))
    if target == "t1_gcd_X_current_next_gt1":
        return int(math.gcd(row["X"], t1["X"]) > 1)
    if target == "t1_gcd_F_current_next_gt1":
        return int(math.gcd(row["F"], t1["F"]) > 1)
    if target == "t2_X_omega_same":
        t2 = dynamic_values(row, 2)
        return int(factor_count(t2["X"]) == factor_count(row["X"]))
    if target == "t2_F_squarefree_same":
        t2 = dynamic_values(row, 2)
        return int(is_squarefree(t2["F"]) == is_squarefree(row["F"]))
    raise ValueError(f"unknown target: {target}")


FeatureFunc = Callable[[dict[str, int]], object]


def feature_sets() -> dict[str, tuple[str, FeatureFunc]]:
    return {
        "b_only9": ("non_orbit", lambda row: qa_mod_coord(row["b"], 9)),
        "e_only9": ("non_orbit", lambda row: qa_mod_coord(row["e"], 9)),
        "b_only24": ("non_orbit", lambda row: qa_mod_coord(row["b"], 24)),
        "e_only24": ("non_orbit", lambda row: qa_mod_coord(row["e"], 24)),
        "be_pair9": ("non_orbit", lambda row: (qa_mod_coord(row["b"], 9), qa_mod_coord(row["e"], 9))),
        "be_pair24": ("non_orbit", lambda row: (qa_mod_coord(row["b"], 24), qa_mod_coord(row["e"], 24))),
        "current_factor_signature": (
            "non_orbit",
            lambda row: (
                cap(factor_count(row["X"])),
                cap(factor_count(row["F"])),
                int(is_squarefree(row["X"])),
                int(is_squarefree(row["F"])),
                int(is_semiprime(row["X"])),
                int(is_semiprime(row["F"])),
            ),
        ),
        "dynamic_factor_signature": (
            "non_orbit",
            lambda row: (
                cap(factor_count(row["X"])),
                cap(factor_count(dynamic_values(row, 1)["X"])),
                cap(factor_count(row["F"])),
                cap(factor_count(dynamic_values(row, 1)["F"])),
                int(is_squarefree(row["X"])),
                int(is_squarefree(dynamic_values(row, 1)["X"])),
                int(is_squarefree(row["F"])),
                int(is_squarefree(dynamic_values(row, 1)["F"])),
            ),
        ),
        "qa_orbit_family9": ("orbit", lambda row: orbit_family_code(row, 9)),
        "qa_orbit_id9": ("orbit", lambda row: orbit_id(row, 9)),
        "qa_orbit_family24": ("orbit", lambda row: orbit_family_code(row, 24)),
        "qa_orbit_id24": ("orbit", lambda row: orbit_id(row, 24)),
        "qa_orbit_family9_24": ("orbit", lambda row: (orbit_family_code(row, 9), orbit_family_code(row, 24))),
    }


def square_rows(start: int, end: int) -> list[dict[str, int]]:
    return [qa_values(b, e) for b in range(start, end + 1) for e in range(start, end + 1)]


def score_predictions(predictions: list[int], labels: list[int]) -> dict[str, float | int]:
    tp = sum(1 for pred, truth in zip(predictions, labels) if pred == 1 and truth == 1)
    fp = sum(1 for pred, truth in zip(predictions, labels) if pred == 1 and truth == 0)
    fn = sum(1 for pred, truth in zip(predictions, labels) if pred == 0 and truth == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = sum(labels) / len(labels) if labels else 0.0
    return {
        "predicted_positive": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
    }


def best_threshold(scores: list[float], labels: list[int]) -> float:
    grouped: list[tuple[float, int, int]] = []
    for score, label in sorted(zip(scores, labels), reverse=True):
        if grouped and grouped[-1][0] == score:
            prior_score, prior_total, prior_positive = grouped[-1]
            grouped[-1] = (prior_score, prior_total + 1, prior_positive + label)
        else:
            grouped.append((score, 1, label))
    total_positive = sum(labels)
    predicted_positive = 0
    true_positive = 0
    best_f1 = -1.0
    selected = grouped[0][0] if grouped else 0.0
    for threshold, total, positive in grouped:
        predicted_positive += total
        true_positive += positive
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / total_positive if total_positive else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            selected = threshold
    return selected


def train_category_rate(categories: list[object], labels: list[int]) -> dict[str, object]:
    base = sum(labels) / len(labels) if labels else 0.0
    counts: dict[str, list[int]] = {}
    for category, label in zip(categories, labels):
        key = repr(category)
        if key not in counts:
            counts[key] = [0, 0]
        counts[key][0] += 1
        counts[key][1] += label
    rates = {key: positive / total for key, (total, positive) in counts.items()}
    scores = [rates.get(repr(category), base) for category in categories]
    return {"base_rate": base, "rates": rates, "threshold": best_threshold(scores, labels)}


def evaluate_feature(train_categories: list[object], train_labels: list[int], test_categories: list[object], test_labels: list[int]) -> dict[str, float | int]:
    model = train_category_rate(train_categories, train_labels)
    rates = model["rates"]
    base = float(model["base_rate"])
    threshold = float(model["threshold"])
    scores = [rates.get(repr(category), base) for category in test_categories]
    predictions = [int(score >= threshold) for score in scores]
    return score_predictions(predictions, test_labels)


def null_summary(
    train_categories: list[object],
    train_labels: list[int],
    test_categories: list[object],
    test_labels: list[int],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    lifts = []
    f1s = []
    for _ in range(iterations):
        shuffled = list(train_labels)
        rng.shuffle(shuffled)
        row = evaluate_feature(train_categories, shuffled, test_categories, test_labels)
        lifts.append(float(row["lift"]))
        f1s.append(float(row["f1"]))
    return {
        "null_lift_mean": sum(lifts) / len(lifts) if lifts else 0.0,
        "null_lift_max": max(lifts) if lifts else 0.0,
        "null_f1_mean": sum(f1s) / len(f1s) if f1s else 0.0,
        "null_f1_max": max(f1s) if f1s else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def verdict_for(row: dict[str, object], margin: float) -> str:
    if row["target_class"] == "orbit_derived_control":
        return "ORBIT_DERIVED_CONTROL"
    if row["train_positive_rows"] < row["min_positive"] or row["train_negative_rows"] < row["min_positive"]:
        return "DEGENERATE_OR_LOW_TRAIN_SUPPORT"
    if row["test_positive_rows"] < row["min_positive"] or row["test_negative_rows"] < row["min_positive"]:
        return "DEGENERATE_OR_LOW_TEST_SUPPORT"
    if float(row["best_orbit_lift"]) <= float(row["best_orbit_null_lift_max"]):
        return "ORBIT_NULL_COMPETITIVE"
    if float(row["orbit_specific_margin"]) >= margin:
        return "ORBIT_DYNAMIC_CANDIDATE"
    return "NON_ORBIT_BASELINE_COMPETITIVE"


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [piece.strip() for piece in args.targets.split(",") if piece.strip()]
    train_rows = square_rows(args.train_start, args.train_end)
    test_rows = square_rows(args.test_start, args.test_end)
    features = feature_sets()
    train_categories = {name: [func(row) for row in train_rows] for name, (_kind, func) in features.items()}
    test_categories = {name: [func(row) for row in test_rows] for name, (_kind, func) in features.items()}
    feature_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        train_labels = [label_for(row, target) for row in train_rows]
        test_labels = [label_for(row, target) for row in test_rows]
        train_positive = sum(train_labels)
        test_positive = sum(test_labels)
        best_orbit: dict[str, object] | None = None
        best_non_orbit: dict[str, object] | None = None
        for feature_index, (feature_name, (feature_kind, _func)) in enumerate(features.items()):
            if train_positive < args.min_positive or len(train_labels) - train_positive < args.min_positive or test_positive < args.min_positive:
                observed = {"precision": None, "recall": None, "f1": None, "lift": None, "base_rate": test_positive / len(test_labels), "predicted_positive": None}
                controls = {"null_lift_mean": None, "null_lift_max": None, "null_f1_mean": None, "null_f1_max": None}
            else:
                observed = evaluate_feature(train_categories[feature_name], train_labels, test_categories[feature_name], test_labels)
                controls = null_summary(
                    train_categories[feature_name],
                    train_labels,
                    test_categories[feature_name],
                    test_labels,
                    args.null_iterations,
                    args.seed + target_index * 1231 + feature_index * 827,
                )
            payload = {
                "target": target,
                "target_class": "orbit_derived_control" if target in ORBIT_DERIVED_TARGETS else "non_circular_dynamic",
                "feature_set": feature_name,
                "feature_kind": feature_kind,
                "train_window": f"square_{args.train_start}_{args.train_end}",
                "test_window": f"square_{args.test_start}_{args.test_end}",
                "model": "category_rate_threshold",
                "train_positive_rows": train_positive,
                "test_positive_rows": test_positive,
                "test_base_rate": observed["base_rate"],
                "precision": observed["precision"],
                "recall": observed["recall"],
                "f1": observed["f1"],
                "observed_lift": observed["lift"],
                "predicted_positive": observed["predicted_positive"],
                "null_lift_mean": controls["null_lift_mean"],
                "null_lift_max": controls["null_lift_max"],
                "null_f1_mean": controls["null_f1_mean"],
                "null_f1_max": controls["null_f1_max"],
            }
            feature_rows.append({**payload, "hash": domain_sha256(f"{DOMAIN}.feature", canonical_json(payload))})
            if observed["lift"] is not None:
                if feature_kind == "orbit":
                    if best_orbit is None or float(observed["lift"]) > float(best_orbit["observed_lift"]):
                        best_orbit = {**payload}
                else:
                    if best_non_orbit is None or float(observed["lift"]) > float(best_non_orbit["observed_lift"]):
                        best_non_orbit = {**payload}
        best_orbit_lift = 0.0 if best_orbit is None else float(best_orbit["observed_lift"])
        best_non_orbit_lift = 0.0 if best_non_orbit is None else float(best_non_orbit["observed_lift"])
        best_orbit_null = 0.0 if best_orbit is None else float(best_orbit["null_lift_max"] or 0.0)
        summary = {
            "target": target,
            "target_class": "orbit_derived_control" if target in ORBIT_DERIVED_TARGETS else "non_circular_dynamic",
            "train_positive_rows": train_positive,
            "train_negative_rows": len(train_labels) - train_positive,
            "test_positive_rows": test_positive,
            "test_negative_rows": len(test_labels) - test_positive,
            "min_positive": args.min_positive,
            "test_base_rate": test_positive / len(test_labels),
            "best_orbit_feature": "" if best_orbit is None else best_orbit["feature_set"],
            "best_orbit_lift": best_orbit_lift,
            "best_orbit_null_lift_max": best_orbit_null,
            "best_non_orbit_feature": "" if best_non_orbit is None else best_non_orbit["feature_set"],
            "best_non_orbit_lift": best_non_orbit_lift,
            "orbit_specific_margin": best_orbit_lift - best_non_orbit_lift,
        }
        summary["verdict"] = verdict_for(summary, args.orbit_margin)
        summary["hash"] = domain_sha256(f"{DOMAIN}.target", canonical_json(summary))
        target_rows.append(summary)
    target_rows.sort(
        key=lambda row: (
            str(row["verdict"]) == "ORBIT_DYNAMIC_CANDIDATE",
            str(row["verdict"]) == "ORBIT_DERIVED_CONTROL",
            float(row["orbit_specific_margin"]),
            float(row["best_orbit_lift"]),
        ),
        reverse=True,
    )
    feature_rows.sort(key=lambda row: (str(row["target"]), float(row["observed_lift"] or 0.0)), reverse=True)
    target_csv = out_dir / args.target_summary_csv
    feature_csv = out_dir / args.feature_leaderboard_csv
    write_csv(target_csv, target_rows)
    write_csv(feature_csv, feature_rows)
    verdict_counts: dict[str, int] = {}
    for row in target_rows:
        verdict = str(row["verdict"])
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    payload = {
        "stage_id": "qa_quantum_arithmetic_stage31_orbit_dynamic_transition",
        "hypothesis": (
            "Orbit-derived controls should be detected as circular, while non-circular T-step arithmetic-change "
            "targets should only count if orbit features beat coordinate and dynamic factor baselines."
        ),
        "parameters": {
            "train_window": f"square_{args.train_start}_{args.train_end}",
            "test_window": f"square_{args.test_start}_{args.test_end}",
            "targets": targets,
            "orbit_derived_targets": sorted(ORBIT_DERIVED_TARGETS),
            "feature_sets": {name: kind for name, (kind, _func) in features.items()},
            "orbit_moduli": list(MODULI),
            "null_iterations": args.null_iterations,
            "min_positive": args.min_positive,
            "orbit_margin": args.orbit_margin,
            "seed": args.seed,
        },
        "artifacts": {
            "target_summary_csv": str(target_csv),
            "feature_leaderboard_csv": str(feature_csv),
        },
        "target_rows": len(target_rows),
        "feature_rows": len(feature_rows),
        "verdict_counts": verdict_counts,
        "top_targets": target_rows[:20],
        "honest_interpretation": (
            "Orbit-derived controls are harness checks, not discoveries. Non-circular dynamic labels remain "
            "candidate-only if orbit beats both nulls and the best non-orbit dynamic leakage baseline."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    json_path = out_dir / args.summary_json
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(
            out_dir=tmp,
            train_start=1,
            train_end=20,
            test_start=21,
            test_end=36,
            targets="orbit9_satellite_control,t1_X_omega_increases,t1_X_squarefree_flip,t1_gcd_X_current_next_gt1",
            null_iterations=2,
            min_positive=1,
            orbit_margin=0.05,
            seed=3131,
            target_summary_csv="stage31_target_summary.csv",
            feature_leaderboard_csv="stage31_feature_leaderboard.csv",
            summary_json="stage31_summary.json",
        )
        payload = run(args)
        ok = (
            payload["target_rows"] == 4
            and payload["feature_rows"] == 4 * len(feature_sets())
            and Path(tmp, "stage31_target_summary.csv").exists()
            and Path(tmp, "stage31_feature_leaderboard.csv").exists()
            and Path(tmp, "stage31_summary.json").exists()
            and len(payload["canonical_hash"]) == 64
        )
        return {"ok": ok, "target_rows": payload["target_rows"], "feature_rows": payload["feature_rows"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=100)
    parser.add_argument("--test-start", type=int, default=101)
    parser.add_argument("--test-end", type=int, default=300)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--null-iterations", type=int, default=50)
    parser.add_argument("--min-positive", type=int, default=10)
    parser.add_argument("--orbit-margin", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3131)
    parser.add_argument("--target-summary-csv", default="qa_quantum_arithmetic_stage31_orbit_dynamic_target_summary.csv")
    parser.add_argument("--feature-leaderboard-csv", default="qa_quantum_arithmetic_stage31_orbit_dynamic_feature_leaderboard.csv")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage31_orbit_dynamic_transition.json")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        result = self_test()
        print(canonical_json(result))
        return 0 if result["ok"] else 1
    payload = run(args)
    print(
        canonical_json(
            {
                "ok": True,
                "stage_id": payload["stage_id"],
                "target_rows": payload["target_rows"],
                "feature_rows": payload["feature_rows"],
                "verdict_counts": payload["verdict_counts"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
