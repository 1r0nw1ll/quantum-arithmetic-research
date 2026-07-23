#!/usr/bin/env python3
"""Stage 18 consolidated ledger for QA arithmetic mining results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path


DOMAIN = "QA_QUANTUM_ARITHMETIC_CONSOLIDATED_LEDGER_STAGE18.v1"


LEDGER_FILES = [
    "qa_quantum_arithmetic_scale_stage1_leaderboard.csv",
    "qa_quantum_arithmetic_pattern_targets_stage2_leaderboard.csv",
    "qa_quantum_arithmetic_stronger_nulls_stage3_leaderboard.csv",
    "qa_quantum_arithmetic_generalization_stage4_leaderboard.csv",
    "qa_quantum_arithmetic_result_ledger_stage6.csv",
    "qa_quantum_arithmetic_sweep_targets_stage7_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier1_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier23_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_fermat_features_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier4_geometry_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier5_radius_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier6_smoothness_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier7a_conics_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier7b_eccentricity_leaderboard.csv",
    "qa_quantum_arithmetic_stage8_tier7c_directrix_leaderboard.csv",
    "qa_quantum_arithmetic_stage9_cross_scale_leaderboard.csv",
    "qa_quantum_arithmetic_stage10_latus_targets_leaderboard.csv",
    "qa_quantum_arithmetic_stage11_axis_eccentricity_targets_leaderboard.csv",
    "qa_quantum_arithmetic_stage12_confocal_targets_leaderboard.csv",
    "qa_quantum_arithmetic_stage13_director_circle_targets_leaderboard.csv",
    "qa_quantum_arithmetic_stage14_evolute_curvature_targets_leaderboard.csv",
    "qa_quantum_arithmetic_stage15_tier7b_polar_pack_leaderboard.csv",
    "qa_quantum_arithmetic_stage17_tier7b_cross_scale_leaderboard.csv",
]


def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, payload: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if value not in {"", "None", "null"}:
            return value
    return ""


def to_float(value: str) -> float | None:
    if value in {"", "None", "null"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> int | None:
    if value in {"", "None", "null"}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def stage_from_name(name: str) -> str:
    pieces = name.replace("qa_quantum_arithmetic_", "").replace("_leaderboard.csv", "").replace(".csv", "")
    if pieces == "scale_stage1":
        return "stage1_scale"
    if pieces == "pattern_targets_stage2":
        return "stage2_targets"
    if pieces == "stronger_nulls_stage3":
        return "stage3_nulls"
    if pieces == "generalization_stage4":
        return "stage4_generalization"
    if pieces == "result_ledger_stage6":
        return "stage6_result_ledger"
    if pieces == "sweep_targets_stage7":
        return "stage7_sweep"
    return pieces


def identity_status_map(identity_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in identity_rows:
        for target in row["target_labels"].split(","):
            out[target.strip()] = {
                "algebraic_status": row["status"],
                "algebraic_claim": row["claim"],
                "algebraic_reason": row["algebraic_reason"],
            }
    return out


def normalize_row(source_file: str, row: dict[str, str], identities: dict[str, dict[str, str]]) -> dict[str, object]:
    target = row.get("target", "")
    observed_lift = first_nonempty(row, ["observed_lift", "lift", "threshold_lift", "top_1pct_lift"])
    null_lift = first_nonempty(row, ["null_max_lift", "null_lift_max", "same_density_top1pct_null_max_lift"])
    observed = to_float(observed_lift)
    null_value = to_float(null_lift)
    identity = identities.get(target, {})
    test_window = first_nonempty(row, ["test_window", "window"])
    out = {
        "source_stage": stage_from_name(source_file),
        "source_file": source_file,
        "target": target,
        "feature_set": first_nonempty(row, ["feature_set"]),
        "model": first_nonempty(row, ["model"]),
        "train_window": first_nonempty(row, ["train_window"]),
        "test_window": test_window,
        "verdict": first_nonempty(row, ["verdict", "serious_result"]),
        "algebraic_status": identity.get("algebraic_status", "EMPIRICAL_OPEN"),
        "algebraic_claim": identity.get("algebraic_claim", ""),
        "train_rows": to_int(first_nonempty(row, ["train_rows"])),
        "test_rows": to_int(first_nonempty(row, ["test_rows", "rows_evaluated"])),
        "train_positive_rows": to_int(first_nonempty(row, ["train_positive_rows"])),
        "test_positive_rows": to_int(first_nonempty(row, ["test_positive_rows", "positive_rows"])),
        "base_rate": to_float(first_nonempty(row, ["test_base_rate", "base_rate"])),
        "precision": to_float(first_nonempty(row, ["precision", "top_1pct_precision"])),
        "recall": to_float(first_nonempty(row, ["recall"])),
        "f1": to_float(first_nonempty(row, ["f1", "threshold_f1"])),
        "observed_lift": observed,
        "null_max_lift": null_value,
        "null_margin": (observed - null_value) if observed is not None and null_value is not None else None,
        "average_precision": to_float(first_nonempty(row, ["average_precision"])),
        "top1pct_lift": to_float(first_nonempty(row, ["top1pct_lift", "top_1pct_lift"])),
        "top1pct_hits": to_int(first_nonempty(row, ["top1pct_hits", "top_1pct_hits"])),
        "row_hash": first_nonempty(row, ["hash"]),
    }
    payload = {key: value for key, value in out.items() if key != "row_hash"}
    if not out["row_hash"]:
        out["row_hash"] = domain_sha256(f"{DOMAIN}.row", canonical_json(payload))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def target_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["target"]), []).append(row)
    summaries = []
    for target, target_rows in grouped.items():
        empirical = [row for row in target_rows if row["algebraic_status"] == "EMPIRICAL_OPEN"]
        no_parity = [row for row in empirical if row["feature_set"] == "no_parity"]
        cross_scale = [row for row in empirical if "cross_scale" in str(row["source_stage"])]
        lifts = [float(row["observed_lift"]) for row in empirical if row["observed_lift"] is not None]
        no_parity_lifts = [float(row["observed_lift"]) for row in no_parity if row["observed_lift"] is not None]
        margins = [float(row["null_margin"]) for row in empirical if row["null_margin"] is not None]
        persistent = sum(1 for row in empirical if "PERSISTENT" in str(row["verdict"]) or str(row["verdict"]) == "True")
        weak = sum(1 for row in empirical if "WEAK" in str(row["verdict"]) or "NULL" in str(row["verdict"]))
        low_support = sum(1 for row in empirical if "LOW" in str(row["verdict"]))
        identity = next((row for row in target_rows if row["algebraic_status"] != "EMPIRICAL_OPEN"), None)
        summaries.append(
            {
                "target": target,
                "algebraic_status": identity["algebraic_status"] if identity else "EMPIRICAL_OPEN",
                "algebraic_claim": identity["algebraic_claim"] if identity else "",
                "rows_total": len(target_rows),
                "empirical_rows": len(empirical),
                "cross_scale_rows": len(cross_scale),
                "persistent_rows": persistent,
                "weak_or_null_rows": weak,
                "low_support_rows": low_support,
                "best_lift": max(lifts) if lifts else None,
                "best_no_parity_lift": max(no_parity_lifts) if no_parity_lifts else None,
                "min_no_parity_lift": min(no_parity_lifts) if no_parity_lifts else None,
                "best_null_margin": max(margins) if margins else None,
                "best_top1pct_lift": max(
                    [float(row["top1pct_lift"]) for row in empirical if row["top1pct_lift"] is not None],
                    default=None,
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            0 if row["algebraic_status"] == "EMPIRICAL_OPEN" else 1,
            -float(row["best_no_parity_lift"] or 0.0),
            -float(row["best_lift"] or 0.0),
        )
    )
    return summaries


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    identity_path = out_dir / "qa_quantum_arithmetic_stage16_identity_audit_ledger.csv"
    identity_rows = read_csv(identity_path) if identity_path.exists() else []
    identities = identity_status_map(identity_rows)
    ledger_rows: list[dict[str, object]] = []
    sources_read = []
    for name in LEDGER_FILES:
        path = out_dir / name
        if not path.exists():
            continue
        sources_read.append(name)
        for row in read_csv(path):
            ledger_rows.append(normalize_row(name, row, identities))
    summaries = target_summary(ledger_rows)
    ledger_path = out_dir / args.ledger_csv
    summary_path = out_dir / args.target_summary_csv
    write_csv(ledger_path, ledger_rows)
    write_csv(summary_path, summaries)
    status_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    for row in ledger_rows:
        status = str(row["algebraic_status"])
        verdict = str(row["verdict"])
        status_counts[status] = status_counts.get(status, 0) + 1
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
    payload = {
        "stage_id": "qa_quantum_arithmetic_consolidated_ledger_stage18",
        "hypothesis": (
            "A normalized ledger can turn the scattered QA mining stage outputs into a single reviewable table with "
            "support, lift, null margin, verdict, and algebraic status."
        ),
        "parameters": {"out_dir": str(out_dir), "sources_read": sources_read},
        "artifacts": {"ledger_csv": str(ledger_path), "target_summary_csv": str(summary_path)},
        "row_count": len(ledger_rows),
        "target_count": len(summaries),
        "algebraic_status_counts": status_counts,
        "verdict_counts": verdict_counts,
        "top_empirical_targets": [
            row for row in summaries if row["algebraic_status"] == "EMPIRICAL_OPEN"
        ][:20],
        "honest_interpretation": (
            "This is a synthesis artifact, not a new experiment. It identifies what should be cross-scale-confirmed, "
            "proved, deprioritized, or excluded as closed algebra."
        ),
    }
    payload["canonical_hash"] = domain_sha256(DOMAIN, canonical_json(payload))
    json_path = out_dir / args.summary_json
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return payload


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "qa_quantum_arithmetic_stage16_identity_audit_ledger.csv").write_text(
            "claim,target_labels,status,algebraic_reason,audit_mismatches,checked_pairs,hash\n"
            "closed,a_closed,PROVEN_EMPTY,reason,0,4,h\n",
            encoding="utf-8",
        )
        (out / "qa_quantum_arithmetic_scale_stage1_leaderboard.csv").write_text(
            "target,window,rows_evaluated,positive_rows,base_rate,precision,recall,f1,lift,null_lift_max,verdict\n"
            "a_open,w,10,2,0.2,0.5,0.5,0.5,2.5,1.1,PERSISTENT_SIGNAL\n"
            "a_closed,w,10,0,0,,,,,,LOW_TRAIN_SUPPORT\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            out_dir=tmp,
            summary_json="stage18_selftest.json",
            ledger_csv="stage18_selftest_ledger.csv",
            target_summary_csv="stage18_selftest_target_summary.csv",
        )
        payload = run(args)
        ok = (
            payload["row_count"] == 2
            and payload["target_count"] == 2
            and payload["algebraic_status_counts"]["PROVEN_EMPTY"] == 1
            and Path(tmp, "stage18_selftest_ledger.csv").exists()
        )
        return {"ok": ok, "rows": payload["row_count"], "targets": payload["target_count"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/qa_quantum_arithmetic_mining_001")
    parser.add_argument("--summary-json", default="qa_quantum_arithmetic_stage18_consolidated_ledger.json")
    parser.add_argument("--ledger-csv", default="qa_quantum_arithmetic_stage18_consolidated_ledger.csv")
    parser.add_argument("--target-summary-csv", default="qa_quantum_arithmetic_stage18_target_summary.csv")
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
                "row_count": payload["row_count"],
                "target_count": payload["target_count"],
                "artifacts": payload["artifacts"],
                "canonical_hash": payload["canonical_hash"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
