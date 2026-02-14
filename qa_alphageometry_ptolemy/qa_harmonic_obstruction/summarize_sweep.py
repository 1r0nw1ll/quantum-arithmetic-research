#!/usr/bin/env python3
"""Summarize a harmonic obstruction sweep into markdown, CSV, and JSON."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


EPISODE_RE = re.compile(
    r"^EP_HO_a(?P<alpha_index>\d+)_(?P<alpha_slug>[0-9]+-[0-9]+)_w(?P<window>\d+)_(?P<gen>g[A-Za-z0-9]+)\.json$"
)
RUN_RE = re.compile(
    r"^RUN-HO-a(?P<alpha_index>\d+)-w(?P<window>\d+)-(?P<gen>g[A-Za-z0-9]+)(?:-k(?P<k>\d+))?$"
)
KNOWN_RECEIPT_STATUSES = frozenset({"RETURN_FOUND", "NO_RETURN_WITHIN_K"})
PHASE_LAW_FAIL_TYPES = (
    "NON_MONOTONE_RETURN",
    "MISSING_K_LEVEL",
    "DUPLICATE_K_LEVEL",
    "UNKNOWN_RECEIPT_STATUS",
)


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_ref(repo_root: Path, ref_path: str) -> Path:
    p = Path(ref_path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _parse_case_identity(run_obj: Dict[str, Any]) -> Dict[str, Any]:
    run_id = str(run_obj.get("run_id", ""))
    run_match = RUN_RE.match(run_id)

    episode_ref = (
        run_obj.get("inputs", {})
        .get("episode_ref", {})
        .get("path_or_hash", "")
    )
    ep_name = Path(str(episode_ref)).name
    ep_match = EPISODE_RE.match(ep_name)

    planned_k = int(run_obj.get("inputs", {}).get("k", 0) or 0)

    if ep_match is not None:
        alpha_index = int(ep_match.group("alpha_index"))
        alpha = ep_match.group("alpha_slug").replace("-", "/")
        window = int(ep_match.group("window"))
        gen = ep_match.group("gen")
    elif run_match is not None:
        alpha_index = int(run_match.group("alpha_index"))
        alpha = f"a{alpha_index:02d}"
        window = int(run_match.group("window"))
        gen = run_match.group("gen")
        if planned_k <= 0 and run_match.group("k"):
            planned_k = int(run_match.group("k"))
    else:
        alpha_index = -1
        alpha = "unknown"
        window = -1
        gen = "unknown"

    return {
        "run_id": run_id,
        "episode_ref": str(episode_ref),
        "episode_name": ep_name,
        "alpha_index": alpha_index,
        "alpha": alpha,
        "window": window,
        "generator_set": gen,
        "planned_k": planned_k,
    }


def _find_artifact_path(run_obj: Dict[str, Any], schema_id: str) -> str:
    artifacts = run_obj.get("outputs", {}).get("artifacts", [])
    for item in artifacts:
        if item.get("schema_id") == schema_id:
            return str(item.get("path_or_hash", ""))
    return ""


def _cluster_key(fail_type: str, invariant_diff: Dict[str, Any]) -> str:
    return f"{fail_type}|{_canonical(invariant_diff)}"


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _compute_critical_k_phase_law(
    by_case: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    k_domain: List[int],
    max_violation_samples: int = 25,
) -> Dict[str, Any]:
    kd = sorted({int(k) for k in k_domain})
    violations_by_fail_type: Counter = Counter()
    violations_by_gen: Dict[str, Counter] = defaultdict(Counter)
    violation_samples: List[Dict[str, Any]] = []

    cases_total = 0
    cases_with_return = 0
    cases_with_return_at_smallest_k = 0
    cases_phase_law_ok = 0
    cases_phase_law_violations = 0

    def _add_violation(
        alpha: str,
        alpha_index: int,
        window: int,
        gen: str,
        fail_type: str,
        invariant_diff: Dict[str, Any],
    ) -> None:
        violations_by_fail_type[fail_type] += 1
        violations_by_gen[gen][fail_type] += 1
        if len(violation_samples) < max_violation_samples:
            violation_samples.append(
                {
                    "case_key": {
                        "alpha": alpha,
                        "alpha_index": alpha_index,
                        "window": window,
                        "generator_set": gen,
                    },
                    "fail_type": fail_type,
                    "invariant_diff": invariant_diff,
                }
            )

    for (alpha, window, gen), case_rows in sorted(by_case.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        cases_total += 1
        alpha_index = int(case_rows[0].get("alpha_index", -1)) if case_rows else -1

        status_by_k: Dict[int, str] = {}
        dup_ks: List[int] = []
        unknown_statuses: List[Tuple[int, str]] = []
        for r in case_rows:
            pk = int(r.get("planned_k", 0) or 0)
            rs = str(r.get("receipt_status", "MISSING"))
            if pk <= 0:
                continue
            if pk in status_by_k:
                dup_ks.append(pk)
                continue
            status_by_k[pk] = rs
            if rs not in KNOWN_RECEIPT_STATUSES:
                unknown_statuses.append((pk, rs))

        missing = [k for k in kd if k not in status_by_k]
        case_has_violation = False

        if dup_ks:
            case_has_violation = True
            _add_violation(
                alpha,
                alpha_index,
                window,
                gen,
                "DUPLICATE_K_LEVEL",
                {
                    "k_domain": kd,
                    "duplicate_k": sorted({int(x) for x in dup_ks}),
                },
            )
        if unknown_statuses:
            case_has_violation = True
            _add_violation(
                alpha,
                alpha_index,
                window,
                gen,
                "UNKNOWN_RECEIPT_STATUS",
                {
                    "k_domain": kd,
                    "unknown_statuses": [{"k": int(k), "status": s} for k, s in unknown_statuses],
                },
            )
        if missing:
            case_has_violation = True
            _add_violation(
                alpha,
                alpha_index,
                window,
                gen,
                "MISSING_K_LEVEL",
                {
                    "k_domain": kd,
                    "missing_k": [int(k) for k in missing],
                    "present_k": [int(k) for k in sorted(status_by_k.keys())],
                },
            )

        if case_has_violation:
            cases_phase_law_violations += 1
            continue

        seq = [(k, status_by_k[k]) for k in kd]
        if seq and seq[0][1] == "RETURN_FOUND":
            cases_with_return_at_smallest_k += 1
        if any(s == "RETURN_FOUND" for _, s in seq):
            cases_with_return += 1

        first_return_seen = False
        monotone_violation = None
        for i, (k, s) in enumerate(seq):
            if s == "RETURN_FOUND":
                first_return_seen = True
                continue
            if first_return_seen:
                k_prev, s_prev = seq[i - 1]
                monotone_violation = {
                    "k_domain": kd,
                    "status_by_k": {str(k0): str(s0) for (k0, s0) in seq},
                    "first_violation": {
                        "k_prev": int(k_prev),
                        "status_prev": str(s_prev),
                        "k_next": int(k),
                        "status_next": str(s),
                    },
                }
                break

        if monotone_violation is None:
            cases_phase_law_ok += 1
        else:
            cases_phase_law_violations += 1
            _add_violation(
                alpha,
                alpha_index,
                window,
                gen,
                "NON_MONOTONE_RETURN",
                monotone_violation,
            )

    violations_by_generator_set = []
    for gen in sorted({g for (_, _, g) in by_case.keys()}):
        counts = {ft: int(violations_by_gen[gen].get(ft, 0)) for ft in PHASE_LAW_FAIL_TYPES}
        violations_by_generator_set.append(
            {
                "generator_set": gen,
                "counts": counts,
                "total": int(sum(counts.values())),
            }
        )

    return {
        "schema_version": "PHASE_LAW.v1",
        "rule_id": "MONOTONE_RETURN_IN_K",
        "k_domain": kd,
        "strict_order": True,
        "summary": {
            "cases_total": int(cases_total),
            "cases_with_return": int(cases_with_return),
            "cases_with_return_at_smallest_k": int(cases_with_return_at_smallest_k),
            "cases_phase_law_ok": int(cases_phase_law_ok),
            "cases_phase_law_violations": int(cases_phase_law_violations),
        },
        "violations_by_fail_type": {ft: int(violations_by_fail_type.get(ft, 0)) for ft in PHASE_LAW_FAIL_TYPES},
        "violations_by_generator_set": violations_by_generator_set,
        "violation_samples": violation_samples,
    }


def summarize(out_dir: Path) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    run_paths = sorted(Path(p) for p in glob.glob(str(out_dir / "run_*.json")))
    bundle_paths = sorted(Path(p) for p in glob.glob(str(out_dir / "bundle_*.json")))

    if not run_paths:
        raise RuntimeError(f"No run_*.json files found in: {out_dir}")
    if not bundle_paths:
        raise RuntimeError(f"No bundle_*.json files found in: {out_dir}")

    # Use deterministic choice if multiple bundles exist.
    bundle_path = bundle_paths[-1]
    bundle = _read_json(bundle_path)

    rows: List[Dict[str, Any]] = []
    run_status_counts: Counter = Counter()
    receipt_status_counts: Counter = Counter()
    by_window_gen: Dict[Tuple[int, str], Counter] = defaultdict(Counter)
    by_window_gen_k: Dict[Tuple[int, str, int], Counter] = defaultdict(Counter)
    receipt_by_k: Dict[int, Counter] = defaultdict(Counter)
    obstruction_clusters: Dict[str, Dict[str, Any]] = {}
    receipt_gallery: List[Dict[str, Any]] = []

    for run_path in run_paths:
        run_obj = _read_json(run_path)
        case = _parse_case_identity(run_obj)
        run_status = str(run_obj.get("result", {}).get("status", "UNKNOWN"))
        run_fail_type = str(run_obj.get("result", {}).get("fail_type", ""))
        steps_total = len(run_obj.get("execution", {}).get("steps", []))

        frontier_ref = _find_artifact_path(run_obj, "QA_FRONTIER_SNAPSHOT_SCHEMA.v1")
        receipt_ref = _find_artifact_path(run_obj, "QA_BOUNDED_RETURN_RECEIPT_SCHEMA.v1")

        frontier_size = 0
        visited_size = 0
        if frontier_ref:
            frontier_path = _resolve_ref(repo_root, frontier_ref)
            if frontier_path.exists():
                frontier_obj = _read_json(frontier_path)
                frontier_size = len(frontier_obj.get("frontier", []))
                visited_size = len(frontier_obj.get("visited", []))

        receipt_status = "MISSING"
        receipt_fail_type = ""
        receipt_k = 0
        receipt_id = ""
        receipt_invariant: Dict[str, Any] = {}
        visited_nodes = 0
        receipt_path = None
        if receipt_ref:
            receipt_path = _resolve_ref(repo_root, receipt_ref)
            if receipt_path.exists():
                receipt_obj = _read_json(receipt_path)
                result_obj = receipt_obj.get("result", {})
                receipt_status = str(result_obj.get("status", "UNKNOWN"))
                receipt_fail_type = str(result_obj.get("fail_type", ""))
                receipt_k = int(receipt_obj.get("k", 0))
                receipt_id = str(receipt_obj.get("receipt_id", ""))
                receipt_invariant = (
                    result_obj.get("invariant_diff", {})
                    if isinstance(result_obj.get("invariant_diff", {}), dict)
                    else {}
                )
                visited_nodes = int(receipt_invariant.get("visited_nodes", 0))

                # Treat non-return statuses as obstructions for clustering/reporting.
                if receipt_status != "RETURN_FOUND":
                    fail_type = receipt_fail_type or receipt_status
                    key = _cluster_key(fail_type, receipt_invariant)
                    entry = obstruction_clusters.get(key)
                    if entry is None:
                        entry = {
                            "fail_type": fail_type,
                            "invariant_diff": receipt_invariant,
                            "count": 0,
                            "sample_run_ids": [],
                            "sample_receipt_ids": [],
                        }
                        obstruction_clusters[key] = entry
                    entry["count"] += 1
                    if len(entry["sample_run_ids"]) < 5:
                        entry["sample_run_ids"].append(case["run_id"])
                    if receipt_id and len(entry["sample_receipt_ids"]) < 5:
                        entry["sample_receipt_ids"].append(receipt_id)

        run_status_counts[run_status] += 1
        receipt_status_counts[receipt_status] += 1
        by_window_gen[(case["window"], case["generator_set"])][run_status] += 1
        by_window_gen_k[(case["window"], case["generator_set"], case["planned_k"])][run_status] += 1
        receipt_by_k[case["planned_k"]][receipt_status] += 1

        row = {
            **case,
            "run_status": run_status,
            "run_fail_type": run_fail_type,
            "steps_total": steps_total,
            "frontier_size": frontier_size,
            "visited_size": visited_size,
            "receipt_status": receipt_status,
            "receipt_fail_type": receipt_fail_type,
            "receipt_k": receipt_k,
            "visited_nodes": visited_nodes,
            "receipt_id": receipt_id,
            "receipt_ref": receipt_ref,
            "run_path": str(run_path),
        }
        rows.append(row)

        if receipt_status != "MISSING" and len(receipt_gallery) < 10:
            receipt_gallery.append(
                {
                    "run_id": case["run_id"],
                    "alpha": case["alpha"],
                    "window": case["window"],
                    "generator_set": case["generator_set"],
                    "receipt_status": receipt_status,
                    "receipt_fail_type": receipt_fail_type or receipt_status,
                    "receipt_id": receipt_id,
                    "receipt_ref": receipt_ref,
                }
            )

    rows.sort(key=lambda x: x["run_id"])
    receipt_gallery.sort(key=lambda x: x["run_id"])

    sorted_clusters = sorted(
        obstruction_clusters.values(),
        key=lambda x: (-int(x["count"]), str(x["fail_type"]), _canonical(x["invariant_diff"])),
    )
    for i, cluster in enumerate(sorted_clusters, start=1):
        cluster["rank"] = i

    windows = sorted({int(r["window"]) for r in rows})
    gensets = sorted({str(r["generator_set"]) for r in rows})
    k_values = sorted({int(r["planned_k"]) for r in rows})
    alphas = sorted({str(r["alpha"]) for r in rows}, key=lambda s: (len(s), s))

    window_gen_rows = []
    for (window, gen), counts in sorted(by_window_gen.items(), key=lambda x: (x[0][0], x[0][1])):
        window_gen_rows.append(
            {
                "window": int(window),
                "generator_set": str(gen),
                "success": int(counts.get("SUCCESS", 0)),
                "fail": int(counts.get("FAIL", 0)),
                "total": int(sum(counts.values())),
            }
        )

    window_gen_k_rows = []
    for (window, gen, k), counts in sorted(by_window_gen_k.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        window_gen_k_rows.append(
            {
                "window": int(window),
                "generator_set": str(gen),
                "k": int(k),
                "success": int(counts.get("SUCCESS", 0)),
                "fail": int(counts.get("FAIL", 0)),
                "total": int(sum(counts.values())),
            }
        )

    receipt_by_k_rows = []
    for k in sorted(receipt_by_k.keys()):
        c = receipt_by_k[k]
        receipt_by_k_rows.append(
            {
                "k": int(k),
                "status_counts": dict(sorted(c.items())),
                "total": int(sum(c.values())),
            }
        )

    # Critical depth invariant:
    # k* = smallest planned_k for which receipt_status == RETURN_FOUND
    # for each (alpha, window, generator_set) case family.
    by_case: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(str(row["alpha"]), int(row["window"]), str(row["generator_set"]))].append(row)

    critical_k_rows: List[Dict[str, Any]] = []
    critical_k_hist_by_gen: Dict[str, Counter] = defaultdict(Counter)
    for (alpha, window, gen), case_rows in sorted(by_case.items(), key=lambda x: (x[0][2], x[0][1], x[0][0])):
        sorted_case_rows = sorted(case_rows, key=lambda r: (int(r["planned_k"]), str(r["run_id"])))
        alpha_index = int(sorted_case_rows[0]["alpha_index"]) if sorted_case_rows else -1
        return_ks = sorted(
            {
                int(r["planned_k"])
                for r in sorted_case_rows
                if str(r.get("receipt_status", "")) == "RETURN_FOUND"
            }
        )
        critical_k = int(return_ks[0]) if return_ks else None
        status_by_k: Dict[str, str] = {}
        for r in sorted_case_rows:
            status_by_k[str(int(r["planned_k"]))] = str(r.get("receipt_status", "MISSING"))

        critical_k_rows.append(
            {
                "alpha": alpha,
                "alpha_index": alpha_index,
                "window": window,
                "generator_set": gen,
                "critical_k": critical_k,
                "return_observed": bool(return_ks),
                "status_by_k": status_by_k,
            }
        )
        hist_key = "NONE" if critical_k is None else str(critical_k)
        critical_k_hist_by_gen[gen][hist_key] += 1

    critical_k_hist_rows = []
    for gen in sorted(critical_k_hist_by_gen.keys()):
        counts = critical_k_hist_by_gen[gen]
        critical_k_hist_rows.append(
            {
                "generator_set": gen,
                "counts": {k: int(v) for k, v in sorted(counts.items(), key=lambda kv: (kv[0] != "NONE", kv[0]))},
                "total_cases": int(sum(counts.values())),
            }
        )

    phase_law = _compute_critical_k_phase_law(by_case=by_case, k_domain=k_values)

    summary = {
        "schema_id": "QA_HARMONIC_SWEEP_REPORT.v1",
        "generated_utc": _now_utc(),
        "out_dir": str(out_dir.resolve()),
        "bundle": {
            "path": str(bundle_path),
            "bundle_id": str(bundle.get("bundle_id", "")),
            "created_utc": str(bundle.get("created_utc", "")),
            "this_bundle_hash": str(bundle.get("hash_chain", {}).get("this_bundle_hash", "")),
            "runs_total": int(bundle.get("summary", {}).get("runs_total", 0)),
            "runs_success": int(bundle.get("summary", {}).get("runs_success", 0)),
        },
        "coverage": {
            "runs": len(rows),
            "alpha_count": len(alphas),
            "window_count": len(windows),
            "generator_set_count": len(gensets),
            "k_count": len(k_values),
            "alphas": alphas,
            "windows": windows,
            "generator_sets": gensets,
            "k_values": k_values,
        },
        "run_status_counts": dict(sorted(run_status_counts.items())),
        "receipt_status_counts": dict(sorted(receipt_status_counts.items())),
        "receipt_status_by_k": receipt_by_k_rows,
        "critical_k_by_case": critical_k_rows,
        "critical_k_histogram_by_generator_set": critical_k_hist_rows,
        "critical_k_phase_law": phase_law,
        "window_generator_status": window_gen_rows,
        "window_generator_k_status": window_gen_k_rows,
        "obstruction_clusters": sorted_clusters,
        "receipt_gallery": receipt_gallery,
        "cases": rows,
    }
    summary["report_hash"] = _sha256(_canonical(summary))
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "alpha",
        "alpha_index",
        "window",
        "generator_set",
        "planned_k",
        "run_status",
        "run_fail_type",
        "steps_total",
        "frontier_size",
        "visited_size",
        "receipt_status",
        "receipt_fail_type",
        "receipt_k",
        "visited_nodes",
        "receipt_id",
        "receipt_ref",
        "episode_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    bundle = summary["bundle"]
    coverage = summary["coverage"]

    lines: List[str] = []
    lines.append(f"# Harmonic Sweep Report: `{bundle['bundle_id'][:16]}`")
    lines.append("")
    lines.append(f"- Bundle: `{bundle['path']}`")
    lines.append(f"- Bundle hash: `{bundle['this_bundle_hash']}`")
    lines.append(f"- Generated UTC: `{summary['generated_utc']}`")
    lines.append(f"- Out dir: `{summary['out_dir']}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Runs: `{coverage['runs']}`")
    lines.append(f"- Alpha values: `{coverage['alpha_count']}`")
    lines.append(f"- Windows: `{coverage['window_count']}`")
    lines.append(f"- Generator sets: `{coverage['generator_set_count']}`")
    lines.append(f"- k values: `{coverage['k_values']}`")
    lines.append("")
    lines.append("## Status Summary")
    lines.append(f"- Run status counts: `{summary['run_status_counts']}`")
    lines.append(f"- Receipt status counts: `{summary['receipt_status_counts']}`")
    lines.append("")
    lines.append("## Window x Generator Status")
    w_rows = [
        [
            str(item["window"]),
            str(item["generator_set"]),
            str(item["success"]),
            str(item["fail"]),
            str(item["total"]),
        ]
        for item in summary["window_generator_status"]
    ]
    lines.append(_markdown_table(["window", "generator_set", "success", "fail", "total"], w_rows))
    lines.append("")
    lines.append("## k x Window x Generator Status")
    wk_rows = [
        [
            str(item["k"]),
            str(item["window"]),
            str(item["generator_set"]),
            str(item["success"]),
            str(item["fail"]),
            str(item["total"]),
        ]
        for item in summary["window_generator_k_status"]
    ]
    lines.append(_markdown_table(["k", "window", "generator_set", "success", "fail", "total"], wk_rows))
    lines.append("")
    lines.append("## Critical k (k*)")
    lines.append("k* is the smallest planned k with `RETURN_FOUND` for each (alpha, window, generator_set).")
    lines.append("")
    hk_rows = []
    for item in summary["critical_k_histogram_by_generator_set"]:
        hk_rows.append(
            [
                str(item["generator_set"]),
                str(item["counts"]),
                str(item["total_cases"]),
            ]
        )
    lines.append(_markdown_table(["generator_set", "k*_histogram", "total_cases"], hk_rows))
    lines.append("")
    pl = summary["critical_k_phase_law"]
    lines.append("Phase law summary:")
    lines.append(f"- Rule: `{pl['rule_id']}`")
    lines.append(f"- k domain: `{pl['k_domain']}`")
    lines.append(f"- Summary: `{pl['summary']}`")
    lines.append(f"- Violations by fail type: `{pl['violations_by_fail_type']}`")
    lines.append("")
    ck_rows = []
    for item in summary["critical_k_by_case"][:40]:
        ck_rows.append(
            [
                str(item["generator_set"]),
                str(item["window"]),
                str(item["alpha"]),
                str(item["critical_k"]) if item["critical_k"] is not None else "NONE",
                str(item["status_by_k"]),
            ]
        )
    lines.append("First 40 cases:")
    lines.append(_markdown_table(["generator_set", "window", "alpha", "critical_k", "status_by_k"], ck_rows))
    lines.append("")
    lines.append("## Top Obstruction Clusters")

    top_clusters = summary["obstruction_clusters"][:10]
    if not top_clusters:
        lines.append("No obstruction clusters detected.")
    else:
        c_rows = []
        for c in top_clusters:
            c_rows.append(
                [
                    str(c["rank"]),
                    str(c["count"]),
                    str(c["fail_type"]),
                    f"`{_canonical(c['invariant_diff'])}`",
                    ", ".join(c["sample_run_ids"]),
                ]
            )
        lines.append(_markdown_table(["rank", "count", "fail_type", "invariant_diff", "sample_run_ids"], c_rows))
    lines.append("")
    lines.append("## Receipt Gallery")
    g_rows = []
    for g in summary["receipt_gallery"]:
        g_rows.append(
            [
                str(g["run_id"]),
                str(g["alpha"]),
                str(g["window"]),
                str(g["generator_set"]),
                str(g["receipt_status"]),
                str(g["receipt_fail_type"]),
                str(g["receipt_id"]),
            ]
        )
    lines.append(
        _markdown_table(
            ["run_id", "alpha", "window", "generator_set", "receipt_status", "receipt_fail_type", "receipt_id"],
            g_rows,
        )
    )
    lines.append("")
    lines.append(f"- Report hash: `{summary['report_hash']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize harmonic sweep outputs.")
    ap.add_argument("--out_dir", required=True, help="Directory containing run_*.json and bundle_*.json")
    ap.add_argument(
        "--out_prefix",
        default="harmonic_report",
        help="Output prefix for report files (default: harmonic_report)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    summary = summarize(out_dir)

    bundle_short = str(summary["bundle"]["bundle_id"])[:8]
    prefix = f"{args.out_prefix}_{bundle_short}"

    md_path = out_dir / f"{prefix}.md"
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"

    _write_markdown(md_path, summary)
    _write_csv(csv_path, summary["cases"])
    json_path.write_text(_canonical(summary) + "\n", encoding="utf-8")

    print(f"WROTE {md_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {json_path}")
    print(f"REPORT_META out_json={str(json_path)} report_hash={summary['report_hash']} critical_k_phase_law={1 if 'critical_k_phase_law' in summary else 0}", flush=True)
    print(f"REPORT_HASH {summary['report_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
