#!/usr/bin/env python3
"""Build a bounded prioritized replay sample from the SINQA ledger."""

from __future__ import annotations

QA_COMPLIANCE = (
    "self_improving_neural_qa_prioritized_replay — read-only prioritized replay "
    "sampler over accepted/rejected SINQA ledger evidence"
)

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = Path("results/self_improving_neural_qa/ledger.jsonl")
DEFAULT_OUT_DIR = Path("results/self_improving_neural_qa/prioritized_replay")
SCHEMA_VERSION = "QA_SELF_IMPROVING_NEURAL_QA_PRIORITIZED_REPLAY.v0"
FILE_HASH_PREFIX = b"qa_self_improving_neural_qa_file_v0\x00"
SAMPLE_HASH_DOMAIN = "qa_self_improving_neural_qa_prioritized_replay_v0"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_sha256(domain: str, obj: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + canonical_json(obj).encode("utf-8")).hexdigest()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(FILE_HASH_PREFIX)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{path}: file not found"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: empty line")
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            errors.append(f"line {line_no}: row must be object")
    return rows, errors


def packet(row: dict[str, Any]) -> dict[str, Any] | None:
    obj = row.get("packet")
    return obj if isinstance(obj, dict) else None


def obj_field(parent: dict[str, Any], key: str) -> dict[str, Any]:
    obj = parent.get(key)
    return obj if isinstance(obj, dict) else {}


def int_field(parent: dict[str, Any], key: str) -> int:
    val = parent.get(key)
    return val if isinstance(val, int) and not isinstance(val, bool) else 0


def classify_domain(pkt: dict[str, Any]) -> str:
    cand = obj_field(pkt, "candidate")
    ev = obj_field(pkt, "evidence")
    text = " ".join(
        str(part).lower()
        for part in [cand.get("artifact_ref"), cand.get("description"), ev.get("source_replay_ref")]
        if part is not None
    )
    if "general_ml" in text or "self_improving_neural_qa/general_ml" in text:
        return "general_ml"
    if "hsi" in text or "salinas" in text or "paviau" in text or "pavia" in text:
        return "hsi"
    if cand.get("kind") in {"capacity_patch", "configuration_patch"}:
        return "config"
    return "other"


def source_status(pkt: dict[str, Any]) -> dict[str, Any]:
    ev = obj_field(pkt, "evidence")
    ref = ev.get("source_replay_ref")
    expected_hash = ev.get("source_replay_hash")
    if not isinstance(ref, str) or not ref:
        return {"source_ref": None, "source_hash": None, "source_exists": False, "source_hash_ok": False}
    path = resolve_path(ref)
    exists = path.exists()
    actual = file_hash(path) if exists else None
    return {
        "source_ref": ref,
        "source_hash": expected_hash if isinstance(expected_hash, str) else None,
        "source_exists": exists,
        "source_hash_ok": exists and isinstance(expected_hash, str) and actual == expected_hash,
        "computed_source_hash": actual,
    }


def priority_item(row: dict[str, Any], line_no: int, total_rows: int) -> dict[str, Any] | None:
    pkt = packet(row)
    if pkt is None:
        return None
    gate = obj_field(pkt, "replay_gate")
    promotion = obj_field(pkt, "promotion")
    cand = obj_field(pkt, "candidate")
    decision = promotion.get("decision")
    if decision not in {"accepted", "rejected"}:
        return None
    fixed = int_field(gate, "new_failures_fixed")
    harmed = int_field(gate, "protected_cases_harmed")
    protected = int_field(gate, "protected_cases_replayed")
    recency = line_no / max(total_rows, 1)
    rejected = 1 if decision == "rejected" else 0
    protected_harm = 1 if harmed > 0 else 0
    no_fix = 1 if fixed <= 0 else 0
    source = source_status(pkt)
    missing_penalty = 0 if source["source_hash_ok"] else 1
    score = (
        protected_harm * 1000
        + rejected * 250
        + no_fix * 125
        + min(harmed, 20) * 25
        + min(fixed, 100) * 3
        + min(protected, 20_000) / 1000.0
        + recency
        - missing_penalty * 50
    )
    return {
        "line": line_no,
        "packet_hash": row.get("packet_hash"),
        "update_id": pkt.get("update_id"),
        "decision": decision,
        "candidate_kind": cand.get("kind"),
        "domain": classify_domain(pkt),
        "priority_score": round(score, 6),
        "priority_reasons": {
            "rejected": bool(rejected),
            "protected_harm": bool(protected_harm),
            "no_positive_fix": bool(no_fix),
            "recency": round(recency, 6),
        },
        "replay_gate": {
            "fixed": fixed,
            "protected": protected,
            "harmed": harmed,
            "deterministic": gate.get("deterministic_replay") is True,
        },
        **source,
    }


def count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_sample(ledger: Path, max_items: int, min_harm_items: int, max_per_domain: int) -> dict[str, Any]:
    rows, errors = read_jsonl(ledger)
    candidates = [
        item
        for line_no, row in enumerate(rows, start=1)
        if (item := priority_item(row, line_no, len(rows))) is not None
    ]
    candidates.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            str(item.get("update_id") or ""),
            int(item["line"]),
        )
    )
    eligible = [item for item in candidates if item["source_hash_ok"] is True]
    unavailable = len(candidates) - len(eligible)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_domains: dict[str, int] = {}

    def can_add(item: dict[str, Any]) -> bool:
        if len(selected) >= max_items:
            return False
        if str(item.get("update_id")) in selected_ids:
            return False
        domain = str(item["domain"])
        return max_per_domain <= 0 or selected_domains.get(domain, 0) < max_per_domain

    def add_item(item: dict[str, Any]) -> None:
        selected.append(item)
        selected_ids.add(str(item.get("update_id")))
        domain = str(item["domain"])
        selected_domains[domain] = selected_domains.get(domain, 0) + 1

    harm_items = [item for item in eligible if item["priority_reasons"]["protected_harm"]]
    for item in harm_items[: max(0, min_harm_items)]:
        if can_add(item):
            add_item(item)
    for item in eligible:
        if can_add(item):
            add_item(item)
        if len(selected) >= max_items:
            break
    selected.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            str(item.get("update_id") or ""),
            int(item["line"]),
        )
    )
    missing = sum(1 for item in selected if not item["source_exists"])
    hash_bad = sum(1 for item in selected if item["source_exists"] and not item["source_hash_ok"])
    core = {
        "schema_version": SCHEMA_VERSION,
        "ledger": repo_relative(ledger),
        "ledger_rows": len(rows),
        "max_items": max_items,
        "min_harm_items": min_harm_items,
        "max_per_domain": max_per_domain,
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "unavailable_count": unavailable,
        "selected_count": len(selected),
        "selected_domains": count_values([str(item["domain"]) for item in selected]),
        "selected_decisions": count_values([str(item["decision"]) for item in selected]),
        "source_missing": missing,
        "source_hash_mismatch": hash_bad,
        "items": selected,
    }
    result = dict(core)
    result["sample_hash"] = domain_sha256(SAMPLE_HASH_DOMAIN, core)
    result["ok"] = not errors and missing == 0 and hash_bad == 0 and len(selected) <= max_items
    result["errors"] = errors
    return result


def write_sample(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qa_sinqa_prioritized_replay_{result['sample_hash'][:16]}.json"
    path.write_text(canonical_json(result), encoding="utf-8")
    return path


def print_text(result: dict[str, Any]) -> None:
    print(f"SINQA prioritized replay: {'OK' if result['ok'] else 'ATTENTION'}")
    print(
        f"selected={result['selected_count']} candidates={result['candidate_count']} "
        f"eligible={result['eligible_count']} unavailable={result['unavailable_count']} "
        f"domains={result['selected_domains']} decisions={result['selected_decisions']} "
        f"missing={result['source_missing']} hash_mismatch={result['source_hash_mismatch']}"
    )


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.json"
        source.write_text(canonical_json({"fixed_ensemble_errors": 1}), encoding="utf-8")
        source_hash = file_hash(source)
        packets = []
        for idx, (decision, fixed, harmed) in enumerate(
            [("accepted", 2, 0), ("rejected", 0, 0), ("rejected", 4, 2)],
            start=1,
        ):
            pkt = {
                "update_id": f"sinqa.v0.prioritized.selftest.{idx}",
                "candidate": {"kind": "correction_rule", "description": "general_ml selftest"},
                "evidence": {"source_replay_ref": str(source), "source_replay_hash": source_hash},
                "replay_gate": {
                    "new_failures_fixed": fixed,
                    "protected_cases_replayed": 5,
                    "protected_cases_harmed": harmed,
                    "deterministic_replay": True,
                },
                "promotion": {"decision": decision},
            }
            packets.append({"packet_hash": hashlib.sha256(str(idx).encode()).hexdigest(), "packet": pkt})
        ledger = root / "ledger.jsonl"
        ledger.write_text("\n".join(canonical_json(row) for row in packets) + "\n", encoding="utf-8")
        result = build_sample(ledger, max_items=2, min_harm_items=1, max_per_domain=0)
        out = write_sample(result, root / "out")
        loaded = json.loads(out.read_text(encoding="utf-8"))
        ok = (
            result["ok"] is True
            and result["selected_count"] == 2
            and result["items"][0]["priority_reasons"]["protected_harm"] is True
            and loaded["sample_hash"] == result["sample_hash"]
        )
        return {"ok": ok, "result": result, "out": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-items", type=int, default=32)
    parser.add_argument("--min-harm-items", type=int, default=8)
    parser.add_argument("--max-per-domain", type=int, default=16)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        result = self_test()
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1
    if args.max_items < 1:
        print(json.dumps({"ok": False, "errors": ["--max-items must be >= 1"]}, sort_keys=True))
        return 2
    if args.min_harm_items < 0:
        print(json.dumps({"ok": False, "errors": ["--min-harm-items must be >= 0"]}, sort_keys=True))
        return 2
    if args.max_per_domain < 0:
        print(json.dumps({"ok": False, "errors": ["--max-per-domain must be >= 0"]}, sort_keys=True))
        return 2
    result = build_sample(resolve_path(args.ledger), args.max_items, args.min_harm_items, args.max_per_domain)
    if args.write:
        out = write_sample(result, resolve_path(args.out_dir))
        result = {"out": str(out), **result}
    if args.text:
        print_text(result)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
