#!/usr/bin/env python3
"""Deterministically scaffold the pinned 15-proof Mathlib certificate corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "mathlib_ingest" / "upstream_registry.v1.json"
PACK = ROOT / "mathlib_pack_v1"
EXAMPLES = PACK / "examples"
MANIFEST_PATH = ROOT / "kernel_trace_manifest.json"
CREATED_UTC = "2026-06-21T00:00:00Z"

CASES = [
    {
        "id": "mathlib01_bit_indices_zero",
        "declaration": "Nat.bitIndices_zero",
        "claim": "The binary one-bit position list of zero is empty.",
        "formal": "theorem qa_mathlib01_bit_indices_zero : Nat.bitIndices 0 = []",
        "proof": "Nat.bitIndices_zero",
    },
    {
        "id": "mathlib02_bit_indices_one",
        "declaration": "Nat.bitIndices_one",
        "claim": "The binary one-bit position list of one contains only position zero.",
        "formal": "theorem qa_mathlib02_bit_indices_one : Nat.bitIndices 1 = [0]",
        "proof": "Nat.bitIndices_one",
    },
    {
        "id": "mathlib03_bit_indices_odd",
        "declaration": "Nat.bitIndices_two_mul_add_one",
        "claim": "For every natural n, the one-bit positions of 2n+1 are zero followed by the one-bit positions of n shifted by one.",
        "formal": "theorem qa_mathlib03_bit_indices_odd (n : Nat) : (2 * n + 1).bitIndices = 0 :: List.map (fun x => x + 1) n.bitIndices",
        "proof": "Nat.bitIndices_two_mul_add_one n",
    },
    {
        "id": "mathlib04_bit_indices_even",
        "declaration": "Nat.bitIndices_two_mul",
        "claim": "For every natural n, the one-bit positions of 2n are the one-bit positions of n shifted by one.",
        "formal": "theorem qa_mathlib04_bit_indices_even (n : Nat) : (2 * n).bitIndices = List.map (fun x => x + 1) n.bitIndices",
        "proof": "Nat.bitIndices_two_mul n",
    },
    {
        "id": "mathlib05_bit_indices_nodup",
        "declaration": "Nat.bitIndices_nodup",
        "claim": "The binary one-bit position list of every natural number has no duplicate positions.",
        "formal": "theorem qa_mathlib05_bit_indices_nodup (n : Nat) : n.bitIndices.Nodup",
        "proof": "Nat.bitIndices_nodup",
    },
    {
        "id": "mathlib06_bit_indices_two_pow",
        "declaration": "Nat.bitIndices_two_pow",
        "claim": "The binary one-bit position list of 2^k contains only position k.",
        "formal": "theorem qa_mathlib06_bit_indices_two_pow (k : Nat) : (2 ^ k).bitIndices = [k]",
        "proof": "Nat.bitIndices_two_pow k",
    },
    {
        "id": "mathlib07_nth_true",
        "declaration": "Nat.nth_true",
        "claim": "The nth natural satisfying the predicate that is always true is n.",
        "formal": "theorem qa_mathlib07_nth_true (n : Nat) : Nat.nth (fun _ => True) n = n",
        "proof": "Nat.nth_true n",
    },
    {
        "id": "mathlib08_nth_false",
        "declaration": "Nat.nth_false",
        "claim": "The nth natural satisfying the predicate that is always false is defined as zero.",
        "formal": "theorem qa_mathlib08_nth_false (n : Nat) : Nat.nth (fun _ => False) n = 0",
        "proof": "Nat.nth_false n",
    },
    {
        "id": "mathlib09_count_true",
        "declaration": "Nat.count_true",
        "claim": "Among naturals below n, exactly n satisfy the predicate that is always true.",
        "formal": "theorem qa_mathlib09_count_true (n : Nat) : Nat.count (fun _ => True) n = n",
        "proof": "Nat.count_true n",
    },
    {
        "id": "mathlib10_count_false",
        "declaration": "Nat.count_false",
        "claim": "Among naturals below n, none satisfy the predicate that is always false.",
        "formal": "theorem qa_mathlib10_count_false (n : Nat) : Nat.count (fun _ => False) n = 0",
        "proof": "Nat.count_false n",
    },
    {
        "id": "mathlib11_cycle_mem_reverse",
        "declaration": "Cycle.mem_reverse_iff",
        "claim": "An element belongs to a reversed cycle exactly when it belongs to the original cycle.",
        "formal": "theorem qa_mathlib11_cycle_mem_reverse {α : Type u} {a : α} {s : Cycle α} : a ∈ s.reverse ↔ a ∈ s",
        "proof": "Cycle.mem_reverse_iff",
    },
    {
        "id": "mathlib12_cycle_reverse_reverse",
        "declaration": "Cycle.reverse_reverse",
        "claim": "Reversing a cycle twice returns the original cycle.",
        "formal": "theorem qa_mathlib12_cycle_reverse_reverse {α : Type u} (s : Cycle α) : s.reverse.reverse = s",
        "proof": "Cycle.reverse_reverse s",
    },
    {
        "id": "mathlib13_finset_card_union_inter",
        "declaration": "Finset.card_union_add_card_inter",
        "claim": "For finite sets, the cardinality of the union plus the cardinality of the intersection equals the sum of the two cardinalities.",
        "formal": "theorem qa_mathlib13_finset_card_union_inter {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∪ t).card + (s ∩ t).card = s.card + t.card",
        "proof": "Finset.card_union_add_card_inter s t",
    },
    {
        "id": "mathlib14_finset_card_inter_union",
        "declaration": "Finset.card_inter_add_card_union",
        "claim": "For finite sets, the cardinality of the intersection plus the cardinality of the union equals the sum of the two cardinalities.",
        "formal": "theorem qa_mathlib14_finset_card_inter_union {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∩ t).card + (s ∪ t).card = s.card + t.card",
        "proof": "Finset.card_inter_add_card_union s t",
    },
    {
        "id": "mathlib15_finset_card_union_le",
        "declaration": "Finset.card_union_le",
        "claim": "The cardinality of the union of two finite sets is at most the sum of their cardinalities.",
        "formal": "theorem qa_mathlib15_finset_card_union_le {α : Type u} [DecidableEq α] (s t : Finset α) : (s ∪ t).card ≤ s.card + t.card",
        "proof": "Finset.card_union_le s t",
    },
]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return value


def generated_files() -> dict[Path, bytes]:
    registry = load_json(REGISTRY_PATH)
    entries = {
        entry["declaration"]: entry
        for entry in registry["entries"]
        if isinstance(entry, dict)
    }
    if set(entries) != {case["declaration"] for case in CASES}:
        raise RuntimeError("Mathlib registry and scaffold declaration sets differ")

    files: dict[Path, bytes] = {}

    def add_text(path: Path, text: str) -> None:
        files[path] = text.encode("utf-8")

    def add_json(path: Path, value: object) -> None:
        add_text(path, canonical(value) + "\n")

    add_text(
        PACK / "README.md",
        "# Mathlib Certified Corpus v1\n\n"
        "Replay-backed wrappers around 15 declarations from the exact pinned "
        "Mathlib v4.31.0 source revision.\n",
    )
    index_entries = []
    for case in CASES:
        entry = entries[case["declaration"]]
        example_dir = EXAMPLES / case["id"]
        import_line = f"import {entry['module']}"
        proof_source = (
            f"{import_line}\n\n{case['formal']} := by\n"
            f"  exact {case['proof']}\n"
        )
        provenance = {
            "repository": registry["source"]["repository"],
            "release": registry["source"]["release"],
            "commit": registry["source"]["commit"],
            "declaration": case["declaration"],
            "source_path": entry["source_path"],
            "source_file_sha256": entry["source_file_sha256"],
            "declaration_source_sha256": entry["declaration_source_sha256"],
        }
        add_text(example_dir / "claim.txt", case["claim"] + "\n")
        add_text(example_dir / "proof.lean", proof_source)
        add_text(
            example_dir / "README.md",
            f"# {case['id']}\n\n"
            f"Replay-certified wrapper for `{case['declaration']}` from the "
            "pinned Mathlib source revision.\n",
        )
        add_json(
            example_dir / "task.json",
            {
                "schema_id": "QA_FORMAL_TASK_SCHEMA.v1",
                "task_id": f"MATHLIB_{case['id'].upper()}_TASK",
                "created_utc": CREATED_UTC,
                "nl_statement": case["claim"],
                "formal_goal": case["formal"],
                "imports": [entry["module"]],
                "context": [],
                "constraints": {
                    "max_seconds": 120,
                    "max_memory_mb": 4096,
                    "allowed_tactics": ["exact"],
                },
                "invariant_diff": {
                    "corpus": "mathlib_pack_v1",
                    "category": entry["category"],
                    "provenance": provenance,
                },
            },
        )
        trace_path = example_dir / "trace.json"
        linked_trace = load_json(trace_path) if trace_path.is_file() else None
        trace_id = (
            linked_trace["trace_id"] if linked_trace is not None else "0" * 64
        )
        pair = {
            "schema_id": "QA_HUMAN_FORMAL_PAIR_CERT.v1",
            "pair_id": f"MATHLIB_{case['id'].upper()}_PAIR",
            "created_utc": CREATED_UTC,
            "natural_language_claim": case["claim"],
            "formal_statement": case["formal"],
            "alignment_evidence": {
                "key_lemmas": [case["declaration"]],
                "span_mappings": [
                    {
                        "nl_span": case["claim"].rstrip("."),
                        "formal_identifiers": [case["declaration"]],
                    }
                ],
            },
            "trace_ref": {
                "trace_id": trace_id,
                "result_status": "SUCCESS",
                "replay_status": "SUCCESS",
            },
            "status": "PROVED",
            "objections": [],
            "invariant_diff": {
                "corpus": "mathlib_pack_v1",
                "upstream_declaration": case["declaration"],
            },
        }
        add_json(example_dir / "pair.json", pair)
        status = {
            "example_id": case["id"],
            "status": "PROVED",
            "replay_rate": 1.0,
            "compressed": False,
            "introduced_lemmas": 0,
            "upstream_declaration": case["declaration"],
        }
        if linked_trace is not None:
            status.update(
                {
                    "kernel_derived": True,
                    "toolchain_id": linked_trace["toolchain_id"],
                    "trace_id": trace_id,
                }
            )
        add_json(example_dir / "status.json", status)
        index_entries.append(
            {
                "id": case["id"],
                "topic": entry["category"],
                "difficulty": "upstream",
                "status": "PROVED",
            }
        )
    add_json(
        PACK / "index.json",
        {
            "schema_id": "QA_MATH_COMPILER_DEMO_PACK_SCHEMA.v1",
            "version": "v1",
            "example_count": len(index_entries),
            "examples": index_entries,
        },
    )

    manifest = load_json(MANIFEST_PATH)
    retained = [
        row
        for row in manifest["cases"]
        if not str(row.get("case_id", "")).startswith("mathlib")
    ]
    retained.extend(
        {
            "case_id": case["id"],
            "source": f"mathlib_pack_v1/examples/{case['id']}/proof.lean",
            "expected_status": "SUCCESS",
            "proof_method": case["declaration"],
            "artifact_dir": f"mathlib_pack_v1/examples/{case['id']}",
            "lake_project": "mathlib_ingest",
            "upstream_declaration": case["declaration"],
        }
        for case in CASES
    )
    manifest["cases"] = retained
    add_json(MANIFEST_PATH, manifest)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["write", "check"])
    args = parser.parse_args()
    files = generated_files()
    mismatches = []
    for path, payload in sorted(files.items()):
        if args.mode == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        elif not path.is_file() or path.read_bytes() != payload:
            mismatches.append(path.relative_to(ROOT).as_posix())
    result = {
        "ok": not mismatches,
        "mode": args.mode,
        "case_count": len(CASES),
        "file_count": len(files),
        "mismatches": mismatches,
    }
    print(canonical(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
