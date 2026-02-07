#!/usr/bin/env python3
"""
qa_arag_validator.py

Strict validator for QA A-RAG interface semantics + witness + counterexample packs.

This validator composes with datastore [18] and datastore view [20] families.
"""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
    from .qa_datastore_validator import validate_semantics_cert as validate_store_semantics_cert
    from .qa_datastore_view_validator import (
        ValidationFail,
        _is_hex64,
        _require,
        _manifest_hashable_copy,
        _enforce_manifest,
        _validate_root_snapshot,
        validate_view_semantics_cert,
        validate_witness_pack as validate_view_witness_pack,
        store_leaf_hash,
        verify_inclusion_proof,
    )
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical
    from qa_datastore_validator import validate_semantics_cert as validate_store_semantics_cert
    from qa_datastore_view_validator import (
        ValidationFail,
        _is_hex64,
        _require,
        _manifest_hashable_copy,
        _enforce_manifest,
        _validate_root_snapshot,
        validate_view_semantics_cert,
        validate_witness_pack as validate_view_witness_pack,
        store_leaf_hash,
        verify_inclusion_proof,
    )


HEX64_ZERO = "0" * 64
ARAG_TOOLS = {"keyword_search", "semantic_search", "chunk_read"}
ARAG_VIEW_KINDS = {"KEYWORD_VIEW", "SEMANTIC_VIEW", "CHUNK_STORE"}


def validate_arag_semantics_cert(arag_sem: Dict[str, Any]) -> Dict[str, Any]:
    _require(arag_sem.get("schema_id") == "QA_ARAG_INTERFACE_CERT.v1",
             "SCHEMA_MISMATCH", "bad A-RAG semantics schema_id")
    _require(arag_sem.get("version") == 1, "SCHEMA_MISMATCH", "bad A-RAG semantics version")
    _require(arag_sem.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in A-RAG semantics")
    _require(arag_sem.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in A-RAG semantics")

    tool_set = arag_sem.get("tool_set")
    _require(isinstance(tool_set, list) and len(tool_set) == 3,
             "SCHEMA_MISMATCH", "tool_set must contain exactly 3 tools")
    _require(set(tool_set) == ARAG_TOOLS,
             "SCHEMA_MISMATCH", f"tool_set must be {sorted(ARAG_TOOLS)}")

    tool_to_view_kind = arag_sem.get("tool_to_view_kind")
    _require(isinstance(tool_to_view_kind, dict),
             "SCHEMA_MISMATCH", "tool_to_view_kind must be object")
    _require(set(tool_to_view_kind.keys()) == ARAG_TOOLS,
             "SCHEMA_MISMATCH", f"tool_to_view_kind keys must be {sorted(ARAG_TOOLS)}")
    for tool_name, view_kind in tool_to_view_kind.items():
        _require(isinstance(view_kind, str) and view_kind in ARAG_VIEW_KINDS,
                 "SCHEMA_MISMATCH",
                 f"tool_to_view_kind[{tool_name}] must be one of {sorted(ARAG_VIEW_KINDS)}")

    fail_types = arag_sem.get("fail_types")
    _require(isinstance(fail_types, list) and len(fail_types) >= 6,
             "SCHEMA_MISMATCH", "fail_types must be non-empty list")
    _require(all(isinstance(ft, str) and ft for ft in fail_types),
             "SCHEMA_MISMATCH", "fail_types entries must be non-empty strings")

    budget = arag_sem.get("budget_model")
    _require(isinstance(budget, dict), "SCHEMA_MISMATCH", "budget_model must be object")
    max_steps = budget.get("max_steps")
    max_tokens = budget.get("max_retrieved_tokens")
    _require(isinstance(max_steps, int) and max_steps >= 1,
             "SCHEMA_MISMATCH", "budget_model.max_steps must be integer >=1")
    _require(isinstance(max_tokens, int) and max_tokens >= 1,
             "SCHEMA_MISMATCH", "budget_model.max_retrieved_tokens must be integer >=1")
    _require(isinstance(budget.get("enforce_on_validation"), bool),
             "SCHEMA_MISMATCH", "budget_model.enforce_on_validation must be boolean")

    trace_contract = arag_sem.get("trace_contract")
    _require(isinstance(trace_contract, dict), "SCHEMA_MISMATCH", "trace_contract must be object")
    for key in ("step_index_monotone", "root_binding_required", "proof_per_retrieval_step", "view_root_provenance_by_kind"):
        _require(isinstance(trace_contract.get(key), bool), "SCHEMA_MISMATCH", f"trace_contract.{key} must be boolean")

    hash_domains = arag_sem.get("hash_domains")
    _require(isinstance(hash_domains, dict), "SCHEMA_MISMATCH", "hash_domains must be object")
    for key in ("trace_step", "cert"):
        _require(isinstance(hash_domains.get(key), str) and bool(hash_domains.get(key)),
                 "SCHEMA_MISMATCH", f"hash_domains.{key} must be non-empty string")
    _require(hash_domains["trace_step"] != hash_domains["cert"],
             "DOMAIN_SEP_VIOLATION", "A-RAG hash_domains must be distinct")

    _enforce_manifest(arag_sem, "arag_semantics")

    return {
        "tools": set(tool_set),
        "tool_to_view_kind": dict(tool_to_view_kind),
        "fail_types": set(fail_types),
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "enforce_budget": budget["enforce_on_validation"],
        "require_monotone_steps": trace_contract["step_index_monotone"],
        "require_root_binding": trace_contract["root_binding_required"],
        "require_proofs": trace_contract["proof_per_retrieval_step"],
        "require_view_root_provenance_by_kind": trace_contract["view_root_provenance_by_kind"],
    }


def _validate_store_read(
    *,
    label: str,
    read: Dict[str, Any],
    store_root_hash: str,
    store_leaf_domain: str,
    store_node_domain: str,
) -> None:
    _require(isinstance(read, dict), "SCHEMA_MISMATCH", f"{label} must be object")
    base_key = read.get("base_key")
    record_hash_hex = read.get("record_hash")
    proof = read.get("proof")

    _require(isinstance(base_key, str) and bool(base_key), "SCHEMA_MISMATCH", f"{label}.base_key must be non-empty string")
    _require(_is_hex64(record_hash_hex), "SCHEMA_MISMATCH", f"{label}.record_hash must be 64-hex")
    _require(isinstance(proof, dict), "SCHEMA_MISMATCH", f"{label}.proof must be object")
    _require(proof.get("proof_type") == "INCLUSION", "SCHEMA_MISMATCH", f"{label}.proof.proof_type must be INCLUSION")
    _require(proof.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", f"{label}.proof.hash_alg must be sha256")

    leaf = store_leaf_hash(store_leaf_domain, base_key, record_hash_hex)
    declared_leaf = proof.get("leaf_hash")
    _require(
        declared_leaf == leaf,
        "HASH_MISMATCH",
        f"{label} leaf hash mismatch",
        {"declared_leaf_hash": declared_leaf, "computed_leaf_hash": leaf},
    )

    _require(proof.get("root_hash") == store_root_hash,
             "UNVERIFIABLE_PROOF", f"{label}.proof.root_hash must match store_root_snapshot.root_hash")

    verify_inclusion_proof(
        leaf_hash_hex=leaf,
        root_hash_hex=store_root_hash,
        path=proof.get("path", []),
        node_domain=store_node_domain,
    )


def _build_single_view_test_witness(
    *,
    test_id: str,
    lookup: Dict[str, Any],
    store_root_snapshot: Dict[str, Any],
    view_root_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    pack = {
        "schema_id": "QA_DATASTORE_VIEW_WITNESS_PACK.v1",
        "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
        "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "store_root_snapshot": store_root_snapshot,
        "view_root_snapshot": view_root_snapshot,
        "tests": [{
            "test_id": test_id,
            "op": "VIEW_GET",
            "view_key": lookup.get("view_key"),
            "expected_posting": lookup.get("expected_posting"),
            "view_proof": lookup.get("view_proof"),
            "store_proofs_by_base_key": lookup.get("store_proofs_by_base_key", {}),
        }],
        "manifest": {
            "hash_alg": "sha256",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }
    pack["manifest"]["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(pack))
    return pack


def validate_witness_pack(
    pack: Dict[str, Any],
    arag_cfg: Dict[str, Any],
    store_cfg: Dict[str, Any],
    view_cfg: Dict[str, Any],
) -> None:
    _require(pack.get("schema_id") == "QA_ARAG_WITNESS_PACK.v1",
             "SCHEMA_MISMATCH", "bad A-RAG witness pack schema_id")
    _require(pack.get("arag_semantics_schema_id") == "QA_ARAG_INTERFACE_CERT.v1",
             "SCHEMA_MISMATCH", "bad arag_semantics_schema_id in witness pack")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in witness pack")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in witness pack")

    store_root_hash, _ = _validate_root_snapshot(pack.get("store_root_snapshot"), store_cfg["cert_domain"], "store_root_snapshot")
    view_root_snapshot = pack.get("view_root_snapshot")
    view_root_hash, _ = _validate_root_snapshot(
        view_root_snapshot, view_cfg["cert_domain"], "view_root_snapshot"
    )
    view_root_snapshot_id = view_root_snapshot.get("snapshot_id") if isinstance(view_root_snapshot, dict) else None
    view_root_snapshots = pack.get("view_root_snapshots")
    resolved_view_roots_by_kind: Dict[str, Dict[str, Any]] = {}
    if view_root_snapshots is not None:
        _require(isinstance(view_root_snapshots, dict),
                 "SCHEMA_MISMATCH", "view_root_snapshots must be object when present")
        _require(set(view_root_snapshots.keys()) == {"KEYWORD_VIEW", "SEMANTIC_VIEW"},
                 "SCHEMA_MISMATCH", "view_root_snapshots keys must be exactly KEYWORD_VIEW and SEMANTIC_VIEW")
        for kind in ("KEYWORD_VIEW", "SEMANTIC_VIEW"):
            snap = view_root_snapshots[kind]
            snap_root_hash, _ = _validate_root_snapshot(
                snap, view_cfg["cert_domain"], f"view_root_snapshots.{kind}"
            )
            snap_id = snap.get("snapshot_id") if isinstance(snap, dict) else None
            if arag_cfg["require_view_root_provenance_by_kind"]:
                _require(
                    isinstance(snap_id, str) and bool(snap_id),
                    "SCHEMA_MISMATCH",
                    f"view_root_snapshots.{kind}.snapshot_id required when view_root_provenance_by_kind is enabled",
                )
            resolved_view_roots_by_kind[kind] = {
                "snapshot": snap,
                "root_hash": snap_root_hash,
                "snapshot_id": snap_id,
            }
        if arag_cfg["require_view_root_provenance_by_kind"]:
            _require(
                isinstance(view_root_snapshot_id, str) and bool(view_root_snapshot_id),
                "SCHEMA_MISMATCH",
                "view_root_snapshot.snapshot_id required when view_root_provenance_by_kind is enabled",
            )
        _require(
            view_root_hash == resolved_view_roots_by_kind["KEYWORD_VIEW"]["root_hash"]
            and view_root_snapshot_id == resolved_view_roots_by_kind["KEYWORD_VIEW"]["snapshot_id"],
            "SCHEMA_MISMATCH",
            "view_root_snapshot (default view root alias) must mirror view_root_snapshots.KEYWORD_VIEW",
        )
    elif arag_cfg["require_view_root_provenance_by_kind"]:
        _require(False, "SCHEMA_MISMATCH", "trace_contract.view_root_provenance_by_kind requires view_root_snapshots")

    question = pack.get("question")
    final_answer = pack.get("final_answer")
    claimed_success = pack.get("claimed_success")
    steps = pack.get("steps")

    _require(isinstance(question, str) and bool(question), "SCHEMA_MISMATCH", "question must be non-empty string")
    _require(isinstance(final_answer, str) and bool(final_answer), "SCHEMA_MISMATCH", "final_answer must be non-empty string")
    _require(isinstance(claimed_success, bool), "SCHEMA_MISMATCH", "claimed_success must be boolean")
    _require(claimed_success is True, "SCHEMA_MISMATCH", "witness pack must represent success trace")
    _require(isinstance(steps, list) and len(steps) > 0, "SCHEMA_MISMATCH", "steps must be non-empty list")

    _enforce_manifest(pack, "arag_witness_pack")

    total_tokens = 0
    seen_indices: List[int] = []

    for i, step in enumerate(steps):
        _require(isinstance(step, dict), "SCHEMA_MISMATCH", f"steps[{i}] must be object")

        step_index = step.get("step_index")
        tool_name = step.get("tool_name")
        view_kind = step.get("view_kind")
        query = step.get("query")
        token_cost = step.get("retrieved_token_cost")
        proof_bundle = step.get("proof_bundle")

        _require(isinstance(step_index, int) and step_index >= 0,
                 "SCHEMA_MISMATCH", f"steps[{i}].step_index must be integer >= 0")
        _require(tool_name in arag_cfg["tools"],
                 "SCHEMA_MISMATCH", f"steps[{i}].tool_name must be in semantics tool_set")
        _require(isinstance(view_kind, str) and bool(view_kind),
                 "SCHEMA_MISMATCH", f"steps[{i}].view_kind must be non-empty string")
        expected_view_kind = arag_cfg["tool_to_view_kind"][tool_name]
        _require(
            view_kind == expected_view_kind,
            "WRONG_GENERATOR_SELECTION",
            f"steps[{i}] view_kind does not match tool_name contract",
            {"tool_name": tool_name, "expected_view_kind": expected_view_kind, "actual_view_kind": view_kind},
        )
        _require(isinstance(query, str) and bool(query),
                 "SCHEMA_MISMATCH", f"steps[{i}].query must be non-empty string")
        _require(isinstance(token_cost, int) and token_cost >= 0,
                 "SCHEMA_MISMATCH", f"steps[{i}].retrieved_token_cost must be integer >= 0")
        _require(isinstance(proof_bundle, dict),
                 "SCHEMA_MISMATCH", f"steps[{i}].proof_bundle must be object")

        seen_indices.append(step_index)
        total_tokens += token_cost

        if arag_cfg["require_root_binding"]:
            expected_view_root_hash = view_root_hash
            expected_view_snapshot_id = view_root_snapshot_id
            if view_kind in ("KEYWORD_VIEW", "SEMANTIC_VIEW") and view_root_snapshots is not None:
                expected_view_root_hash = resolved_view_roots_by_kind[view_kind]["root_hash"]
                expected_view_snapshot_id = resolved_view_roots_by_kind[view_kind]["snapshot_id"]
            _require(proof_bundle.get("store_root_hash") == store_root_hash,
                     "UNVERIFIABLE_PROOF", f"steps[{i}] proof_bundle.store_root_hash mismatch")
            _require(proof_bundle.get("view_root_hash") == expected_view_root_hash,
                     "UNVERIFIABLE_PROOF", f"steps[{i}] proof_bundle.view_root_hash mismatch")
            if view_kind in ("KEYWORD_VIEW", "SEMANTIC_VIEW") and arag_cfg["require_view_root_provenance_by_kind"]:
                declared_view_snapshot_id = proof_bundle.get("view_snapshot_id")
                _require(
                    isinstance(declared_view_snapshot_id, str) and bool(declared_view_snapshot_id),
                    "SCHEMA_MISMATCH",
                    f"steps[{i}] proof_bundle.view_snapshot_id required for typed view steps",
                )
                _require(
                    declared_view_snapshot_id == expected_view_snapshot_id,
                    "UNVERIFIABLE_PROOF",
                    f"steps[{i}] proof_bundle.view_snapshot_id mismatch for {view_kind}",
                    {
                        "expected_view_snapshot_id": expected_view_snapshot_id,
                        "actual_view_snapshot_id": declared_view_snapshot_id,
                    },
                )

        view_lookup = proof_bundle.get("view_lookup")
        store_reads = proof_bundle.get("store_reads")

        if tool_name in ("keyword_search", "semantic_search"):
            _require(view_lookup is not None, "UNVERIFIABLE_PROOF",
                     f"steps[{i}] {tool_name} requires view_lookup proof bundle")
            _require(store_reads in (None, []), "SCHEMA_MISMATCH",
                     f"steps[{i}] {tool_name} must not include store_reads")
            tmp_view_pack = _build_single_view_test_witness(
                test_id=f"ARAG_STEP::{step_index}",
                lookup=view_lookup,
                store_root_snapshot=pack.get("store_root_snapshot"),
                view_root_snapshot=(
                    resolved_view_roots_by_kind[view_kind]["snapshot"]
                    if view_kind in ("KEYWORD_VIEW", "SEMANTIC_VIEW") and view_root_snapshots is not None
                    else pack.get("view_root_snapshot")
                ),
            )
            validate_view_witness_pack(tmp_view_pack, store_cfg, view_cfg)

        elif tool_name == "chunk_read":
            _require(view_lookup in (None, {}), "SCHEMA_MISMATCH",
                     f"steps[{i}] chunk_read must not include view_lookup")
            _require(isinstance(store_reads, list) and len(store_reads) > 0,
                     "UNVERIFIABLE_PROOF", f"steps[{i}] chunk_read requires non-empty store_reads")

            seen_base_keys = set()
            for j, read in enumerate(store_reads):
                _validate_store_read(
                    label=f"steps[{i}].proof_bundle.store_reads[{j}]",
                    read=read,
                    store_root_hash=store_root_hash,
                    store_leaf_domain=store_cfg["leaf_domain"],
                    store_node_domain=store_cfg["node_domain"],
                )
                base_key = read["base_key"]
                _require(base_key not in seen_base_keys, "SCHEMA_MISMATCH",
                         f"steps[{i}] duplicate base_key in store_reads: {base_key}")
                seen_base_keys.add(base_key)

    if arag_cfg["require_monotone_steps"]:
        _require(seen_indices == sorted(seen_indices),
                 "SCHEMA_MISMATCH", "step_index values must be monotone non-decreasing")

    if arag_cfg["enforce_budget"]:
        _require(len(steps) <= arag_cfg["max_steps"],
                 "BUDGET_EXCEEDED",
                 "step budget exceeded",
                 {"max_steps": arag_cfg["max_steps"], "actual_steps": len(steps)})
        _require(total_tokens <= arag_cfg["max_tokens"],
                 "BUDGET_EXCEEDED",
                 "retrieved token budget exceeded",
                 {"max_retrieved_tokens": arag_cfg["max_tokens"], "actual_retrieved_tokens": total_tokens})


def validate_counterexamples_pack(
    pack: Dict[str, Any],
    arag_cfg: Dict[str, Any],
    store_cfg: Dict[str, Any],
    view_cfg: Dict[str, Any],
) -> None:
    _require(pack.get("schema_id") == "QA_ARAG_COUNTEREXAMPLES_PACK.v1",
             "SCHEMA_MISMATCH", "bad A-RAG counterexamples pack schema_id")
    _require(pack.get("arag_semantics_schema_id") == "QA_ARAG_INTERFACE_CERT.v1",
             "SCHEMA_MISMATCH", "bad arag_semantics_schema_id in counterexamples pack")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in counterexamples pack")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in counterexamples pack")

    _validate_root_snapshot(pack.get("store_root_snapshot"), store_cfg["cert_domain"], "store_root_snapshot")
    _validate_root_snapshot(pack.get("view_root_snapshot"), view_cfg["cert_domain"], "view_root_snapshot")

    cases = pack.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0,
             "SCHEMA_MISMATCH", "cases must be non-empty list")

    _enforce_manifest(pack, "arag_counterexamples_pack")

    allowed_fail_types = arag_cfg["fail_types"]

    for i, case in enumerate(cases):
        _require(isinstance(case, dict), "SCHEMA_MISMATCH", f"cases[{i}] must be object")
        tamper_mode = case.get("tamper_mode")
        _require(isinstance(tamper_mode, str) and bool(tamper_mode),
                 "SCHEMA_MISMATCH", f"cases[{i}].tamper_mode must be non-empty string")

        expected_fail_type = case.get("expected_fail_type")
        _require(expected_fail_type in allowed_fail_types,
                 "SCHEMA_MISMATCH",
                 f"cases[{i}].expected_fail_type not present in arag semantics fail_types: {expected_fail_type}")

        synthetic = {
            "schema_id": "QA_ARAG_WITNESS_PACK.v1",
            "arag_semantics_schema_id": "QA_ARAG_INTERFACE_CERT.v1",
            "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
            "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
            "store_root_snapshot": pack.get("store_root_snapshot"),
            "view_root_snapshot": pack.get("view_root_snapshot"),
            "view_root_snapshots": pack.get("view_root_snapshots"),
            "question": case.get("question"),
            "steps": case.get("steps"),
            "final_answer": case.get("final_answer"),
            "claimed_success": case.get("claimed_success"),
            "manifest": {
                "hash_alg": "sha256",
                "canonical_json_sha256": HEX64_ZERO,
            },
        }
        synthetic["manifest"]["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(synthetic))

        try:
            validate_witness_pack(synthetic, arag_cfg, store_cfg, view_cfg)
        except ValidationFail as vf:
            if vf.fail_type != expected_fail_type:
                raise ValidationFail(
                    "FORK_DETECTED",
                    "counterexample failed, but with unexpected fail_type",
                    {
                        "case_id": case.get("case_id", f"case_{i}"),
                        "expected_fail_type": expected_fail_type,
                        "actual_fail_type": vf.fail_type,
                        "actual_msg": vf.msg,
                    },
                )
            continue

        raise ValidationFail(
            "UNVERIFIABLE_PROOF",
            "counterexample unexpectedly validated (it must fail)",
            {"case_id": case.get("case_id", f"case_{i}")},
        )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rehash_manifest_file(path: str) -> str:
    doc = _load_json(path)
    _require(isinstance(doc, dict), "SCHEMA_MISMATCH", "rehash target must be a JSON object")
    manifest = doc.get("manifest")
    _require(isinstance(manifest, dict), "SCHEMA_MISMATCH", "rehash target must contain manifest object")
    _require(manifest.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "manifest.hash_alg must be sha256")

    manifest["canonical_json_sha256"] = HEX64_ZERO
    manifest["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(doc))

    with open(path, "w", encoding="utf-8") as f:
        f.write(canonical_json_compact(doc))
        f.write("\n")

    return manifest["canonical_json_sha256"]


def validate_all(
    *,
    store_semantics_path: str,
    view_semantics_path: str,
    arag_semantics_path: str,
    witness_path: str,
    counterexamples_path: str,
) -> None:
    store_sem = _load_json(store_semantics_path)
    store_cfg = validate_store_semantics_cert(store_sem)

    view_sem = _load_json(view_semantics_path)
    view_cfg = validate_view_semantics_cert(view_sem)

    arag_sem = _load_json(arag_semantics_path)
    arag_cfg = validate_arag_semantics_cert(arag_sem)

    witness = _load_json(witness_path)
    validate_witness_pack(witness, arag_cfg, store_cfg, view_cfg)

    counterexamples = _load_json(counterexamples_path)
    validate_counterexamples_pack(counterexamples, arag_cfg, store_cfg, view_cfg)


def _demo_paths() -> Dict[str, str]:
    base = __file__.rsplit("/", 1)[0]
    return {
        "store_semantics": f"{base}/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json",
        "view_semantics": f"{base}/certs/QA_DATASTORE_VIEW_CERT.v1.json",
        "arag_semantics": f"{base}/certs/QA_ARAG_INTERFACE_CERT.v1.json",
        "witness": f"{base}/certs/witness/QA_ARAG_WITNESS_PACK.v1.json",
        "counterexamples": f"{base}/certs/counterexamples/QA_ARAG_COUNTEREXAMPLES_PACK.v1.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA A-RAG semantics/witness/counterexample packs.")
    parser.add_argument("--rehash", default="", help="Recompute manifest.canonical_json_sha256 for a JSON file and exit")
    parser.add_argument("--store-semantics", default="", help="Path to QA_DATASTORE_SEMANTICS cert JSON")
    parser.add_argument("--view-semantics", default="", help="Path to QA_DATASTORE_VIEW_CERT cert JSON")
    parser.add_argument("--arag-semantics", default="", help="Path to QA_ARAG_INTERFACE_CERT cert JSON")
    parser.add_argument("--witness", default="", help="Path to A-RAG witness pack JSON")
    parser.add_argument("--counterexamples", default="", help="Path to A-RAG counterexamples pack JSON")
    parser.add_argument("--demo", action="store_true", help="Validate built-in demo certs under certs/")
    args = parser.parse_args()

    if args.rehash:
        try:
            digest = rehash_manifest_file(args.rehash)
        except ValidationFail as e:
            print(f"FAIL: {e}")
            return 1
        except Exception as e:  # pragma: no cover
            print(f"FAIL: unexpected error: {e}")
            return 1
        print(f"OK: rehashed manifest for {args.rehash}")
        print(f"  canonical_json_sha256 = {digest}")
        return 0

    if args.demo:
        paths = _demo_paths()
        store_semantics_path = paths["store_semantics"]
        view_semantics_path = paths["view_semantics"]
        arag_semantics_path = paths["arag_semantics"]
        witness_path = paths["witness"]
        counterexamples_path = paths["counterexamples"]
    else:
        if not (args.store_semantics and args.view_semantics and args.arag_semantics and args.witness and args.counterexamples):
            parser.error(
                "Provide --store-semantics --view-semantics --arag-semantics --witness --counterexamples, or use --demo."
            )
        store_semantics_path = args.store_semantics
        view_semantics_path = args.view_semantics
        arag_semantics_path = args.arag_semantics
        witness_path = args.witness
        counterexamples_path = args.counterexamples

    try:
        validate_all(
            store_semantics_path=store_semantics_path,
            view_semantics_path=view_semantics_path,
            arag_semantics_path=arag_semantics_path,
            witness_path=witness_path,
            counterexamples_path=counterexamples_path,
        )
    except ValidationFail as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"FAIL: unexpected error: {e}")
        return 1

    print("OK: A-RAG semantics + witness + counterexamples validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
