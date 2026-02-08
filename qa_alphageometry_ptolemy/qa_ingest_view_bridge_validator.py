#!/usr/bin/env python3
"""
qa_ingest_view_bridge_validator.py

Validator for QA ingest->view bridge family.

This family composes:
- QA_DATASTORE_SEMANTICS_CERT.v1 (store root + proofs)
- QA_DATASTORE_VIEW_CERT.v1 (view root + proofs)

Bridge-specific guarantees:
- view entries are explicitly grounded in ingested documents
- each document reference carries an ingest inclusion proof
- typed view root provenance can be enforced (KEYWORD_VIEW/SEMANTIC_VIEW)
- deterministic failure taxonomy and counterexample matching
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
    from .qa_datastore_validator import (
        validate_semantics_cert as validate_store_semantics_cert,
        verify_inclusion_proof as verify_merkle_inclusion,
    )
    from .qa_datastore_view_validator import (
        ValidationFail,
        _is_hex64,
        _require,
        _validate_root_snapshot,
        _manifest_hashable_copy,
        _enforce_manifest,
        validate_view_semantics_cert,
        validate_witness_pack as validate_view_witness_pack,
    )
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical
    from qa_datastore_validator import (
        validate_semantics_cert as validate_store_semantics_cert,
        verify_inclusion_proof as verify_merkle_inclusion,
    )
    from qa_datastore_view_validator import (
        ValidationFail,
        _is_hex64,
        _require,
        _validate_root_snapshot,
        _manifest_hashable_copy,
        _enforce_manifest,
        validate_view_semantics_cert,
        validate_witness_pack as validate_view_witness_pack,
    )


HEX64_ZERO = "0" * 64
BRIDGE_VIEW_KINDS = {"KEYWORD_VIEW", "SEMANTIC_VIEW"}


def canonical_json_dumps(obj: Any) -> str:
    return canonical_json_compact(obj)


def ds_sha256(domain: str, payload: bytes) -> str:
    _require(isinstance(domain, str) and bool(domain), "DOMAIN_SEP_VIOLATION", "hash domain must be non-empty")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def record_hash(record_domain: str, record_obj: Dict[str, Any]) -> str:
    _require(isinstance(record_obj, dict), "SCHEMA_MISMATCH", "doc_record must be object")
    canonical = canonical_json_dumps(record_obj)
    _require(json.loads(canonical) == record_obj, "NON_CANONICAL_JSON", "doc_record is not canonical-json roundtrippable")
    return ds_sha256(record_domain, canonical.encode("utf-8"))


def ingest_leaf_hash(ingest_leaf_domain: str, doc_id: str, doc_record_hash_hex: str) -> str:
    _require(isinstance(doc_id, str) and bool(doc_id), "SCHEMA_MISMATCH", "doc_id must be non-empty string")
    _require(_is_hex64(doc_record_hash_hex), "SCHEMA_MISMATCH", "doc_record_hash must be 64-hex")
    payload = doc_id.encode("utf-8") + b"\x00" + doc_record_hash_hex.encode("ascii")
    return ds_sha256(ingest_leaf_domain, payload)


def _assert_strictly_increasing_strings(values: List[str], label: str) -> None:
    for i in range(1, len(values)):
        _require(
            values[i - 1] < values[i],
            "SCHEMA_MISMATCH",
            f"{label} must be strictly increasing at index {i}",
            {"left": values[i - 1], "right": values[i]},
        )


def _manifest_hash_for_doc(doc: Dict[str, Any]) -> str:
    return sha256_canonical(_manifest_hashable_copy(doc))


def rehash_manifest_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    _require(isinstance(doc, dict), "SCHEMA_MISMATCH", "rehash target must be JSON object")
    manifest = doc.get("manifest")
    _require(isinstance(manifest, dict), "SCHEMA_MISMATCH", "rehash target must contain manifest object")
    _require(manifest.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "manifest.hash_alg must be sha256")

    manifest["canonical_json_sha256"] = HEX64_ZERO
    manifest["canonical_json_sha256"] = _manifest_hash_for_doc(doc)

    with open(path, "w", encoding="utf-8") as f:
        f.write(canonical_json_dumps(doc))
        f.write("\n")

    return manifest["canonical_json_sha256"]


def _validate_ingest_root_snapshot(snapshot: Dict[str, Any], cert_domain: str) -> Tuple[str, Optional[List[str]]]:
    _require(isinstance(snapshot, dict), "SCHEMA_MISMATCH", "ingest_root_snapshot must be object")
    _require(snapshot.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "ingest_root_snapshot.hash_alg must be sha256")

    root_hash = snapshot.get("root_hash")
    _require(_is_hex64(root_hash), "SCHEMA_MISMATCH", "ingest_root_snapshot.root_hash must be 64-hex")

    doc_ids = snapshot.get("doc_ids")
    if doc_ids is None:
        return root_hash, None

    _require(isinstance(doc_ids, list), "SCHEMA_MISMATCH", "ingest_root_snapshot.doc_ids must be list when present")
    _require(all(isinstance(d, str) and bool(d) for d in doc_ids),
             "SCHEMA_MISMATCH", "ingest_root_snapshot.doc_ids entries must be non-empty strings")
    _assert_strictly_increasing_strings(doc_ids, "ingest_root_snapshot.doc_ids")

    declared = snapshot.get("doc_ids_hash")
    _require(_is_hex64(declared), "SCHEMA_MISMATCH", "ingest_root_snapshot.doc_ids_hash must be 64-hex when doc_ids present")
    computed = ds_sha256(cert_domain, canonical_json_dumps(doc_ids).encode("utf-8"))
    _require(
        declared == computed,
        "HASH_MISMATCH",
        "ingest_root_snapshot.doc_ids_hash does not match computed hash",
        {"declared_doc_ids_hash": declared, "computed_doc_ids_hash": computed},
    )
    return root_hash, doc_ids


def validate_bridge_semantics_cert(bridge_sem: Dict[str, Any]) -> Dict[str, Any]:
    _require(bridge_sem.get("schema_id") == "QA_INGEST_VIEW_BRIDGE_CERT.v1",
             "SCHEMA_MISMATCH", "bad bridge semantics schema_id")

    domains = bridge_sem.get("domains")
    _require(isinstance(domains, dict), "SCHEMA_MISMATCH", "domains must be object")
    for key in ("cert_domain", "record_domain", "ingest_leaf_domain", "ingest_node_domain", "view_node_domain"):
        _require(isinstance(domains.get(key), str) and bool(domains.get(key)),
                 "SCHEMA_MISMATCH", f"domains.{key} must be non-empty string")
    domain_values = [
        domains["cert_domain"],
        domains["record_domain"],
        domains["ingest_leaf_domain"],
        domains["ingest_node_domain"],
        domains["view_node_domain"],
    ]
    _require(len(set(domain_values)) == len(domain_values),
             "DOMAIN_SEP_VIOLATION", "bridge domains must be distinct")

    fail_types = bridge_sem.get("fail_types")
    _require(isinstance(fail_types, list) and len(fail_types) > 0,
             "SCHEMA_MISMATCH", "fail_types must be non-empty list")
    _require(all(isinstance(ft, str) and bool(ft) for ft in fail_types),
             "SCHEMA_MISMATCH", "fail_types entries must be non-empty strings")

    contract = bridge_sem.get("bridge_contract")
    _require(isinstance(contract, dict), "SCHEMA_MISMATCH", "bridge_contract must be object")
    for key in (
        "requires_store_family",
        "requires_view_family",
        "requires_doc_inclusion_proofs",
        "requires_view_to_doc_links",
        "require_typed_view_roots",
    ):
        _require(isinstance(contract.get(key), bool),
                 "SCHEMA_MISMATCH", f"bridge_contract.{key} must be boolean")

    budget = contract.get("budget_model")
    _require(isinstance(budget, dict), "SCHEMA_MISMATCH", "bridge_contract.budget_model must be object")
    _require(isinstance(budget.get("enforce_on_validation"), bool),
             "SCHEMA_MISMATCH", "budget_model.enforce_on_validation must be boolean")
    _require(isinstance(budget.get("max_entries"), int) and budget.get("max_entries") >= 1,
             "SCHEMA_MISMATCH", "budget_model.max_entries must be integer >=1")
    _require(isinstance(budget.get("max_total_tokens"), int) and budget.get("max_total_tokens") >= 1,
             "SCHEMA_MISMATCH", "budget_model.max_total_tokens must be integer >=1")

    _enforce_manifest(bridge_sem, "bridge_semantics")

    return {
        "domains": domains,
        "fail_types": set(fail_types),
        "contract": contract,
        "budget": budget,
    }


def _verify_ingest_doc_ref(
    *,
    doc_ref: Dict[str, Any],
    ingest_root_hash: str,
    bridge_cfg: Dict[str, Any],
    label: str,
) -> None:
    _require(isinstance(doc_ref, dict), "SCHEMA_MISMATCH", f"{label} must be object")

    doc_id = doc_ref.get("doc_id")
    doc_record_hash_hex = doc_ref.get("doc_record_hash")
    ingest_proof = doc_ref.get("ingest_inclusion_proof")

    _require(isinstance(doc_id, str) and bool(doc_id), "SCHEMA_MISMATCH", f"{label}.doc_id must be non-empty string")
    _require(_is_hex64(doc_record_hash_hex), "SCHEMA_MISMATCH", f"{label}.doc_record_hash must be 64-hex")
    _require(isinstance(ingest_proof, dict), "SCHEMA_MISMATCH", f"{label}.ingest_inclusion_proof must be object")

    doc_record = doc_ref.get("doc_record")
    if doc_record is not None:
        computed_doc_record_hash = record_hash(bridge_cfg["domains"]["record_domain"], doc_record)
        _require(
            computed_doc_record_hash == doc_record_hash_hex,
            "INGEST_DOC_HASH_MISMATCH",
            f"{label}.doc_record_hash does not match canonical doc_record hash",
            {
                "declared_doc_record_hash": doc_record_hash_hex,
                "computed_doc_record_hash": computed_doc_record_hash,
            },
        )

    if bridge_cfg["contract"]["requires_doc_inclusion_proofs"]:
        _require(ingest_proof.get("proof_type") == "INCLUSION",
                 "SCHEMA_MISMATCH", f"{label}.ingest_inclusion_proof.proof_type must be INCLUSION")
        _require(ingest_proof.get("hash_alg") == "sha256",
                 "SCHEMA_MISMATCH", f"{label}.ingest_inclusion_proof.hash_alg must be sha256")

        computed_leaf = ingest_leaf_hash(
            bridge_cfg["domains"]["ingest_leaf_domain"],
            doc_id,
            doc_record_hash_hex,
        )
        declared_leaf = ingest_proof.get("leaf_hash")
        _require(
            declared_leaf == computed_leaf,
            "DOMAIN_SEP_VIOLATION",
            f"{label} ingest proof leaf hash mismatch",
            {"declared_leaf_hash": declared_leaf, "computed_leaf_hash": computed_leaf},
        )

        _require(
            ingest_proof.get("root_hash") == ingest_root_hash,
            "UNVERIFIABLE_PROOF",
            f"{label} ingest proof root hash mismatch",
            {"declared_root_hash": ingest_proof.get("root_hash"), "expected_root_hash": ingest_root_hash},
        )

        verify_merkle_inclusion(
            leaf_hash_hex=computed_leaf,
            root_hash_hex=ingest_root_hash,
            path=ingest_proof.get("path", []),
            node_domain=bridge_cfg["domains"]["ingest_node_domain"],
        )


def _build_single_view_test_witness(
    *,
    entry: Dict[str, Any],
    store_root_snapshot: Dict[str, Any],
    view_root_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    proof_bundle = entry.get("proof_bundle")
    view_lookup = proof_bundle.get("view_lookup") if isinstance(proof_bundle, dict) else None
    _require(isinstance(view_lookup, dict), "VIEW_DERIVATION_UNSOUND", "entry proof_bundle.view_lookup must be object")

    return {
        "schema_id": "QA_DATASTORE_VIEW_WITNESS_PACK.v1",
        "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
        "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "store_root_snapshot": store_root_snapshot,
        "view_root_snapshot": view_root_snapshot,
        "tests": [
            {
                "test_id": f"BRIDGE::{entry.get('entry_id', 'entry')}",
                "op": "VIEW_GET",
                "view_key": entry.get("view_key"),
                "expected_posting": view_lookup.get("expected_posting"),
                "view_proof": view_lookup.get("view_proof"),
                "store_proofs_by_base_key": view_lookup.get("store_proofs_by_base_key", {}),
            }
        ],
        "manifest": {
            "hash_alg": "sha256",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }


def _validate_witness_pack_obj(
    pack: Dict[str, Any],
    bridge_cfg: Dict[str, Any],
    store_cfg: Dict[str, Any],
    view_cfg: Dict[str, Any],
    *,
    enforce_manifest: bool,
) -> None:
    _require(pack.get("schema_id") == "QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1",
             "SCHEMA_MISMATCH", "bad bridge witness pack schema_id")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in bridge witness")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in bridge witness")
    _require(pack.get("bridge_semantics_schema_id") == "QA_INGEST_VIEW_BRIDGE_CERT.v1",
             "SCHEMA_MISMATCH", "bad bridge_semantics_schema_id in bridge witness")

    if enforce_manifest:
        _enforce_manifest(pack, "bridge_witness_pack")

    store_root_snapshot = pack.get("store_root_snapshot")
    view_root_snapshot = pack.get("view_root_snapshot")
    ingest_root_snapshot = pack.get("ingest_root_snapshot")
    _require(isinstance(store_root_snapshot, dict), "SCHEMA_MISMATCH", "store_root_snapshot must be object")
    _require(isinstance(view_root_snapshot, dict), "SCHEMA_MISMATCH", "view_root_snapshot must be object")
    _require(isinstance(ingest_root_snapshot, dict), "SCHEMA_MISMATCH", "ingest_root_snapshot must be object")

    store_root_hash, _ = _validate_root_snapshot(store_root_snapshot, store_cfg["cert_domain"], "store_root_snapshot")
    view_root_hash, _ = _validate_root_snapshot(view_root_snapshot, view_cfg["cert_domain"], "view_root_snapshot")
    ingest_root_hash, _ = _validate_ingest_root_snapshot(ingest_root_snapshot, bridge_cfg["domains"]["cert_domain"])

    view_root_snapshot_id = view_root_snapshot.get("snapshot_id") if isinstance(view_root_snapshot, dict) else None

    view_root_snapshots = pack.get("view_root_snapshots")
    resolved_view_roots_by_kind: Dict[str, Dict[str, Any]] = {}
    if bridge_cfg["contract"]["require_typed_view_roots"]:
        _require(isinstance(view_root_snapshots, dict), "SCHEMA_MISMATCH", "view_root_snapshots required when typed roots enabled")

    if view_root_snapshots is not None:
        _require(isinstance(view_root_snapshots, dict), "SCHEMA_MISMATCH", "view_root_snapshots must be object when present")
        _require(set(view_root_snapshots.keys()) == BRIDGE_VIEW_KINDS,
                 "SCHEMA_MISMATCH", "view_root_snapshots keys must be KEYWORD_VIEW and SEMANTIC_VIEW")

        for kind in ("KEYWORD_VIEW", "SEMANTIC_VIEW"):
            snap = view_root_snapshots[kind]
            snap_root_hash, _ = _validate_root_snapshot(snap, view_cfg["cert_domain"], f"view_root_snapshots.{kind}")
            snap_id = snap.get("snapshot_id") if isinstance(snap, dict) else None
            _require(isinstance(snap_id, str) and bool(snap_id),
                     "SCHEMA_MISMATCH", f"view_root_snapshots.{kind}.snapshot_id must be non-empty string")
            resolved_view_roots_by_kind[kind] = {
                "snapshot": snap,
                "root_hash": snap_root_hash,
                "snapshot_id": snap_id,
            }

        _require(
            isinstance(view_root_snapshot_id, str) and bool(view_root_snapshot_id),
            "SCHEMA_MISMATCH",
            "view_root_snapshot.snapshot_id required when view_root_snapshots is present",
        )
        _require(
            view_root_hash == resolved_view_roots_by_kind["KEYWORD_VIEW"]["root_hash"]
            and view_root_snapshot_id == resolved_view_roots_by_kind["KEYWORD_VIEW"]["snapshot_id"],
            "SCHEMA_MISMATCH",
            "view_root_snapshot (default view root alias) must mirror view_root_snapshots.KEYWORD_VIEW",
        )

    entries = pack.get("entries")
    _require(isinstance(entries, list) and len(entries) > 0,
             "SCHEMA_MISMATCH", "entries must be non-empty list")

    claimed_success = pack.get("claimed_success")
    _require(claimed_success is True, "SCHEMA_MISMATCH", "claimed_success must be true for witness pack")

    if bridge_cfg["budget"]["enforce_on_validation"]:
        _require(
            len(entries) <= bridge_cfg["budget"]["max_entries"],
            "BUDGET_EXCEEDED",
            "entry count exceeds budget max_entries",
            {
                "max_entries": bridge_cfg["budget"]["max_entries"],
                "actual_entries": len(entries),
            },
        )

    total_tokens = 0
    seen_entry_ids = set()

    for i, entry in enumerate(entries):
        _require(isinstance(entry, dict), "SCHEMA_MISMATCH", f"entries[{i}] must be object")

        entry_id = entry.get("entry_id")
        view_kind = entry.get("view_kind")
        view_key = entry.get("view_key")
        token_cost = entry.get("token_cost")
        doc_refs = entry.get("doc_refs")
        proof_bundle = entry.get("proof_bundle")

        _require(isinstance(entry_id, str) and bool(entry_id), "SCHEMA_MISMATCH", f"entries[{i}].entry_id must be non-empty string")
        _require(entry_id not in seen_entry_ids, "FORK_DETECTED", f"duplicate entry_id: {entry_id}")
        seen_entry_ids.add(entry_id)

        _require(view_kind in BRIDGE_VIEW_KINDS, "SCHEMA_MISMATCH", f"entries[{i}].view_kind invalid")
        _require(isinstance(view_key, str) and bool(view_key), "SCHEMA_MISMATCH", f"entries[{i}].view_key must be non-empty string")
        _require(isinstance(token_cost, int) and token_cost >= 0,
                 "SCHEMA_MISMATCH", f"entries[{i}].token_cost must be non-negative integer")
        total_tokens += token_cost

        if bridge_cfg["contract"]["requires_view_to_doc_links"]:
            _require(isinstance(doc_refs, list) and len(doc_refs) > 0,
                     "VIEW_ENTRY_UNGROUNDED", f"entries[{i}] missing doc_refs grounding")
        else:
            _require(isinstance(doc_refs, list), "SCHEMA_MISMATCH", f"entries[{i}].doc_refs must be list")

        _require(isinstance(proof_bundle, dict), "SCHEMA_MISMATCH", f"entries[{i}].proof_bundle must be object")
        _require(proof_bundle.get("store_root_hash") == store_root_hash,
                 "UNVERIFIABLE_PROOF", f"entries[{i}] proof_bundle.store_root_hash mismatch")

        expected_view_root_hash = view_root_hash
        expected_view_snapshot_id = view_root_snapshot_id
        if view_kind in resolved_view_roots_by_kind:
            expected_view_root_hash = resolved_view_roots_by_kind[view_kind]["root_hash"]
            expected_view_snapshot_id = resolved_view_roots_by_kind[view_kind]["snapshot_id"]

        _require(proof_bundle.get("view_root_hash") == expected_view_root_hash,
                 "UNVERIFIABLE_PROOF", f"entries[{i}] proof_bundle.view_root_hash mismatch")

        if bridge_cfg["contract"]["require_typed_view_roots"]:
            declared_sid = proof_bundle.get("view_snapshot_id")
            _require(isinstance(declared_sid, str) and bool(declared_sid),
                     "SCHEMA_MISMATCH", f"entries[{i}] proof_bundle.view_snapshot_id required")
            _require(
                declared_sid == expected_view_snapshot_id,
                "UNVERIFIABLE_PROOF",
                f"entries[{i}] proof_bundle.view_snapshot_id mismatch for {view_kind}",
                {
                    "expected_view_snapshot_id": expected_view_snapshot_id,
                    "actual_view_snapshot_id": declared_sid,
                },
            )

        synthetic_view_pack = _build_single_view_test_witness(
            entry=entry,
            store_root_snapshot=store_root_snapshot,
            view_root_snapshot=(
                resolved_view_roots_by_kind[view_kind]["snapshot"]
                if view_kind in resolved_view_roots_by_kind
                else view_root_snapshot
            ),
        )
        synthetic_view_pack["manifest"]["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(synthetic_view_pack))

        try:
            validate_view_witness_pack(synthetic_view_pack, store_cfg, view_cfg)
        except ValidationFail as e:
            raise ValidationFail(
                "VIEW_DERIVATION_UNSOUND",
                f"entries[{i}] failed view derivation validation: {e}",
            )

        for j, doc_ref in enumerate(doc_refs):
            _verify_ingest_doc_ref(
                doc_ref=doc_ref,
                ingest_root_hash=ingest_root_hash,
                bridge_cfg=bridge_cfg,
                label=f"entries[{i}].doc_refs[{j}]",
            )

    if bridge_cfg["budget"]["enforce_on_validation"]:
        _require(
            total_tokens <= bridge_cfg["budget"]["max_total_tokens"],
            "BUDGET_EXCEEDED",
            "total token_cost exceeds budget",
            {
                "max_total_tokens": bridge_cfg["budget"]["max_total_tokens"],
                "actual_total_tokens": total_tokens,
            },
        )


def validate_bridge_witness_pack(
    *,
    pack: Dict[str, Any],
    bridge_cfg: Dict[str, Any],
    store_cfg: Dict[str, Any],
    view_cfg: Dict[str, Any],
) -> None:
    _validate_witness_pack_obj(
        pack,
        bridge_cfg,
        store_cfg,
        view_cfg,
        enforce_manifest=True,
    )


def validate_bridge_counterexamples_pack(
    *,
    pack: Dict[str, Any],
    bridge_cfg: Dict[str, Any],
    store_cfg: Dict[str, Any],
    view_cfg: Dict[str, Any],
) -> None:
    _require(pack.get("schema_id") == "QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1",
             "SCHEMA_MISMATCH", "bad bridge counterexamples schema_id")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in bridge counterexamples")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in bridge counterexamples")
    _require(pack.get("bridge_semantics_schema_id") == "QA_INGEST_VIEW_BRIDGE_CERT.v1",
             "SCHEMA_MISMATCH", "bad bridge_semantics_schema_id in bridge counterexamples")

    _enforce_manifest(pack, "bridge_counterexamples_pack")

    store_root_snapshot = pack.get("store_root_snapshot")
    view_root_snapshot = pack.get("view_root_snapshot")
    ingest_root_snapshot = pack.get("ingest_root_snapshot")
    view_root_snapshots = pack.get("view_root_snapshots")

    _require(isinstance(store_root_snapshot, dict), "SCHEMA_MISMATCH", "store_root_snapshot must be object")
    _require(isinstance(view_root_snapshot, dict), "SCHEMA_MISMATCH", "view_root_snapshot must be object")
    _require(isinstance(ingest_root_snapshot, dict), "SCHEMA_MISMATCH", "ingest_root_snapshot must be object")

    _validate_root_snapshot(store_root_snapshot, store_cfg["cert_domain"], "store_root_snapshot")
    _validate_root_snapshot(view_root_snapshot, view_cfg["cert_domain"], "view_root_snapshot")
    _validate_ingest_root_snapshot(ingest_root_snapshot, bridge_cfg["domains"]["cert_domain"])

    cases = pack.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0,
             "SCHEMA_MISMATCH", "cases must be non-empty list")

    for i, case in enumerate(cases):
        _require(isinstance(case, dict), "SCHEMA_MISMATCH", f"cases[{i}] must be object")
        tamper_mode = case.get("tamper_mode")
        _require(isinstance(tamper_mode, str) and bool(tamper_mode),
                 "SCHEMA_MISMATCH", f"cases[{i}].tamper_mode must be non-empty string")

        expected_fail_type = case.get("expected_fail_type")
        _require(expected_fail_type in bridge_cfg["fail_types"],
                 "SCHEMA_MISMATCH",
                 f"cases[{i}].expected_fail_type not present in bridge semantics fail_types: {expected_fail_type}")

        synthetic = {
            "schema_id": "QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1",
            "manifest": {
                "hash_alg": "sha256",
                "canonical_json_sha256": HEX64_ZERO,
            },
            "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
            "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
            "bridge_semantics_schema_id": "QA_INGEST_VIEW_BRIDGE_CERT.v1",
            "store_root_snapshot": store_root_snapshot,
            "view_root_snapshot": view_root_snapshot,
            "view_root_snapshots": view_root_snapshots,
            "ingest_root_snapshot": ingest_root_snapshot,
            "entries": case.get("entries"),
            "claimed_success": True,
        }
        synthetic["manifest"]["canonical_json_sha256"] = _manifest_hash_for_doc(synthetic)

        try:
            _validate_witness_pack_obj(
                synthetic,
                bridge_cfg,
                store_cfg,
                view_cfg,
                enforce_manifest=True,
            )
        except ValidationFail as e:
            if e.fail_type != expected_fail_type:
                raise ValidationFail(
                    "FORK_DETECTED",
                    "counterexample failed, but with unexpected fail_type",
                    {
                        "case_id": case.get("case_id", f"case_{i}"),
                        "expected_fail_type": expected_fail_type,
                        "actual_fail_type": e.fail_type,
                        "actual_msg": e.msg,
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


def validate_all(
    *,
    store_semantics_path: str,
    view_semantics_path: str,
    bridge_semantics_path: str,
    witness_path: str,
    counterexamples_path: str,
) -> None:
    store_sem = _load_json(store_semantics_path)
    store_cfg = validate_store_semantics_cert(store_sem)

    view_sem = _load_json(view_semantics_path)
    view_cfg = validate_view_semantics_cert(view_sem)

    bridge_sem = _load_json(bridge_semantics_path)
    bridge_cfg = validate_bridge_semantics_cert(bridge_sem)

    witness = _load_json(witness_path)
    validate_bridge_witness_pack(pack=witness, bridge_cfg=bridge_cfg, store_cfg=store_cfg, view_cfg=view_cfg)

    counterexamples = _load_json(counterexamples_path)
    validate_bridge_counterexamples_pack(
        pack=counterexamples,
        bridge_cfg=bridge_cfg,
        store_cfg=store_cfg,
        view_cfg=view_cfg,
    )


def _demo_paths() -> Dict[str, str]:
    base = __file__.rsplit("/", 1)[0]
    return {
        "store_semantics": f"{base}/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json",
        "view_semantics": f"{base}/certs/QA_DATASTORE_VIEW_CERT.v1.json",
        "bridge_semantics": f"{base}/certs/QA_INGEST_VIEW_BRIDGE_CERT.v1.json",
        "witness": f"{base}/certs/witness/QA_INGEST_VIEW_BRIDGE_WITNESS_PACK.v1.json",
        "counterexamples": f"{base}/certs/counterexamples/QA_INGEST_VIEW_BRIDGE_COUNTEREXAMPLES_PACK.v1.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA ingest->view bridge semantics/witness/counterexample packs.")
    parser.add_argument("--rehash", default="", help="Recompute manifest.canonical_json_sha256 for a JSON file and exit")
    parser.add_argument("--store-semantics", default="", help="Path to QA_DATASTORE_SEMANTICS cert JSON")
    parser.add_argument("--view-semantics", default="", help="Path to QA_DATASTORE_VIEW_CERT cert JSON")
    parser.add_argument("--bridge-semantics", default="", help="Path to QA_INGEST_VIEW_BRIDGE cert JSON")
    parser.add_argument("--witness", default="", help="Path to bridge witness pack JSON")
    parser.add_argument("--counterexamples", default="", help="Path to bridge counterexamples pack JSON")
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
        bridge_semantics_path = paths["bridge_semantics"]
        witness_path = paths["witness"]
        counterexamples_path = paths["counterexamples"]
    else:
        if not (args.store_semantics and args.view_semantics and args.bridge_semantics and args.witness and args.counterexamples):
            parser.error(
                "Provide --store-semantics --view-semantics --bridge-semantics --witness --counterexamples, or use --demo."
            )
        store_semantics_path = args.store_semantics
        view_semantics_path = args.view_semantics
        bridge_semantics_path = args.bridge_semantics
        witness_path = args.witness
        counterexamples_path = args.counterexamples

    try:
        validate_all(
            store_semantics_path=store_semantics_path,
            view_semantics_path=view_semantics_path,
            bridge_semantics_path=bridge_semantics_path,
            witness_path=witness_path,
            counterexamples_path=counterexamples_path,
        )
    except ValidationFail as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"FAIL: unexpected error: {e}")
        return 1

    print("OK: ingest->view bridge semantics + witness + counterexamples validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
