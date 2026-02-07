#!/usr/bin/env python3
"""
qa_datastore_build_snapshot.py

Build a deterministic Merkle snapshot and inclusion proofs from sorted datastore items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Dict, List, Tuple

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical


# --- Constants ---

HEX64_ZERO = "0" * 64


# --- Hashing Helpers ---

def canonical_json_dumps(obj: Any) -> str:
    return canonical_json_compact(obj)


def ds_sha256(domain: str, payload: bytes) -> str:
    if not domain:
        raise ValueError("domain must be non-empty")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def keys_hash(cert_domain: str, keys: List[str]) -> str:
    payload = canonical_json_dumps(keys).encode("utf-8")
    return ds_sha256(cert_domain, payload)


def record_hash(record_domain: str, record_obj: Dict[str, Any]) -> str:
    serialized = canonical_json_dumps(record_obj)
    if json.loads(serialized) != record_obj:
        raise ValueError("record is not canonical-json roundtrippable")
    return ds_sha256(record_domain, serialized.encode("utf-8"))


def keyed_leaf_hash(leaf_domain: str, key: str, rec_hash_hex: str) -> str:
    payload = key.encode("utf-8") + b"\x00" + rec_hash_hex.encode("ascii")
    return ds_sha256(leaf_domain, payload)


def merkle_parent_hash(node_domain: str, left_hex: str, right_hex: str) -> str:
    payload = left_hex.encode("ascii") + b"\x00" + right_hex.encode("ascii")
    return ds_sha256(node_domain, payload)


# --- Merkle Builder ---

def _assert_strictly_sorted(keys: List[str]) -> None:
    for i in range(1, len(keys)):
        if not (keys[i - 1] < keys[i]):
            raise ValueError(
                f"keys must be strictly increasing; violation at {i-1}='{keys[i-1]}' and {i}='{keys[i]}'"
            )


def build_merkle_with_paths(
    node_domain: str,
    leaf_hashes: List[str],
) -> Tuple[str, List[List[Dict[str, str]]]]:
    """
    Returns:
        root_hash, paths
    where paths[i] is list of {side, sibling_hash} steps from leaf -> root.

    Deterministic odd-node rule:
        duplicate last node at each level.
    """
    if not leaf_hashes:
        raise ValueError("need at least one leaf")

    n = len(leaf_hashes)
    paths: List[List[Dict[str, str]]] = [[] for _ in range(n)]
    current: List[Tuple[str, List[int]]] = [(h, [i]) for i, h in enumerate(leaf_hashes)]

    while len(current) > 1:
        nxt: List[Tuple[str, List[int]]] = []
        i = 0
        while i < len(current):
            left_hash, left_indices = current[i]

            if i + 1 < len(current):
                right_hash, right_indices = current[i + 1]

                for idx in left_indices:
                    paths[idx].append({"side": "R", "sibling_hash": right_hash})
                for idx in right_indices:
                    paths[idx].append({"side": "L", "sibling_hash": left_hash})

                combined_indices = left_indices + right_indices
                i += 2
            else:
                # Duplicate the last node when cardinality is odd.
                right_hash = left_hash
                for idx in left_indices:
                    paths[idx].append({"side": "R", "sibling_hash": right_hash})
                combined_indices = list(left_indices)
                i += 1

            parent_hash = merkle_parent_hash(node_domain, left_hash, right_hash)
            nxt.append((parent_hash, combined_indices))

        current = nxt

    return current[0][0], paths


# --- Snapshot + Pack Builders ---

def build_snapshot(
    items: List[Dict[str, Any]],
    *,
    snapshot_id: str,
    record_domain: str,
    leaf_domain: str,
    node_domain: str,
    cert_domain: str,
) -> Dict[str, Any]:
    keys = [it["key"] for it in items]
    _assert_strictly_sorted(keys)

    rec_hashes: List[str] = []
    leaf_hashes: List[str] = []
    for it in items:
        rh = record_hash(record_domain, it["record"])
        rec_hashes.append(rh)
        leaf_hashes.append(keyed_leaf_hash(leaf_domain, it["key"], rh))

    root_hash, paths = build_merkle_with_paths(node_domain, leaf_hashes)

    proofs_by_key: Dict[str, Any] = {}
    leaf_hashes_by_key: Dict[str, str] = {}
    for i, it in enumerate(items):
        key = it["key"]
        leaf_hashes_by_key[key] = leaf_hashes[i]
        proofs_by_key[key] = {
            "proof_type": "INCLUSION",
            "hash_alg": "sha256",
            "leaf_hash": leaf_hashes[i],
            "root_hash": root_hash,
            "path": paths[i],
        }

    snapshot = {
        "snapshot_id": snapshot_id,
        "hash_alg": "sha256",
        "root_hash": root_hash,
        "keys": keys,
        "keys_hash": keys_hash(cert_domain, keys),
        "leaf_hashes_by_key": leaf_hashes_by_key,
        "proofs_by_key": proofs_by_key,
        "manifest": {
            "hash_alg": "sha256",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }
    snapshot["manifest"]["canonical_json_sha256"] = _manifest_hash_for_obj_excluding_manifest(snapshot)
    return snapshot


def _manifest_hashable_copy(doc: Dict[str, Any]) -> Dict[str, Any]:
    clone = json.loads(json.dumps(doc))
    if isinstance(clone.get("manifest"), dict):
        clone["manifest"]["canonical_json_sha256"] = HEX64_ZERO
    return clone


def _manifest_hash_for_obj_excluding_manifest(doc: Dict[str, Any]) -> str:
    return sha256_canonical(_manifest_hashable_copy(doc))


def build_non_inclusion_range_proof(snapshot: Dict[str, Any], query_key: str) -> Dict[str, Any]:
    keys = snapshot["keys"]
    proofs = snapshot["proofs_by_key"]
    leaf_hashes = snapshot["leaf_hashes_by_key"]

    if query_key in proofs:
        raise ValueError(f"query_key exists in snapshot: {query_key}")

    pred_key = None
    succ_key = None

    for k in keys:
        if k < query_key:
            pred_key = k
        elif query_key < k:
            succ_key = k
            break

    predecessor = None
    successor = None

    if pred_key is not None:
        predecessor = {
            "leaf_key": pred_key,
            "leaf_hash": leaf_hashes[pred_key],
            "proof": proofs[pred_key],
        }

    if succ_key is not None:
        successor = {
            "leaf_key": succ_key,
            "leaf_hash": leaf_hashes[succ_key],
            "proof": proofs[succ_key],
        }

    if predecessor is None and successor is None:
        raise ValueError("cannot build non-inclusion proof with empty keyset")

    return {
        "proof_type": "NON_INCLUSION_RANGE",
        "hash_alg": "sha256",
        "leaf_hash": hashlib.sha256(f"NON_MEMBER::{query_key}".encode("utf-8")).hexdigest(),
        "root_hash": snapshot["root_hash"],
        "path": [],
        "range": {
            "query_key": query_key,
            "predecessor": predecessor,
            "successor": successor,
        },
    }


def build_witness_pack(
    snapshot: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    snapshot_id: str,
    missing_keys: List[str],
) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []

    for item in items:
        key = item["key"]
        tests.append({
            "test_id": f"GET::{key}",
            "op": "GET",
            "key": key,
            "expected": item["record"],
            "proof": snapshot["proofs_by_key"][key],
        })

    for key in missing_keys:
        tests.append({
            "test_id": f"GET::{key}",
            "op": "GET",
            "key": key,
            "expected": None,
            "proof": build_non_inclusion_range_proof(snapshot, key),
        })

    witness = {
        "schema_id": "QA_DATASTORE_WITNESS_PACK.v1",
        "semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "root_snapshot": {
            "root_hash": snapshot["root_hash"],
            "hash_alg": "sha256",
            "snapshot_id": snapshot_id,
            "keys": snapshot["keys"],
            "keys_hash": snapshot["keys_hash"],
        },
        "tests": tests,
        "manifest": {
            "hash_alg": "sha256",
            "canonical_json_sha256": HEX64_ZERO,
        },
    }
    witness["manifest"]["canonical_json_sha256"] = _manifest_hash_for_obj_excluding_manifest(witness)
    return witness


# --- IO ---

def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    items = obj.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("input JSON must contain non-empty 'items' list")

    parsed: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{i}] must be object")
        key = item.get("key")
        record = item.get("record")
        if not isinstance(key, str) or not key:
            raise ValueError(f"items[{i}].key must be non-empty string")
        if not isinstance(record, dict):
            raise ValueError(f"items[{i}].record must be object")
        parsed.append({"key": key, "record": record})

    _assert_strictly_sorted([it["key"] for it in parsed])
    return parsed


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(canonical_json_dumps(obj))
        f.write("\n")


# --- CLI ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic QA datastore snapshots and witness packs.")
    parser.add_argument("--input", required=True, help="Input JSON with {items:[{key,record},...]} sorted by key")
    parser.add_argument("--out_snapshot", required=True, help="Output snapshot JSON path")
    parser.add_argument("--out_witness_pack", default="", help="Optional witness pack output path")
    parser.add_argument("--missing-key", action="append", default=[],
                        help="Optional absent key to include as NON_INCLUSION_RANGE witness test")
    parser.add_argument("--snapshot_id", default="SNAPSHOT.v1")
    parser.add_argument("--dom_record", default="QA/RECORD/v1")
    parser.add_argument("--dom_leaf", default="QA/LEAF/v1")
    parser.add_argument("--dom_node", default="QA/MERKLE_NODE/v1")
    parser.add_argument("--dom_cert", default="QA/CERT/v1")
    args = parser.parse_args()

    items = load_items(args.input)
    snapshot = build_snapshot(
        items,
        snapshot_id=args.snapshot_id,
        record_domain=args.dom_record,
        leaf_domain=args.dom_leaf,
        node_domain=args.dom_node,
        cert_domain=args.dom_cert,
    )
    write_json(args.out_snapshot, snapshot)

    if args.out_witness_pack:
        witness = build_witness_pack(
            snapshot,
            items,
            snapshot_id=args.snapshot_id,
            missing_keys=args.missing_key,
        )
        write_json(args.out_witness_pack, witness)

    print("OK: built snapshot")
    print(f"root_hash={snapshot['root_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
