#!/usr/bin/env python3
"""
qa_datastore_view_validator.py

Deterministic validator for QA datastore view semantics + witness + counterexample packs.

Implements:
1. Manifest self-hash enforcement
2. Domain-separated hashing for posting/view leaves
3. Dual-root proof checks (view root + store root)
4. Optional keys/keys_hash + adjacency guardrails
5. Counterexample fail_type taxonomy enforcement
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical


HEX64_ZERO = "0" * 64


@dataclass
class ValidationFail(Exception):
    fail_type: str
    msg: str
    invariant_diff: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.invariant_diff:
            return f"{self.fail_type}: {self.msg} | invariant_diff={self.invariant_diff}"
        return f"{self.fail_type}: {self.msg}"


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require(condition: bool, fail_type: str, msg: str,
             invariant_diff: Optional[Dict[str, Any]] = None) -> None:
    if not condition:
        raise ValidationFail(fail_type, msg, invariant_diff)


def canonical_json_dumps(obj: Any) -> str:
    return canonical_json_compact(obj)


def _manifest_hashable_copy(doc: Dict[str, Any]) -> Dict[str, Any]:
    cpy = copy.deepcopy(doc)
    manifest = cpy.get("manifest")
    if isinstance(manifest, dict) and "canonical_json_sha256" in manifest:
        manifest["canonical_json_sha256"] = HEX64_ZERO
    return cpy


def _manifest_hash_for_obj_excluding_manifest(doc: Dict[str, Any], label: str) -> str:
    _require(isinstance(doc, dict), "SCHEMA_MISMATCH", f"{label} must be object")
    manifest = doc.get("manifest")
    _require(isinstance(manifest, dict), "SCHEMA_MISMATCH", f"{label}.manifest must be object")
    return sha256_canonical(_manifest_hashable_copy(doc))


def _enforce_manifest(doc: Dict[str, Any], label: str) -> None:
    manifest = doc.get("manifest")
    _require(isinstance(manifest, dict), "SCHEMA_MISMATCH", f"{label}.manifest must be object")

    hash_alg = manifest.get("hash_alg")
    claimed = manifest.get("canonical_json_sha256")
    _require(hash_alg == "sha256", "SCHEMA_MISMATCH", f"{label}.manifest.hash_alg must be sha256")
    _require(_is_hex64(claimed), "SCHEMA_MISMATCH", f"{label}.manifest.canonical_json_sha256 must be 64-hex")

    recomputed = _manifest_hash_for_obj_excluding_manifest(doc, label)
    _require(
        recomputed == claimed,
        "HASH_MISMATCH",
        f"{label} manifest canonical_json_sha256 mismatch",
        {"expected": claimed, "actual": recomputed},
    )


def _assert_strictly_increasing_strings(values: List[str], label: str) -> None:
    for i in range(1, len(values)):
        _require(
            values[i - 1] < values[i],
            "SCHEMA_MISMATCH",
            f"{label} must be strictly increasing at index {i}",
            {"left": values[i - 1], "right": values[i]},
        )


def ds_sha256(domain: str, payload: bytes) -> str:
    _require(isinstance(domain, str) and bool(domain), "DOMAIN_SEP_VIOLATION", "hash domain must be non-empty")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def merkle_parent_hash(node_domain: str, left_hex: str, right_hex: str) -> str:
    _require(_is_hex64(left_hex), "SCHEMA_MISMATCH", "left child hash must be 64-hex")
    _require(_is_hex64(right_hex), "SCHEMA_MISMATCH", "right child hash must be 64-hex")
    payload = left_hex.encode("ascii") + b"\x00" + right_hex.encode("ascii")
    return ds_sha256(node_domain, payload)


def keys_hash(cert_domain: str, keys: List[str]) -> str:
    return ds_sha256(cert_domain, canonical_json_dumps(keys).encode("utf-8"))


def posting_hash(posting_domain: str, posting: List[str]) -> str:
    _require(isinstance(posting, list), "SCHEMA_MISMATCH", "posting must be list")
    _require(all(isinstance(k, str) and bool(k) for k in posting),
             "SCHEMA_MISMATCH", "posting entries must be non-empty strings")
    _assert_strictly_increasing_strings(posting, "expected_posting")
    return ds_sha256(posting_domain, canonical_json_dumps(posting).encode("utf-8"))


def view_leaf_hash(view_leaf_domain: str, view_key: str, posting_hash_hex: str) -> str:
    _require(isinstance(view_key, str) and bool(view_key), "SCHEMA_MISMATCH", "view_key must be non-empty string")
    _require(_is_hex64(posting_hash_hex), "SCHEMA_MISMATCH", "posting_hash must be 64-hex")
    payload = view_key.encode("utf-8") + b"\x00" + posting_hash_hex.encode("ascii")
    return ds_sha256(view_leaf_domain, payload)


def store_leaf_hash(store_leaf_domain: str, base_key: str, record_hash_hex: str) -> str:
    _require(isinstance(base_key, str) and bool(base_key), "SCHEMA_MISMATCH", "base_key must be non-empty string")
    _require(_is_hex64(record_hash_hex), "SCHEMA_MISMATCH", "record_hash must be 64-hex")
    payload = base_key.encode("utf-8") + b"\x00" + record_hash_hex.encode("ascii")
    return ds_sha256(store_leaf_domain, payload)


def verify_inclusion_proof(
    *,
    leaf_hash_hex: str,
    root_hash_hex: str,
    path: List[Dict[str, Any]],
    node_domain: str,
) -> None:
    _require(_is_hex64(leaf_hash_hex), "SCHEMA_MISMATCH", "leaf_hash must be 64-hex")
    _require(_is_hex64(root_hash_hex), "SCHEMA_MISMATCH", "root_hash must be 64-hex")
    _require(isinstance(path, list), "SCHEMA_MISMATCH", "path must be list")

    cur = leaf_hash_hex
    for i, step in enumerate(path):
        _require(isinstance(step, dict), "SCHEMA_MISMATCH", f"path[{i}] must be object")
        side = step.get("side")
        sibling = step.get("sibling_hash")

        _require(side in ("L", "R"), "SCHEMA_MISMATCH", f"path[{i}].side must be L or R")
        _require(_is_hex64(sibling), "SCHEMA_MISMATCH", f"path[{i}].sibling_hash must be 64-hex")

        if side == "L":
            cur = merkle_parent_hash(node_domain, sibling, cur)
        else:
            cur = merkle_parent_hash(node_domain, cur, sibling)

    _require(
        cur == root_hash_hex,
        "UNVERIFIABLE_PROOF",
        "inclusion path does not resolve to declared root",
        {"computed_root": cur, "declared_root": root_hash_hex},
    )


def _verify_range_bound(
    *,
    bound: Dict[str, Any],
    bound_name: str,
    root_hash_hex: str,
    node_domain: str,
) -> str:
    leaf_key = bound.get("leaf_key")
    leaf_hash = bound.get("leaf_hash")
    proof = bound.get("proof")

    _require(isinstance(leaf_key, str) and bool(leaf_key),
             "SCHEMA_MISMATCH", f"{bound_name}.leaf_key must be non-empty string")
    _require(_is_hex64(leaf_hash), "SCHEMA_MISMATCH", f"{bound_name}.leaf_hash must be 64-hex")
    _require(isinstance(proof, dict), "SCHEMA_MISMATCH", f"{bound_name}.proof must be object")

    verify_inclusion_proof(
        leaf_hash_hex=leaf_hash,
        root_hash_hex=root_hash_hex,
        path=proof.get("path", []),
        node_domain=node_domain,
    )
    return leaf_key


def verify_non_inclusion_range_proof(
    *,
    query_key: str,
    predecessor: Optional[Dict[str, Any]],
    successor: Optional[Dict[str, Any]],
    root_hash_hex: str,
    node_domain: str,
) -> Tuple[Optional[str], Optional[str]]:
    _require(isinstance(query_key, str) and bool(query_key),
             "SCHEMA_MISMATCH", "range.query_key must be non-empty string")
    _require(predecessor is not None or successor is not None,
             "SCHEMA_MISMATCH", "range proof must provide predecessor and/or successor")

    pred_key: Optional[str] = None
    succ_key: Optional[str] = None

    if predecessor is not None:
        _require(isinstance(predecessor, dict), "SCHEMA_MISMATCH", "predecessor must be object or null")
        pred_key = _verify_range_bound(
            bound=predecessor,
            bound_name="predecessor",
            root_hash_hex=root_hash_hex,
            node_domain=node_domain,
        )
        _require(pred_key < query_key, "UNVERIFIABLE_PROOF", "query_key must be > predecessor.leaf_key")

    if successor is not None:
        _require(isinstance(successor, dict), "SCHEMA_MISMATCH", "successor must be object or null")
        succ_key = _verify_range_bound(
            bound=successor,
            bound_name="successor",
            root_hash_hex=root_hash_hex,
            node_domain=node_domain,
        )
        _require(query_key < succ_key, "UNVERIFIABLE_PROOF", "query_key must be < successor.leaf_key")

    if pred_key is not None and succ_key is not None:
        _require(pred_key < succ_key, "UNVERIFIABLE_PROOF", "predecessor.leaf_key must be < successor.leaf_key")

    return pred_key, succ_key


def verify_non_inclusion_adjacency(
    *,
    keys: List[str],
    query_key: str,
    pred_key: Optional[str],
    succ_key: Optional[str],
) -> None:
    _require(query_key not in keys, "UNVERIFIABLE_PROOF", "query_key appears in root snapshot key list")
    idx = bisect_left(keys, query_key)
    expected_pred = keys[idx - 1] if idx > 0 else None
    expected_succ = keys[idx] if idx < len(keys) else None
    _require(
        pred_key == expected_pred and succ_key == expected_succ,
        "UNVERIFIABLE_PROOF",
        "non-inclusion bounds are not adjacent neighbors in root snapshot keys",
        {
            "expected_pred": expected_pred,
            "actual_pred": pred_key,
            "expected_succ": expected_succ,
            "actual_succ": succ_key,
        },
    )


def _validate_root_snapshot(snapshot: Dict[str, Any], cert_domain: str, label: str) -> Tuple[str, Optional[List[str]]]:
    _require(isinstance(snapshot, dict), "SCHEMA_MISMATCH", f"{label} must be object")
    _require(snapshot.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", f"{label}.hash_alg must be sha256")

    root_hash = snapshot.get("root_hash")
    _require(_is_hex64(root_hash), "SCHEMA_MISMATCH", f"{label}.root_hash must be 64-hex")

    keys = snapshot.get("keys")
    if keys is None:
        return root_hash, None

    _require(isinstance(keys, list), "SCHEMA_MISMATCH", f"{label}.keys must be array when present")
    _require(all(isinstance(k, str) and bool(k) for k in keys),
             "SCHEMA_MISMATCH", f"{label}.keys entries must be non-empty strings")
    _assert_strictly_increasing_strings(keys, f"{label}.keys")

    declared_keys_hash = snapshot.get("keys_hash")
    _require(_is_hex64(declared_keys_hash), "SCHEMA_MISMATCH",
             f"{label}.keys_hash must be 64-hex when keys is present")

    computed_keys_hash = keys_hash(cert_domain, keys)
    _require(
        declared_keys_hash == computed_keys_hash,
        "HASH_MISMATCH",
        f"{label}.keys_hash does not match computed hash over keys",
        {"declared_keys_hash": declared_keys_hash, "computed_keys_hash": computed_keys_hash},
    )

    return root_hash, keys


def validate_store_semantics_cert(store_sem: Dict[str, Any]) -> Dict[str, Any]:
    _require(store_sem.get("schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store semantics schema_id")
    domains = store_sem.get("hash_domains")
    _require(isinstance(domains, dict), "SCHEMA_MISMATCH", "store semantics hash_domains must be object")
    for key in ("record", "keyed_leaf", "merkle_node", "cert"):
        _require(isinstance(domains.get(key), str) and bool(domains.get(key)),
                 "SCHEMA_MISMATCH", f"store hash_domains.{key} must be non-empty string")

    _enforce_manifest(store_sem, "store_semantics")

    return {
        "leaf_domain": domains["keyed_leaf"],
        "node_domain": domains["merkle_node"],
        "cert_domain": domains["cert"],
    }


def validate_view_semantics_cert(view_sem: Dict[str, Any]) -> Dict[str, Any]:
    _require(view_sem.get("schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view semantics schema_id")
    _require(view_sem.get("version") == 1, "SCHEMA_MISMATCH", "bad view semantics version")
    _require(view_sem.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in view semantics cert")
    _require(view_sem.get("view_kind") == "POSTING_LIST", "SCHEMA_MISMATCH", "view_kind must be POSTING_LIST")

    projection = view_sem.get("projection")
    _require(isinstance(projection, dict), "SCHEMA_MISMATCH", "projection must be object")
    _require(isinstance(projection.get("name"), str) and bool(projection.get("name")),
             "SCHEMA_MISMATCH", "projection.name must be non-empty string")
    _require(projection.get("mode") in ("SCALE_PRESERVING", "OBSERVER_PROJECTION"),
             "SCHEMA_MISMATCH", "projection.mode must be SCALE_PRESERVING or OBSERVER_PROJECTION")

    params = projection.get("params", None)
    _require(isinstance(params, dict) or params is None,
             "SCHEMA_MISMATCH", "projection.params must be object or null")
    params_hash_declared = projection.get("params_canonical_json_sha256")
    _require(_is_hex64(params_hash_declared),
             "SCHEMA_MISMATCH", "projection.params_canonical_json_sha256 must be 64-hex")
    params_hash_computed = sha256_canonical(params)
    _require(params_hash_declared == params_hash_computed,
             "HASH_MISMATCH",
             "projection.params_canonical_json_sha256 mismatch",
             {"declared": params_hash_declared, "computed": params_hash_computed})

    domains = view_sem.get("hash_domains")
    _require(isinstance(domains, dict), "SCHEMA_MISMATCH", "hash_domains must be object")
    for key in ("posting", "view_leaf", "merkle_node", "cert"):
        _require(isinstance(domains.get(key), str) and bool(domains.get(key)),
                 "SCHEMA_MISMATCH", f"hash_domains.{key} must be non-empty string")
    values = [domains["posting"], domains["view_leaf"], domains["merkle_node"], domains["cert"]]
    _require(len(set(values)) == len(values), "DOMAIN_SEP_VIOLATION", "view hash_domains must be distinct")

    merkle = view_sem.get("merkle")
    _require(isinstance(merkle, dict), "SCHEMA_MISMATCH", "merkle must be object")
    _require(merkle.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "merkle.hash_alg must be sha256")
    _require(merkle.get("pair_order") == "LEFT_RIGHT_CONCAT",
             "SCHEMA_MISMATCH", "merkle.pair_order must be LEFT_RIGHT_CONCAT")
    _require(merkle.get("path_sides") == "EXPLICIT_LR",
             "SCHEMA_MISMATCH", "merkle.path_sides must be EXPLICIT_LR")
    _require(merkle.get("odd_leaf_padding") == "DUPLICATE_LAST",
             "SCHEMA_MISMATCH", "merkle.odd_leaf_padding must be DUPLICATE_LAST")

    fail_types = view_sem.get("fail_types")
    _require(isinstance(fail_types, list) and len(fail_types) > 0,
             "SCHEMA_MISMATCH", "fail_types must be non-empty list")
    _require(all(isinstance(ft, str) and ft for ft in fail_types),
             "SCHEMA_MISMATCH", "fail_types entries must be non-empty strings")

    invariants = view_sem.get("invariants")
    _require(isinstance(invariants, list) and len(invariants) >= 4,
             "SCHEMA_MISMATCH", "invariants must be list with at least 4 entries")
    _require(all(isinstance(inv, str) and inv for inv in invariants),
             "SCHEMA_MISMATCH", "invariants entries must be non-empty strings")

    _enforce_manifest(view_sem, "view_semantics")

    return {
        "fail_types": set(fail_types),
        "posting_domain": domains["posting"],
        "view_leaf_domain": domains["view_leaf"],
        "node_domain": domains["merkle_node"],
        "cert_domain": domains["cert"],
    }


def _validate_store_proof_entry(
    *,
    base_key: str,
    entry: Dict[str, Any],
    store_root_hash: str,
    store_leaf_domain: str,
    store_node_domain: str,
    label: str,
) -> None:
    _require(isinstance(entry, dict), "SCHEMA_MISMATCH", f"{label} must be object")
    record_hash_hex = entry.get("record_hash")
    proof = entry.get("proof")

    _require(_is_hex64(record_hash_hex), "SCHEMA_MISMATCH", f"{label}.record_hash must be 64-hex")
    _require(isinstance(proof, dict), "SCHEMA_MISMATCH", f"{label}.proof must be object")
    _require(proof.get("proof_type") == "INCLUSION", "SCHEMA_MISMATCH", f"{label}.proof.proof_type must be INCLUSION")
    _require(proof.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", f"{label}.proof.hash_alg must be sha256")

    computed_leaf = store_leaf_hash(store_leaf_domain, base_key, record_hash_hex)
    declared_leaf = proof.get("leaf_hash")
    _require(
        declared_leaf == computed_leaf,
        "HASH_MISMATCH",
        f"{label} leaf hash mismatch for base_key={base_key}",
        {"declared_leaf_hash": declared_leaf, "computed_leaf_hash": computed_leaf},
    )

    declared_root = proof.get("root_hash")
    _require(declared_root == store_root_hash,
             "UNVERIFIABLE_PROOF", f"{label}.proof.root_hash must match store_root_snapshot.root_hash")

    verify_inclusion_proof(
        leaf_hash_hex=computed_leaf,
        root_hash_hex=store_root_hash,
        path=proof.get("path", []),
        node_domain=store_node_domain,
    )


def validate_witness_pack(pack: Dict[str, Any], store_cfg: Dict[str, Any], view_cfg: Dict[str, Any]) -> None:
    _require(pack.get("schema_id") == "QA_DATASTORE_VIEW_WITNESS_PACK.v1",
             "SCHEMA_MISMATCH", "bad witness pack schema_id")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in witness pack")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in witness pack")

    store_root_hash, _ = _validate_root_snapshot(
        pack.get("store_root_snapshot"), store_cfg["cert_domain"], "store_root_snapshot"
    )
    view_root_hash, view_keys = _validate_root_snapshot(
        pack.get("view_root_snapshot"), view_cfg["cert_domain"], "view_root_snapshot"
    )

    tests = pack.get("tests")
    _require(isinstance(tests, list) and len(tests) > 0,
             "SCHEMA_MISMATCH", "tests must be non-empty list")

    _enforce_manifest(pack, "view_witness_pack")

    for i, test in enumerate(tests):
        _require(isinstance(test, dict), "SCHEMA_MISMATCH", f"tests[{i}] must be object")
        _require(test.get("op") == "VIEW_GET", "SCHEMA_MISMATCH", f"tests[{i}].op must be VIEW_GET")

        view_key = test.get("view_key")
        expected_posting = test.get("expected_posting")
        view_proof = test.get("view_proof")
        store_proofs = test.get("store_proofs_by_base_key")

        _require(isinstance(view_key, str) and bool(view_key), "SCHEMA_MISMATCH",
                 f"tests[{i}].view_key must be non-empty string")
        _require(isinstance(view_proof, dict), "SCHEMA_MISMATCH", f"tests[{i}].view_proof must be object")
        _require(isinstance(store_proofs, dict), "SCHEMA_MISMATCH",
                 f"tests[{i}].store_proofs_by_base_key must be object")
        _require(view_proof.get("hash_alg") == "sha256", "SCHEMA_MISMATCH",
                 f"tests[{i}].view_proof.hash_alg must be sha256")

        if expected_posting is None:
            _require(view_proof.get("proof_type") == "NON_INCLUSION_RANGE", "SCHEMA_MISMATCH",
                     f"tests[{i}] expected_posting null requires NON_INCLUSION_RANGE proof")
            range_obj = view_proof.get("range")
            _require(isinstance(range_obj, dict), "SCHEMA_MISMATCH",
                     f"tests[{i}].view_proof.range must be object")
            query_key = range_obj.get("query_key")
            _require(query_key == view_key, "SCHEMA_MISMATCH",
                     f"tests[{i}].view_proof.range.query_key must equal tests[{i}].view_key")

            pred_key, succ_key = verify_non_inclusion_range_proof(
                query_key=query_key,
                predecessor=range_obj.get("predecessor"),
                successor=range_obj.get("successor"),
                root_hash_hex=view_root_hash,
                node_domain=view_cfg["node_domain"],
            )
            if view_keys is not None:
                verify_non_inclusion_adjacency(
                    keys=view_keys,
                    query_key=query_key,
                    pred_key=pred_key,
                    succ_key=succ_key,
                )
            _require(len(store_proofs) == 0, "SCHEMA_MISMATCH",
                     f"tests[{i}].store_proofs_by_base_key must be empty for non-inclusion tests")
            continue

        _require(isinstance(expected_posting, list), "SCHEMA_MISMATCH",
                 f"tests[{i}].expected_posting must be array or null")
        _require(view_proof.get("proof_type") == "INCLUSION", "SCHEMA_MISMATCH",
                 f"tests[{i}] expected_posting array requires INCLUSION proof")

        ph = posting_hash(view_cfg["posting_domain"], expected_posting)
        leaf = view_leaf_hash(view_cfg["view_leaf_domain"], view_key, ph)

        declared_leaf = view_proof.get("leaf_hash")
        _require(
            declared_leaf == leaf,
            "HASH_MISMATCH",
            f"tests[{i}] declared view leaf hash mismatch",
            {"declared_leaf_hash": declared_leaf, "computed_leaf_hash": leaf},
        )

        declared_root = view_proof.get("root_hash")
        _require(declared_root == view_root_hash,
                 "UNVERIFIABLE_PROOF", f"tests[{i}].view_proof.root_hash must match view_root_snapshot.root_hash")

        verify_inclusion_proof(
            leaf_hash_hex=leaf,
            root_hash_hex=view_root_hash,
            path=view_proof.get("path", []),
            node_domain=view_cfg["node_domain"],
        )

        expected_set = set(expected_posting)
        actual_set = set(store_proofs.keys())
        _require(expected_set == actual_set, "SCHEMA_MISMATCH",
                 f"tests[{i}] store_proofs_by_base_key keys must match expected_posting")

        for base_key in expected_posting:
            _validate_store_proof_entry(
                base_key=base_key,
                entry=store_proofs[base_key],
                store_root_hash=store_root_hash,
                store_leaf_domain=store_cfg["leaf_domain"],
                store_node_domain=store_cfg["node_domain"],
                label=f"tests[{i}].store_proofs_by_base_key[{base_key}]",
            )


def validate_counterexamples_pack(pack: Dict[str, Any], store_cfg: Dict[str, Any], view_cfg: Dict[str, Any]) -> None:
    _require(pack.get("schema_id") == "QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1",
             "SCHEMA_MISMATCH", "bad counterexamples pack schema_id")
    _require(pack.get("view_semantics_schema_id") == "QA_DATASTORE_VIEW_CERT.v1",
             "SCHEMA_MISMATCH", "bad view_semantics_schema_id in counterexamples pack")
    _require(pack.get("store_semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad store_semantics_schema_id in counterexamples pack")

    _validate_root_snapshot(pack.get("store_root_snapshot"), store_cfg["cert_domain"], "store_root_snapshot")
    _validate_root_snapshot(pack.get("view_root_snapshot"), view_cfg["cert_domain"], "view_root_snapshot")

    cases = pack.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0,
             "SCHEMA_MISMATCH", "cases must be non-empty list")

    _enforce_manifest(pack, "view_counterexamples_pack")

    allowed_fail_types = view_cfg["fail_types"]

    for i, case in enumerate(cases):
        _require(isinstance(case, dict), "SCHEMA_MISMATCH", f"cases[{i}] must be object")

        tamper_mode = case.get("tamper_mode")
        _require(isinstance(tamper_mode, str) and bool(tamper_mode),
                 "SCHEMA_MISMATCH", f"cases[{i}].tamper_mode must be non-empty string")

        expected_fail_type = case.get("expected_fail_type")
        _require(expected_fail_type in allowed_fail_types,
                 "SCHEMA_MISMATCH",
                 f"cases[{i}].expected_fail_type not present in view semantics fail_types: {expected_fail_type}")

        synthetic = {
            "schema_id": "QA_DATASTORE_VIEW_WITNESS_PACK.v1",
            "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
            "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
            "store_root_snapshot": pack.get("store_root_snapshot"),
            "view_root_snapshot": pack.get("view_root_snapshot"),
            "tests": [{
                "test_id": case.get("case_id", f"case_{i}"),
                "op": case.get("op", "VIEW_GET"),
                "view_key": case.get("view_key"),
                "expected_posting": case.get("expected_posting"),
                "view_proof": case.get("view_proof"),
                "store_proofs_by_base_key": case.get("store_proofs_by_base_key", {}),
            }],
            "manifest": {
                "hash_alg": "sha256",
                "canonical_json_sha256": HEX64_ZERO,
            },
        }
        synthetic["manifest"]["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(synthetic))

        try:
            validate_witness_pack(synthetic, store_cfg, view_cfg)
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


def validate_all(
    *,
    store_semantics_path: str,
    view_semantics_path: str,
    witness_path: str,
    counterexamples_path: str,
) -> None:
    store_sem = _load_json(store_semantics_path)
    store_cfg = validate_store_semantics_cert(store_sem)

    view_sem = _load_json(view_semantics_path)
    view_cfg = validate_view_semantics_cert(view_sem)

    witness = _load_json(witness_path)
    validate_witness_pack(witness, store_cfg, view_cfg)

    counterexamples = _load_json(counterexamples_path)
    validate_counterexamples_pack(counterexamples, store_cfg, view_cfg)


def _demo_paths() -> Dict[str, str]:
    base = __file__.rsplit("/", 1)[0]
    return {
        "store_semantics": f"{base}/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json",
        "view_semantics": f"{base}/certs/QA_DATASTORE_VIEW_CERT.v1.json",
        "witness": f"{base}/certs/witness/QA_DATASTORE_VIEW_WITNESS_PACK.v1.json",
        "counterexamples": f"{base}/certs/counterexamples/QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA datastore view semantics/witness/counterexample packs.")
    parser.add_argument("--store-semantics", default="", help="Path to QA_DATASTORE_SEMANTICS cert JSON")
    parser.add_argument("--view-semantics", default="", help="Path to QA_DATASTORE_VIEW_CERT cert JSON")
    parser.add_argument("--witness", default="", help="Path to view witness pack JSON")
    parser.add_argument("--counterexamples", default="", help="Path to view counterexamples pack JSON")
    parser.add_argument("--demo", action="store_true", help="Validate built-in demo certs under certs/")
    args = parser.parse_args()

    if args.demo:
        paths = _demo_paths()
        store_semantics_path = paths["store_semantics"]
        view_semantics_path = paths["view_semantics"]
        witness_path = paths["witness"]
        counterexamples_path = paths["counterexamples"]
    else:
        if not (args.store_semantics and args.view_semantics and args.witness and args.counterexamples):
            parser.error("Provide --store-semantics --view-semantics --witness --counterexamples, or use --demo.")
        store_semantics_path = args.store_semantics
        view_semantics_path = args.view_semantics
        witness_path = args.witness
        counterexamples_path = args.counterexamples

    try:
        validate_all(
            store_semantics_path=store_semantics_path,
            view_semantics_path=view_semantics_path,
            witness_path=witness_path,
            counterexamples_path=counterexamples_path,
        )
    except ValidationFail as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"FAIL: unexpected error: {e}")
        return 1

    print("OK: datastore view semantics + witness + counterexamples validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
