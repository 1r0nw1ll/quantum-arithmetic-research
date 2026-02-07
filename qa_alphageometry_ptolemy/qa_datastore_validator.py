#!/usr/bin/env python3
"""
qa_datastore_validator.py

Deterministic validator for QA datastore semantics + witness + counterexample packs.

Implements:
1. Canonical JSON checks
2. Domain-separated hashing
3. Merkle inclusion and non-inclusion (range) proof verification
4. fail_type taxonomy enforcement for counterexamples
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical


# --- Constants ---

HEX64_ZERO = "0" * 64


# --- Failure Type ---

@dataclass
class ValidationFail(Exception):
    fail_type: str
    msg: str
    invariant_diff: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.invariant_diff:
            return f"{self.fail_type}: {self.msg} | invariant_diff={self.invariant_diff}"
        return f"{self.fail_type}: {self.msg}"


# --- Helpers ---

def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require(condition: bool, fail_type: str, msg: str,
             invariant_diff: Optional[Dict[str, Any]] = None) -> None:
    if not condition:
        raise ValidationFail(fail_type, msg, invariant_diff)


def canonical_json_dumps(obj: Any) -> str:
    """Compact deterministic JSON (single source from qa_cert_core)."""
    return canonical_json_compact(obj)


def _canonical_roundtrip_equal(obj: Any) -> bool:
    try:
        return json.loads(canonical_json_dumps(obj)) == obj
    except Exception:
        return False


def _assert_strictly_increasing_keys(keys: List[str], label: str) -> None:
    for i in range(1, len(keys)):
        _require(
            keys[i - 1] < keys[i],
            "SCHEMA_MISMATCH",
            f"{label} keys must be strictly increasing at index {i}",
            {"left": keys[i - 1], "right": keys[i]},
        )


def _keys_hash(cert_domain: str, keys: List[str]) -> str:
    payload = canonical_json_dumps(keys).encode("utf-8")
    return ds_sha256(cert_domain, payload)


def _manifest_hashable_copy(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce copy used for self-hash verification:
    manifest.canonical_json_sha256 is zeroed before hashing.
    """
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
        {
            "expected": claimed,
            "actual": recomputed,
        },
    )


# --- Domain-Separated Hashing ---

def ds_sha256(domain: str, payload: bytes) -> str:
    _require(isinstance(domain, str) and bool(domain), "DOMAIN_SEP_VIOLATION", "hash domain must be non-empty")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def record_hash(record_domain: str, record_obj: Any) -> str:
    _require(isinstance(record_obj, dict), "SCHEMA_MISMATCH", "record must be an object")
    _require(_canonical_roundtrip_equal(record_obj), "NON_CANONICAL_JSON", "record is not canonical-json roundtrippable")
    payload = canonical_json_dumps(record_obj).encode("utf-8")
    return ds_sha256(record_domain, payload)


def keyed_leaf_hash(leaf_domain: str, key: str, rec_hash_hex: str) -> str:
    _require(isinstance(key, str) and bool(key), "SCHEMA_MISMATCH", "key must be non-empty string")
    _require(_is_hex64(rec_hash_hex), "SCHEMA_MISMATCH", "record hash must be 64-hex")
    payload = key.encode("utf-8") + b"\x00" + rec_hash_hex.encode("ascii")
    return ds_sha256(leaf_domain, payload)


def merkle_parent_hash(node_domain: str, left_hex: str, right_hex: str) -> str:
    _require(_is_hex64(left_hex), "SCHEMA_MISMATCH", "left child hash must be 64-hex")
    _require(_is_hex64(right_hex), "SCHEMA_MISMATCH", "right child hash must be 64-hex")
    payload = left_hex.encode("ascii") + b"\x00" + right_hex.encode("ascii")
    return ds_sha256(node_domain, payload)


# --- Merkle Verification ---

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


def _verify_bound(
    *,
    bound: Dict[str, Any],
    bound_name: str,
    root_hash_hex: str,
    node_domain: str,
) -> str:
    leaf_key = bound.get("leaf_key")
    leaf_hash = bound.get("leaf_hash")
    proof = bound.get("proof")

    _require(isinstance(leaf_key, str), "SCHEMA_MISMATCH", f"{bound_name}.leaf_key must be string")
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
) -> tuple[Optional[str], Optional[str]]:
    _require(isinstance(query_key, str) and bool(query_key),
             "SCHEMA_MISMATCH", "range.query_key must be non-empty string")
    _require(predecessor is not None or successor is not None,
             "SCHEMA_MISMATCH", "range proof must provide predecessor and/or successor")

    pred_key: Optional[str] = None
    succ_key: Optional[str] = None

    if predecessor is not None:
        _require(isinstance(predecessor, dict), "SCHEMA_MISMATCH", "predecessor must be object or null")
        pred_key = _verify_bound(
            bound=predecessor,
            bound_name="predecessor",
            root_hash_hex=root_hash_hex,
            node_domain=node_domain,
        )
        _require(pred_key < query_key, "UNVERIFIABLE_PROOF", "query_key must be > predecessor.leaf_key")

    if successor is not None:
        _require(isinstance(successor, dict), "SCHEMA_MISMATCH", "successor must be object or null")
        succ_key = _verify_bound(
            bound=successor,
            bound_name="successor",
            root_hash_hex=root_hash_hex,
            node_domain=node_domain,
        )
        _require(query_key < succ_key, "UNVERIFIABLE_PROOF", "query_key must be < successor.leaf_key")

    if pred_key is not None and succ_key is not None:
        _require(pred_key < succ_key, "UNVERIFIABLE_PROOF", "predecessor.leaf_key must be < successor.leaf_key")
    return pred_key, succ_key


def _verify_non_inclusion_adjacency(
    *,
    snapshot_keys: List[str],
    query_key: str,
    pred_key: Optional[str],
    succ_key: Optional[str],
) -> None:
    _require(query_key not in snapshot_keys, "UNVERIFIABLE_PROOF", "query_key appears in snapshot key list")
    idx = bisect_left(snapshot_keys, query_key)
    expected_pred = snapshot_keys[idx - 1] if idx > 0 else None
    expected_succ = snapshot_keys[idx] if idx < len(snapshot_keys) else None
    _require(
        pred_key == expected_pred and succ_key == expected_succ,
        "UNVERIFIABLE_PROOF",
        "non-inclusion bounds are not adjacent neighbors in root_snapshot.keys",
        {
            "expected_pred": expected_pred,
            "actual_pred": pred_key,
            "expected_succ": expected_succ,
            "actual_succ": succ_key,
        },
    )


# --- Semantics Validation ---

def validate_semantics_cert(semantics: Dict[str, Any]) -> Dict[str, Any]:
    _require(semantics.get("schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad semantics schema_id")
    _require(semantics.get("version") == 1, "SCHEMA_MISMATCH", "bad semantics version")

    domains = semantics.get("hash_domains")
    _require(isinstance(domains, dict), "SCHEMA_MISMATCH", "hash_domains must be object")

    for key in ("record", "keyed_leaf", "merkle_node", "cert"):
        _require(isinstance(domains.get(key), str) and bool(domains.get(key)),
                 "SCHEMA_MISMATCH", f"hash_domains.{key} must be non-empty string")

    domain_values = [domains["record"], domains["keyed_leaf"], domains["merkle_node"], domains["cert"]]
    _require(len(set(domain_values)) == len(domain_values),
             "DOMAIN_SEP_VIOLATION", "hash_domains must be distinct")

    merkle = semantics.get("merkle")
    _require(isinstance(merkle, dict), "SCHEMA_MISMATCH", "merkle must be object")
    _require(merkle.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "merkle.hash_alg must be sha256")
    _require(merkle.get("pair_order") == "LEFT_RIGHT_CONCAT",
             "SCHEMA_MISMATCH", "merkle.pair_order must be LEFT_RIGHT_CONCAT")
    _require(merkle.get("path_sides") == "EXPLICIT_LR",
             "SCHEMA_MISMATCH", "merkle.path_sides must be EXPLICIT_LR")

    fail_types = semantics.get("fail_types")
    _require(isinstance(fail_types, list) and len(fail_types) > 0,
             "SCHEMA_MISMATCH", "fail_types must be non-empty list")
    _require(all(isinstance(ft, str) and ft for ft in fail_types),
             "SCHEMA_MISMATCH", "fail_types entries must be non-empty strings")

    _enforce_manifest(semantics, "semantics")

    return {
        "fail_types": set(fail_types),
        "record_domain": domains["record"],
        "leaf_domain": domains["keyed_leaf"],
        "node_domain": domains["merkle_node"],
        "cert_domain": domains["cert"],
    }


# --- Witness Validation ---

def validate_witness_pack(pack: Dict[str, Any], semantics_cfg: Dict[str, Any]) -> None:
    _require(pack.get("schema_id") == "QA_DATASTORE_WITNESS_PACK.v1",
             "SCHEMA_MISMATCH", "bad witness pack schema_id")
    _require(pack.get("semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad semantics_schema_id in witness pack")

    root_snapshot = pack.get("root_snapshot")
    _require(isinstance(root_snapshot, dict), "SCHEMA_MISMATCH", "root_snapshot must be object")
    root_hash = root_snapshot.get("root_hash")
    _require(_is_hex64(root_hash), "SCHEMA_MISMATCH", "root_snapshot.root_hash must be 64-hex")
    snapshot_keys = root_snapshot.get("keys")
    if snapshot_keys is not None:
        _require(isinstance(snapshot_keys, list), "SCHEMA_MISMATCH", "root_snapshot.keys must be array when present")
        _require(all(isinstance(k, str) for k in snapshot_keys), "SCHEMA_MISMATCH",
                 "root_snapshot.keys entries must be strings")
        _assert_strictly_increasing_keys(snapshot_keys, "root_snapshot")
        declared_keys_hash = root_snapshot.get("keys_hash")
        _require(_is_hex64(declared_keys_hash), "SCHEMA_MISMATCH",
                 "root_snapshot.keys_hash must be 64-hex when root_snapshot.keys is present")
        computed_keys_hash = _keys_hash(semantics_cfg["cert_domain"], snapshot_keys)
        _require(
            declared_keys_hash == computed_keys_hash,
            "HASH_MISMATCH",
            "root_snapshot.keys_hash does not match computed hash over keys",
            {"declared_keys_hash": declared_keys_hash, "computed_keys_hash": computed_keys_hash},
        )

    tests = pack.get("tests")
    _require(isinstance(tests, list) and len(tests) > 0,
             "SCHEMA_MISMATCH", "tests must be non-empty list")

    _enforce_manifest(pack, "witness_pack")

    for i, test in enumerate(tests):
        _require(isinstance(test, dict), "SCHEMA_MISMATCH", f"tests[{i}] must be object")
        _require(test.get("op") == "GET", "SCHEMA_MISMATCH", f"tests[{i}].op must be GET")

        key = test.get("key")
        expected = test.get("expected")
        proof = test.get("proof")

        _require(isinstance(key, str) and bool(key), "SCHEMA_MISMATCH", f"tests[{i}].key must be non-empty")
        _require(isinstance(proof, dict), "SCHEMA_MISMATCH", f"tests[{i}].proof must be object")
        _require(proof.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", f"tests[{i}].proof.hash_alg must be sha256")

        if expected is None:
            _require(proof.get("proof_type") == "NON_INCLUSION_RANGE",
                     "SCHEMA_MISMATCH", f"tests[{i}] expected null requires NON_INCLUSION_RANGE proof")
            range_obj = proof.get("range")
            _require(isinstance(range_obj, dict), "SCHEMA_MISMATCH", f"tests[{i}].proof.range must be object")
            query_key = range_obj.get("query_key")
            _require(query_key == key,
                     "SCHEMA_MISMATCH",
                     f"tests[{i}].proof.range.query_key must equal tests[{i}].key")

            pred_key, succ_key = verify_non_inclusion_range_proof(
                query_key=query_key,
                predecessor=range_obj.get("predecessor"),
                successor=range_obj.get("successor"),
                root_hash_hex=root_hash,
                node_domain=semantics_cfg["node_domain"],
            )
            if snapshot_keys is not None:
                _verify_non_inclusion_adjacency(
                    snapshot_keys=snapshot_keys,
                    query_key=query_key,
                    pred_key=pred_key,
                    succ_key=succ_key,
                )
            continue

        _require(isinstance(expected, dict), "SCHEMA_MISMATCH", f"tests[{i}].expected must be object or null")
        _require(proof.get("proof_type") == "INCLUSION",
                 "SCHEMA_MISMATCH", f"tests[{i}] expected object requires INCLUSION proof")

        rh = record_hash(semantics_cfg["record_domain"], expected)
        leaf = keyed_leaf_hash(semantics_cfg["leaf_domain"], key, rh)

        declared_leaf = proof.get("leaf_hash")
        _require(
            declared_leaf == leaf,
            "HASH_MISMATCH",
            f"tests[{i}] declared leaf_hash does not match computed keyed leaf",
            {"declared_leaf_hash": declared_leaf, "computed_leaf_hash": leaf},
        )

        declared_root = proof.get("root_hash")
        _require(declared_root == root_hash,
                 "UNVERIFIABLE_PROOF", f"tests[{i}] proof.root_hash must match pack root_snapshot.root_hash")

        verify_inclusion_proof(
            leaf_hash_hex=leaf,
            root_hash_hex=root_hash,
            path=proof.get("path", []),
            node_domain=semantics_cfg["node_domain"],
        )


# --- Counterexample Validation ---

def validate_counterexamples_pack(pack: Dict[str, Any], semantics_cfg: Dict[str, Any]) -> None:
    _require(pack.get("schema_id") == "QA_DATASTORE_COUNTEREXAMPLES_PACK.v1",
             "SCHEMA_MISMATCH", "bad counterexamples pack schema_id")
    _require(pack.get("semantics_schema_id") == "QA_DATASTORE_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad semantics_schema_id in counterexamples pack")

    root_snapshot = pack.get("root_snapshot")
    _require(isinstance(root_snapshot, dict), "SCHEMA_MISMATCH", "root_snapshot must be object")
    root_hash = root_snapshot.get("root_hash")
    _require(_is_hex64(root_hash), "SCHEMA_MISMATCH", "root_snapshot.root_hash must be 64-hex")
    snapshot_keys = root_snapshot.get("keys")
    if snapshot_keys is not None:
        _require(isinstance(snapshot_keys, list), "SCHEMA_MISMATCH", "root_snapshot.keys must be array when present")
        _require(all(isinstance(k, str) for k in snapshot_keys), "SCHEMA_MISMATCH",
                 "root_snapshot.keys entries must be strings")
        _assert_strictly_increasing_keys(snapshot_keys, "root_snapshot")
        declared_keys_hash = root_snapshot.get("keys_hash")
        _require(_is_hex64(declared_keys_hash), "SCHEMA_MISMATCH",
                 "root_snapshot.keys_hash must be 64-hex when root_snapshot.keys is present")
        computed_keys_hash = _keys_hash(semantics_cfg["cert_domain"], snapshot_keys)
        _require(
            declared_keys_hash == computed_keys_hash,
            "HASH_MISMATCH",
            "root_snapshot.keys_hash does not match computed hash over keys",
            {"declared_keys_hash": declared_keys_hash, "computed_keys_hash": computed_keys_hash},
        )

    cases = pack.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0,
             "SCHEMA_MISMATCH", "cases must be non-empty list")

    _enforce_manifest(pack, "counterexamples_pack")

    allowed_fail_types = semantics_cfg["fail_types"]

    for i, case in enumerate(cases):
        _require(isinstance(case, dict), "SCHEMA_MISMATCH", f"cases[{i}] must be object")
        tamper_mode = case.get("tamper_mode")
        _require(isinstance(tamper_mode, str) and bool(tamper_mode),
                 "SCHEMA_MISMATCH", f"cases[{i}].tamper_mode must be non-empty string")
        expected_fail_type = case.get("expected_fail_type")
        _require(expected_fail_type in allowed_fail_types,
                 "SCHEMA_MISMATCH",
                 f"cases[{i}].expected_fail_type not present in semantics.fail_types: {expected_fail_type}")

        synthetic_witness = {
            "schema_id": "QA_DATASTORE_WITNESS_PACK.v1",
            "semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
            "root_snapshot": {
                "root_hash": root_hash,
                "hash_alg": "sha256",
                "snapshot_id": pack.get("root_snapshot", {}).get("snapshot_id", "")
            },
            "tests": [{
                "test_id": case.get("case_id", f"case_{i}"),
                "op": case.get("op", "GET"),
                "key": case.get("key"),
                "expected": case.get("expected"),
                "proof": case.get("proof"),
            }],
            "manifest": {
                "hash_alg": "sha256",
                "canonical_json_sha256": HEX64_ZERO,
            },
        }
        synthetic_witness["manifest"]["canonical_json_sha256"] = sha256_canonical(
            _manifest_hashable_copy(synthetic_witness)
        )

        try:
            validate_witness_pack(synthetic_witness, semantics_cfg)
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


# --- Entrypoints ---

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_all(semantics_path: str, witness_path: str, counterexamples_path: str) -> None:
    semantics = _load_json(semantics_path)
    semantics_cfg = validate_semantics_cert(semantics)

    witness = _load_json(witness_path)
    validate_witness_pack(witness, semantics_cfg)

    counterexamples = _load_json(counterexamples_path)
    validate_counterexamples_pack(counterexamples, semantics_cfg)


def _demo_paths() -> Dict[str, str]:
    base = __file__.rsplit("/", 1)[0]
    return {
        "semantics": f"{base}/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json",
        "witness": f"{base}/certs/witness/QA_DATASTORE_WITNESS_PACK.v1.json",
        "counterexamples": f"{base}/certs/counterexamples/QA_DATASTORE_COUNTEREXAMPLES_PACK.v1.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA datastore semantics/witness/counterexample packs.")
    parser.add_argument("--semantics", default="", help="Path to semantics cert JSON")
    parser.add_argument("--witness", default="", help="Path to witness pack JSON")
    parser.add_argument("--counterexamples", default="", help="Path to counterexamples pack JSON")
    parser.add_argument("--demo", action="store_true", help="Validate built-in demo certs under certs/")
    args = parser.parse_args()

    if args.demo:
        paths = _demo_paths()
        semantics_path = paths["semantics"]
        witness_path = paths["witness"]
        counterexamples_path = paths["counterexamples"]
    else:
        if not (args.semantics and args.witness and args.counterexamples):
            parser.error("Provide --semantics --witness --counterexamples, or use --demo.")
        semantics_path = args.semantics
        witness_path = args.witness
        counterexamples_path = args.counterexamples

    try:
        validate_all(semantics_path, witness_path, counterexamples_path)
    except ValidationFail as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"FAIL: unexpected error: {e}")
        return 1

    print("OK: datastore semantics + witness + counterexamples validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
