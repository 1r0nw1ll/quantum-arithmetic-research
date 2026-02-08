#!/usr/bin/env python3
"""
qa_ingest_validator.py

Strict validator for QA ingestion semantics + witness + counterexample packs.

Implements:
1) canonical JSON + manifest self-hash enforcement
2) domain-separated hashing and deterministic chunk/doc merkle roots
3) extraction/normalization/chunking invariant checks from source containers
4) fail-type strictness for counterexamples
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

try:
    from .qa_cert_core import canonical_json_compact, sha256_canonical
    from .qa_datastore_view_validator import ValidationFail, _is_hex64, _manifest_hashable_copy, _enforce_manifest, _require
except ImportError:
    from qa_cert_core import canonical_json_compact, sha256_canonical
    from qa_datastore_view_validator import ValidationFail, _is_hex64, _manifest_hashable_copy, _enforce_manifest, _require


HEX64_ZERO = "0" * 64


def canonical_json_dumps(obj: Any) -> str:
    return canonical_json_compact(obj)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_source_path(source_ref: str) -> Path:
    """
    Resolve source_ref deterministically across common invocation roots.

    Priority:
    1) path as provided (absolute or cwd-relative)
    2) repo-root-relative (parent of qa_alphageometry_ptolemy/)
    """
    p = Path(source_ref)
    if p.exists():
        return p

    repo_root_candidate = Path(__file__).resolve().parent.parent / source_ref
    if repo_root_candidate.exists():
        return repo_root_candidate

    return p


def ds_sha256(domain: str, payload: bytes) -> str:
    _require(isinstance(domain, str) and bool(domain), "DOMAIN_SEP_VIOLATION", "hash domain must be non-empty")
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + payload).hexdigest()


def merkle_parent_hash(node_domain: str, left_hex: str, right_hex: str) -> str:
    _require(_is_hex64(left_hex), "SCHEMA_MISMATCH", "left child hash must be 64-hex")
    _require(_is_hex64(right_hex), "SCHEMA_MISMATCH", "right child hash must be 64-hex")
    payload = left_hex.encode("ascii") + b"\x00" + right_hex.encode("ascii")
    return ds_sha256(node_domain, payload)


def merkle_root(node_domain: str, leaves: List[str]) -> str:
    _require(isinstance(leaves, list) and len(leaves) > 0, "SCHEMA_MISMATCH", "merkle leaves must be non-empty list")
    _require(all(_is_hex64(h) for h in leaves), "SCHEMA_MISMATCH", "merkle leaves must be 64-hex")

    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = level + [level[-1]]
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            nxt.append(merkle_parent_hash(node_domain, level[i], level[i + 1]))
        level = nxt
    return level[0]


def _assert_strictly_increasing(values: List[str], label: str) -> None:
    for i in range(1, len(values)):
        _require(
            values[i - 1] < values[i],
            "SCHEMA_MISMATCH",
            f"{label} must be strictly increasing at index {i}",
            {"left": values[i - 1], "right": values[i]},
        )


def _extract_text_odt(path: str) -> Tuple[str, str]:
    """
    Returns (extracted_text, container_sig_sha256).
    container_sig_sha256 is sha256(content.xml bytes).
    """
    with zipfile.ZipFile(path, "r") as zf:
        try:
            content_xml = zf.read("content.xml")
        except KeyError:
            raise ValidationFail("SCHEMA_MISMATCH", "odt container missing content.xml")

    container_sig = _sha256_bytes(content_xml)

    try:
        root = ET.fromstring(content_xml)
    except ET.ParseError:
        raise ValidationFail("SCHEMA_MISMATCH", "content.xml parse error")

    extracted = "\n".join(t for t in root.itertext())
    return extracted, container_sig


def extract_text_from_container(path: str, container_type: str, allow_ocr: bool) -> Tuple[str, str]:
    _require(isinstance(container_type, str) and bool(container_type), "SCHEMA_MISMATCH", "container_type must be non-empty string")
    _require(os.path.exists(path), "INGEST_DOC_MISSING", f"source file not found: {path}")

    ctype = container_type.lower()
    if ctype == "odt":
        return _extract_text_odt(path)

    if ctype in ("txt", "md", "markdown"):
        raw = _read_bytes(path)
        sig = _sha256_bytes(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValidationFail("SCHEMA_MISMATCH", "text container is not UTF-8")
        return text, sig

    if ctype == "pdf" and allow_ocr:
        # v1 fail-closed: OCR path intentionally unsupported in validator runtime.
        raise ValidationFail("SCHEMA_MISMATCH", "pdf OCR extraction not implemented in v1 validator")

    raise ValidationFail("SCHEMA_MISMATCH", f"unsupported container_type: {container_type}")


def normalize_text(text: str, cfg: Dict[str, Any]) -> str:
    _require(isinstance(text, str), "SCHEMA_MISMATCH", "text must be string")

    out = text

    if cfg.get("line_endings") == "LF":
        out = out.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.get("unicode_nfkc"):
        out = unicodedata.normalize("NFKC", out)

    if cfg.get("remove_control_chars"):
        out = "".join(ch for ch in out if (ord(ch) >= 32 or ch in "\n\t"))

    if cfg.get("lowercase"):
        out = out.lower()

    if cfg.get("whitespace_collapse"):
        out = re.sub(r"\s+", " ", out).strip()

    return out


def chunk_text(text: str, chunk_cfg: Dict[str, Any]) -> List[str]:
    strategy = chunk_cfg.get("strategy")
    _require(strategy in ("fixed_chars", "paragraph"), "SCHEMA_MISMATCH", "chunking.strategy must be fixed_chars or paragraph")

    if strategy == "paragraph":
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return parts

    max_chars = chunk_cfg.get("max_chars")
    overlap = chunk_cfg.get("overlap_chars")
    min_chars = chunk_cfg.get("min_chars")

    _require(isinstance(max_chars, int) and max_chars >= 1, "SCHEMA_MISMATCH", "chunking.max_chars must be int >=1")
    _require(isinstance(overlap, int) and overlap >= 0, "SCHEMA_MISMATCH", "chunking.overlap_chars must be int >=0")
    _require(isinstance(min_chars, int) and min_chars >= 1, "SCHEMA_MISMATCH", "chunking.min_chars must be int >=1")
    _require(overlap < max_chars, "SCHEMA_MISMATCH", "chunking.overlap_chars must be < max_chars")

    chunks: List[str] = []
    n = len(text)
    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        if len(chunk) >= min_chars or end == n:
            chunks.append(chunk)
        if end == n:
            break
        nxt = end - overlap
        if nxt <= start:
            nxt = start + 1
        start = nxt

    return chunks


def _doc_leaf_hash(doc_root_domain: str, doc_id: str, normalized_text_sha256: str, chunk_root_hash: str) -> str:
    _require(_is_hex64(normalized_text_sha256), "SCHEMA_MISMATCH", "normalized_text_sha256 must be 64-hex")
    _require(_is_hex64(chunk_root_hash), "SCHEMA_MISMATCH", "chunk_root_hash must be 64-hex")
    payload = (
        doc_id.encode("utf-8")
        + b"\x00"
        + normalized_text_sha256.encode("ascii")
        + b"\x00"
        + chunk_root_hash.encode("ascii")
    )
    return ds_sha256(doc_root_domain, payload)


def _manifest_hash_for_doc(doc: Dict[str, Any]) -> str:
    return sha256_canonical(_manifest_hashable_copy(doc))


def rehash_manifest_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    _require(isinstance(doc, dict), "SCHEMA_MISMATCH", "rehash target must be JSON object")
    man = doc.get("manifest")
    _require(isinstance(man, dict), "SCHEMA_MISMATCH", "rehash target must contain manifest object")
    _require(man.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "manifest.hash_alg must be sha256")

    man["canonical_json_sha256"] = HEX64_ZERO
    man["canonical_json_sha256"] = _manifest_hash_for_doc(doc)

    with open(path, "w", encoding="utf-8") as f:
        f.write(canonical_json_dumps(doc))
        f.write("\n")

    return man["canonical_json_sha256"]


def validate_ingest_semantics_cert(sem: Dict[str, Any]) -> Dict[str, Any]:
    _require(sem.get("schema_id") == "QA_INGEST_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad ingest semantics schema_id")

    _require(sem.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "hash_alg must be sha256")
    _require(sem.get("canonical_json") is True, "SCHEMA_MISMATCH", "canonical_json must be true")

    cert_domain = sem.get("cert_domain")
    doc_root_domain = sem.get("doc_root_domain")
    chunk_domain = sem.get("chunk_domain")
    for name, value in (("cert_domain", cert_domain), ("doc_root_domain", doc_root_domain), ("chunk_domain", chunk_domain)):
        _require(isinstance(value, str) and bool(value), "SCHEMA_MISMATCH", f"{name} must be non-empty string")

    _require(len({cert_domain, doc_root_domain, chunk_domain}) == 3,
             "DOMAIN_SEP_VIOLATION", "cert_domain/doc_root_domain/chunk_domain must be distinct")

    contract = sem.get("ingest_contract")
    _require(isinstance(contract, dict), "SCHEMA_MISMATCH", "ingest_contract must be object")

    accepted = contract.get("accepted_container_types")
    _require(isinstance(accepted, list) and len(accepted) > 0,
             "SCHEMA_MISMATCH", "accepted_container_types must be non-empty list")
    _require(all(isinstance(x, str) and bool(x) for x in accepted),
             "SCHEMA_MISMATCH", "accepted_container_types entries must be non-empty strings")

    text_extraction = contract.get("text_extraction")
    _require(isinstance(text_extraction, dict), "SCHEMA_MISMATCH", "text_extraction must be object")
    for key in ("preserve_order", "strip_markup", "allow_ocr"):
        _require(isinstance(text_extraction.get(key), bool), "SCHEMA_MISMATCH", f"text_extraction.{key} must be boolean")
    _require(isinstance(text_extraction.get("method"), str) and bool(text_extraction.get("method")),
             "SCHEMA_MISMATCH", "text_extraction.method must be non-empty string")

    normalization = contract.get("normalization")
    _require(isinstance(normalization, dict), "SCHEMA_MISMATCH", "normalization must be object")
    for key in ("unicode_nfkc", "whitespace_collapse", "lowercase", "remove_control_chars"):
        _require(isinstance(normalization.get(key), bool), "SCHEMA_MISMATCH", f"normalization.{key} must be boolean")
    _require(normalization.get("line_endings") == "LF", "SCHEMA_MISMATCH", "normalization.line_endings must be LF")

    chunking = contract.get("chunking")
    _require(isinstance(chunking, dict), "SCHEMA_MISMATCH", "chunking must be object")
    _require(chunking.get("strategy") in ("fixed_chars", "paragraph"),
             "SCHEMA_MISMATCH", "chunking.strategy must be fixed_chars or paragraph")
    for key in ("max_chars", "overlap_chars", "min_chars"):
        _require(isinstance(chunking.get(key), int), "SCHEMA_MISMATCH", f"chunking.{key} must be integer")

    budgets = contract.get("budgets")
    _require(isinstance(budgets, dict), "SCHEMA_MISMATCH", "budgets must be object")
    for key in ("max_docs", "max_total_chars", "max_total_chunks"):
        _require(isinstance(budgets.get(key), int) and budgets.get(key) >= 1,
                 "SCHEMA_MISMATCH", f"budgets.{key} must be integer >=1")

    provenance = contract.get("provenance")
    _require(isinstance(provenance, dict), "SCHEMA_MISMATCH", "provenance must be object")
    for key in ("require_source_path", "require_file_hash", "require_container_sig"):
        _require(isinstance(provenance.get(key), bool), "SCHEMA_MISMATCH", f"provenance.{key} must be boolean")

    fail_types = sem.get("fail_types")
    _require(isinstance(fail_types, list) and len(fail_types) > 0,
             "SCHEMA_MISMATCH", "fail_types must be non-empty list")
    _require(all(isinstance(ft, str) and bool(ft) for ft in fail_types),
             "SCHEMA_MISMATCH", "fail_types entries must be non-empty strings")

    _enforce_manifest(sem, "ingest_semantics")

    return {
        "cert_domain": cert_domain,
        "doc_root_domain": doc_root_domain,
        "chunk_domain": chunk_domain,
        "contract": contract,
        "accepted_container_types": {x.lower() for x in accepted},
        "fail_types": set(fail_types),
    }


def _validate_doc_root_snapshot(snapshot: Dict[str, Any], cert_domain: str) -> Tuple[str, Optional[List[str]]]:
    _require(isinstance(snapshot, dict), "SCHEMA_MISMATCH", "doc_root_snapshot must be object")
    _require(snapshot.get("hash_alg") == "sha256", "SCHEMA_MISMATCH", "doc_root_snapshot.hash_alg must be sha256")
    root_hash = snapshot.get("root_hash")
    _require(_is_hex64(root_hash), "SCHEMA_MISMATCH", "doc_root_snapshot.root_hash must be 64-hex")

    keys = snapshot.get("keys")
    if keys is None:
        return root_hash, None

    _require(isinstance(keys, list), "SCHEMA_MISMATCH", "doc_root_snapshot.keys must be array when present")
    _require(all(isinstance(k, str) and bool(k) for k in keys),
             "SCHEMA_MISMATCH", "doc_root_snapshot.keys entries must be non-empty strings")
    _assert_strictly_increasing(keys, "doc_root_snapshot.keys")

    declared_keys_hash = snapshot.get("keys_hash")
    _require(_is_hex64(declared_keys_hash),
             "SCHEMA_MISMATCH", "doc_root_snapshot.keys_hash must be 64-hex when keys present")
    computed_keys_hash = ds_sha256(cert_domain, canonical_json_dumps(keys).encode("utf-8"))
    _require(
        declared_keys_hash == computed_keys_hash,
        "HASH_MISMATCH",
        "doc_root_snapshot.keys_hash mismatch",
        {"declared_keys_hash": declared_keys_hash, "computed_keys_hash": computed_keys_hash},
    )

    return root_hash, keys


def _validate_witness_pack_obj(pack: Dict[str, Any], sem_cfg: Dict[str, Any], *, enforce_manifest: bool) -> None:
    _require(pack.get("schema_id") == "QA_INGEST_WITNESS_PACK.v1",
             "SCHEMA_MISMATCH", "bad ingest witness schema_id")
    _require(pack.get("ingest_semantics_schema_id") == "QA_INGEST_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad ingest_semantics_schema_id in witness pack")

    if enforce_manifest:
        _enforce_manifest(pack, "ingest_witness_pack")

    doc_root_snapshot = pack.get("doc_root_snapshot")
    _require(isinstance(doc_root_snapshot, dict), "SCHEMA_MISMATCH", "doc_root_snapshot must be object")
    doc_root_hash, snapshot_keys = _validate_doc_root_snapshot(doc_root_snapshot, sem_cfg["cert_domain"])

    docs = pack.get("docs")
    _require(isinstance(docs, list) and len(docs) > 0, "SCHEMA_MISMATCH", "docs must be non-empty list")
    _require(pack.get("claimed_success") is True, "SCHEMA_MISMATCH", "claimed_success must be true")

    budgets = sem_cfg["contract"]["budgets"]
    if len(docs) > budgets["max_docs"]:
        raise ValidationFail(
            "BUDGET_EXCEEDED",
            "document count exceeds max_docs",
            {"max_docs": budgets["max_docs"], "actual_docs": len(docs)},
        )

    doc_ids: List[str] = []
    doc_leaf_hashes: List[str] = []
    total_chars = 0
    total_chunks = 0

    provenance = sem_cfg["contract"]["provenance"]
    text_extraction_cfg = sem_cfg["contract"]["text_extraction"]
    norm_cfg = sem_cfg["contract"]["normalization"]
    chunk_cfg = sem_cfg["contract"]["chunking"]

    seen_doc_ids = set()

    for i, doc in enumerate(docs):
        _require(isinstance(doc, dict), "SCHEMA_MISMATCH", f"docs[{i}] must be object")

        doc_id = doc.get("doc_id")
        source_ref = doc.get("source_ref")
        container_type = doc.get("container_type")

        _require(isinstance(doc_id, str) and bool(doc_id), "SCHEMA_MISMATCH", f"docs[{i}].doc_id must be non-empty string")
        _require(doc_id not in seen_doc_ids, "FORK_DETECTED", f"duplicate doc_id: {doc_id}")
        seen_doc_ids.add(doc_id)
        doc_ids.append(doc_id)

        _require(isinstance(source_ref, str) and bool(source_ref), "SCHEMA_MISMATCH", f"docs[{i}].source_ref must be non-empty string")
        _require(isinstance(container_type, str) and bool(container_type), "SCHEMA_MISMATCH", f"docs[{i}].container_type must be non-empty string")
        _require(container_type.lower() in sem_cfg["accepted_container_types"],
                 "SCHEMA_MISMATCH", f"docs[{i}].container_type not accepted by ingest semantics")

        resolved_source_ref = _resolve_source_path(source_ref)
        if provenance["require_source_path"]:
            _require(
                resolved_source_ref.exists(),
                "INGEST_DOC_MISSING",
                f"docs[{i}] source not found: {source_ref}",
                {"source_ref": source_ref, "resolved_path": str(resolved_source_ref)},
            )

        file_sha_declared = doc.get("file_sha256")
        container_sig_declared = doc.get("container_sig_sha256")
        extracted_declared = doc.get("extracted_text_sha256")
        normalized_declared = doc.get("normalized_text_sha256")
        chunk_root_declared = doc.get("chunk_root_hash")
        chunk_count_declared = doc.get("chunk_count")
        total_chars_declared = doc.get("total_chars")
        chunk_hashes_declared = doc.get("chunk_hashes")

        _require(_is_hex64(file_sha_declared), "SCHEMA_MISMATCH", f"docs[{i}].file_sha256 must be 64-hex")
        _require(_is_hex64(container_sig_declared), "SCHEMA_MISMATCH", f"docs[{i}].container_sig_sha256 must be 64-hex")
        _require(_is_hex64(extracted_declared), "SCHEMA_MISMATCH", f"docs[{i}].extracted_text_sha256 must be 64-hex")
        _require(_is_hex64(normalized_declared), "SCHEMA_MISMATCH", f"docs[{i}].normalized_text_sha256 must be 64-hex")
        _require(_is_hex64(chunk_root_declared), "SCHEMA_MISMATCH", f"docs[{i}].chunk_root_hash must be 64-hex")
        _require(isinstance(chunk_count_declared, int) and chunk_count_declared >= 0,
                 "SCHEMA_MISMATCH", f"docs[{i}].chunk_count must be integer >=0")
        _require(isinstance(total_chars_declared, int) and total_chars_declared >= 0,
                 "SCHEMA_MISMATCH", f"docs[{i}].total_chars must be integer >=0")
        _require(isinstance(chunk_hashes_declared, list), "SCHEMA_MISMATCH", f"docs[{i}].chunk_hashes must be list")
        _require(all(_is_hex64(h) for h in chunk_hashes_declared),
                 "SCHEMA_MISMATCH", f"docs[{i}].chunk_hashes entries must be 64-hex")

        raw_bytes = _read_bytes(str(resolved_source_ref))
        file_sha_actual = _sha256_bytes(raw_bytes)
        if provenance["require_file_hash"] and file_sha_actual != file_sha_declared:
            raise ValidationFail(
                "INGEST_DOC_HASH_MISMATCH",
                f"docs[{i}] file_sha256 mismatch",
                {"declared_file_sha256": file_sha_declared, "computed_file_sha256": file_sha_actual},
            )

        extracted_text, container_sig_actual = extract_text_from_container(
            str(resolved_source_ref),
            container_type,
            allow_ocr=text_extraction_cfg.get("allow_ocr", False),
        )

        if provenance["require_container_sig"] and container_sig_actual != container_sig_declared:
            raise ValidationFail(
                "CONTAINER_SIG_TAMPER",
                f"docs[{i}] container_sig_sha256 mismatch",
                {
                    "declared_container_sig_sha256": container_sig_declared,
                    "computed_container_sig_sha256": container_sig_actual,
                },
            )

        extracted_actual = _sha256_bytes(extracted_text.encode("utf-8"))
        if extracted_actual != extracted_declared:
            raise ValidationFail(
                "EXTRACTION_DRIFT",
                f"docs[{i}] extracted_text_sha256 mismatch",
                {"declared_extracted_text_sha256": extracted_declared, "computed_extracted_text_sha256": extracted_actual},
            )

        normalized_text = normalize_text(extracted_text, norm_cfg)
        normalized_actual = _sha256_bytes(normalized_text.encode("utf-8"))
        if normalized_actual != normalized_declared:
            raise ValidationFail(
                "NORMALIZATION_DRIFT",
                f"docs[{i}] normalized_text_sha256 mismatch",
                {"declared_normalized_text_sha256": normalized_declared, "computed_normalized_text_sha256": normalized_actual},
            )

        chunks = chunk_text(normalized_text, chunk_cfg)
        chunk_hashes_actual = [ds_sha256(sem_cfg["chunk_domain"], c.encode("utf-8")) for c in chunks]

        if chunk_hashes_actual != chunk_hashes_declared:
            raise ValidationFail(
                "CHUNKING_DRIFT",
                f"docs[{i}] chunk_hashes mismatch",
                {
                    "declared_chunk_count": len(chunk_hashes_declared),
                    "computed_chunk_count": len(chunk_hashes_actual),
                },
            )

        chunk_root_actual = merkle_root(sem_cfg["chunk_domain"], chunk_hashes_actual) if chunk_hashes_actual else HEX64_ZERO
        if chunk_root_actual != chunk_root_declared:
            raise ValidationFail(
                "CHUNKING_DRIFT",
                f"docs[{i}] chunk_root_hash mismatch",
                {"declared_chunk_root_hash": chunk_root_declared, "computed_chunk_root_hash": chunk_root_actual},
            )

        if chunk_count_declared != len(chunk_hashes_actual):
            raise ValidationFail(
                "CHUNKING_DRIFT",
                f"docs[{i}] chunk_count mismatch",
                {"declared_chunk_count": chunk_count_declared, "computed_chunk_count": len(chunk_hashes_actual)},
            )

        if total_chars_declared != len(normalized_text):
            raise ValidationFail(
                "CHUNKING_DRIFT",
                f"docs[{i}] total_chars mismatch",
                {"declared_total_chars": total_chars_declared, "computed_total_chars": len(normalized_text)},
            )

        total_chars += total_chars_declared
        total_chunks += chunk_count_declared

        doc_leaf_hashes.append(_doc_leaf_hash(sem_cfg["doc_root_domain"], doc_id, normalized_actual, chunk_root_actual))

    _assert_strictly_increasing(doc_ids, "docs.doc_id")

    if snapshot_keys is not None:
        _require(doc_ids == snapshot_keys,
                 "UNVERIFIABLE_PROOF",
                 "doc_root_snapshot.keys must match witness doc_id ordering",
                 {"snapshot_keys": snapshot_keys, "doc_ids": doc_ids})

    doc_root_actual = merkle_root(sem_cfg["doc_root_domain"], doc_leaf_hashes)
    _require(
        doc_root_actual == doc_root_hash,
        "UNVERIFIABLE_PROOF",
        "doc_root_snapshot.root_hash does not match computed doc merkle root",
        {"declared_root_hash": doc_root_hash, "computed_root_hash": doc_root_actual},
    )

    if total_chars > budgets["max_total_chars"]:
        raise ValidationFail(
            "BUDGET_EXCEEDED",
            "total chars exceed budget",
            {"max_total_chars": budgets["max_total_chars"], "actual_total_chars": total_chars},
        )

    if total_chunks > budgets["max_total_chunks"]:
        raise ValidationFail(
            "BUDGET_EXCEEDED",
            "total chunks exceed budget",
            {"max_total_chunks": budgets["max_total_chunks"], "actual_total_chunks": total_chunks},
        )


def validate_witness_pack(pack: Dict[str, Any], sem_cfg: Dict[str, Any]) -> None:
    _validate_witness_pack_obj(pack, sem_cfg, enforce_manifest=True)


def validate_counterexamples_pack(pack: Dict[str, Any], sem_cfg: Dict[str, Any]) -> None:
    _require(pack.get("schema_id") == "QA_INGEST_COUNTEREXAMPLES_PACK.v1",
             "SCHEMA_MISMATCH", "bad ingest counterexamples schema_id")
    _require(pack.get("ingest_semantics_schema_id") == "QA_INGEST_SEMANTICS_CERT.v1",
             "SCHEMA_MISMATCH", "bad ingest_semantics_schema_id in counterexamples pack")

    _enforce_manifest(pack, "ingest_counterexamples_pack")

    root_snapshot = pack.get("doc_root_snapshot")
    _require(isinstance(root_snapshot, dict), "SCHEMA_MISMATCH", "doc_root_snapshot must be object")

    cases = pack.get("cases")
    _require(isinstance(cases, list) and len(cases) > 0,
             "SCHEMA_MISMATCH", "cases must be non-empty list")

    for i, case in enumerate(cases):
        _require(isinstance(case, dict), "SCHEMA_MISMATCH", f"cases[{i}] must be object")
        tamper_mode = case.get("tamper_mode")
        _require(isinstance(tamper_mode, str) and bool(tamper_mode),
                 "SCHEMA_MISMATCH", f"cases[{i}].tamper_mode must be non-empty string")

        expected_fail_type = case.get("expected_fail_type")
        _require(expected_fail_type in sem_cfg["fail_types"],
                 "SCHEMA_MISMATCH",
                 f"cases[{i}].expected_fail_type not present in semantics.fail_types: {expected_fail_type}")

        synthetic = {
            "schema_id": "QA_INGEST_WITNESS_PACK.v1",
            "ingest_semantics_schema_id": "QA_INGEST_SEMANTICS_CERT.v1",
            "doc_root_snapshot": copy.deepcopy(root_snapshot),
            "docs": case.get("docs"),
            "claimed_success": True,
            "manifest": {
                "hash_alg": "sha256",
                "canonical_json_sha256": HEX64_ZERO,
            },
        }
        synthetic["manifest"]["canonical_json_sha256"] = _manifest_hash_for_doc(synthetic)

        try:
            _validate_witness_pack_obj(synthetic, sem_cfg, enforce_manifest=True)
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


def validate_all(*, semantics_path: str, witness_path: str, counterexamples_path: str) -> None:
    sem = _load_json(semantics_path)
    sem_cfg = validate_ingest_semantics_cert(sem)

    witness = _load_json(witness_path)
    validate_witness_pack(witness, sem_cfg)

    counterexamples = _load_json(counterexamples_path)
    validate_counterexamples_pack(counterexamples, sem_cfg)


def _demo_paths() -> Dict[str, str]:
    base = __file__.rsplit("/", 1)[0]
    return {
        "semantics": f"{base}/certs/QA_INGEST_SEMANTICS_CERT.v1.json",
        "witness": f"{base}/certs/witness/QA_INGEST_WITNESS_PACK.v1.json",
        "counterexamples": f"{base}/certs/counterexamples/QA_INGEST_COUNTEREXAMPLES_PACK.v1.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QA ingestion semantics/witness/counterexample packs.")
    parser.add_argument("--rehash", default="", help="Recompute manifest.canonical_json_sha256 for a JSON file and exit")
    parser.add_argument("--semantics", default="", help="Path to QA_INGEST_SEMANTICS cert JSON")
    parser.add_argument("--witness", default="", help="Path to ingest witness pack JSON")
    parser.add_argument("--counterexamples", default="", help="Path to ingest counterexamples pack JSON")
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
        validate_all(
            semantics_path=semantics_path,
            witness_path=witness_path,
            counterexamples_path=counterexamples_path,
        )
    except ValidationFail as e:
        print(f"FAIL: {e}")
        return 1
    except Exception as e:  # pragma: no cover
        print(f"FAIL: unexpected error: {e}")
        return 1

    print("OK: ingest semantics + witness + counterexamples validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
