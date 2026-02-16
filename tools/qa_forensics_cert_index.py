"""
QA-Certified Forensics Index (Datastore + View)

Purpose
  - Turn the fast-but-informal local search index (`_forensics/qa_local_search.sqlite`)
    into a QA-style, auditable snapshot:
      * Store:  base_key (repo path) -> record (file metadata + forensics signals)
      * View:   view_key -> posting list of base_keys (small, high-signal lists)

Why this exists (as a "cold" data analyst tool)
  - You can rebuild the index, get Merkle roots, and validate witness packs.
  - Local agents can read the emitted JSON and use it as a compact "where are the
    results / what should I open next" structure.

Entry points
  - Build: python tools/qa_forensics_cert_index.py build
  - View keys: python tools/qa_forensics_cert_index.py view-keys
  - View get: python tools/qa_forensics_cert_index.py view-get "view:forensics/hotspot_top_k" --limit 50
  - Store get: python tools/qa_forensics_cert_index.py store-get qa_alphageometry_ptolemy/qa_meta_validator.py
  - Validate (store): python -m qa_alphageometry_ptolemy.qa_datastore_validator --semantics ... --witness ... --counterexamples ...
  - Validate (view):  python -m qa_alphageometry_ptolemy.qa_datastore_view_validator --store-semantics ... --view-semantics ... --witness ... --counterexamples ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qa_alphageometry_ptolemy.qa_cert_core import canonical_json_compact, sha256_canonical
from qa_alphageometry_ptolemy.qa_datastore_build_snapshot import (
    HEX64_ZERO,
    build_merkle_with_paths,
    build_non_inclusion_range_proof,
    build_snapshot,
    build_witness_pack,
    ds_sha256,
    keys_hash,
    record_hash,
)


UTC = dt.timezone.utc


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_compact(obj) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json_compact(row))
            handle.write("\n")


def _manifest_hashable_copy(doc: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(doc))
    manifest = clone.get("manifest")
    if isinstance(manifest, dict) and "canonical_json_sha256" in manifest:
        manifest["canonical_json_sha256"] = HEX64_ZERO
    return clone


def _set_manifest(doc: dict[str, Any]) -> None:
    manifest = doc.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("doc.manifest must be object")
    manifest["hash_alg"] = "sha256"
    manifest["canonical_json_sha256"] = sha256_canonical(_manifest_hashable_copy(doc))


def _connect_sqlite(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _load_build_info(con: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = con.execute("SELECT k, v FROM build_info").fetchall()
    except sqlite3.Error:
        return {}
    return {str(r["k"]): str(r["v"]) for r in rows}


def _load_script_artifacts(con: sqlite3.Connection) -> dict[str, list[str]]:
    artifacts: dict[str, list[str]] = {}
    try:
        rows = con.execute(
            "SELECT script, artifact_path FROM script_artifacts ORDER BY script, artifact_path"
        ).fetchall()
    except sqlite3.Error:
        return artifacts
    for r in rows:
        script = str(r["script"])
        artifact_path = str(r["artifact_path"])
        artifacts.setdefault(script, []).append(artifact_path)
    return artifacts


def _load_file_meta(con: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        return list(
            con.execute(
                """
                SELECT
                  path, ext, size_bytes, mtime_utc, tracked,
                  title, category,
                  hotspot_score, hotspot_evidence, hotspot_claims,
                  chat_mentions, artifact_count
                FROM file_meta
                ORDER BY path
                """
            ).fetchall()
        )
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed reading file_meta from sqlite index: {e}") from e


def _load_store_semantics() -> dict[str, Any]:
    path = Path("qa_alphageometry_ptolemy/certs/QA_DATASTORE_SEMANTICS_CERT.v1.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_view_semantics(*, top_k: int, view_keys: list[str]) -> dict[str, Any]:
    # Reuse the established domains; projection describes *this* view’s intent.
    params = {
        "source": "_forensics/qa_local_search.sqlite:file_meta",
        "definition": "Small, high-signal posting lists for repo forensics review.",
        "top_k": int(top_k),
        "view_keys": list(view_keys),
    }
    doc: dict[str, Any] = {
        "schema_id": "QA_DATASTORE_VIEW_CERT.v1",
        "version": 1,
        "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "view_kind": "POSTING_LIST",
        "projection": {
            "name": "repo_forensics_posting_view",
            "mode": "OBSERVER_PROJECTION",
            "params": params,
            "params_canonical_json_sha256": sha256_canonical(params),
        },
        "hash_domains": {
            "posting": "QA/VIEW_POSTING/v1",
            "view_leaf": "QA/VIEW_LEAF/v1",
            "merkle_node": "QA/VIEW_MERKLE_NODE/v1",
            "cert": "QA/VIEW_CERT/v1",
        },
        "merkle": {
            "hash_alg": "sha256",
            "pair_order": "LEFT_RIGHT_CONCAT",
            "path_sides": "EXPLICIT_LR",
            "odd_leaf_padding": "DUPLICATE_LAST",
        },
        "fail_types": [
            "KEY_NOT_FOUND",
            "HASH_MISMATCH",
            "SCHEMA_MISMATCH",
            "DOMAIN_SEP_VIOLATION",
            "NON_CANONICAL_JSON",
            "UNVERIFIABLE_PROOF",
            "FORK_DETECTED",
            "MIGRATION_NON_INVERTIBLE",
            "SCALE_COLLAPSE",
        ],
        "invariants": [
            "Manifest self-hash uses zeroed canonical_json_sha256 field",
            "posting_hash = ds_sha256(dom_posting, canonical_json(posting_list))",
            "view_leaf_hash = ds_sha256(dom_view_leaf, view_key || 0x00 || posting_hash)",
            "Merkle internal hash uses ds_sha256(dom_merkle_node, left || 0x00 || right)",
            "When view_root_snapshot.keys is present, enforce keys_hash binding + non-inclusion adjacency",
        ],
        "manifest": {"hash_alg": "sha256", "canonical_json_sha256": HEX64_ZERO},
    }
    _set_manifest(doc)
    return doc


def _posting_hash(posting_domain: str, posting: list[str]) -> str:
    payload = canonical_json_compact(posting).encode("utf-8")
    return ds_sha256(posting_domain, payload)


def _view_leaf_hash(view_leaf_domain: str, view_key: str, posting_hash_hex: str) -> str:
    payload = view_key.encode("utf-8") + b"\x00" + posting_hash_hex.encode("ascii")
    return ds_sha256(view_leaf_domain, payload)


def _build_view_snapshot(
    *,
    snapshot_id: str,
    view_semantics: dict[str, Any],
    postings_by_key: dict[str, list[str]],
) -> dict[str, Any]:
    domains = view_semantics["hash_domains"]
    posting_domain = str(domains["posting"])
    view_leaf_domain = str(domains["view_leaf"])
    node_domain = str(domains["merkle_node"])
    cert_domain = str(domains["cert"])

    keys = sorted(postings_by_key.keys())
    leaf_hashes: list[str] = []
    proofs_by_key: dict[str, Any] = {}
    leaf_hashes_by_key: dict[str, str] = {}
    posting_hashes_by_key: dict[str, str] = {}

    posting_hashes: list[str] = []
    for k in keys:
        posting = postings_by_key[k]
        ph = _posting_hash(posting_domain, posting)
        posting_hashes.append(ph)
        posting_hashes_by_key[k] = ph
        leaf = _view_leaf_hash(view_leaf_domain, k, ph)
        leaf_hashes.append(leaf)
        leaf_hashes_by_key[k] = leaf

    root_hash, paths = build_merkle_with_paths(node_domain, leaf_hashes)
    for i, k in enumerate(keys):
        proofs_by_key[k] = {
            "proof_type": "INCLUSION",
            "hash_alg": "sha256",
            "leaf_hash": leaf_hashes[i],
            "root_hash": root_hash,
            "path": paths[i],
        }

    snapshot: dict[str, Any] = {
        "schema_id": "QA_DATASTORE_VIEW_SNAPSHOT.v1",
        "snapshot_id": snapshot_id,
        "hash_alg": "sha256",
        "root_hash": root_hash,
        "keys": keys,
        "keys_hash": keys_hash(cert_domain, keys),
        "posting_hashes_by_key": posting_hashes_by_key,
        "leaf_hashes_by_key": leaf_hashes_by_key,
        "proofs_by_key": proofs_by_key,
        "manifest": {"hash_alg": "sha256", "canonical_json_sha256": HEX64_ZERO},
    }
    _set_manifest(snapshot)
    return snapshot


def _pick_top_k(rows: list[dict[str, Any]], key: str, *, top_k: int) -> list[dict[str, Any]]:
    def score(r: dict[str, Any]) -> tuple[int, str]:
        try:
            s = int(r.get(key) or 0)
        except Exception:
            s = 0
        return (s, str(r.get("path") or ""))

    # Descending by metric, tie-break by path.
    ranked = sorted(rows, key=score, reverse=True)
    out: list[dict[str, Any]] = []
    for r in ranked:
        try:
            if int(r.get(key) or 0) <= 0:
                continue
        except Exception:
            continue
        out.append(r)
        if len(out) >= top_k:
            break
    return out


def _posting_from_candidates(
    *,
    present_keys: set[str],
    candidates: list[str],
) -> list[str]:
    return sorted({c for c in candidates if c in present_keys})


def build_index(*, db_path: Path, out_dir: Path, top_k: int) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Missing sqlite index: {db_path}. Run `python tools/qa_local_search.py build` first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    store_semantics = _load_store_semantics()
    store_domains = store_semantics["hash_domains"]
    record_domain = str(store_domains["record"])
    leaf_domain = str(store_domains["keyed_leaf"])
    node_domain = str(store_domains["merkle_node"])
    cert_domain = str(store_domains["cert"])

    con = _connect_sqlite(db_path)
    try:
        build_info = _load_build_info(con)
        artifacts_by_script = _load_script_artifacts(con)
        file_rows = _load_file_meta(con)
    finally:
        con.close()

    # Build store items from sqlite metadata.
    items: list[dict[str, Any]] = []
    record_hash_by_key: dict[str, str] = {}
    records_jsonl: list[dict[str, Any]] = []

    for r in file_rows:
        path = str(r["path"])
        rec: dict[str, Any] = {
            "path": path,
            "ext": str(r["ext"] or ""),
            "size_bytes": int(r["size_bytes"] or 0),
            "mtime_utc": str(r["mtime_utc"] or ""),
            "tracked": int(r["tracked"] or 0),
            "title": str(r["title"] or ""),
            "category": str(r["category"] or ""),
            "hotspot": {
                "score": int(r["hotspot_score"] or 0),
                "evidence": int(r["hotspot_evidence"] or 0),
                "claims": int(r["hotspot_claims"] or 0),
            },
            "chat_mentions": int(r["chat_mentions"] or 0),
            "artifact_count": int(r["artifact_count"] or 0),
            "artifact_paths": artifacts_by_script.get(path, []),
        }
        items.append({"key": path, "record": rec})

    items.sort(key=lambda it: it["key"])
    snapshot_stamp = dt.datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"QA_FORensicsStore.{snapshot_stamp}.v1"

    store_snapshot = build_snapshot(
        items,
        snapshot_id=snapshot_id,
        record_domain=record_domain,
        leaf_domain=leaf_domain,
        node_domain=node_domain,
        cert_domain=cert_domain,
    )

    # Record hashes (for view store_proofs_by_base_key entries).
    for it in items:
        key = str(it["key"])
        rh = record_hash(record_domain, it["record"])
        record_hash_by_key[key] = rh
        records_jsonl.append({"key": key, "record_hash": rh, "record": it["record"]})

    # Store witness pack (small sample + a few non-inclusion tests).
    flat_rows: list[dict[str, Any]] = []
    for it in items:
        rec = it["record"]
        flat_rows.append(
            {
                "path": it["key"],
                "hotspot_score": rec["hotspot"]["score"],
                "chat_mentions": rec["chat_mentions"],
                "artifact_count": len(rec.get("artifact_paths") or []),
            }
        )

    sample_keys: set[str] = set()
    if items:
        sample_keys.add(str(items[0]["key"]))
        sample_keys.add(str(items[-1]["key"]))

    for metric in ("hotspot_score", "chat_mentions", "artifact_count"):
        for row in _pick_top_k(flat_rows, metric, top_k=min(10, max(1, top_k))):
            sample_keys.add(str(row["path"]))

    sample_items = [it for it in items if it["key"] in sample_keys]
    sample_items.sort(key=lambda it: it["key"])
    missing_keys = [
        "__qa_forensics_nonexistent__",
        "zzzzzzzzzz_nonexistent",
        "Documents/__nope__.md",
    ]
    store_witness = build_witness_pack(
        store_snapshot,
        sample_items,
        snapshot_id=snapshot_id,
        missing_keys=missing_keys,
    )

    # Minimal counterexample pack: tamper leaf_hash to force HASH_MISMATCH.
    first_inclusion = next((t for t in store_witness["tests"] if t.get("expected") is not None), None)
    if first_inclusion is None:
        raise RuntimeError("No inclusion tests generated for store witness pack.")
    bad_proof = json.loads(json.dumps(first_inclusion["proof"]))
    bad_proof["leaf_hash"] = HEX64_ZERO
    store_counterexamples: dict[str, Any] = {
        "schema_id": "QA_DATASTORE_COUNTEREXAMPLES_PACK.v1",
        "semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "root_snapshot": store_witness["root_snapshot"],
        "cases": [
            {
                "case_id": "CE_HASH_MISMATCH_001",
                "tamper_mode": "HASH_MISMATCH",
                "expected_fail_type": "HASH_MISMATCH",
                "op": "GET",
                "key": first_inclusion["key"],
                "expected": first_inclusion["expected"],
                "proof": bad_proof,
            }
        ],
        "manifest": {"hash_alg": "sha256", "canonical_json_sha256": HEX64_ZERO},
    }
    _set_manifest(store_counterexamples)

    # View: small, high-signal posting lists.
    hotspot_top = _pick_top_k(flat_rows, "hotspot_score", top_k=top_k)
    chat_top = _pick_top_k(flat_rows, "chat_mentions", top_k=top_k)
    artifact_top = _pick_top_k(flat_rows, "artifact_count", top_k=top_k)

    present_store_keys = set(record_hash_by_key.keys())

    ontology_candidates = [
        "QA_AXIOMS_BLOCK.md",
        "QA_CONTROL_THEOREMS.md",
        "qa_alphageometry_ptolemy/qa_cert_core.py",
        "qa_alphageometry_ptolemy/qa_certificate.py",
        "qa_alphageometry_ptolemy/qa_datastore_validator.py",
        "qa_alphageometry_ptolemy/qa_datastore_view_validator.py",
        "qa_alphageometry_ptolemy/qa_meta_validator.py",
        "qa_alphageometry_ptolemy/QA_DECISION_CERTIFICATE_SPINE.md",
        "qa_alphageometry_ptolemy/QA_MAP_CANONICAL.md",
        "qa_alphageometry_ptolemy/QACertificateSpine.tla",
        "qa_alphageometry_ptolemy/QARM_v02_Failures.tla",
    ]

    execution_candidates = [
        "qa_build_pipeline.py",
        "qa_graph_query.py",
        "qa_knowledge_graph.py",
        "qa_entity_extractor.py",
        "qa_entity_encoder.py",
        "qa_chunk_ingest.py",
        "qa_repo_ingest.py",
        "tools/project_forensics.py",
        "tools/qa_local_search.py",
        "tools/qa_forensics_cert_index.py",
        "tools/generate_results_registry.py",
        "qa_alphageometry_ptolemy/qa_verify.py",
        "qa_alphageometry_ptolemy/qa_meta_validator.py",
    ]

    results_candidates = [
        "Documents/RESULTS_REGISTRY.md",
        "Documents/PROJECT_FORENSICS_CONSOLIDATION.md",
        "FINAL_PUBLICATION_REPORT.md",
        "REAL_FINAL_RESULTS.md",
        "REAL_RESULTS_ANALYSIS.md",
        "QA_UNIFIED_FRAMEWORK_SUMMARY.md",
    ]

    meta_candidates = [
        "AGENTS.md",
        "README.md",
        "CONTRIBUTING.md",
        "PAPER_SUBMISSION_README.md",
        "START_HERE_2026-01-10.md",
        "QA_PIPELINE_README.md",
        "QA_PIPELINE_AXIOM_DRIFT.md",
        "HANDOFF.md",
        "MULTI_AI_COLLABORATION_GUIDE.md",
    ]

    postings_by_key: dict[str, list[str]] = {
        "view:forensics/hotspot_top_k": sorted({str(r["path"]) for r in hotspot_top}),
        "view:forensics/chat_top_k": sorted({str(r["path"]) for r in chat_top}),
        "view:forensics/artifact_producers_top_k": sorted({str(r["path"]) for r in artifact_top}),
        "view:cartography/ontology_spine": _posting_from_candidates(
            present_keys=present_store_keys,
            candidates=ontology_candidates,
        ),
        "view:cartography/execution_spine": _posting_from_candidates(
            present_keys=present_store_keys,
            candidates=execution_candidates,
        ),
        "view:cartography/results_spine": _posting_from_candidates(
            present_keys=present_store_keys,
            candidates=results_candidates,
        ),
        "view:cartography/meta_spine": _posting_from_candidates(
            present_keys=present_store_keys,
            candidates=meta_candidates,
        ),
    }
    view_keys = sorted(postings_by_key.keys())

    view_semantics = _make_view_semantics(top_k=top_k, view_keys=view_keys)
    view_snapshot_id = f"QA_FORensicsView.{snapshot_stamp}.v1"
    view_snapshot = _build_view_snapshot(
        snapshot_id=view_snapshot_id,
        view_semantics=view_semantics,
        postings_by_key=postings_by_key,
    )

    # View witness pack: include all (small) view keys + one non-inclusion test.
    view_tests: list[dict[str, Any]] = []
    for vk in view_keys:
        posting = postings_by_key[vk]
        store_proofs = {
            base_key: {
                "record_hash": record_hash_by_key[base_key],
                "proof": store_snapshot["proofs_by_key"][base_key],
            }
            for base_key in posting
        }
        view_tests.append(
            {
                "test_id": f"VIEW_GET::{vk}",
                "op": "VIEW_GET",
                "view_key": vk,
                "expected_posting": posting,
                "view_proof": view_snapshot["proofs_by_key"][vk],
                "store_proofs_by_base_key": store_proofs,
            }
        )

    missing_view_key = "view:forensics/__missing__"
    view_tests.append(
        {
            "test_id": f"VIEW_GET::{missing_view_key}",
            "op": "VIEW_GET",
            "view_key": missing_view_key,
            "expected_posting": None,
            "view_proof": build_non_inclusion_range_proof(view_snapshot, missing_view_key),
            "store_proofs_by_base_key": {},
        }
    )

    view_witness: dict[str, Any] = {
        "schema_id": "QA_DATASTORE_VIEW_WITNESS_PACK.v1",
        "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
        "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "store_root_snapshot": {
            "snapshot_id": snapshot_id,
            "hash_alg": "sha256",
            "root_hash": store_snapshot["root_hash"],
            "keys": store_snapshot["keys"],
            "keys_hash": store_snapshot["keys_hash"],
        },
        "view_root_snapshot": {
            "snapshot_id": view_snapshot_id,
            "hash_alg": "sha256",
            "root_hash": view_snapshot["root_hash"],
            "keys": view_snapshot["keys"],
            "keys_hash": view_snapshot["keys_hash"],
        },
        "tests": view_tests,
        "manifest": {"hash_alg": "sha256", "canonical_json_sha256": HEX64_ZERO},
    }
    _set_manifest(view_witness)

    # Minimal view counterexample pack: tamper view leaf hash.
    first_view_inclusion = next((t for t in view_tests if t.get("expected_posting") is not None), None)
    if first_view_inclusion is None:
        raise RuntimeError("No inclusion tests generated for view witness pack.")
    bad_view_proof = json.loads(json.dumps(first_view_inclusion["view_proof"]))
    bad_view_proof["leaf_hash"] = HEX64_ZERO
    view_counterexamples: dict[str, Any] = {
        "schema_id": "QA_DATASTORE_VIEW_COUNTEREXAMPLES_PACK.v1",
        "view_semantics_schema_id": "QA_DATASTORE_VIEW_CERT.v1",
        "store_semantics_schema_id": "QA_DATASTORE_SEMANTICS_CERT.v1",
        "store_root_snapshot": view_witness["store_root_snapshot"],
        "view_root_snapshot": view_witness["view_root_snapshot"],
        "cases": [
            {
                "case_id": "VIEW_CE_HASH_MISMATCH_001",
                "tamper_mode": "VIEW_LEAF_HASH_TAMPER",
                "expected_fail_type": "HASH_MISMATCH",
                "op": "VIEW_GET",
                "view_key": first_view_inclusion["view_key"],
                "expected_posting": first_view_inclusion["expected_posting"],
                "view_proof": bad_view_proof,
                "store_proofs_by_base_key": first_view_inclusion["store_proofs_by_base_key"],
            }
        ],
        "manifest": {"hash_alg": "sha256", "canonical_json_sha256": HEX64_ZERO},
    }
    _set_manifest(view_counterexamples)

    # Emit artifacts
    _write_json(out_dir / "store_semantics.json", store_semantics)
    _write_json(out_dir / "store_snapshot.json", store_snapshot)
    _write_json(out_dir / "store_witness_pack.json", store_witness)
    _write_json(out_dir / "store_counterexamples_pack.json", store_counterexamples)
    _write_jsonl(out_dir / "store_records.jsonl", records_jsonl)

    _write_json(out_dir / "view_semantics.json", view_semantics)
    _write_json(out_dir / "view_snapshot.json", view_snapshot)
    _write_json(out_dir / "view_postings.json", postings_by_key)
    _write_json(out_dir / "view_witness_pack.json", view_witness)
    _write_json(out_dir / "view_counterexamples_pack.json", view_counterexamples)

    meta = {
        "generated_utc": dt.datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "build_info": build_info,
        "store_snapshot_id": snapshot_id,
        "store_root_hash": store_snapshot["root_hash"],
        "store_keys": len(store_snapshot.get("keys") or []),
        "view_snapshot_id": view_snapshot_id,
        "view_root_hash": view_snapshot["root_hash"],
        "view_keys": len(view_snapshot.get("keys") or []),
        "view_top_k": int(top_k),
    }
    _write_json(out_dir / "META.json", meta)


def _find_latest_index_dir(root: Path) -> Path:
    base = root / "_forensics"
    if not base.exists():
        raise FileNotFoundError("Missing _forensics/.")
    dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("qa_cert_index_")]
    if not dirs:
        raise FileNotFoundError("No qa_cert_index_* directories found under _forensics/. Run `build` first.")
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_store_records(index_dir: Path) -> Iterator[dict[str, Any]]:
    path = index_dir / "store_records.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def view_keys(*, index_dir: Path, json_out: bool) -> int:
    postings = _read_json(index_dir / "view_postings.json")
    keys = sorted(postings.keys()) if isinstance(postings, dict) else []
    if json_out:
        print(json.dumps({"index_dir": str(index_dir), "view_keys": keys}, indent=2))
    else:
        for k in keys:
            print(k)
    return 0


def view_get(*, index_dir: Path, view_key: str, limit: int, json_out: bool) -> int:
    postings = _read_json(index_dir / "view_postings.json")
    if not isinstance(postings, dict) or view_key not in postings:
        print(f"[qa_forensics_cert_index] missing view_key: {view_key}", file=sys.stderr)
        return 2
    posting = postings[view_key]
    if not isinstance(posting, list):
        print(f"[qa_forensics_cert_index] invalid posting list for: {view_key}", file=sys.stderr)
        return 2

    shown = posting[: max(0, int(limit))] if limit > 0 else posting
    if json_out:
        print(
            json.dumps(
                {
                    "index_dir": str(index_dir),
                    "view_key": view_key,
                    "posting_size": len(posting),
                    "posting": shown,
                },
                indent=2,
            )
        )
    else:
        print(f"{view_key}  (n={len(posting)})")
        for p in shown:
            print(p)
        if limit > 0 and len(posting) > limit:
            print(f"... (+{len(posting) - limit} more)")
    return 0


def store_get(*, index_dir: Path, key: str, json_out: bool) -> int:
    for row in _iter_store_records(index_dir):
        if str(row.get("key")) == key:
            if json_out:
                print(json.dumps(row, indent=2))
            else:
                rec = row.get("record") or {}
                print(f"Key: {key}")
                print(f"Record hash: {row.get('record_hash')}")
                if isinstance(rec, dict):
                    print(f"Title: {rec.get('title')}")
                    print(f"Category: {rec.get('category')}")
                    print(f"Ext: {rec.get('ext')}  Tracked: {rec.get('tracked')}")
                    print(f"Size: {rec.get('size_bytes')}  Mtime: {rec.get('mtime_utc')}")
                    hs = rec.get("hotspot") or {}
                    if isinstance(hs, dict):
                        print(
                            f"Hotspot: score={hs.get('score')} evidence={hs.get('evidence')} claims={hs.get('claims')}"
                        )
                    print(f"Chat mentions: {rec.get('chat_mentions')}  Artifact count: {rec.get('artifact_count')}")
                    artifacts = rec.get("artifact_paths") or []
                    if isinstance(artifacts, list) and artifacts:
                        print("Artifacts:")
                        for a in artifacts[:25]:
                            print(f"  - {a}")
                        if len(artifacts) > 25:
                            print(f"  - (+{len(artifacts) - 25} more)")
            return 0

    print(f"[qa_forensics_cert_index] missing store key: {key}", file=sys.stderr)
    return 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build QA-certified datastore/view from local forensics sqlite index.")
    parser.add_argument("--db", default="_forensics/qa_local_search.sqlite", help="Input sqlite index path.")
    parser.add_argument(
        "--out",
        default="",
        help="Index directory. For `build`, this is the output directory; for query commands, this is the input directory (default: latest under _forensics/).",
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-K for view posting lists.")
    parser.add_argument("--json", dest="json_out", action="store_true", help="JSON output (for agents).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="Build QA-certified store + view artifacts")
    sub.add_parser("view-keys", help="List view keys from an existing index directory")
    vg = sub.add_parser("view-get", help="Get posting list for a view key")
    vg.add_argument("view_key", help="View key")
    vg.add_argument("--limit", type=int, default=200, help="Max posting entries to print (0=all)")
    sg = sub.add_parser("store-get", help="Get one store record by base key (repo path)")
    sg.add_argument("key", help="Base key (repo-relative path)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = (root / db_path).resolve()

    index_dir = Path(args.out) if args.out else None
    if index_dir is None:
        if args.cmd == "build":
            stamp = dt.datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            index_dir = Path("_forensics") / f"qa_cert_index_{stamp}"
        else:
            index_dir = _find_latest_index_dir(root)
    if not index_dir.is_absolute():
        index_dir = (root / index_dir).resolve()

    if args.cmd == "build":
        build_index(db_path=db_path, out_dir=index_dir, top_k=int(args.top_k))
        print(f"[qa_forensics_cert_index] wrote: {index_dir}")
        return 0

    if args.cmd == "view-keys":
        return view_keys(index_dir=index_dir, json_out=bool(args.json_out))

    if args.cmd == "view-get":
        return view_get(
            index_dir=index_dir,
            view_key=str(args.view_key),
            limit=int(args.limit),
            json_out=bool(args.json_out),
        )

    if args.cmd == "store-get":
        return store_get(index_dir=index_dir, key=str(args.key), json_out=bool(args.json_out))

    raise RuntimeError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main(list(sys.argv[1:])))
