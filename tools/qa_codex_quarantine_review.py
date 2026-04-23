#!/usr/bin/env python3
"""Review and clear Claude Python quarantine packets.

This is the Codex-side batch review tool for the Claude PreToolUse gate.
Claude Python writes land in the worktree and their audit packets land in
`llm_qa_wrapper/quarantine/pending/`. Codex reviews those packets at the
cert-submission boundary and approves or rejects the batch here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


QA_COMPLIANCE = {
    "observer": "codex_quarantine_review_gate",
    "state_alphabet": "quarantine packet metadata and review ledger records only",
}

REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUARANTINE_DIR = REPO / "llm_qa_wrapper" / "quarantine"
SUPPORTED_PACKET_SCHEMAS = {
    "QA_CLAUDE_PYTHON_QUARANTINE.v1",
    "QA_CLAUDE_FORMAL_PUBLICATION_QUARANTINE.v1",
}


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _quarantine_root(raw: str | None) -> Path:
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    return Path(os.environ.get("LLM_QA_WRAPPER_QUARANTINE_DIR", DEFAULT_QUARANTINE_DIR)).resolve()


def _pending_packets(root: Path) -> list[Path]:
    pending = root / "pending"
    if not pending.exists():
        return []
    return sorted(p for p in pending.glob("*.json") if p.is_file())


def _load_packet(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: packet JSON must be an object")
    # The same Codex review flow clears Python and formal-publication packets.
    if obj.get("schema_version") not in SUPPORTED_PACKET_SCHEMAS:
        raise ValueError(f"{path}: unexpected schema_version")
    return obj


def _append_review(root: Path, packet_path: Path, packet: dict[str, Any], decision: str, reviewer: str, notes: str) -> dict[str, Any]:
    packet_sha = _sha256_file(packet_path)
    record_payload = {
        "schema_version": "QA_CODEX_QUARANTINE_REVIEW.v1",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviewer": reviewer,
        "decision": decision,
        "packet_path": str(packet_path),
        "packet_sha256": packet_sha,
        "payload_sha256": packet.get("payload_sha256"),
        "target_rel": packet.get("target_rel"),
        "tool_name": packet.get("tool_name"),
        "rollback": packet.get("rollback"),
        "notes": notes,
    }
    record = dict(record_payload)
    record["review_sha256"] = hashlib.sha256(
        b"QA_CODEX_QUARANTINE_REVIEW.v1\x00" + _canonical_json(record_payload)
    ).hexdigest()

    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "codex_reviews.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return record


def _rollback_packet(packet: dict[str, Any]) -> dict[str, Any]:
    target_raw = packet.get("target_path")
    if not isinstance(target_raw, str) or not target_raw.strip():
        return {"status": "advisory_manual_revert_required", "reason": "missing target_path"}

    target = Path(target_raw)
    if not packet.get("original_snapshot_available"):
        return {"status": "advisory_manual_revert_required", "reason": "original snapshot unavailable"}

    if packet.get("original_exists"):
        content_b64 = packet.get("original_content_b64")
        if not isinstance(content_b64, str):
            return {"status": "advisory_manual_revert_required", "reason": "missing original_content_b64"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(content_b64.encode("ascii")))
        return {"status": "restored_original", "target_path": str(target)}

    if target.exists():
        target.unlink()
        return {"status": "removed_new_file", "target_path": str(target)}
    return {"status": "no_op_new_file_absent", "target_path": str(target)}


def _review_packets(root: Path, decision: str, reviewer: str, notes: str, packet_args: list[str], all_pending: bool) -> int:
    if all_pending:
        packets = _pending_packets(root)
    else:
        packets = [Path(p).resolve() for p in packet_args]
    if not packets:
        print("qa_codex_quarantine_review: no pending packets")
        return 0

    dest_dir = root / ("approved" if decision == "approve" else "rejected")
    dest_dir.mkdir(parents=True, exist_ok=True)
    reviewed = []
    for packet_path in packets:
        packet = _load_packet(packet_path)
        if decision == "reject":
            packet["rollback"] = _rollback_packet(packet)
        else:
            packet["rollback"] = {"status": "not_requested"}
        record = _append_review(root, packet_path, packet, decision, reviewer, notes)
        dest = dest_dir / packet_path.name
        packet_path.write_text(
            json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(packet_path), str(dest))
        reviewed.append({
            "packet": str(dest),
            "target_rel": packet.get("target_rel"),
            "decision": decision,
            "rollback": packet.get("rollback"),
            "review_sha256": record["review_sha256"],
        })
    print(json.dumps({"ok": True, "reviewed": reviewed}, indent=2, sort_keys=True))
    return 0


def _list_packets(root: Path, as_json: bool) -> int:
    rows = []
    for path in _pending_packets(root):
        try:
            packet = _load_packet(path)
        except Exception as exc:
            rows.append({"path": str(path), "error": str(exc)})
            continue
        rows.append({
            "path": str(path),
            "target_rel": packet.get("target_rel"),
            "tool_name": packet.get("tool_name"),
            "deny_reasons": packet.get("deny_reasons", []),
            "original_snapshot_available": packet.get("original_snapshot_available"),
            "payload_sha256": packet.get("payload_sha256"),
        })
    if as_json:
        print(json.dumps({"ok": True, "pending": rows}, indent=2, sort_keys=True))
    else:
        if not rows:
            print("qa_codex_quarantine_review: no pending packets")
        for row in rows:
            print(f"{row.get('payload_sha256', '')[:16]} {row.get('target_rel') or row['path']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review Claude Python quarantine packets")
    parser.add_argument("--quarantine-dir", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list", help="List pending packets")
    list_p.add_argument("--json", action="store_true")

    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} pending quarantine packets")
        p.add_argument("packets", nargs="*", help="Packet paths. Use --all for every pending packet.")
        p.add_argument("--all", action="store_true")
        p.add_argument("--reviewer", default="codex")
        p.add_argument("--notes", required=True)

    args = parser.parse_args(argv)
    root = _quarantine_root(args.quarantine_dir)
    if args.cmd == "list":
        return _list_packets(root, args.json)
    if args.cmd in {"approve", "reject"}:
        if not args.all and not args.packets:
            print("qa_codex_quarantine_review: pass packet paths or --all", file=sys.stderr)
            return 2
        return _review_packets(root, args.cmd, args.reviewer, args.notes, args.packets, args.all)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
