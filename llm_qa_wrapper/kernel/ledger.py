# noqa: FIREWALL-2 (ledger plumbing — no mev/loop content)
"""
ledger.py — Append-only hash-chained ledger.

Implements the TLA+ `ledger` variable and the Lean `Ledger = List
CertRecord` abstraction. Writes are serialized by an in-process lock
and an OS-level O_APPEND flag. Each write is followed by fsync.

Matches:
- cert_gate.tla AppendLedger action
- cert_gate.tla LedgerTailHash operator
- LedgerInvariants.lean `valid` predicate and `append_preserves_valid`
"""
from __future__ import annotations

QA_COMPLIANCE = {
    "observer": "LLM_QA_WRAPPER_LEDGER",
    "state_alphabet": "append-only sequence of CertRecord entries "
                      "forming a SHA-256 hash chain from GENESIS",
    "rationale": "Implements Lean valid predicate and TLA+ ledger "
                 "invariants (LedgerChainValid, Composition). "
                 "fsync after each append provides crash-consistency "
                 "at the line boundary.",
}

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from .cert import CertRecord, GENESIS


@dataclass
class LedgerVerifyResult:
    """Result of verify_chain(). Structured rather than a bool so a
    caller can see which entry broke the chain."""

    ok: bool
    entries_checked: int
    broken_at: Optional[int] = None
    reason: Optional[str] = None


class Ledger:
    """Append-only hash-chained ledger.

    Stores entries in JSONL files under `ledger_dir/`. One file per
    UTC date. Each line is a CertRecord.to_json() serialization.

    Invariants maintained (matches cert_gate.tla Inv_LedgerChainValid):

      1. entries[0].prev == GENESIS
      2. entries[i].prev == entries[i-1].self_hash for all i > 0
      3. No duplicate self_hash across the ledger
      4. Every append is followed by fsync

    Concurrency: a single threading.Lock serializes all mutations.
    Read operations (verify_chain, iterate, tail_hash) do not hold
    the write lock.
    """

    def __init__(self, ledger_dir: Path, filename: str = "ledger.jsonl"):
        self.dir = Path(ledger_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / filename
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._index: set[bytes] = set()       # self_hashes currently present
        self._tail_hash: bytes = GENESIS
        self._count: int = 0
        self._rebuild_state_from_disk()

    def _rebuild_state_from_disk(self) -> None:
        """Read the ledger file and repopulate in-memory state.

        Called once at __init__. Verifies chain integrity on load;
        raises if the on-disk ledger is corrupted.
        """
        with self.path.open("r", encoding="utf-8") as f:
            prev_expected = GENESIS
            count = 0
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cert = CertRecord.from_json(obj)
                except Exception as e:
                    raise RuntimeError(
                        f"Ledger corrupted at line {line_no}: {e}"
                    ) from e
                if cert.prev != prev_expected:
                    raise RuntimeError(
                        f"Ledger chain broken at line {line_no}: "
                        f"expected prev={prev_expected.hex()}, "
                        f"got prev={cert.prev.hex()}"
                    )
                if cert.self_hash in self._index:
                    raise RuntimeError(
                        f"Ledger duplicate at line {line_no}: "
                        f"{cert.self_hash.hex()}"
                    )
                self._index.add(cert.self_hash)
                prev_expected = cert.self_hash
                count += 1
            self._tail_hash = prev_expected
            self._count = count

    def tail_hash(self) -> bytes:
        """The hash a new entry must chain against.

        Matches TLA+ LedgerTailHash operator. Safe to call without
        the lock; returns a snapshot of the atomic field.
        """
        return self._tail_hash

    def __len__(self) -> int:
        return self._count

    def contains(self, self_hash: bytes) -> bool:
        """Check if a self_hash is already in the ledger. O(1)."""
        return self_hash in self._index

    def append(self, cert: CertRecord) -> int:
        """Append a cert to the ledger. Returns the new offset.

        Enforces AppendLedger preconditions from cert_gate.tla:
          - cert not already in ledger
          - cert.prev == LedgerTailHash
          - open file with O_APPEND, write line, fsync

        Raises:
            ValueError if any precondition fails.
            OSError if the filesystem write or fsync fails.
        """
        with self._lock:
            # Precondition 1: no duplicate
            if cert.self_hash in self._index:
                raise ValueError(
                    f"Cert {cert.self_hash.hex()[:16]}... already in ledger"
                )
            # Precondition 2: chain order (cert.prev == tail_hash)
            if cert.prev != self._tail_hash:
                raise ValueError(
                    f"Cert prev={cert.prev.hex()[:16]}... does not match "
                    f"ledger tail={self._tail_hash.hex()[:16]}..."
                )
            # Append line to file with O_APPEND + fsync
            fd = os.open(
                str(self.path),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o644,
            )
            try:
                line = json.dumps(cert.to_json(), sort_keys=True).encode("utf-8")
                os.write(fd, line + b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
            # Update in-memory state (atomic under the lock)
            self._index.add(cert.self_hash)
            self._tail_hash = cert.self_hash
            new_offset = self._count
            self._count += 1
            return new_offset

    def iterate(self) -> Iterator[CertRecord]:
        """Iterate all ledger entries in insertion order.

        Re-reads the file from disk. Does NOT hold the write lock.
        """
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield CertRecord.from_json(obj)

    def verify_chain(self) -> LedgerVerifyResult:
        """Re-verify the entire chain from disk.

        Returns a structured result that identifies where the chain
        broke, if anywhere. Does NOT hold the write lock.
        """
        prev_expected = GENESIS
        seen: set[bytes] = set()
        count = 0
        for i, cert in enumerate(self.iterate()):
            count += 1
            if cert.prev != prev_expected:
                return LedgerVerifyResult(
                    ok=False,
                    entries_checked=count,
                    broken_at=i,
                    reason=(
                        f"Expected prev={prev_expected.hex()[:16]}..., "
                        f"got prev={cert.prev.hex()[:16]}..."
                    ),
                )
            if cert.self_hash in seen:
                return LedgerVerifyResult(
                    ok=False,
                    entries_checked=count,
                    broken_at=i,
                    reason=f"Duplicate cert {cert.self_hash.hex()[:16]}...",
                )
            seen.add(cert.self_hash)
            prev_expected = cert.self_hash
        return LedgerVerifyResult(ok=True, entries_checked=count)

    def snapshot_state(self) -> dict:
        """Return an in-memory state snapshot for tests / conformance."""
        with self._lock:
            return {
                "count": self._count,
                "tail_hash_hex": self._tail_hash.hex(),
                "index_size": len(self._index),
            }

    def reset_for_testing(self) -> None:
        """Destroy the ledger file and reset in-memory state.

        DANGER: data loss. Only called by test harnesses.
        """
        with self._lock:
            self.path.write_text("")
            self._index.clear()
            self._tail_hash = GENESIS
            self._count = 0
