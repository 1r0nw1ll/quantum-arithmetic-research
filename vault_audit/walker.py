"""
Filesystem crawler for Obsidian-style vaults.

Responsibilities:
    * Traverse one or more root directories.
    * Emit structured records containing path, modification time, size, and
      content checksum metadata suitable for downstream auditing.
    * Optionally stream file payloads (e.g. markdown bodies) when requested.

The implementation will be coordinated with Codex (code generation) and
Gemini (analysis routines).  This module only defines the scaffolding used
for that collaboration.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class VaultRecord:
    """Lightweight container describing a single vault artefact."""

    root: Path
    path: Path
    size_bytes: int
    mtime_ns: int
    sha256: Optional[str] = None
    payload: Optional[bytes] = None


class VaultWalker:
    """Collects `VaultRecord` entries while traversing the filesystem tree."""

    DEFAULT_BLOB_LIMIT = 5 * 1024 * 1024  # 5 MiB

    def __init__(
        self,
        roots: Iterable[Path],
        include_blobs: bool = False,
        hash_payloads: bool = True,
        blob_limit: int = DEFAULT_BLOB_LIMIT,
        include_globs: Optional[Sequence[str]] = None,
        exclude_globs: Optional[Sequence[str]] = None,
    ) -> None:
        self.roots = [Path(root) for root in roots]
        self.include_blobs = include_blobs
        self.hash_payloads = hash_payloads
        self.blob_limit = blob_limit
        self.include_globs = list(include_globs or [])
        self.exclude_globs = list(exclude_globs or [])

    def walk(self) -> Iterator[VaultRecord]:
        """
        Yield `VaultRecord` objects for each file discovered beneath the roots.

        Traversal is depth-first and yields metadata for every file located
        beneath the configured roots. Symlinks and directories are skipped.
        """
        for root in self.roots:
            resolved_root = root.resolve()
            if not resolved_root.exists():
                raise FileNotFoundError(f"Vault root does not exist: {resolved_root}")
            if not resolved_root.is_dir():
                raise NotADirectoryError(f"Vault root is not a directory: {resolved_root}")

            for candidate in resolved_root.rglob("*"):
                try:
                    if not candidate.is_file():
                        continue
                except OSError as exc:
                    logger.warning("Skipping %s due to access error: %s", candidate, exc)
                    continue

                if not self._should_include(resolved_root, candidate):
                    continue

                try:
                    stat = candidate.stat()
                except OSError as exc:
                    logger.warning("Unable to stat %s: %s", candidate, exc)
                    continue

                payload: Optional[bytes] = None
                if self.include_blobs:
                    try:
                        payload = self._read_payload(candidate, stat.st_size)
                    except ValueError as exc:
                        logger.warning("Skipping payload for %s: %s", candidate, exc)

                digest: Optional[str] = None
                if self.hash_payloads:
                    if payload is not None:
                        digest = hashlib.sha256(payload).hexdigest()
                    else:
                        digest = self._hash_file(candidate)

                yield VaultRecord(
                    root=resolved_root,
                    path=candidate,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    sha256=digest,
                    payload=payload,
                )

    def stream_payload(self, path: Path) -> bytes:
        """
        Return the raw payload for `path` when `include_blobs` is enabled.

        ValueError is raised if payload capture was not enabled or the file
        exceeds the configured blob size limit.
        """
        if not self.include_blobs:
            raise ValueError("Payload streaming requested but include_blobs=False.")

        path = Path(path)
        stat = path.stat()
        return self._read_payload(path, stat.st_size)

    # --- Internal helpers -------------------------------------------------

    def _hash_file(self, path: Path) -> str:
        """Compute the SHA-256 digest for `path` without loading the entire file."""
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _read_payload(self, path: Path, size_bytes: int) -> bytes:
        """Read the file payload enforcing the configured blob limit."""
        if size_bytes > self.blob_limit:
            raise ValueError(
                f"File {path} exceeds blob limit ({size_bytes} > {self.blob_limit})"
            )
        return path.read_bytes()

    def _should_include(self, resolved_root: Path, candidate: Path) -> bool:
        """Determine whether `candidate` should be yielded based on glob filters."""
        try:
            rel_path = candidate.relative_to(resolved_root).as_posix()
        except ValueError:
            # Symlink pointing outside the root; treat as included only if explicitly allowed
            rel_path = candidate.as_posix()

        if self.exclude_globs and any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.exclude_globs):
            return False

        if self.include_globs:
            return any(fnmatch.fnmatch(rel_path, pattern) for pattern in self.include_globs)

        return True
