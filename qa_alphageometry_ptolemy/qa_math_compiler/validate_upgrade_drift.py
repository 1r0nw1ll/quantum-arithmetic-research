#!/usr/bin/env python3
"""Mathlib upgrade drift detection self-tests for Family [31].

Simulates four upstream mutation types against a minimal source snapshot
and verifies verify_source_checkout() classifies each correctly.

Mutation types:
  T1  SOURCE_FILE_HASH_MISMATCH     — content changed (byte flip)
  T2  DECLARATION_HASH_MISMATCH     — declaration text changed, file hash updated
  T3  SOURCE_FILE_MISSING           — source file deleted (simulates move/rename)
  T4  SOURCE_COMMIT_MISMATCH        — git HEAD differs from pinned commit

Usage:
  python3 validate_upgrade_drift.py
  python3 validate_upgrade_drift.py --json-only   # suppress narrative output

Output: {"ok": bool, "checks": [{"name": str, "ok": bool, "detail": str}, ...]}
"""

import copy, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).parent
MATHLIB_INGEST = ROOT / "mathlib_ingest"
MATHLIB_PKG = MATHLIB_INGEST / ".lake/packages/mathlib"

sys.path.insert(0, str(MATHLIB_INGEST))
from validate_mathlib_ingest import verify_source_checkout, sha256_bytes  # noqa: E402

# Three BitIndices entries — real Mathlib provenance, share one source file
PROBE_DECLARATIONS = [
    "Nat.bitIndices_zero",
    "Nat.bitIndices_one",
    "Nat.bitIndices_two_mul_add_one",
]
PROBE_SOURCE = "Mathlib/Data/Nat/BitIndices.lean"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    return json.loads((MATHLIB_INGEST / "upstream_registry.v1.json").read_text())


def _entries_for(registry: dict, declarations: list[str]) -> list[dict]:
    by_name = {e["declaration"]: e for e in registry["entries"]}
    return [by_name[d] for d in declarations]


def _build_snapshot(dest: Path, registry: dict, include_probe: bool = True) -> None:
    """Copy probe source file + fixed metadata files into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    if include_probe:
        dst = dest / PROBE_SOURCE
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(MATHLIB_PKG / PROBE_SOURCE, dst)
    for fixed in ("LICENSE", "lean-toolchain", "lake-manifest.json"):
        shutil.copy2(MATHLIB_PKG / fixed, dest / fixed)


def _git_init(root: Path) -> str:
    """Init a git repo, make an empty commit, return the commit hash."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "drift-test",
        "GIT_AUTHOR_EMAIL": "drift@test",
        "GIT_COMMITTER_NAME": "drift-test",
        "GIT_COMMITTER_EMAIL": "drift@test",
        "GIT_AUTHOR_DATE": "2020-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2020-01-01T00:00:00Z",
    }
    subprocess.run(["git", "init", "--quiet", str(root)], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, env=env)
    subprocess.run(
        ["git", "-C", str(root), "commit", "--quiet", "--allow-empty", "-m", "snapshot"],
        check=True, env=env,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _run(registry: dict, entries: list[dict], source_root: Path) -> list[str]:
    """Call verify_source_checkout with a minimal registry dict."""
    errors: list[str] = []
    verify_source_checkout({"source": registry["source"], "entries": entries}, source_root, errors)
    return errors


def _check(name: str, errors: list[str],
           required: list[str], forbidden: list[str] | None = None) -> dict:
    ok = all(r in errors for r in required)
    if forbidden:
        ok = ok and not any(f in errors for f in forbidden)
    detail = f"errors={errors}"
    if not ok:
        miss = [r for r in required if r not in errors]
        if miss:
            detail += f"  MISSING:{miss}"
        if forbidden:
            hit = [f for f in forbidden if f in errors]
            if hit:
                detail += f"  UNEXPECTED:{hit}"
    return {"name": name, "ok": ok, "detail": detail}


# ── test cases ────────────────────────────────────────────────────────────────

def run_tests() -> list[dict]:
    registry = _load_registry()
    entries = _entries_for(registry, PROBE_DECLARATIONS)
    checks: list[dict] = []

    # T0: clean run against the real pinned Mathlib package
    errors = _run(registry, entries, MATHLIB_PKG)
    checks.append(_check(
        "T0_clean_real_source", errors,
        required=[],
        forbidden=[
            f"SOURCE_FILE_MISSING:{PROBE_SOURCE}",
            f"SOURCE_FILE_HASH_MISMATCH:{PROBE_SOURCE}",
            "DECLARATION_HASH_MISMATCH:Nat.bitIndices_zero",
            "SOURCE_GIT_HEAD_UNAVAILABLE",
            "SOURCE_COMMIT_MISMATCH",
        ],
    ))

    with tempfile.TemporaryDirectory(prefix="qa_drift_") as tmp:
        tmp_path = Path(tmp)

        # T1: Mutation A — flip a byte in the probe source file
        snap = tmp_path / "T1"
        _build_snapshot(snap, registry)
        _git_init(snap)
        probe = snap / PROBE_SOURCE
        raw = probe.read_bytes()
        probe.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
        errors = _run(registry, entries, snap)
        checks.append(_check(
            "T1_file_byte_flip", errors,
            required=[f"SOURCE_FILE_HASH_MISMATCH:{PROBE_SOURCE}"],
            forbidden=[f"SOURCE_FILE_MISSING:{PROBE_SOURCE}"],
        ))

        # T2: Mutation B — change a declaration line; update file hash in test-registry
        # but leave declaration_source_sha256 unchanged → DECLARATION_HASH_MISMATCH
        snap = tmp_path / "T2"
        _build_snapshot(snap, registry)
        _git_init(snap)
        probe = snap / PROBE_SOURCE
        src_lines = bytearray(probe.read_bytes()).split(b"\n")
        e0 = entries[0]  # Nat.bitIndices_zero
        line_idx = e0["start_line"] - 1
        src_lines[line_idx] += b"  -- drift"
        new_bytes = b"\n".join(src_lines)
        probe.write_bytes(new_bytes)
        # Build single-entry test-registry with new file hash but old declaration hash
        e0_b = copy.deepcopy(e0)
        e0_b["source_file_sha256"] = sha256_bytes(new_bytes)
        errors = _run(registry, [e0_b], snap)
        checks.append(_check(
            "T2_declaration_text_mutated", errors,
            required=[f"DECLARATION_HASH_MISMATCH:{e0['declaration']}"],
        ))

        # T3: Mutation C — delete the probe source file
        snap = tmp_path / "T3"
        _build_snapshot(snap, registry, include_probe=False)
        _git_init(snap)
        errors = _run(registry, entries, snap)
        checks.append(_check(
            "T3_source_file_deleted", errors,
            required=[f"SOURCE_FILE_MISSING:{PROBE_SOURCE}"],
        ))

        # T4: Mutation D — fake git commit ≠ PINNED_COMMIT
        snap = tmp_path / "T4"
        _build_snapshot(snap, registry)
        commit = _git_init(snap)
        errors = _run(registry, entries, snap)
        checks.append(_check(
            "T4_commit_mismatch", errors,
            required=["SOURCE_COMMIT_MISMATCH"],
        ))
        # Sanity: T4 should have NO file/declaration errors (only commit mismatch)
        checks.append(_check(
            "T4_no_file_errors_when_clean", errors,
            required=[],
            forbidden=[
                f"SOURCE_FILE_MISSING:{PROBE_SOURCE}",
                f"SOURCE_FILE_HASH_MISMATCH:{PROBE_SOURCE}",
                "DECLARATION_HASH_MISMATCH:Nat.bitIndices_zero",
            ],
        ))

    return checks


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    json_only = "--json-only" in sys.argv
    checks = run_tests()
    all_ok = all(c["ok"] for c in checks)
    result = {"ok": all_ok, "checks": checks}
    if not json_only:
        for c in checks:
            status = "PASS" if c["ok"] else "FAIL"
            print(f"[{status}] {c['name']}")
            if not c["ok"]:
                print(f"       {c['detail']}")
    print(json.dumps(result))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
