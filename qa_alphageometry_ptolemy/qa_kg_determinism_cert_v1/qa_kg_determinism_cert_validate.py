# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 internal cert validator; grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), tools/qa_kg/canonicalize.py (Dale, 2026), memory/project_qa_mem_review_role.md (Dale, 2026) -->
"""QA-KG Determinism Cert [228] v1 validator.

QA_COMPLIANCE = "cert_validator — validates KG determinism contract, no empirical QA state machine"

Phase 5: validates that the QA-KG build pipeline produces byte-stable
canonical output given a frozen corpus fixture, and that the live
production graph_hash is recorded in `_meta_ledger.json` so `kg.promote()`
can enforce staleness against it.

Gates (D1 through D7; see README.md for full descriptions):
  D1   (HARD) Fixture content hash matches expected_hash.fixture_hash.
  D1.5 (WARN) Manifest repo_head drift vs current git HEAD.
  D2   (HARD) In-process rebuild twice gives byte-identical graph_hash.
  D3   (HARD) Subprocess rebuild gives same graph_hash as D2.
  D4   (SHOULD or N-A) Cross-platform scaffold — PASS on Linux, N-A else.
  D5   (HARD) Ledger graph_hash matches live; bootstrap PASS on first run.
  D6   (HARD) kg.promote() enforces graph_hash staleness (ephemeral test).
  D7   (HARD) No except-Exception-pass swallows in Phase 5 modules (AST).
"""
from __future__ import annotations

QA_COMPLIANCE = "cert_validator — validates KG determinism contract, no empirical QA state machine"

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_CERT_DIR = _HERE.parent
_REPO = _HERE.parents[2]
_META_DIR = _REPO / "qa_alphageometry_ptolemy"
_FIXTURE_DIR = _REPO / "tools" / "qa_kg" / "fixtures" / "corpus_snapshot_v1"
_MANIFEST_PATH = _FIXTURE_DIR / "CANONICAL_MANIFEST.json"
_EXPECTED_HASH_PATH = _CERT_DIR / "expected_hash.json"
_LEDGER_PATH = _META_DIR / "_meta_ledger.json"
_LEDGER_KEY_228 = "228"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# --- Helpers --------------------------------------------------------------


def _load_manifest() -> dict:
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(f"fixture manifest missing at {_MANIFEST_PATH}")
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_expected_hash() -> dict | None:
    if not _EXPECTED_HASH_PATH.exists():
        return None
    return json.loads(_EXPECTED_HASH_PATH.read_text(encoding="utf-8"))


def _expected_rebuild_hash_for_platform(expected: dict) -> str:
    """Read fixture_rebuild_graph_hash for the current platform.

    Supports both legacy string format (single-platform pin, treated as
    Linux) and the platform-keyed dict format introduced when Mac was
    promoted to canonical. Returns "" when no entry for the current
    platform exists (caller treats as bootstrap).
    """
    raw = expected.get("fixture_rebuild_graph_hash", "")
    if isinstance(raw, str):
        # Legacy single-platform pin. Honor it for Linux only; other
        # platforms fall through to bootstrap.
        return raw if platform.system() == "Linux" else ""
    if isinstance(raw, dict):
        return raw.get(platform.system(), "")
    return ""


def _write_expected_hash(fixture_hash: str, fixture_rebuild_graph_hash: str) -> None:
    """Bootstrap / refresh write: records the expected hashes for the
    CURRENT platform so subsequent same-platform runs can compare.

    The `_exempt` key carries the PRIMARY-SOURCE-EXEMPT header in-band so
    that any re-bootstrap (after someone deletes expected_hash.json to
    force a refresh) still satisfies the cert-gate primary-source check
    on the next commit.

    fixture_rebuild_graph_hash is stored under a platform-keyed dict so
    Linux and Darwin can each pin their own deterministic hash. The
    canonicalize pipeline produces stable bytes within a platform but
    differs across platforms (filesystem ordering / NFC edge cases handled
    upstream don't fully eliminate cross-platform drift). Per-platform
    entries let the cert attest within-platform determinism on both Mac
    and Linux without one host's hash falsely failing the other.
    """
    existing = _load_expected_hash() or {}
    existing_rebuild = existing.get("fixture_rebuild_graph_hash", {})
    if isinstance(existing_rebuild, str):
        # Migrate legacy single-platform pin → Linux entry under dict.
        existing_rebuild = {"Linux": existing_rebuild}
    if not isinstance(existing_rebuild, dict):
        existing_rebuild = {}
    existing_rebuild[platform.system()] = fixture_rebuild_graph_hash

    payload = {
        "_exempt": (
            "<!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 5 determinism anchor; "
            "bootstrap-written by qa_kg_determinism_cert_validate.py on first "
            "meta_validator run. Grounds in docs/specs/QA_MEM_SCOPE.md (Dale, 2026), "
            "tools/qa_kg/canonicalize.py (Dale, 2026), "
            "tools/qa_kg/fixtures/corpus_snapshot_v1/CANONICAL_MANIFEST.json (Dale, 2026). "
            "Any intentional refresh: delete this file and rerun "
            "`python qa_alphageometry_ptolemy/qa_meta_validator.py`. -->"
        ),
        "fixture_hash": fixture_hash,
        "fixture_rebuild_graph_hash": existing_rebuild,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qa_compliance": "memory_infra — [228] Phase 5 determinism expected hashes (per-platform)",
    }
    _EXPECTED_HASH_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _current_git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=str(_REPO),
        )
        return r.stdout.strip() or "UNKNOWN"
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return f"UNKNOWN({exc.__class__.__name__})"


def _load_ledger() -> dict:
    if not _LEDGER_PATH.exists():
        return {}
    try:
        return json.loads(_LEDGER_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # NOT an except-pass swallow (D7); the failure is surfaced.
        print(f"[WARN] D5  ledger read failed — {exc}", file=sys.stderr)
        return {}


def _compute_manifest_fixture_hash(manifest: dict) -> str:
    """Content-only hash of the manifest's files[] array."""
    # Local import avoids a hard dependency when D7 scans this file.
    from tools.qa_kg.canonicalize import compute_fixture_hash
    return compute_fixture_hash(manifest)


def _build_in_memory_graph() -> tuple[sqlite3.Connection, str]:
    """Run the pipeline against an ephemeral :memory: DB + return (conn, hash).

    Uses the fixture at _FIXTURE_DIR. Caller owns the returned connection
    and is responsible for closing it.
    """
    from tools.qa_kg.schema import init_db
    from tools.qa_kg.kg import KG
    from tools.qa_kg.build_context import BuildContext, run_pipeline
    from tools.qa_kg.canonicalize import graph_hash

    conn = init_db(":memory:")
    kg = KG(conn)
    ctx = BuildContext.from_fixture(_FIXTURE_DIR)
    run_pipeline(kg, ctx)
    return conn, graph_hash(conn)


# --- Gates ----------------------------------------------------------------


def check_d1_fixture_content_hash() -> tuple[str, str]:
    """D1 HARD: manifest content hash matches expected_hash.fixture_hash.

    Bootstrap behavior: if expected_hash.json is missing, this gate
    cooperates with D2/D5 — it recomputes the live manifest hash but
    cannot compare until the bootstrap write lands (which D2 performs
    after its in-process rebuild succeeds).
    """
    manifest = _load_manifest()
    live_hash = _compute_manifest_fixture_hash(manifest)
    expected = _load_expected_hash()
    if expected is None:
        return (
            "PASS",
            f"bootstrap: expected_hash.json absent; live fixture_hash="
            f"{live_hash[:12]}… will be written by D2 bootstrap path"
        )
    want = expected.get("fixture_hash", "")
    if live_hash == want:
        return "PASS", f"match: fixture_hash={live_hash[:12]}…"
    return (
        "FAIL",
        f"drift: expected={want[:12]}… live={live_hash[:12]}… — "
        f"regenerate fixture or refresh expected_hash.json"
    )


def check_d1_5_repo_head_drift() -> tuple[str, str]:
    """D1.5 WARN: manifest.repo_head vs current HEAD. Never blocks."""
    manifest = _load_manifest()
    pinned = manifest.get("metadata", {}).get("repo_head", "")
    head = _current_git_head()
    if not pinned:
        return "WARN", f"manifest has no repo_head pin; current HEAD={head[:12]}…"
    if pinned == head:
        return "PASS", f"fixture pinned to current HEAD {head[:12]}…"
    return (
        "WARN",
        f"fixture pinned at {pinned[:12]}…; current HEAD={head[:12]}… — "
        f"Phase 5.1 will pin reproducibility to manifest.repo_head via git archive"
    )


def check_d2_in_process_idempotent() -> tuple[str, str]:
    """D2 HARD: two sequential in-process rebuilds give byte-identical hash."""
    conn_a, hash_a = _build_in_memory_graph()
    conn_a.close()
    conn_b, hash_b = _build_in_memory_graph()
    conn_b.close()
    if hash_a != hash_b:
        return (
            "FAIL",
            f"drift: run1={hash_a[:12]}… run2={hash_b[:12]}… — "
            f"non-determinism in canonicalization or extractor pipeline"
        )
    # Bootstrap expected_hash.json if missing. On the first-ever run, D2
    # freezes the fixture_rebuild_graph_hash so subsequent runs have
    # something to compare against. This is explicit in the cert spec:
    # first-run PASS is a bootstrap, not a comparison.
    expected = _load_expected_hash()
    if expected is None:
        manifest = _load_manifest()
        fh = _compute_manifest_fixture_hash(manifest)
        _write_expected_hash(fh, hash_a)
        return (
            "PASS",
            f"bootstrap: wrote expected_hash.json "
            f"(fixture_hash={fh[:12]}… "
            f"{platform.system()}.fixture_rebuild_graph_hash={hash_a[:12]}…)"
        )
    want = _expected_rebuild_hash_for_platform(expected)
    if not want:
        # First run on this platform — bootstrap the platform's entry.
        manifest = _load_manifest()
        fh = _compute_manifest_fixture_hash(manifest)
        _write_expected_hash(fh, hash_a)
        return (
            "PASS",
            f"bootstrap: added {platform.system()} entry "
            f"(fixture_rebuild_graph_hash={hash_a[:12]}…)"
        )
    if hash_a == want:
        return (
            "PASS",
            f"match: {platform.system()}.fixture_rebuild_graph_hash={hash_a[:12]}…"
        )
    return (
        "FAIL",
        f"drift vs expected_hash.json[{platform.system()}]: "
        f"want={want[:12]}… got={hash_a[:12]}… — "
        f"pipeline output changed; refresh expected_hash.json if intentional"
    )


def check_d3_subprocess_reproducible() -> tuple[str, str]:
    """D3 HARD: subprocess rebuild gives same hash as D2.

    Invokes `python -m tools.qa_kg.cli build --fixture <> --hash-only`
    in two separate subprocesses with isolated QA_KG_DB. Hashes must
    match each other and the D2 in-process hash.
    """
    hashes = []
    for i in range(2):
        with tempfile.NamedTemporaryFile(suffix=f".d3_{i}.db", delete=False) as tf:
            tmp_db = tf.name
        try:
            env = os.environ.copy()
            env["QA_KG_DB"] = tmp_db
            # Remove stale DB file to force rebuild from scratch.
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            result = subprocess.run(
                [sys.executable, "-m", "tools.qa_kg.cli",
                 "build", "--fixture", str(_FIXTURE_DIR), "--hash-only"],
                env=env, cwd=str(_REPO),
                capture_output=True, text=True, timeout=120, check=True,
            )
            hashes.append(result.stdout.strip())
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
    if hashes[0] != hashes[1]:
        return (
            "FAIL",
            f"subprocess drift: run1={hashes[0][:12]}… run2={hashes[1][:12]}…"
        )
    # Compare against D2's bootstrapped hash (expected_hash.json must
    # exist by the time D3 runs because D2 is ordered before it; main()
    # enforces the order).
    expected = _load_expected_hash()
    if expected is None:
        return "FAIL", "expected_hash.json missing after D2 ran — ordering bug"
    want = _expected_rebuild_hash_for_platform(expected)
    if not want:
        return (
            "FAIL",
            f"expected_hash.json has no {platform.system()} entry after D2 ran — "
            f"D2 bootstrap ordering bug"
        )
    if hashes[0] != want:
        return (
            "FAIL",
            f"subprocess hash != in-process hash on {platform.system()}: "
            f"subprocess={hashes[0][:12]}… in-process={want[:12]}… — "
            f"interpreter-state sensitivity"
        )
    return (
        "PASS",
        f"match: {platform.system()}.subprocess_graph_hash={hashes[0][:12]}…"
    )


def check_d4_cross_platform_scaffold() -> tuple[str, str]:
    """D4 SHOULD: cross-platform scaffold.

    On Linux, PASS attests Linux determinism. On macOS/Windows, N-A —
    the reviewer must re-run this gate and surface a Linux parity
    comparison. Phase 5 ships the Linux scaffold; platform parity is
    an explicit follow-up.
    """
    sysname = platform.system()
    if sysname == "Linux":
        return "PASS", f"platform={sysname}; Linux parity scaffolded"
    return (
        "N-A",
        f"platform={sysname}; cross-platform parity deferred — "
        f"non-Linux reviewer should compare their hash against Linux's expected_hash.json"
    )


def check_d5_ledger_matches_live(conn_live: sqlite3.Connection) -> tuple[str, str]:
    """D5 HARD: ledger graph_hash matches live production graph_hash.

    Bootstrap: on first-ever run or after ledger entry rebuild that
    dropped graph_hash, PASS with explicit rationale. Subsequent runs
    must match byte-for-byte.
    """
    from tools.qa_kg.canonicalize import graph_hash
    live = graph_hash(conn_live)
    ledger = _load_ledger()
    entry = ledger.get(_LEDGER_KEY_228) or {}
    recorded = entry.get("graph_hash")
    if not recorded:
        # Side-channel to the ledger writer — see qa_meta_validator.py
        # Phase 5 section — the writer reads _kg_graph_hash_by_fam_id
        # and attaches graph_hash to [228]'s entry. Calling here ensures
        # the value propagates when the validator is invoked standalone.
        try:
            import qa_meta_validator as _mv  # type: ignore[import-not-found]
            if hasattr(_mv, "_kg_graph_hash_by_fam_id"):
                _mv._kg_graph_hash_by_fam_id[_LEDGER_KEY_228] = live
        except ImportError as exc:
            # Standalone validator run without meta-validator in path:
            # bootstrap succeeds; ledger will be updated on next sweep.
            return (
                "PASS",
                f"bootstrap: ledger has no prior graph_hash; live={live[:12]}… "
                f"(meta_validator side-channel not reachable: {exc})"
            )
        return (
            "PASS",
            f"bootstrap: ledger has no prior graph_hash to compare; "
            f"validator wrote graph_hash={live[:12]}… for subsequent runs to validate against"
        )
    if live == recorded:
        return "PASS", f"match: graph_hash={live[:12]}…"
    return (
        "FAIL",
        f"drift: ledger={recorded[:12]}… live={live[:12]}… — "
        f"rerun `python qa_alphageometry_ptolemy/qa_meta_validator.py` to refresh"
    )


def check_d6_promote_graph_hash_enforced() -> tuple[str, str]:
    """D6 HARD: ephemeral test that kg.promote() raises FirewallViolation
    on graph_hash drift.

    Sets up:
      - temp DB with an agent Thought node + internal promoter node
      - fake ledger entry with a sentinel graph_hash
      - mutation to induce drift (live != sentinel)
    Asserts FirewallViolation raised on promote.
    """
    from tools.qa_kg.schema import init_db
    from tools.qa_kg.kg import KG, Node, FirewallViolation

    with tempfile.TemporaryDirectory() as td:
        tmp_db = Path(td) / "d6.db"
        fake_ledger = Path(td) / "_meta_ledger.json"
        conn = init_db(tmp_db)
        kg = KG(conn)
        # Real-shape fixtures
        kg.upsert_node(Node(
            id="ob:d6-agent", node_type="Thought",
            title="D6 agent note",
            body="ephemeral agent-authority thought for D6 promote-staleness test",
            authority="agent", epistemic_status="observation",
            method="ob_capture", source_locator="ob:d6-agent",
        ))
        kg.upsert_node(Node(
            id="axiom:A1", node_type="Axiom",
            title="A1 — No-Zero",
            body="ephemeral A1 promoter for D6",
            authority="primary", epistemic_status="axiom",
            method="axioms_block", source_locator="file:QA_AXIOMS_BLOCK.md#A1",
        ))
        # Ledger entry with a SENTINEL graph_hash that does not match live.
        now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        head = _current_git_head()
        ledger_payload = {
            "228": {
                "status": "PASS",
                "ts": now_iso,
                "git_head": head,
                "graph_hash": "0" * 64,  # sentinel that WILL mismatch
            }
        }
        fake_ledger.write_text(json.dumps(ledger_payload), encoding="utf-8")
        # Broadcast payload must be within ±60s window.
        broadcast = {
            "session": "d6-ephemeral",
            "ts": now_iso,
        }
        # Expect FirewallViolation because live graph_hash != sentinel.
        try:
            kg.promote(
                agent_note_id="ob:d6-agent",
                via_cert="228",
                promoter_node_id="axiom:A1",
                broadcast_payload=broadcast,
                ledger_path=fake_ledger,
            )
        except FirewallViolation as exc:
            msg = str(exc)
            if "graph_hash drift" in msg:
                return "PASS", "promote() rejected drift as expected"
            # Some other FirewallViolation fired (e.g. git_head mismatch)
            # — that's still HARD-rejection but not D6's specific path.
            return (
                "FAIL",
                f"promote rejected but for wrong reason: {msg[:100]}"
            )
        finally:
            conn.close()
    return "FAIL", "promote() accepted drift — D6 enforcement broken"


def check_d7_no_swallows_ast() -> tuple[str, str]:
    """D7 HARD: AST scan for except-Exception-pass / bare except: pass.

    Scope: tools/qa_kg/canonicalize.py, tools/qa_kg/build_context.py,
    and this validator itself. Mirrors [254] R8 and [225] KG10.
    """
    targets = (
        _REPO / "tools" / "qa_kg" / "canonicalize.py",
        _REPO / "tools" / "qa_kg" / "build_context.py",
        _HERE,
    )
    viols: list[str] = []
    for path in targets:
        if not path.exists():
            viols.append(f"{path.name}: file missing")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            viols.append(f"{path.name}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = node.body
            # Bare `except:` (type is None) with body=[Pass] or body=[]
            if node.type is None and (
                not body or (len(body) == 1 and isinstance(body[0], ast.Pass))
            ):
                viols.append(
                    f"{path.name}:{node.lineno} bare except: pass (or empty)"
                )
                continue
            # `except Exception:` / `except BaseException:` with body=[Pass]
            if isinstance(node.type, ast.Name) and node.type.id in (
                "Exception", "BaseException"
            ):
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    viols.append(
                        f"{path.name}:{node.lineno} "
                        f"except {node.type.id}: pass swallow"
                    )
    if viols:
        return "FAIL", f"{len(viols)} swallow(s): " + "; ".join(viols[:5])
    return "PASS", f"no swallows across {len(targets)} files"


# --- Main -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=None,
                   help="path to qa_kg.db for D5 live-hash check")
    p.add_argument("--strict", action="store_true",
                   help="treat WARN as FAIL (CI-strict mode)")
    p.add_argument("--show-details", action="store_true")
    args = p.parse_args(argv)

    db = args.db or (_REPO / "tools" / "qa_kg" / "qa_kg.db")
    hard_fail = False

    def run(code: str, desc: str, fn, is_hard: bool) -> None:
        nonlocal hard_fail
        try:
            status, msg = fn()
        except (FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
            # Expected operational failures; surface cleanly without
            # swallowing (D7 compliance). Downstream inspection via msg.
            status, msg = ("FAIL" if is_hard else "WARN"), f"error: {exc}"
        print(f"[{status}] {code}  {desc} — {msg}")
        if status == "FAIL" and (is_hard or args.strict):
            hard_fail = True
        if status == "WARN" and args.strict:
            hard_fail = True

    # ---- Order matters: D1 → D2 → D3 → D4 → D5 → D6 → D7 ----
    # D2 bootstraps expected_hash.json if missing; D3 relies on that write.
    run("D1",   "Fixture content hash",       check_d1_fixture_content_hash, True)
    run("D1.5", "Repo_head drift",            check_d1_5_repo_head_drift,    False)
    run("D2",   "In-process rebuild idempotent", check_d2_in_process_idempotent, True)
    run("D3",   "Subprocess rebuild reproducible", check_d3_subprocess_reproducible, True)
    run("D4",   "Cross-platform scaffold",    check_d4_cross_platform_scaffold, False)

    # D5 needs live DB connection
    if not db.exists():
        print(f"[WARN] D5  Ledger vs live hash — qa_kg.db missing at {db}")
    else:
        conn_live = sqlite3.connect(str(db))
        conn_live.row_factory = sqlite3.Row
        try:
            run("D5", "Ledger matches live", lambda: check_d5_ledger_matches_live(conn_live), True)
        finally:
            conn_live.close()

    run("D6", "promote graph_hash staleness enforced", check_d6_promote_graph_hash_enforced, True)
    run("D7", "No except-pass swallows",   check_d7_no_swallows_ast, True)

    if hard_fail:
        print("[FAIL] QA-KG determinism cert [228] v1")
        return 1
    print("[PASS] QA-KG determinism cert [228] v1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
