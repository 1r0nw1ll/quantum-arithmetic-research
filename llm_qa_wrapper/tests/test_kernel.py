# noqa: DECL-1 (test file)
"""
test_kernel.py — unit tests for llm_qa_wrapper.kernel.

Exercises the action-for-action mapping from cert_gate.tla to the
Python kernel. Each test case corresponds to one TLA+ action or one
rejection path and asserts the kernel state matches the TLA+ post-state.

Run:
    python llm_qa_wrapper/tests/test_kernel.py

Exit 0 = all tests pass; nonzero = at least one failure.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "llm_qa_wrapper"))

from kernel import (  # noqa: E402
    CertRecord,
    GENESIS,
    Gate,
    GateDecision,
    Ledger,
    gated_tool,
    payload_hash,
)
from kernel.hooks import decorated_tool_names  # noqa: E402


# ── Test harness plumbing ────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def test(name: str):
    def decorator(fn):
        def runner():
            try:
                fn()
                _results.append((name, True, ""))
                print(f"[PASS] {name}")
            except AssertionError as e:
                _results.append((name, False, str(e)))
                print(f"[FAIL] {name}: {e}")
            except Exception as e:
                _results.append((name, False, f"unexpected {type(e).__name__}: {e}"))
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return runner
    return decorator


def make_gate() -> Gate:
    tmpdir = tempfile.mkdtemp(prefix="qa_wrapper_test_")
    ledger = Ledger(Path(tmpdir))
    return Gate(ledger=ledger)


# ── TLA+ action: RequestToolCall ─────────────────────────────────────

@test("RequestToolCall adds to pending")
def t_request():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    assert isinstance(rid, str) and len(rid) > 0
    snap = g.snapshot_state()
    assert snap["pending_count"] == 1
    assert snap["certs_count"] == 0


# ── TLA+ action: IssueCert ───────────────────────────────────────────

@test("IssueCert produces a valid cert with GENESIS prev on first issue")
def t_issue_first():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    cert = g.issue_cert(rid)
    assert cert is not None
    assert cert.prev == GENESIS
    assert cert.counter == 0
    assert cert.agent == "claude"
    assert cert.tool == "bash"
    assert cert.payload_hash_bytes == payload_hash({"cmd": "ls"})
    snap = g.snapshot_state()
    assert snap["pending_count"] == 0
    assert snap["certs_count"] == 1


@test("IssueCert second cert chains against first")
def t_issue_chain():
    g = make_gate()
    rid1 = g.submit_request("claude", "bash", {"cmd": "ls"})
    cert1 = g.issue_cert(rid1)
    assert cert1 is not None

    rid2 = g.submit_request("claude", "bash", {"cmd": "pwd"})
    cert2 = g.issue_cert(rid2)
    assert cert2 is not None
    assert cert2.prev == cert1.self_hash
    assert cert2.counter == cert1.counter + 1
    assert cert2.self_hash != cert1.self_hash


@test("IssueCert on unknown request_id returns None and logs denial")
def t_issue_unknown():
    g = make_gate()
    cert = g.issue_cert("nonexistent")
    assert cert is None
    assert g.last_denial is not None
    assert g.last_denial.reason == "NOT_PENDING"


# ── TLA+ action: Deny (via policy) ──────────────────────────────────

@test("Deny policy rejects cert and records denial")
def t_deny():
    def always_deny(a, t, p):
        return (GateDecision.DENY, "POLICY_TEST")
    g = make_gate()
    g.policy = always_deny
    rid = g.submit_request("claude", "bash", {"cmd": "rm -rf /"})
    cert = g.issue_cert(rid)
    assert cert is None
    assert g.last_denial is not None
    assert g.last_denial.reason == "POLICY_TEST"
    snap = g.snapshot_state()
    assert snap["denied_count"] == 1
    assert snap["certs_count"] == 0


# ── TLA+ action: Execute ─────────────────────────────────────────────

@test("Execute marks cert as executed")
def t_execute():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    cert = g.issue_cert(rid)
    ok = g.execute(cert)
    assert ok is True
    snap = g.snapshot_state()
    assert snap["executed_count"] == 1


@test("Execute of a replayed cert returns False (no-double-exec)")
def t_execute_replay():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    cert = g.issue_cert(rid)
    assert g.execute(cert) is True
    assert g.execute(cert) is False  # replay rejected


@test("Execute of a forged (non-issued) cert returns False")
def t_execute_forgery():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    cert = g.issue_cert(rid)
    # Construct a fake cert with different fields
    forged = CertRecord.create("adv", "bash", {"cmd": "evil"}, GENESIS, 99)
    assert g.execute(forged) is False
    # Legitimate one still works
    assert g.execute(cert) is True


# ── TLA+ action: AppendLedger + LedgerTailHash ─────────────────────

@test("AppendLedger in cert-chain order succeeds")
def t_append_ok():
    g = make_gate()
    rid1 = g.submit_request("claude", "bash", {"cmd": "ls"})
    c1 = g.issue_cert(rid1)
    g.execute(c1)  # atomic execute+append

    rid2 = g.submit_request("claude", "bash", {"cmd": "pwd"})
    c2 = g.issue_cert(rid2)
    g.execute(c2)  # atomic execute+append

    assert len(g.ledger) == 2
    vr = g.ledger.verify_chain()
    assert vr.ok, f"chain broken: {vr.reason}"


@test("Direct Ledger.append rejects out-of-order cert.prev mismatch")
def t_append_out_of_order():
    # Bypass the gate to test ledger-level enforcement directly:
    # construct two certs with chained prev values, then append c2 first.
    g = make_gate()
    rid1 = g.submit_request("claude", "bash", {"cmd": "ls"})
    c1 = g.issue_cert(rid1)
    rid2 = g.submit_request("claude", "bash", {"cmd": "pwd"})
    c2 = g.issue_cert(rid2)
    # Neither has been executed (no append yet). Try direct append of c2.
    try:
        g.ledger.append(c2)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "does not match ledger tail" in str(e)


@test("Direct Ledger.append rejects duplicate cert")
def t_append_duplicate():
    g = make_gate()
    rid = g.submit_request("claude", "bash", {"cmd": "ls"})
    c = g.issue_cert(rid)
    g.execute(c)  # atomic execute+append, now in ledger
    try:
        g.ledger.append(c)  # direct re-append should reject
        raise AssertionError("should have raised on duplicate")
    except ValueError as e:
        assert "already in ledger" in str(e)


@test("Ledger.verify_chain reports ok on a fresh empty ledger")
def t_verify_empty():
    g = make_gate()
    vr = g.ledger.verify_chain()
    assert vr.ok
    assert vr.entries_checked == 0


# ── @gated_tool decorator ────────────────────────────────────────────

@test("@gated_tool wraps a function, produces allowed=True on success")
def t_decorator_allow():
    g = make_gate()

    @gated_tool(g, tool_name="echo.test")
    def echo(x):
        return f"echo:{x}"

    result = echo("hello")
    assert result.allowed is True
    assert result.value == "echo:hello"
    assert result.cert is not None
    assert len(g.ledger) == 1


@test("@gated_tool denies when policy says DENY")
def t_decorator_deny():
    def always_deny(a, t, p):
        return (GateDecision.DENY, "BLOCKED")
    g = make_gate()
    g.policy = always_deny

    @gated_tool(g, tool_name="echo.denied")
    def echo(x):
        return f"echo:{x}"

    result = echo("hello")
    assert result.allowed is False
    assert result.value is None
    assert result.deny_reason == "BLOCKED"
    assert len(g.ledger) == 0  # denied call does NOT hit ledger


@test("@gated_tool registers the function in _REGISTRY")
def t_decorator_registry():
    g = make_gate()

    @gated_tool(g, tool_name="registry.probe")
    def probe():
        return 42

    assert "registry.probe" in decorated_tool_names()


# ── Concurrency test (per ARCHITECTURE.md §5) ────────────────────────

@test("Concurrent IssueCert preserves hash chain under 20-thread race")
def t_concurrent_issue():
    g = make_gate()
    results: list[CertRecord] = []
    lock = threading.Lock()

    def worker(i):
        rid = g.submit_request("claude", "bash", {"i": i})
        cert = g.issue_cert(rid)
        with lock:
            results.append(cert)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 20 certs issued
    assert len(results) == 20
    # All counters are unique and cover 0..19
    counters = sorted(c.counter for c in results)
    assert counters == list(range(20))
    # Hash chain is well-formed: sort by counter, then each prev = previous self_hash
    by_ctr = sorted(results, key=lambda c: c.counter)
    assert by_ctr[0].prev == GENESIS
    for i in range(1, 20):
        assert by_ctr[i].prev == by_ctr[i - 1].self_hash, \
            f"chain broken at counter {i}"


# ── CertRecord tampering detection ──────────────────────────────────

@test("CertRecord construction rejects tampered self_hash")
def t_tampering():
    c = CertRecord.create("claude", "bash", {"cmd": "ls"}, GENESIS, 0)
    try:
        CertRecord(
            agent=c.agent,
            tool=c.tool,
            payload_hash_bytes=c.payload_hash_bytes,
            prev=c.prev,
            counter=c.counter,
            self_hash=GENESIS,  # tampered
        )
        raise AssertionError("should have rejected tampered self_hash")
    except ValueError as e:
        assert "tampered" in str(e) or "does not match" in str(e)


# ── Ledger persistence across restart ──────────────────────────────

@test("Ledger rebuilds state from disk on __init__")
def t_ledger_persistence():
    tmpdir = tempfile.mkdtemp(prefix="qa_persist_test_")
    ledger1 = Ledger(Path(tmpdir))
    gate1 = Gate(ledger=ledger1)
    for i in range(3):
        rid = gate1.submit_request("claude", "bash", {"i": i})
        c = gate1.issue_cert(rid)
        gate1.execute(c)  # atomic execute+append
    tail_after_writes = gate1.ledger.tail_hash()
    assert len(gate1.ledger) == 3

    # "Restart" — new Ledger instance on the same directory
    ledger2 = Ledger(Path(tmpdir))
    assert len(ledger2) == 3
    assert ledger2.tail_hash() == tail_after_writes
    vr = ledger2.verify_chain()
    assert vr.ok


# ── Run all tests ──────────────────────────────────────────────────

def main() -> int:
    tests = [
        t_request, t_issue_first, t_issue_chain, t_issue_unknown,
        t_deny, t_execute, t_execute_replay, t_execute_forgery,
        t_append_ok, t_append_out_of_order, t_append_duplicate,
        t_verify_empty,
        t_decorator_allow, t_decorator_deny, t_decorator_registry,
        t_concurrent_issue,
        t_tampering, t_ledger_persistence,
    ]
    print("=== LLM QA Wrapper kernel unit tests ===\n")
    for t in tests:
        t()

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
