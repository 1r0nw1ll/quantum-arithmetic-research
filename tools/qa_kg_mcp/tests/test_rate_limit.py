# <!-- PRIMARY-SOURCE-EXEMPT: reason=QA-KG Phase 6 rate-limit tests -->
"""Phase 6 rate limit module tests (_agent_writes.json).

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

Run: python -m tools.qa_kg_mcp.tests.test_rate_limit
"""
from __future__ import annotations

QA_COMPLIANCE = "memory_infra — graph over project artifacts, not empirical QA state"

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.qa_kg_mcp.rate_limit import (
    RateLimitExceeded,
    decay_on_session_done,
    get_count,
    increment,
    reset_session,
)


def _fresh_ledger() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="qa_kg_rl_"))
    return tmp / "_agent_writes.json"


def test_increment_from_scratch_starts_at_one():
    p = _fresh_ledger()
    n = increment("s", ledger_path=p)
    assert n == 1
    data = json.loads(p.read_text())
    assert data["s"]["count"] == 1
    assert data["s"]["first_ts"]
    assert data["s"]["last_ts"]


def test_increment_accumulates_and_preserves_first_ts():
    p = _fresh_ledger()
    increment("s", ledger_path=p)
    first_ts_1 = json.loads(p.read_text())["s"]["first_ts"]
    n = increment("s", ledger_path=p)
    data = json.loads(p.read_text())
    assert n == 2
    assert data["s"]["count"] == 2
    assert data["s"]["first_ts"] == first_ts_1


def test_increment_isolates_sessions():
    p = _fresh_ledger()
    increment("s1", ledger_path=p)
    increment("s1", ledger_path=p)
    increment("s2", ledger_path=p)
    data = json.loads(p.read_text())
    assert data["s1"]["count"] == 2
    assert data["s2"]["count"] == 1


def test_rate_limit_raises_at_cap():
    """Plan W4: simulate 51 promote calls; 51st raises."""
    p = _fresh_ledger()
    for i in range(50):
        increment("s", ledger_path=p, max_writes=50)
    try:
        increment("s", ledger_path=p, max_writes=50)
    except RateLimitExceeded:
        pass
    else:
        raise AssertionError("cap must raise RateLimitExceeded")
    assert get_count("s", ledger_path=p) == 50


def test_decay_removes_session_entry():
    p = _fresh_ledger()
    increment("s", ledger_path=p)
    increment("s", ledger_path=p)
    decay_on_session_done("s", ledger_path=p)
    assert get_count("s", ledger_path=p) == 0
    data = json.loads(p.read_text())
    assert "s" not in data


def test_reset_session_is_explicit_and_reports_found():
    p = _fresh_ledger()
    increment("s", ledger_path=p)
    assert reset_session("s", ledger_path=p) is True
    assert reset_session("s", ledger_path=p) is False
    assert get_count("s", ledger_path=p) == 0


def test_increment_rejects_empty_session():
    p = _fresh_ledger()
    try:
        increment("", ledger_path=p)
    except ValueError:
        pass
    else:
        raise AssertionError("empty session must raise")


def test_missing_ledger_file_treated_as_empty():
    tmp = Path(tempfile.mkdtemp(prefix="qa_kg_rl_"))
    p = tmp / "never_created.json"
    assert get_count("s", ledger_path=p) == 0
    n = increment("s", ledger_path=p)
    assert n == 1
    assert p.exists()


def test_corrupt_ledger_recovers_to_empty():
    p = _fresh_ledger()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not-json-at-all")
    n = increment("s", ledger_path=p)
    assert n == 1


TESTS = [
    test_increment_from_scratch_starts_at_one,
    test_increment_accumulates_and_preserves_first_ts,
    test_increment_isolates_sessions,
    test_rate_limit_raises_at_cap,
    test_decay_removes_session_entry,
    test_reset_session_is_explicit_and_reports_found,
    test_increment_rejects_empty_session,
    test_missing_ledger_file_treated_as_empty,
    test_corrupt_ledger_recovers_to_empty,
]


if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            print(f"FAIL {t.__name__}: {exc}")
            failed += 1
    if failed:
        sys.exit(1)
    print(f"\n{len(TESTS)}/{len(TESTS)} PASS")
