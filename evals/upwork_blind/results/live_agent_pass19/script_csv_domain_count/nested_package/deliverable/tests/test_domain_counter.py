import textwrap
from pathlib import Path

from deliverable.lib.core.v1 import count_domains_from_csv


def test_basic_counting(tmp_path):
    csv = tmp_path / "users.csv"
    content = "name,email\nAlice,alice@example.com\nBob,BOB@example.org\nCarol,carol@example.co.uk\nDave,daveexample.com\nEve,eve@example.com\n"
    csv.write_text(content, encoding='utf-8')
    counts = count_domains_from_csv(str(csv))
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert items == [
        ("example.com", 2),
        ("example.co.uk", 1),
        ("example.org", 1),
    ]


def test_malformed_rows_skipped(tmp_path):
    csv = tmp_path / "users.csv"
    content = "name,email\nAlice,alice@example.com\nMallory,malloryexample.com\n"
    csv.write_text(content, encoding='utf-8')
    counts = count_domains_from_csv(str(csv))
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert items == [("example.com", 1)]
