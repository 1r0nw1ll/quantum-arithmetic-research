from __future__ import annotations

from io import StringIO
import subprocess
import sys
from pathlib import Path

import pytest

from domain_count import extract_domain, load_domain_counts, sorted_counts


EXTRACT_CASES = [
    ("alice@example.com", "example.com", "basic-address", "normal lowercase domain"),
    ("Alice@Example.COM", "example.com", "mixed-case-domain", "domain is normalized"),
    ("name+tag@sub.example.org", "sub.example.org", "subdomain", "subdomains are preserved"),
    (" spaced@Example.com ", "example.com", "outer-whitespace", "surrounding whitespace is tolerated"),
    ("missing-at-symbol", None, "missing-at", "malformed values are skipped"),
    ("missing-domain@", None, "empty-domain", "empty domains are rejected"),
    ("two@parts@Example.com", "example.com", "multiple-ats", "split uses the last at-sign"),
]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(raw_value, expected) for raw_value, expected, _label, _note in EXTRACT_CASES],
    ids=[label for _raw_value, _expected, label, _note in EXTRACT_CASES],
)
def test_extract_domain(raw_value: str, expected: str | None) -> None:
    assert extract_domain(raw_value) == expected


COUNT_CASES = [
    (
        "alice@example.com\nbob@example.com\ncarol@test.com\n",
        [("example.com", 2), ("test.com", 1)],
        "first-column-basic",
        "counts repeated domains from the first column",
    ),
    (
        "email,name\nalice@example.com,Alice\nbob@test.com,Bob\ncarol@example.com,Carol\n",
        [("example.com", 2), ("test.com", 1)],
        "email-header-selected",
        "uses the named email column when present",
    ),
    (
        "name,email,team\nAlice,ALICE@Example.com,red\nBob,bob@test.com,blue\nCara,cara@example.COM,red\n",
        [("example.com", 2), ("test.com", 1)],
        "non-first-email-column",
        "finds the email column even when it is not first",
    ),
    (
        "email\nalice@example.com\ninvalid-row\nbob@example.com\nmissingatsign\ncarol@test.com\n",
        [("example.com", 2), ("test.com", 1)],
        "skip-malformed-with-header",
        "bad rows without at-signs are ignored",
    ),
    (
        "alice@example.com\n\ninvalid\nbob@test.com\nsolo@\ncarol@EXAMPLE.com\n",
        [("example.com", 2), ("test.com", 1)],
        "skip-malformed-no-header",
        "blank lines and malformed rows do not crash",
    ),
    (
        "email\nalice@b.com\nbob@a.com\ncara@b.com\ndan@a.com\n",
        [("a.com", 2), ("b.com", 2)],
        "tie-sorted-by-domain",
        "ties are ordered alphabetically by domain",
    ),
    (
        "email\nleft@z.com\nright@y.com\nagain@y.com\nthird@x.com\nfourth@y.com\n",
        [("y.com", 3), ("x.com", 1), ("z.com", 1)],
        "count-descending",
        "higher counts sort ahead of lower counts",
    ),
    (
        "",
        [],
        "empty-file",
        "an empty CSV produces no output rows",
    ),
]


@pytest.mark.parametrize(
    ("csv_text", "expected"),
    [(csv_text, expected) for csv_text, expected, _label, _note in COUNT_CASES],
    ids=[label for _csv_text, _expected, label, _note in COUNT_CASES],
)
def test_load_domain_counts_and_sorting(csv_text: str, expected: list[tuple[str, int]]) -> None:
    counts = load_domain_counts(StringIO(csv_text))
    assert sorted_counts(counts) == expected


def test_cli_outputs_tab_separated_sorted_counts(tmp_path: Path) -> None:
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(
        "email,name\n"
        "alice@Example.com,Alice\n"
        "bob@test.com,Bob\n"
        "carol@example.com,Carol\n"
        "not-an-email,Skipped\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "domain_count.py", str(csv_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "2\texample.com\n1\ttest.com\n"
    assert result.stderr == ""
