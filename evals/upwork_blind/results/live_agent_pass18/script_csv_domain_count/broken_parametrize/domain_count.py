#!/usr/bin/env python3
"""Count distinct email domains from a CSV file."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def extract_domain(value: str) -> str | None:
    """Return a normalized domain from an email-like value."""
    if "@" not in value:
        return None

    _, domain = value.rsplit("@", 1)
    domain = domain.strip().lower()
    if not domain:
        return None
    return domain


def count_domains(rows: Iterable[list[str]], email_index: int = 0) -> Counter[str]:
    """Count valid domains from CSV rows."""
    counts: Counter[str] = Counter()

    for row in rows:
        if not row or email_index >= len(row):
            continue

        domain = extract_domain(row[email_index].strip())
        if domain is None:
            continue

        counts[domain] += 1

    return counts


def load_domain_counts(handle: Iterable[str]) -> Counter[str]:
    """Read a CSV stream and count domains from the first column or email column."""
    reader = csv.reader(handle)

    try:
        first_row = next(reader)
    except StopIteration:
        return Counter()

    email_index = 0
    normalized_first_row = [cell.strip().lower() for cell in first_row]
    if "email" in normalized_first_row:
        email_index = normalized_first_row.index("email")
        data_rows: Iterable[list[str]] = reader
    else:
        data_rows = [first_row, *reader]

    return count_domains(data_rows, email_index=email_index)


def sorted_counts(counts: Counter[str]) -> list[tuple[str, int]]:
    """Return domain counts sorted by count desc, domain asc."""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python3 domain_count.py <csv-file>", file=sys.stderr)
        return 2

    csv_path = Path(args[0])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        counts = load_domain_counts(handle)

    for domain, count in sorted_counts(counts):
        print(f"{count}\t{domain}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
