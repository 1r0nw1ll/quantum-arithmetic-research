#!/usr/bin/env python3
"""Count distinct email domains from a CSV file."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from typing import Iterable


def extract_email(row: list[str], header: list[str] | None) -> str | None:
    """Return the candidate email value for a CSV row."""
    if not row:
        return None

    if header is not None:
        try:
            email_index = header.index("email")
        except ValueError:
            email_index = 0
    else:
        email_index = 0

    if email_index >= len(row):
        return None
    return row[email_index].strip()


def count_domains(rows: Iterable[list[str]]) -> Counter[str]:
    """Count domains from CSV rows, skipping malformed email values."""
    iterator = iter(rows)
    try:
        first_row = next(iterator)
    except StopIteration:
        return Counter()

    header = None
    first_email = extract_email(first_row, None)
    normalized_first = [value.strip().lower() for value in first_row]
    if "email" in normalized_first:
        header = normalized_first
    else:
        iterator = iter([first_row, *iterator])

    counts: Counter[str] = Counter()
    for row in iterator:
        email = extract_email(row, header)
        if not email or "@" not in email:
            continue

        _, domain = email.rsplit("@", 1)
        domain = domain.strip().lower()
        if not domain:
            continue
        counts[domain] += 1

    return counts


def format_counts(counts: Counter[str]) -> str:
    lines = [f"{count}\t{domain}" for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count email domains from a CSV file.")
    parser.add_argument("csv_file", help="Path to the CSV file to process.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    with open(args.csv_file, newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        counts = count_domains(reader)

    output = format_counts(counts)
    if output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
