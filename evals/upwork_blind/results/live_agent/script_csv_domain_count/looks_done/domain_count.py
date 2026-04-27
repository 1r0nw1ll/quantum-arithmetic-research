#!/usr/bin/env python3
"""Count email domains from a CSV file."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from itertools import chain
from pathlib import Path


def extract_domains(csv_path: str) -> Counter[str]:
    counts: Counter[str] = Counter()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)

        try:
            first_row = next(reader)
        except StopIteration:
            return counts

        email_index = 0
        rows = reader

        lowered_first_row = [value.strip().lower() for value in first_row]
        if "email" in lowered_first_row:
            email_index = lowered_first_row.index("email")
            rows = reader
        else:
            rows = chain([first_row], reader)

        for row in rows:
            if not row or email_index >= len(row):
                continue

            email = row[email_index].strip()
            if "@" not in email:
                continue

            _, domain = email.rsplit("@", 1)
            domain = domain.strip().lower()
            if not domain:
                continue

            counts[domain] += 1

    return counts


def main(argv: list[str]) -> int:
    script_name = Path(argv[0]).name if argv else "domain_count.py"

    if len(argv) == 2 and argv[1] in {"-h", "--help"}:
        print(f"Usage: python3 {script_name} <csv_file>")
        return 0

    if len(argv) != 2:
        print(f"Usage: python3 {script_name} <csv_file>", file=sys.stderr)
        return 1

    counts = extract_domains(argv[1])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
