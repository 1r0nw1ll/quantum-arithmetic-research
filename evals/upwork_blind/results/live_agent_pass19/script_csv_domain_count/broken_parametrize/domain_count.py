#!/usr/bin/env python3
"""CSV Email-Domain Counter CLI

Reads a CSV file and outputs distinct email domains with counts.
- Uses the standard library `csv` module
- Domain counting is case-insensitive (lowercased)
- Skips rows where the email column is missing or malformed (no '@')
- Handles CSV with an 'email' header or no header (first column as fallback)
- Outputs lines: count<TAB>domain, sorted by count desc, then domain asc
"""
from __future__ import annotations
import sys
import csv
from collections import Counter


def parse_email_domain(email: str) -> str | None:
    if not email:
        return None
    email = email.strip()
    if "@" not in email:
        return None
    return email.split("@", 1)[1].lower()


def read_domains_from_csv(path: str) -> Counter:
    counts: Counter[str] = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is None:
            return counts
        # Determine if the first row is a header that contains an 'email' column
        header = [c.strip() for c in first_row]
        email_col_index: int = -1
        # check header for 'email' (case-insensitive)
        for idx, col in enumerate(header):
            if col.lower() == "email":
                email_col_index = idx
                break
        if email_col_index != -1:
            # skip header; process remaining rows
            for row in reader:
                if len(row) <= email_col_index:
                    continue
                domain = parse_email_domain(row[email_col_index])
                if domain:
                    counts[domain] += 1
        else:
            # First row is data; use first column as email
            domain = parse_email_domain(first_row[0] if len(first_row) > 0 else "")
            if domain:
                counts[domain] += 1
            for row in reader:
                if not row:
                    continue
                domain = parse_email_domain(row[0] if len(row) > 0 else "")
                if domain:
                    counts[domain] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        print("Usage: domain_count.py <path-to-csv>", file=sys.stderr)
        return 2
    path = argv[0]
    counts = read_domains_from_csv(path)
    # sort by count desc, then domain asc
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out = sys.stdout
    for domain, cnt in items:
        print(f"{cnt}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
