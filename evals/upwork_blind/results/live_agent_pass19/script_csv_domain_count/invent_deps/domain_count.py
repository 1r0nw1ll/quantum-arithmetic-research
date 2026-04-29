#!/usr/bin/env python3
import csv
import sys
from collections import Counter


def main():
    if len(sys.argv) != 2:
        print("Usage: domain_count.py <path_to_csv>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    counts = Counter()
    try:
        with open(path, newline='', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        return
    first = rows[0]
    has_header = isinstance(first, list) and any(isinstance(v, str) and v.lower() == 'email' for v in first)
    if has_header:
        email_col = None
        for i, v in enumerate(first):
            if isinstance(v, str) and v.lower() == 'email':
                email_col = i
                break
        if email_col is None:
            email_col = 0
        data_rows = rows[1:]
    else:
        email_col = 0
        data_rows = rows
    for row in data_rows:
        if not isinstance(row, list) or len(row) <= email_col:
            continue
        email = row[email_col]
        if not isinstance(email, str):
            continue
        if '@' not in email:
            continue
        domain = email.rsplit('@', 1)[-1].lower()
        counts[domain] += 1
    # sort by count desc, then domain asc
    for domain, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{count}\t{domain}")


if __name__ == "__main__":
    main()
