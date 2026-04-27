#!/usr/bin/env python3
import csv
import sys
from collections import Counter


def iter_emails(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return

        email_index = None
        for index, value in enumerate(first_row):
            if value.strip().lower() == "email":
                email_index = index
                break

        if email_index is not None:
            rows = reader
        else:
            email_index = 0
            rows = [first_row]
            rows.extend(reader)

        for row in rows:
            if email_index >= len(row):
                continue
            yield row[email_index].strip()


def count_domains(csv_path):
    counts = Counter()
    for email in iter_emails(csv_path):
        if "@" not in email:
            continue
        _, domain = email.rsplit("@", 1)
        domain = domain.strip().lower()
        if not domain:
            continue
        counts[domain] += 1
    return counts


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("Usage: python3 domain_count.py <csv-file>", file=sys.stderr)
        return 1

    counts = count_domains(argv[0])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
