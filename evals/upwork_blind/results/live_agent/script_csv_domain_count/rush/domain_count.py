#!/usr/bin/env python3
import csv
import sys
from collections import Counter


def extract_domain(email):
    if "@" not in email:
        return None
    local, domain = email.rsplit("@", 1)
    if not local or not domain:
        return None
    return domain.lower()


def count_domains(csv_path):
    counts = Counter()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            first_row = next(reader)
        except StopIteration:
            return counts

        email_index = 0
        header_value = first_row[0].strip().lower() if first_row else ""
        if "email" in [cell.strip().lower() for cell in first_row]:
            email_index = [cell.strip().lower() for cell in first_row].index("email")
        else:
            domain = extract_domain(first_row[0] if first_row else "")
            if domain:
                counts[domain] += 1

        for row in reader:
            if email_index >= len(row):
                continue
            domain = extract_domain(row[email_index].strip())
            if domain:
                counts[domain] += 1
    return counts


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 domain_count.py <csv_file>", file=sys.stderr)
        return 1

    counts = count_domains(argv[1])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
