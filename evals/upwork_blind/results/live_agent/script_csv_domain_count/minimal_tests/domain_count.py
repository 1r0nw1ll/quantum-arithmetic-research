#!/usr/bin/env python3
import csv
import sys
from collections import Counter


def extract_domain(value: str) -> str | None:
    if "@" not in value:
        return None
    local_part, domain = value.split("@", 1)
    if not local_part or not domain:
        return None
    return domain.lower()


def count_domains(csv_path: str) -> Counter[str]:
    counts: Counter[str] = Counter()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            return counts

        email_index = 0
        data_rows = [first_row]

        normalized_headers = [cell.strip().lower() for cell in first_row]
        if "email" in normalized_headers:
            email_index = normalized_headers.index("email")
            data_rows = []

        for row in data_rows:
            if len(row) <= email_index:
                continue
            domain = extract_domain(row[email_index].strip())
            if domain is not None:
                counts[domain] += 1

        for row in reader:
            if len(row) <= email_index:
                continue
            domain = extract_domain(row[email_index].strip())
            if domain is not None:
                counts[domain] += 1

    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <csv-file>", file=sys.stderr)
        return 1

    counts = count_domains(argv[1])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
