from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


def _extract_domain(email: str) -> str | None:
    if "@" not in email:
        return None

    local_part, domain = email.rsplit("@", 1)
    if not local_part or not domain:
        return None

    return domain.lower()


def count_domains(emails: Iterable[str]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()

    for email in emails:
        domain = _extract_domain(email)
        if domain is not None:
            counter[domain] += 1

    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def count_domains_from_csv(csv_path: str | Path) -> list[tuple[str, int]]:
    path = Path(csv_path)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]
    email_index = _resolve_email_column_index(header)
    selected_index = email_index if email_index is not None else 0
    data_rows = rows[1:] if email_index is not None else rows

    emails = []
    for row in data_rows:
        if not row:
            continue
        if selected_index >= len(row):
            continue
        emails.append(row[selected_index].strip())

    return count_domains(emails)


def _resolve_email_column_index(header: Sequence[str]) -> int | None:
    for index, name in enumerate(header):
        if name.strip().lower() == "email":
            return index
    return None


def format_counts(domain_counts: Iterable[tuple[str, int]]) -> str:
    return "\n".join(f"{count}\t{domain}" for domain, count in domain_counts)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("Usage: python3 domain_count.py <csv-file>", file=sys.stderr)
        return 1

    output = format_counts(count_domains_from_csv(args[0]))
    if output:
        print(output)
    return 0
