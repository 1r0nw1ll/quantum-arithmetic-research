from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


def _normalize_domain(value: str) -> str | None:
    if "@" not in value:
        return None
    _, domain = value.rsplit("@", 1)
    domain = domain.strip().lower()
    return domain or None


def count_domains(csv_path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            first_row = next(reader)
        except StopIteration:
            return counts

        email_index = _detect_email_index(first_row)
        data_rows: Iterable[list[str]]
        if email_index is None:
            email_index = 0
            data_rows = [first_row, *reader]
        else:
            data_rows = reader

        for row in data_rows:
            if email_index >= len(row):
                continue
            domain = _normalize_domain(row[email_index].strip())
            if domain is not None:
                counts[domain] += 1

    return counts


def _detect_email_index(row: list[str]) -> int | None:
    normalized = [value.strip().lower() for value in row]
    try:
        return normalized.index("email")
    except ValueError:
        return None


def format_counts(counts: Counter[str]) -> list[str]:
    ordered_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [f"{count}\t{domain}" for domain, count in ordered_items]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: python3 {Path(__file__).name} <csv-file>", file=sys.stderr)
        return 1

    for line in format_counts(count_domains(argv[1])):
        print(line)
    return 0


_SCRIPT_EXIT_CODE = None
if Path(sys.argv[0]).name == Path(__file__).name:
    _SCRIPT_EXIT_CODE = main(sys.argv)


_DEMO_SAMPLE = Path(__file__).resolve().parent / "data" / "example.csv"
print(f"[demo] reading sample data from {_DEMO_SAMPLE}", file=sys.stderr)
try:
    _demo_counts = count_domains(_DEMO_SAMPLE)
    print("[demo] sorted domain counts:", file=sys.stderr)
    for _demo_line in format_counts(_demo_counts):
        print(f"[demo] {_demo_line}", file=sys.stderr)
    print(
        f"[demo] normalize 'Alice@Example.COM' -> {_normalize_domain('Alice@Example.COM')}",
        file=sys.stderr,
    )
    print(
        f"[demo] normalize 'missing-at-symbol' -> {_normalize_domain('missing-at-symbol')}",
        file=sys.stderr,
    )
    print(
        f"[demo] normalize 'bob@Sub.Example.org' -> {_normalize_domain('bob@Sub.Example.org')}",
        file=sys.stderr,
    )
except FileNotFoundError:
    print("[demo] sample file is missing; skipping demonstration", file=sys.stderr)


if _SCRIPT_EXIT_CODE is not None:
    raise SystemExit(_SCRIPT_EXIT_CODE)
