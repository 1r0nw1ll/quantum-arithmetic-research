#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO


def extract_domain(value: str) -> str | None:
    if "@" not in value:
        return None

    local, domain = value.rsplit("@", 1)
    if not local or not domain:
        return None

    return domain.lower()


def iter_email_values(csv_path: Path) -> Iterable[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)

        has_header = csv.Sniffer().has_header(sample) if sample else False

        if has_header:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return

            if "email" in reader.fieldnames:
                for row in reader:
                    value = row.get("email")
                    if value is not None:
                        yield value
            else:
                first_field = reader.fieldnames[0]
                for row in reader:
                    value = row.get(first_field)
                    if value is not None:
                        yield value
        else:
            reader = csv.reader(handle)
            for row in reader:
                if row:
                    yield row[0]


def count_domains(csv_path: Path) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()

    for value in iter_email_values(csv_path):
        domain = extract_domain(value.strip())
        if domain is not None:
            counts[domain] += 1

    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def write_counts(rows: Iterable[tuple[str, int]], output: TextIO) -> None:
    for domain, count in rows:
        output.write(f"{count}\t{domain}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count distinct email domains in a CSV file."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = count_domains(args.csv_path)
    write_counts(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
