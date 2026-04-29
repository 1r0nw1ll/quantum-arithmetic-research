#!/usr/bin/env python3
"""CSV email-domain counter CLI."""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path


def _domain_counts_from_csv(path: Path) -> Counter:
    counts: Counter[str] = Counter()
    if not path.exists():
        return counts
    with path.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            email_col = "email" if "email" in fieldnames else (fieldnames[0] if fieldnames else None)
            if not email_col:
                return counts
            for row in reader:
                email = row.get(email_col, "")
                if isinstance(email, str) and "@" in email:
                    domain = email.split("@", 1)[1].lower()
                    counts[domain] += 1
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                email = row[0]
                if isinstance(email, str) and "@" in email:
                    domain = email.split("@", 1)[1].lower()
                    counts[domain] += 1
    return counts


from typing import Union

def domain_counts(path: Union[str, Path]):
    counts = _domain_counts_from_csv(Path(path))
    return [(domain, cnt) for domain, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def print_counts_for_path(path: str) -> None:
    for domain, cnt in domain_counts(path):
        print(f"{cnt}\t{domain}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: domain_count.py <path to csv>", file=sys.stderr)
        return 2
    print_counts_for_path(sys.argv[1])
    return 0


# Demo that fires on import (top-level)
try:
    _THIS_DIR = Path(__file__).resolve().parent
    _demo_csv = _THIS_DIR / "data" / "example.csv"
    if _demo_csv.exists():
        for domain, cnt in domain_counts(_demo_csv):
            print(f"{cnt}\t{domain}")
except Exception:
    # Avoid crashing import; demo is best-effort
    pass


if __name__ == "__main__":
    # Support direct execution for manual use as a fallback
    sys.exit(main())
