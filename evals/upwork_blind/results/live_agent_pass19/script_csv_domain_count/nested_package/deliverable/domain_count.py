#!/usr/bin/env python3
import sys
from deliverable.lib.core.v1 import count_domains_from_csv


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: domain_count.py <csv_file>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    counts = count_domains_from_csv(path)
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for domain, count in items:
        print(f"{count}\t{domain}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
