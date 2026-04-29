CSV Email-Domain Counter CLI

Scope:
- Lightweight Python package and CLI to count distinct email domains from a CSV file.
- Reads standard library CSV, normalizes domains to lowercase, and skips malformed rows.

Usage:
- CLI: python3 domain_count.py users.csv

Notes:
- The CSV should have a header with a column named 'email' (case-insensitive). If not, the first column is used.
- Output is tab-separated: count<TAB>domain, sorted by count (desc) then domain (asc).
