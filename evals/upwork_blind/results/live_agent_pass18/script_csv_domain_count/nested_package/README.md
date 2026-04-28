# CSV email-domain counter CLI

## Scope

This project provides a small Python CLI that reads a CSV file, extracts email domains from either the `email` column or the first column, skips malformed rows that do not contain `@`, and prints domain counts sorted by descending count and ascending domain for ties.

## Usage

```bash
python3 domain_count.py users.csv
```

Output is written to stdout as tab-separated lines:

```text
count<TAB>domain
```
