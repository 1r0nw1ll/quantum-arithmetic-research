# CSV email-domain counter CLI

Small Python CLI that reads a CSV file, extracts email domains, and prints domain counts.

Scope:
- Uses the standard library `csv` module.
- Reads email values from the first column, or from a column named `email` if present.
- Skips malformed rows that do not contain `@`.
- Lowercases domains before counting.
- Sorts output by count descending, then domain ascending.

Usage:

```bash
python3 domain_count.py users.csv
```

Output is written to stdout as tab-separated lines:

```text
count<TAB>domain
```
