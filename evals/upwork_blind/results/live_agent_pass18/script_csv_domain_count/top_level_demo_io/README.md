# CSV email-domain counter CLI

Scope: `domain_count.py` reads a CSV file, extracts email addresses from either the first column or a column named `email`, skips malformed rows that lack `@`, and prints domain counts sorted by descending count and ascending domain for ties.

Usage:

```bash
python3 domain_count.py users.csv
```

Output is written to `stdout` as tab-separated `count<TAB>domain` lines. The module also includes a small top-level demonstration that reads `data/example.csv` and prints sample behavior to `stderr`.
