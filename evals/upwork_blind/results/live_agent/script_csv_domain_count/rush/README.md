# CSV Email-Domain Counter CLI

Scope: reads a CSV file, extracts email domains from the `email` column when present or from the first column otherwise, skips malformed rows, and prints domain counts.

Usage:

```bash
python3 domain_count.py users.csv
```

Output is written to stdout as tab-separated `count<TAB>domain`, sorted by descending count and then ascending domain.
