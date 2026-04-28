# CSV email-domain counter CLI

Scope: count distinct email domains from a CSV using the standard library `csv` module, lowercasing domains and skipping rows whose selected email field lacks `@`.

Usage:

```bash
python3 domain_count.py users.csv
```

Output is written to stdout as tab-separated `count<TAB>domain`, sorted by descending count and then ascending domain.
