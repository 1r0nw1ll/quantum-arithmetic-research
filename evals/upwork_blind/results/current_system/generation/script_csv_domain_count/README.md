# CSV email-domain counter

Reads a CSV with an `email` column and prints distinct domains with their
counts, sorted descending by count, then ascending by domain.

## Usage

```bash
python3 domain_count.py users.csv
```

Output is tab-separated `count<TAB>domain` on stdout.

## Scope

- CSV must have an `email` header.
- Malformed rows (missing `@`) are skipped silently.
- Domains are lowercased before counting.
- Whitespace is trimmed from the email field.
