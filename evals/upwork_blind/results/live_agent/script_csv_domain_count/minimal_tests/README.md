# CSV Email-Domain Counter CLI

Small Python CLI that reads a CSV file, extracts email domains from either:

- a column named `email`, or
- the first column when no `email` header is present

It prints domain counts as tab-separated `count<TAB>domain`, sorted by descending count and then ascending domain.

## Usage

```bash
python3 domain_count.py users.csv
```

Rows without a valid `@` in the selected email field are skipped.
