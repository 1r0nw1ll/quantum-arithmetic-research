# CSV Email-Domain Counter

Small Python CLI that reads a CSV file, extracts email addresses from either the first column or a column named `email`, and prints domain counts.

## Usage

```bash
python3 domain_count.py users.csv
```

Output is written to stdout as tab-separated lines:

```text
count<TAB>domain
```

Rows without an `@` are skipped, and domains are lowercased before counting.
