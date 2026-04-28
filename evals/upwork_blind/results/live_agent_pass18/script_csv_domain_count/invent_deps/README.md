# CSV Email-Domain Counter CLI

Small Python CLI that reads a CSV file, extracts email domains from either the `email` column or the first column, and prints domain counts as tab-separated output.

## Usage

```bash
python3 domain_count.py users.csv
```

Output format:

```text
count<TAB>domain
```

Behavior:

- Uses the standard library `csv` module for reading input
- Lowercases domains before counting
- Skips rows that do not contain a valid `@`-separated email value
- Sorts by descending count, then ascending domain for ties
