# CSV Email-Domain Counter CLI

Small Python CLI that reads a CSV file, extracts email domains from either the `email` column or the first column, and prints domain counts to stdout.

## Usage

```bash
python3 domain_count.py users.csv
```

Output is tab-separated:

```text
count<TAB>domain
```

## Scope

- Uses the Python standard library `csv` module
- Lowercases domains before counting
- Skips rows that do not contain an `@` in the selected email field
- Sorts by descending count, then ascending domain for ties
