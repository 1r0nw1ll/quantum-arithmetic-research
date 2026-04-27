# CSV Email-Domain Counter CLI

This deliverable provides a production-ready command-line utility for counting distinct email domains from CSV input. It is intentionally compact, uses only Python's standard library, and covers the required malformed-row handling so the tool remains reliable on imperfect data.

## Scope

- Reads a CSV file with the standard library `csv` module.
- Uses the `email` column when present, otherwise falls back to the first column.
- Skips rows that do not contain an `@` character instead of failing.
- Normalizes domains to lowercase before counting.
- Sorts output by descending count, then ascending domain for ties.
- Prints results to stdout as tab-separated `count<TAB>domain`.

## Usage

```bash
python3 domain_count.py users.csv
```

Example output:

```text
12	example.com
4	test.com
1	other.org
```

## Testing

Run the included tests with:

```bash
python3 -m unittest -v
```

The test suite covers basic counting, sort order, and malformed-row skipping behavior.
