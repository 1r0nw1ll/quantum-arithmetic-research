# CSV email-domain counter CLI

Write a Python CLI that:

- reads a CSV file whose first column (or a column named `email`) is an email address
- outputs the distinct email domains with their counts
- sorted descending by count, then ascending by domain for ties

Invocation:

```
python3 domain_count.py users.csv
```

Output format is tab-separated `count<TAB>domain` on stdout.

Requirements to handle honestly:

- rows missing the `@` character should be skipped (not crash)
- domains should be lowercased before counting
- the CSV should be read with the standard library `csv` module
- include a short README with scope + usage
- include a test file that exercises basic counting and malformed-row skipping
