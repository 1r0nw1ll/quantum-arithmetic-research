# CSV email-domain counter CLI

Scope: Read a CSV file and count distinct email domains from the first column or a column named `email` (case-insensitive). Output is sorted by count (desc) and domain (asc) with lines in the format `count<TAB>domain`.

Usage:

```bash
python3 domain_count.py users.csv
```

Output:

```
2	domain.com
2	example.com
```

Notes:
- Rows missing the `@` character are skipped.
- Domains are lowercased before counting.
- The CSV is read with the standard library `csv` module.
- The script handles a header row that includes a column named `email` (case-insensitive); otherwise it uses the first column.
