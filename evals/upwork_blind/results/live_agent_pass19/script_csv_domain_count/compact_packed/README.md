CSV Email-Domain Counter
========================
A tiny Python CLI that counts distinct email domains from a CSV file.

Scope
-----
- Reads a CSV file where the first column is an email address, or a column named `email`.
- Outputs domain counts, one per line, as `count<TAB>domain` on stdout.
- Domains are lowercased before counting.
- Rows missing the `@` character are skipped.

Usage
-----
```
python3 domain_count.py users.csv
```

Notes
-----
- Uses only the standard library `csv` module.
- Sorting is by count descending, then domain ascending for ties.
- Includes basic tests and a small README.
