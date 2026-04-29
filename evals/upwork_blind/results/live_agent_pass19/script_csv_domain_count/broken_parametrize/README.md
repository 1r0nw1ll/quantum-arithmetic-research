CSV Email-Domain Counter CLI
=============================

Scope
- A small Python CLI that counts distinct email domains from a CSV file.
- Reads email addresses from the first column or a column named 'email'.
- Normalizes domains to lowercase and ignores malformed rows.
- Outputs lines in the format: count<TAB>domain
- Sorts by count descending, then domain ascending.

Usage
- Run: python3 domain_count.py path/to/file.csv
- If the file has a header with an 'email' column, that column will be used. Otherwise the first column is used.

Notes
- Implemented with the standard library `csv` module.
- Includes a test suite exercising basic counting and malformed-row skipping.
