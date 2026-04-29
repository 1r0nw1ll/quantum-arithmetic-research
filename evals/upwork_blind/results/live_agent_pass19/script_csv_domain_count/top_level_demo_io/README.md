CSV Email-Domain Counter
========================

Scope
-----
Small Python CLI that counts distinct email domains from a CSV file.
The first column (or a column named email) should contain email addresses.

Usage
-----
- CLI: python3 domain_count.py <path/to/users.csv>
- Output: lines of `count<TAB>domain` sorted by count (desc) then domain (asc).

Notes
-----
- Rows without an @ are skipped.
- Domains are lowercased before counting.
- The CSV reader uses the standard library `csv` module.

Demo
----
Upon import, the module runs a tiny demo against `data/example.csv` (a small sample).
