#!/usr/bin/env python3
import sys, csv
from collections import Counter

def main(path):
    rows = list(csv.reader(open(path, newline='', encoding='utf-8')))
    header = rows[0] if rows else []
    email_idx = next((i for i,v in enumerate(header) if str(v).lower()=='email'), 0) if rows else 0
    data = rows[1:] if header and any(str(v).lower()=='email' for v in header) else rows
    domains = Counter((row[email_idx].lower().rsplit('@',1)[-1] for row in data if len(row) > email_idx and '@' in row[email_idx]))
    for dom, cnt in sorted(domains.items(), key=lambda t: (-t[1], t[0])):
        print(f"{cnt}\t{dom}")

if __name__ == '__main__':
    main(sys.argv[1])
