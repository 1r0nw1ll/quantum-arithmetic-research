import csv
import sys
from collections import Counter


def count_domains(path):
    counts = Counter()
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            email = (row.get('email') or '').strip().lower()
            if '@' not in email:
                continue
            domain = email.split('@', 1)[1]
            if domain:
                counts[domain] += 1
    return counts


def main(argv):
    if len(argv) != 2:
        print('Usage: python3 domain_count.py <path-to-users.csv>', file=sys.stderr)
        return 2
    counts = count_domains(argv[1])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f'{count}\t{domain}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
