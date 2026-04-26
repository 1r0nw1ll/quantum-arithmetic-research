import csv
import sys
from collections import Counter


def count_domains(path):
    counts = Counter()
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            email = row.get('email', '')
            if '@' in email:
                domain = email.split('@', 1)[1].lower()
                counts[domain] += 1
    return counts


def main(argv):
    counts = count_domains(argv[1])
    for domain, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f'{count}\t{domain}')


if __name__ == '__main__':
    main(sys.argv)
