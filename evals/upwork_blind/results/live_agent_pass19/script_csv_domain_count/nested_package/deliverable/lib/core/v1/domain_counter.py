import csv
from typing import Dict


def count_domains_from_csv(filepath: str) -> Dict[str, int]:
    """
    Count email domains from a CSV file.

    Rules:
    - If the CSV has a header with a column named 'email' (case-insensitive), use that column.
    - Otherwise, fall back to using the first column.
    - Skip rows where the value doesn't contain an '@'.
    - Domains are lowercased for counting.
    - Returns a mapping domain -> count.
    """
    counts: Dict[str, int] = {}

    # First attempt: use DictReader and look for an 'email' column
    email_field = None
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames:
                for fn in fieldnames:
                    if isinstance(fn, str) and fn.lower() == 'email':
                        email_field = fn
                        break
            if email_field is not None:
                for row in reader:
                    val = row.get(email_field)
                    if not val:
                        continue
                    val = val.strip()
                    if '@' not in val:
                        continue
                    domain = val.rsplit('@', 1)[-1].lower()
                    if domain:
                        counts[domain] = counts.get(domain, 0) + 1
                return counts
    except FileNotFoundError:
        raise
    except Exception:
        # Fall back to second approach if something goes wrong
        pass

    # Fallback: read rows and treat the first column as the email value
    try:
        with open(filepath, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                val = row[0] if len(row) > 0 else ''
                if not isinstance(val, str):
                    continue
                val = val.strip()
                if '@' not in val:
                    continue
                domain = val.rsplit('@', 1)[-1].lower()
                if domain:
                    counts[domain] = counts.get(domain, 0) + 1
    except FileNotFoundError:
        raise

    return counts
