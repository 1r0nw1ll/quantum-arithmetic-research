import csv
import tempfile
import os
from domain_count import count_domains


def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['email'])
        writer.writeheader()
        for r in rows:
            writer.writerow({'email': r})
    return path


def test_basic_counts():
    path = _write_csv(['a@foo.com', 'b@foo.com', 'c@bar.org'])
    c = count_domains(path)
    assert c['foo.com'] == 2
    assert c['bar.org'] == 1


def test_skips_malformed():
    path = _write_csv(['noatsign', '', 'a@foo.com'])
    c = count_domains(path)
    assert sum(c.values()) == 1
