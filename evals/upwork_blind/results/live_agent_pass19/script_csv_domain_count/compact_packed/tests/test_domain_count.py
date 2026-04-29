import sys
import subprocess
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(csv_content, tmp_path):
    p = tmp_path / "users.csv"
    p.write_text(csv_content, encoding='utf-8')
    out = subprocess.run([PY, str(ROOT / 'domain_count.py'), str(p)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return out.stdout


def test_basic_counting(tmp_path):
    csv = textwrap.dedent("""\
    email
    alice@example.com
    bob@example.com
    charlie@domain.org
    alice@example.com
    malformed
    """)
    assert run(csv, tmp_path) == "2\texample.com\n1\tdomain.org\n"


def test_malformed_rows_skipped(tmp_path):
    csv = textwrap.dedent("""\
    email
    not-an-email
    bob@domain.com
    """)
    assert run(csv, tmp_path) == "1\tdomain.com\n"


def test_no_header_uses_first_column(tmp_path):
    csv = textwrap.dedent("""\
    alice@example.com
    bob@domain.net
    """)
    assert run(csv, tmp_path) == "1\tdomain.net\n1\texample.com\n"
