import subprocess
import sys
from pathlib import Path


def run_cli(tmp_path, csv_text):
    csv_path = tmp_path / "users.csv"
    csv_path.write_text(csv_text, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "domain_count.py", str(csv_path)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_basic_counting_with_email_header(tmp_path):
    output = run_cli(
        tmp_path,
        "email,name\nAlice@Example.com,Alice\nbob@test.com,Bob\ncarol@example.com,Carol\n",
    )
    assert output == "2\texample.com\n1\ttest.com\n"


def test_skips_malformed_rows(tmp_path):
    output = run_cli(
        tmp_path,
        "not-an-email\nuser@example.com\nmissing-at-sign\nsecond@EXAMPLE.com\n",
    )
    assert output == "2\texample.com\n"
