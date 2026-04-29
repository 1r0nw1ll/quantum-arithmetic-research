import os
import sys
import subprocess
import tempfile
import pytest

def _script_path():
    # domain_count.py is located at repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "domain_count.py"))

# Test cases: each entry is (input_csv, expected_output_lines, label, note)
# The test function uses all four values; label/note are advisory metadata.

@pytest.mark.parametrize(
    "input_csv, expected, label, note",
    [
        ("email,name\nalice@example.com,A\nbob@example.com,B\ncharlie@domain.org,C\n",
         ["2\texample.com", "1\tdomain.org"],
         "Header present with multiple domains",
         "Counts domains case-insensitively; example.com appears twice"),
        ("alice@example.com,B\nbob@example.org,C\n",
         ["1\texample.com", "1\texample.org"],
         "No header, first column emails",
         "Ties broken by domain alphabetical order"),
        ("email\nalice@ExAmPlE.CoM\nbob@example.com\n",
         ["2\texample.com"],
         "Header with email column; case-insensitive domain",
         "Domains counted as lower-case"),
        ("email,name\nnot-an-email,foo\nvalid@domain.tld,Bar\n",
         ["1\tdomain.tld"],
         "Skip malformed rows",
         "Only valid emails counted"),
        ("name,addr\nalice@example.org,addr1\ncarol@example.org,addr2\n",
         ["2\texample.org"],
         "Emails from same domain without header on first column",
         "Counts domain.org twice"),
    ],
)

def test_domain_count_cli(input_csv: str, expected, label: str, note: str):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8") as tf:
        tf.write(input_csv)
        tf.flush()
        csv_path = tf.name

    script_path = _script_path()
    result = subprocess.run([sys.executable, script_path, csv_path], capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    output_lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert output_lines == expected, f"{label}: expected {expected}, got {output_lines}. Note: {note}"
    os.unlink(csv_path)
