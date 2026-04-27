import subprocess
import sys
from pathlib import Path


def test_basic_counting_and_malformed_rows(tmp_path: Path) -> None:
    csv_file = tmp_path / "users.csv"
    csv_file.write_text(
        "email,name\n"
        "Alice@Example.com,Alice\n"
        "bad-row,Bad\n"
        "bob@test.com,Bob\n"
        "charlie@example.com,Charlie\n"
        ",Blank\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "domain_count.py", str(csv_file)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "2\texample.com\n1\ttest.com\n"
