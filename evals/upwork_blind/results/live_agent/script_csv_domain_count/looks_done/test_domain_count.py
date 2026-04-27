import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from domain_count import extract_domains


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "domain_count.py"


class DomainCountTests(unittest.TestCase):
    def write_csv(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "users.csv"
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return path

    def test_counts_email_column_and_skips_malformed_rows(self) -> None:
        path = self.write_csv(
            """\
            email,name
            Alice@Example.com,Alice
            bob@test.com,Bob
            invalid-row,Nope
            carol@example.com,Carol
            dave@TEST.com,Dave
            eve@,Eve
            """
        )

        self.assertEqual(
            extract_domains(str(path)),
            {"example.com": 2, "test.com": 2},
        )

    def test_uses_first_column_when_no_email_header(self) -> None:
        path = self.write_csv(
            """\
            alice@example.com,Alice
            bad-row,Bad
            bob@sample.org,Bob
            """
        )

        self.assertEqual(
            extract_domains(str(path)),
            {"example.com": 1, "sample.org": 1},
        )

    def test_cli_output_is_sorted(self) -> None:
        path = self.write_csv(
            """\
            email
            zed@beta.com
            amy@alpha.com
            bob@beta.com
            """
        )

        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "2\tbeta.com\n1\talpha.com\n")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
