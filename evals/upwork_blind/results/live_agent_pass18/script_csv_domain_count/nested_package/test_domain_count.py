import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from deliverable.lib.core.v1 import count_domains_from_csv


ROOT = Path(__file__).resolve().parent


class DomainCountTests(unittest.TestCase):
    def test_counts_email_column_and_skips_malformed_rows(self) -> None:
        csv_text = textwrap.dedent(
            """\
            name,email
            Alice,Alice@Example.com
            Bob,invalid-address
            Carol,carol@test.org
            Dave,dave@example.com
            Eve,
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "users.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            result = count_domains_from_csv(csv_path)

        self.assertEqual(result, [("example.com", 2), ("test.org", 1)])

    def test_cli_uses_first_column_when_email_column_missing(self) -> None:
        csv_text = textwrap.dedent(
            """\
            first,second
            user@beta.com,ignored
            broken-value,ignored
            admin@alpha.com,ignored
            other@beta.com,ignored
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "users.csv"
            csv_path.write_text(csv_text, encoding="utf-8")

            completed = subprocess.run(
                ["python3", str(ROOT / "domain_count.py"), str(csv_path)],
                check=True,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )

        self.assertEqual(completed.stdout.strip().splitlines(), ["2\tbeta.com", "1\talpha.com"])


if __name__ == "__main__":
    unittest.main()
