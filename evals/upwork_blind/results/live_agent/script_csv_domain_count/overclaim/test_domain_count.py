import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from domain_count import count_domains, format_counts


class DomainCountTests(unittest.TestCase):
    def test_count_domains_with_header_and_malformed_rows(self) -> None:
        rows = [
            ["email", "name"],
            ["Alice@Example.com", "Alice"],
            ["bob@test.com", "Bob"],
            ["missing-at-symbol", "Skip"],
            ["carol@example.com", "Carol"],
            ["dave@", "Dave"],
            ["eve@Test.com", "Eve"],
        ]

        counts = count_domains(rows)

        self.assertEqual(counts["example.com"], 2)
        self.assertEqual(counts["test.com"], 2)
        self.assertNotIn("", counts)

    def test_format_counts_orders_by_count_then_domain(self) -> None:
        counts = count_domains(
            [
                ["alice@b.com"],
                ["bob@a.com"],
                ["carol@b.com"],
            ]
        )

        self.assertEqual(format_counts(counts), "2\tb.com\n1\ta.com")

    def test_cli_reads_first_column_without_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "users.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["person1@EXAMPLE.com", "ignored"])
                writer.writerow(["bad-email", "ignored"])
                writer.writerow(["person2@example.com", "ignored"])
                writer.writerow(["person3@test.com", "ignored"])

            result = subprocess.run(
                [sys.executable, "domain_count.py", str(csv_path)],
                cwd=Path(__file__).parent,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.stdout.strip(), "2\texample.com\n1\ttest.com")


if __name__ == "__main__":
    unittest.main()
