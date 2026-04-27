import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import domain_count


ROOT = Path(__file__).resolve().parent


class DomainCountTests(unittest.TestCase):
    def write_csv(self, rows):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        csv_path = Path(temp_dir.name) / "users.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        return csv_path

    def test_counts_domains_from_email_column(self):
        csv_path = self.write_csv(
            [
                ["name", "email"],
                ["Alice", "Alice@Example.com"],
                ["Bob", "bob@test.com"],
                ["Cara", "cara@example.com"],
            ]
        )

        counts = domain_count.count_domains(str(csv_path))

        self.assertEqual(counts, {"example.com": 2, "test.com": 1})

    def test_skips_malformed_rows_in_first_column_mode(self):
        csv_path = self.write_csv(
            [
                ["alice@example.com", "Alice"],
                ["not-an-email", "Broken"],
                ["BOB@EXAMPLE.COM", "Bob"],
                ["carol@test.com", "Carol"],
                ["@", "EmptyDomain"],
            ]
        )

        result = subprocess.run(
            [sys.executable, str(ROOT / "domain_count.py"), str(csv_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout, "2\texample.com\n1\ttest.com\n")


if __name__ == "__main__":
    unittest.main()
