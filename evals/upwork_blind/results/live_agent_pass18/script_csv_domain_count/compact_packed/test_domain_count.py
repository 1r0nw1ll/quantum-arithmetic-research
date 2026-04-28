import csv, subprocess, sys, tempfile, unittest
from pathlib import Path


class DomainCountTests(unittest.TestCase):
    def run_cli(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "users.csv"
            with p.open("w", newline="") as f:
                csv.writer(f).writerows(rows)
            return subprocess.run([sys.executable, str(Path(__file__).with_name("domain_count.py")), str(p)], capture_output=True, text=True, check=True).stdout

    def test_counts_first_column(self): self.assertEqual(self.run_cli([["a@x.com"], ["b@x.com"], ["c@y.com"], ["bad"], ["D@Y.COM"]]), "2\tx.com\n2\ty.com\n")

    def test_prefers_email_column_and_skips_malformed(self): self.assertEqual(self.run_cli([["name", "email"], ["Ann", "ann@z.com"], ["Bob", "oops"], ["Cid", "CID@z.com"], ["Dee", "dee@a.com"]]), "2\tz.com\n1\ta.com\n")


if __name__ == "__main__":
    unittest.main()
