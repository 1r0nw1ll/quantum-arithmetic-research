import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "domain_count.py"


def load_domain_count_module():
    spec = importlib.util.spec_from_file_location("domain_count_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


class DomainCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_domain_count_module()

    def write_csv(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "input.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_counts_domains_from_email_header(self):
        path = self.write_csv(
            "name,email\n"
            "Alice,Alice@Example.com\n"
            "Bob,bob@example.com\n"
            "Carol,carol@sample.org\n"
        )

        counts = self.module.count_domains(path)

        self.assertEqual(counts["example.com"], 2)
        self.assertEqual(counts["sample.org"], 1)

    def test_uses_first_column_and_skips_malformed_rows(self):
        path = self.write_csv(
            "Alice@Example.com,alice\n"
            "missing-at-symbol,skip-me\n"
            "bob@sample.org,bob\n"
            ",blank\n"
            "charlie@EXAMPLE.com,charlie\n"
        )

        counts = self.module.count_domains(path)

        self.assertEqual(
            self.module.format_counts(counts),
            ["2\texample.com", "1\tsample.org"],
        )


if __name__ == "__main__":
    unittest.main()
