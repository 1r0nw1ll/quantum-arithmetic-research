import os
import sys
import tempfile
import subprocess
import unittest


class DomainCountScriptTests(unittest.TestCase):
    def _run_script(self, csv_text: str):
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.csv', encoding='utf-8') as f:
            f.write(csv_text)
            csv_path = f.name
        try:
            script_path = os.path.abspath('domain_count.py')
            result = subprocess.run([sys.executable, script_path, csv_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        finally:
            try:
                os.remove(csv_path)
            except OSError:
                pass
        return result

    def test_basic_count_and_sorting(self):
        csv = "email\nalice@example.com\nbob@domain.com\nalice@example.com\ncarol@domain.com\n"
        res = self._run_script(csv)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout, "2\tdomain.com\n2	example.com\n")
        
    def test_skip_malformed_row(self):
        csv = "email\nnot-an-email\nuser@valid.com\nbademail\nanother@domain.org\n"
        res = self._run_script(csv)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout, "1\tdomain.org\n1\tvalid.com\n")

    def test_header_case_insensitive_email_column(self):
        csv = "Email\nuser@Example.CoM\nother@example.com\n"
        res = self._run_script(csv)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(res.stdout, "2\texample.com\n")


if __name__ == '__main__':
    unittest.main()
