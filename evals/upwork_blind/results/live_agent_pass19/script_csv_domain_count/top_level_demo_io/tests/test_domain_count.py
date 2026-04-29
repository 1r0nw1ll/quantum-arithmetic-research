import importlib.util
from pathlib import Path

# Lazy load the domain_count module from file, to avoid import-path issues
_MODULE_PATH = Path(__file__).resolve().parents[1] / "domain_count.py"
_spec = importlib.util.spec_from_file_location("domain_count", str(_MODULE_PATH))
_domain_count = importlib.util.module_from_spec(_spec)  # type: ignore
if _spec and _spec.loader:
    _spec.loader.exec_module(_domain_count)  # type: ignore
domain_count = _domain_count  # type: ignore


def _write_csv(tmp_path: Path, rows):
    p = tmp_path / "test.csv"
    with p.open("w", newline='', encoding="utf-8") as f:
        f.write("email\n")
        for r in rows:
            f.write(r + "\n")
    return p


def test_basic_counting_and_sorting(tmp_path):
    rows = [
        "alice@example.com",
        "bob@example.com",
        "carol@other.org",
        "dave@other.org",
        "eve@alpha.com",
        "frank@alpha.com",
    ]
    path = _write_csv(tmp_path, rows)
    result = domain_count.domain_counts(path)
    assert result == [("alpha.com", 2), ("example.com", 2), ("other.org", 2)]


def test_skip_malformed_rows(tmp_path):
    rows = [
        "alice@example.com",
        "not-an-email",
        "carol@example.org",
    ]
    path = _write_csv(tmp_path, rows)
    result = domain_count.domain_counts(path)
    assert result == [("example.com", 1), ("example.org", 1)]
