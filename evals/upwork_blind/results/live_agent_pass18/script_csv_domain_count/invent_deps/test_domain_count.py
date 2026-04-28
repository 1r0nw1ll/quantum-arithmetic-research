from pathlib import Path

from domain_count import count_domains


def write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_counts_email_column_and_lowercases_domains(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "users.csv",
        "\n".join(
            [
                "email,name",
                "Alice@Example.com,Alice",
                "bob@example.com,Bob",
                "carol@test.com,Carol",
                "dave@TEST.com,Dave",
            ]
        )
        + "\n",
    )

    assert count_domains(csv_path) == [("example.com", 2), ("test.com", 2)]


def test_skips_rows_without_valid_email_in_first_column(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "users.csv",
        "\n".join(
            [
                "alice@example.com,ok",
                "missing-at-symbol,bad",
                "@missing-local,bad",
                "missing-domain@,bad",
                "bob@test.com,ok",
                "carol@example.com,ok",
            ]
        )
        + "\n",
    )

    assert count_domains(csv_path) == [("example.com", 2), ("test.com", 1)]
