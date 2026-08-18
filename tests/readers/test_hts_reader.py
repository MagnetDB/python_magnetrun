"""Unit tests for python_magnetrun.readers.hts_reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from python_magnetrun.readers.hts_reader import HtsReader


@pytest.fixture()
def hts_file(tmp_path: Path) -> Path:
    """Synthetic HTS file with unit-bearing column headers."""
    content = (
        "Temps [s];I_H1 [A];U_H1 [V];Field [T]\n"
        "0.0;100.0;2.5;1.2\n"
        "1.0;200.0;3.0;2.4\n"
        "2.0;300.0;3.5;3.6\n"
    )
    f = tmp_path / "run.hts"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture()
def hts_file_no_units(tmp_path: Path) -> Path:
    """HTS file where some columns lack unit suffixes."""
    content = "t;I;U\n0.0;1.0;2.0\n1.0;2.0;3.0\n"
    f = tmp_path / "run_no_units.hts"
    f.write_text(content, encoding="utf-8")
    return f


class TestHtsReader:
    def setup_method(self):
        self.reader = HtsReader()

    def test_sep(self):
        assert self.reader.sep == ";"

    def test_defs_file(self):
        assert self.reader.defs_file == "feelpp-defs.json"

    def test_validate_ok(self, hts_file):
        assert self.reader.validate(hts_file) is True

    def test_validate_missing_file(self):
        with pytest.raises(FileNotFoundError):
            self.reader.validate(Path("/nonexistent/file.hts"))

    def test_validate_wrong_separator(self, tmp_path):
        f = tmp_path / "bad.hts"
        f.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="does not appear to be an HTS file"):
            self.reader.validate(f)

    def test_read_returns_dataframe(self, hts_file):
        df = self.reader.read(hts_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_read_strips_unit_suffix(self, hts_file):
        df = self.reader.read(hts_file)
        assert "Temps" in df.columns
        assert "I_H1" in df.columns
        assert "U_H1" in df.columns
        assert "Field" in df.columns

    def test_read_no_raw_brackets_in_columns(self, hts_file):
        df = self.reader.read(hts_file)
        for col in df.columns:
            assert "[" not in col, f"bracket found in column name: {col!r}"

    def test_extracted_units_returns_dict(self, hts_file):
        units = self.reader.extracted_units(hts_file)
        assert isinstance(units, dict)
        assert units.get("Temps") == "s"
        assert units.get("I_H1") == "A"
        assert units.get("U_H1") == "V"
        assert units.get("Field") == "T"

    def test_extracted_units_no_unit_columns(self, hts_file_no_units):
        units = self.reader.extracted_units(hts_file_no_units)
        for v in units.values():
            assert v == ""

    def test_read_data_values(self, hts_file):
        df = self.reader.read(hts_file)
        assert df["I_H1"].iloc[0] == pytest.approx(100.0)
        assert df["I_H1"].iloc[1] == pytest.approx(200.0)
