"""Unit tests for python_magnetrun.readers.csv_readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from python_magnetrun.readers.base import Reader
from python_magnetrun.readers.csv_readers import (
    BProfileReader,
    CsvReader,
    EnsightReader,
    FeelppReader,
    PupitreReader,
)

DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_PUPITRE = DATA_DIR / "sample_pupitre.txt"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reader_cls",
    [PupitreReader, BProfileReader, EnsightReader, CsvReader],
)
def test_reader_satisfies_protocol(reader_cls):
    """Each reader must satisfy the Reader protocol."""
    assert isinstance(reader_cls(), Reader)


def test_feelpp_reader_satisfies_protocol():
    assert isinstance(FeelppReader(skip_rows=0), Reader)


# ---------------------------------------------------------------------------
# PupitreReader
# ---------------------------------------------------------------------------


class TestPupitreReader:
    def setup_method(self):
        self.reader = PupitreReader()

    def test_read_kwargs_keys(self):
        kw = self.reader.read_kwargs()
        assert "sep" in kw
        assert "engine" in kw
        assert "skiprows" in kw
        assert "on_bad_lines" in kw

    def test_read_kwargs_values(self):
        kw = self.reader.read_kwargs()
        assert kw["skiprows"] == 1
        assert kw["sep"] == r"\s+"
        assert kw["engine"] == "python"

    def test_validate_ok(self):
        assert self.reader.validate(SAMPLE_PUPITRE) is True

    def test_read_stub_returns_one_row(self):
        df = self.reader.read_stub(SAMPLE_PUPITRE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_read_returns_dataframe(self):
        df = self.reader.read(SAMPLE_PUPITRE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_read_stub_columns_match_read(self):
        stub = self.reader.read_stub(SAMPLE_PUPITRE)
        full = self.reader.read(SAMPLE_PUPITRE)
        assert list(stub.columns) == list(full.columns)

    def test_validate_missing_file(self):
        from python_magnetrun.utils.validation import FileFormatError

        with pytest.raises((FileNotFoundError, FileFormatError, OSError)):
            self.reader.validate(Path("/nonexistent/file.txt"))

    def test_defs_file_attribute(self):
        assert self.reader.defs_file == "pupitre-defs.json"


# ---------------------------------------------------------------------------
# BProfileReader
# ---------------------------------------------------------------------------


class TestBProfileReader:
    def setup_method(self):
        self.reader = BProfileReader()

    def test_read_kwargs_keys(self):
        kw = self.reader.read_kwargs()
        assert "sep" in kw
        assert "engine" in kw
        assert "skiprows" in kw

    def test_skip_rows_is_zero(self):
        assert self.reader.skip_rows == 0

    def test_defs_file_is_none(self):
        assert self.reader.defs_file is None

    def test_expected_cols(self):
        assert "Index" in self.reader.expected_cols
        assert "Position" in self.reader.expected_cols
        assert "Profile" in self.reader.expected_cols


# ---------------------------------------------------------------------------
# EnsightReader
# ---------------------------------------------------------------------------


class TestEnsightReader:
    def setup_method(self):
        self.reader = EnsightReader()

    def test_skip_rows_is_two(self):
        assert self.reader.skip_rows == 2

    def test_sep_is_comma(self):
        assert self.reader.sep == ","

    def test_read_kwargs_keys(self):
        kw = self.reader.read_kwargs()
        assert kw["skiprows"] == 2
        assert kw["sep"] == ","


# ---------------------------------------------------------------------------
# FeelppReader
# ---------------------------------------------------------------------------


class TestFeelppReader:
    def test_default_skip_rows_zero(self):
        r = FeelppReader()
        assert r.skip_rows == 0

    def test_custom_skip_rows(self):
        r = FeelppReader(skip_rows=3)
        assert r.skip_rows == 3
        assert r.read_kwargs()["skiprows"] == 3

    def test_sep_is_comma(self):
        assert FeelppReader().sep == ","

    def test_defs_file(self):
        assert FeelppReader().defs_file == "feelpp-defs.json"


# ---------------------------------------------------------------------------
# CsvReader
# ---------------------------------------------------------------------------


class TestCsvReader:
    def setup_method(self):
        self.reader = CsvReader()

    def test_sep_comma(self):
        assert self.reader.sep == ","

    def test_skip_rows_zero(self):
        assert self.reader.skip_rows == 0

    def test_read_writes_dataframe(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5,6\n")
        df = self.reader.read(csv_file)
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2

    def test_validate_ok(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n")
        assert self.reader.validate(csv_file) is True
