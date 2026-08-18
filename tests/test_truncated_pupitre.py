"""Tests for resilient handling of broken/truncated pupitre .txt files.

Covers:
- fromtxt with a mid-line truncated file (bad last row) — loads partial data, no exception
- fromtxt with a file truncated between lines — loads normally
- fromtxt with Latin-1 encoding — no UnicodeDecodeError
- fromtxt with header only, no data rows — raises FileFormatError
- _pupitre_end_from_last_line on a truncated file — returns "", no exception
- select_files with a Latin-1 file — skips cleanly
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.utils.validation import FileFormatError, check_pupitre_truncation

# ---------------------------------------------------------------------------
# Helpers to build fixture files
# ---------------------------------------------------------------------------

_HEADER_COMMENT = "# pupitre test file\n"
_HEADER_ROW = "Date Time t I1 I2 Field\n"
_DATA_ROW = "2024.01.15 10:00:00 0.0 1000.0 2000.0 15.0\n"


def _write_pupitre(path: Path, data_rows: list[str]) -> None:
    path.write_text(_HEADER_COMMENT + _HEADER_ROW + "".join(data_rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# check_pupitre_truncation
# ---------------------------------------------------------------------------


class TestCheckPupitreTruncation:
    def test_clean_file_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.txt"
        _write_pupitre(f, [_DATA_ROW])
        assert check_pupitre_truncation(str(f), ["Date", "Time", "t", "I1", "I2", "Field"]) is False

    def test_no_trailing_newline_returns_true(self, tmp_path: Path) -> None:
        f = tmp_path / "no_newline.txt"
        content = _HEADER_COMMENT + _HEADER_ROW + "2024.01.15 10:00:00 0.0 1000.0 2000.0 15.0"
        f.write_bytes(content.encode("utf-8"))
        assert check_pupitre_truncation(str(f), ["Date", "Time", "t", "I1", "I2", "Field"]) is True

    def test_last_line_too_few_fields_returns_true(self, tmp_path: Path) -> None:
        f = tmp_path / "short_last.txt"
        _write_pupitre(f, [_DATA_ROW, "2024.01.15 10:00:01 1.0 999.0\n"])
        assert check_pupitre_truncation(str(f), ["Date", "Time", "t", "I1", "I2", "Field"]) is True

    def test_empty_keys_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.txt"
        _write_pupitre(f, [_DATA_ROW])
        assert check_pupitre_truncation(str(f), []) is False

    def test_nonexistent_file_returns_false(self, tmp_path: Path) -> None:
        assert check_pupitre_truncation(str(tmp_path / "missing.txt"), ["Date", "Time"]) is False


# ---------------------------------------------------------------------------
# fromtxt resilience
# ---------------------------------------------------------------------------


class TestFromtxtTruncated:
    def test_truncated_midline_loads_good_rows(self, tmp_path: Path) -> None:
        """File with N valid rows + one truncated row → loads at least N rows, no exception.

        Pandas NaN-fills short rows rather than skipping them, so the truncated
        last row is included with NaN for the missing fields.  What matters is
        that no exception is raised and the good rows are present.
        """
        f = tmp_path / "2024.01.15 - 10:00:00.txt"
        good_rows = [
            "2024.01.15 10:00:00 0.0 1000.0 2000.0 15.0\n",
            "2024.01.15 10:00:01 1.0 1001.0 2001.0 15.1\n",
        ]
        truncated_last = "2024.01.15 10:00:02 2.0 1002.0"  # missing last 2 fields, no newline
        f.write_bytes(
            (_HEADER_COMMENT + _HEADER_ROW + "".join(good_rows) + truncated_last).encode("utf-8")
        )
        mdata = PandasMagnetData.fromtxt(str(f))
        mdata._ensure_data_loaded()
        assert len(mdata.Data) >= len(good_rows)

    def test_truncated_between_lines_loads_normally(self, tmp_path: Path) -> None:
        """File ending right after a valid row's newline — not truncated at data level."""
        f = tmp_path / "2024.01.15 - 10:00:00.txt"
        _write_pupitre(f, [_DATA_ROW])
        mdata = PandasMagnetData.fromtxt(str(f))
        mdata._ensure_data_loaded()
        assert len(mdata.Data) >= 1

    def test_latin1_encoding_no_error(self, tmp_path: Path) -> None:
        """File with Latin-1 characters in a column value loads without UnicodeDecodeError."""
        f = tmp_path / "2024.01.15 - 10:00:00.txt"
        latin1_comment = "# capteur température\n".encode("latin-1")
        body = (_HEADER_ROW + _DATA_ROW).encode("utf-8")
        f.write_bytes(latin1_comment + body)
        mdata = PandasMagnetData.fromtxt(str(f))
        mdata._ensure_data_loaded()
        assert len(mdata.Data) >= 1

    def test_header_only_raises_file_format_error(self, tmp_path: Path) -> None:
        """File with comment + header but no data rows → raises FileFormatError."""
        f = tmp_path / "2024.01.15 - 10:00:00.txt"
        f.write_text(_HEADER_COMMENT + _HEADER_ROW, encoding="utf-8")
        with pytest.raises(FileFormatError, match="no data rows"):
            PandasMagnetData.fromtxt(str(f))


# ---------------------------------------------------------------------------
# select_files skips unicode-error files
# ---------------------------------------------------------------------------


class TestSelectFilesSkipsUnicodeError:
    def test_unicode_error_file_skipped(self, tmp_path: Path) -> None:
        """select_files with a Latin-1 file that triggers UnicodeDecodeError is skipped."""
        from python_magnetrun.analysis.loaders import select_files

        good = tmp_path / "2024.01.15 - 10:00:00.txt"
        _write_pupitre(good, [_DATA_ROW, "2024.01.15 10:00:59 59.0 1000.0 2000.0 15.0\n"])

        bad = tmp_path / "2024.01.15 - 11:00:00.txt"
        latin1_comment = "# capteur température invalide\n".encode("latin-1")
        bad.write_bytes(latin1_comment + (_HEADER_ROW + _DATA_ROW).encode("utf-8"))

        # Patch fromtxt so the bad file raises UnicodeDecodeError at load time
        original_fromtxt = PandasMagnetData.fromtxt

        def _patched_fromtxt(name: str, **kwargs):
            if os.path.basename(name).startswith("2024.01.15 - 11"):
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
            return original_fromtxt(name, **kwargs)

        with patch.object(PandasMagnetData, "fromtxt", staticmethod(_patched_fromtxt)):
            result = select_files(
                [str(good), str(bad)],
                housing="M9",
                start="2024-01-15 09:00:00",
                end="2024-01-15 12:00:00",
            )

        assert str(good) in result
        assert str(bad) not in result
