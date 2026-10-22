"""Tests for resilient handling of broken/truncated pupitre .txt files.

Covers:
- fromtxt with a mid-line truncated file (bad last row) — loads partial data, no exception
- fromtxt with a file truncated between lines — loads normally
- fromtxt with Latin-1 encoding — no UnicodeDecodeError
- fromtxt with header only, no data rows — raises FileFormatError
- _pupitre_end_from_last_line on a truncated file — returns "", no exception
- select_files with a Latin-1 file — skips cleanly
- fromtxt (pandas and polars) on a file whose tail holds stale disk data
  (NUL padding, then rows from an older run) — tail rows dropped and reported
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from python_magnetrun.magnetdata_pandas import PandasMagnetData
from python_magnetrun.magnetdata_polars import PolarsMagnetData
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


# ---------------------------------------------------------------------------
# Stale disk data after a crash (tab-separated, like real pupitre files)
# ---------------------------------------------------------------------------

# Modelled on M10 "2024.12.05 - 11:16:02.txt": the acquisition stopped at
# 12:15:32, the rest of the file is NUL padding followed by rows copied from
# an older M10 run (2014.07.27), with fewer fields and no final newline.
_TAB_PARAMS = "1\tM1\t100\t100\t0.1\t0\t0\t0.0035\n"
_TAB_HEADER = "Date\tTime\tField\tIcoil1\t\n"


def _tab_row(date: str, time: str, field: float) -> str:
    return f"{date}\t{time}\t{field}\t1000.0\t\n"


_GOOD_TAB_ROWS = [
    _tab_row("2024.12.05", "12:15:30", 28.7457),
    _tab_row("2024.12.05", "12:15:31", 28.7517),
    _tab_row("2024.12.05", "12:15:32", 28.7578),
]
_NUL_LINE = "\x00" * 64 + ".0\t20897.5\t182.4\n"
_OLD_ROWS = "2014.07.27\t22:38:31\t22.2053\n" "2014.07.27\t22:38:32\t22.2091"

_BACKENDS = pytest.mark.parametrize(
    "backend", [PandasMagnetData, PolarsMagnetData], ids=["pandas", "polars"]
)


def _write_tab(path: Path, body: str) -> None:
    path.write_bytes((_TAB_PARAMS + _TAB_HEADER + body).encode("utf-8"))


class TestStaleDiskTail:
    @_BACKENDS
    def test_nul_line_and_old_rows_dropped(self, tmp_path: Path, backend) -> None:
        f = tmp_path / "2024.12.05 - 12:15:30.txt"
        _write_tab(f, "".join(_GOOD_TAB_ROWS) + _NUL_LINE + _OLD_ROWS)
        mdata = backend.fromtxt(str(f))
        assert len(mdata.Data) == len(_GOOD_TAB_ROWS)
        assert float(mdata.Data["Field"].max()) == pytest.approx(28.7578)
        assert mdata.rows_dropped == 3
        assert mdata.truncated is True

    @_BACKENDS
    def test_old_rows_without_nul_line_dropped(self, tmp_path: Path, backend) -> None:
        """A backward jump of more than 1 h cuts the file even without a broken line."""
        f = tmp_path / "2024.12.05 - 12:15:30.txt"
        _write_tab(f, "".join(_GOOD_TAB_ROWS) + _OLD_ROWS + "\n")
        mdata = backend.fromtxt(str(f))
        assert len(mdata.Data) == len(_GOOD_TAB_ROWS)
        assert mdata.rows_dropped == 2
        assert mdata.truncated is True

    @_BACKENDS
    def test_dst_end_step_kept(self, tmp_path: Path, backend) -> None:
        """Clock going back 1 h at the end of summer time is not corruption.

        Real case: M9 "2018.10.27 - 18:12:19.txt" steps from 02:59:59 back
        to 02:00:00 (-3599 s).
        """
        f = tmp_path / "2018.10.28 - 01:00:00.txt"
        rows = [
            _tab_row("2018.10.28", "01:00:00", 10.0),
            _tab_row("2018.10.28", "02:59:59", 10.0),
            _tab_row("2018.10.28", "02:00:00", 10.0),
            _tab_row("2018.10.28", "02:00:01", 10.0),
        ]
        _write_tab(f, "".join(rows))
        mdata = backend.fromtxt(str(f))
        assert len(mdata.Data) == len(rows)
        assert mdata.rows_dropped == 0
        assert mdata.truncated is False

    @_BACKENDS
    def test_clean_file_not_truncated(self, tmp_path: Path, backend) -> None:
        f = tmp_path / "2024.12.05 - 12:15:30.txt"
        _write_tab(f, "".join(_GOOD_TAB_ROWS))
        mdata = backend.fromtxt(str(f))
        assert len(mdata.Data) == len(_GOOD_TAB_ROWS)
        assert mdata.rows_dropped == 0
        assert mdata.truncated is False

    @_BACKENDS
    def test_missing_final_newline_is_truncated(self, tmp_path: Path, backend) -> None:
        """No row dropped, but the tail check still flags the file."""
        f = tmp_path / "2024.12.05 - 12:15:30.txt"
        _write_tab(f, "".join(_GOOD_TAB_ROWS).rstrip("\n"))
        mdata = backend.fromtxt(str(f))
        assert mdata.rows_dropped == 0
        assert mdata.truncated is True
