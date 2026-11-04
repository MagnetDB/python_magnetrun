"""Tests for the end timestamps returned by ``extract_data`` (metadata-only path).

Covers:
- Archive TDMS files: end taken from the file modification time when it is
  plausible, otherwise from the TDMS properties
- Overview TDMS files: always from the TDMS properties (their modification
  time is unreliable)
- Pupitre files whose last line is in a corrupt tail (stale rows from an
  older run, or binary data): end taken from the valid rows instead
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from python_magnetrun.utils.files import TIMESTAMP_FORMAT, extract_data, select_files

_PROPS = "python_magnetrun.utils.files._tdms_end_from_properties"


def _touch(path: Path, mtime: datetime) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


# ---------------------------------------------------------------------------
# Archive / Overview TDMS
# ---------------------------------------------------------------------------


class TestTdmsEndTime:
    def test_archive_uses_plausible_mtime(self, tmp_path: Path) -> None:
        f = _touch(
            tmp_path / "Fichiers_Archive" / "M9_Archive_250601-1749.tdms",
            datetime(2025, 6, 1, 22, 5, 33),
        )
        with patch(_PROPS) as props:
            start, end, skip = extract_data(str(f), "M9")
        props.assert_not_called()
        assert start == "2025-06-01 17:49:00"
        assert end == "2025-06-01 22:05:33"
        assert skip is False

    def test_archive_mtime_before_start_falls_back(self, tmp_path: Path) -> None:
        f = _touch(
            tmp_path / "Fichiers_Archive" / "M9_Archive_250601-1749.tdms",
            datetime(2025, 6, 1, 17, 0, 0),
        )
        with patch(_PROPS, return_value="2025-06-01 22:05:00") as props:
            _, end, _ = extract_data(str(f), "M9")
        props.assert_called_once()
        assert end == "2025-06-01 22:05:00"

    def test_archive_mtime_more_than_a_day_late_falls_back(self, tmp_path: Path) -> None:
        f = _touch(
            tmp_path / "Fichiers_Archive" / "M9_Archive_250601-1749.tdms",
            datetime(2025, 6, 3, 9, 0, 0),
        )
        with patch(_PROPS, return_value="2025-06-01 22:05:00") as props:
            _, end, _ = extract_data(str(f), "M9")
        props.assert_called_once()
        assert end == "2025-06-01 22:05:00"

    def test_overview_ignores_mtime(self, tmp_path: Path) -> None:
        f = _touch(
            tmp_path / "Overview" / "M9_Overview_250601-1749.tdms",
            datetime(2025, 6, 1, 22, 5, 33),
        )
        with patch(_PROPS, return_value="2025-06-01 20:00:00") as props:
            _, end, _ = extract_data(str(f), "M9")
        props.assert_called_once()
        assert end == "2025-06-01 20:00:00"


# ---------------------------------------------------------------------------
# Pupitre files with a corrupt tail
# ---------------------------------------------------------------------------

# Same layout as the stale-tail fixtures of test_truncated_pupitre.py.
_TAB_PARAMS = "1\tM1\t100\t100\t0.1\t0\t0\t0.0035\n"
_TAB_HEADER = "Date\tTime\tField\tIcoil1\t\n"


def _tab_rows(date: str, hour: int, n: int) -> str:
    return "".join(
        f"{date}\t{hour:02d}:{i // 60:02d}:{i % 60:02d}\t{20.0 + i / 1000}\t1000.0\t\n"
        for i in range(n)
    )


class TestPupitreEndTime:
    def test_clean_file_uses_last_line(self, tmp_path: Path) -> None:
        f = tmp_path / "2024.12.05 - 12:00:00.txt"
        f.write_text(_TAB_PARAMS + _TAB_HEADER + _tab_rows("2024.12.05", 12, 120))
        _, end, _ = extract_data(str(f), "M10")
        assert end == "2024-12-05 12:01:59"

    def test_stale_tail_uses_valid_rows(self, tmp_path: Path) -> None:
        """Last line comes from an older run: end must not be before start."""
        f = tmp_path / "2024.12.05 - 12:00:00.txt"
        body = (
            _tab_rows("2024.12.05", 12, 120)
            + "\x00" * 64
            + ".0\t20897.5\t182.4\n"
            + "2014.07.27\t22:38:31\t22.2053\n"
            + "2014.07.27\t22:38:32\t22.2091"
        )
        f.write_bytes((_TAB_PARAMS + _TAB_HEADER + body).encode("utf-8"))
        start, end, _ = extract_data(str(f), "M10")
        assert start == "2024-12-05 12:00:00"
        assert end == "2024-12-05 12:01:59"

    def test_binary_tail_uses_valid_rows(self, tmp_path: Path) -> None:
        """Last line is binary junk that cannot be parsed."""
        f = tmp_path / "2024.12.05 - 12:00:00.txt"
        binary = b"\x00" * 64 + b"o\x9e\xf7\xefi\xaa9r\t\x9c\x8d+D6\n\xeb]\xde<\xef\x19H\n\x13J\x92"
        f.write_bytes(
            (_TAB_PARAMS + _TAB_HEADER + _tab_rows("2024.12.05", 12, 400)).encode("utf-8") + binary
        )
        _, end, _ = extract_data(str(f), "M10")
        assert end == "2024-12-05 12:06:39"

    def test_stale_tail_file_selected(self, tmp_path: Path) -> None:
        f = tmp_path / "2024.12.05 - 12:00:00.txt"
        body = _tab_rows("2024.12.05", 12, 120) + "2014.07.27\t22:38:31\t22.2053\n"
        f.write_bytes((_TAB_PARAMS + _TAB_HEADER + body).encode("utf-8"))
        selected = select_files(
            [str(f)], "M10", start="2024-12-05 11:00:00", end="2024-12-05 13:00:00"
        )
        assert selected == [str(f)]

    def test_end_is_a_timestamp_string(self, tmp_path: Path) -> None:
        f = tmp_path / "2024.12.05 - 12:00:00.txt"
        f.write_text(_TAB_PARAMS + _TAB_HEADER + _tab_rows("2024.12.05", 12, 60))
        _, end, _ = extract_data(str(f), "M10")
        datetime.strptime(end, TIMESTAMP_FORMAT)
