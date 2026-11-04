"""Tests for EXPERIENCES_LOG session records (``python_magnetrun.analysis.sessions``)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from python_magnetrun.analysis.loaders import FileDiscovery, FileSet
from python_magnetrun.analysis.sessions import (
    Session,
    discover_sessions,
    find_unlogged,
    load_sessions,
    parse_hlog_timestamp,
    sessions_to_records,
)

_HEADER = (
    "timestamp,UserCode,UserName,OperatorName,Magnet,ElecGroup,HStart,HStop,"
    "MinuteCount,Paimant,Pmax,Phydraulique,Ptotale,dDepassement,nDepassement,"
    "Cout,Qtotale,QCout\n"
)


def _hlog(dt: datetime) -> str:
    """Format like EXPERIENCES_LOG: 24-hour clock with a spurious AM/PM marker."""
    marker = "PM" if dt.hour >= 12 else "AM"
    return f"{dt.month}/{dt.day}/{dt:%y}  {dt:%H:%M:%S}.{dt.microsecond // 1000:03d} {marker}"


def _row(
    user: str,
    magnet: str,
    group: int,
    hstart: datetime,
    hstop: datetime | None,
    minutes: int | None = None,
    paimant: float | None = 1.0,
) -> str:
    stop = _hlog(hstop) if hstop else ""
    mc = "" if minutes is None else str(minutes)
    pa = "" if paimant is None else str(paimant)
    return f"{hstart:%Y-%m-%d %H:%M:%S},{user},,,{magnet},{group},{_hlog(hstart)},{stop},{mc},{pa},0.0,,{pa},,,,,\n"


def _write_log(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "EXPERIENCES_LOG.csv"
    path.write_text(_HEADER + "".join(rows))
    return path


D = datetime  # shorthand


# ---------------------------------------------------------------------------
# parse_hlog_timestamp
# ---------------------------------------------------------------------------


class TestParseHlogTimestamp:
    def test_spurious_pm_marker(self) -> None:
        assert parse_hlog_timestamp("8/31/06  14:50:56.078 PM") == D(2006, 8, 31, 14, 50, 56, 78000)

    def test_am_marker(self) -> None:
        assert parse_hlog_timestamp("3/1/16  01:03:30.687 AM") == D(2016, 3, 1, 1, 3, 30, 687000)

    def test_blank_is_none(self) -> None:
        assert parse_hlog_timestamp("") is None
        assert parse_hlog_timestamp(None) is None


# ---------------------------------------------------------------------------
# load_sessions
# ---------------------------------------------------------------------------


class TestLoadSessions:
    def test_exploit_variants_are_one_operator(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("EXPLOIT.", "M9i", 1, D(2025, 6, 1, 10, 0), D(2025, 6, 1, 11, 0)),
            _row("EXPLOIT", "M9e", 2, D(2025, 6, 1, 10, 0, 5), D(2025, 6, 1, 11, 0)),
        ])
        sessions = load_sessions(log, "M9", D(2025, 1, 1))
        assert len(sessions) == 1
        assert sessions[0].users == ["EXPLOIT"]
        assert sessions[0].operator is True

    def test_user_session_not_operator(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GSC06-124", "M9i", 1, D(2025, 6, 1, 10, 0), D(2025, 6, 1, 11, 0)),
            _row("EXPLOIT.", "M9e", 2, D(2025, 6, 1, 10, 0), D(2025, 6, 1, 11, 0)),
        ])
        (s,) = load_sessions(log, "M9", D(2025, 1, 1))
        assert s.users == ["EXPLOIT", "GSC06-124"]
        assert s.operator is False

    def test_exact_duplicates_removed(self, tmp_path: Path) -> None:
        row = _row("GSC06-124", "M9i", 1, D(2025, 6, 1, 10, 0), D(2025, 6, 1, 11, 0), 60)
        log = _write_log(tmp_path, [row, row])
        (s,) = load_sessions(log, "M9", D(2025, 1, 1))
        assert len(s.coils) == 1

    def test_unclosed_superseded_within_10_min_dropped(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("EXPLOIT.", "M9i", 1, D(2025, 6, 3, 14, 2, 25), None),
            _row("EXPLOIT.", "M9i", 1, D(2025, 6, 3, 14, 3, 9), D(2025, 6, 3, 14, 32, 37), 29),
        ])
        (s,) = load_sessions(log, "M9", D(2025, 1, 1))
        assert s.start == D(2025, 6, 3, 14, 3, 9)
        assert s.stop_capped is False

    def test_unclosed_capped_at_next_start_on_same_coil(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 10, 0), None),
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 15, 0), D(2025, 6, 3, 16, 0)),
        ])
        first, second = load_sessions(log, "M9", D(2025, 1, 1))
        assert first.stop == D(2025, 6, 3, 15, 0)
        assert first.stop_capped is True
        assert second.start == D(2025, 6, 3, 15, 0)

    def test_unclosed_capped_at_12_hours(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 10, 0), None),
            _row("GSC06-124", "M9i", 1, D(2025, 6, 5, 10, 0), D(2025, 6, 5, 11, 0)),
        ])
        first, _ = load_sessions(log, "M9", D(2025, 1, 1))
        assert first.stop == D(2025, 6, 3, 22, 0)
        assert first.stop_capped is True

    def test_unclosed_last_row_capped_at_12_hours(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [_row("GSC06-124", "M9i", 1, D(2025, 6, 3, 10, 0), None)])
        (s,) = load_sessions(log, "M9", D(2025, 1, 1))
        assert s.stop == D(2025, 6, 3, 22, 0)

    def test_overlapping_coils_merged(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 18, 24, 27), D(2025, 6, 3, 22, 51, 33), 263, 23.8),
            _row("GSC06-124", "M9e", 2, D(2025, 6, 3, 18, 24, 33), D(2025, 6, 3, 18, 25, 29), 1, 0.0),
            _row("GSC06-124", "M9e", 2, D(2025, 6, 3, 18, 25, 50), D(2025, 6, 3, 22, 51, 40), 263, 11.0),
        ])
        (s,) = load_sessions(log, "M9", D(2025, 1, 1))
        assert s.start == D(2025, 6, 3, 18, 24, 27)
        assert s.stop == D(2025, 6, 3, 22, 51, 40)
        assert [c["coil"] for c in s.coils] == ["M9i", "M9e", "M9e"]
        assert s.coils[0]["elec_group"] == 1
        assert s.coils[0]["minute_count"] == 263
        assert s.coils[0]["paimant"] == pytest.approx(23.8)

    def test_disjoint_windows_not_merged(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 10, 0), D(2025, 6, 3, 11, 0)),
            _row("GSC06-124", "M9i", 1, D(2025, 6, 3, 11, 30), D(2025, 6, 3, 12, 0)),
        ])
        assert len(load_sessions(log, "M9", D(2025, 1, 1))) == 2

    def test_housing_and_period_filter(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("A", "M9i", 1, D(2017, 12, 31, 10, 0), D(2017, 12, 31, 11, 0)),
            _row("B", "M9i", 1, D(2018, 1, 2, 10, 0), D(2018, 1, 2, 11, 0)),
            _row("C", "M10e", 1, D(2018, 1, 2, 10, 0), D(2018, 1, 2, 11, 0)),
            _row("D", "M9i", 1, D(2019, 1, 2, 10, 0), D(2019, 1, 2, 11, 0)),
        ])
        sessions = load_sessions(log, "M9", D(2018, 1, 1), D(2019, 1, 1))
        assert [s.users for s in sessions] == [["B"]]

    def test_long_session_kept(self, tmp_path: Path) -> None:
        log = _write_log(tmp_path, [
            _row("GMS06-219", "M9i", 1, D(2020, 12, 19, 10, 2), D(2020, 12, 20, 15, 5), 1692),
        ])
        (s,) = load_sessions(log, "M9", D(2020, 1, 1))
        assert s.stop == D(2020, 12, 20, 15, 5)
        assert s.stop_capped is False

    def test_timestamp_column_ignored(self, tmp_path: Path) -> None:
        """The CSV ``timestamp`` drifts from HStart since 2020; only HStart counts."""
        row = _row("GSC06-124", "M9i", 1, D(2025, 6, 1, 17, 47, 11), D(2025, 6, 1, 22, 4, 24))
        row = "2025-06-02 01:15:47" + row[len("2025-06-01 17:47:11"):]
        (s,) = load_sessions(_write_log(tmp_path, [row]), "M9", D(2025, 1, 1))
        assert s.start == D(2025, 6, 1, 17, 47, 11)


# ---------------------------------------------------------------------------
# FileDiscovery.discover_window and the session passes (file system faked)
# ---------------------------------------------------------------------------

# basename -> (start, end) of each fake file
_RANGES: dict[str, tuple[str, str]] = {}


def _fake_extract(file: str, housing: str, *args, **kwargs) -> tuple[str, str, bool]:
    start, end = _RANGES[Path(file).name]
    return start, end, False


def _make_tree(tmp_path: Path, files: dict[str, tuple[str, str]]) -> FileDiscovery:
    """Create empty files under a pigbrother/pupitre tree and register their ranges."""
    _RANGES.clear()
    pb, pup = tmp_path / "pigbrother" / "M9", tmp_path / "pupitre" / "M9"
    dirs = {
        "Overview": pb / "Overview", "Archive": pb / "Fichiers_Archive",
        "Default": pb / "Fichiers_Default", "Spikes": pb / "Fichiers_Spike",
        "ManuelTrig": pb / "Fichiers_Manuel_Trig", "Stats": pb / "Fichiers_stats",
    }
    for name, rng in files.items():
        directory = pup if name.endswith(".txt") else dirs[name.split("_")[1]]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(b"")
        _RANGES[name] = rng
    return FileDiscovery(
        pupitre_datadir=tmp_path / "pupitre", pigbrother_datadir=tmp_path / "pigbrother"
    )


def _names(paths: list[str]) -> list[str]:
    return [Path(p).name for p in paths]


_WINDOW_FILES = {
    "M9_Archive_250531-2200.tdms": ("2025-05-31 22:00:00", "2025-06-01 03:00:00"),
    "M9_Overview_250601-0040.tdms": ("2025-06-01 00:40:00", "2025-06-01 01:30:00"),
    "M9_Overview_250601-0300.tdms": ("2025-06-01 03:00:00", "2025-06-01 04:00:00"),
    "2025.05.31 - 23:50:00.txt": ("2025-05-31 23:50:00", "2025-06-01 00:45:00"),
    "2025.06.01 - 03:00:00.txt": ("2025-06-01 03:00:00", "2025-06-01 03:30:00"),
    "M9_Default_250601-010000_DefautNums.tdms": ("2025-06-01 01:00:00", "2025-06-01 01:00:01"),
}


class TestDiscoverWindow:
    def test_window_crossing_midnight(self, tmp_path: Path) -> None:
        fd = _make_tree(tmp_path, _WINDOW_FILES)
        with patch("python_magnetrun.utils.files.extract_data", side_effect=_fake_extract):
            fs = fd.discover_window("M9", D(2025, 6, 1, 0, 30), D(2025, 6, 1, 2, 0))
        assert _names(fs.archive) == ["M9_Archive_250531-2200.tdms"]
        assert _names(fs.overview) == ["M9_Overview_250601-0040.tdms"]
        assert _names(fs.pupitre) == ["2025.05.31 - 23:50:00.txt"]
        assert _names(fs.default) == ["M9_Default_250601-010000_DefautNums.tdms"]


class TestDiscoverSessions:
    def test_anchor_and_data_end(self, tmp_path: Path) -> None:
        fd = _make_tree(tmp_path, _WINDOW_FILES)
        sessions = [
            Session(housing="M9", start=D(2025, 6, 1, 0, 30), stop=D(2025, 6, 1, 2, 0)),
            Session(housing="M9", start=D(2025, 6, 1, 3, 10), stop=D(2025, 6, 1, 3, 20)),
            Session(housing="M9", start=D(2025, 6, 1, 5, 0), stop=D(2025, 6, 1, 6, 0)),
        ]
        with patch("python_magnetrun.utils.files.extract_data", side_effect=_fake_extract), \
                patch("python_magnetrun.analysis.sessions.extract_data", side_effect=_fake_extract):
            discover_sessions(sessions, fd)
        assert sessions[0].anchor == "overview"
        assert sessions[0].data_end == D(2025, 6, 1, 3, 0)
        assert sessions[1].anchor == "overview"
        assert sessions[2].anchor == "none"
        assert sessions[2].data_end is None


_UNLOGGED_FILES = {
    "2025.06.01 - 10:00:00.txt": ("2025-06-01 10:00:00", "2025-06-01 10:20:00"),
    "2025.06.01 - 14:00:00.txt": ("2025-06-01 14:00:00", "2025-06-01 14:30:00"),
    "2025.06.01 - 14:50:00.txt": ("2025-06-01 14:50:00", "2025-06-01 15:10:00"),
    "2025.06.01 - 18:00:00.txt": ("2025-06-01 18:00:00", "2025-06-01 18:10:00"),
    "2025.06.01 - 19:00:00.txt": ("2025-06-01 19:00:00", "2025-06-01 19:00:20"),
}


class TestFindUnlogged:
    def test_unclaimed_files_chained(self, tmp_path: Path) -> None:
        fd = _make_tree(tmp_path, _UNLOGGED_FILES)
        logged = Session(housing="M9", start=D(2025, 6, 1, 10, 0), stop=D(2025, 6, 1, 10, 30))
        with patch("python_magnetrun.utils.files.extract_data", side_effect=_fake_extract), \
                patch("python_magnetrun.analysis.sessions.extract_data", side_effect=_fake_extract):
            discover_sessions([logged], fd)
            unlogged = find_unlogged([logged], fd, "M9", D(2025, 6, 1), D(2025, 6, 2))
        assert [(u.start, u.stop) for u in unlogged] == [
            (D(2025, 6, 1, 14, 0), D(2025, 6, 1, 15, 10)),
            (D(2025, 6, 1, 18, 0), D(2025, 6, 1, 18, 10)),
        ]
        assert all(u.logged is False for u in unlogged)
        assert _names(unlogged[0].files.pupitre) == [
            "2025.06.01 - 14:00:00.txt", "2025.06.01 - 14:50:00.txt",
        ]
        assert unlogged[0].anchor == "pupitre"


# ---------------------------------------------------------------------------
# sessions_to_records
# ---------------------------------------------------------------------------


class TestSessionsToRecords:
    def test_record_fields(self) -> None:
        s = Session(
            housing="M9", start=D(2025, 6, 1, 17, 47, 11), stop=D(2025, 6, 1, 22, 4, 25),
            users=["GMS12-224"],
            coils=[{"coil": "M9i", "elec_group": 1, "start": D(2025, 6, 1, 17, 47, 11),
                    "stop": D(2025, 6, 1, 22, 4, 24), "minute_count": 253, "paimant": 13.9}],
            files=FileSet(pupitre=["/p/2025.06.01 - 17:49:00.txt"]), anchor="pupitre",
            data_end=D(2025, 6, 1, 22, 0),
        )
        (r,) = sessions_to_records([s])
        assert r["id"] == "M9_20250601-174711"
        assert r["start"] == "2025-06-01 17:47:11"
        assert r["data_end"] == "2025-06-01 22:00:00"
        assert r["logged"] is True
        assert r["pupitre"] == ["/p/2025.06.01 - 17:49:00.txt"]
        assert r["overview"] == []
        assert r["coils"][0]["stop"] == "2025-06-01 22:04:24"
