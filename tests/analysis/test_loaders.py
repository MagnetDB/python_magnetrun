"""
Tests for the loaders module.

These tests verify the file loading and discovery functionality.
Some tests use mocking since actual data files may not be available.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from python_magnetrun.analysis.loaders import (
    # Constants
    TIMESTAMP_FORMAT,
    # Classes
    FileDiscovery,
    # Dataclasses
    FileMetadata,
    FileSet,
    # Functions
    convert_to_timestamp,
    find_files,
    merge_data,
)


class TestTimestampFormat:
    """Test timestamp format constant."""

    def test_format_string(self):
        """Format should be standard datetime format."""
        assert TIMESTAMP_FORMAT == "%Y-%m-%d %H:%M:%S"

    def test_format_works(self):
        """Format should work with datetime."""
        dt = datetime(2024, 11, 6, 16, 43, 0)
        formatted = dt.strftime(TIMESTAMP_FORMAT)
        assert formatted == "2024-11-06 16:43:00"


class TestConvertToTimestamp:
    """Test convert_to_timestamp function."""

    def test_default_format_tdms(self):
        """Test default TDMS format (yymmdd HHMM)."""
        ts, fmt = convert_to_timestamp("241106", "1643")
        assert fmt == "2024-11-06 16:43:00"
        assert isinstance(ts, float)
        assert ts > 0

    def test_pupitre_format(self):
        """Test pupitre format (YYYY.MM.DD HH:MM:SS)."""
        ts, fmt = convert_to_timestamp(
            "2024.11.06", "16:43:00", date_format="%Y.%m.%d", time_format="%H:%M:%S"
        )
        assert fmt == "2024-11-06 16:43:00"

    def test_tdms_with_seconds(self):
        """Test TDMS format with seconds (yymmdd HHMMSS)."""
        ts, fmt = convert_to_timestamp(
            "241106", "164300", date_format="%y%m%d", time_format="%H%M%S"
        )
        assert fmt == "2024-11-06 16:43:00"

    def test_timestamp_is_unix(self):
        """Timestamp should be valid Unix timestamp."""
        ts, fmt = convert_to_timestamp("241106", "1643")
        # Convert back to datetime
        dt = datetime.fromtimestamp(ts)
        assert dt.year == 2024
        assert dt.month == 11
        assert dt.day == 6
        assert dt.hour == 16
        assert dt.minute == 43

    def test_different_dates(self):
        """Test various date formats."""
        test_cases = [
            (("240101", "0000"), "2024-01-01 00:00:00"),
            (("241231", "2359"), "2024-12-31 23:59:00"),
            (("250615", "1200"), "2025-06-15 12:00:00"),
        ]
        for (date, time), expected in test_cases:
            _, fmt = convert_to_timestamp(date, time)
            assert fmt == expected


class TestFileMetadata:
    """Test FileMetadata dataclass."""

    def test_basic_creation(self):
        """Test basic metadata creation."""
        meta = FileMetadata(
            filepath="/data/M9_Overview_241106-1643.tdms",
            site="M9",
            mode="Overview",
            start_time="2024-11-06 16:43:00",
            end_time="2024-11-06 17:43:00",
        )
        assert meta.site == "M9"
        assert meta.mode == "Overview"

    def test_filename_property(self):
        """filename property should return stem."""
        meta = FileMetadata(filepath="/data/M9_Overview_241106-1643.tdms")
        assert meta.filename == "M9_Overview_241106-1643"

    def test_extension_property(self):
        """extension property should return suffix."""
        meta = FileMetadata(filepath="/data/M9_Overview_241106-1643.tdms")
        assert meta.extension == ".tdms"

        meta_txt = FileMetadata(filepath="/data/2024.11.06 - 16:43:00.txt")
        assert meta_txt.extension == ".txt"

    def test_overlaps_within_range(self):
        """overlaps should return True when file is within range."""
        meta = FileMetadata(
            filepath="test.tdms",
            start_time="2024-11-06 16:45:00",
            end_time="2024-11-06 17:00:00",
        )
        assert meta.overlaps("2024-11-06 16:00:00", "2024-11-06 18:00:00")

    def test_overlaps_outside_range(self):
        """overlaps should return False when file is outside range."""
        meta = FileMetadata(
            filepath="test.tdms",
            start_time="2024-11-06 19:00:00",
            end_time="2024-11-06 20:00:00",
        )
        assert not meta.overlaps("2024-11-06 16:00:00", "2024-11-06 18:00:00")

    def test_overlaps_empty_times(self):
        """overlaps should return False when times not set."""
        meta = FileMetadata(filepath="test.tdms")
        assert not meta.overlaps("2024-11-06 16:00:00", "2024-11-06 18:00:00")


class TestFileSet:
    """Test FileSet dataclass."""

    def test_empty_fileset(self):
        """Empty FileSet should have empty lists."""
        fs = FileSet()
        assert fs.overview == []
        assert fs.archive == []
        assert fs.pupitre == []
        assert fs.default == []
        assert fs.trigger == []
        assert fs.spike == []

    def test_with_files(self):
        """FileSet should store file lists."""
        fs = FileSet(
            overview=["overview1.tdms"],
            archive=["archive1.tdms", "archive2.tdms"],
            pupitre=["pupitre1.txt"],
        )
        assert len(fs.overview) == 1
        assert len(fs.archive) == 2
        assert len(fs.pupitre) == 1

    def test_len(self):
        """len() should return total file count."""
        fs = FileSet(
            overview=["o1.tdms"],
            archive=["a1.tdms", "a2.tdms"],
            pupitre=["p1.txt"],
            default=["d1.tdms"],
        )
        assert len(fs) == 5

    def test_has_archive(self):
        """has_archive should check archive list."""
        fs_empty = FileSet()
        assert not fs_empty.has_archive

        fs_with = FileSet(archive=["a1.tdms"])
        assert fs_with.has_archive

    def test_has_pupitre(self):
        """has_pupitre should check pupitre list."""
        fs_empty = FileSet()
        assert not fs_empty.has_pupitre

        fs_with = FileSet(pupitre=["p1.txt"])
        assert fs_with.has_pupitre

    def test_has_incidents(self):
        """has_incidents should check all incident lists."""
        fs_empty = FileSet()
        assert not fs_empty.has_incidents

        fs_default = FileSet(default=["d1.tdms"])
        assert fs_default.has_incidents

        fs_trigger = FileSet(trigger=["t1.tdms"])
        assert fs_trigger.has_incidents

        fs_spike = FileSet(spike=["s1.tdms"])
        assert fs_spike.has_incidents

    def test_to_dict(self):
        """to_dict should return proper dict."""
        fs = FileSet(
            overview=["o1.tdms"],
            archive=["a1.tdms"],
        )
        d = fs.to_dict()
        assert d["overview"] == ["o1.tdms"]
        assert d["archive"] == ["a1.tdms"]
        assert "pupitre" in d
        assert "default" in d
        assert "trigger" in d
        assert "spike" in d

    def test_from_dict(self):
        """from_dict should create FileSet from dict."""
        d = {
            "overview": ["o1.tdms"],
            "archive": ["a1.tdms", "a2.tdms"],
            "pupitre": [],
            "default": [],
            "trigger": [],
            "spike": [],
        }
        fs = FileSet.from_dict(d)
        assert fs.overview == ["o1.tdms"]
        assert len(fs.archive) == 2

    def test_roundtrip(self):
        """to_dict and from_dict should be inverses."""
        original = FileSet(
            overview=["o1.tdms"],
            archive=["a1.tdms"],
            pupitre=["p1.txt"],
            default=["d1.tdms"],
            trigger=["t1.tdms"],
            spike=["s1.tdms"],
        )
        roundtrip = FileSet.from_dict(original.to_dict())
        assert roundtrip.overview == original.overview
        assert roundtrip.archive == original.archive
        assert roundtrip.pupitre == original.pupitre
        assert roundtrip.default == original.default
        assert roundtrip.trigger == original.trigger
        assert roundtrip.spike == original.spike


class TestFindFiles:
    """Test find_files function."""

    def test_returns_five_patterns(self):
        """find_files should return 5 glob patterns."""
        patterns = find_files(
            "/data/Overview/M9_Overview_241106-1643.tdms",
            "M9",
            "241106",
            "1643",
            pupitre_datadir="/pupitre",
        )
        assert len(patterns) == 5

    def test_pupitre_pattern(self):
        """Pupitre pattern should use correct date format."""
        pupitre, *_ = find_files(
            "/data/Overview/M9_Overview_241106-1643.tdms",
            "M9",
            "241106",
            "1643",
            pupitre_datadir="/pupitre",
        )
        # Pattern should be: /pupitre/M9/2024.11.06*.txt
        assert "M9" in pupitre
        assert "2024.11.06" in pupitre
        assert "*.txt" in pupitre

    def test_archive_pattern(self):
        """Archive pattern should replace Overview with Archive."""
        _, archive, *_ = find_files(
            "/data/Overview/M9_Overview_241106-1643.tdms",
            "M9",
            "241106",
            "1643",
        )
        assert "Fichiers_Archive" in archive
        assert "M9_Archive_241106" in archive
        assert "*.tdms" in archive

    def test_incident_patterns(self):
        """Incident patterns should use correct directories."""
        _, _, default, trigger, spike = find_files(
            "/data/Overview/M9_Overview_241106-1643.tdms",
            "M9",
            "241106",
            "1643",
        )
        assert "Fichiers_Default" in default
        assert "Fichiers_Manuel_Trig" in trigger
        assert "Fichiers_Spike" in spike


class TestMergeData:
    """Test merge_data function."""

    def test_single_dataframe(self):
        """Single DataFrame should be returned as-is."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = merge_data([df])
        assert result is df

    def test_multiple_dataframes(self):
        """Multiple DataFrames should be concatenated."""
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        result = merge_data([df1, df2])
        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 3, 4]

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            merge_data([])


class TestFileDiscovery:
    """Test FileDiscovery class."""

    def test_init_default(self):
        """Default init should set directories."""
        fd = FileDiscovery()
        assert fd.pupitre_datadir is not None
        assert fd.log_datadir == fd.pupitre_datadir

    def test_init_custom_dirs(self):
        """Custom directories should be set."""
        fd = FileDiscovery(pupitre_datadir="/custom/pupitre", log_datadir="/custom/logs")
        assert str(fd.pupitre_datadir) == "/custom/pupitre"
        assert str(fd.log_datadir) == "/custom/logs"

    def test_init_path_conversion(self):
        """String paths should be converted to Path."""
        fd = FileDiscovery(pupitre_datadir="/some/path")
        assert isinstance(fd.pupitre_datadir, Path)


class TestIntegration:
    """Integration tests (may require mocking)."""

    @patch("python_magnetrun.analysis.loaders.glob.glob")
    @patch("python_magnetrun.analysis.loaders.extract_data")
    def test_file_discovery_workflow(self, mock_extract, mock_glob):
        """Test complete discovery workflow with mocks."""
        # Setup mocks
        mock_extract.return_value = ("2024-11-06 16:43:00", "2024-11-06 17:43:00", False)
        mock_glob.return_value = []  # No files found

        fd = FileDiscovery(pupitre_datadir="/data")
        fs = fd.discover("/data/Overview/M9_Overview_241106-1643.tdms")

        # Should return FileSet with overview
        assert len(fs.overview) == 1
        assert fs.overview[0] == "/data/Overview/M9_Overview_241106-1643.tdms"
