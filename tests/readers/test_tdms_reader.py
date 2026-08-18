"""Unit tests for python_magnetrun.readers.tdms_reader."""

from __future__ import annotations

import pytest

from python_magnetrun.readers.tdms_reader import TdmsReader


class TestTdmsReader:
    def setup_method(self):
        self.reader = TdmsReader()

    def test_required_group(self):
        assert self.reader.required_group == "Courants_Alimentations"

    def test_t_offsets_present(self):
        assert "Overview" in self.reader.t_offsets
        assert "Archive" in self.reader.t_offsets

    def test_t_offset_overview(self):
        assert self.reader.t_offset_for("run_Overview_2025.tdms") == pytest.approx(0.5)

    def test_t_offset_archive(self):
        expected = 1 / 240.0
        assert self.reader.t_offset_for("run_Archive_2025.tdms") == pytest.approx(
            expected
        )

    def test_t_offset_unknown_file(self):
        assert self.reader.t_offset_for("run_normal_2025.tdms") == pytest.approx(0.0)

    def test_defs_file(self):
        assert self.reader.defs_file == "pigbrother-defs.json"

    def test_validate_missing_file(self):
        from python_magnetrun.utils.validation import FileFormatError

        with pytest.raises((FileNotFoundError, FileFormatError, RuntimeError, OSError)):
            self.reader.validate("/nonexistent/file.tdms")
