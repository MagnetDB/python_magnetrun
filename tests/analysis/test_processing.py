"""
Tests for the processing module.

These tests verify the main orchestration functionality
including data structures, configuration, and processing utilities.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from python_magnetrun.analysis.config import get_housing_config, get_time_offset
from python_magnetrun.analysis.loaders import FileSet
from python_magnetrun.analysis.processing import (
    # Main dataclasses
    OverviewRecord,
    # Configuration
    ProcessingConfig,
    ProcessingResult,
    add_time_column_with_offset,
    # Utilities
    create_overview_dict,
    summarize_record,
)


class TestProcessingConfig:
    """Test ProcessingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()

        assert config.group == "Courants_Alimentations"
        assert config.tkey == "t"
        assert config.synchronize is True
        assert config.compute_lag is False
        assert config.dry_run is False
        assert config.downsample_percent == 100.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            pupitre_datadir="/custom/path",
            synchronize=False,
            compute_lag=True,
            debug=True,
            downsample_percent=10.0,
        )

        assert config.pupitre_datadir == "/custom/path"
        assert config.synchronize is False
        assert config.compute_lag is True
        assert config.debug is True
        assert config.downsample_percent == 10.0


class TestOverviewRecord:
    """Test OverviewRecord dataclass."""

    def test_creation(self):
        """Test basic record creation."""
        record = OverviewRecord(
            filename="M9_Overview_241106-091500",
            housing="M9",
            mode="Overview",
        )

        assert record.filename == "M9_Overview_241106-091500"
        assert record.housing == "M9"
        assert record.mode == "Overview"
        assert record.t0 is None
        assert record.duration == 0.0

    def test_data_defaults(self):
        """Test default data structures."""
        record = OverviewRecord(filename="test", housing="M9")

        assert "overview" in record.data
        assert "archive" in record.data
        assert "pupitre" in record.data
        assert "default" in record.data
        assert "trigger" in record.data
        assert "spike" in record.data

        assert isinstance(record.data["overview"], pd.DataFrame)
        assert isinstance(record.data["default"], list)

    def test_has_data_empty(self):
        """Test has_data with empty data."""
        record = OverviewRecord(filename="test", housing="M9")

        assert record.has_data("overview") is False
        assert record.has_data("archive") is False
        assert record.has_data("default") is False

    def test_has_data_with_data(self):
        """Test has_data with actual data."""
        record = OverviewRecord(filename="test", housing="M9")

        # Add overview data
        record.data["overview"] = pd.DataFrame(
            {
                "t": [0, 1, 2],
                "value": [1.0, 2.0, 3.0],
            }
        )

        assert record.has_data("overview") is True

        # Add incident data
        record.data["default"] = [pd.DataFrame({"t": [0]})]
        assert record.has_data("default") is True

    def test_get_overview(self):
        """Test get_overview method."""
        record = OverviewRecord(filename="test", housing="M9")

        df = pd.DataFrame({"t": [0, 1], "I": [10, 20]})
        record.data["overview"] = df

        result = record.get_overview()
        pd.testing.assert_frame_equal(result, df)

    def test_get_incidents(self):
        """Test get_incidents method."""
        record = OverviewRecord(filename="test", housing="M9")

        record.data["default"] = [pd.DataFrame({"t": [0]})]
        record.data["trigger"] = [pd.DataFrame({"t": [1]}), pd.DataFrame({"t": [2]})]

        incidents = record.get_incidents()

        assert len(incidents["default"]) == 1
        assert len(incidents["trigger"]) == 2
        assert len(incidents["spike"]) == 0


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        result = ProcessingResult()

        assert result.records == {}
        assert result.housing == ""
        assert result.keys == []
        assert result.errors == []

    def test_len(self):
        """Test __len__ method."""
        result = ProcessingResult()
        assert len(result) == 0

        result.records["file1"] = OverviewRecord(filename="file1", housing="M9")
        assert len(result) == 1

        result.records["file2"] = OverviewRecord(filename="file2", housing="M9")
        assert len(result) == 2

    def test_get_record(self):
        """Test get_record method."""
        result = ProcessingResult()
        record = OverviewRecord(filename="file1", housing="M9")
        result.records["file1"] = record

        assert result.get_record("file1") is record
        assert result.get_record("nonexistent") is None


class TestComputeTimeOffset:
    """Test get_time_offset function (replaces deleted compute_time_offset)."""

    def test_1hz(self):
        """Test offset for 1 Hz sampling."""
        offset = get_time_offset(1.0)
        assert offset == 0.5  # Half second

    def test_120hz(self):
        """Test offset for 120 Hz (archive) sampling."""
        offset = get_time_offset(120.0)
        assert abs(offset - 1 / 240) < 1e-10

    def test_4800hz(self):
        """Test offset for 4800 Hz (incidents) sampling."""
        offset = get_time_offset(4800.0)
        assert abs(offset - 1 / 9600) < 1e-10


class TestAddTimeColumnWithOffset:
    """Test add_time_column_with_offset function."""

    def test_basic_addition(self):
        """Test adding time column with offset."""
        t0 = datetime(2024, 11, 6, 9, 15, 0)
        df = pd.DataFrame(
            {
                "timestamp": [
                    t0,
                    t0 + timedelta(seconds=1),
                    t0 + timedelta(seconds=2),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )

        result = add_time_column_with_offset(df, t0, sampling_rate=1.0)

        assert "t" in result.columns
        assert abs(result["t"].iloc[0] - 0.5) < 1e-10  # 0 + offset
        assert abs(result["t"].iloc[1] - 1.5) < 1e-10  # 1 + offset
        assert abs(result["t"].iloc[2] - 2.5) < 1e-10  # 2 + offset

    def test_preserves_original(self):
        """Test that original DataFrame is not modified."""
        t0 = datetime(2024, 11, 6, 9, 15, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0, t0 + timedelta(seconds=1)],
                "value": [1.0, 2.0],
            }
        )

        result = add_time_column_with_offset(df, t0, sampling_rate=1.0)

        # Original should not have 't' column
        assert "t" not in df.columns
        assert "t" in result.columns


class TestGetHousingConfig:
    """Test get_housing_config function."""

    def test_valid_site(self):
        """Test getting config for valid site."""
        config = get_housing_config("M9")

        assert config is not None
        assert config.name == "M9"

    def test_invalid_site(self):
        """Test error for invalid site."""
        with pytest.raises(ValueError) as exc_info:
            get_housing_config("INVALID")

        assert "Unknown housing" in str(exc_info.value)


class TestSummarizeRecord:
    """Test summarize_record function."""

    def test_empty_record(self):
        """Test summarizing empty record."""
        record = OverviewRecord(filename="test", housing="M9")

        summary = summarize_record(record)

        assert summary["filename"] == "test"
        assert summary["housing"] == "M9"
        assert summary["has_overview"] is False
        assert summary["has_archive"] is False
        assert summary["n_default"] == 0

    def test_record_with_data(self):
        """Test summarizing record with data."""
        record = OverviewRecord(
            filename="M9_Overview_test",
            housing="M9",
            mode="Overview",
            duration=3600.0,
            teb=25.5,
            BP=3.2,
        )
        record.t0 = datetime(2024, 11, 6, 9, 15, 0)
        record.data["overview"] = pd.DataFrame({"t": [0, 1, 2]})
        record.signatures = {"Courant_GR1": {"min": 0, "max": 100}}

        summary = summarize_record(record)

        assert summary["duration"] == 3600.0
        assert summary["has_overview"] is True
        assert summary["teb"] == 25.5
        assert "Courant_GR1" in summary["signatures"]


class TestCreateOverviewDict:
    """Test create_overview_dict function."""

    def test_empty_result(self):
        """Test with empty ProcessingResult."""
        result = ProcessingResult()

        overview_dict = create_overview_dict(result)

        assert overview_dict == {}

    def test_with_records(self):
        """Test with ProcessingResult containing records."""
        result = ProcessingResult()

        record = OverviewRecord(
            filename="test_file",
            housing="M9",
            mode="Overview",
        )
        record.sources = FileSet(
            overview=["file1.tdms"],
            archive=["file2.tdms"],
            pupitre=[],
        )
        record.t0 = datetime(2024, 11, 6, 9, 15, 0)
        record.teb = 25.0
        record.BP = 3.0

        result.records["test_file"] = record

        overview_dict = create_overview_dict(result)

        assert "test_file" in overview_dict
        assert overview_dict["test_file"]["mode"] == "Overview"
        assert overview_dict["test_file"]["t0"] == record.t0
        assert overview_dict["test_file"]["teb"] == 25.0


class TestIntegration:
    """Integration tests for processing module."""

    def test_record_workflow(self):
        """Test complete record creation workflow."""
        # Create record
        record = OverviewRecord(
            filename="M9_Overview_241106-091500",
            housing="M9",
            mode="Overview",
        )

        # Set timestamp
        record.t0 = datetime(2024, 11, 6, 9, 15, 0)

        # Add overview data
        record.data["overview"] = pd.DataFrame(
            {
                "timestamp": [record.t0 + timedelta(seconds=i) for i in range(100)],
                "t": list(range(100)),
                "Courant_GR1": np.sin(np.linspace(0, 2 * np.pi, 100)) * 100,
            }
        )

        # Add signature
        record.signatures["Courant_GR1"] = {
            "min": -100,
            "max": 100,
            "mean": 0,
        }

        # Add sync info
        record.sync_info["timeshift"] = timedelta(seconds=1.5)

        # Verify
        assert record.has_data("overview")
        assert not record.has_data("archive")
        assert "Courant_GR1" in record.signatures

        # Summarize
        summary = summarize_record(record)
        assert summary["has_overview"]
        assert summary["duration"] == 0.0  # Not set directly

        # Create result
        result = ProcessingResult(housing="M9")
        result.records[record.filename] = record

        # Convert to legacy format
        overview_dict = create_overview_dict(result)
        assert record.filename in overview_dict

    def test_config_variations(self):
        """Test different configuration combinations."""
        # Analysis-focused config
        analysis_config = ProcessingConfig(
            synchronize=True,
            compute_lag=True,
            compute_distance=True,
            debug=False,
        )
        assert analysis_config.synchronize
        assert analysis_config.compute_lag

        # Quick preview config
        preview_config = ProcessingConfig(
            dry_run=True,
            downsample_percent=1.0,
        )
        assert preview_config.dry_run
        assert preview_config.downsample_percent == 1.0

        # Debug config
        debug_config = ProcessingConfig(
            debug=True,
            show=True,
            save=True,
        )
        assert debug_config.debug
        assert debug_config.show
        assert debug_config.save
