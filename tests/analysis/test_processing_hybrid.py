"""Tests for hybrid data loading in analysis/processing.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from python_magnetrun.analysis.processing import load_hybrid_data, load_hybrid_incidents_data

# A valid-looking kHz source path: parents[3]=base, parents[1].name=date_str,
# Path(f).name[:2] = "10" (parseable as int → hours_set).
_KHZ_SOURCE = "/data/exp1/kHz/2025-11-05/FEPC-LNCMI/10HOST_Courant_GR1.bin"
_KEY = "FEPC-LNCMI/Courant_GR1"
_KHZ_ORIGIN = datetime(2025, 11, 5, 10, 0, 0)


def _make_record() -> MagicMock:
    record = MagicMock()
    record.housing = "M9"
    record.sources.hybrid_kHz = [_KHZ_SOURCE]
    record.sources.hybrid_rms = []
    record.sources.hybrid_vprocess = []
    return record


def _make_hc() -> MagicMock:
    hc = MagicMock()
    hc.get_hybrid_group_keys.return_value = [_KEY]
    return hc


def _make_hrun(time_array: np.ndarray, data_array: np.ndarray) -> MagicMock:
    hrun = MagicMock()
    hrun.get_time_range.return_value = (_KHZ_ORIGIN, _KHZ_ORIGIN)
    hrun.getData.return_value = (data_array, time_array)
    return hrun


class TestLoadHybridDataTimeAlignment:
    """load_hybrid_data() applies kHz → overview time offset when reference_t0 given."""

    def _call(
        self, reference_t0, time_array: np.ndarray, data_array: np.ndarray
    ) -> pd.DataFrame:
        hrun = _make_hrun(time_array, data_array)
        with patch(
            "python_magnetrun.hybrid.hybrid_run.HybridRun"
        ) as MockHybridRun:
            MockHybridRun.fromdir.return_value = hrun
            return load_hybrid_data(
                _make_record(),
                _make_hc(),
                "GR1",
                ["Courant_GR1"],
                htype="kHz",
                reference_t0=reference_t0,
            )

    def test_no_reference_t0_leaves_t_unchanged(self):
        """Without reference_t0 the t column equals the raw kHz elapsed seconds."""
        time_array = np.array([0.0, 0.1, 0.2])
        df = self._call(None, time_array, np.ones(3))
        assert not df.empty
        np.testing.assert_allclose(df["t"].values, time_array)

    def test_reference_t0_applies_negative_offset(self):
        """reference_t0 after kHz origin → negative offset shifts t back."""
        time_array = np.array([0.0, 60.0, 120.0])
        # archive starts 2.5 min after the kHz origin → offset = -150 s
        reference_t0 = datetime(2025, 11, 5, 10, 2, 30)
        expected_offset = (_KHZ_ORIGIN - reference_t0).total_seconds()  # -150.0

        df = self._call(reference_t0, time_array, np.ones(3))
        assert not df.empty
        np.testing.assert_allclose(df["t"].values, time_array + expected_offset)

    def test_reference_t0_before_khz_origin_gives_positive_offset(self):
        """reference_t0 60 s before kHz origin → t shifted forward by 60 s."""
        time_array = np.array([0.0, 1.0])
        reference_t0 = datetime(2025, 11, 5, 9, 59, 0)
        expected_offset = (_KHZ_ORIGIN - reference_t0).total_seconds()  # +60.0

        df = self._call(reference_t0, time_array, np.ones(2))
        assert not df.empty
        np.testing.assert_allclose(df["t"].values, time_array + expected_offset)

    def test_offset_debug_logged(self, caplog):
        """DEBUG log is emitted when reference_t0 is provided."""
        import logging

        time_array = np.array([0.0])
        reference_t0 = datetime(2025, 11, 5, 10, 2, 30)

        with caplog.at_level(logging.DEBUG, logger="python_magnetrun.analysis.processing"):
            self._call(reference_t0, time_array, np.ones(1))

        assert any("t_offset" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Helpers for load_hybrid_incidents_data tests
# ---------------------------------------------------------------------------

_TDIR_1 = "/data/exp1/trigger/TRIGGER__2025-11-05__10-02"
_TDIR_2 = "/data/exp1/trigger/TRIGGER__2025-11-05__10-15"
_REFERENCE_T0 = datetime(2025, 11, 5, 10, 0, 0)


def _make_trigger_record(trigger_dirs: list[str]) -> MagicMock:
    record = MagicMock()
    record.sources.hybrid_trigger = trigger_dirs
    return record


def _make_trigger_info(approx_ts: datetime | None, dirname_ts: datetime) -> MagicMock:
    info = MagicMock()
    info.trigger_approx_timestamp = approx_ts
    info.timestamp = dirname_ts
    info.trigger_name = "TRIGGER__2025-11-05__10-02"
    return info


class TestLoadHybridIncidentsData:
    """load_hybrid_incidents_data() parses trigger dirs and returns aligned t values."""

    def test_empty_sources_returns_empty_list(self):
        record = _make_trigger_record([])
        result = load_hybrid_incidents_data(record, MagicMock(), "GR1", [], _REFERENCE_T0)
        assert result == {"hybrid_trigger": []}

    def test_approx_timestamp_preferred_over_dirname(self):
        """trigger_approx_timestamp (ms precision) is used when available."""
        approx_ts = datetime(2025, 11, 5, 10, 2, 16, 921000)  # 136.921 s after reference
        dirname_ts = datetime(2025, 11, 5, 10, 2)              # 120 s — minute precision
        info = _make_trigger_info(approx_ts, dirname_ts)

        with patch(
            "python_magnetrun.hybrid.trigger.trigger_reader.parse_trigger_directory",
            return_value=info,
        ):
            result = load_hybrid_incidents_data(
                _make_trigger_record([_TDIR_1]), MagicMock(), "GR1", [], _REFERENCE_T0
            )

        dfs = result["hybrid_trigger"]
        assert len(dfs) == 1
        expected_t = (approx_ts - _REFERENCE_T0).total_seconds()
        assert abs(dfs[0]["t"].iloc[0] - expected_t) < 1e-6

    def test_falls_back_to_dirname_timestamp_when_approx_missing(self):
        """When trigger_approx_timestamp is None, dirname timestamp is used."""
        dirname_ts = datetime(2025, 11, 5, 10, 2)
        info = _make_trigger_info(None, dirname_ts)

        with patch(
            "python_magnetrun.hybrid.trigger.trigger_reader.parse_trigger_directory",
            return_value=info,
        ):
            result = load_hybrid_incidents_data(
                _make_trigger_record([_TDIR_1]), MagicMock(), "GR1", [], _REFERENCE_T0
            )

        dfs = result["hybrid_trigger"]
        assert len(dfs) == 1
        expected_t = (dirname_ts - _REFERENCE_T0).total_seconds()
        assert abs(dfs[0]["t"].iloc[0] - expected_t) < 1e-6

    def test_multiple_triggers_produce_multiple_dataframes(self):
        """One DataFrame per trigger directory."""
        ts1 = datetime(2025, 11, 5, 10, 2, 16)
        ts2 = datetime(2025, 11, 5, 10, 15, 3)
        info1 = _make_trigger_info(ts1, ts1)
        info2 = _make_trigger_info(ts2, ts2)

        with patch(
            "python_magnetrun.hybrid.trigger.trigger_reader.parse_trigger_directory",
            side_effect=[info1, info2],
        ):
            result = load_hybrid_incidents_data(
                _make_trigger_record([_TDIR_1, _TDIR_2]),
                MagicMock(), "GR1", [], _REFERENCE_T0,
            )

        dfs = result["hybrid_trigger"]
        assert len(dfs) == 2
        assert abs(dfs[0]["t"].iloc[0] - (ts1 - _REFERENCE_T0).total_seconds()) < 1e-6
        assert abs(dfs[1]["t"].iloc[0] - (ts2 - _REFERENCE_T0).total_seconds()) < 1e-6

    def test_bad_directory_is_skipped(self):
        """OSError on one directory does not prevent others from loading."""
        ts_good = datetime(2025, 11, 5, 10, 15, 3)
        info_good = _make_trigger_info(ts_good, ts_good)

        with patch(
            "python_magnetrun.hybrid.trigger.trigger_reader.parse_trigger_directory",
            side_effect=[OSError("not found"), info_good],
        ):
            result = load_hybrid_incidents_data(
                _make_trigger_record([_TDIR_1, _TDIR_2]),
                MagicMock(), "GR1", [], _REFERENCE_T0,
            )

        dfs = result["hybrid_trigger"]
        assert len(dfs) == 1  # only the good one
