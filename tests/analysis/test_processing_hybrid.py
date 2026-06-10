"""Tests for hybrid data loading in analysis/processing.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from python_magnetrun.analysis.processing import load_hybrid_data

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
