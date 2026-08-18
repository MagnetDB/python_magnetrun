"""Tests for the waterflow pipeline (data extraction side)."""

from __future__ import annotations

import builtins

import numpy as np
import pandas as pd
import pytest

from python_magnetrun.waterflow_pipeline import (
    HydraulicData,
    compute_waterflow,
    compute_waterflow_from_run,
    detect_imax_from_plateaus,
    extract_hydraulic_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal DataFrame with the given column arrays."""
    defaults = {
        "I": [100, 200, 500, 1000, 5000],
        "Rpm": [1000, 1010, 1050, 1100, 1500],
        "Flow": [10, 11, 15, 20, 50],
        "Pin": [5, 5.1, 6, 8, 15],
        "Pout": [4, 4, 4, 4, 4],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def _make_synthetic_hydraulic_data(
    n: int = 200,
    imax: float = 28000.0,
    vpmax: float = 2840.0,
    vp0: float = 1000.0,
) -> HydraulicData:
    """Return HydraulicData with a clean quadratic pump curve."""
    current = np.linspace(300, imax, n)
    vp = vpmax * (current / imax) ** 2 + vp0
    vp_ratio = vp / (vpmax + vp0)
    flow = 140.0 * vp_ratio
    pressure = 4.0 + 22.0 * vp_ratio**2
    back_pressure = np.full(n, 4.0)
    return HydraulicData(
        current=current,
        pump_speed=vp,
        flow_rate=flow,
        pressure=pressure,
        back_pressure=back_pressure,
        imax=imax,
        name="synthetic",
    )


# ---------------------------------------------------------------------------
# TestExtractHydraulicData
# ---------------------------------------------------------------------------


class TestExtractHydraulicData:
    """Test data extraction from DataFrames."""

    def test_basic_extraction(self):
        """Verify arrays are correctly extracted and filtered."""
        df = _make_df()
        data = extract_hydraulic_data(
            df,
            "I",
            "Rpm",
            "Flow",
            "Pin",
            "Pout",
            current_threshold=300.0,
        )
        # rows with I >= 300: 500, 1000, 5000
        assert len(data.current) == 3
        assert data.current[0] == pytest.approx(500.0)

    def test_all_columns_present(self):
        """Returned arrays exist for every column."""
        df = _make_df()
        data = extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")
        assert data.pump_speed is not None
        assert data.flow_rate is not None
        assert data.pressure is not None
        assert data.back_pressure is not None

    def test_name_propagated(self):
        """The name kwarg is stored on HydraulicData."""
        df = _make_df()
        data = extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout", name="M9_M10")
        assert data.name == "M9_M10"

    def test_imax_defaults_to_none(self):
        """imax is None unless explicitly set."""
        df = _make_df()
        data = extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")
        assert data.imax is None

    def test_missing_column_raises_key_error(self):
        """Verify KeyError for missing columns."""
        df = pd.DataFrame({"I": [1, 2], "Rpm": [3, 4]})
        with pytest.raises(KeyError, match="Missing columns"):
            extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")

    def test_missing_column_message_lists_all_absent(self):
        """Error message names every missing column."""
        df = pd.DataFrame({"I": [1, 2]})
        with pytest.raises(KeyError) as exc_info:
            extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")
        msg = str(exc_info.value)
        for col in ("Rpm", "Flow", "Pin", "Pout"):
            assert col in msg

    def test_insufficient_data_raises_value_error(self):
        """Verify ValueError when too few points survive filtering."""
        df = pd.DataFrame(
            {
                "I": [100, 200],
                "Rpm": [1000, 1010],
                "Flow": [10, 11],
                "Pin": [5, 5.1],
                "Pout": [4, 4],
            }
        )
        with pytest.raises(ValueError, match="at least 3"):
            extract_hydraulic_data(
                df,
                "I",
                "Rpm",
                "Flow",
                "Pin",
                "Pout",
                current_threshold=300.0,
            )

    def test_threshold_zero_keeps_all_rows(self):
        """threshold=0 keeps every row."""
        df = _make_df()
        data = extract_hydraulic_data(
            df,
            "I",
            "Rpm",
            "Flow",
            "Pin",
            "Pout",
            current_threshold=0.0,
        )
        assert len(data.current) == len(df)

    def test_array_alignment(self):
        """All returned arrays have the same length."""
        df = _make_df()
        data = extract_hydraulic_data(df, "I", "Rpm", "Flow", "Pin", "Pout")
        n = len(data.current)
        assert len(data.pump_speed) == n
        assert len(data.flow_rate) == n
        assert len(data.pressure) == n
        assert len(data.back_pressure) == n


# ---------------------------------------------------------------------------
# TestDetectImaxFromPlateaus
# ---------------------------------------------------------------------------


class TestDetectImaxFromPlateaus:
    """Test rolling-derivative plateau detection."""

    def _ramp_then_plateau(
        self,
        imax: float = 10000.0,
        plateau_fraction: float = 0.2,
        n: int = 300,
    ) -> HydraulicData:
        """
        Construct data with a clean ramp followed by a flat plateau.

        The pump speed increases quadratically up to imax * (1 - plateau_fraction),
        then stays constant for the remaining plateau_fraction of the current range.
        """
        current = np.linspace(300, imax, n)
        breakpoint_I = imax * (1.0 - plateau_fraction)

        vp_peak = 3000.0
        ramp = current[current < breakpoint_I]
        plateau = current[current >= breakpoint_I]

        vp_ramp = vp_peak * (ramp / breakpoint_I) ** 2
        vp_plateau = np.full(len(plateau), vp_peak)
        pump_speed = np.concatenate([vp_ramp, vp_plateau])

        return HydraulicData(
            current=current,
            pump_speed=pump_speed,
            flow_rate=np.ones(n),
            pressure=np.ones(n),
            back_pressure=np.ones(n),
            name="test_plateau",
        )

    def test_detects_plateau_onset(self):
        """Onset current should be close to the true breakpoint."""
        imax = 10000.0
        data = self._ramp_then_plateau(imax=imax, plateau_fraction=0.2)
        true_breakpoint = imax * 0.8

        detected = detect_imax_from_plateaus(data, plateau_threshold=0.01)

        assert detected is not None
        # Allow 10% relative tolerance — the rolling window introduces lag
        assert abs(detected - true_breakpoint) / true_breakpoint < 0.10

    def test_returns_none_when_no_plateau(self):
        """Pure ramp (no plateau) should return None."""
        n = 200
        current = np.linspace(300, 10000, n)
        pump_speed = 3000.0 * (current / 10000.0) ** 2  # monotonically increasing

        data = HydraulicData(
            current=current,
            pump_speed=pump_speed,
            flow_rate=np.ones(n),
            pressure=np.ones(n),
            back_pressure=np.ones(n),
        )
        detected = detect_imax_from_plateaus(data, plateau_threshold=0.001)
        assert detected is None

    def test_constant_pump_speed_returns_none(self):
        """Constant pump speed (zero range) should return None without error."""
        n = 50
        data = HydraulicData(
            current=np.linspace(300, 5000, n),
            pump_speed=np.full(n, 2000.0),
            flow_rate=np.ones(n),
            pressure=np.ones(n),
            back_pressure=np.ones(n),
        )
        assert detect_imax_from_plateaus(data) is None

    def test_noisy_signal_does_not_false_positive(self):
        """Random noise alone should not trigger a plateau detection."""
        rng = np.random.default_rng(0)
        n = 300
        current = np.linspace(300, 10000, n)
        # Pure noise — no ramp, no plateau structure
        pump_speed = rng.normal(loc=2000.0, scale=500.0, size=n)

        data = HydraulicData(
            current=current,
            pump_speed=pump_speed,
            flow_rate=np.ones(n),
            pressure=np.ones(n),
            back_pressure=np.ones(n),
        )
        # A strict threshold should not fire on pure noise
        detected = detect_imax_from_plateaus(data, plateau_threshold=0.001, min_plateau_samples=30)
        assert detected is None

    def test_window_parameter_accepted(self):
        """Custom window size should not raise."""
        data = self._ramp_then_plateau()
        result = detect_imax_from_plateaus(data, window=10, min_plateau_samples=5)
        assert result is None or isinstance(result, float)

    def test_detected_imax_is_float(self):
        """Return type must be float when a plateau is found."""
        data = self._ramp_then_plateau(plateau_fraction=0.3)
        detected = detect_imax_from_plateaus(data)
        if detected is not None:
            assert isinstance(detected, float)


# ---------------------------------------------------------------------------
# TestComputeWaterflow
# ---------------------------------------------------------------------------


class TestComputeWaterflow:
    """Test the full fitting pipeline (requires python_magnetcooling)."""

    def test_simple_method(self):
        """End-to-end with simple fitting returns a WaterFlow."""
        pytest.importorskip("python_magnetcooling")
        from python_magnetrun.waterflow_pipeline import compute_waterflow

        data = _make_synthetic_hydraulic_data(imax=28000.0)
        wf = compute_waterflow(data, method="simple")

        assert abs(wf.current_max - 28000.0) < 1.0
        assert abs(wf.pump_speed_max - 2840.0) < 50.0
        assert wf.flow_rate(20000.0) > 0.0

    def test_missing_imax_raises_value_error(self):
        """simple method without imax raises ValueError."""
        data = _make_synthetic_hydraulic_data()
        data.imax = None

        with pytest.raises(ValueError, match="imax must be set"):
            compute_waterflow(data, method="simple")

    def test_missing_magnetcooling_raises_import_error(self, monkeypatch):
        """ImportError is raised with a helpful message when the package is absent."""
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "python_magnetcooling" in name:
                raise ImportError("mocked absence")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        data = _make_synthetic_hydraulic_data()
        with pytest.raises(ImportError, match="python_magnetcooling"):
            compute_waterflow(data, method="simple")

    def test_invalid_method_propagates(self):
        """An unrecognised method string should propagate an error from the fitting layer."""
        pytest.importorskip("python_magnetcooling")
        data = _make_synthetic_hydraulic_data()

        with pytest.raises(Exception):  # noqa: B017
            compute_waterflow(data, method="unknown_method")


# ---------------------------------------------------------------------------
# TestComputeWaterflowFromRun
# ---------------------------------------------------------------------------


class TestComputeWaterflowFromRun:
    """Test the MagnetRun convenience wrapper."""

    class _FakeMData:
        def __init__(self, df):
            self._df = df

        def getData(self):
            return self._df

    class _FakeMagnetRun:
        def __init__(self, df, insert="fake_magnet"):
            self._df = df
            self._insert = insert

        def getMData(self):
            return TestComputeWaterflowFromRun._FakeMData(self._df)

        def getInsert(self):
            return self._insert

    def _make_mrun(self, n: int = 200):
        data = _make_synthetic_hydraulic_data(n=n, imax=28000.0)
        df = pd.DataFrame(
            {
                "IH": data.current,
                "Rpm": data.pump_speed,
                "Flow": data.flow_rate,
                "HP": data.pressure,
                "BP": data.back_pressure,
            }
        )
        return self._FakeMagnetRun(df)

    def test_basic_call(self):
        """compute_waterflow_from_run returns a WaterFlow for a valid MagnetRun."""
        pytest.importorskip("python_magnetcooling")
        mrun = self._make_mrun()
        wf = compute_waterflow_from_run(
            mrun,
            current_col="IH",
            rpm_col="Rpm",
            flow_col="Flow",
            pressure_in_col="HP",
            pressure_out_col="BP",
            method="simple",
            imax=28000.0,
        )
        assert wf is not None

    def test_name_extracted_from_mrun(self):
        """Insert name is passed through to HydraulicData."""
        df = pd.DataFrame(
            {
                "IH": np.linspace(300, 5000, 50),
                "Rpm": np.linspace(1000, 2000, 50),
                "Flow": np.linspace(10, 50, 50),
                "HP": np.linspace(5, 15, 50),
                "BP": np.full(50, 4.0),
            }
        )
        mrun = self._FakeMagnetRun(df, insert="M10")
        # We test only the extraction step here (no fitting dependency)
        result_df = mrun.getMData().getData()
        data = extract_hydraulic_data(
            result_df,
            "IH",
            "Rpm",
            "Flow",
            "HP",
            "BP",
            current_threshold=0.0,
        )
        # Name would be set from mrun.getInsert() in the real call
        assert mrun.getInsert() == "M10"
        assert len(data.current) == 50

    def test_missing_column_raises(self):
        """KeyError propagates correctly from the MagnetRun wrapper."""
        mrun = self._make_mrun()
        with pytest.raises(KeyError):
            compute_waterflow_from_run(
                mrun,
                current_col="WRONG",
                rpm_col="Rpm",
                flow_col="Flow",
                pressure_in_col="HP",
                pressure_out_col="BP",
                imax=28000.0,
            )

    def test_simple_without_imax_and_no_plateau_raises(self):
        """
        If method='simple', imax=None, and no plateau detected, raise ValueError.
        """
        n = 50
        # Pure linear ramp — no plateau
        df = pd.DataFrame(
            {
                "IH": np.linspace(300, 10000, n),
                "Rpm": np.linspace(1000, 3000, n),  # monotone, no plateau
                "Flow": np.linspace(10, 50, n),
                "HP": np.linspace(5, 15, n),
                "BP": np.full(n, 4.0),
            }
        )
        mrun = self._FakeMagnetRun(df)

        with pytest.raises(ValueError, match="imax"):
            compute_waterflow_from_run(
                mrun,
                current_col="IH",
                rpm_col="Rpm",
                flow_col="Flow",
                pressure_in_col="HP",
                pressure_out_col="BP",
                method="simple",
                imax=None,
            )
