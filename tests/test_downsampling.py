"""Tests for python_magnetrun.utils.downsampling — M4, NaN-M4, RDP, VW methods."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from python_magnetrun.utils.downsampling import (
    DownsampleConfig,
    downsample_arrays,
    downsample_dataframe,
)

try:
    import tsdownsample  # noqa: F401

    HAS_TSDOWNSAMPLE = True
except ImportError:
    HAS_TSDOWNSAMPLE = False

try:
    import simplification  # noqa: F401

    HAS_SIMPLIFICATION = True
except ImportError:
    HAS_SIMPLIFICATION = False

needs_tsdownsample = pytest.mark.skipif(not HAS_TSDOWNSAMPLE, reason="tsdownsample not installed")
needs_simplification = pytest.mark.skipif(not HAS_SIMPLIFICATION, reason="simplification not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sin_wave(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.sin(2 * np.pi * t).astype(np.float64)
    return t, y


def _plateau_spike(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    spike_start, spike_end = int(n * 0.45), int(n * 0.55)
    y[spike_start:spike_end] = np.linspace(0, 1, spike_end - spike_start)
    return t, y


# ===========================================================================
# M4 tests
# ===========================================================================

@needs_tsdownsample
def test_m4_returns_n_out_points():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="m4")
    t_out, y_out = downsample_arrays(y, t, cfg)
    assert len(t_out) == 200
    assert len(y_out) == 200


@needs_tsdownsample
def test_m4_indices_are_sorted():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="m4")
    t_out, _ = downsample_arrays(y, t, cfg)
    assert np.all(np.diff(t_out) >= 0)


@needs_tsdownsample
def test_m4_no_op_when_data_le_n_out():
    t, y = _sin_wave(100)
    cfg = DownsampleConfig(n_out=200, method="m4")
    t_out, y_out = downsample_arrays(y, t, cfg)
    np.testing.assert_array_equal(t_out, t)
    np.testing.assert_array_equal(y_out, y)


@needs_tsdownsample
def test_m4_downsample_dataframe():
    t, y = _sin_wave(1000)
    df = pd.DataFrame({"t": t, "y": y})
    cfg = DownsampleConfig(n_out=200, method="m4")
    result = downsample_dataframe(df, "t", ["y"], cfg)
    assert len(result) == 200
    assert list(result.columns) == ["t", "y"]


@patch("python_magnetrun.utils.downsampling.HAS_TSDOWNSAMPLE", False)
def test_m4_fallback_stride_no_tsdownsample():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="m4")
    t_out, y_out = downsample_arrays(y, t, cfg)
    assert len(t_out) <= 200
    assert len(t_out) > 0


# ===========================================================================
# NaN-M4 tests
# ===========================================================================

@needs_tsdownsample
def test_nan_m4_returns_n_out_points():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="nan_m4")
    t_out, y_out = downsample_arrays(y, t, cfg)
    assert len(t_out) == 200


@needs_tsdownsample
def test_nan_m4_preserves_nan_in_output():
    t, y = _sin_wave(1000)
    y_nan = y.copy()
    y_nan[100:110] = np.nan
    cfg = DownsampleConfig(n_out=200, method="nan_m4")
    _, y_out = downsample_arrays(y_nan, t, cfg)
    assert np.any(np.isnan(y_out))


@needs_tsdownsample
def test_nan_m4_dataframe():
    t, y = _sin_wave(1000)
    y_nan = y.copy()
    y_nan[200:210] = np.nan
    df = pd.DataFrame({"t": t, "y": y_nan})
    cfg = DownsampleConfig(n_out=200, method="nan_m4")
    result = downsample_dataframe(df, "t", ["y"], cfg)
    assert len(result) == 200
    assert result["y"].isna().any()


@patch("python_magnetrun.utils.downsampling.HAS_TSDOWNSAMPLE", False)
def test_nan_m4_fallback_stride_no_tsdownsample():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="nan_m4")
    t_out, _ = downsample_arrays(y, t, cfg)
    assert len(t_out) <= 200
    assert len(t_out) > 0


# ===========================================================================
# RDP / VW tests
# ===========================================================================

@needs_simplification
def test_rdp_with_epsilon_reduces_points():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=1000, method="rdp", epsilon=0.01)
    t_out, y_out = downsample_arrays(y, t, cfg)
    assert len(t_out) < 1000
    assert len(t_out) > 0


@needs_simplification
def test_rdp_indices_sorted():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=1000, method="rdp", epsilon=0.01)
    t_out, _ = downsample_arrays(y, t, cfg)
    assert np.all(np.diff(t_out) >= 0)


@needs_simplification
def test_rdp_n_out_cap():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=50, method="rdp", epsilon=1e-9)
    t_out, _ = downsample_arrays(y, t, cfg)
    assert len(t_out) <= 50


def test_rdp_raises_without_epsilon():
    """ValueError must be raised when epsilon is None and simplification is available."""
    if not HAS_SIMPLIFICATION:
        pytest.skip("simplification not installed — fallback path used, no error raised")
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="rdp")
    with pytest.raises(ValueError, match="epsilon"):
        downsample_arrays(y, t, cfg)


@needs_simplification
def test_rdp_plateau_awareness():
    t, y = _plateau_spike(1000)
    cfg = DownsampleConfig(n_out=1000, method="rdp", epsilon=0.001)
    t_out, _ = downsample_arrays(y, t, cfg)
    n_plateau = int(np.sum(t_out <= 0.45))
    n_spike = int(np.sum((t_out >= 0.45) & (t_out <= 0.55)))
    assert n_spike / max(n_plateau, 1) > (0.10 / 0.45)


@needs_simplification
def test_vw_with_epsilon():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=1000, method="vw", epsilon=0.01)
    t_out, _ = downsample_arrays(y, t, cfg)
    assert len(t_out) < 1000


@patch("python_magnetrun.utils.downsampling.HAS_SIMPLIFICATION", False)
def test_rdp_fallback_no_simplification():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig(n_out=200, method="rdp", epsilon=0.01)
    t_out, _ = downsample_arrays(y, t, cfg)
    assert len(t_out) <= 200
    assert len(t_out) > 0


@needs_simplification
def test_rdp_downsample_dataframe():
    t, y = _sin_wave(1000)
    df = pd.DataFrame({"t": t, "y": y})
    cfg = DownsampleConfig(n_out=1000, method="rdp", epsilon=0.01)
    result = downsample_dataframe(df, "t", ["y"], cfg)
    assert len(result) < 1000
    assert len(result) > 0


@needs_simplification
def test_from_n_out_rdp_convergence():
    t, y = _sin_wave(1000)
    cfg = DownsampleConfig.from_n_out_rdp(y, t, n_out=200, method="rdp", tol=0.15)
    assert cfg.epsilon is not None
    assert cfg.method == "rdp"
    t_out, _ = downsample_arrays(y, t, cfg)
    assert abs(len(t_out) - 200) <= 0.15 * 200
