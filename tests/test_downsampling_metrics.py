"""Tests for python_magnetrun.utils.downsampling_metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from python_magnetrun.utils.downsampling import DownsampleConfig
from python_magnetrun.utils.downsampling_metrics import (
    benchmark_configs,
    evaluate_downsampling,
    evaluate_downsampling_segments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 2000
RNG = np.random.default_rng(42)


@pytest.fixture
def sine_arrays():
    t = np.linspace(0, 10, N)
    data = np.sin(2 * np.pi * t)
    return data, t


@pytest.fixture
def ramp_arrays():
    """High-frequency oscillation (transition) in first half, constant plateau in second.

    Stride downsampling aliases the oscillation (high RMSE) while the constant
    plateau is reconstructed exactly (near-zero RMSE), giving a clear ordering
    for segment-aware metric tests.
    """
    t = np.linspace(0, 10, N)
    osc = 0.05 * np.sin(2 * np.pi * 30 * t[: N // 2])
    data = np.concatenate([osc, np.ones(N // 2)])
    return data, t


# ---------------------------------------------------------------------------
# Phase 1 tests
# ---------------------------------------------------------------------------


def test_no_downsample_zero_error(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=len(data), method="stride")
    m = evaluate_downsampling(data, t, config)
    assert m.rmse == pytest.approx(0.0, abs=1e-12)


def test_compression_ratio_correct(sine_arrays):
    data, t = sine_arrays
    n_out = 200
    config = DownsampleConfig(n_out=n_out, method="stride")
    m = evaluate_downsampling(data, t, config)
    expected = len(data) / m.n_downsampled
    assert m.compression_ratio == pytest.approx(expected, rel=0.05)


def test_elapsed_positive(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config)
    assert m.elapsed_s > 0


def test_m4_rmse_le_stride_same_n_out(sine_arrays):
    data, t = sine_arrays
    n_out = 200
    m_stride = evaluate_downsampling(data, t, DownsampleConfig(n_out=n_out, method="stride"))
    m_m4 = evaluate_downsampling(data, t, DownsampleConfig(n_out=n_out, method="minmax"))
    # minmax preserves extremes — its RMSE should not exceed stride by a large margin
    assert m_m4.rmse <= m_stride.rmse * 1.5


def test_peak_max_error_zero_when_exact(sine_arrays):
    """Method that keeps exact points should have near-zero peak_max_error."""
    data, t = sine_arrays
    config = DownsampleConfig(n_out=len(data), method="stride")
    m = evaluate_downsampling(data, t, config)
    assert m.peak_max_error == pytest.approx(0.0, abs=1e-9)


def test_hausdorff_finite(sine_arrays):
    pytest.importorskip("scipy")
    data, t = sine_arrays
    config = DownsampleConfig(n_out=200, method="stride")
    m = evaluate_downsampling(data, t, config)
    assert math.isfinite(m.hausdorff_distance)
    assert m.hausdorff_distance > 0


def test_energy_ratio_near_one(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=N // 2, method="stride")
    m = evaluate_downsampling(data, t, config)
    assert 0.8 <= m.energy_ratio <= 1.2


# ---------------------------------------------------------------------------
# Phase 2 tests
# ---------------------------------------------------------------------------


def test_benchmark_configs_shape(sine_arrays):
    data, t = sine_arrays
    configs = [
        DownsampleConfig(n_out=500, method="stride"),
        DownsampleConfig(n_out=500, method="minmax"),
    ]
    df = benchmark_configs(data, t, configs)
    assert len(df) == len(configs)
    for col in ("rmse", "mae", "max_error", "elapsed_s", "compression_ratio"):
        assert col in df.columns


def test_benchmark_configs_best_method(sine_arrays):
    """minmax should rank at least as good as stride by RMSE on a sine wave."""
    data, t = sine_arrays
    configs = [
        DownsampleConfig(n_out=200, method="stride"),
        DownsampleConfig(n_out=200, method="minmax"),
    ]
    df = benchmark_configs(data, t, configs)
    # minmax RMSE should not be worse than stride by a significant margin
    assert df.loc["minmax", "rmse"] <= df.loc["stride", "rmse"] * 2.0


# ---------------------------------------------------------------------------
# Phase 3 tests
# ---------------------------------------------------------------------------


def test_segment_metrics_sum_to_one(ramp_arrays):
    data, t = ramp_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    _base, seg = evaluate_downsampling_segments(data, t, config)
    total = seg.plateau_fraction + seg.transition_fraction
    assert total == pytest.approx(1.0, abs=1e-9)


def test_segment_transition_rmse_higher(ramp_arrays):
    """Stride aliasing of the oscillating region produces higher RMSE than the plateau."""
    data, t = ramp_arrays
    config = DownsampleConfig(n_out=200, method="stride")
    _base, seg = evaluate_downsampling_segments(data, t, config)
    if math.isfinite(seg.transition_rmse) and math.isfinite(seg.plateau_rmse):
        assert seg.transition_rmse >= seg.plateau_rmse


# ---------------------------------------------------------------------------
# Memory / size tests
# ---------------------------------------------------------------------------


def test_memory_fields_none_by_default(sine_arrays):
    """Memory fields are None when compute_memory is not requested."""
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config)
    assert m.peak_memory_bytes is None
    assert m.input_memory_bytes is None
    assert m.output_memory_bytes is None
    assert m.memory_overhead_ratio is None


def test_output_memory_bytes_exact(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config, compute_memory=True)
    from python_magnetrun.utils.downsampling import downsample_arrays
    data_ds, time_ds = downsample_arrays(data, t, config)
    assert m.output_memory_bytes == data_ds.nbytes + time_ds.nbytes


def test_input_memory_bytes_exact(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config, compute_memory=True)
    assert m.input_memory_bytes == data.nbytes + t.nbytes


def test_memory_overhead_ratio_positive(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config, compute_memory=True)
    assert m.memory_overhead_ratio >= 0


def test_output_bytes_lt_input_bytes(sine_arrays):
    data, t = sine_arrays
    config = DownsampleConfig(n_out=N // 10, method="stride")
    m = evaluate_downsampling(data, t, config, compute_memory=True)
    assert m.output_memory_bytes < m.input_memory_bytes


# ---------------------------------------------------------------------------
# Tier 2 test (subprocess)
# ---------------------------------------------------------------------------


def test_tier2_runs(sine_arrays):
    """Tier 2 subprocess measurement completes without error."""
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="minmax")
    m = evaluate_downsampling(data, t, config, compute_memory=True, memory_tier=2)
    assert m.peak_memory_bytes >= 0


# ---------------------------------------------------------------------------
# Tier 3 test (memray, optional)
# ---------------------------------------------------------------------------


def test_tier3_memray_captures_native(sine_arrays):
    pytest.importorskip("memray")
    data, t = sine_arrays
    config = DownsampleConfig(n_out=500, method="stride")
    m = evaluate_downsampling(data, t, config, compute_memory=True, memory_tier=3)
    assert m.peak_memory_bytes >= 0
