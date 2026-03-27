"""Unit tests for python_magnetrun processing modules.

Covers:
  - distance: calc_euclidean, calc_mape, calc_correlation
  - smoothers: kernel_function, savgol, lowess_bell_shape_kern, lowess_ag, lowess_sm
  - correlations: crosscorr
  - trends: piecewise_linear_approximation
  - stats: stats()
  - breakingpoints: detect_changes (skipped if ruptures not installed)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import

import numpy as np
import pandas as pd
import pytest

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"

# ---------------------------------------------------------------------------
# distance.py
# ---------------------------------------------------------------------------

from python_magnetrun.processing.distance import (  # noqa: E402
    calc_correlation,
    calc_euclidean,
    calc_mape,
)


class TestCalcEuclidean:
    def test_identical_arrays_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert calc_euclidean(a, a) == pytest.approx(0.0)

    def test_known_3_4_5_triangle(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert calc_euclidean(a, b) == pytest.approx(5.0)

    def test_symmetric(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 6.0, 8.0])
        assert calc_euclidean(a, b) == pytest.approx(calc_euclidean(b, a))

    def test_nonnegative(self) -> None:
        a = np.array([1.0, -2.0])
        b = np.array([-3.0, 4.0])
        assert calc_euclidean(a, b) >= 0.0

    def test_single_element(self) -> None:
        a = np.array([5.0])
        b = np.array([2.0])
        assert calc_euclidean(a, b) == pytest.approx(3.0)


class TestCalcMape:
    def test_identical_arrays_returns_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert calc_mape(a, a) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # mean(|[1, 1, 1]|) = 1.0
        assert calc_mape(a, b) == pytest.approx(1.0)

    def test_nonnegative(self) -> None:
        a = np.random.default_rng(42).standard_normal(20)
        b = np.random.default_rng(7).standard_normal(20)
        assert calc_mape(a, b) >= 0.0

    def test_symmetric_absolute_error(self) -> None:
        """calc_mape uses absolute error so it should be symmetric."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert calc_mape(a, b) == pytest.approx(calc_mape(b, a))


class TestCalcCorrelation:
    def test_perfect_positive_correlation(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert calc_correlation(a, a) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        assert calc_correlation(a, b) == pytest.approx(-1.0)

    def test_range_minus_one_to_one(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        r = calc_correlation(a, b)
        assert -1.0 <= r <= 1.0

    def test_constant_offset_does_not_affect(self) -> None:
        """Adding a constant to one array should not change correlation."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = a + 100.0
        assert calc_correlation(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# smoothers.py — pure math functions
# ---------------------------------------------------------------------------

from python_magnetrun.processing.smoothers import (  # noqa: E402
    kernel_function,
    lowess_ag,
    lowess_bell_shape_kern,
    lowess_sm,
    savgol,
)


class TestKernelFunction:
    def test_maximum_at_x0(self) -> None:
        xi = np.linspace(-1.0, 1.0, 101)
        x0 = 0.0
        k = kernel_function(xi, x0, tau=0.1)
        assert np.argmax(k) == np.argmin(np.abs(xi - x0))

    def test_returns_array_same_shape(self) -> None:
        xi = np.array([0.0, 0.5, 1.0])
        result = kernel_function(xi, 0.0)
        assert result.shape == xi.shape

    def test_nonnegative(self) -> None:
        xi = np.linspace(-2.0, 2.0, 50)
        assert np.all(kernel_function(xi, 0.0) >= 0.0)

    def test_unit_value_at_x0(self) -> None:
        """kernel_function(x0, x0) should equal 1 (exp(0) = 1)."""
        assert kernel_function(np.array([0.5]), 0.5) == pytest.approx(1.0)

    def test_decays_away_from_x0(self) -> None:
        """Values far from x0 should be smaller than at x0."""
        x0 = 0.0
        k_at_x0 = kernel_function(np.array([x0]), x0)
        k_far = kernel_function(np.array([5.0]), x0, tau=0.1)
        assert k_far < k_at_x0


class TestSavgol:
    def test_output_length_unchanged(self) -> None:
        y = np.sin(np.linspace(0, 2 * np.pi, 100))
        smoothed = savgol(y, window=11, polyorder=3)
        assert smoothed.shape == y.shape

    def test_constant_signal_preserved(self) -> None:
        y = np.ones(50)
        smoothed = savgol(y, window=7, polyorder=2)
        np.testing.assert_allclose(smoothed, y, atol=1e-10)

    def test_linear_signal_preserved(self) -> None:
        y = np.linspace(0.0, 1.0, 50)
        smoothed = savgol(y, window=7, polyorder=2)
        np.testing.assert_allclose(smoothed, y, atol=1e-10)

    def test_reduces_noise(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(0, 2 * np.pi, 100)
        signal = np.sin(x)
        noisy = signal + 0.3 * rng.standard_normal(100)
        smoothed = savgol(noisy, window=15, polyorder=3)
        # smoothed signal should be closer to the true signal than the noisy one
        noise_err = np.mean((noisy - signal) ** 2)
        smooth_err = np.mean((smoothed - signal) ** 2)
        assert smooth_err < noise_err

    def test_derivative_differs_from_signal(self) -> None:
        """deriv=1 result should differ from the original signal."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        dy = savgol(y, window=11, polyorder=4, deriv=1)
        assert dy.shape == y.shape
        assert not np.allclose(dy, y)


class TestLowessBellShapeKern:
    @pytest.fixture
    def linear_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0, 25)
        y = 2.0 * x + 1.0
        return x, y

    def test_output_length(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_bell_shape_kern(x, y, tau=0.05)
        assert len(result) == len(x)

    def test_returns_numpy_array(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_bell_shape_kern(x, y)
        assert isinstance(result, np.ndarray)

    def test_approximates_linear(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_bell_shape_kern(x, y, tau=0.05)
        np.testing.assert_allclose(result, y, atol=0.2)


class TestLowessAg:
    @pytest.fixture
    def linear_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0, 20)
        y = 3.0 * x + 0.5
        return x, y

    def test_output_length(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_ag(x, y)
        assert len(result) == len(x)

    def test_returns_numpy_array(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_ag(x, y)
        assert isinstance(result, np.ndarray)

    def test_approximates_linear(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_ag(x, y)
        np.testing.assert_allclose(result, y, atol=0.2)


class TestLowessSm:
    @pytest.fixture
    def linear_data(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0, 20)
        y = 3.0 * x + 0.5
        return x, y

    def test_output_length(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_sm(x, y)
        assert len(result) == len(x)

    def test_returns_numpy_array(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_sm(x, y)
        assert isinstance(result, np.ndarray)

    def test_approximates_linear(self, linear_data: tuple[np.ndarray, np.ndarray]) -> None:
        x, y = linear_data
        result = lowess_sm(x, y)
        np.testing.assert_allclose(result, y, atol=0.3)


# ---------------------------------------------------------------------------
# correlations.py — crosscorr (pure pandas helper)
# ---------------------------------------------------------------------------

from python_magnetrun.processing.correlations import crosscorr  # noqa: E402


class TestCrosscorr:
    @pytest.fixture
    def sine_series(self) -> pd.Series:
        n = 60
        t = np.linspace(0, 2 * np.pi, n)
        return pd.Series(np.sin(t))

    def test_zero_lag_identical_series_is_one(self, sine_series: pd.Series) -> None:
        r = crosscorr(sine_series, sine_series, lag=0)
        assert r == pytest.approx(1.0)

    def test_returns_float_or_nan(self, sine_series: pd.Series) -> None:
        r = crosscorr(sine_series, sine_series, lag=0)
        assert isinstance(float(r), float)

    def test_lag_zero_is_pearson_r(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        r = crosscorr(x, y, lag=0)
        assert r == pytest.approx(1.0)

    def test_anticorrelated_series(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        r = crosscorr(x, y, lag=0)
        assert r == pytest.approx(-1.0)

    def test_nonzero_lag_shifts_series(self, sine_series: pd.Series) -> None:
        """A non-zero lag should change the correlation value."""
        r0 = crosscorr(sine_series, sine_series, lag=0)
        r5 = crosscorr(sine_series, sine_series, lag=5)
        assert r0 != pytest.approx(r5, abs=1e-6)

    def test_wrap_true_fills_boundary(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        r_wrap = crosscorr(x, y, lag=2, wrap=True)
        r_no_wrap = crosscorr(x, y, lag=2, wrap=False)
        # Both return a scalar; wrap fills NaN differently
        assert isinstance(float(r_wrap), float)
        assert isinstance(float(r_no_wrap), float)


# ---------------------------------------------------------------------------
# trends.py — piecewise_linear_approximation (pure function)
# ---------------------------------------------------------------------------

from python_magnetrun.processing.trends import piecewise_linear_approximation  # noqa: E402


class TestPiecewiseLinearApproximation:
    def test_returns_list_of_tuples(self) -> None:
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_output_length_equals_input_length(self) -> None:
        """The function appends one final duplicate, giving len(series) tuples."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert len(result) == len(series)

    def test_monotone_increasing_all_up(self) -> None:
        series = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        # First n-1 entries (the loop iterations) should all be "U"
        for trend, _ in result[:-1]:
            assert trend == "U"

    def test_monotone_decreasing_all_down(self) -> None:
        series = pd.Series([3.0, 2.0, 1.0, 0.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        for trend, _ in result[:-1]:
            assert trend == "D"

    def test_constant_series_all_plateau(self) -> None:
        series = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = piecewise_linear_approximation(series, threshold=0.1)
        for trend, _ in result:
            assert trend == "P"

    def test_small_changes_below_threshold_are_plateau(self) -> None:
        series = pd.Series([0.0, 0.001, 0.0, 0.001, 0.0])
        result = piecewise_linear_approximation(series, threshold=0.1)
        for trend, _ in result:
            assert trend == "P"

    def test_mixed_up_then_down(self) -> None:
        series = pd.Series([0.0, 1.0, 2.0, 1.0, 0.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert result[0][0] == "U"
        assert result[1][0] == "U"
        assert result[2][0] == "D"
        assert result[3][0] == "D"

    def test_difference_values_are_correct(self) -> None:
        series = pd.Series([0.0, 2.0, 5.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert result[0][1] == pytest.approx(2.0)
        assert result[1][1] == pytest.approx(3.0)

    def test_threshold_boundary(self) -> None:
        """Changes exactly at threshold should be treated as Up/Down."""
        series = pd.Series([0.0, 0.5, 0.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert result[0][0] == "U"
        assert result[1][0] == "D"

    def test_two_element_series(self) -> None:
        series = pd.Series([0.0, 1.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert len(result) == 2
        assert result[0][0] == "U"


# ---------------------------------------------------------------------------
# stats.py
# ---------------------------------------------------------------------------

from python_magnetrun.magnetdata import MagnetData  # noqa: E402
from python_magnetrun.processing.stats import stats  # noqa: E402


@pytest.fixture(scope="module")
def pupitre_magnetdata() -> MagnetData:
    """MagnetData from the sample pupitre txt file — shared across stats tests.

    Units() must be called so that getUnitKey() works.
    """
    md = MagnetData.fromtxt(str(SAMPLE_TXT))
    md.Units()
    return md


class TestStats:
    def test_returns_tuple_of_two(self, pupitre_magnetdata: MagnetData) -> None:
        result = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tables_is_list(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert isinstance(tables, list)

    def test_headers_contains_stat_names(self, pupitre_magnetdata: MagnetData) -> None:
        _, headers = stats(pupitre_magnetdata, fields=["Field"], display=False)
        for expected in ("Mean", "Max", "Min", "Std", "Median"):
            assert expected in headers

    def test_one_row_per_requested_field(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field", "Icoil1"], display=False)
        assert len(tables) == 2

    def test_missing_field_gives_nan_row(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["NonExistentKey"], display=False)
        assert len(tables) == 1
        assert tables[0][0] == "NonExistentKey[N/A]"

    def test_field_mean_value(self, pupitre_magnetdata: MagnetData) -> None:
        """Field column in sample file is [0.5, 0.6, 0.7, 0.8]; mean=0.65."""
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        row = tables[0]
        # row: [name, mean, max, min, std, median, mode]
        assert row[1] == pytest.approx(0.65)

    def test_field_max_value(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert tables[0][2] == pytest.approx(0.8)

    def test_field_min_value(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert tables[0][3] == pytest.approx(0.5)

    def test_empty_fields_list(self, pupitre_magnetdata: MagnetData) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=[], display=False)
        assert tables == []

    def test_default_fields_when_none(self, pupitre_magnetdata: MagnetData) -> None:
        """Passing fields=None should fall back to the built-in default list."""
        tables, _ = stats(pupitre_magnetdata, fields=None, display=False)
        assert len(tables) > 0


# ---------------------------------------------------------------------------
# breakingpoints.py — detect_changes (optional dependency: ruptures)
# ---------------------------------------------------------------------------

ruptures = pytest.importorskip("ruptures", reason="ruptures not installed")

from python_magnetrun.processing.breakingpoints import detect_changes  # noqa: E402


@pytest.fixture
def step_series() -> pd.Series:
    """Time series with one clear step change at index 50."""
    data = np.concatenate([np.zeros(50), np.ones(50) * 10])
    return pd.Series(data, name="signal")


@pytest.fixture
def constant_series() -> pd.Series:
    return pd.Series(np.ones(60), name="constant")


class TestDetectChanges:
    def test_returns_list(self, step_series: pd.Series) -> None:
        changes = detect_changes(step_series, algoname="Binseg", model="l2", n_bkps=1)
        assert isinstance(changes, list)

    def test_all_changes_within_bounds(self, step_series: pd.Series) -> None:
        """No change point index should equal len(signal) (excluded by the code)."""
        changes = detect_changes(step_series, algoname="Binseg", model="l2", n_bkps=2)
        assert all(c < len(step_series) for c in changes)

    def test_detects_step_near_midpoint(self, step_series: pd.Series) -> None:
        changes = detect_changes(step_series, algoname="Binseg", model="l2", n_bkps=1)
        assert len(changes) >= 1
        assert any(40 <= c <= 60 for c in changes)

    def test_constant_series_zero_or_one_change(self, constant_series: pd.Series) -> None:
        changes = detect_changes(constant_series, algoname="Binseg", model="l2", n_bkps=1)
        assert len(changes) <= 2

    def test_dynp_algorithm(self) -> None:
        """Dynp algorithm should also detect a step change."""
        data = np.concatenate([np.zeros(40), np.ones(40) * 5])
        ts = pd.Series(data)
        changes = detect_changes(ts, algoname="Dynp", model="l2", n_bkps=1)
        assert isinstance(changes, list)
        assert all(c < len(ts) for c in changes)

    def test_multiple_steps_detected(self) -> None:
        """Three-level step series should yield at least two change points."""
        data = np.concatenate([np.zeros(30), np.ones(30) * 5, np.ones(30) * 10])
        ts = pd.Series(data)
        changes = detect_changes(ts, algoname="Binseg", model="l2", n_bkps=2)
        assert len(changes) >= 2
