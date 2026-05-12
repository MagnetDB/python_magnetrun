"""Unit tests for python_magnetrun processing modules.

Covers:
  - distance: calc_euclidean, calc_mape, calc_correlation
  - smoothers: kernel_function, savgol, lowess_bell_shape_kern, lowess_ag, lowess_sm
  - correlations: crosscorr
  - trends: piecewise_linear_approximation
  - stats: stats()
  - hysteresis: remove_outliers, remove_outliers_by_x_range, hysteresis_model,
                multi_level_hysteresis, relay_hysteresis, continuous_hysteresis
  - plateaux: tuple_type
  - breakingpoints: detect_changes (skipped if ruptures not installed)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before any plt import

import numpy as np
import pandas as pd
import pytest

SAMPLE_TXT = Path(__file__).parent / "data" / "sample_pupitre.txt"

# Optional dependency guards — do NOT use pytest.importorskip at module level as it
# causes the entire file to be skipped when the package is absent.

try:
    import ruptures as _ruptures  # noqa: F401

    from python_magnetrun.processing.breakingpoints import detect_changes

    _has_ruptures = True
except ImportError:
    _has_ruptures = False

needs_ruptures = pytest.mark.skipif(not _has_ruptures, reason="ruptures not installed")

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
        noise_err = np.mean((noisy - signal) ** 2)
        smooth_err = np.mean((smoothed - signal) ** 2)
        assert smooth_err < noise_err

    def test_derivative_differs_from_signal(self) -> None:
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
        r0 = crosscorr(sine_series, sine_series, lag=0)
        r5 = crosscorr(sine_series, sine_series, lag=5)
        assert r0 != pytest.approx(r5, abs=1e-6)

    def test_wrap_true_fills_boundary(self) -> None:
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        r_wrap = crosscorr(x, y, lag=2, wrap=True)
        r_no_wrap = crosscorr(x, y, lag=2, wrap=False)
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
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
        assert len(result) == len(series)

    def test_monotone_increasing_all_up(self) -> None:
        series = pd.Series([0.0, 1.0, 2.0, 3.0])
        result = piecewise_linear_approximation(series, threshold=0.5)
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

from python_magnetrun.magnetdata import load_magnetdata  # noqa: E402
from python_magnetrun.magnetdata_base import MagnetDataBase  # noqa: E402
from python_magnetrun.processing.stats import stats  # noqa: E402


@pytest.fixture(scope="module")
def pupitre_magnetdata() -> MagnetDataBase:
    """PandasMagnetData from the sample pupitre txt file — shared across stats tests."""
    md = load_magnetdata(str(SAMPLE_TXT))
    md.Units()
    return md


class TestStats:
    def test_returns_tuple_of_two(self, pupitre_magnetdata: MagnetDataBase) -> None:
        result = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tables_is_list(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert isinstance(tables, list)

    def test_headers_contains_stat_names(self, pupitre_magnetdata: MagnetDataBase) -> None:
        _, headers = stats(pupitre_magnetdata, fields=["Field"], display=False)
        for expected in ("Mean", "Max", "Min", "Std", "Median"):
            assert expected in headers

    def test_one_row_per_requested_field(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field", "Icoil1"], display=False)
        assert len(tables) == 2

    def test_missing_field_gives_nan_row(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["NonExistentKey"], display=False)
        assert len(tables) == 1
        assert tables[0][0] == "NonExistentKey[N/A]"

    def test_field_mean_value(self, pupitre_magnetdata: MagnetDataBase) -> None:
        """Field column in sample file is [0.5, 0.6, 0.7, 0.8]; mean=0.65."""
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        row = tables[0]
        assert row[1] == pytest.approx(0.65)

    def test_field_max_value(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert tables[0][2] == pytest.approx(0.8)

    def test_field_min_value(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=["Field"], display=False)
        assert tables[0][3] == pytest.approx(0.5)

    def test_empty_fields_list(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=[], display=False)
        assert tables == []

    def test_default_fields_when_none(self, pupitre_magnetdata: MagnetDataBase) -> None:
        tables, _ = stats(pupitre_magnetdata, fields=None, display=False)
        assert len(tables) > 0


# ---------------------------------------------------------------------------
# plateaux.py — tuple_type (pure stdlib function)
# ---------------------------------------------------------------------------

from python_magnetrun.processing.plateaux import tuple_type  # noqa: E402


class TestTupleType:
    def test_basic_two_element(self) -> None:
        result = tuple_type("(Field,Icoil1)")
        assert result == ("Field", "Icoil1")

    def test_without_parens(self) -> None:
        result = tuple_type("a,b,c")
        assert result == ("a", "b", "c")

    def test_single_element(self) -> None:
        result = tuple_type("(Field)")
        assert result == ("Field",)

    def test_returns_tuple(self) -> None:
        assert isinstance(tuple_type("(x,y)"), tuple)

    def test_whitespace_preserved_in_elements(self) -> None:
        """Elements are mapped as-is; no stripping is done by tuple_type."""
        result = tuple_type("( x , y )")
        assert result == (" x ", " y ")


# ---------------------------------------------------------------------------
# hysteresis.py — pure numeric functions
# ---------------------------------------------------------------------------

from python_magnetrun.processing.hysteresis import (  # noqa: E402
    continuous_hysteresis,
    hysteresis_model,
    multi_level_hysteresis,
    relay_hysteresis,
    remove_outliers,
    remove_outliers_by_x_range,
)


@pytest.fixture
def clean_xy_df() -> pd.DataFrame:
    """Small DataFrame with x and y columns, no outliers."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 40)
    y = 2.0 * x + rng.normal(0, 0.02, 40)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def xy_df_with_spike(clean_xy_df: pd.DataFrame) -> pd.DataFrame:
    """Same DataFrame with two obvious outliers injected."""
    df = clean_xy_df.copy()
    df.loc[5, "y"] = 100.0    # spike high
    df.loc[10, "y"] = -100.0  # spike low
    return df


class TestRemoveOutliers:
    def test_returns_dataframe(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers(clean_xy_df, method="iqr")
        assert isinstance(result, pd.DataFrame)

    def test_columns_preserved(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers(clean_xy_df, method="iqr")
        assert "x" in result.columns and "y" in result.columns

    def test_iqr_removes_spikes(self, xy_df_with_spike: pd.DataFrame) -> None:
        result = remove_outliers(xy_df_with_spike, method="iqr", threshold=1.5)
        assert len(result) < len(xy_df_with_spike)

    def test_iqr_spike_values_gone(self, xy_df_with_spike: pd.DataFrame) -> None:
        result = remove_outliers(xy_df_with_spike, method="iqr", threshold=1.5)
        assert (result["y"].abs() < 10.0).all()

    def test_zscore_removes_spikes(self, xy_df_with_spike: pd.DataFrame) -> None:
        result = remove_outliers(xy_df_with_spike, method="zscore", threshold=2.0)
        assert (result["y"].abs() < 10.0).all()

    def test_mad_removes_spikes(self, xy_df_with_spike: pd.DataFrame) -> None:
        result = remove_outliers(xy_df_with_spike, method="mad", threshold=2.0)
        assert (result["y"].abs() < 10.0).all()

    def test_unknown_method_raises(self, clean_xy_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            remove_outliers(clean_xy_df, method="bogus")

    def test_clean_data_mostly_kept(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers(clean_xy_df, method="iqr", threshold=3.0)
        assert len(result) >= len(clean_xy_df) * 0.9

    def test_reset_index(self, xy_df_with_spike: pd.DataFrame) -> None:
        result = remove_outliers(xy_df_with_spike, method="iqr")
        assert result.index.tolist() == list(range(len(result)))


class TestRemoveOutliersByXRange:
    def test_removes_below_x_min(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers_by_x_range(clean_xy_df, x_min=0.5)
        assert (result["x"] >= 0.5).all()

    def test_removes_above_x_max(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers_by_x_range(clean_xy_df, x_max=0.5)
        assert (result["x"] <= 0.5).all()

    def test_both_bounds(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers_by_x_range(clean_xy_df, x_min=0.2, x_max=0.8)
        assert (result["x"] >= 0.2).all() and (result["x"] <= 0.8).all()

    def test_no_bounds_returns_all(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers_by_x_range(clean_xy_df)
        assert len(result) == len(clean_xy_df)

    def test_returns_dataframe(self, clean_xy_df: pd.DataFrame) -> None:
        assert isinstance(remove_outliers_by_x_range(clean_xy_df, x_min=0.0), pd.DataFrame)

    def test_reset_index(self, clean_xy_df: pd.DataFrame) -> None:
        result = remove_outliers_by_x_range(clean_xy_df, x_min=0.3)
        assert result.index.tolist() == list(range(len(result)))


class TestHysteresisModel:
    def test_output_length(self) -> None:
        x = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert len(out) == len(x)

    def test_stays_low_below_ascending_threshold(self) -> None:
        x = np.linspace(0.0, 0.7, 20)
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert (out == 0.0).all()

    def test_switches_high_above_ascending_threshold(self) -> None:
        x = np.array([0.0, 0.9, 0.9])
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert out[-1] == pytest.approx(1.0)

    def test_switches_low_below_descending_threshold(self) -> None:
        x = np.array([0.0, 0.9, 0.1])
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert out[-1] == pytest.approx(0.0)

    def test_hysteresis_band_stays_high(self) -> None:
        """Inside [desc, asc] band, state should not change."""
        x = np.array([0.0, 0.9, 0.5])  # 0.9 → high, 0.5 is in [0.2, 0.8] → stays high
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert out[-1] == pytest.approx(1.0)

    def test_custom_output_values(self) -> None:
        x = np.array([0.0, 0.9])
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2,
                               low_value=5.0, high_value=10.0)
        assert out[-1] == pytest.approx(10.0)

    def test_invalid_thresholds_raises(self) -> None:
        with pytest.raises(ValueError):
            hysteresis_model(np.array([0.5]), ascending_threshold=0.2, descending_threshold=0.8)

    def test_full_cycle_returns_to_low(self) -> None:
        x = np.concatenate([np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)])
        out = hysteresis_model(x, ascending_threshold=0.8, descending_threshold=0.2)
        assert out[-1] == pytest.approx(0.0)


class TestMultiLevelHysteresis:
    @pytest.fixture
    def two_level_params(self) -> dict:
        return {
            "thresholds": [(0.4, 0.1), (0.7, 0.3)],
            "low_values": [0.0, 0.5],
            "high_values": [0.5, 1.0],
        }

    def test_output_length(self, two_level_params: dict) -> None:
        x = np.linspace(0.0, 1.0, 30)
        out = multi_level_hysteresis(x, **two_level_params)
        assert len(out) == len(x)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            multi_level_hysteresis(
                np.array([0.5]),
                thresholds=[(0.4, 0.1)],
                low_values=[0.0, 0.5],
                high_values=[1.0],
            )

    def test_unordered_ascending_thresholds_raises(self) -> None:
        with pytest.raises(ValueError, match="ascending thresholds must be in ascending order"):
            multi_level_hysteresis(
                np.array([0.5]),
                thresholds=[(0.7, 0.1), (0.4, 0.3)],
                low_values=[0.0, 0.5],
                high_values=[0.5, 1.0],
            )

    def test_descending_not_less_than_ascending_raises(self) -> None:
        with pytest.raises(ValueError):
            multi_level_hysteresis(
                np.array([0.5]),
                thresholds=[(0.4, 0.5), (0.7, 0.8)],  # desc >= asc
                low_values=[0.0, 0.5],
                high_values=[0.5, 1.0],
            )

    def test_ramp_up_reaches_highest_level(self, two_level_params: dict) -> None:
        x = np.linspace(0.0, 1.0, 100)
        out = multi_level_hysteresis(x, **two_level_params)
        assert out[-1] == pytest.approx(1.0)

    def test_ramp_down_returns_to_low(self, two_level_params: dict) -> None:
        x = np.concatenate([np.linspace(0.0, 1.0, 50), np.linspace(1.0, 0.0, 50)])
        out = multi_level_hysteresis(x, **two_level_params)
        assert out[-1] == pytest.approx(0.0)


class TestRelayHysteresis:
    def test_output_length(self) -> None:
        x = np.linspace(-1.0, 1.0, 50)
        out = relay_hysteresis(x, center=0.0, width=0.4)
        assert len(out) == len(x)

    def test_starts_low(self) -> None:
        x = np.array([-0.5, -0.4, -0.3])
        out = relay_hysteresis(x, center=0.0, width=0.2)
        assert out[0] == pytest.approx(0.0)

    def test_switches_to_high_above_upper_edge(self) -> None:
        # center=0, width=0.4 → upper edge = center + width/2 = 0.2
        x = np.array([-1.0, 0.3])
        out = relay_hysteresis(x, center=0.0, width=0.4)
        assert out[-1] == pytest.approx(1.0)

    def test_switches_to_low_below_lower_edge(self) -> None:
        x = np.array([-1.0, 0.5, -0.5])
        out = relay_hysteresis(x, center=0.0, width=0.4)
        assert out[-1] == pytest.approx(0.0)

    def test_custom_output_values(self) -> None:
        x = np.array([-1.0, 0.5])
        out = relay_hysteresis(x, center=0.0, width=0.4, low_value=2.0, high_value=7.0)
        assert out[-1] == pytest.approx(7.0)


class TestContinuousHysteresis:
    def test_output_length(self) -> None:
        x = np.linspace(-1.0, 1.0, 50)
        out = continuous_hysteresis(x, center=0.0, width=0.4)
        assert len(out) == len(x)

    def test_output_bounded(self) -> None:
        x = np.linspace(-2.0, 2.0, 100)
        out = continuous_hysteresis(x, center=0.0, width=0.6, low_value=0.0, high_value=1.0)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_custom_output_values_bounded(self) -> None:
        x = np.linspace(-2.0, 2.0, 100)
        out = continuous_hysteresis(x, center=0.0, width=0.4, low_value=3.0, high_value=8.0)
        assert out.min() >= 3.0 - 1e-6
        assert out.max() <= 8.0 + 1e-6

    def test_transitions_are_smooth(self) -> None:
        """No consecutive output jump should be discontinuously large."""
        x = np.linspace(-1.0, 1.0, 200)
        out = continuous_hysteresis(x, center=0.0, width=0.4, slope=5)
        max_jump = np.max(np.abs(np.diff(out)))
        assert max_jump < 0.2


# ---------------------------------------------------------------------------
# breakingpoints.py — detect_changes (skipped if ruptures not installed)
# ---------------------------------------------------------------------------


@pytest.fixture
def step_series() -> pd.Series:
    """Time series with one clear step change at index 50."""
    data = np.concatenate([np.zeros(50), np.ones(50) * 10])
    return pd.Series(data, name="signal")


@pytest.fixture
def constant_series() -> pd.Series:
    return pd.Series(np.ones(60), name="constant")


@needs_ruptures
class TestDetectChanges:
    def test_returns_list(self, step_series: pd.Series) -> None:
        changes = detect_changes(step_series, algoname="Binseg", model="l2", n_bkps=1)
        assert isinstance(changes, list)

    def test_all_changes_within_bounds(self, step_series: pd.Series) -> None:
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
        data = np.concatenate([np.zeros(40), np.ones(40) * 5])
        ts = pd.Series(data)
        changes = detect_changes(ts, algoname="Dynp", model="l2", n_bkps=1)
        assert isinstance(changes, list)
        assert all(c < len(ts) for c in changes)

    def test_multiple_steps_detected(self) -> None:
        data = np.concatenate([np.zeros(30), np.ones(30) * 5, np.ones(30) * 10])
        ts = pd.Series(data)
        changes = detect_changes(ts, algoname="Binseg", model="l2", n_bkps=2)
        assert len(changes) >= 2


# ---------------------------------------------------------------------------
# signal.py — normalize_signal, _otsu_threshold, binarize_signal
# ---------------------------------------------------------------------------

from python_magnetrun.processing.signal import (  # noqa: E402
    _otsu_threshold,
    binarize_signal,
    normalize_signal,
)


class TestNormalizeSignal:
    def test_all_zeros_unchanged(self) -> None:
        data = np.zeros(10)
        result = normalize_signal(data)
        np.testing.assert_array_equal(result, data)

    def test_max_abs_is_one(self) -> None:
        data = np.array([1.0, -2.0, 0.5])
        result = normalize_signal(data)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_known_values(self) -> None:
        data = np.array([2.0, 4.0, 6.0])
        result = normalize_signal(data)
        np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1.0])

    def test_output_shape_preserved(self) -> None:
        data = np.ones(20)
        result = normalize_signal(data)
        assert result.shape == data.shape

    def test_negative_values(self) -> None:
        data = np.array([-3.0, 0.0, 3.0])
        result = normalize_signal(data)
        assert np.max(np.abs(result)) == pytest.approx(1.0)


class TestOtsuThreshold:
    def test_empty_returns_zero(self) -> None:
        assert _otsu_threshold(np.array([])) == pytest.approx(0.0)

    def test_all_zeros_returns_zero(self) -> None:
        assert _otsu_threshold(np.zeros(10)) == pytest.approx(0.0)

    def test_bimodal_signal_finds_threshold_between_modes(self) -> None:
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 0.01, 200)
        signal = rng.uniform(0.5, 1.0, 200)
        data = np.concatenate([noise, signal])
        threshold = _otsu_threshold(data)
        assert 0.01 < threshold < 0.5

    def test_returns_float(self) -> None:
        data = np.array([0.1, 0.5, 0.9])
        assert isinstance(_otsu_threshold(data), float)


class TestBinarizeSignal:
    @pytest.fixture
    def bimodal(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        off = rng.uniform(0.0, 0.05, 100)
        on = rng.uniform(0.8, 1.0, 100)
        return np.concatenate([off, on])

    def test_fixed_output_is_binary(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, tolerance=0.1, method="fixed")
        assert set(np.unique(result)).issubset({0, 1})

    def test_otsu_output_is_binary(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="otsu")
        assert set(np.unique(result)).issubset({0, 1})

    def test_noise_output_is_binary(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="noise")
        assert set(np.unique(result)).issubset({0, 1})

    def test_unknown_method_raises(self, bimodal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            binarize_signal(bimodal, method="bogus")

    def test_output_length_matches_input(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="fixed")
        assert len(result) == len(bimodal)

    def test_all_zeros_all_off(self) -> None:
        result = binarize_signal(np.zeros(50), method="fixed", tolerance=0.005)
        assert np.all(result == 0)

    def test_normalize_false_path(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="fixed", tolerance=0.1, normalize=False)
        assert set(np.unique(result)).issubset({0, 1})

    def test_otsu_off_population_classified_correctly(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="otsu")
        assert result[:100].mean() < 0.5

    def test_otsu_on_population_classified_correctly(self, bimodal: np.ndarray) -> None:
        result = binarize_signal(bimodal, method="otsu")
        assert result[100:].mean() > 0.5
