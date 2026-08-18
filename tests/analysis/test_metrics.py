"""
Tests for the metrics module.

These tests verify distance metrics, cross-correlation,
and DTW functionality.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from python_magnetrun.analysis.metrics import (
    CorrelationResult,
    # Dataclasses
    DistanceResult,
    DTWResult,
    TLCCResult,
    calc_correlation,
    # Distance functions
    calc_euclidean,
    calc_mae,
    calc_mape,
    # Convenience
    compare_series,
    compute_all_distances,
    compute_pearson_correlation,
    compute_rolling_tlcc,
    compute_tlcc,
    compute_windowed_tlcc,
    # Cross-correlation functions
    crosscorr,
)


class TestDistanceResult:
    """Test DistanceResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = DistanceResult(
            euclidean=1.0,
            mae=0.5,
            mape=0.1,
            correlation=0.95,
        )
        assert result.euclidean == 1.0
        assert result.mae == 0.5
        assert result.mape == 0.1
        assert result.correlation == 0.95

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = DistanceResult(
            euclidean=1.0,
            mae=0.5,
            mape=0.1,
            correlation=0.95,
        )
        d = result.to_dict()
        assert d["euclidean"] == 1.0
        assert d["correlation"] == 0.95
        # rmse, max_error, mahalanobis are optional fields with NaN defaults
        assert {"euclidean", "mae", "mape", "correlation", "rmse", "max_error", "mahalanobis"} == set(d)


class TestDTWResult:
    """Test DTWResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = DTWResult(
            distance=10.5,
            path=[(0, 0), (1, 1), (2, 2)],
            similarity_score=3.5,
            normalized_distance=0.5,
        )
        assert result.distance == 10.5
        assert result.path_length == 3

    def test_empty_path(self):
        """Test with empty path."""
        result = DTWResult(distance=5.0)
        assert result.path == []
        assert result.path_length == 0


class TestCorrelationResult:
    """Test CorrelationResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = CorrelationResult(
            pearson_r=0.95,
            p_value=0.001,
            lag=5,
            max_correlation=0.98,
        )
        assert result.pearson_r == 0.95
        assert result.p_value == 0.001


class TestTLCCResult:
    """Test TLCCResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        correlations = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
        lags = np.array([-2, -1, 0, 1, 2])

        result = TLCCResult(
            correlations=correlations,
            lags=lags,
            optimal_lag=0,
            optimal_correlation=0.9,
            offset=0.0,
        )
        assert result.optimal_lag == 0
        assert result.optimal_correlation == 0.9


class TestCalcEuclidean:
    """Test calc_euclidean function."""

    def test_identical_arrays(self):
        """Distance between identical arrays is 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert calc_euclidean(a, a) == 0.0

    def test_simple_case(self):
        """Test simple distance calculation."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([3.0, 4.0, 0.0])
        # sqrt(9 + 16 + 0) = 5
        assert calc_euclidean(a, b) == 5.0

    def test_single_element(self):
        """Test with single element arrays."""
        a = np.array([5.0])
        b = np.array([8.0])
        assert calc_euclidean(a, b) == 3.0


class TestCalcMAE:
    """Test calc_mae function."""

    def test_identical_arrays(self):
        """MAE between identical arrays is 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert calc_mae(a, a) == 0.0

    def test_simple_case(self):
        """Test simple MAE calculation."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        # mean(|1|, |1|, |1|) = 1.0
        assert calc_mae(a, b) == 1.0

    def test_mixed_errors(self):
        """Test with mixed positive/negative errors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 4.0, 3.0])
        # mean(|1|, |2|, |0|) = 1.0
        assert calc_mae(a, b) == 1.0


class TestCalcMAPE:
    """Test calc_mape function."""

    def test_identical_arrays(self):
        """MAPE between identical arrays is 0."""
        a = np.array([1.0, 2.0, 3.0])
        assert calc_mape(a, a) == 0.0

    def test_simple_case(self):
        """Test simple MAPE calculation."""
        a = np.array([100.0, 200.0])
        b = np.array([110.0, 180.0])
        # mean(|10/100|, |20/200|) = mean(0.1, 0.1) = 0.1
        assert abs(calc_mape(a, b) - 0.1) < 1e-10


class TestCalcCorrelation:
    """Test calc_correlation function."""

    def test_perfect_correlation(self):
        """Perfectly correlated arrays have r=1."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert abs(calc_correlation(a, b) - 1.0) < 1e-10

    def test_perfect_negative_correlation(self):
        """Perfectly anti-correlated arrays have r=-1."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(calc_correlation(a, b) - (-1.0)) < 1e-10

    def test_no_correlation(self):
        """Test with uncorrelated data."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 1.0, 1.0, 1.0])  # Constant - zero variance
        # When one array is constant, correlation is 0
        assert calc_correlation(a, b) == 0.0


class TestComputeAllDistances:
    """Test compute_all_distances function."""

    def test_returns_distance_result(self):
        """Should return DistanceResult dataclass."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        result = compute_all_distances(a, b)

        assert isinstance(result, DistanceResult)
        assert result.euclidean > 0
        assert result.mae > 0
        assert result.correlation > 0.9


class TestCrosscorr:
    """Test crosscorr function."""

    def test_zero_lag(self):
        """Test correlation at zero lag."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([1.1, 2.0, 3.1, 3.9, 5.0])

        corr = crosscorr(x, y, lag=0)
        assert corr > 0.99

    def test_positive_lag(self):
        """Test with positive lag."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])  # y is x shifted by 1

        # At lag=1, y.shift(1) aligns with x
        corr_lag1 = crosscorr(x, y, lag=1)
        corr_lag0 = crosscorr(x, y, lag=0)

        # Correlation should be higher at lag=1
        assert corr_lag1 > corr_lag0


class TestComputeTLCC:
    """Test compute_tlcc function."""

    def test_aligned_series(self):
        """Test with aligned series (optimal lag should be ~0)."""
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(100)))
        y = x + np.random.randn(100) * 0.1  # Small noise

        result = compute_tlcc(x, y, seconds=1, fps=10)

        assert isinstance(result, TLCCResult)
        assert abs(result.optimal_lag) <= 2  # Near zero
        assert result.optimal_correlation > 0.9

    def test_shifted_series(self):
        """Test with shifted series."""
        np.random.seed(42)
        base = pd.Series(np.cumsum(np.random.randn(100)))
        x = base.iloc[5:].reset_index(drop=True)
        y = base.iloc[:-5].reset_index(drop=True)

        result = compute_tlcc(x, y, seconds=1, fps=10)

        # Should detect the shift
        assert isinstance(result, TLCCResult)


class TestComputeWindowedTLCC:
    """Test compute_windowed_tlcc function."""

    def test_basic_windowed(self):
        """Test basic windowed TLCC."""
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(200)))
        y = x + np.random.randn(200) * 0.1

        result = compute_windowed_tlcc(x, y, seconds=0.5, fps=10, n_windows=4)

        assert result.shape[0] == 4  # 4 windows
        assert result.shape[1] > 0  # Multiple lags


class TestComputeRollingTLCC:
    """Test compute_rolling_tlcc function."""

    def test_basic_rolling(self):
        """Test basic rolling TLCC."""
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(500)))
        y = x + np.random.randn(500) * 0.1

        result = compute_rolling_tlcc(
            x, y, seconds=0.5, fps=10, window_size=100, step_size=50, max_samples=400
        )

        assert result.shape[0] > 0  # At least one window
        assert result.shape[1] > 0  # Multiple lags


class TestComputePearsonCorrelation:
    """Test compute_pearson_correlation function."""

    def test_high_correlation(self):
        """Test with highly correlated data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.0, 3.1, 3.9, 5.0])

        result = compute_pearson_correlation(x, y)

        assert isinstance(result, CorrelationResult)
        assert result.pearson_r > 0.99
        assert result.p_value < 0.01

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        result = compute_pearson_correlation(x, y)

        # Should still compute (with NaN removed)
        assert not np.isnan(result.pearson_r)


class TestCompareSeries:
    """Test compare_series convenience function."""

    def test_basic_comparison(self):
        """Test basic series comparison."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.0, 3.1, 3.9, 5.0])

        result = compare_series(actual, predicted, compute_dtw=False)

        assert "distances" in result
        assert "pearson" in result
        assert isinstance(result["distances"], DistanceResult)
        assert isinstance(result["pearson"], CorrelationResult)

    @patch("python_magnetrun.analysis.metrics.compute_dtw_distance")
    def test_with_dtw(self, mock_dtw):
        """Test with DTW computation (mocked)."""
        mock_dtw.return_value = DTWResult(distance=1.5)

        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.0, 3.1])

        result = compare_series(actual, predicted, compute_dtw=True)

        mock_dtw.assert_called_once()
        assert "dtw" in result


class TestDTWFunctions:
    """Test DTW-related functions (with optional dtaidistance)."""

    def test_dtw_import_error_handling(self):
        """Test graceful handling when dtaidistance not available."""
        from python_magnetrun.analysis.metrics import compute_dtw_distance

        # This will either work (if dtaidistance is installed)
        # or raise ImportError (if not)
        try:
            result = compute_dtw_distance(np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.0, 3.1]))
            assert isinstance(result, DTWResult)
        except ImportError as e:
            assert "dtaidistance" in str(e)


class TestIntegration:
    """Integration tests for metrics module."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        np.random.seed(42)

        # Create test time series
        t = np.linspace(0, 10, 100)
        actual = np.sin(t) + np.random.randn(100) * 0.1
        predicted = np.sin(t + 0.1) + np.random.randn(100) * 0.1

        # Compute all distances
        distances = compute_all_distances(actual, predicted)
        assert distances.correlation > 0.9

        # Compute TLCC
        x = pd.Series(actual)
        y = pd.Series(predicted)
        tlcc_result = compute_tlcc(x, y, seconds=1, fps=10)
        assert abs(tlcc_result.optimal_lag) <= 5

        # Full comparison
        comparison = compare_series(actual, predicted, compute_dtw=False)
        assert comparison["distances"].correlation > 0.9
