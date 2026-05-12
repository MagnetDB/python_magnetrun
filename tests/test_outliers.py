"""Tests for python_magnetrun.hybrid.outliers (canonical outlier module)."""

import numpy as np
import pytest

from python_magnetrun.hybrid.outliers import (
    OutlierDetector,
    OutlierResult,
    analyze_outliers,
    detect_outliers,
    find_outlier_segments,
    get_outlier_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200
SPIKE_INDICES = [10, 50, 100, 150, 190]
SPIKE_VALUE = 1_000.0


@pytest.fixture
def clean_series() -> np.ndarray:
    t = np.linspace(0, 4 * np.pi, N)
    return np.sin(t) + RNG.normal(0, 0.05, N)


@pytest.fixture
def series_with_outliers(clean_series: np.ndarray) -> np.ndarray:
    data = clean_series.copy()
    data[SPIKE_INDICES] = SPIKE_VALUE
    return data


@pytest.fixture
def time_array() -> np.ndarray:
    return np.linspace(0, 10, N)


# ---------------------------------------------------------------------------
# TestOutlierDetector
# ---------------------------------------------------------------------------


class TestOutlierDetector:
    @pytest.mark.parametrize("method", ["iqr", "zscore", "mad"])
    def test_detects_spikes(self, series_with_outliers, method):
        detector = OutlierDetector(method)
        result = detector.detect(series_with_outliers)
        assert isinstance(result, OutlierResult)
        assert result.n_outliers > 0
        # All injected spikes must be flagged
        for idx in SPIKE_INDICES:
            assert result.mask[idx], f"{method}: spike at index {idx} not detected"

    def test_isolation_forest_detects_spikes(self, series_with_outliers):
        detector = OutlierDetector("isolation_forest", threshold=0.05)
        result = detector.detect(series_with_outliers)
        assert isinstance(result, OutlierResult)
        assert result.n_outliers > 0
        for idx in SPIKE_INDICES:
            assert result.mask[idx], f"isolation_forest: spike at index {idx} not detected"

    def test_isolation_forest_rejects_rolling(self, series_with_outliers):
        detector = OutlierDetector("isolation_forest", window_size=30)
        with pytest.raises(ValueError, match="isolation_forest does not support rolling"):
            detector.detect(series_with_outliers)

    def test_percentile_detects_extremes(self):
        # percentile clips at the 1st/99th percentile — use a value strictly
        # beyond the computed bound (not at it, which uses strict >)
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 500)
        data[0] = 1e6  # far beyond any reasonable 99th percentile
        detector = OutlierDetector("percentile")
        result = detector.detect(data)
        assert isinstance(result, OutlierResult)
        assert result.mask[0], "extreme value not flagged by percentile method"

    @pytest.mark.parametrize("method", ["iqr", "zscore", "mad"])
    def test_no_false_positives_on_clean_data(self, clean_series, method):
        detector = OutlierDetector(method)
        result = detector.detect(clean_series)
        # Allow at most 1% false positives on Gaussian noise
        assert result.outlier_ratio < 0.01, (
            f"{method}: {result.outlier_ratio:.2%} false-positive rate on clean data"
        )

    @pytest.mark.parametrize("method", ["iqr", "zscore", "mad", "percentile"])
    def test_rolling_detection(self, series_with_outliers, method):
        detector = OutlierDetector(method, window_size=30)
        result = detector.detect(series_with_outliers)
        assert isinstance(result, OutlierResult)
        assert result.n_outliers > 0

    def test_result_totals_consistent(self, series_with_outliers):
        detector = OutlierDetector("iqr")
        result = detector.detect(series_with_outliers)
        assert result.n_total == N
        assert result.n_outliers == int(result.mask.sum())

    def test_indices_match_mask(self, series_with_outliers):
        detector = OutlierDetector("zscore")
        result = detector.detect(series_with_outliers)
        expected = np.where(result.mask)[0]
        np.testing.assert_array_equal(result.indices, expected)


# ---------------------------------------------------------------------------
# TestOutlierResult
# ---------------------------------------------------------------------------


class TestOutlierResult:
    @pytest.fixture
    def result(self, series_with_outliers) -> OutlierResult:
        return OutlierDetector("iqr").detect(series_with_outliers)

    def test_outlier_ratio(self, result):
        assert 0 < result.outlier_ratio <= 1
        assert result.outlier_ratio == pytest.approx(result.n_outliers / result.n_total)

    def test_outlier_percentage(self, result):
        assert result.outlier_percentage == pytest.approx(result.outlier_ratio * 100)

    def test_get_clean_mask(self, result):
        clean = result.get_clean_mask()
        assert clean.dtype == bool
        assert clean.sum() == result.n_total - result.n_outliers

    def test_summary_returns_string(self, result):
        s = result.summary()
        assert isinstance(s, str)
        assert str(result.n_outliers) in s

    def test_apply_remove(self, series_with_outliers, result):
        clean = result.apply_to_data(series_with_outliers, strategy="remove")
        assert isinstance(clean, np.ndarray)
        assert len(clean) == N - result.n_outliers
        assert SPIKE_VALUE not in clean

    def test_apply_nan(self, series_with_outliers, result):
        out = result.apply_to_data(series_with_outliers, strategy="nan")
        assert len(out) == N
        assert np.any(np.isnan(out))
        assert np.isnan(out[SPIKE_INDICES[0]])

    def test_apply_interpolate(self, series_with_outliers, result):
        out = result.apply_to_data(series_with_outliers, strategy="interpolate")
        assert len(out) == N
        assert not np.any(np.isnan(out))
        # Interpolated values must be far below the original spike
        for idx in SPIKE_INDICES:
            assert out[idx] < SPIKE_VALUE / 10

    def test_apply_median(self, series_with_outliers, result):
        out = result.apply_to_data(series_with_outliers, strategy="median")
        assert len(out) == N
        assert SPIKE_VALUE not in out[SPIKE_INDICES]

    def test_apply_remove_with_time(self, series_with_outliers, time_array, result):
        clean_data, clean_time = result.apply_to_data(
            series_with_outliers, time=time_array, strategy="remove"
        )
        assert len(clean_data) == len(clean_time)
        assert len(clean_data) == N - result.n_outliers


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_detect_outliers_functional_api(self, series_with_outliers):
        mask = detect_outliers(series_with_outliers, method="iqr", threshold=1.5)
        assert mask.dtype == bool
        assert len(mask) == N
        for idx in SPIKE_INDICES:
            assert mask[idx], f"spike at {idx} not detected"

    def test_detect_outliers_returns_false_on_clean(self, clean_series):
        mask = detect_outliers(clean_series, method="iqr")
        assert mask.sum() / N < 0.01

    def test_find_outlier_segments_empty_mask(self):
        mask = np.zeros(100, dtype=bool)
        segments = find_outlier_segments(mask)
        assert segments == []

    def test_find_outlier_segments_single_block(self):
        mask = np.zeros(100, dtype=bool)
        mask[20:30] = True
        segments = find_outlier_segments(mask, min_gap=5)
        assert len(segments) == 1
        start, end = segments[0]
        assert start == 20
        assert end == 30

    def test_find_outlier_segments_merge_close(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:15] = True
        mask[17:22] = True  # gap of 2, will be merged with min_gap=5
        segments = find_outlier_segments(mask, min_gap=5)
        assert len(segments) == 1

    def test_find_outlier_segments_keep_separate(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:15] = True
        mask[40:45] = True  # gap >> min_gap
        segments = find_outlier_segments(mask, min_gap=5)
        assert len(segments) == 2

    def test_get_outlier_summary_default_methods(self, series_with_outliers):
        summary = get_outlier_summary(series_with_outliers)
        assert set(summary.keys()) == {"iqr", "zscore", "mad"}
        for result in summary.values():
            assert isinstance(result, OutlierResult)
            assert result.n_outliers > 0

    def test_get_outlier_summary_custom_methods(self, series_with_outliers):
        summary = get_outlier_summary(series_with_outliers, methods=["iqr", "percentile"])
        assert set(summary.keys()) == {"iqr", "percentile"}

    def test_analyze_outliers_keys(self, series_with_outliers, time_array):
        analysis = analyze_outliers(
            series_with_outliers, time=time_array, method="iqr", threshold=1.5
        )
        required = {
            "method", "threshold", "n_total", "n_outliers",
            "outlier_percentage", "statistics", "n_segments", "segments",
        }
        assert required.issubset(analysis.keys())
        assert analysis["n_total"] == N
        assert analysis["n_outliers"] > 0
        assert "outlier_stats" in analysis
        assert "outlier_time_stats" in analysis

    def test_analyze_outliers_no_time(self, series_with_outliers):
        analysis = analyze_outliers(series_with_outliers, method="mad")
        assert "outlier_time_stats" not in analysis

    def test_isolation_forest_via_detect_outliers(self, series_with_outliers):
        mask = detect_outliers(series_with_outliers, method="isolation_forest", threshold=0.05)
        assert mask.dtype == bool
        assert mask.sum() > 0
        for idx in SPIKE_INDICES:
            assert mask[idx], f"spike at {idx} not flagged via detect_outliers"
