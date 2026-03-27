"""
Tests for the synchronization module.

These tests verify the time synchronization functionality including
basic synchronization, lag computation, and regime matching.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd

from python_magnetrun.analysis.synchronization import (
    LagResult,
    RegimeMatch,
    # Dataclasses
    SyncResult,
    add_time_column,
    apply_lag_correction,
    check_lag_reliability,
    compute_regime_score,
    compute_simple_timeshift,
    find_best_matching_regime,
    get_timestamp_info,
    # Functions
    synchronize_data,
)


class TestSyncResult:
    """Test SyncResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = SyncResult(
            timeshift=timedelta(seconds=10),
            df=pd.DataFrame({"a": [1, 2, 3]}),
            source_t0=datetime(2024, 11, 6, 16, 0, 0),
            target_t0=datetime(2024, 11, 6, 16, 0, 10),
        )
        assert result.timeshift == timedelta(seconds=10)
        assert len(result.df) == 3

    def test_shift_seconds(self):
        """Test shift_seconds property."""
        result = SyncResult(
            timeshift=timedelta(seconds=5.5),
            df=pd.DataFrame(),
            source_t0=datetime.now(),
            target_t0=datetime.now(),
        )
        assert result.shift_seconds == 5.5


class TestLagResult:
    """Test LagResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        result = LagResult(
            lag=timedelta(seconds=2),
            correlation=0.95,
            confidence=0.9,
            method="cross_correlation",
        )
        assert result.lag == timedelta(seconds=2)
        assert result.correlation == 0.95

    def test_seconds_property(self):
        """Test seconds property."""
        result = LagResult(lag=timedelta(seconds=-3.5))
        assert result.seconds == -3.5

    def test_defaults(self):
        """Test default values."""
        result = LagResult(lag=timedelta(seconds=1))
        assert result.correlation == 0.0
        assert result.confidence == 0.0
        assert result.method == "cross_correlation"


class TestRegimeMatch:
    """Test RegimeMatch dataclass."""

    def test_creation(self):
        """Test basic creation."""
        match = RegimeMatch(
            regime="U",
            match_regime="U",
            score=0.5,
            lags=(1.0, 2.0),
            indices=(0, 1),
        )
        assert match.regime == "U"
        assert match.score == 0.5
        assert match.lags == (1.0, 2.0)

    def test_is_reliable_true(self):
        """Test is_reliable with finite lags."""
        match = RegimeMatch(
            regime="U",
            match_regime="U",
            score=0.5,
            lags=(1.0, 2.0),
            indices=(0, 1),
        )
        assert match.is_reliable

    def test_is_reliable_false(self):
        """Test is_reliable with infinite lags."""
        match = RegimeMatch(
            regime="P",
            match_regime="P",
            score=float("inf"),
            lags=(float("inf"), float("inf")),
            indices=(0, 0),
        )
        assert not match.is_reliable


class TestSynchronizeData:
    """Test synchronize_data function."""

    def test_basic_sync(self):
        """Test basic synchronization."""
        # Create test DataFrame
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [
                    t0,
                    t0 + timedelta(seconds=1),
                    t0 + timedelta(seconds=2),
                ],
                "value": [1.0, 2.0, 3.0],
                "t": [0.0, 1.0, 2.0],
            }
        )

        # Sync to new reference time
        target_t0 = datetime(2024, 11, 6, 16, 0, 10)
        timeshift, df_synced = synchronize_data(df, target_t0)

        # Check timeshift
        assert timeshift == timedelta(seconds=10)

        # Check that timestamps were shifted
        assert df_synced["timestamp"].iloc[0] == target_t0

        # Check that t column was recalculated
        assert df_synced["t"].iloc[0] == 0.0
        assert df_synced["t"].iloc[1] == 1.0

    def test_negative_timeshift(self):
        """Test synchronization with negative timeshift."""
        t0 = datetime(2024, 11, 6, 16, 0, 10)
        df = pd.DataFrame(
            {
                "timestamp": [t0],
                "value": [1.0],
                "t": [0.0],
            }
        )

        target_t0 = datetime(2024, 11, 6, 16, 0, 0)
        timeshift, df_synced = synchronize_data(df, target_t0)

        assert timeshift == timedelta(seconds=-10)

    def test_preserves_data(self):
        """Test that data values are preserved."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0],
                "value": [42.0],
                "other": ["test"],
                "t": [0.0],
            }
        )

        _, df_synced = synchronize_data(df, t0 + timedelta(seconds=5))

        assert df_synced["value"].iloc[0] == 42.0
        assert df_synced["other"].iloc[0] == "test"


class TestApplyLagCorrection:
    """Test apply_lag_correction function."""

    def test_timedelta_lag(self):
        """Test with timedelta lag."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0, t0 + timedelta(seconds=1)],
                "value": [1.0, 2.0],
            }
        )

        lag = timedelta(seconds=5)
        df_corrected = apply_lag_correction(df, lag)

        # Lag is subtracted
        expected_t0 = t0 - timedelta(seconds=5)
        assert df_corrected["timestamp"].iloc[0] == expected_t0

    def test_float_lag(self):
        """Test with float lag (seconds)."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0],
                "value": [1.0],
            }
        )

        df_corrected = apply_lag_correction(df, 3.0)

        expected_t0 = t0 - timedelta(seconds=3)
        assert df_corrected["timestamp"].iloc[0] == expected_t0

    def test_with_reference_t0(self):
        """Test lag correction with time column recalculation."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0, t0 + timedelta(seconds=1)],
                "value": [1.0, 2.0],
                "t": [0.0, 1.0],
            }
        )

        reference_t0 = t0 - timedelta(seconds=5)
        df_corrected = apply_lag_correction(df, 2.0, reference_t0=reference_t0)

        # t column should be recalculated relative to reference
        # After lag correction of 2s, first timestamp is t0 - 2s
        # Relative to reference_t0 (t0 - 5s), that's 3s
        assert df_corrected["t"].iloc[0] == 3.0


class TestComputeRegimeScore:
    """Test compute_regime_score function."""

    def test_matching_regimes(self):
        """Test with matching regime types."""
        score, tscore, lags = compute_regime_score(
            regime="U",
            value=(0.0, 100.0),
            time=(0.0, 10.0),
            reference_regime="U",
            reference_value=(0.0, 100.0),
            reference_time=(1.0, 11.0),
        )

        # Same value range -> score = 0
        assert score == 0.0

        # Same duration -> tscore = 0
        assert tscore == 0.0

        # Lags should be -1.0, -1.0 (source ahead of reference)
        assert lags == (-1.0, -1.0)

    def test_non_matching_regimes(self):
        """Test with different regime types."""
        score, tscore, lags = compute_regime_score(
            regime="U",
            value=(0.0, 100.0),
            time=(0.0, 10.0),
            reference_regime="D",  # Different!
            reference_value=(100.0, 0.0),
            reference_time=(0.0, 10.0),
        )

        # Should return infinity for non-matching regimes
        assert score == float("inf")
        assert lags == (float("inf"), float("inf"))

    def test_different_value_ranges(self):
        """Test with different value ranges."""
        score, tscore, lags = compute_regime_score(
            regime="U",
            value=(0.0, 100.0),  # Range = 100
            time=(0.0, 10.0),
            reference_regime="U",
            reference_value=(0.0, 50.0),  # Range = 50
            reference_time=(0.0, 10.0),
        )

        # Score should be |100 - 50| = 50
        assert score == 50.0


class TestFindBestMatchingRegime:
    """Test find_best_matching_regime function."""

    def test_with_mock_signatures(self):
        """Test with mock signature objects."""
        # Create mock signatures
        signature = Mock()
        signature.regimes = ["U", "P", "D"]
        signature.times = [0.0, 10.0, 20.0, 30.0]
        signature.values = [0.0, 100.0, 100.0, 0.0]

        reference = Mock()
        reference.regimes = ["U", "P", "D"]
        reference.times = [1.0, 11.0, 21.0, 31.0]  # Shifted by 1s
        reference.values = [0.0, 100.0, 100.0, 0.0]

        matches = find_best_matching_regime(signature, reference)

        # Should find matches for U and D regimes
        assert len(matches) >= 1

        # Check first match
        match = matches[0]
        assert match.regime == "U"
        assert match.match_regime == "U"

    def test_empty_signatures(self):
        """Test with empty signatures."""
        signature = Mock()
        signature.regimes = []
        signature.times = []
        signature.values = []

        reference = Mock()
        reference.regimes = []
        reference.times = []
        reference.values = []

        matches = find_best_matching_regime(signature, reference)
        assert matches == []


class TestCheckLagReliability:
    """Test check_lag_reliability function."""

    def test_reliable_lag(self):
        """Test with reliable lag (small ratio)."""
        # Lag is 5% of duration
        assert check_lag_reliability(lags=(5.0, 5.0), duration=100.0, threshold_ratio=0.2)

    def test_unreliable_lag(self):
        """Test with unreliable lag (large ratio)."""
        # Lag is 50% of duration
        assert not check_lag_reliability(lags=(50.0, 50.0), duration=100.0, threshold_ratio=0.2)

    def test_zero_duration(self):
        """Test with zero duration."""
        assert not check_lag_reliability(lags=(1.0, 1.0), duration=0.0)

    def test_default_threshold(self):
        """Test with default threshold ratio."""
        # Should use LAG_THRESHOLD_RATIO (0.2)
        result = check_lag_reliability(lags=(10.0, 10.0), duration=100.0)
        # 10% is less than 20% threshold
        assert result


class TestComputeSimpleTimeshift:
    """Test compute_simple_timeshift function."""

    def test_positive_shift(self):
        """Test positive timeshift."""
        source = datetime(2024, 11, 6, 16, 0, 0)
        target = datetime(2024, 11, 6, 16, 0, 10)

        shift = compute_simple_timeshift(source, target)

        assert shift == timedelta(seconds=10)

    def test_negative_shift(self):
        """Test negative timeshift."""
        source = datetime(2024, 11, 6, 16, 0, 10)
        target = datetime(2024, 11, 6, 16, 0, 0)

        shift = compute_simple_timeshift(source, target)

        assert shift == timedelta(seconds=-10)


class TestAddTimeColumn:
    """Test add_time_column function."""

    def test_add_new_column(self):
        """Test adding t column to DataFrame without one."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
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

        df_result = add_time_column(df, t0)

        assert "t" in df_result.columns
        assert df_result["t"].iloc[0] == 0.0
        assert df_result["t"].iloc[1] == 1.0
        assert df_result["t"].iloc[2] == 2.0

    def test_replace_existing_column(self):
        """Test replacing existing t column."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0, t0 + timedelta(seconds=1)],
                "value": [1.0, 2.0],
                "t": [999.0, 999.0],  # Old values
            }
        )

        df_result = add_time_column(df, t0)

        # t should be recalculated, not 999
        assert df_result["t"].iloc[0] == 0.0
        assert df_result["t"].iloc[1] == 1.0


class TestGetTimestampInfo:
    """Test get_timestamp_info function."""

    def test_basic_info(self):
        """Test getting timestamp info."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [
                    t0,
                    t0 + timedelta(seconds=30),
                    t0 + timedelta(seconds=60),
                ],
                "value": [1.0, 2.0, 3.0],
            }
        )

        info = get_timestamp_info(df)

        assert info["t0"] == t0
        assert info["t_end"] == t0 + timedelta(seconds=60)
        assert info["duration"] == 60.0
        assert info["n_samples"] == 3

    def test_single_row(self):
        """Test with single row DataFrame."""
        t0 = datetime(2024, 11, 6, 16, 0, 0)
        df = pd.DataFrame(
            {
                "timestamp": [t0],
                "value": [1.0],
            }
        )

        info = get_timestamp_info(df)

        assert info["duration"] == 0.0
        assert info["n_samples"] == 1


class TestIntegration:
    """Integration tests for synchronization workflow."""

    def test_full_sync_workflow(self):
        """Test complete synchronization workflow."""
        # Create source data
        source_t0 = datetime(2024, 11, 6, 16, 0, 0)
        df_source = pd.DataFrame(
            {
                "timestamp": [source_t0 + timedelta(seconds=i) for i in range(10)],
                "value": list(range(10)),
                "t": list(range(10)),
            }
        )

        # Create target reference
        target_t0 = datetime(2024, 11, 6, 16, 0, 5)

        # Synchronize
        timeshift, df_synced = synchronize_data(df_source, target_t0)

        # Verify
        assert timeshift.total_seconds() == 5.0
        assert df_synced["timestamp"].iloc[0] == target_t0

        # Apply additional lag correction
        lag = 1.0
        df_corrected = apply_lag_correction(df_synced, lag, reference_t0=target_t0)

        # First timestamp should be target_t0 - 1s
        expected_t0 = target_t0 - timedelta(seconds=1)
        assert df_corrected["timestamp"].iloc[0] == expected_t0
