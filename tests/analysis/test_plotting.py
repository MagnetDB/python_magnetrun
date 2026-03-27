"""
Tests for the plotting module.

These tests verify downsampling functionality and plot creation.
Note: Most plotting tests use mocking to avoid actual display/save operations.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from python_magnetrun.analysis.plotting import (
    DEFAULT_COLORS,
    # Constants
    DEFAULT_STYLE,
    PlotColors,
    # Dataclasses
    PlotStyle,
    create_figure_grid,
    downsample_dataframe,
    # Downsampling functions
    downsample_for_plot,
    downsample_minmax,
    estimate_downsample_percent,
    plot_incidents_markers,
    # Plotting functions
    plot_regimes,
)


class TestPlotStyle:
    """Test PlotStyle dataclass."""

    def test_default_values(self):
        """Test default PlotStyle values."""
        style = PlotStyle()
        assert style.figsize == (12, 5)
        assert style.dpi == 300
        assert style.grid is True

    def test_custom_values(self):
        """Test custom PlotStyle values."""
        style = PlotStyle(figsize=(10, 8), dpi=150, grid=False)
        assert style.figsize == (10, 8)
        assert style.dpi == 150
        assert style.grid is False


class TestPlotColors:
    """Test PlotColors dataclass."""

    def test_default_values(self):
        """Test default PlotColors values."""
        colors = PlotColors()
        assert colors.overview == "blue"
        assert colors.archive == "red"
        assert colors.pupitre == "green"

    def test_get_regime_color(self):
        """Test regime color mapping."""
        colors = PlotColors()
        assert colors.get_regime_color("U") == colors.regime_up
        assert colors.get_regime_color("D") == colors.regime_down
        assert colors.get_regime_color("P") == colors.regime_plateau
        assert colors.get_regime_color("X") == "gray"  # Unknown


class TestDefaultInstances:
    """Test default style and color instances."""

    def test_default_style_exists(self):
        """Test DEFAULT_STYLE is properly configured."""
        assert DEFAULT_STYLE is not None
        assert isinstance(DEFAULT_STYLE, PlotStyle)

    def test_default_colors_exists(self):
        """Test DEFAULT_COLORS is properly configured."""
        assert DEFAULT_COLORS is not None
        assert isinstance(DEFAULT_COLORS, PlotColors)


class TestDownsampleForPlot:
    """Test downsample_for_plot function."""

    def test_no_downsampling(self):
        """Test with 100% (no downsampling)."""
        x = np.arange(100)
        y = np.sin(x)

        x_ds, y_ds = downsample_for_plot(x, y, percent=100.0)

        assert len(x_ds) == 100
        np.testing.assert_array_equal(x_ds, x)
        np.testing.assert_array_equal(y_ds, y)

    def test_50_percent_downsampling(self):
        """Test 50% downsampling."""
        x = np.arange(100)
        y = np.sin(x)

        x_ds, y_ds = downsample_for_plot(x, y, percent=50.0)

        # Should have approximately 50 points
        assert len(x_ds) == 50
        assert len(y_ds) == 50

    def test_10_percent_downsampling(self):
        """Test 10% downsampling."""
        x = np.arange(1000)
        y = np.random.randn(1000)

        x_ds, y_ds = downsample_for_plot(x, y, percent=10.0)

        # Should have approximately 100 points
        assert len(x_ds) <= 110  # Allow some tolerance
        assert len(x_ds) >= 90

    def test_1_percent_downsampling(self):
        """Test 1% downsampling for large datasets."""
        x = np.arange(1000000)
        y = np.sin(x / 1000)

        x_ds, y_ds = downsample_for_plot(x, y, percent=1.0)

        # Should have approximately 10000 points
        assert len(x_ds) <= 11000
        assert len(x_ds) >= 9000

    def test_zero_percent(self):
        """Test with 0% (edge case)."""
        x = np.arange(100)
        y = np.sin(x)

        x_ds, y_ds = downsample_for_plot(x, y, percent=0.0)

        # Should return at least one point
        assert len(x_ds) >= 1

    def test_over_100_percent(self):
        """Test with >100% (no effect)."""
        x = np.arange(100)
        y = np.sin(x)

        x_ds, y_ds = downsample_for_plot(x, y, percent=150.0)

        assert len(x_ds) == 100

    def test_preserves_endpoints(self):
        """Test that first point is preserved."""
        x = np.arange(100)
        y = np.sin(x)

        x_ds, y_ds = downsample_for_plot(x, y, percent=10.0)

        # First point should be preserved
        assert x_ds[0] == x[0]
        assert y_ds[0] == y[0]


class TestDownsampleDataframe:
    """Test downsample_dataframe function."""

    def test_no_downsampling(self):
        """Test with 100% (no downsampling)."""
        df = pd.DataFrame(
            {
                "t": np.arange(100),
                "value": np.random.randn(100),
            }
        )

        df_ds = downsample_dataframe(df, percent=100.0)

        assert len(df_ds) == 100

    def test_50_percent(self):
        """Test 50% downsampling."""
        df = pd.DataFrame(
            {
                "t": np.arange(100),
                "value": np.random.randn(100),
            }
        )

        df_ds = downsample_dataframe(df, percent=50.0)

        assert len(df_ds) == 50

    def test_preserves_columns(self):
        """Test that all columns are preserved."""
        df = pd.DataFrame(
            {
                "t": np.arange(100),
                "value1": np.random.randn(100),
                "value2": np.random.randn(100),
            }
        )

        df_ds = downsample_dataframe(df, percent=20.0)

        assert list(df_ds.columns) == ["t", "value1", "value2"]


class TestDownsampleMinmax:
    """Test downsample_minmax function."""

    def test_preserves_extrema(self):
        """Test that min/max values are preserved in each bin."""
        np.random.seed(42)
        x = np.arange(1000)
        y = np.random.randn(1000)

        x_ds, y_ds = downsample_minmax(x, y, n_bins=10)

        # Should have 2 points per bin (min and max)
        assert len(x_ds) == 20

        # Global min and max should be in result
        assert y.min() in y_ds or np.isclose(y_ds, y.min()).any()
        assert y.max() in y_ds or np.isclose(y_ds, y.max()).any()

    def test_small_dataset(self):
        """Test with dataset smaller than n_bins * 2."""
        x = np.arange(10)
        y = np.sin(x)

        x_ds, y_ds = downsample_minmax(x, y, n_bins=100)

        # Should return original data
        np.testing.assert_array_equal(x_ds, x)
        np.testing.assert_array_equal(y_ds, y)


class TestEstimateDownsamplePercent:
    """Test estimate_downsample_percent function."""

    def test_small_dataset(self):
        """Test with small dataset (no downsampling needed)."""
        percent = estimate_downsample_percent(5000, target_points=10000)
        assert percent == 100.0

    def test_large_dataset(self):
        """Test with large dataset."""
        percent = estimate_downsample_percent(1000000, target_points=10000)
        assert percent == 1.0

    def test_exact_target(self):
        """Test when n_points equals target."""
        percent = estimate_downsample_percent(10000, target_points=10000)
        assert percent == 100.0

    def test_custom_target(self):
        """Test with custom target points."""
        percent = estimate_downsample_percent(50000, target_points=5000)
        assert percent == 10.0


class TestPlotRegimes:
    """Test plot_regimes function."""

    @patch("python_magnetrun.analysis.plotting.plt")
    def test_adds_spans(self, mock_plt):
        """Test that axvspan is called correctly."""
        mock_ax = MagicMock()

        regimes = ["U", "P", "D"]
        times = [0, 10, 50, 100]

        plot_regimes(mock_ax, regimes, times)

        # Should call axvspan 3 times (for each regime)
        assert mock_ax.axvspan.call_count == 3

    def test_empty_regimes(self):
        """Test with empty regimes list."""
        mock_ax = MagicMock()

        plot_regimes(mock_ax, [], [])

        # Should not call axvspan
        mock_ax.axvspan.assert_not_called()

    def test_single_time(self):
        """Test with single time point."""
        mock_ax = MagicMock()

        plot_regimes(mock_ax, ["U"], [0])

        # Should not call axvspan (need at least 2 time points)
        mock_ax.axvspan.assert_not_called()


class TestPlotIncidentsMarkers:
    """Test plot_incidents_markers function."""

    def test_adds_vertical_lines(self):
        """Test that axvline is called for each incident."""
        mock_ax = MagicMock()

        incident_times = [10.0, 25.0, 50.0]

        plot_incidents_markers(mock_ax, incident_times)

        assert mock_ax.axvline.call_count == 3

    def test_empty_incidents(self):
        """Test with no incidents."""
        mock_ax = MagicMock()

        plot_incidents_markers(mock_ax, [])

        mock_ax.axvline.assert_not_called()

    def test_custom_style(self):
        """Test with custom color and alpha."""
        mock_ax = MagicMock()

        plot_incidents_markers(mock_ax, [10.0], color="blue", alpha=0.5, linestyle="-")

        mock_ax.axvline.assert_called_once_with(10.0, color="blue", alpha=0.5, linestyle="-")


class TestCreateFigureGrid:
    """Test create_figure_grid function."""

    @patch("python_magnetrun.analysis.plotting.plt")
    def test_single_plot(self, mock_plt):
        """Test grid with single plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        fig, axes = create_figure_grid(1, n_cols=2)

        assert fig is mock_fig
        mock_plt.subplots.assert_called_once()

    @patch("python_magnetrun.analysis.plotting.plt")
    def test_multiple_plots(self, mock_plt):
        """Test grid with multiple plots."""
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock(), MagicMock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        fig, axes = create_figure_grid(2, n_cols=2)

        assert fig is mock_fig


class TestPlotDataFunction:
    """Test plot_data function with mocked matplotlib."""

    @patch("python_magnetrun.analysis.plotting.plt")
    def test_basic_plot_creation(self, mock_plt):
        """Test that plot_data creates a figure."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Create test data
        df_overview = pd.DataFrame(
            {
                "t": np.arange(100),
                "Courant_GR1": np.random.randn(100),
            }
        )
        df_archive = pd.DataFrame(
            {
                "t": np.arange(100),
                "Courant_GR1": np.random.randn(100),
            }
        )
        df_pupitre = pd.DataFrame(
            {
                "t": np.arange(100),
                "IH": np.random.randn(100),
            }
        )

        channels_dict = {"Courant_GR1": "Courant_GR1"}
        pupitre_dict = {"M9": {"Courant_GR1": "IH"}}

        from python_magnetrun.analysis.plotting import plot_data

        # Call function
        result = plot_data(
            df_overview,
            df_archive,
            df_pupitre,
            None,
            channels_dict,
            pupitre_dict,
            "M9",
            tkey="t",
            key="Courant_GR1",
            title="Test",
            msg="",
            show=False,
            save=False,
            downsample_percent=50.0,
        )

        # Verify figure was created
        mock_plt.subplots.assert_called_once()


class TestIntegration:
    """Integration tests for plotting module."""

    def test_downsampling_workflow(self):
        """Test complete downsampling workflow."""
        # Create large dataset
        n_points = 100000
        x = np.arange(n_points)
        y = np.sin(x / 1000) + np.random.randn(n_points) * 0.1

        # Estimate appropriate downsampling
        percent = estimate_downsample_percent(n_points, target_points=5000)
        assert percent < 10.0

        # Apply downsampling
        x_ds, y_ds = downsample_for_plot(x, y, percent)

        # Verify size reduction
        assert len(x_ds) < n_points / 10
        assert len(x_ds) > 0

    def test_dataframe_and_array_consistency(self):
        """Test that DataFrame and array downsampling are consistent."""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "t": np.arange(1000),
                "value": np.random.randn(1000),
            }
        )

        # Downsample DataFrame
        df_ds = downsample_dataframe(df, percent=10.0)

        # Downsample arrays
        x_ds, y_ds = downsample_for_plot(df["t"].values, df["value"].values, percent=10.0)

        # Should have same length
        assert len(df_ds) == len(x_ds)
