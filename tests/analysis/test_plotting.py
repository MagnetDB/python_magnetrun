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
    estimate_downsample_percent,
    plot_incidents_markers,
    # Plotting functions
    plot_regimes,
)
from python_magnetrun.utils.downsampling import (
    DownsampleConfig,
    downsample_arrays,
    downsample_dataframe,
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


class TestDownsampleArrays:
    """Test downsample_arrays via DownsampleConfig (canonical API)."""

    def test_no_downsampling(self):
        x = np.arange(100, dtype=float)
        y = np.sin(x)
        config = DownsampleConfig(n_out=200)
        y_ds, x_ds = downsample_arrays(y, x, config)
        assert len(x_ds) == 100

    def test_stride_50_percent(self):
        x = np.arange(100, dtype=float)
        y = np.sin(x)
        config = DownsampleConfig.from_percent(len(x), 50.0)
        y_ds, x_ds = downsample_arrays(y, x, config)
        assert len(x_ds) == 50

    def test_stride_10_percent(self):
        x = np.arange(1000, dtype=float)
        y = np.random.randn(1000)
        config = DownsampleConfig.from_percent(len(x), 10.0)
        y_ds, x_ds = downsample_arrays(y, x, config)
        assert 90 <= len(x_ds) <= 110

    def test_minmax_preserves_extrema(self):
        np.random.seed(42)
        x = np.arange(1000, dtype=float)
        y = np.random.randn(1000)
        config = DownsampleConfig(n_out=20, method="minmax")
        y_ds, x_ds = downsample_arrays(y, x, config)
        assert len(x_ds) == 20
        assert np.isclose(y_ds, y.min()).any() or y.min() in y_ds

    def test_first_point_preserved(self):
        x = np.arange(100, dtype=float)
        y = np.sin(x)
        config = DownsampleConfig.from_percent(len(x), 10.0)
        y_ds, x_ds = downsample_arrays(y, x, config)
        assert x_ds[0] == x[0]


class TestDownsampleDataframe:
    """Test downsample_dataframe (canonical API from utils.downsampling)."""

    def test_no_downsampling(self):
        df = pd.DataFrame({"t": np.arange(100, dtype=float), "value": np.random.randn(100)})
        config = DownsampleConfig(n_out=200)
        df_ds = downsample_dataframe(df, "t", ["value"], config)
        assert len(df_ds) == 100

    def test_stride_50_percent(self):
        df = pd.DataFrame({"t": np.arange(100, dtype=float), "value": np.random.randn(100)})
        config = DownsampleConfig.from_percent(len(df), 50.0)
        df_ds = downsample_dataframe(df, "t", ["value"], config)
        assert len(df_ds) == 50

    def test_preserves_columns(self):
        df = pd.DataFrame(
            {
                "t": np.arange(100, dtype=float),
                "value1": np.random.randn(100),
                "value2": np.random.randn(100),
            }
        )
        config = DownsampleConfig.from_percent(len(df), 20.0)
        df_ds = downsample_dataframe(df, "t", ["value1", "value2"], config)
        assert list(df_ds.columns) == ["t", "value1", "value2"]


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


class TestPlotDataUnits:
    """Verify unit propagation through plot_data()."""

    @staticmethod
    def _overview_with_unit(unit_symbol: str, pint_unit):
        """Return df_overview with attrs['units'] set for 'Courant_GR1'."""
        import pint as _pint  # noqa: F401 (ensure available)

        df = pd.DataFrame({"t": np.arange(5, dtype=float), "Courant_GR1": np.ones(5)})
        df.attrs["units"] = {"Courant_GR1": (unit_symbol, pint_unit)}
        return df

    def test_merged_attrs_populated(self):
        """units_map is stored on merged.attrs['units'] after pd.concat.

        Uses the non-direct (plot_overlay) path so that merged is passed as
        the first positional arg and can be captured.
        """
        import pint

        from python_magnetrun.plotting.backend import PlottingBackend

        ureg = pint.UnitRegistry()

        df_overview = self._overview_with_unit("A", ureg.ampere)
        df_archive = pd.DataFrame({"t": np.arange(5, dtype=float)})
        df_pupitre = pd.DataFrame({"t": np.arange(5, dtype=float)})

        # A MagicMock that passes isinstance(b, PlottingBackend) — enough to
        # satisfy get_backend (which returns it unchanged) but is NOT a
        # MatplotlibBackend, so the non-direct plot_overlay path is taken.
        mock_backend = MagicMock(spec=PlottingBackend)

        captured: dict = {}

        def fake_plot_overlay(merged_df, *args, **kwargs):
            captured["merged"] = merged_df
            fig = MagicMock()
            fig._magnetrun_xlabel = ""
            fig._magnetrun_ylabel = ""
            return fig

        from python_magnetrun.analysis.plotting import plot_data

        with patch("python_magnetrun.analysis.plotting.plot_overlay", side_effect=fake_plot_overlay):
            plot_data(
                df_overview,
                df_archive,
                df_pupitre,
                None,
                channels_dict={},
                pupitre_dict=None,
                housing="M9",
                tkey="t",
                key="Courant_GR1",
                title="Test",
                msg="",
                show=False,
                save=False,
                backend=mock_backend,
            )

        assert "merged" in captured, "plot_overlay was never called"
        merged = captured["merged"]
        units = merged.attrs.get("units", {})
        assert "Overview: Courant_GR1" in units, f"Expected 'Overview: Courant_GR1' in units, got {list(units)}"
        symbol, unit = units["Overview: Courant_GR1"]
        assert symbol == "A"
        assert unit == ureg.ampere

    @patch("python_magnetrun.analysis.plotting.plt")
    def test_ylabel_from_pupitre_when_overview_empty(self, mock_plt):
        """ylabel is derived from pupitre attrs when overview+archive have no units."""
        import pint

        ureg = pint.UnitRegistry()

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_fig._magnetrun_axes = [mock_ax]

        df_overview = pd.DataFrame({"t": np.arange(5, dtype=float)})
        df_archive = pd.DataFrame({"t": np.arange(5, dtype=float)})
        df_pupitre = pd.DataFrame({"t": np.arange(5, dtype=float), "IH": np.ones(5)})
        df_pupitre.attrs["units"] = {"IH": ("A", ureg.ampere)}

        from python_magnetrun.analysis.plotting import plot_data

        plot_data(
            df_overview,
            df_archive,
            df_pupitre,
            None,
            channels_dict={},
            pupitre_dict={"M9": {"Courant_GR1": "IH"}},
            housing="M9",
            tkey="t",
            key="Courant_GR1",
            title="Test",
            msg="",
            show=False,
            save=False,
        )

        mock_ax.set_ylabel.assert_called_once()
        ylabel_arg = mock_ax.set_ylabel.call_args[0][0]
        assert "A" in ylabel_arg


class TestIntegration:
    """Integration tests for plotting module."""

    def test_downsampling_workflow(self):
        """Test complete downsampling workflow."""
        n_points = 100000
        x = np.arange(n_points, dtype=float)
        y = np.sin(x / 1000) + np.random.randn(n_points) * 0.1

        percent = estimate_downsample_percent(n_points, target_points=5000)
        assert percent < 10.0

        config = DownsampleConfig.from_percent(n_points, percent)
        y_ds, x_ds = downsample_arrays(y, x, config)

        assert len(x_ds) < n_points / 10
        assert len(x_ds) > 0

    def test_dataframe_and_array_consistency(self):
        """Test that DataFrame and array downsampling give the same point count."""
        np.random.seed(42)
        n = 1000
        t = np.arange(n, dtype=float)
        values = np.random.randn(n)
        df = pd.DataFrame({"t": t, "value": values})

        config = DownsampleConfig.from_percent(n, 10.0)
        df_ds = downsample_dataframe(df, "t", ["value"], config)
        y_ds, x_ds = downsample_arrays(values, t, config)

        assert len(df_ds) == len(x_ds)
