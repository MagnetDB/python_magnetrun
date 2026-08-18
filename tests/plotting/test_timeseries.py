"""Tests for plot_subplots and plot_overlay."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from python_magnetrun.plotting.timeseries import plot_overlay, plot_subplots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def simple_df():
    t = np.linspace(0, 10, 200)
    return pd.DataFrame({"t": t, "A": np.sin(t), "B": np.cos(t), "C": t * 0.1})


class TestPlotSubplots:
    def test_returns_figure_matplotlib(self, simple_df):
        import matplotlib.figure

        fig = plot_subplots(simple_df, ["A", "B"], backend="matplotlib")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_axes_count(self, simple_df):
        fig = plot_subplots(simple_df, ["A", "B", "C"], backend="matplotlib")
        assert len(fig._magnetrun_axes) == 3

    def test_each_ax_has_one_line(self, simple_df):
        fig = plot_subplots(simple_df, ["A", "B"], backend="matplotlib")
        for ax in fig._magnetrun_axes:
            assert len(ax.lines) == 1

    def test_plotly_trace_count(self, simple_df):
        pytest.importorskip("plotly")
        fig = plot_subplots(simple_df, ["A", "B"], backend="plotly")
        assert len(fig.data) == 2

    def test_empty_fields_raises(self, simple_df):
        with pytest.raises(ValueError):
            plot_subplots(simple_df, [], backend="matplotlib")

    def test_missing_field_skipped(self, simple_df):
        fig = plot_subplots(simple_df, ["A", "MISSING"], backend="matplotlib")
        # Second subplot created but has no lines
        assert len(fig._magnetrun_axes) == 2
        assert len(fig._magnetrun_axes[0].lines) == 1
        assert len(fig._magnetrun_axes[1].lines) == 0

    def test_with_downsample(self, simple_df):
        from python_magnetrun.utils.downsampling import DownsampleConfig

        cfg = DownsampleConfig(n_out=50)
        fig = plot_subplots(simple_df, ["A"], downsample=cfg, backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        # Should have at most 50 points
        assert len(ax.lines[0].get_xdata()) <= 50

    def test_title_set(self, simple_df):
        fig = plot_subplots(simple_df, ["A"], backend="matplotlib", title="My title")
        assert fig._suptitle.get_text() == "My title"


class TestPlotOverlay:
    def test_single_axes(self, simple_df):
        fig = plot_overlay(simple_df, ["A", "B"], backend="matplotlib")
        assert len(fig._magnetrun_axes) == 1

    def test_multiple_series_on_one_axes(self, simple_df):
        fig = plot_overlay(simple_df, ["A", "B", "C"], backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        assert len(ax.lines) == 3

    def test_normalize_scales_to_one(self, simple_df):
        fig = plot_overlay(simple_df, ["A", "B"], normalize=True, backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        for line in ax.lines:
            y = line.get_ydata()
            assert float(np.abs(y).max()) <= 1.0 + 1e-9

    def test_normalize_label_contains_max(self, simple_df):
        fig = plot_overlay(simple_df, ["A"], normalize=True, backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        label = ax.lines[0].get_label()
        assert "max=" in label

    def test_shared_unit_goes_to_ylabel_not_legend(self, simple_df):
        # Single field → shared unit → "T" in ylabel, not legend label.
        fig = plot_overlay(
            simple_df,
            ["A"],
            normalize=True,
            display_units={"A": "T"},
            backend="matplotlib",
        )
        ax = fig._magnetrun_axes[0]
        assert ax.get_ylabel() == "[T]"
        assert "T" not in ax.lines[0].get_label()

    def test_mixed_units_go_to_legend(self, simple_df):
        # Two fields with different display symbols → unit in each legend label.
        fig = plot_overlay(
            simple_df,
            ["A", "B"],
            display_units={"A": "T", "B": "A"},
            backend="matplotlib",
        )
        ax = fig._magnetrun_axes[0]
        labels = [line.get_label() for line in ax.lines]
        assert any("[T]" in lbl for lbl in labels)
        assert any("[A]" in lbl for lbl in labels)
        assert ax.get_ylabel() == ""

    def test_no_normalize_preserves_values(self, simple_df):
        fig = plot_overlay(simple_df, ["A"], normalize=False, backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        y = ax.lines[0].get_ydata()
        expected = simple_df["A"].to_numpy()
        np.testing.assert_allclose(y, expected)

    def test_plotly_trace_count(self, simple_df):
        pytest.importorskip("plotly")
        fig = plot_overlay(simple_df, ["A", "B"], backend="plotly")
        assert len(fig.data) == 2

    def test_empty_fields_raises(self, simple_df):
        with pytest.raises(ValueError):
            plot_overlay(simple_df, [], backend="matplotlib")

    def test_normalize_and_display_units_do_not_raise(self, simple_df):
        # normalize and display_units are independent and compose — must not raise.
        fig = plot_overlay(
            simple_df,
            ["A"],
            normalize=True,
            display_units={"A": "T"},
            backend="matplotlib",
        )
        assert fig is not None


class TestPlotSubplotsNormalize:
    def test_normalize_scales_to_one(self, simple_df):
        fig = plot_subplots(simple_df, ["A", "B"], normalize=True, backend="matplotlib")
        for ax in fig._magnetrun_axes:
            if ax.lines:
                y = ax.lines[0].get_ydata()
                assert float(np.abs(y).max()) <= 1.0 + 1e-9

    def test_normalize_label_contains_max(self, simple_df):
        fig = plot_subplots(simple_df, ["A"], normalize=True, backend="matplotlib")
        ax = fig._magnetrun_axes[0]
        label = ax.lines[0].get_label()
        assert "max=" in label

    def test_normalize_and_display_units_do_not_raise(self, simple_df):
        fig = plot_subplots(
            simple_df,
            ["A"],
            normalize=True,
            display_units={"A": "T"},
            backend="matplotlib",
        )
        assert fig is not None
