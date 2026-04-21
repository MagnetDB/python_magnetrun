"""Tests for plotting backend protocol and factory."""

import json

import numpy as np
import pytest

from python_magnetrun.plotting.backend import get_backend
from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend


class TestGetBackend:
    def test_default_is_matplotlib(self):
        b = get_backend()
        assert isinstance(b, MatplotlibBackend)

    def test_matplotlib_explicit(self):
        b = get_backend("matplotlib")
        assert isinstance(b, MatplotlibBackend)

    def test_plotly_returns_plotly_backend(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = get_backend("plotly")
        assert isinstance(b, PlotlyBackend)

    def test_plotly_resampler_returns_resampler_backend(self):
        pytest.importorskip("plotly_resampler")
        from python_magnetrun.plotting.plotly_resampler_backend import PlotlyResamplerBackend

        b = get_backend("plotly-resampler")
        assert isinstance(b, PlotlyResamplerBackend)
        assert not b._widget

    def test_plotly_widget_sets_widget_flag(self):
        pytest.importorskip("plotly_resampler")
        from python_magnetrun.plotting.plotly_resampler_backend import PlotlyResamplerBackend

        b = get_backend("plotly-widget")
        assert isinstance(b, PlotlyResamplerBackend)
        assert b._widget


class TestMatplotlibBackend:
    def test_subplots_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(3)
        assert len(fig._magnetrun_axes) == 3

    def test_subplots_single(self):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(1)
        assert len(fig._magnetrun_axes) == 1

    def test_add_series(self):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(1)
        t = np.linspace(0, 10, 100)
        y = np.sin(t)
        b.add_series(fig, 0, t, y, label="sin")
        ax = fig._magnetrun_axes[0]
        assert len(ax.lines) == 1

    def test_add_series_normalize(self):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(1)
        t = np.linspace(0, 10, 50)
        y = np.full(50, 5.0)
        b.add_series(fig, 0, t, y, label="const", normalize=True)
        ax = fig._magnetrun_axes[0]
        line = ax.lines[0]
        # After normalization by abs max (5.0), all values should be 1.0
        np.testing.assert_allclose(line.get_ydata(), 1.0)
        assert "max=5" in line.get_label()

    def test_to_json_raises(self):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(1)
        with pytest.raises(NotImplementedError):
            b.to_json(fig)

    def test_save(self, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        b = MatplotlibBackend()
        fig = b.subplots(1)
        out = tmp_path / "test.png"
        b.save(fig, out)
        assert out.exists()


class TestPlotlyBackend:
    def test_subplots_returns_figure(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(2)
        assert len(fig.data) == 0  # no traces yet

    def test_add_series(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(2)
        t = np.linspace(0, 10, 50)
        y = np.sin(t)
        b.add_series(fig, 0, t, y, label="sin")
        b.add_series(fig, 1, t, np.cos(t), label="cos")
        assert len(fig.data) == 2

    def test_to_json_returns_valid_json(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(1)
        t = np.arange(10, dtype=float)
        b.add_series(fig, 0, t, t * 2, label="line")
        js = b.to_json(fig)
        parsed = json.loads(js)
        assert "data" in parsed


class TestPlotlyResamplerBackend:
    def test_requires_plotly_resampler(self):
        try:
            import plotly_resampler  # noqa: F401
        except ImportError:
            from python_magnetrun.plotting.plotly_resampler_backend import PlotlyResamplerBackend

            b = PlotlyResamplerBackend()
            with pytest.raises(ImportError):
                b.subplots(1)
            return
        # If installed, skip this test
        pytest.skip("plotly-resampler is installed")

    def test_save_raises(self):
        pytest.importorskip("plotly_resampler")
        from python_magnetrun.plotting.plotly_resampler_backend import PlotlyResamplerBackend

        b = PlotlyResamplerBackend()
        fig = b.subplots(1)
        with pytest.raises(NotImplementedError):
            b.save(fig, "out.png")

    def test_to_json_raises(self):
        pytest.importorskip("plotly_resampler")
        from python_magnetrun.plotting.plotly_resampler_backend import PlotlyResamplerBackend

        b = PlotlyResamplerBackend()
        fig = b.subplots(1)
        with pytest.raises(NotImplementedError):
            b.to_json(fig)
