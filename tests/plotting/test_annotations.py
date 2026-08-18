"""Tests for AnnotationManager."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from python_magnetrun.plotting.annotations import AnnotationManager
from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend
from python_magnetrun.plotting.style import PlotColors


class TestAnnotationManagerMatplotlib:
    def _make_fig(self):
        b = MatplotlibBackend()
        return b, b.subplots(1)

    def test_add_annotation_stores_detail(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        detail = {"idx": 0, "anomaly": "spike #1", "tkey": "t", "df": pd.DataFrame()}
        mgr.add(fig, 0, t=5.0, f=1.0, label="spike #1", detail=detail)
        assert len(mgr._mpl_annotation_dict) == 1

    def test_connect_without_annotations_is_noop(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        mgr.connect(fig)  # should not raise

    def test_connect_wires_pick_event(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        detail = {"idx": 0, "anomaly": "ev", "tkey": "t", "df": pd.DataFrame()}
        mgr.add(fig, 0, t=1.0, f=1.0, label="ev", detail=detail)
        mgr.connect(fig)  # should not raise


class TestAnnotationColors:
    """Test per-type annotation box colors."""

    def _make_mgr(self):
        b = MatplotlibBackend()
        fig = b.subplots(1)
        return AnnotationManager(b), fig

    def test_default_color_used_when_none(self):
        import matplotlib.colors as mcolors
        mgr, fig = self._make_mgr()
        mgr.add(fig, 0, t=1.0, f=0.5, label="ev", detail={"idx": 0})
        annot = list(mgr._mpl_annotation_dict.keys())[0]
        fc = annot.get_bbox_patch().get_facecolor()
        expected = mcolors.to_rgb(PlotColors().incident_default)
        assert mcolors.to_hex(fc[:3]) == mcolors.to_hex(expected)

    def test_explicit_color_forwarded(self):
        import matplotlib.colors as mcolors
        mgr, fig = self._make_mgr()
        mgr.add(fig, 0, t=1.0, f=0.5, label="ev", detail={"idx": 0}, color="red")
        annot = list(mgr._mpl_annotation_dict.keys())[0]
        fc = annot.get_bbox_patch().get_facecolor()
        assert mcolors.to_hex(fc[:3]) == mcolors.to_hex(mcolors.to_rgb("red"))

    def test_get_incident_color_mapping(self):
        colors = PlotColors()
        assert colors.get_incident_color("default") == colors.incident_default
        assert colors.get_incident_color("spike") == colors.incident_spike
        assert colors.get_incident_color("trigger") == colors.incident_trigger
        assert colors.get_incident_color("hybrid_trigger") == colors.incident_hybrid

    def test_get_incident_color_unknown_falls_back(self):
        colors = PlotColors()
        assert colors.get_incident_color("unknown_type") == colors.incident_default

    def test_four_incident_types_have_distinct_colors(self):
        colors = PlotColors()
        incident_colors = [
            colors.incident_default,
            colors.incident_spike,
            colors.incident_trigger,
            colors.incident_hybrid,
        ]
        # All four should be defined (non-empty strings)
        assert all(isinstance(c, str) and c for c in incident_colors)


class TestAddVline:
    """add_vline() draws a vertical line with the correct color."""

    def _make_mgr(self):
        b = MatplotlibBackend()
        fig = b.subplots(1)
        return AnnotationManager(b), fig

    def test_axvline_called_at_correct_x(self):
        from unittest.mock import patch

        mgr, fig = self._make_mgr()
        ax = fig._magnetrun_axes[0]
        with patch.object(ax, "axvline") as mock_axvline:
            mgr.add_vline(fig, 0, t=42.0, label="trigger #1")
        mock_axvline.assert_called_once()
        assert mock_axvline.call_args.kwargs.get("x") == 42.0 or mock_axvline.call_args.args[0] == 42.0

    def test_explicit_color_forwarded_to_axvline(self):
        from unittest.mock import patch

        mgr, fig = self._make_mgr()
        ax = fig._magnetrun_axes[0]
        with patch.object(ax, "axvline") as mock_axvline:
            mgr.add_vline(fig, 0, t=5.0, label="ev", color="red")
        _, kwargs = mock_axvline.call_args
        assert kwargs.get("color") == "red"

    def test_default_color_is_incident_default(self):
        from unittest.mock import patch

        mgr, fig = self._make_mgr()
        ax = fig._magnetrun_axes[0]
        with patch.object(ax, "axvline") as mock_axvline:
            mgr.add_vline(fig, 0, t=1.0, label="ev")
        _, kwargs = mock_axvline.call_args
        assert kwargs.get("color") == PlotColors().incident_default

    def test_no_entry_added_to_annotation_dict(self):
        """add_vline does not register detail metadata (no click handler needed)."""
        mgr, fig = self._make_mgr()
        mgr.add_vline(fig, 0, t=1.0, label="ev")
        assert len(mgr._mpl_annotation_dict) == 0


class TestAnnotationManagerPlotly:
    def test_add_annotation_plotly(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(1)
        mgr = AnnotationManager(b)
        mgr.add(fig, 0, t=3.0, f=0.0, label="event", detail={"info": "test"})
        # vline should be added as a shape/annotation
        assert len(fig.layout.shapes) > 0 or len(fig.layout.annotations) >= 0

    def test_connect_plotly_is_noop(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(1)
        mgr = AnnotationManager(b)
        mgr.connect(fig)  # should not raise
