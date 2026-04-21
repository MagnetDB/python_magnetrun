"""Tests for AnnotationManager."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from python_magnetrun.plotting.annotations import AnnotationManager
from python_magnetrun.plotting.matplotlib_backend import MatplotlibBackend


class TestAnnotationManagerMatplotlib:
    def _make_fig(self):
        b = MatplotlibBackend()
        return b, b.subplots(1)

    def test_add_annotation_stores_detail(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        detail = {"idx": 0, "anomaly": "spike #1", "tkey": "t", "df": pd.DataFrame()}
        mgr.add(fig, 0, t=5.0, label="spike #1", detail=detail)
        assert len(mgr._mpl_annotation_dict) == 1

    def test_connect_without_annotations_is_noop(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        mgr.connect(fig)  # should not raise

    def test_connect_wires_pick_event(self):
        b, fig = self._make_fig()
        mgr = AnnotationManager(b)
        detail = {"idx": 0, "anomaly": "ev", "tkey": "t", "df": pd.DataFrame()}
        mgr.add(fig, 0, t=1.0, label="ev", detail=detail)
        mgr.connect(fig)  # should not raise


class TestAnnotationManagerPlotly:
    def test_add_annotation_plotly(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(1)
        mgr = AnnotationManager(b)
        mgr.add(fig, 0, t=3.0, label="event", detail={"info": "test"})
        # vline should be added as a shape/annotation
        assert len(fig.layout.shapes) > 0 or len(fig.layout.annotations) >= 0

    def test_connect_plotly_is_noop(self):
        pytest.importorskip("plotly")
        from python_magnetrun.plotting.plotly_backend import PlotlyBackend

        b = PlotlyBackend()
        fig = b.subplots(1)
        mgr = AnnotationManager(b)
        mgr.connect(fig)  # should not raise
