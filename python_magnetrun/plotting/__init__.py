"""``python_magnetrun.plotting`` — backend-agnostic plotting subpackage.

Public API
----------
``PlotStyle``, ``PlotColors``
    Style and color configuration dataclasses.
``get_backend(name)``
    Factory returning a :class:`PlottingBackend` instance.
``plot_subplots(data, fields, ...)``
    N fields as stacked subplots sharing a time axis.
``plot_overlay(data, fields, ...)``
    N fields on one axes with optional normalization.
``AnnotationManager``
    Backend-agnostic clickable annotation manager.

Backend selection
-----------------
| Goal                          | ``backend=``              |
|-------------------------------|---------------------------|
| Static PNG / SVG file         | ``"matplotlib"`` (default)|
| JSON for JS frontend / REST   | ``"plotly"``              |
| Jupyter / marimo notebook     | ``"plotly"``              |
| voilà dashboard               | ``"plotly"``              |
| Dash app with huge datasets   | ``"plotly-resampler"``    |
| Jupyter / marimo ipywidget    | ``"plotly-widget"``       |
"""

from .annotations import AnnotationManager
from .backend import PlottingBackend, get_backend
from .style import (
    BUNDLED_CONFIG_PATH,
    DEFAULT_COLORS,
    DEFAULT_STYLE,
    PlotColors,
    PlotConfig,
    PlotStyle,
    load_plot_config,
    save_plot_config,
)
from .timeseries import plot_overlay, plot_subplots
from .utils import format_axis_label, format_legend_label, resolve_legend_labels

__all__ = [
    "PlotStyle",
    "PlotColors",
    "DEFAULT_STYLE",
    "DEFAULT_COLORS",
    "PlottingBackend",
    "get_backend",
    "plot_subplots",
    "plot_overlay",
    "AnnotationManager",
    "PlotConfig",
    "load_plot_config",
    "save_plot_config",
    "BUNDLED_CONFIG_PATH",
    "format_axis_label",
    "format_legend_label",
    "resolve_legend_labels",
]
