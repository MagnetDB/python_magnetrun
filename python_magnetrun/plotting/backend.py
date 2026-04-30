"""PlottingBackend protocol and ``get_backend()`` factory.

Every backend must implement ``PlottingBackend``.  Use ``get_backend(name)``
to obtain a backend instance without importing the concrete class directly.

Supported names
---------------
``"matplotlib"`` (default)
    Static PNG / SVG output, Jupyter inline.
``"plotly"``
    Interactive HTML, JSON export for REST/JS frontends.  Requires
    ``plotly`` (install with ``pip install python_magnetrun[plotting]``).
``"plotly-resampler"``
    Plotly figure wrapped in ``FigureResampler`` for view-dependent
    on-the-fly aggregation.  Requires a live Python kernel (Jupyter,
    voilà, marimo, Dash).  Requires ``plotly-resampler``
    (``pip install python_magnetrun[resampler]``).
``"plotly-widget"``
    Same as ``"plotly-resampler"`` but uses ``FigureWidgetResampler``
    (ipywidget) for Jupyter / marimo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .style import PlotStyle

__all__ = ["PlottingBackend", "get_backend"]


class PlottingBackend(Protocol):
    """Minimal contract every plotting backend must satisfy."""

    def subplots(
        self,
        n: int,
        *,
        share_x: bool = True,
        style: PlotStyle | None = None,
    ) -> Any:
        """Return an opaque figure handle with *n* sub-axes."""
        ...

    def add_series(
        self,
        fig: Any,
        ax_idx: int,
        t: np.ndarray,
        y: np.ndarray,
        label: str,
        *,
        normalize: bool = False,
        color: str | None = None,
        ylabel: str | None = None,
        marker: str | None = None,
        linestyle: str | None = None,
        markevery: int | None = None,
        alpha: float | None = None,
    ) -> None:
        """Plot *(t, y)* on the *ax_idx*-th axes.

        When *normalize* is ``True`` the series is divided by its absolute
        maximum and the legend label is amended with ``"(max=<value>)"``.
        """
        ...

    def add_scatter(
        self,
        fig: Any,
        ax_idx: int,
        t: np.ndarray,
        y: np.ndarray,
        label: str,
        *,
        color: str = "red",
        size: int = 10,
        alpha: float = 0.7,
    ) -> None:
        """Overlay scatter points on the *ax_idx*-th axes.

        Used primarily for outlier highlighting: plot the outlier points as
        a separate scatter series on top of the main line.
        """
        ...

    def add_annotation(
        self,
        fig: Any,
        ax_idx: int,
        t: float,
        f: float,
        label: str,
        detail: dict | None = None,
    ) -> None:
        """Add a clickable annotation at (t, f) on the *ax_idx*-th axes."""
        ...

    def save(self, fig: Any, path: Path, *, dpi: int = 300) -> None:
        """Save *fig* to *path*."""
        ...

    def show(self, fig: Any) -> None:
        """Display *fig* interactively."""
        ...

    def finalize(self, fig: Any, *, xlabel: str = "t [s]") -> None:
        """Apply final touches (axis labels, tight layout, etc.)."""
        ...

    def to_json(self, fig: Any) -> str:
        """Serialise to a self-contained JSON string (Plotly spec).

        Used by REST API endpoints and JS frontends.
        Raises ``NotImplementedError`` for backends that do not support it.
        """
        ...


def get_backend(name: str | PlottingBackend | None = "matplotlib") -> PlottingBackend:
    """Return a backend instance by name.

    Parameters
    ----------
    name:
        One of ``"matplotlib"``, ``"plotly"``, ``"plotly-resampler"``,
        ``"plotly-widget"``.  A ``PlottingBackend`` instance is returned
        unchanged.  ``None`` defaults to ``"matplotlib"``.

    ``"plotly-resampler"`` and ``"plotly-widget"`` require a live Python
    kernel (Jupyter / voilà / marimo / Dash) and the ``plotly-resampler``
    package.  Use ``"plotly"`` for static export or the REST/JS path.
    """
    if not isinstance(name, str | type(None)):
        return name  # already a PlottingBackend instance
    if name == "plotly":
        from .plotly_backend import PlotlyBackend

        return PlotlyBackend()
    if name in ("plotly-resampler", "plotly-widget"):
        from .plotly_resampler_backend import PlotlyResamplerBackend

        return PlotlyResamplerBackend(widget=(name == "plotly-widget"))
    from .matplotlib_backend import MatplotlibBackend

    return MatplotlibBackend()
